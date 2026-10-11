"""Leasing one code-execution sandbox to a caller across requests.

A container id (``otari_cntr_…``) names a sandbox this deployment leased on a
caller's behalf. Within one request the tool loop already shares it across
every code call; this module is what lets the *next* request pick it up, so
the files a run wrote, the packages it installed and the variables it set are
still there. It is the gateway's counterpart to Anthropic's ``container`` and
OpenAI's ``code_interpreter`` container: a client reads the id off one
response and sends it with the next.

What the registry decides:

* **Ownership.** A lease is bound to the user and workspace that took it, and
  a resume by anyone else is answered exactly as an unknown id, so an id never
  reveals that another tenant's sandbox exists.
* **Lifetime.** Two clocks. The idle clock restarts on every use and is what
  the provider is told to hold the sandbox for; the hard clock starts at the
  first lease and never restarts, so a client resuming forever still gives the
  sandbox back. Past either, the id is gone.
* **Provider.** A lease belongs to the provider that made it. A deployment
  that switched providers cannot resume the other's sandbox, and is told so
  the same way.
* **Exclusivity.** One request at a time holds a sandbox. Two sharing one
  workspace would interleave their code and each collect the other's files, so
  a resume while another request is running is refused with
  :class:`ContainerBusyError` rather than admitted.
* **Fencing.** Each claim carries a token minted for that claim alone, both on
  the first take and on the takeover of one whose deadline passed. Only the
  request still holding it may renew, record, release or forget the row, so a
  request that outlived its claim and lost it to a takeover finishes its own
  response without touching the new holder's session handle, clocks or
  occupancy. A live request renews the claim on a bounded timer; a dead process
  stops renewing and the claim passes at its deadline.

Standalone only: it keeps its rows in the local database. Reclaiming the
sandbox itself is the provider's job, told the idle timeout at release; the
sweep (``container_sweeper``) only drops rows whose clocks ran out.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.tools_exceptions import ContainerOnManagedCredentialError
from gateway.repositories.code_execution import (
    SandboxContainerRow,
    claim_container_row,
    delete_container_rows,
    forget_container_row,
    get_container_row,
    insert_container_row,
    record_container_row,
    release_container_claim,
    renew_container_claim,
)

# The gateway's own container ids. OpenAI issues ``cntr_``-prefixed ids and
# Anthropic ``container_``-prefixed ones, so a reserved prefix is what lets an
# id the gateway minted be told from one that names a provider's container.
CONTAINER_ID_PREFIX = "otari_cntr_"
# What a request sends to ask for a sandbox that outlives it, in place of an id
# it does not have yet. OpenAI's own spelling on a ``code_interpreter`` entry is
# the object ``{"type": "auto"}``, which means the same thing.
CONTAINER_AUTO = "auto"
# How long a request's claim on a container lasts before another request may
# take it. The ceiling on a sandbox session's lease
# (``sandbox_backend._MAX_SESSION_TTL_S``), because a request cannot outlive the
# sandbox it is running code in, so nothing legitimate is displaced and a claim
# a crashed gateway never released expires with the lease it was protecting.
CONTAINER_CLAIM_TTL_S = 3600.0
# How often a still-running request extends its claim. Far shorter than the
# claim TTL, so a healthy request survives several missed renewals while a dead
# process's claim passes at one TTL plus at most one interval after its last
# renewal.
CONTAINER_CLAIM_RENEW_INTERVAL_S = 300.0


def new_container_id() -> str:
    """A fresh container id, minted once per lease and kept across resumes."""
    return f"{CONTAINER_ID_PREFIX}{uuid.uuid4().hex}"


def new_claim_token() -> str:
    """A fresh, non-reusable identity for one claim or one expired-claim takeover."""
    return uuid.uuid4().hex


def requested_container(value: object) -> str | None:
    """What a ``container`` field asks for: an id to resume, ``auto``, or nothing.

    ``auto`` is how a request asks for a sandbox that outlives it, and is what
    OpenAI's ``{"type": "auto"}`` object already means on a ``code_interpreter``
    entry; the string spelling is for the dialects with no object form, which is
    Anthropic's top-level field and the gateway's own entry. An id asks for that
    sandbox back. Absent, which is every request written before this existed,
    asks for neither and gets the sandbox released with the request.
    """
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        return CONTAINER_AUTO if cleaned.lower() == CONTAINER_AUTO else cleaned
    if isinstance(value, dict):
        nested = value.get("id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
        kind = value.get("type")
        return CONTAINER_AUTO if isinstance(kind, str) and kind.strip().lower() == CONTAINER_AUTO else None
    return None


def gateway_container_value(raw: object) -> str | None:
    """A ``container`` value only this gateway could have named, or ``None``.

    Which is an id it minted, or its own ``auto`` spelling. Deliberately string
    only: the object form is the provider's own (OpenAI's ``{"type": "auto"}``
    on a ``code_interpreter`` entry), which means something upstream and must
    reach it untouched. A provider's own id is not ours either, and passes.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.lower() == CONTAINER_AUTO or cleaned.startswith(CONTAINER_ID_PREFIX):
        return cleaned
    return None


def check_container_on_credential(container: object, *, managed_credential: bool) -> None:
    """Refuse a caller-chosen container id when the upstream account is not the caller's.

    A container id names an execution environment and the files uploaded into it,
    scoped to the *provider account* that minted it, not to an otari tenant. A
    managed credential is one the platform owns and many workspaces share, so
    forwarding an id the caller picked would let anyone holding one resume another
    tenant's container and read its workspace. A container id is minted by the
    provider and carries the caller's claim on it, so it cannot be namespaced:
    forward or refuse are the only answers, and on a shared account the answer is
    refuse.

    ``auto`` passes: it names no container, only asks this gateway to hold its
    own sandbox, and never reaches the provider.

    Raises:
        ContainerOnManagedCredentialError: ``managed_credential`` and ``container`` is not ``auto``.
    """
    if managed_credential and requested_container(container) != CONTAINER_AUTO:
        raise ContainerOnManagedCredentialError()


class ContainerNotFoundError(LookupError):
    """The container is unknown, expired, another tenant's, or another provider's.

    One error for all four on purpose: telling them apart would tell a caller
    which ids exist.
    """


class ContainerBusyError(RuntimeError):
    """The container is this caller's, and another of their requests is using it.

    Distinct from unknown because the answer is different: the id is good and
    retrying works, where an unknown id has to be dropped. Only reachable for a
    container the caller owns, so it reveals nothing about anyone else's.
    """


@dataclass(frozen=True)
class ContainerLease:
    """One held sandbox: the id a client resumes it by and what it maps to.

    ``provider_session_id`` is the provider's own handle, the one the port
    resumes. ``expires_at`` is the idle clock and ``hard_expires_at`` the hard
    one, both as reported to the client; the sandbox may in fact live a little
    longer, never shorter. ``claim_token`` is the fencing identity the resolving
    request took and every owner-scoped write has to name; ``None`` on a lease
    for a sandbox no prior request held.
    """

    container_id: str
    provider: str
    provider_session_id: str
    expires_at: datetime
    hard_expires_at: datetime
    claim_token: str | None = None


class SandboxContainers(Protocol):
    """What the sandbox backend needs to hold a session past its request.

    Implemented by :class:`SandboxContainerRegistry`; a Protocol so the backend
    does not depend on the database-backed one and a test can hand it a stub.
    """

    def keep_alive_s(self, resumed: ContainerLease | None) -> float:
        """How long the provider should hold the session after this request."""
        ...

    def lease(self, container_id: str, provider_session_id: str, *, resumed: ContainerLease | None) -> ContainerLease:
        """The lease this request holds, with both clocks set from now."""
        ...

    async def record(self, lease: ContainerLease) -> bool:
        """Persist the lease so the next request can resume it.

        Returns whether this caller still owned the claim; ``False`` when a
        takeover happened first, in which case nothing was written.
        """
        ...

    async def renew(self, lease: ContainerLease) -> bool:
        """Extend the claim a still-running request holds, if it still owns it."""
        ...

    async def release(self, container_id: str, *, claim_token: str) -> None:
        """Give back a claim taken at admission that no lease will be recorded against."""
        ...

    async def forget(self, container_id: str, *, claim_token: str) -> None:
        """Drop the mapping once the provider says the session is gone."""
        ...


class SandboxContainerRegistry:
    """The database-backed :class:`SandboxContainers`, scoped to one caller.

    Built per request by the route once the billed user and workspace are
    known. Writes go through the request's Unit of Work, one block each, for
    the reason the file bridge's do: the request session is released before
    the provider is dispatched, so nothing here holds a transaction open while
    code runs. The claim heartbeat alone runs on blocks of its own, because it
    fires from a background task while the request session is parked.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        user_id: str,
        workspace_id: uuid.UUID,
        provider: str,
        idle_ttl_s: int,
        max_lifetime_s: int,
    ) -> None:
        self._uow = uow
        self._user_id = user_id
        self._workspace_id = workspace_id
        self._provider = provider
        self._idle_ttl_s = idle_ttl_s
        self._max_lifetime_s = max_lifetime_s

    @property
    def idle_ttl_s(self) -> int:
        return self._idle_ttl_s

    def keep_alive_s(self, resumed: ContainerLease | None) -> float:
        # Never past the hard clock, and never so short the provider reclaims
        # the sandbox before the row saying so is written.
        if resumed is None:
            return float(self._idle_ttl_s)
        remaining = (resumed.hard_expires_at - datetime.now(UTC)).total_seconds()
        return max(1.0, min(float(self._idle_ttl_s), remaining))

    def lease(self, container_id: str, provider_session_id: str, *, resumed: ContainerLease | None) -> ContainerLease:
        now = datetime.now(UTC)
        hard = resumed.hard_expires_at if resumed is not None else now + timedelta(seconds=self._max_lifetime_s)
        return ContainerLease(
            container_id=container_id,
            provider=self._provider,
            provider_session_id=provider_session_id,
            expires_at=min(now + timedelta(seconds=self._idle_ttl_s), hard),
            hard_expires_at=hard,
            # Ownership is the admission claim's for the whole request, however
            # the lease object gets rebuilt once the session opens.
            claim_token=resumed.claim_token if resumed is not None else None,
        )

    async def resolve(self, container_id: str) -> ContainerLease:
        """The lease behind ``container_id`` for this caller, claimed for this request.

        Raises :class:`ContainerNotFoundError` when the id is unknown, expired,
        another tenant's or another provider's, and :class:`ContainerBusyError`
        when it is this caller's own and another of their requests holds it.

        A row whose clocks ran out is dropped on the way unless a live claim
        still runs code in it (a long request past its idle clock), so the next
        resume does not read a dead row but an in-use one is not swept out from
        under its request. Every write lands before either error is raised,
        which is why the block closes first.
        """
        if not container_id.startswith(CONTAINER_ID_PREFIX):
            raise ContainerNotFoundError(container_id)
        now = datetime.now(UTC)
        claim_token = new_claim_token()
        async with self._uow:
            row = await get_container_row(self._uow, container_id)
            claim_live = row is not None and row.in_use_until is not None and row.in_use_until > now
            expired = row is not None and (row.expires_at <= now or row.hard_expires_at <= now) and not claim_live
            if expired:
                await delete_container_rows(self._uow, [container_id])
            resumable = (
                row is not None
                and not expired
                and row.user_id == self._user_id
                and row.workspace_id == self._workspace_id
                and row.provider == self._provider
            )
            # Never claim past the hard clock: at that deadline the row belongs
            # to the sweep, and a claim reaching past it would delay reclaim.
            claimed = False
            if resumable:
                assert row is not None  # ``resumable`` already says so
                until = min(now + timedelta(seconds=CONTAINER_CLAIM_TTL_S), row.hard_expires_at)
                claimed = await claim_container_row(
                    self._uow,
                    container_id,
                    now=now,
                    until=until,
                    claim_token=claim_token,
                )
        if row is None or not resumable:
            raise ContainerNotFoundError(container_id)
        if not claimed:
            raise ContainerBusyError(container_id)
        return _lease_from_row(row, claim_token=claim_token)

    async def record(self, lease: ContainerLease) -> bool:
        """Record the lease a held session ended with, or no-op having lost the claim.

        A fresh sandbox (no claim, no row) inserts it. A resumed one is updated
        only where this request still holds the claim token; a takeover turns
        that into a no-op rather than a write over the new holder's row.
        """
        now = datetime.now(UTC)
        async with self._uow:
            if lease.claim_token is None:
                await insert_container_row(
                    self._uow,
                    SandboxContainerRow(
                        id=lease.container_id,
                        user_id=self._user_id,
                        workspace_id=self._workspace_id,
                        provider=lease.provider,
                        provider_session_id=lease.provider_session_id,
                        created_at=now,
                        last_used_at=now,
                        expires_at=lease.expires_at,
                        hard_expires_at=lease.hard_expires_at,
                    ),
                )
                return True
            return await record_container_row(
                self._uow,
                SandboxContainerRow(
                    id=lease.container_id,
                    user_id=self._user_id,
                    workspace_id=self._workspace_id,
                    provider=lease.provider,
                    provider_session_id=lease.provider_session_id,
                    created_at=now,
                    last_used_at=now,
                    expires_at=lease.expires_at,
                    hard_expires_at=lease.hard_expires_at,
                ),
                claim_token=lease.claim_token,
            )

    async def renew(self, lease: ContainerLease) -> bool:
        """Extend a live request's claim to the next deadline, bounded by the hard clock.

        Runs from the backend's heartbeat on a block of its own, so it never
        shares the parked request session. ``False`` means the claim is no
        longer this caller's (a deadline and a takeover passed) or the hard
        clock ran out; the caller then writes nothing more.
        """
        token = lease.claim_token
        if token is None:
            return False
        now = datetime.now(UTC)
        if lease.hard_expires_at <= now:
            return False
        until = min(now + timedelta(seconds=CONTAINER_CLAIM_TTL_S), lease.hard_expires_at)
        async with create_unit_of_work() as uow:
            async with uow:
                return await renew_container_claim(uow, lease.container_id, claim_token=token, until=until)

    async def forget(self, container_id: str, *, claim_token: str) -> None:
        async with self._uow:
            await forget_container_row(self._uow, container_id, claim_token=claim_token)

    async def release(self, container_id: str, *, claim_token: str) -> None:
        """Give back a claim this request took and will not record a lease against."""
        async with self._uow:
            await release_container_claim(self._uow, container_id, claim_token=claim_token)


def _lease_from_row(row: SandboxContainerRow, *, claim_token: str) -> ContainerLease:
    return ContainerLease(
        container_id=row.id,
        provider=row.provider,
        provider_session_id=row.provider_session_id,
        expires_at=row.expires_at,
        hard_expires_at=row.hard_expires_at,
        claim_token=claim_token,
    )
