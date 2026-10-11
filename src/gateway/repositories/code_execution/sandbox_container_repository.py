"""Data access for ``sandbox_containers``, the leases a caller resumes a sandbox by.

The registry that decides ownership, lifetime and exclusivity lives in the
service layer; this module is the only place its statements are spelled.

Every statement that moves a held claim carries the owner's ``claim_token`` in
its predicate, so a request whose claim expired under it writes nothing: the
conditional row count is the fence.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, or_, select, update

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import SandboxContainer


@dataclass(frozen=True)
class SandboxContainerRow:
    """One lease, resolved to its columns, with every datetime UTC-aware."""

    id: str
    user_id: str
    workspace_id: uuid.UUID
    provider: str
    provider_session_id: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    hard_expires_at: datetime
    in_use_until: datetime | None = None
    claim_token: str | None = None


def _as_utc(value: datetime) -> datetime:
    # SQLite hands datetimes back naive; they were stored as UTC.
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _row(model: SandboxContainer) -> SandboxContainerRow:
    return SandboxContainerRow(
        id=model.id,
        user_id=model.user_id,
        workspace_id=model.workspace_id,
        provider=model.provider,
        provider_session_id=model.provider_session_id,
        created_at=_as_utc(model.created_at),
        last_used_at=_as_utc(model.last_used_at),
        expires_at=_as_utc(model.expires_at),
        hard_expires_at=_as_utc(model.hard_expires_at),
        in_use_until=_as_utc(model.in_use_until) if model.in_use_until is not None else None,
        claim_token=model.claim_token,
    )


async def get_container_row(uow: UnitOfWork, container_id: str) -> SandboxContainerRow | None:
    """The lease behind ``container_id``, whoever owns it; the caller decides who may use it."""
    db = session_for(uow)
    model = (await db.execute(select(SandboxContainer).where(SandboxContainer.id == container_id))).scalar_one_or_none()
    return _row(model) if model is not None else None


async def insert_container_row(uow: UnitOfWork, row: SandboxContainerRow) -> None:
    """Write a lease no row exists for yet: the first lease of a freshly minted id.

    A fresh container id is unique and unpublished until this request ends, so
    nothing can hold a claim on it; it is written already free. Flushes; the
    caller's block commits.
    """
    db = session_for(uow)
    db.add(
        SandboxContainer(
            id=row.id,
            user_id=row.user_id,
            workspace_id=row.workspace_id,
            provider=row.provider,
            provider_session_id=row.provider_session_id,
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            expires_at=row.expires_at,
            hard_expires_at=row.hard_expires_at,
            in_use_until=None,
            claim_token=None,
        )
    )
    await db.flush()


async def record_container_row(uow: UnitOfWork, row: SandboxContainerRow, *, claim_token: str) -> bool:
    """Refresh a lease the caller still holds the claim on, giving the claim back.

    One conditional update keyed on ``claim_token``: it writes the new session
    handle and idle clock, keeps the hard clock, and frees the claim only for
    the request that owned it. A request whose claim was taken over after the
    deadline matches no row and changes nothing, so it cannot clear the new
    holder's occupancy. The token is intentionally left on the free row, so the
    same request can still forget a session the provider says is gone; the next
    claim replaces it. Returns whether this caller still owned the row. Flushes;
    the caller's block commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            update(SandboxContainer)
            .where(SandboxContainer.id == row.id, SandboxContainer.claim_token == claim_token)
            .values(
                provider_session_id=row.provider_session_id,
                last_used_at=row.last_used_at,
                expires_at=row.expires_at,
                in_use_until=None,
            )
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def upsert_container_row(uow: UnitOfWork, row: SandboxContainerRow) -> None:
    """Write or replace a whole lease, including one nobody claims.

    Administrative and test seeding path; request code claims, records and
    releases through the owner-scoped statements instead. ``created_at`` and
    ``hard_expires_at`` are kept from the existing row on a replace, so the hard
    clock never restarts. Flushes; the caller's block commits.
    """
    db = session_for(uow)
    existing = (await db.execute(select(SandboxContainer).where(SandboxContainer.id == row.id))).scalar_one_or_none()
    if existing is None:
        db.add(
            SandboxContainer(
                id=row.id,
                user_id=row.user_id,
                workspace_id=row.workspace_id,
                provider=row.provider,
                provider_session_id=row.provider_session_id,
                created_at=row.created_at,
                last_used_at=row.last_used_at,
                expires_at=row.expires_at,
                hard_expires_at=row.hard_expires_at,
                in_use_until=row.in_use_until,
                claim_token=row.claim_token,
            )
        )
    else:
        existing.user_id = row.user_id
        existing.workspace_id = row.workspace_id
        existing.provider = row.provider
        existing.provider_session_id = row.provider_session_id
        existing.last_used_at = row.last_used_at
        existing.expires_at = row.expires_at
        existing.hard_expires_at = row.hard_expires_at
        existing.in_use_until = row.in_use_until
        existing.claim_token = row.claim_token
    await db.flush()


async def claim_container_row(
    uow: UnitOfWork, container_id: str, *, now: datetime, until: datetime, claim_token: str
) -> bool:
    """Take the exclusive claim on a lease under a fresh fencing identity.

    Returns whether this caller got it. One conditional update, so two requests
    racing for the same container cannot both win: the loser sees no row updated.
    Winning mints ``claim_token`` as this claim's non-reusable identity, both on
    a free row and on the takeover of one whose ``in_use_until`` is past; every
    later owner-scoped write has to name it. Flushes; the caller's block commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            update(SandboxContainer)
            .where(
                SandboxContainer.id == container_id,
                or_(SandboxContainer.in_use_until.is_(None), SandboxContainer.in_use_until <= now),
            )
            .values(in_use_until=until, claim_token=claim_token)
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def renew_container_claim(
    uow: UnitOfWork, container_id: str, *, claim_token: str, until: datetime
) -> bool:
    """Extend a claim the caller still holds, returning whether it still does.

    The heartbeat a long-running request sends so a healthy tool loop is not
    taken over for outliving the claim deadline. A request whose claim was taken
    over matches no row: it has lost ownership and its exit must write nothing.
    Moves neither clock; the claim alone is what is renewed. Flushes; the
    caller's block commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            update(SandboxContainer)
            .where(SandboxContainer.id == container_id, SandboxContainer.claim_token == claim_token)
            .values(in_use_until=until)
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def release_container_claim(uow: UnitOfWork, container_id: str, *, claim_token: str) -> bool:
    """Give a claim back without touching the clocks or the session handle.

    Only the request holding ``claim_token`` may, so one whose claim was taken
    over cannot free the new holder's claim. Returns whether the claim was this
    caller's. Flushes; the caller's block commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            update(SandboxContainer)
            .where(SandboxContainer.id == container_id, SandboxContainer.claim_token == claim_token)
            .values(in_use_until=None, claim_token=None)
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def forget_container_row(uow: UnitOfWork, container_id: str, *, claim_token: str) -> bool:
    """Drop a lease the caller still owns the claim on (its provider session is gone).

    Returns whether this caller still owned it; a stale owner deletes nothing.
    Flushes; the caller's block commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            delete(SandboxContainer).where(
                SandboxContainer.id == container_id, SandboxContainer.claim_token == claim_token
            )
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def delete_container_rows(uow: UnitOfWork, container_ids: Sequence[str]) -> None:
    """Drop the named leases regardless of claim, for the sweep and expired-read paths.

    System actions rather than an owner's: a clock the registry enforces says
    the row is gone, so a still-written token must not keep it. Flushes; the
    caller's block commits.
    """
    if not container_ids:
        return
    db = session_for(uow)
    await db.execute(delete(SandboxContainer).where(SandboxContainer.id.in_(list(container_ids))))
    await db.flush()


async def expired_container_ids(uow: UnitOfWork, *, now: datetime, batch_size: int) -> list[str]:
    """Ids of up to ``batch_size`` leases past either clock and held by no live claim.

    A request still inside its claim is actively using the sandbox, so its idle
    clock pauses until it records a lease: only rows that are free, or held by a
    claim whose deadline passed (a crashed gateway's), are swept. A claim is
    never extended past the hard clock, so the hard deadline still sweeps.
    """
    db = session_for(uow)
    result = await db.execute(
        select(SandboxContainer.id)
        .where(
            or_(SandboxContainer.expires_at <= now, SandboxContainer.hard_expires_at <= now),
            or_(SandboxContainer.in_use_until.is_(None), SandboxContainer.in_use_until <= now),
        )
        .order_by(SandboxContainer.expires_at)
        .limit(batch_size)
    )
    return list(result.scalars().all())
