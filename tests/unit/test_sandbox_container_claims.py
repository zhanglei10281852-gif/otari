"""Fencing on a resumable sandbox container's claim.

The claim token is what keeps a request whose deadline passed while it was
still running from writing the row a takeover now owns: renew, record, release
and forget all match it, and the conditional row count is the fence. These run
on SQLite with just the ``sandbox_containers`` table; the end-to-end resume
behavior lives in ``tests/integration/test_sandbox_container_reuse.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, init_db, reset_db
from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.base import Base
from gateway.models.tools import SandboxContainer
from gateway.repositories.code_execution import (
    SandboxContainerRow,
    claim_container_row,
    get_container_row,
    upsert_container_row,
)
from gateway.services.code_execution import (
    ContainerBusyError,
    ContainerLease,
    ContainerNotFoundError,
    SandboxContainerRegistry,
)

USER = "user-1"
WORKSPACE = uuid.uuid4()
PROVIDER = "fake-provider"
CONTAINER = "otari_cntr_fenced000000000000000000001"


@pytest_asyncio.fixture
async def database(tmp_path: Path) -> AsyncIterator[None]:
    reset_db()
    init_db(GatewayConfig(database_url=f"sqlite+aiosqlite:///{tmp_path / 'claims.db'}", auto_migrate=False))
    async with create_session() as session:
        uow = UnitOfWork(session)
        async with uow:
            db = session_for(uow)
            # The lease table cascades to users and workspaces; create just the
            # parent keys this table points at, plus the rows the seeds use.
            await db.execute(text("CREATE TABLE users (user_id VARCHAR PRIMARY KEY)"))
            await db.execute(text("CREATE TABLE workspace (id CHAR(32) PRIMARY KEY)"))
            await db.execute(text("INSERT INTO users (user_id) VALUES (:user_id)"), {"user_id": USER})
            await db.execute(
                text("INSERT INTO workspace (id) VALUES (:workspace_id)"),
                {"workspace_id": WORKSPACE.hex},
            )
            table = Base.metadata.tables[SandboxContainer.__tablename__]
            await db.run_sync(lambda sync: table.create(sync.connection()))
    yield
    reset_db()


async def _registry(session: AsyncSession) -> SandboxContainerRegistry:
    return SandboxContainerRegistry(
        uow=UnitOfWork(session),
        user_id=USER,
        workspace_id=WORKSPACE,
        provider=PROVIDER,
        idle_ttl_s=600,
        max_lifetime_s=3600,
    )


def _row(
    container_id: str = CONTAINER,
    *,
    now: datetime | None = None,
    expires_in_s: float = 600.0,
    hard_in_s: float = 3600.0,
    in_use_until: datetime | None = None,
    claim_token: str | None = None,
    provider_session_id: str = "sbx-1",
) -> SandboxContainerRow:
    now = now or datetime.now(UTC)
    return SandboxContainerRow(
        id=container_id,
        user_id=USER,
        workspace_id=WORKSPACE,
        provider=PROVIDER,
        provider_session_id=provider_session_id,
        created_at=now - timedelta(minutes=10),
        last_used_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(seconds=expires_in_s),
        hard_expires_at=now + timedelta(seconds=hard_in_s),
        in_use_until=in_use_until,
        claim_token=claim_token,
    )


async def _seed(session: AsyncSession, row: SandboxContainerRow) -> None:
    uow = UnitOfWork(session)
    async with uow:
        await upsert_container_row(uow, row)


def _lease(token: str, *, hard_in_s: float = 3600.0, session_id: str = "sbx-1") -> ContainerLease:
    now = datetime.now(UTC)
    return ContainerLease(
        container_id=CONTAINER,
        provider=PROVIDER,
        provider_session_id=session_id,
        expires_at=now + timedelta(seconds=600),
        hard_expires_at=now + timedelta(seconds=hard_in_s),
        claim_token=token,
    )


@pytest.mark.asyncio
async def test_resolve_mints_a_token_and_a_second_resolve_is_busy(database: None) -> None:
    async with create_session() as session:
        await _seed(session, _row())
        registry = await _registry(session)

        lease = await registry.resolve(CONTAINER)

        assert lease.claim_token is not None
        uow = UnitOfWork(session)
        async with uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None
        assert stored.in_use_until is not None
        assert stored.claim_token == lease.claim_token

        with pytest.raises(ContainerBusyError):
            await registry.resolve(CONTAINER)


@pytest.mark.asyncio
async def test_a_live_claim_holds_an_idle_expired_row_as_busy(database: None) -> None:
    """A long request past the idle clock is still using the sandbox, not dead."""
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(
            session,
            _row(now=now, expires_in_s=-10, in_use_until=now + timedelta(minutes=10), claim_token="held"),
        )
        registry = await _registry(session)

        with pytest.raises(ContainerBusyError):
            await registry.resolve(CONTAINER)

        # The row was not dropped as an expired one.
        uow = UnitOfWork(session)
        async with uow:
            assert await get_container_row(uow, CONTAINER) is not None


@pytest.mark.asyncio
async def test_an_idle_expired_row_with_no_live_claim_is_dropped(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, expires_in_s=-10))
        registry = await _registry(session)

        with pytest.raises(ContainerNotFoundError):
            await registry.resolve(CONTAINER)

        uow = UnitOfWork(session)
        async with uow:
            assert await get_container_row(uow, CONTAINER) is None


@pytest.mark.asyncio
async def test_takeover_mints_a_new_token_and_every_old_token_write_is_a_noop(database: None) -> None:
    """The race in the bug report: the old request must not touch the new holder's row."""
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(
            session,
            _row(
                now=now,
                in_use_until=now - timedelta(seconds=1),
                claim_token="old-token",
                provider_session_id="sbx-1",
            ),
        )
        registry = await _registry(session)

        taken = await registry.resolve(CONTAINER)
        assert taken.claim_token != "old-token"

        old = _lease("old-token")
        # Renewal is refused first.
        assert await registry.renew(old) is False
        # Recording a completed run frees nothing and changes no field.
        assert await registry.record(old) is False
        # Releasing frees nothing.
        async with UnitOfWork(session) as uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None
        assert stored.claim_token == taken.claim_token
        assert stored.in_use_until is not None and stored.in_use_until > now
        assert stored.provider_session_id == "sbx-1"
        # Forgetting deletes nothing.
        await registry.forget(CONTAINER, claim_token="old-token")
        uow = UnitOfWork(session)
        async with uow:
            assert await get_container_row(uow, CONTAINER) is not None


@pytest.mark.asyncio
async def test_normal_record_frees_the_claim_keeps_the_token_and_refreshes_clocks(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(
            session,
            _row(
                now=now,
                expires_in_s=10,
                in_use_until=now + timedelta(hours=1),
                claim_token="owner",
                provider_session_id="sbx-1",
            ),
        )
        registry = await _registry(session)

        held = await registry.record(_lease("owner"))
        assert held is True

        uow = UnitOfWork(session)
        async with uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None
        assert stored.in_use_until is None, "recording gives the sandbox back to the next request"
        assert stored.claim_token == "owner", "the last owner stays named so it can forget a gone session"
        assert stored.provider_session_id == "sbx-1"
        assert stored.expires_at > now + timedelta(seconds=500), "the idle clock restarts"
        assert stored.hard_expires_at <= now + timedelta(seconds=3601), "the hard clock never restarts"


@pytest.mark.asyncio
async def test_forget_after_record_drops_the_gone_session_mapping(database: None) -> None:
    """Provider session gone mid-run: record, then forget by the same owner deletes."""
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, in_use_until=now + timedelta(hours=1), claim_token="owner"))
        registry = await _registry(session)

        assert await registry.record(_lease("owner")) is True
        await registry.forget(CONTAINER, claim_token="owner")

        uow = UnitOfWork(session)
        async with uow:
            assert await get_container_row(uow, CONTAINER) is None


@pytest.mark.asyncio
async def test_forget_after_a_new_takeover_leaves_the_new_holder_alone(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, in_use_until=now + timedelta(hours=1), claim_token="owner"))
        registry = await _registry(session)

        # The old request finishes and records; the next request then claims.
        assert await registry.record(_lease("owner")) is True
        later = datetime.now(UTC)
        async with UnitOfWork(session) as uow:
            assert await claim_container_row(
                uow,
                CONTAINER,
                now=later,
                until=later + timedelta(hours=1),
                claim_token="next-owner",
            )

        # The provider tells the old request its session is gone a beat too late.
        await registry.forget(CONTAINER, claim_token="owner")

        async with UnitOfWork(session) as uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None and stored.claim_token == "next-owner"


@pytest.mark.asyncio
async def test_release_is_owner_scoped(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, in_use_until=now + timedelta(minutes=5), claim_token="owner"))
        registry = await _registry(session)

        await registry.release(CONTAINER, claim_token="someone-else")
        uow = UnitOfWork(session)
        async with uow:
            still_claimed = await get_container_row(uow, CONTAINER)
        assert still_claimed is not None and still_claimed.in_use_until is not None

        await registry.release(CONTAINER, claim_token="owner")
        async with UnitOfWork(session) as uow:
            freed = await get_container_row(uow, CONTAINER)
        assert freed is not None and freed.in_use_until is None and freed.claim_token is None


@pytest.mark.asyncio
async def test_renew_extends_the_claim_and_is_bounded_by_the_hard_clock(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, in_use_until=now + timedelta(minutes=1), claim_token="owner"))
        registry = await _registry(session)

        assert await registry.renew(_lease("owner")) is True
        async with UnitOfWork(session) as uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None and stored.in_use_until is not None
        assert stored.in_use_until > now + timedelta(minutes=50)
        # Neither clock moves on renewal.
        assert stored.expires_at <= now + timedelta(seconds=601)

        # At the hard clock there is nothing left to renew.
        assert await registry.renew(_lease("owner", hard_in_s=-1)) is False


@pytest.mark.asyncio
async def test_a_fresh_lease_is_inserted_with_no_claim(database: None) -> None:
    async with create_session() as session:
        registry = await _registry(session)
        fresh = registry.lease("otari_cntr_brandnew0000000000000000000", "sbx-9", resumed=None)
        assert fresh.claim_token is None

        assert await registry.record(fresh) is True

        uow = UnitOfWork(session)
        async with uow:
            stored = await get_container_row(uow, fresh.container_id)
        assert stored is not None
        assert stored.in_use_until is None and stored.claim_token is None
        assert stored.provider_session_id == "sbx-9"


@pytest.mark.asyncio
async def test_a_claim_is_never_extended_past_the_hard_clock(database: None) -> None:
    now = datetime.now(UTC)
    async with create_session() as session:
        await _seed(session, _row(now=now, hard_in_s=10))
        registry = await _registry(session)

        lease = await registry.resolve(CONTAINER)

        uow = UnitOfWork(session)
        async with uow:
            stored = await get_container_row(uow, CONTAINER)
        assert stored is not None and stored.in_use_until is not None
        assert stored.in_use_until <= lease.hard_expires_at
