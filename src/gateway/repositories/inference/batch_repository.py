"""Data access for locally-recorded batches driven by the settlement worker."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import BatchLocalStatus, BatchRecord
from gateway.models.usage import UsageLog
from gateway.repositories.base_repository import BaseRepository


class BatchRepository(BaseRepository[BatchRecord, Never, Never]):
    """Discover, lease and finish recorded batches in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, BatchRecord)

    async def list_due(self, now: datetime, *, limit: int) -> list[BatchRecord]:
        """Return in-progress batches that are unleased (or lapsed) and due a poll.

        The lease claim, not this read, serializes replicas: two gateways may
        read the same batch, but only one claims it. Oldest due first, by the
        originating created time where no poll has been scheduled.
        """
        result = await self.db.execute(
            select(BatchRecord)
            .where(
                BatchRecord.local_status == BatchLocalStatus.IN_PROGRESS,
                or_(BatchRecord.leased_until.is_(None), BatchRecord.leased_until <= now),
                or_(BatchRecord.next_poll_at.is_(None), BatchRecord.next_poll_at <= now),
            )
            .order_by(func.coalesce(BatchRecord.next_poll_at, BatchRecord.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def claim_lease(
        self,
        batch_id: str,
        *,
        owner: str,
        token: str,
        now: datetime,
        lease_end: datetime,
    ) -> bool:
        """Claim the one polling lease for a batch, committing before any provider call.

        Matches while the batch is in progress and the lease is free, lapsed, or
        already this owner's. The UPDATE takes the row lock at once, so a
        concurrent claim on another gateway loses. If this owner later stalls
        mid-poll, the lease lapses at ``lease_end`` and another replica claims
        with a new token.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(BatchRecord)
                .where(
                    BatchRecord.id == batch_id,
                    BatchRecord.local_status == BatchLocalStatus.IN_PROGRESS,
                    or_(
                        BatchRecord.leased_until.is_(None),
                        BatchRecord.leased_until <= now,
                        BatchRecord.lease_owner == owner,
                    ),
                )
                .values(lease_owner=owner, lease_token=token, leased_until=lease_end, last_error=None)
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def claim_results(self, batch_id: str, *, now: datetime) -> bool:
        """Claim the one-time results gate (NULL -> now), in the settlement transaction.

        Same atomic transition the manual GET results path uses, so the two
        racing each other cannot both bill: the winner logs usage and folds
        spend, the loser sees False and only updates the local state.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(BatchRecord)
                .where(BatchRecord.id == batch_id, BatchRecord.results_accounted_at.is_(None))
                .values(results_accounted_at=now)
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def defer(
        self,
        batch_id: str,
        *,
        token: str,
        attempt: int,
        next_poll_at: datetime,
        error: str | None,
    ) -> bool:
        """Schedule the bounded-backoff retry, holding the lease until the batch is due.

        Tied to this claim's ``token`` so a worker whose lease was taken over
        cannot move a batch the new owner now drives. The lease passes exactly
        when the batch is due, letting any replica claim the retry.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(BatchRecord)
                .where(
                    BatchRecord.id == batch_id,
                    BatchRecord.lease_token == token,
                    BatchRecord.local_status == BatchLocalStatus.IN_PROGRESS,
                )
                .values(
                    attempt_count=attempt,
                    next_poll_at=next_poll_at,
                    leased_until=next_poll_at,
                    last_error=error,
                )
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def finish_terminal(self, batch_id: str, *, token: str, status: BatchLocalStatus) -> bool:
        """Record a distinct local terminal for a batch that produced no result charges.

        Reached for provider failed/cancelled/expired batches: no results are
        fetched, no usage or spend is written, and the batch never enters the
        due set again. The results billing gate stays NULL, only completed
        batches set it; ``local_status`` is what marks the batch finished.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(BatchRecord)
                .where(
                    BatchRecord.id == batch_id,
                    BatchRecord.lease_token == token,
                    BatchRecord.local_status == BatchLocalStatus.IN_PROGRESS,
                )
                .values(
                    local_status=status,
                    lease_owner=None,
                    lease_token=None,
                    leased_until=None,
                    next_poll_at=None,
                )
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def settle(self, batch_id: str, *, token: str) -> bool:
        """Mark the batch locally settled and release its lease, in the billing transaction."""
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(BatchRecord)
                .where(
                    BatchRecord.id == batch_id,
                    BatchRecord.lease_token == token,
                    BatchRecord.local_status == BatchLocalStatus.IN_PROGRESS,
                )
                .values(
                    local_status=BatchLocalStatus.SETTLED,
                    lease_owner=None,
                    lease_token=None,
                    leased_until=None,
                    next_poll_at=None,
                )
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    def build_results_usage_log(
        self,
        *,
        record: BatchRecord,
        model: str,
        workspace_id: Any,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cache_read_tokens: int,
        cost: Any,
        counts_toward_budget: bool,
        now: datetime,
    ) -> UsageLog:
        """Construct the batch results usage row with all attribution resolved.

        Built here rather than in the service because it is a row of the usage
        table; every value follows the creator (workspace, model, cost), while
        ``api_key_id`` records the originating key. The caller puts it through
        the log writer in the same transaction as the results claim.
        """
        return UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            api_key_id=record.api_key_id,
            user_id=record.user_id,
            timestamp=now,
            model=model,
            provider=record.provider,
            endpoint="/v1/batches/results",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cost=cost,
            status="success",
            counts_toward_budget=counts_toward_budget,
        )
