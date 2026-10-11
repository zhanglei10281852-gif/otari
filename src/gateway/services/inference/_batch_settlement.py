"""Background settlement of locally-recorded asynchronous batches.

A recorded batch advances itself instead of waiting for a client to read its
results: the worker claims a short polling lease (so two gateway replicas
never poll the same batch together), fetches the provider batch, and either
schedules a bounded-backoff retry, records a distinct charge-free local
terminal, or bills a completed batch exactly once. Billing uses the same
pricing rules and the same atomic results gate as the manual GET results
path, so the two racing each other cannot double count. Provider batches with
no local record are outside this mechanism and keep their request-time
behavior.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from any_llm import LLMProvider
from any_llm.exceptions import BatchNotCompleteError

from gateway.core.config import GatewayConfig
from gateway.core.metered_pricing import calculate_token_cost, quantize_cost
from gateway.core.unit_of_work import UnitOfWork
from gateway.core.usage import cache_read_tokens_of
from gateway.log_config import logger
from gateway.models.inference import BatchLocalStatus, BatchRecord
from gateway.models.pricing import ModelPricing
from gateway.repositories.inference import InferenceRepositories
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs

if TYPE_CHECKING:
    from datetime import datetime

# One usage-log endpoint identifier, shared with the manual results route.
USAGE_ENDPOINT_RESULTS = "/v1/batches/results"

# Provider terminal statuses that produce no result charges, mapped to the
# distinct local terminal. "canceled" is the provider spelling; "cancelled"
# tolerated for providers that use it.
_PROVIDER_TERMINAL: dict[str, BatchLocalStatus] = {
    "failed": BatchLocalStatus.FAILED,
    "canceled": BatchLocalStatus.CANCELLED,
    "cancelled": BatchLocalStatus.CANCELLED,
    "expired": BatchLocalStatus.EXPIRED,
}
# Provider status that triggers results billing. Everything else nonterminal
# (validating, in_progress, finalizing, cancelling) is retried with backoff.
_PROVIDER_COMPLETED = "completed"

# Bound on the exponent so the cap arithmetic cannot overflow on old rows.
_BACKOFF_EXPONENT_CEILING = 12


def backoff_delay(attempts_made: int, *, base_sec: float, cap_sec: float) -> float:
    """Exponential delay before the next poll, bounded at ``cap_sec``.

    ``attempts_made`` polls have already run; the first retry waits one base
    interval and it doubles from there. A bound, not a retry count: a batch is
    polled for as long as it stays in progress, never in a tight loop.
    """
    exponent = min(max(attempts_made, 0), _BACKOFF_EXPONENT_CEILING)
    return min(cap_sec, base_sec * (2**exponent))


@dataclass(frozen=True)
class BatchTotals:
    """Sums and per-line usage for a completed batch's results."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    lines: list[tuple[int, int, int]]


def summarize_batch_results(result: Any) -> BatchTotals:
    """Sum per-request token usage from batch results, keeping each line for pricing.

    Mirrors the manual path: cached prompt tokens are read the OpenAI way from
    ``prompt_tokens_details``, and pricing happens a line at a time even though
    the row reports the batch totals.
    """
    model = "batch"
    prompt_tokens = completion_tokens = total_tokens = cache_read_tokens = 0
    lines: list[tuple[int, int, int]] = []
    for item in result.results:
        usage = item.result.usage if item.result is not None else None
        if usage is None:
            continue
        line_prompt = usage.prompt_tokens or 0
        line_completion = usage.completion_tokens or 0
        line_cache_read = cache_read_tokens_of(usage)
        prompt_tokens += line_prompt
        completion_tokens += line_completion
        cache_read_tokens += line_cache_read
        total_tokens += usage.total_tokens or 0
        lines.append((line_prompt, line_completion, line_cache_read))
        if item.result is not None:
            model = item.result.model
    return BatchTotals(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        lines=lines,
    )


def resolve_batch_provider(
    config: GatewayConfig,
    provider: str,
    *,
    workspace_id: uuid.UUID | None,
) -> tuple[LLMProvider, dict[str, Any]]:
    """Resolve a batch's provider name into (implementation, call kwargs).

    A configured instance resolves to its type with the instance credentials;
    any other value is parsed as a provider and the kwargs fall back to the
    organization overlay of ``workspace_id``, the batch's own stored origin, so
    a lifecycle call reaches the same credential source the batch was created
    with. Raises UnsupportedProviderError on an unparseable provider.
    """
    if provider in config.providers:
        impl = LLMProvider(config.provider_instance_type(provider))
        return impl, get_provider_kwargs(config, impl, instance=provider)
    impl = LLMProvider.from_string(provider)
    return impl, get_provider_kwargs(config, impl, workspace_id=workspace_id)


class BatchSettlementPorts(Protocol):
    """Session-bound operations the service needs from old-shape services, wired per tick."""

    async def find_pricing(
        self,
        provider: str,
        model: str,
        organization_id: uuid.UUID | None,
    ) -> ModelPricing | None: ...

    # Folds the cost into the creator's spend through the single reconcile writer.
    async def fold_spend(self, user_id: str, cost: Decimal) -> None: ...

    # Workspace/org attribution fallbacks for records predating the workspace column.
    async def workspace_for_key(self, api_key_id: str | None) -> uuid.UUID: ...

    async def organization_for_workspace(
        self,
        workspace_id: uuid.UUID | None,
    ) -> uuid.UUID | None: ...

    # Whether the originating key was budget exempt; a missing key is enforced.
    async def origin_key_exempt(self, api_key_id: str | None) -> bool: ...


@dataclass(frozen=True)
class SettlementReport:
    """What one worker tick did, for logging and tests."""

    claimed: int = 0
    settled: int = 0
    terminal: int = 0
    deferred: int = 0
    skipped: int = 0


class BatchSettlementService:
    """Advance due recorded batches, one claimed lease and one transactional step at a time."""

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: InferenceRepositories,
        ports: BatchSettlementPorts,
        config: GatewayConfig,
        log_writer: LogWriter,
        *,
        owner: str,
    ) -> None:
        self._uow = uow
        self._repos = repositories
        self._ports = ports
        self._config = config
        self._log_writer = log_writer
        self._owner = owner

    async def _now(self) -> datetime:
        # A repository call needs an open block; this nests into an outer one
        # when called mid-step and commits on its own otherwise.
        async with self._uow:
            return await self._repos.idempotency.get_database_time()

    async def run_once(self, *, limit: int) -> SettlementReport:
        """Advance up to ``limit`` due batches, isolating failures per batch."""
        now = await self._now()
        async with self._uow:
            due = await self._repos.batch.list_due(now, limit=limit)
        report = SettlementReport()
        for record in due:
            try:
                outcome = await self._advance(record)
            except Exception:
                # An escaping error would only end this batch's processing this tick;
                # the lease lapses on its own and the next tick reclaims the batch.
                logger.warning("Unexpected error advancing batch '%s'", record.id, exc_info=True)
                outcome = "skipped"
            report = _bump(report, outcome)
        return report

    async def _advance(self, record: BatchRecord) -> str:
        """Claim the lease, read the provider batch, and branch on its status."""
        token = str(uuid.uuid4())
        now = await self._now()
        lease_end = now + timedelta(seconds=self._config.batch_settlement_lease_sec)
        async with self._uow:
            claimed = await self._repos.batch.claim_lease(
                record.id,
                owner=self._owner,
                token=token,
                now=now,
                lease_end=lease_end,
            )
        if not claimed:
            # Another replica claimed it between the read and the claim.
            return "skipped"

        try:
            provider_enum, kwargs = resolve_batch_provider(
                self._config,
                record.provider,
                workspace_id=record.workspace_id,
            )
            batch = await self._retrieve(provider_enum, record.id, kwargs)
            status = str(batch.status)
            terminal = _PROVIDER_TERMINAL.get(status)
            if terminal is not None:
                async with self._uow:
                    await self._repos.batch.finish_terminal(record.id, token=token, status=terminal)
                logger.info("Batch '%s' reached local terminal '%s'", record.id, terminal)
                return "terminal"
            if status != _PROVIDER_COMPLETED:
                await self._defer(record, token, f"provider status '{status}'")
                return "deferred"
            return await self._settle_completed(record, token, provider_enum, kwargs)
        except BatchNotCompleteError as exc:
            await self._defer(record, token, f"results not complete (status: {exc.batch_status})")
            return "deferred"
        except Exception as exc:
            # Provider 429/5xx or any temporary failure: do not settle, retry later.
            try:
                await self._defer(record, token, str(exc))
            except Exception:
                # Even the schedule write failed; the lease lapses and another tick reclaims.
                logger.warning(
                    "Could not defer batch '%s' after a failure; its lease will lapse",
                    record.id,
                    exc_info=True,
                )
            return "deferred"

    async def _retrieve(
        self,
        provider_enum: LLMProvider,
        batch_id: str,
        kwargs: dict[str, Any],
    ) -> Any:
        from any_llm.api import aretrieve_batch

        return await aretrieve_batch(provider=provider_enum, batch_id=batch_id, **kwargs)

    async def _defer(self, record: BatchRecord, token: str, error: str) -> None:
        """Schedule the bounded-backoff retry and hold the lease until due."""
        attempts = record.attempt_count + 1
        delay = backoff_delay(
            record.attempt_count,
            base_sec=self._config.batch_settlement_backoff_base_sec,
            cap_sec=self._config.batch_settlement_backoff_cap_sec,
        )
        next_poll_at = await self._now() + timedelta(seconds=delay)
        async with self._uow:
            moved = await self._repos.batch.defer(
                record.id,
                token=token,
                attempt=attempts,
                next_poll_at=next_poll_at,
                error=error,
            )
        if not moved:
            logger.info("Batch '%s' lease was taken over; skipping its deferral", record.id)

    async def _settle_completed(
        self,
        record: BatchRecord,
        token: str,
        provider_enum: LLMProvider,
        kwargs: dict[str, Any],
    ) -> str:
        """Fetch and price results, then claim the gate and bill exactly once."""
        from any_llm.api import aretrieve_batch_results

        result = await aretrieve_batch_results(provider=provider_enum, batch_id=record.id, **kwargs)
        totals = summarize_batch_results(result)

        # Attribution follows the creator: stored workspace when present,
        # otherwise the originating key's workspace, priced at that org's rates.
        workspace_id = record.workspace_id or await self._ports.workspace_for_key(record.api_key_id)
        organization_id = await self._ports.organization_for_workspace(workspace_id)
        exempt = await self._ports.origin_key_exempt(record.api_key_id)

        cost: Decimal | None = None
        if totals.total_tokens:
            pricing = await self._ports.find_pricing(provider_enum.value, totals.model, organization_id)
            if pricing is not None:
                cost = quantize_cost(
                    sum(
                        (
                            calculate_token_cost(
                                pricing,
                                input_tokens=line_prompt,
                                output_tokens=line_completion,
                                cache_read_tokens=line_cache_read,
                                cache_tokens_included=True,
                                quantize=False,
                            )
                            for line_prompt, line_completion, line_cache_read in totals.lines
                        ),
                        Decimal(0),
                    )
                )

        # Billable tokens with no price row: leave the gate unclaimed and retry,
        # exactly like the manual path, so pricing appearing later still bills once.
        if totals.total_tokens and cost is None and not exempt:
            await self._defer(record, token, "pricing not configured")
            return "deferred"

        now = await self._now()
        async with self._uow:
            won = await self._repos.batch.claim_results(record.id, now=now)
            if won:
                usage_log = self._repos.batch.build_results_usage_log(
                    record=record,
                    model=totals.model,
                    workspace_id=workspace_id,
                    prompt_tokens=totals.prompt_tokens,
                    completion_tokens=totals.completion_tokens,
                    total_tokens=totals.total_tokens,
                    cache_read_tokens=totals.cache_read_tokens,
                    cost=cost,
                    counts_toward_budget=not exempt,
                    now=now,
                )
                await self._log_writer.put(usage_log)
            # Release the lease and mark settled in the same transaction as the
            # claim whether this worker won the gate or a concurrent retrieval did.
            await self._repos.batch.settle(record.id, token=token)
            if won and cost and not exempt:
                # Commits claim, local state and the spend fold atomically.
                await self._ports.fold_spend(record.user_id, cost)
        if won:
            logger.info(
                "Settled batch '%s': %d tokens, cost %s",
                record.id,
                totals.total_tokens,
                cost,
            )
        return "settled"


def _bump(report: SettlementReport, outcome: str) -> SettlementReport:
    """Add one to the report field named by ``outcome``."""
    values = {**report.__dict__, outcome: getattr(report, outcome) + 1}
    return SettlementReport(**values)
