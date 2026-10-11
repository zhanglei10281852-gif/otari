"""Unit tests for background batch settlement: backoff, summaries and the service state machine."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.models.inference import BatchLocalStatus
from gateway.services.inference import (
    BatchSettlementService,
    backoff_delay,
    summarize_batch_results,
)
from gateway.services.inference._batch_settlement import (
    _PROVIDER_COMPLETED,
    _PROVIDER_TERMINAL,
)


class TestBackoff:
    def test_first_retry_is_one_base(self) -> None:
        assert backoff_delay(0, base_sec=10, cap_sec=1800) == 10

    def test_doubles_each_retry(self) -> None:
        assert backoff_delay(2, base_sec=10, cap_sec=1800) == 40

    def test_is_bounded(self) -> None:
        assert backoff_delay(100, base_sec=10, cap_sec=300) == 300

    def test_negative_attempts_treated_as_zero(self) -> None:
        assert backoff_delay(-3, base_sec=10, cap_sec=1800) == 10


class TestProviderStatusMap:
    def test_terminals_map_to_distinct_local_states(self) -> None:
        assert _PROVIDER_TERMINAL["failed"] == BatchLocalStatus.FAILED
        assert _PROVIDER_TERMINAL["canceled"] == BatchLocalStatus.CANCELLED
        assert _PROVIDER_TERMINAL["cancelled"] == BatchLocalStatus.CANCELLED
        assert _PROVIDER_TERMINAL["expired"] == BatchLocalStatus.EXPIRED

    def test_completed_is_not_in_the_terminal_map(self) -> None:
        # A completed batch is billable, not a charge-free terminal.
        assert _PROVIDER_COMPLETED not in _PROVIDER_TERMINAL


class TestSummarizeBatchResults:
    def _result(self, *, prompt: int, completion: int, total: int, cached: int = 0) -> SimpleNamespace:
        usage = SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
        return SimpleNamespace(model="gpt-4o-mini", usage=usage)

    def _batch_result(self, items: list[SimpleNamespace | None]) -> SimpleNamespace:
        wrapped = [SimpleNamespace(result=item) for item in items]
        return SimpleNamespace(results=wrapped)

    def test_sums_tokens_and_keeps_lines(self) -> None:
        batch = self._batch_result(
            [self._result(prompt=100, completion=50, total=150), self._result(prompt=20, completion=5, total=25)]
        )
        totals = summarize_batch_results(batch)
        assert totals.prompt_tokens == 120
        assert totals.completion_tokens == 55
        assert totals.total_tokens == 175
        assert totals.lines == [(100, 50, 0), (20, 5, 0)]
        assert totals.model == "gpt-4o-mini"

    def test_carries_cached_tokens(self) -> None:
        batch = self._batch_result([self._result(prompt=100, completion=5, total=105, cached=80)])
        totals = summarize_batch_results(batch)
        assert totals.cache_read_tokens == 80
        assert totals.lines == [(100, 5, 80)]

    def test_items_without_result_or_usage_are_ignored(self) -> None:
        batch = self._batch_result([None, SimpleNamespace(model=None, usage=None)])
        totals = summarize_batch_results(batch)
        assert totals.total_tokens == 0
        assert totals.lines == []
        assert totals.model == "batch"


# ---------------------------------------------------------------------------
# Service state machine
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


class _FakeUow:
    """No-op async context manager: the fake repositories record the writes."""

    async def __aenter__(self) -> _FakeUow:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakeBatchRepo:
    def __init__(self, *, lease_claim_wins: bool = True) -> None:
        self.lease_claim_wins = lease_claim_wins
        self.claimed: list[dict[str, object]] = []
        self.deferred: list[dict[str, object]] = []
        self.terminals: list[BatchLocalStatus] = []
        self.settled: list[str] = []
        self.results_gate_claims = 0

    async def list_due(self, now: datetime, *, limit: int) -> list[SimpleNamespace]:
        return list(self._due)

    async def claim_lease(self, batch_id: str, **kwargs: object) -> bool:
        if not self.lease_claim_wins:
            return False
        self.claimed.append({"batch_id": batch_id, **kwargs})
        return True

    async def claim_results(self, batch_id: str, *, now: datetime) -> bool:
        self.results_gate_claims += 1
        return not self._gate_lost

    async def defer(self, batch_id: str, **kwargs: object) -> bool:
        self.deferred.append({"batch_id": batch_id, **kwargs})
        return True

    async def finish_terminal(self, batch_id: str, *, token: str, status: BatchLocalStatus) -> bool:
        self.terminals.append(status)
        return True

    async def settle(self, batch_id: str, *, token: str) -> bool:
        self.settled.append(batch_id)
        return True

    def build_results_usage_log(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


def _record(batch_id: str = "b1", *, attempt: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        id=batch_id,
        provider="openai",
        workspace_id=None,
        attempt_count=attempt,
        api_key_id="k1",
        user_id="u1",
    )


def _build_service(
    batch_repo: _FakeBatchRepo,
    *,
    records: list[SimpleNamespace],
    ports: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    provider_batch: SimpleNamespace | None = None,
) -> BatchSettlementService:
    batch_repo._due = records  # type: ignore[attr-defined]
    batch_repo._gate_lost = False  # type: ignore[attr-defined]
    uow = _FakeUow()
    time_repo = SimpleNamespace(get_database_time=AsyncMock(return_value=_NOW))
    repos = SimpleNamespace(idempotency=time_repo, batch=batch_repo)
    log_writer = SimpleNamespace(put=AsyncMock())
    config = SimpleNamespace(
        batch_settlement_lease_sec=60,
        batch_settlement_backoff_base_sec=10,
        batch_settlement_backoff_cap_sec=1800,
    )
    service = BatchSettlementService(uow, repos, ports, config, log_writer, owner="worker-1")  # type: ignore[arg-type]

    monkeypatch.setattr(
        "gateway.services.inference._batch_settlement.resolve_batch_provider",
        lambda *args, **kwargs: (SimpleNamespace(value="openai"), {}),
    )
    monkeypatch.setattr(
        "any_llm.api.aretrieve_batch",
        AsyncMock(return_value=provider_batch or SimpleNamespace(status="in_progress")),
    )
    return service


def _ports(
    *,
    pricing: object | None = None,
    workspace: uuid.UUID | None = None,
    org: uuid.UUID | None = None,
    exempt: bool = False,
) -> tuple[SimpleNamespace, list[tuple[str, object]]]:
    folds: list[tuple[str, object]] = []

    async def fold_spend(user_id: str, cost: object) -> None:
        folds.append((user_id, cost))

    ports = SimpleNamespace(
        find_pricing=AsyncMock(return_value=pricing),
        fold_spend=fold_spend,
        workspace_for_key=AsyncMock(return_value=workspace or uuid.UUID("00000000-0000-0000-0000-000000000001")),
        organization_for_workspace=AsyncMock(return_value=org),
        origin_key_exempt=AsyncMock(return_value=exempt),
    )
    return ports, folds


class TestServiceAdvance:
    @pytest.mark.asyncio
    async def test_in_progress_provider_batch_is_deferred_with_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        ports, folds = _ports()
        service = _build_service(repo, records=[_record()], ports=ports, monkeypatch=monkeypatch)

        report = await service.run_once(limit=10)

        assert report.deferred == 1
        assert repo.deferred[0]["next_poll_at"] == _NOW + timedelta(seconds=10)
        assert repo.deferred[0]["attempt"] == 1
        assert repo.settled == []
        assert folds == []

    @pytest.mark.asyncio
    async def test_completed_batch_is_billed_once_and_settled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        pricing = SimpleNamespace()
        ports, folds = _ports(pricing=pricing)
        service = _build_service(
            repo,
            records=[_record()],
            ports=ports,
            monkeypatch=monkeypatch,
            provider_batch=SimpleNamespace(status="completed"),
        )
        result_item = SimpleNamespace(
            result=SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )
        )
        monkeypatch.setattr(
            "any_llm.api.aretrieve_batch_results",
            AsyncMock(return_value=SimpleNamespace(results=[result_item])),
        )
        # Price through the same cost core, with the arithmetic pinned for the test.
        monkeypatch.setattr(
            "gateway.services.inference._batch_settlement.calculate_token_cost",
            lambda *args, **kwargs: Decimal("0.0002"),
        )

        report = await service.run_once(limit=10)

        assert report.settled == 1
        assert repo.results_gate_claims == 1
        assert repo.settled == ["b1"]
        assert folds == [("u1", Decimal("0.0002"))]
        ports.find_pricing.assert_awaited_once()
        args = ports.find_pricing.await_args
        assert args.args[:2] == ("openai", "gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_completed_batch_without_pricing_defers_without_claiming_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        ports, folds = _ports(pricing=None)
        service = _build_service(
            repo,
            records=[_record()],
            ports=ports,
            monkeypatch=monkeypatch,
            provider_batch=SimpleNamespace(status="completed"),
        )
        # Results carry billable tokens.
        result_item = SimpleNamespace(
            result=SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )
        )
        monkeypatch.setattr(
            "any_llm.api.aretrieve_batch_results",
            AsyncMock(return_value=SimpleNamespace(results=[result_item])),
        )

        report = await service.run_once(limit=10)

        assert report.deferred == 1
        assert repo.results_gate_claims == 0
        assert repo.settled == []
        assert folds == []

    @pytest.mark.asyncio
    async def test_completed_batch_whose_gate_was_lost_is_settled_without_billing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        pricing = SimpleNamespace()
        ports, folds = _ports(pricing=pricing)
        service = _build_service(
            repo,
            records=[_record()],
            ports=ports,
            monkeypatch=monkeypatch,
            provider_batch=SimpleNamespace(status="completed"),
        )
        repo._gate_lost = True  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "any_llm.api.aretrieve_batch_results",
            AsyncMock(return_value=SimpleNamespace(results=[])),
        )

        report = await service.run_once(limit=10)

        assert report.settled == 1
        assert repo.settled == ["b1"]
        assert folds == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider_status", "local_status"),
        [
            ("failed", BatchLocalStatus.FAILED),
            ("canceled", BatchLocalStatus.CANCELLED),
            ("cancelled", BatchLocalStatus.CANCELLED),
            ("expired", BatchLocalStatus.EXPIRED),
        ],
    )
    async def test_terminal_provider_statuses_finish_without_billing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_status: str,
        local_status: BatchLocalStatus,
    ) -> None:
        repo = _FakeBatchRepo()
        ports, folds = _ports()
        service = _build_service(
            repo,
            records=[_record()],
            ports=ports,
            monkeypatch=monkeypatch,
            provider_batch=SimpleNamespace(status=provider_status),
        )

        report = await service.run_once(limit=10)

        assert report.terminal == 1
        assert repo.terminals == [local_status]
        assert repo.results_gate_claims == 0
        assert folds == []

    @pytest.mark.asyncio
    async def test_provider_failure_is_deferred_not_settled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        ports, folds = _ports()
        service = _build_service(repo, records=[_record()], ports=ports, monkeypatch=monkeypatch)
        monkeypatch.setattr(
            "any_llm.api.aretrieve_batch",
            AsyncMock(side_effect=RuntimeError("502 from provider")),
        )

        report = await service.run_once(limit=10)

        assert report.deferred == 1
        assert repo.settled == []
        assert folds == []
        assert "502 from provider" in str(repo.deferred[0]["error"])

    @pytest.mark.asyncio
    async def test_batch_not_complete_is_deferred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from any_llm.exceptions import BatchNotCompleteError

        repo = _FakeBatchRepo()
        ports, folds = _ports()
        service = _build_service(
            repo,
            records=[_record()],
            ports=ports,
            monkeypatch=monkeypatch,
            provider_batch=SimpleNamespace(status="completed"),
        )
        monkeypatch.setattr(
            "any_llm.api.aretrieve_batch_results",
            AsyncMock(side_effect=BatchNotCompleteError(batch_id="b1", status="completed")),
        )

        report = await service.run_once(limit=10)

        assert report.deferred == 1
        assert repo.settled == []
        assert folds == []

    @pytest.mark.asyncio
    async def test_failed_lease_claim_is_skipped_without_calling_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retrieve = AsyncMock()
        monkeypatch.setattr("any_llm.api.aretrieve_batch", retrieve)
        repo = _FakeBatchRepo(lease_claim_wins=False)
        ports, _folds = _ports()
        service = _build_service(repo, records=[_record()], ports=ports, monkeypatch=monkeypatch)

        report = await service.run_once(limit=10)

        assert report.skipped == 1
        retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_bad_record_does_not_stop_the_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _FakeBatchRepo()
        ports, _folds = _ports()
        service = _build_service(
            repo,
            records=[_record("b1"), _record("b2")],
            ports=ports,
            monkeypatch=monkeypatch,
        )

        report = await service.run_once(limit=10)

        assert report.deferred == 2
        assert [d["batch_id"] for d in repo.deferred] == ["b1", "b2"]


class TestBuildBatchSettlement:
    def test_returns_none_when_disabled(self) -> None:
        from gateway.main import _build_batch_settlement

        config = SimpleNamespace(
            batch_settlement_enabled=False,
            batch_settlement_interval_sec=60,
        )
        assert _build_batch_settlement(config, AsyncMock()) is None  # type: ignore[arg-type]

    def test_returns_none_for_nonpositive_interval(self) -> None:
        from gateway.main import _build_batch_settlement

        config = SimpleNamespace(
            batch_settlement_enabled=True,
            batch_settlement_interval_sec=0,
        )
        assert _build_batch_settlement(config, AsyncMock()) is None  # type: ignore[arg-type]
