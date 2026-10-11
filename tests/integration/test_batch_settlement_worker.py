"""Integration tests for background settlement of locally-recorded batches."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.batch import BatchResult, BatchResultItem
from any_llm.types.completion import CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, API_ROOT


@contextmanager
def _patch_provider(*, retrieve: AsyncMock, results: AsyncMock) -> Generator[None]:
    """Patch the provider functions in both the route's and the SDK's namespace.

    The route binds the names at import, so patching ``any_llm.api`` only would
    not redirect manual calls; one shared mock keeps a side_effect sequence
    coherent across tick and route.
    """
    with (
        patch("gateway.api.routes.batches.aretrieve_batch", retrieve),
        patch("gateway.api.routes.batches.aretrieve_batch_results", results),
        patch("any_llm.api.aretrieve_batch", retrieve),
        patch("any_llm.api.aretrieve_batch_results", results),
    ):
        yield


async def _run_tick(app: Any, owner: str = "test-worker") -> Any:
    """Run one production-shaped settlement tick on the app's event loop."""
    from gateway.api.deps import build_batch_settlement_service
    from gateway.core.unit_of_work import create_worker_unit_of_work

    async with create_worker_unit_of_work() as (uow, db):
        service = build_batch_settlement_service(
            uow,
            db,
            app.state.config,
            app.state.log_writer,
            owner=owner,
        )
        return await service.run_once(limit=10)


def _tick(client: TestClient, owner: str = "test-worker") -> Any:
    return client.portal.call(_run_tick, client.app, owner)


def _local_status(postgres_url: str, batch_id: str) -> str:
    engine = create_engine(postgres_url)
    try:
        with engine.connect() as conn:
            value = conn.execute(
                text("SELECT local_status FROM batches WHERE id = :id"),
                {"id": batch_id},
            ).scalar_one()
    finally:
        engine.dispose()
    return str(value)


@pytest.fixture
def _zero_backoff() -> Generator[None]:
    # Retries are due immediately so a test can advance a batch tick to tick.
    with patch(
        "gateway.services.inference._batch_settlement.backoff_delay",
        return_value=0.0,
    ):
        yield


def _results(tokens: CompletionUsage | None = None) -> BatchResult:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "gpt-4o-mini",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "42"},
            }
        ],
    }
    if tokens is not None:
        payload["usage"] = tokens.model_dump()
    from any_llm.types.completion import ChatCompletion

    completion = ChatCompletion.model_validate(payload)
    return BatchResult(results=[BatchResultItem(custom_id="req-1", result=completion, error=None)])


def _create_batch(client: TestClient, header: dict[str, str], batch_id: str = "batch_bg") -> str:
    from .test_batches_endpoint import _create_batch_body, _mock_batch

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    with (
        patch(
            "gateway.api.routes.batches.acreate_batch",
            new_callable=AsyncMock,
            return_value=_mock_batch(id=batch_id),
        ),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post(f"{API_ROOT}/batches", json=_create_batch_body(), headers=header)
    assert resp.status_code == 200
    return str(resp.json()["id"])


class TestBackgroundSettlement:
    def test_completed_batch_is_billed_without_a_results_read(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        postgres_url: str,
        _zero_backoff: None,
    ) -> None:
        """The whole point: no client reads results, yet usage and spend land."""
        from .test_batches_endpoint import _mock_batch

        # Pricing exists so the completed batch bills on the second tick.
        pricing = client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": "openai:gpt-4o-mini",
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            },
            headers=master_key_header,
        )
        assert pricing.status_code == 200

        batch_id = _create_batch(client, api_key_header)
        tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        retrieve = AsyncMock(
            side_effect=[
                _mock_batch(id=batch_id, status="in_progress"),
                _mock_batch(id=batch_id, status="completed"),
            ]
        )
        results = AsyncMock(return_value=_results(tokens))
        with (
            patch("any_llm.api.aretrieve_batch", retrieve),
            patch("any_llm.api.aretrieve_batch_results", results),
        ):
            first = _tick(client)
            # Still in flight: deferred, not billed.
            assert first.deferred == 1
            usage_before = client.get(
                f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header
            ).json()
            assert [row for row in usage_before if row["endpoint"] == "/v1/batches/results"] == []

            second = _tick(client)
            assert second.settled == 1

        assert _local_status(postgres_url, batch_id) == "settled"

        rows = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
        results_rows = [row for row in rows if row["endpoint"] == "/v1/batches/results"]
        assert len(results_rows) == 1
        assert results_rows[0]["total_tokens"] == 150
        expected_cost = 100 / 1_000_000 + 100 / 1_000_000
        assert results_rows[0]["cost"] == pytest.approx(expected_cost)

        user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
        assert user["spend"] == pytest.approx(expected_cost)

    @pytest.mark.parametrize(
        ("provider_status", "local_status"),
        [
            ("failed", "failed"),
            ("cancelled", "cancelled"),
            ("expired", "expired"),
        ],
    )
    def test_terminal_batches_get_a_local_terminal_and_no_charges(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        postgres_url: str,
        provider_status: str,
        local_status: str,
    ) -> None:
        from .test_batches_endpoint import _mock_batch

        batch_id = _create_batch(client, api_key_header, batch_id=f"batch_{provider_status}")
        results_fetch = AsyncMock()
        with (
            patch(
                "any_llm.api.aretrieve_batch",
                AsyncMock(return_value=_mock_batch(id=batch_id, status=provider_status)),
            ),
            patch("any_llm.api.aretrieve_batch_results", results_fetch),
        ):
            report = _tick(client)

        assert report.terminal == 1
        results_fetch.assert_not_awaited()
        assert _local_status(postgres_url, batch_id) == local_status

        rows = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
        assert [row for row in rows if row["endpoint"] == "/v1/batches/results"] == []
        user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
        assert user["spend"] == pytest.approx(0.0)

    def test_tick_after_manual_results_settle_does_not_bill_again(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        _zero_backoff: None,
    ) -> None:
        """Manual GET results wins the gate; the later tick only observes it."""
        from .test_batches_endpoint import _mock_batch

        client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": "openai:gpt-4o-mini",
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            },
            headers=master_key_header,
        )
        batch_id = _create_batch(client, api_key_header)
        tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        retrieve = AsyncMock(return_value=_mock_batch(id=batch_id, status="completed"))
        results = AsyncMock(return_value=_results(tokens))
        with _patch_provider(retrieve=retrieve, results=results):
            manual = client.get(f"{API_ROOT}/batches/{batch_id}/results?provider=openai", headers=api_key_header)
            assert manual.status_code == 200
            report = _tick(client)

        # Already settled by the manual read: not due, nothing billed twice.
        assert report.settled == 0
        rows = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
        results_rows = [row for row in rows if row["endpoint"] == "/v1/batches/results"]
        assert len(results_rows) == 1
        user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
        assert user["spend"] == pytest.approx(100 / 1_000_000 + 100 / 1_000_000)

    def test_manual_results_after_tick_does_not_bill_again(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        _zero_backoff: None,
    ) -> None:
        """The tick wins the gate; the later manual read returns results without billing."""
        from .test_batches_endpoint import _mock_batch

        client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": "openai:gpt-4o-mini",
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            },
            headers=master_key_header,
        )
        batch_id = _create_batch(client, api_key_header)
        tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        retrieve = AsyncMock(return_value=_mock_batch(id=batch_id, status="completed"))
        results = AsyncMock(return_value=_results(tokens))
        with _patch_provider(retrieve=retrieve, results=results):
            report = _tick(client)
            assert report.settled == 1
            manual = client.get(f"{API_ROOT}/batches/{batch_id}/results?provider=openai", headers=api_key_header)
            assert manual.status_code == 200

        rows = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
        results_rows = [row for row in rows if row["endpoint"] == "/v1/batches/results"]
        assert len(results_rows) == 1
        user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
        assert user["spend"] == pytest.approx(100 / 1_000_000 + 100 / 1_000_000)

    def test_concurrent_tick_and_results_bill_exactly_once(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        _zero_backoff: None,
    ) -> None:
        """The atomic results gate is the single billing point under real concurrency."""
        import threading

        from .test_batches_endpoint import _mock_batch

        client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": "openai:gpt-4o-mini",
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            },
            headers=master_key_header,
        )
        batch_id = _create_batch(client, api_key_header)
        tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        errors: list[BaseException] = []

        def _read_results() -> None:
            try:
                resp = client.get(f"{API_ROOT}/batches/{batch_id}/results?provider=openai", headers=api_key_header)
                assert resp.status_code == 200
            except BaseException as exc:  # pragma: no cover - test diagnostics
                errors.append(exc)

        retrieve = AsyncMock(return_value=_mock_batch(id=batch_id, status="completed"))
        results = AsyncMock(return_value=_results(tokens))
        with _patch_provider(retrieve=retrieve, results=results):
            threads = [threading.Thread(target=_read_results) for _ in range(4)]
            for thread in threads:
                thread.start()
            report = _tick(client)
            for thread in threads:
                thread.join(timeout=10)

        assert not errors
        # A reader may have settled the batch before the tick, or the tick won:
        # either way only one bill exists, asserted below.
        assert report.settled in {0, 1}
        rows = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
        results_rows = [row for row in rows if row["endpoint"] == "/v1/batches/results"]
        assert len(results_rows) == 1
        user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
        assert user["spend"] == pytest.approx(100 / 1_000_000 + 100 / 1_000_000)

    def test_legacy_batches_with_no_record_are_not_scanned(
        self,
        client: TestClient,
    ) -> None:
        """No local record means the worker never asks the provider about the batch."""
        retrieve = AsyncMock()
        with patch("any_llm.api.aretrieve_batch", retrieve):
            report = _tick(client)

        assert report.settled == 0
        assert report.terminal == 0
        assert report.deferred == 0
        retrieve.assert_not_awaited()

    def test_lapsed_lease_is_taken_over_by_another_owner_and_billed(
        self,
        client: TestClient,
        api_key_header: dict[str, str],
        master_key_header: dict[str, str],
        api_key_obj: dict[str, Any],
        postgres_url: str,
    ) -> None:
        """After a worker stalls past its lease, a different replica claims and settles."""
        from .test_batches_endpoint import _mock_batch

        client.post(
            f"{API_ROOT}/pricing",
            json={
                "model_key": "openai:gpt-4o-mini",
                "input_price_per_million": 1.0,
                "output_price_per_million": 2.0,
            },
            headers=master_key_header,
        )
        batch_id = _create_batch(client, api_key_header, batch_id="batch_lease")
        tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        # First tick: the provider call fails, so the batch is deferred with the
        # lease held until its next poll and the error recorded.
        retrieve = AsyncMock(
            side_effect=[
                RuntimeError("502 upstream"),
                _mock_batch(id=batch_id, status="completed"),
            ]
        )
        results = AsyncMock(return_value=_results(tokens))
        with (
            patch("any_llm.api.aretrieve_batch", retrieve),
            patch("any_llm.api.aretrieve_batch_results", results),
        ):
            first = _tick(client, owner="worker-1")
            assert first.deferred == 1

        # Simulate reaching the scheduled retry time after the lease lapsed and
        # worker-1 is gone (process restart): another owner may claim it.
        engine = create_engine(postgres_url)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE batches SET leased_until = NOW() - INTERVAL '1 second', "
                        "next_poll_at = NOW() - INTERVAL '1 second' WHERE id = :id"
                    ),
                    {"id": batch_id},
                )
        finally:
            engine.dispose()

        with (
            patch("any_llm.api.aretrieve_batch", retrieve),
            patch("any_llm.api.aretrieve_batch_results", results),
        ):
            second = _tick(client, owner="worker-2")

        assert second.settled == 1
        assert _local_status(postgres_url, batch_id) == "settled"
        rows = client.get(
            f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header
        ).json()
        results_rows = [row for row in rows if row["endpoint"] == "/v1/batches/results"]
        assert len(results_rows) == 1
        user = client.get(
            f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header
        ).json()
        assert user["spend"] == pytest.approx(100 / 1_000_000 + 100 / 1_000_000)


class TestEndToEndWorker:
    def test_short_interval_worker_bills_without_a_results_read(
        self,
        postgres_url: str,
        master_key_header: dict[str, str],
        clean_database: None,
    ) -> None:
        from gateway.core.config import GatewayConfig

        from .conftest import build_test_client
        from .test_batches_endpoint import _create_batch_body, _mock_batch

        config = GatewayConfig(
            database_url=postgres_url,
            master_key="test-master-key",
            host="127.0.0.1",
            port=8000,
            auto_migrate=False,
            require_pricing=False,
            batch_settlement_enabled=True,
            batch_settlement_interval_sec=1,
        )

        class _SupportsBatch:
            SUPPORTS_BATCH = True

        @contextmanager
        def _client() -> Generator[TestClient]:
            yield from build_test_client(config)

        with _client() as client:
            # A user the master-key batch bills, plus the price row.
            user_resp = client.post(
                f"{API_ROOT}/users",
                json={"user_id": "batch-e2e-user"},
                headers=master_key_header,
            )
            assert user_resp.status_code == 200

            pricing = client.post(
                f"{API_ROOT}/pricing",
                json={
                    "model_key": "openai:gpt-4o-mini",
                    "input_price_per_million": 1.0,
                    "output_price_per_million": 2.0,
                },
                headers=master_key_header,
            )
            assert pricing.status_code == 200

            tokens = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            retrieve = AsyncMock(return_value=_mock_batch(id="batch_e2e", status="completed"))
            results = AsyncMock(return_value=_results(tokens))
            # Provider calls patched before the batch exists, so no 1s tick can
            # hit an unmocked provider and push the batch onto a long backoff.
            with _patch_provider(retrieve=retrieve, results=results):
                with (
                    patch(
                        "gateway.api.routes.batches.acreate_batch",
                        new_callable=AsyncMock,
                        return_value=_mock_batch(id="batch_e2e"),
                    ),
                    patch(
                        "gateway.api.routes.batches.AnyLLM.get_provider_class",
                        return_value=_SupportsBatch,
                    ),
                ):
                    created = client.post(
                        f"{API_ROOT}/batches",
                        json=_create_batch_body(user="batch-e2e-user"),
                        headers={API_KEY_HEADER: "Bearer test-master-key"},
                    )
                    assert created.status_code == 200

                # The 1s worker settles it on its own; poll for the spend.
                import time

                deadline = time.monotonic() + 20
                spend = 0.0
                while time.monotonic() < deadline:
                    user = client.get(f"{API_ROOT}/users/batch-e2e-user", headers=master_key_header)
                    if user.status_code == 200:
                        spend = float(user.json()["spend"])
                        if spend > 0:
                            break
                    time.sleep(0.5)

            assert spend == pytest.approx(100 / 1_000_000 + 100 / 1_000_000)
