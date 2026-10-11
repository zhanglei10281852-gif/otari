"""The periodic loop that drives background batch settlement until shutdown."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from gateway.log_config import logger


async def run_batch_settlement_worker(
    interval: float,
    tick: Callable[[], Awaitable[None]],
) -> None:
    """Run one settlement tick every ``interval``; failures are retried on the next tick.

    Each tick opens its own Unit of Work inside ``tick`` (wired by the caller),
    so a failing tick never leaves a transaction open. The fixed sleep, plus the
    per-batch backoff written by the service, means a provider outage or missing
    price causes waiting rather than a busy loop.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            # An escaping error would end this worker, and nothing restarts it.
            logger.warning("Batch settlement tick failed; retrying in %ss", interval, exc_info=True)
