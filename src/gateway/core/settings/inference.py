"""Inference request settings."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED


class InferenceSettings(BaseModel):
    """How a completion request that carries an ``Idempotency-Key`` is deduplicated."""

    idempotency_retention_sec: Annotated[int, OMITTED] = Field(
        default=86400,
        ge=0,
        description=(
            "How long a non-streaming completion sent with an Idempotency-Key is kept, so that a "
            "retry with the same key returns the stored response instead of calling the provider "
            "and billing again. The stored response includes the generated content. 0 ignores the "
            "header. Standalone mode only."
        ),
    )
    idempotency_lease_sec: Annotated[int, OMITTED] = Field(
        default=60,
        ge=3,
        description=(
            "How long a claim on an Idempotency-Key stays valid without being renewed. The request "
            "holding it renews it every third of this while it runs, so this bounds how long a key "
            "stays blocked after the worker running its request dies, not how long a request may take."
        ),
    )
    idempotency_sweep_interval_sec: Annotated[int, OMITTED] = Field(
        default=3600,
        ge=1,
        description="How often expired idempotency records are deleted.",
    )

    batch_settlement_enabled: Annotated[bool, OMITTED] = Field(
        default=True,
        description=(
            "Whether locally-recorded batches are advanced and billed in the background without a client "
            "reading results. Off keeps settlement on the first GET results retrieval as before. "
            "Not available in hybrid mode."
        ),
    )
    batch_settlement_interval_sec: Annotated[int, OMITTED] = Field(
        default=60,
        ge=1,
        description="How often the background worker looks for batches due a poll.",
    )
    batch_settlement_batch_size: Annotated[int, OMITTED] = Field(
        default=10,
        ge=1,
        description="How many due batches one worker tick advances.",
    )
    batch_settlement_lease_sec: Annotated[int, OMITTED] = Field(
        default=60,
        ge=10,
        description=(
            "How long one gateway's polling lease on a batch stays valid. Past this, another replica may "
            "claim the batch if the worker holding it stalled."
        ),
    )
    batch_settlement_backoff_base_sec: Annotated[int, OMITTED] = Field(
        default=10,
        ge=1,
        description=(
            "Base delay before re-polling a still-running or temporarily obstructed batch; it doubles each retry."
        ),
    )
    batch_settlement_backoff_cap_sec: Annotated[int, OMITTED] = Field(
        default=1800,
        ge=5,
        description="Largest delay between polls of a batch, so retries are bounded and never busy-loop.",
    )
