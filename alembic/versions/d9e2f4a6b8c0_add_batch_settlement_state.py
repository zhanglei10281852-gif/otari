"""Add local lifecycle and leasing columns to batches for background settlement.

The background worker advances locally-recorded batches itself: it claims a
polling lease, prices and bills a completed batch exactly once, and leaves a
distinct local terminal for failed/cancelled/expired ones. Existing rows are
seeded ``in_progress`` so they are discovered, while the manual GET results
path keeps its current behavior and stays the fallback when the worker is off.

Revision ID: d9e2f4a6b8c0
Revises: c7e1a4d9b3f2
Create Date: 2026-10-10 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9e2f4a6b8c0"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("batches", sa.Column("lease_owner", sa.String(length=36), nullable=True))
    op.add_column("batches", sa.Column("lease_token", sa.String(length=36), nullable=True))
    op.add_column("batches", sa.Column("leased_until", sa.DateTime(timezone=True), nullable=True))
    op.add_column("batches", sa.Column("next_poll_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("batches", sa.Column("last_error", sa.Text(), nullable=True))
    # Server defaults seed every existing row: they become in_progress with a
    # zero attempt count, the same shape newly created rows have.
    op.add_column(
        "batches",
        sa.Column("local_status", sa.String(length=16), nullable=False, server_default="in_progress"),
    )
    op.add_column(
        "batches",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(op.f("ix_batches_local_status"), "batches", ["local_status"], unique=False)
    op.create_index(op.f("ix_batches_leased_until"), "batches", ["leased_until"], unique=False)
    op.create_index(op.f("ix_batches_next_poll_at"), "batches", ["next_poll_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_batches_next_poll_at"), table_name="batches")
    op.drop_index(op.f("ix_batches_leased_until"), table_name="batches")
    op.drop_index(op.f("ix_batches_local_status"), table_name="batches")
    op.drop_column("batches", "attempt_count")
    op.drop_column("batches", "local_status")
    op.drop_column("batches", "last_error")
    op.drop_column("batches", "next_poll_at")
    op.drop_column("batches", "leased_until")
    op.drop_column("batches", "lease_token")
    op.drop_column("batches", "lease_owner")
