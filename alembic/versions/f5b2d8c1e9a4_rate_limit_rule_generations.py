"""Bind each stored rate-limit rule's counts to its row instance with a generation.

Adds the nullable ``generation`` a stored rule's RPM/TPM windows and concurrency
leases are keyed by. A row written before this column keeps NULL, the
compatibility generation: its counts and slots keep the keys they always used,
deterministically and across restarts. Every rule stored afterwards mints a
fresh generation, so a rule deleted and recreated under the same name starts
from empty windows and an empty concurrency set, and the deleted generation's
late settle and release cannot touch the new one.

Revision ID: f5b2d8c1e9a4
Revises: c7e1a4d9b3f2
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f5b2d8c1e9a4"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``generation``; existing rows stay NULL, which keeps their original counter keys."""
    op.add_column("rate_limit_rules", sa.Column("generation", sa.String(), nullable=True))


def downgrade() -> None:
    """Drop ``generation``; rules go back to counters keyed by name."""
    with op.batch_alter_table("rate_limit_rules") as batch:
        batch.drop_column("generation")
