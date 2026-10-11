"""Add ``claim_token`` to ``sandbox_containers``, the fencing identity of a claim's owner.

``in_use_until`` already says whether one request holds a container, but nothing
said which request: once a long-running request let the deadline pass, another
could take over, and the old request's record, release or delete then landed on
the row the new holder owned, clearing or overwriting its state and putting a
third request into the same workspace.

``claim_token`` is the identity every first claim and every takeover of an
expired claim mints afresh. Owner-scoped statements match it, so a request that
lost its claim cannot move the new holder's session id, clocks or occupancy.

Nullable on purpose: rows written before this migration have no owner (they are
idle, between requests), and a free row keeps the token of its last owner so the
same request can still forget a session the provider said is gone.

Revision ID: e1f2a3b4c5d6
Revises: c7e1a4d9b3f2
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "c7e1a4d9b3f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "sandbox_containers"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("claim_token", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, "claim_token")
