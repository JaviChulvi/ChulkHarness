"""Persist recoverable external effect results (revision 0006)."""
from alembic import op
revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE durable_effects ADD COLUMN result_ref TEXT")
    op.execute("ALTER TABLE durable_effects ADD COLUMN dispatch_token TEXT")
    op.execute("""CREATE TABLE durable_effect_results (
        effect_id TEXT PRIMARY KEY REFERENCES durable_effects(id) ON DELETE CASCADE,
        result_json TEXT NOT NULL, result_digest TEXT NOT NULL, recorded_at TEXT NOT NULL
    )""")


def downgrade() -> None:
    raise RuntimeError("Effect-result migration is forward-only; restore a database backup to roll back")
