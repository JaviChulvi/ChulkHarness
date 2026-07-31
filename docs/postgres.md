# PostgreSQL hosted persistence

Install the optional production reference without changing the lightweight
local SDK installation:

```bash
python -m pip install "chulkharness[postgres]"
```

The extra installs SQLAlchemy Core, psycopg 3, and Alembic. It provides
scope-partitioned sync and native-async stores for durable runs, approvals,
gateway inbox/outbox delivery, and schedule execution. PostgreSQL does not
replace host identity, authorization, secrets, audit retention, or artifacts.

## Engines and migrations

```python
from chulk.postgres import (
    AsyncPostgreSQLRunStore,
    PostgreSQLApprovalStore,
    PostgreSQLGatewayStore,
    PostgreSQLRunStore,
    create_async_postgres_engine,
    create_postgres_engine,
    upgrade_postgres,
)

url = "postgresql+psycopg://chulk@db.example/chulk"
engine = create_postgres_engine(url)
upgrade_postgres(engine)

runs = PostgreSQLRunStore(engine)
approvals = PostgreSQLApprovalStore(engine)
gateway = PostgreSQLGatewayStore(engine)

async_engine = create_async_postgres_engine(url)
async_runs = AsyncPostgreSQLRunStore(async_engine)
```

Run `upgrade_postgres(engine)` during a controlled deployment before workers
start. Migrations are ordered and forward-only. Test upgrades against a restored
production backup; rollback means restoring that backup and its matching
application release.

Revision `0003` adds the parent policy, child lineage/progress, and parent
completion outbox tables. Deploy it before enabling parent/child API calls.
The migration is additive; existing ordinary runs remain unchanged and are
not inferred as parents from `ExecutionScope.parent_run_id`.

Use `ingest_and_submit_run(...)` to commit inbound acceptance and durable run
submission together. Use `complete_run_and_enqueue(...)` to commit a terminal
run and outbound delivery together. Native-async equivalents are included. If
either ledger rejects the operation, the transaction rolls back.

## Pool and timeout policy

Factories enable connection liveness checks and default to five pooled plus ten
overflow connections, a 30-second checkout timeout, and 30-minute recycling.
Keep the total below the database connection budget:

```text
(pool_size + max_overflow) × worker_processes + admin connections
```

Set finite database-side timeouts through psycopg options:

```python
engine = create_postgres_engine(
    url,
    pool_size=8,
    max_overflow=4,
    connect_args={
        "options": (
            "-cstatement_timeout=30000 "
            "-clock_timeout=5000 "
            "-cidle_in_transaction_session_timeout=30000"
        )
    },
)
```

Close shared engines with `engine.dispose()` or
`await async_engine.dispose()`.

## Indexes and workers

The migration indexes durable scope/claim ordering, approval/effect status,
gateway FIFO and reconciliation, and due schedules/triggers. Run and gateway
claims use PostgreSQL row locks with `SKIP LOCKED`; all transitions retain the
existing revisions, leases, uniqueness, and stale-worker checks.

Parent child submission locks the parent row before checking cumulative budget
and fan-out. Parent completion claims use `SKIP LOCKED`; an expired claim is
quarantined as `unknown` because its external delivery outcome may be
ambiguous.

Idempotency uniqueness uses SHA-256 expression indexes over the complete owner
scope and key. The original text remains stored and is compared on every
replay, while companion hash indexes keep equality lookups bounded even for
large, poorly compressible keys.

Inspect plans after representative data volume and run `ANALYZE` after large
migrations or cleanup. Add deployment-specific partial indexes only after
measurement; never remove the shipped uniqueness indexes.

## Backup and cleanup

- Maintain encrypted backups and point-in-time recovery for the host's recovery
  objectives; verify restores before upgrades.
- Record application version, Alembic revision, and backup timestamp together.
- Never delete active leases, pending approvals, unresolved effects, queued
  inbox records, pending parent completions, unknown deliveries, or
  reconciliation-required outcomes.
- Apply retention only to terminal aggregates after audit/support requirements.
  Delete in bounded batches and preserve foreign-key order.
- Monitor old leases, unknown effects, reconciliation queues, approvals, dead
  letters, pool saturation, lock waits, and table/index bloat.

Timestamps remain canonical UTC text for exact compatibility with public records
and SQLite adapters. Run/approval access validates immutable
`ExecutionScope`; gateway and schedules use explicit host-owned profiles.

## Contract gate

```bash
CHULK_POSTGRES_TEST_URL="postgresql+psycopg://postgres:postgres@localhost/chulk" \
  python -m pytest chulk/tests/test_postgres_reference.py -q
```

The gate covers migration, sync/async parity, scope isolation, duplicates,
concurrent workers, leases, effect reconciliation, ordered delivery, and atomic
ownership transfer. It also runs the reusable sync and native-async
parent/child contract, including scope isolation, progress ordering,
approval and retry/resume, bidirectional cancellation, aggregation, and
exactly-once parent delivery.
