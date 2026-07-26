# Hosted scheduling

Local `ScheduledJob` prompt jobs remain a compatibility path for channel
adapters. Hosted applications should schedule immutable definition references
and submit durable runs when an occurrence becomes due.

`HostedScheduledOccurrence` binds:

- the host schedule ID;
- one `GatewayRunTarget` containing the exact `ExecutionScope` and published
  definition digest;
- the timezone-aware scheduled instant; and
- a stable occurrence idempotency key.

`HostedScheduleRunSubmitter` and `AsyncHostedScheduleRunSubmitter` convert that
record into an idempotent `RunSubmission`. The run contains definition and
source-event identity but not a stored raw prompt. Replaying the same
occurrence returns the same run, while recurrence calculation, calendars,
leases, and authorization remain host responsibilities.

Use the same durable worker and approval path described in
[hosted runtime](hosting.md), and the same delivery path described in
[hosted gateway](gateway.md).
