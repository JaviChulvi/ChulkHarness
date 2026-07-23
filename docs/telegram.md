# Telegram adapter

The Telegram adapter lets an allowlisted user talk to a server-hosted Chulk
agent from a private Telegram chat. It uses outbound Bot API long polling, so
the server does not need a public HTTP port, domain, TLS certificate, or
webhook.

## Create and authorize the bot

1. In Telegram, open `@BotFather`, run `/newbot`, and copy the bot token.
2. Obtain your numeric Telegram user id from Telegram's Bot API or a trusted
   user-info bot.
3. Put both values in the project `.env`; never pass the token on the command
   line or commit it:

   ```dotenv
   CHULK_TELEGRAM_BOT_TOKEN=replace-with-botfather-token
   CHULK_TELEGRAM_ALLOWED_USER_IDS=123456789
   ```

Multiple users can be allowlisted with comma-separated numeric ids. The bot
ignores every other user and refuses group chats, even when an allowlisted user
sends the message. Rotate the BotFather token immediately if it appears in
chat, shell history, logs, or version control.

The same `.env` must contain the selected model provider credentials. For
Gemini, for example:

```dotenv
CHULK_LLM_PROVIDER=gemini
CHULK_MODEL=gemini-3.1-flash-lite
CHULK_GEMINI_API_KEY=replace-with-gemini-key
```

Optional web search uses Tavily and keeps its key in the same ignored `.env`:

```dotenv
CHULK_TAVILY_API_KEY=replace-with-tavily-key
CHULK_WEB_SEARCH_MAX_RESULTS=5
CHULK_TELEGRAM_TIMEZONE=Europe/Madrid
CHULK_TELEGRAM_SCHEDULER_POLL_SECONDS=5
CHULK_TELEGRAM_SCHEDULING_ENABLED=true
```

When configured, the bot adds one bounded `web_search` tool. It submits concise
queries to Tavily, returns at most 1–10 result excerpts, and tells the model to
cite the returned source URLs. It does not enable arbitrary URL fetching or
general network access. Search results are untrusted evidence and cannot grant
the agent additional permissions.

## Run

Install the selected provider and start the adapter from the project root:

```bash
python -m pip install -e '.[gemini]'
chulk-telegram
```

The adapter creates one durable SQLite conversation per private Telegram chat.
It restores that conversation after a process restart. Telegram displays the
registered command menu automatically, and the bot refreshes a typing indicator
while an agent request is running. Available commands are:

- `/new` starts a fresh conversation for the chat.
- `/status` reports the provider, model, and short conversation id.
- `/plan <request>` asks the agent to prepare an approval plan.
- `/approve` approves the pending plan.
- `/reject` rejects the pending plan.
- `/reminders` lists active one-off and recurring tasks.
- `/cancel <task-id>` cancels a task shown by `/reminders`.
- `/help` displays command help.

## Reminders and recurring tasks

Scheduling is disabled by default. Set
`CHULK_TELEGRAM_SCHEDULING_ENABLED=true` to let the Telegram adapter attach the
scheduling tools and start its delivery runner. The Telegram agent then
exposes destination-scoped scheduling tools, so an
allowlisted user can say, for example, “At 2026-07-24 09:00 remind me to call
Alex” or “Every 3600 seconds check the project status.” Local date-times use
`CHULK_TELEGRAM_TIMEZONE`; explicit ISO-8601 offsets take precedence.

Jobs are stored in the shared SQLite database. A short lease prevents two
runner iterations from deliberately claiming the same job, expired leases are
recoverable after a crash, and recurring jobs advance from their scheduled
time rather than accumulating drift. At execution time the stored prompt runs
through the chat's normal agent with the same tools and permissions, and its
answer is delivered to that Telegram chat. Failures are sanitized, retained
for inspection, and retried after a bounded delay.

Scheduling and cancellation are destination-scoped side effects. The Telegram
permission callback allows only these dedicated scheduling operations; one
chat cannot list or cancel another chat's jobs. The scheduler does not accept
cron expressions or arbitrary code.

The scheduler itself is channel-neutral and lives under `chulk.scheduling`.
It is not part of the default Python SDK `Agent` or `AsyncAgent` tool set and
does not start background work for SDK hosts. Another adapter or host can opt
in explicitly by constructing a `SQLiteScheduleStore`, binding
`scheduled_job_tools(...)` to its authenticated destination, and running its
own delivery loop. Merely importing or constructing an SDK agent does not
enable scheduling.

Long responses are split into Telegram-sized messages. Attachments, voice
messages, edits, reactions, and group conversations are intentionally ignored
in this first adapter.

The next Telegram polling offset is stored in SQLite after each handled update
and never moves backward, preventing normal service restarts from replaying old
messages. Delivery remains at-least-once: a process failure after sending a
reply but before saving its cursor can cause Telegram to redeliver that update.

## Run continuously with systemd

Create `/etc/systemd/system/chulk-telegram.service` and replace the user and
paths with the account and checkout used on the server:

```ini
[Unit]
Description=Chulk Telegram agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=chulk
WorkingDirectory=/srv/ChulkHarness
EnvironmentFile=/srv/ChulkHarness/.env
ExecStart=/srv/ChulkHarness/.venv/bin/chulk-telegram
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now chulk-telegram
sudo systemctl status chulk-telegram
```

The working directory controls which project files the agent can access. Start
with `CHULK_PERMISSION_PROFILE=read-only`; only enable write or shell behavior
after reviewing the [permission](permissions.md) and [safety](safety.md)
guides. Telegram authorization controls who can submit prompts, while Chulk's
permission policy independently controls what those prompts may cause.

## Operations and security

- Keep `.env`, `.chulk/`, SQLite files, and traces readable only by the service
  account.
- Do not expose the bot process itself to the internet; polling only needs
  outbound HTTPS access to Telegram and the configured model provider.
- Logs intentionally omit the bot token, model-provider keys, prompt content,
  and exception details. Inspect Chulk's sensitive local traces only when
  required.
- Stop the service before changing its token or provider configuration, then
  restart it so the new `.env` values are loaded.
- Set `CHULK_WEB_SEARCH_MAX_RESULTS` conservatively to control context size and
  Tavily credit usage.
