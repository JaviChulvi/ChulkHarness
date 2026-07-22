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

## Run

Install the selected provider and start the adapter from the project root:

```bash
python -m pip install -e '.[gemini]'
chulk-telegram
```

The adapter creates one durable SQLite conversation per private Telegram chat.
It restores that conversation after a process restart. Available commands are:

- `/new` starts a fresh conversation for the chat.
- `/status` reports the provider, model, and short conversation id.
- `/plan <request>` asks the agent to prepare an approval plan.
- `/approve` approves the pending plan.
- `/reject` rejects the pending plan.
- `/help` displays command help.

Long responses are split into Telegram-sized messages. Attachments, voice
messages, edits, reactions, and group conversations are intentionally ignored
in this first adapter.

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
