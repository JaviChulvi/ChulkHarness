const state = {
  token: sessionStorage.getItem("chulk.controlToken") || "",
  csrfToken: "",
  profileId: "",
  conversationId: "",
  socket: null,
  pending: new Map(),
  clientId: sessionStorage.getItem("chulk.clientId") || crypto.randomUUID(),
};

sessionStorage.setItem("chulk.clientId", state.clientId);

const elements = {
  connection: document.querySelector(".connection-pill"),
  connectionLabel: document.querySelector("#connection-label"),
  unlockCard: document.querySelector("#unlock-card"),
  unlockForm: document.querySelector("#unlock-form"),
  unlockError: document.querySelector("#unlock-error"),
  token: document.querySelector("#token"),
  sessionBar: document.querySelector("#session-bar"),
  profile: document.querySelector("#profile"),
  reconnect: document.querySelector("#reconnect"),
  newConversation: document.querySelector("#new-conversation"),
  stopConversation: document.querySelector("#stop-conversation"),
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#empty-state"),
  conversationId: document.querySelector("#conversation-id"),
  commandStrip: document.querySelector("#command-strip"),
  composer: document.querySelector("#composer"),
  message: document.querySelector("#message"),
  send: document.querySelector("#send"),
  pairingForm: document.querySelector("#pairing-form"),
  pairAdapter: document.querySelector("#pair-adapter"),
  pairAccount: document.querySelector("#pair-account"),
  pairProfile: document.querySelector("#pair-profile"),
  pairPrincipal: document.querySelector("#pair-principal"),
  createPairing: document.querySelector("#create-pairing"),
  pairingResult: document.querySelector("#pairing-result"),
  pairingCode: document.querySelector("#pairing-code"),
  pairingExpiry: document.querySelector("#pairing-expiry"),
  copyPairing: document.querySelector("#copy-pairing"),
  refreshRoutes: document.querySelector("#refresh-routes"),
  routeList: document.querySelector("#route-list"),
  eventStatus: document.querySelector("#event-status"),
};

function setConnection(status, label) {
  elements.connection.dataset.status = status;
  elements.connectionLabel.textContent = label;
}

function setStage(stage) {
  const stages = ["auth", "profile", "socket", "turn"];
  const target = stages.indexOf(stage);
  document.querySelectorAll("[data-stage]").forEach((item) => {
    const index = stages.indexOf(item.dataset.stage);
    item.classList.toggle("complete", index < target);
    item.classList.toggle("active", index === target);
  });
}

function setEventStatus(message) {
  elements.eventStatus.textContent = message;
}

function authHeaders(json = false) {
  const headers = { Authorization: `Bearer ${state.token}` };
  if (json) {
    headers["Content-Type"] = "application/json";
    headers["X-Chulk-CSRF"] = state.csrfToken;
  }
  return headers;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...authHeaders(Boolean(options.body)), ...(options.headers || {}) },
  });
  const value = await response.json();
  if (!response.ok) {
    throw new Error(value.error?.message || `Request failed (${response.status})`);
  }
  return value;
}

function fillProfiles(profiles) {
  for (const select of [elements.profile, elements.pairProfile]) {
    select.replaceChildren();
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.id;
      select.append(option);
    }
  }
  state.profileId = elements.profile.value;
  elements.pairProfile.value = state.profileId;
}

function setControls(enabled) {
  elements.sessionBar.hidden = !enabled;
  elements.commandStrip.hidden = !enabled;
  elements.message.disabled = !enabled;
  elements.send.disabled = !enabled;
  for (const element of [
    elements.pairAdapter,
    elements.pairAccount,
    elements.pairProfile,
    elements.pairPrincipal,
    elements.createPairing,
    elements.refreshRoutes,
  ]) {
    element.disabled = !enabled;
  }
}

async function unlock(token) {
  state.token = token.trim();
  if (!state.token) {
    throw new Error("Enter the owner control token.");
  }
  const [session, profiles] = await Promise.all([
    api("/v1/session"),
    api("/v1/profiles"),
  ]);
  if (!profiles.profiles.length) {
    throw new Error("No agent profiles are available.");
  }
  state.csrfToken = session.csrf_token;
  sessionStorage.setItem("chulk.controlToken", state.token);
  fillProfiles(profiles.profiles);
  elements.unlockCard.hidden = true;
  setControls(true);
  setStage("profile");
  await connect();
  await loadRoutes();
}

function socketUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/v1/gateway/ws`;
}

async function connect() {
  if (state.socket && state.socket.readyState < WebSocket.CLOSING) {
    state.socket.close(1000, "reconnect");
  }
  state.profileId = elements.profile.value;
  state.conversationId = "";
  updateConversationId();
  setConnection("locked", "Connecting");
  setEventStatus(`Opening ${state.profileId}`);
  const socket = new WebSocket(
    socketUrl(),
    [`chulk.control.${state.token}`],
  );
  state.socket = socket;
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      type: "hello",
      schema_version: 1,
      profile_id: state.profileId,
      client_id: state.clientId,
    }));
  });
  socket.addEventListener("message", (event) => {
    let frame;
    try {
      frame = JSON.parse(event.data);
    } catch {
      addMessage("system", "The server sent an unreadable frame.", "failed");
      return;
    }
    handleFrame(frame);
  });
  socket.addEventListener("close", (event) => {
    if (state.socket !== socket) return;
    setConnection("error", "Disconnected");
    setStage("profile");
    setEventStatus(`Connection closed (${event.code})`);
    elements.send.disabled = true;
  });
  socket.addEventListener("error", () => {
    setConnection("error", "Connection error");
    setEventStatus("Check the token and local server");
  });
}

function handleFrame(frame) {
  if (frame.type === "ready") {
    setConnection("live", "Connected");
    setStage("socket");
    setEventStatus(`Connected to ${frame.profile_id}`);
    elements.send.disabled = false;
    elements.message.focus();
    return;
  }
  if (frame.type === "message.accepted") {
    const pending = state.pending.get(frame.event_id);
    if (pending) pending.inboxId = frame.inbox_id;
    setStage("turn");
    setEventStatus("Turn accepted");
    return;
  }
  if (frame.type === "message.completed") {
    if (frame.conversation_id && frame.conversation_id !== "none") {
      state.conversationId = frame.conversation_id;
      updateConversationId();
    }
    const roleClass = frame.status === "failed" ? "failed" : "";
    addMessage("assistant", frame.text || frame.error || "No response.", roleClass);
    const eventId = frame.reply_to_event_id;
    if (eventId) state.pending.delete(eventId);
    setStage("socket");
    setEventStatus(frame.status === "failed" ? "Turn failed" : "Turn complete");
    return;
  }
  if (frame.type === "cancel.accepted") {
    addMessage(
      "system",
      frame.cancelled ? "Stop request accepted." : "No active work to stop.",
    );
    setEventStatus("Stop request handled");
    return;
  }
  if (frame.type === "error") {
    addMessage("system", frame.error?.message || "Gateway error.", "failed");
    setEventStatus(frame.error?.code || "Gateway error");
  }
}

function sendMessage(text) {
  const message = text.trim();
  if (!message || !state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return;
  }
  const eventId = crypto.randomUUID();
  state.pending.set(eventId, {});
  state.socket.send(JSON.stringify({
    type: "message",
    schema_version: 1,
    event_id: eventId,
    idempotency_key: `webchat:${state.clientId}:${eventId}`,
    message,
    conversation_id: state.conversationId || null,
    mode: "run",
  }));
  addMessage("user", message);
  setStage("turn");
  setEventStatus("Sending turn");
}

function addMessage(role, text, extraClass = "") {
  elements.emptyState?.remove();
  const item = document.createElement("li");
  item.className = `message ${role} ${extraClass}`.trim();
  const header = document.createElement("header");
  const label = document.createElement("span");
  label.textContent = role === "user" ? "You" : role === "assistant" ? "Chulk" : "System";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const body = document.createElement("p");
  body.textContent = text;
  header.append(label, time);
  item.append(header, body);
  elements.messages.append(item);
  item.scrollIntoView({ block: "end", behavior: "smooth" });
}

function updateConversationId() {
  elements.conversationId.textContent = state.conversationId || "not started";
  elements.conversationId.title = state.conversationId;
}

async function createPairing() {
  const value = await api("/v1/gateway/pairings", {
    method: "POST",
    body: JSON.stringify({
      adapter: elements.pairAdapter.value,
      account_id: elements.pairAccount.value,
      profile_id: elements.pairProfile.value,
      principal_id: elements.pairPrincipal.value.trim() || null,
      ttl_seconds: 600,
    }),
  });
  elements.pairingCode.textContent = value.pairing.code;
  elements.pairingExpiry.textContent = `Expires ${new Date(value.pairing.expires_at).toLocaleString()}`;
  elements.pairingResult.hidden = false;
  setEventStatus(`Pairing ready for ${value.pairing.adapter}`);
}

async function loadRoutes() {
  const value = await api("/v1/gateway/routes");
  elements.routeList.replaceChildren();
  if (!value.routes.length) {
    const empty = document.createElement("li");
    empty.textContent = "No active external channel routes.";
    elements.routeList.append(empty);
    return;
  }
  for (const route of value.routes) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${route.adapter} / ${route.account_id}`;
    const detail = document.createElement("code");
    detail.textContent = `${route.principal_id || "any user"} → ${route.profile_id}`;
    item.append(title, detail);
    elements.routeList.append(item);
  }
}

elements.unlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.unlockError.textContent = "";
  try {
    await unlock(elements.token.value);
  } catch (error) {
    state.token = "";
    sessionStorage.removeItem("chulk.controlToken");
    setConnection("error", "Locked");
    elements.unlockError.textContent = error.message;
  }
});

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = elements.message.value;
  elements.message.value = "";
  sendMessage(value);
});

elements.message.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.profile.addEventListener("change", async () => {
  elements.pairProfile.value = elements.profile.value;
  await connect();
});
elements.reconnect.addEventListener("click", connect);
elements.newConversation.addEventListener("click", () => sendMessage("/new"));
elements.stopConversation.addEventListener("click", () => {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({
    type: "cancel",
    schema_version: 1,
    conversation_id: state.conversationId || null,
  }));
});
elements.commandStrip.addEventListener("click", (event) => {
  const command = event.target.closest("[data-command]")?.dataset.command;
  if (command) sendMessage(command);
});
elements.pairingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await createPairing();
  } catch (error) {
    setEventStatus(error.message);
  }
});
elements.copyPairing.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.pairingCode.textContent);
  setEventStatus("Pairing code copied");
});
elements.refreshRoutes.addEventListener("click", async () => {
  try {
    await loadRoutes();
    setEventStatus("Routes refreshed");
  } catch (error) {
    setEventStatus(error.message);
  }
});

if (state.token) {
  elements.token.value = state.token;
  unlock(state.token).catch(() => {
    sessionStorage.removeItem("chulk.controlToken");
    state.token = "";
    setConnection("locked", "Locked");
  });
}
