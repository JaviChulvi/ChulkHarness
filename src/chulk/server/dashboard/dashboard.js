const state = {
  token: sessionStorage.getItem("chulk.controlToken") || "",
  csrfToken: "",
  profileId: "",
  profiles: [],
  data: {},
  workTab: "goals",
  refreshTimer: null,
  refreshDelay: 10000,
  refreshId: 0,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  unlock: $("#unlock"),
  unlockForm: $("#unlock-form"),
  unlockError: $("#unlock-error"),
  token: $("#token"),
  board: $("#board"),
  headerControls: $("#header-controls"),
  profile: $("#profile"),
  refresh: $("#refresh"),
  connection: $("#connection"),
  connectionLabel: $("#connection-label"),
  profileName: $("#profile-name"),
  briefingCopy: $("#briefing-copy"),
  attentionList: $("#attention-list"),
  attentionCount: $("#attention-count"),
  workList: $("#work-list"),
  conversationList: $("#conversation-list"),
  routeList: $("#route-list"),
  usageList: $("#usage-list"),
  traceList: $("#trace-list"),
  artifactDrawer: $("#artifact-drawer"),
  previewDialog: $("#preview-dialog"),
  previewTitle: $("#preview-title"),
  previewContent: $("#preview-content"),
  toast: $("#toast"),
  updatedAt: $("#updated-at"),
};

function authHeaders(json = false) {
  return {
    Authorization: `Bearer ${state.token}`,
    ...(json ? {
      "Content-Type": "application/json",
      "X-Chulk-CSRF": state.csrfToken,
    } : {}),
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...authHeaders(Boolean(options.body)), ...(options.headers || {}) },
  });
  let value;
  try {
    value = await response.json();
  } catch {
    throw new Error(`Unreadable control response (${response.status})`);
  }
  if (!response.ok) {
    throw new Error(value.error?.message || `Control request failed (${response.status})`);
  }
  return value;
}

function setConnection(status, label) {
  elements.connection.dataset.state = status;
  elements.connectionLabel.textContent = label;
}

async function unlock(token) {
  state.token = token.trim();
  if (!state.token) throw new Error("Enter the owner control token.");
  const [session, profiles] = await Promise.all([
    api("/v1/session"),
    api("/v1/profiles"),
  ]);
  if (!profiles.profiles.length) throw new Error("No profiles are available.");
  state.csrfToken = session.csrf_token;
  state.profiles = profiles.profiles;
  sessionStorage.setItem("chulk.controlToken", state.token);
  fillProfiles();
  elements.unlock.hidden = true;
  elements.board.hidden = false;
  elements.headerControls.hidden = false;
  await refreshDashboard();
}

function fillProfiles() {
  elements.profile.replaceChildren();
  for (const profile of state.profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = `${profile.id} · ${profile.model_profile_id}`;
    elements.profile.append(option);
  }
  state.profileId = elements.profile.value;
}

async function refreshDashboard() {
  clearTimeout(state.refreshTimer);
  if (!state.profileId) return;
  const refreshId = ++state.refreshId;
  const profileId = state.profileId;
  setConnection("loading", "Syncing");
  const root = `/v1/profiles/${encodeURIComponent(profileId)}`;
  try {
    const [
      conversations,
      proposals,
      goals,
      tasks,
      jobs,
      usage,
      traces,
      routes,
      permissions,
    ] = await Promise.all([
      api(`${root}/conversations?limit=30`),
      api(`${root}/learning/proposals?status=pending&limit=30`),
      api(`${root}/goals?limit=50`),
      api(`${root}/tasks?limit=50`),
      api(`${root}/jobs?include_terminal=false&limit=50`),
      api(`${root}/usage?group_by=resource_kind&limit=50`),
      api(`${root}/traces?limit=30`),
      api("/v1/gateway/routes"),
      api(`${root}/permissions?status=pending&limit=100`),
    ]);
    if (refreshId !== state.refreshId) return;
    state.data = {
      conversations: conversations.conversations,
      proposals: proposals.proposals,
      permissions: permissions.permissions,
      goals: goals.goals,
      tasks: tasks.tasks,
      jobs: jobs.jobs,
      usage: usage.groups || usage.entries || [],
      traces: traces.traces,
      routes: routes.routes.filter((route) => route.profile_id === profileId),
    };
    render();
    state.refreshDelay = 10000;
    setConnection("live", "Live");
    elements.updatedAt.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    if (refreshId !== state.refreshId) return;
    setConnection("error", "Reconnecting");
    elements.updatedAt.textContent = error.message;
    state.refreshDelay = Math.min(state.refreshDelay * 2, 60000);
  }
  if (refreshId === state.refreshId) {
    state.refreshTimer = setTimeout(refreshDashboard, state.refreshDelay);
  }
}

function render() {
  const data = state.data;
  elements.profileName.textContent = state.profileId;
  const attention = data.permissions.length + data.proposals.length;
  const terminal = new Set(["completed", "cancelled", "failed", "expired", "budget_exhausted"]);
  const activeWork = ["goals", "tasks", "jobs"].reduce(
    (total, key) => total + data[key].filter((item) => !terminal.has(item.status)).length,
    0,
  );
  $("#metric-attention").textContent = attention;
  $("#metric-work").textContent = activeWork;
  $("#metric-conversations").textContent = data.conversations.length;
  $("#metric-evidence").textContent = data.traces.reduce(
    (total, trace) => total + Number(trace.artifact_count || 0), 0,
  );
  elements.attentionCount.textContent = attention;
  elements.briefingCopy.textContent = attention
    ? `${attention} owner decision${attention === 1 ? "" : "s"} waiting.`
    : "No owner decisions are waiting.";
  renderAttention();
  renderWork();
  renderConversations();
  renderRoutes();
  renderUsage();
  renderTraces();
}

function empty(label) {
  const node = document.createElement("p");
  node.className = "empty";
  node.textContent = label;
  return node;
}

function itemShell(title, status, meta) {
  const item = document.createElement("article");
  item.className = "item";
  item.dataset.state = status;
  const head = document.createElement("div");
  head.className = "item-head";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const stateLabel = document.createElement("span");
  stateLabel.className = "meta";
  stateLabel.textContent = status;
  head.append(strong, stateLabel);
  const detail = document.createElement("span");
  detail.className = "meta";
  detail.textContent = meta;
  item.append(head, detail);
  return item;
}

function actionButton(label, action, value, extraClass = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.dataset.action = action;
  Object.entries(value).forEach(([key, item]) => { button.dataset[key] = item; });
  button.className = extraClass;
  return button;
}

function renderAttention() {
  elements.attentionList.replaceChildren();
  const values = [];
  for (const permission of state.data.permissions) {
    const item = itemShell(
      permission.tool_name || "Tool permission",
      permission.status,
      permission.reason || permission.permission_level,
    );
    const actions = document.createElement("div");
    actions.className = "actions";
    const identity = { kind: "permission", id: permission.id, conversation: permission.conversation_id };
    actions.append(
      actionButton("Allow", "attention", { ...identity, decision: "allow" }),
      actionButton("Deny", "attention", { ...identity, decision: "deny" }, "deny"),
    );
    item.append(actions);
    values.push(item);
  }
  for (const proposal of state.data.proposals) {
    const item = itemShell(
      proposal.target_name || proposal.kind || "Learning proposal",
      proposal.status,
      proposal.summary || proposal.id,
    );
    const actions = document.createElement("div");
    actions.className = "actions";
    const identity = { kind: "proposal", id: proposal.id };
    actions.append(
      actionButton("Approve", "attention", { ...identity, decision: "approve" }),
      actionButton("Reject", "attention", { ...identity, decision: "reject" }, "deny"),
    );
    item.append(actions);
    values.push(item);
  }
  elements.attentionList.append(...(values.length ? values : [empty("Decision queue clear.")]));
}

function nextAction(kind, status) {
  const actions = {
    goal: { draft: "approve", approved: "run", running: "pause", paused: "resume", blocked: "resume" },
    task: { failed: "retry", budget_exhausted: "retry" },
    job: { pending_approval: "approve", active: "pause", running: "pause", paused: "resume" },
  };
  return actions[kind]?.[status] || null;
}

function renderWork() {
  elements.workList.replaceChildren();
  const kind = state.workTab.slice(0, -1);
  const nodes = state.data[state.workTab].map((record) => {
    const title = record.objective || record.title || record.name || record.prompt_preview || record.id;
    const item = itemShell(title, record.status, `${record.id} · revision ${record.revision ?? "—"}`);
    const actions = document.createElement("div");
    actions.className = "actions";
    const advance = nextAction(kind, record.status);
    if (advance) {
      actions.append(actionButton(advance, "work", {
        kind, id: record.id, revision: record.revision, operation: advance,
      }));
    }
    if (!["completed", "cancelled", "expired"].includes(record.status)) {
      actions.append(actionButton("cancel", "work", {
        kind, id: record.id, revision: record.revision, operation: "cancel",
      }, "deny"));
    }
    item.append(actions);
    return item;
  });
  elements.workList.append(...(nodes.length ? nodes : [empty(`No ${state.workTab} to show.`)]));
}

function renderConversations() {
  elements.conversationList.replaceChildren();
  const nodes = state.data.conversations.map((record) => compact(
    record.title || record.id,
    `${record.status} · ${record.turn_count} turns`,
  ));
  elements.conversationList.append(...(nodes.length ? nodes : [empty("No conversations.")]));
}

function renderRoutes() {
  elements.routeList.replaceChildren();
  const nodes = state.data.routes.map((route) => compact(
    `${route.adapter} / ${route.account_id}`,
    route.principal_id || "profile route",
  ));
  elements.routeList.append(...(nodes.length ? nodes : [empty("No routes for this profile.")]));
}

function compact(title, detail) {
  const row = document.createElement("div");
  row.className = "compact";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const small = document.createElement("small");
  small.textContent = detail;
  row.append(strong, small);
  return row;
}

function renderUsage() {
  elements.usageList.replaceChildren();
  for (const record of state.data.usage) {
    const row = document.createElement("tr");
    const cost = record.cost || {};
    const values = [
      record.key || record.resource_kind || "unknown",
      record.total_units || formatAggregateUnits(record) || formatUnits(record.units),
      record.total_cost || (cost.amount ? `${cost.amount} ${cost.currency}` : "unknown"),
      cost.pricing_known === false ? "unknown" : "known",
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value || "—";
      row.append(cell);
    });
    elements.usageList.append(row);
  }
  if (!state.data.usage.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.append(empty("No metered usage."));
    row.append(cell);
    elements.usageList.append(row);
  }
}

function formatUnits(units) {
  if (!units) return "—";
  return Object.entries(units).map(([key, value]) => `${key} ${value}`).join(", ");
}

function formatAggregateUnits(record) {
  const values = [];
  if (record.total_tokens) values.push(`${record.total_tokens} tokens`);
  if (record.model_calls) values.push(`${record.model_calls} model`);
  if (record.tool_calls) values.push(`${record.tool_calls} tool`);
  return values.join(", ");
}

function renderTraces() {
  elements.traceList.replaceChildren();
  const nodes = state.data.traces.map((trace) => actionButton(
    `${trace.conversation_id.slice(0, 8)} · ${trace.artifact_count} artifacts`,
    "trace",
    { conversation: trace.conversation_id },
    "trace-button",
  ));
  elements.traceList.append(...(nodes.length ? nodes : [empty("No trace summaries.")]));
}

async function decideAttention(button) {
  const { kind, id, conversation, decision } = button.dataset;
  if (!window.confirm(`${decision} ${kind} ${id.slice(0, 8)}?`)) return;
  const root = `/v1/profiles/${encodeURIComponent(state.profileId)}`;
  const path = kind === "permission"
    ? `${root}/conversations/${encodeURIComponent(conversation)}/permissions/${encodeURIComponent(id)}`
    : `${root}/learning/proposals/${encodeURIComponent(id)}`;
  const body = kind === "permission"
    ? { decision, idempotency_key: crypto.randomUUID() }
    : { action: decision };
  await api(path, { method: "POST", body: JSON.stringify(body) });
  showToast(`${kind} ${decision}d`);
  await refreshDashboard();
}

async function controlWork(button) {
  const { kind, id, revision, operation } = button.dataset;
  if (!window.confirm(`${operation} ${kind} ${id.slice(0, 8)}?`)) return;
  const plural = { goal: "goals", task: "tasks", job: "jobs" }[kind];
  const body = { action: operation, revision: Number(revision) };
  if (kind === "job") body.idempotency_key = crypto.randomUUID();
  await api(
    `/v1/profiles/${encodeURIComponent(state.profileId)}/${plural}/${encodeURIComponent(id)}/actions`,
    { method: "POST", body: JSON.stringify(body) },
  );
  showToast(`${kind} ${operation} requested`);
  await refreshDashboard();
}

async function loadTrace(button) {
  const conversation = button.dataset.conversation;
  elements.artifactDrawer.replaceChildren(empty("Loading bounded inventory…"));
  const value = await api(
    `/v1/profiles/${encodeURIComponent(state.profileId)}/traces/${encodeURIComponent(conversation)}`,
  );
  elements.artifactDrawer.replaceChildren();
  const artifacts = value.artifacts || [];
  for (const artifact of artifacts) {
    const row = document.createElement("div");
    row.className = "artifact-row";
    row.append(compact(artifact.name || artifact.artifact_id, `${artifact.kind} · ${artifact.size_bytes} bytes`));
    row.append(actionButton("Reveal preview", "artifact", {
      conversation,
      artifact: artifact.artifact_id,
      name: artifact.name || artifact.artifact_id,
    }));
    elements.artifactDrawer.append(row);
  }
  if (!artifacts.length) elements.artifactDrawer.append(empty("No artifacts for this trace."));
}

async function revealArtifact(button) {
  const { conversation, artifact, name } = button.dataset;
  const value = await api(
    `/v1/profiles/${encodeURIComponent(state.profileId)}/conversations/${encodeURIComponent(conversation)}/artifacts/${encodeURIComponent(artifact)}?mode=head_tail&max_bytes=8192`,
  );
  elements.previewTitle.textContent = name;
  elements.previewContent.textContent = value.artifact?.content || "No preview content.";
  elements.previewDialog.showModal();
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  setTimeout(() => elements.toast.classList.remove("visible"), 2500);
}

elements.unlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.unlockError.textContent = "";
  setConnection("loading", "Authenticating");
  try {
    await unlock(elements.token.value);
  } catch (error) {
    state.token = "";
    sessionStorage.removeItem("chulk.controlToken");
    setConnection("error", "Locked");
    elements.unlockError.textContent = error.message;
  }
});

elements.profile.addEventListener("change", async () => {
  state.profileId = elements.profile.value;
  await refreshDashboard();
});
elements.refresh.addEventListener("click", refreshDashboard);
document.querySelectorAll("[data-work-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    state.workTab = button.dataset.workTab;
    document.querySelectorAll("[data-work-tab]").forEach((tab) => {
      tab.setAttribute("aria-selected", String(tab === button));
    });
    renderWork();
  });
});
elements.board.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  button.disabled = true;
  try {
    if (button.dataset.action === "attention") await decideAttention(button);
    if (button.dataset.action === "work") await controlWork(button);
    if (button.dataset.action === "trace") await loadTrace(button);
    if (button.dataset.action === "artifact") await revealArtifact(button);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
});
$("#close-preview").addEventListener("click", () => elements.previewDialog.close());

if (state.token) {
  elements.token.value = state.token;
  unlock(state.token).catch((error) => {
    state.token = "";
    sessionStorage.removeItem("chulk.controlToken");
    setConnection("error", "Locked");
    elements.unlockError.textContent = error.message;
  });
}
