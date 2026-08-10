const state = {
  token: sessionStorage.getItem("chulk.controlToken") || "",
  runs: [],
  selected: "",
  offset: 0,
  limit: 25,
  hasMore: false,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  unlock: $("#unlock"),
  form: $("#unlock-form"),
  token: $("#token"),
  error: $("#error"),
  ledger: $("#ledger"),
  connection: $("#connection"),
  refresh: $("#refresh"),
  filterForm: $("#filter-form"),
  clearFilters: $("#clear-filters"),
  rail: $("#run-rail"),
  detail: $("#runs"),
  updated: $("#updated"),
  previous: $("#prev-page"),
  next: $("#next-page"),
  pageLabel: $("#page-label"),
};

async function api(path) {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${state.token}` },
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Unreadable response (${response.status})`);
  }
  if (!response.ok) {
    throw new Error(body.error?.message || `Request failed (${response.status})`);
  }
  return body;
}

async function unlock() {
  state.token = elements.token.value.trim() || state.token;
  if (!state.token) throw new Error("Enter the control token.");
  await api("/v1/session");
  sessionStorage.setItem("chulk.controlToken", state.token);
  elements.unlock.hidden = true;
  elements.ledger.hidden = false;
  elements.refresh.hidden = false;
  await refresh();
}

function runQuery() {
  const query = new URLSearchParams({
    limit: String(state.limit),
    offset: String(state.offset),
  });
  const values = new FormData(elements.filterForm);
  for (const field of ["suite", "status", "mode", "target", "provider", "model"]) {
    const value = String(values.get(field) || "").trim();
    if (value) query.set(field, value);
  }
  const startedAfter = String(values.get("started_after") || "");
  if (startedAfter) query.set("started_after", new Date(startedAfter).toISOString());
  for (const tag of String(values.get("tag") || "").split(",")) {
    if (tag.trim()) query.append("tag", tag.trim());
  }
  return query;
}

async function refresh() {
  elements.connection.textContent = "Loading";
  const value = await api(`/v1/evals/runs?${runQuery().toString()}`);
  state.runs = value.runs;
  state.hasMore = Boolean(value.has_more);
  if (state.selected && !state.runs.some((run) => run.id === state.selected)) {
    state.selected = "";
    elements.detail.replaceChildren(emptyState("Select a run", "Choose one result from the ledger to inspect its evidence."));
  }
  renderSummary();
  renderRail();
  renderPagination();
  elements.connection.textContent = "Live";
  elements.updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

function renderSummary() {
  const passing = state.runs.filter((run) => run.passed).length;
  $("#run-count").textContent = String(state.runs.length);
  $("#pass-count").textContent = String(passing);
  $("#mean-rate").textContent = state.runs.length
    ? pct(state.runs.reduce((sum, run) => sum + run.pass_rate, 0) / state.runs.length)
    : "—";
}

function renderRail() {
  elements.rail.replaceChildren();
  if (!state.runs.length) {
    elements.rail.append(emptyState("No runs", "No stored evaluation runs match these filters."));
    return;
  }
  for (const run of state.runs) {
    const button = document.createElement("button");
    const quality = run.passed ? "passed" : "failed";
    button.type = "button";
    button.className = `run ${quality} status-${run.status}`;
    button.dataset.id = run.id;
    button.setAttribute("aria-current", String(run.id === state.selected));

    const rate = node("span", "run-rate", pct(run.pass_rate));
    const copy = node("span", "run-copy");
    copy.append(node("strong", "", run.suite_name), node("small", "", formatDate(run.started_at)));
    const status = node(
      "span",
      "run-state",
      run.status === "completed" ? (run.passed ? "pass" : "fail") : run.status,
    );
    button.append(rate, copy, status);
    button.addEventListener("click", () => selectRun(run.id));
    elements.rail.append(button);
  }
}

function renderPagination() {
  elements.previous.disabled = state.offset === 0;
  elements.next.disabled = !state.hasMore;
  elements.pageLabel.textContent = `Page ${Math.floor(state.offset / state.limit) + 1}`;
}

async function selectRun(id) {
  state.selected = id;
  renderRail();
  elements.detail.replaceChildren(emptyState("Loading run", "Reading redacted case evidence."));
  const detail = await api(`/v1/evals/runs/${encodeURIComponent(id)}`);
  renderDetail(detail);
  elements.detail.focus();
}

function renderDetail(detail) {
  const { run, baseline, dimensions, traces } = detail;
  elements.detail.replaceChildren();
  const head = node("header", "run-head");
  const title = node("div");
  const lifecycle = run.status === "completed" ? (run.passed ? "Quality gate passed" : "Quality gate failed") : `Run ${run.status}`;
  title.append(node("p", "kicker", lifecycle), node("h2", "", run.suite_name));
  const identity = node("span", "id");
  identity.append(node("b", "", run.id), node("small", "", `${run.metadata?.mode || "unknown"} · ${formatDate(run.started_at)}`));
  head.append(title, identity);
  elements.detail.append(head);

  elements.detail.append(metricStrip([
    ["Pass rate", pct(run.metrics.pass_rate)],
    ["P95 latency", seconds(run.metrics.p95_latency_seconds)],
    ["Total tokens", integer(run.metrics.total_tokens)],
    ["Total cost", money(run.metrics.total_cost)],
  ]));
  renderFailures(run);
  elements.detail.append(renderBaseline(baseline));
  elements.detail.append(renderDimensions(dimensions));
  const traceIndex = new Map(
    (traces || []).map((trace) => [traceKey(trace.target_name, trace.case_id, trace.trial, trace.turn), trace]),
  );
  const cases = node("section", "case-list");
  cases.append(sectionHeading("Case evidence", `${(run.cases || []).length} target/case results`));
  for (const item of run.cases || []) cases.append(renderCase(item, traceIndex));
  elements.detail.append(cases);
}

function renderFailures(run) {
  const groups = [
    ["Quality thresholds", run.threshold_failures, "quality"],
    ["Operational failures", run.operational_errors, "operational"],
  ];
  for (const [title, failures, kind] of groups) {
    if (!failures?.length) continue;
    const alert = node("section", `failure-block ${kind}`);
    alert.append(node("h3", "", title));
    const list = node("ul");
    for (const failure of failures) list.append(node("li", "", failure));
    alert.append(list);
    elements.detail.append(alert);
  }
}

function renderBaseline(baseline) {
  const section = node("section", "baseline-seam");
  section.append(sectionHeading("Baseline seam", baseline ? `Compared with ${baseline.run.id}` : "No promoted baseline"));
  if (!baseline) {
    section.append(node("p", "muted", "Promote a completed run from the CLI to reveal regression deltas here."));
    return section;
  }
  const comparison = baseline.comparison;
  const deltas = comparison.metric_deltas || {};
  const measures = node("div", "delta-grid");
  for (const [label, value, formatter, lowerIsBetter] of [
    ["Pass rate", deltas.pass_rate, signedPct, false],
    ["P95 latency", deltas.p95_latency_seconds, signedSeconds, true],
    ["Total tokens", deltas.total_tokens, signedInteger, true],
    ["Total cost", deltas.total_cost, signedMoney, true],
  ]) {
    const card = node("div", "delta");
    const numeric = Number(value || 0);
    const regressed = lowerIsBetter ? numeric > 0 : numeric < 0;
    card.classList.add(regressed ? "regressed" : numeric === 0 ? "flat" : "improved");
    card.append(node("b", "", formatter(value)), node("span", "", label));
    measures.append(card);
  }
  section.append(measures);
  const coverage = node("p", "coverage", `Coverage ${pct(comparison.baseline_coverage)} · ${comparison.matched_cases.length} matched · ${comparison.new_cases.length} new · ${comparison.removed_cases.length} removed`);
  section.append(coverage);
  if (comparison.new_cases.length || comparison.removed_cases.length) {
    const changes = node("details", "identity-changes");
    changes.append(node("summary", "", "Case identity changes"));
    changes.append(jsonEvidence({ new_cases: comparison.new_cases, removed_cases: comparison.removed_cases }));
    section.append(changes);
  }
  return section;
}

function renderDimensions(dimensions) {
  const section = node("section", "comparisons");
  section.append(sectionHeading("Comparison matrix", "Target, provider, model, and dataset tags"));
  const grid = node("div", "comparison-grid");
  for (const [label, values] of Object.entries(dimensions || {})) {
    if (!values.length) continue;
    const group = node("div", "comparison-group");
    group.append(node("h3", "", label));
    for (const item of values) {
      const row = node("div", "comparison-row");
      const name = item.provider || item.model
        ? `${item.name} · ${item.provider || "—"}/${item.model || "—"}`
        : item.name;
      row.append(node("span", "", name), node("b", "", item.pass_rate == null ? "—" : pct(item.pass_rate)));
      group.append(row);
    }
    grid.append(group);
  }
  if (!grid.childElementCount) grid.append(node("p", "muted", "No comparison dimensions were recorded."));
  section.append(grid);
  return section;
}

function renderCase(item, traceIndex) {
  const details = node("details", `case ${item.passed ? "pass" : "fail"}`);
  const summary = node("summary");
  summary.append(
    node("span", "", item.case_id),
    node("span", "", item.target_name),
    node("span", "", item.passed ? "pass" : "fail"),
  );
  details.append(summary);
  const body = node("div", "case-body");
  for (const trial of item.trials || []) body.append(renderTrial(item, trial, traceIndex));
  details.append(body);
  return details;
}

function renderTrial(item, trial, traceIndex) {
  const section = node("section", "trial");
  const status = trial.exception ? "execution failed" : `${seconds(trial.duration_seconds)} total`;
  section.append(sectionHeading(`Trial ${trial.trial}`, status));
  if (trial.exception) section.append(node("p", "trial-error", trial.exception));

  const turns = table(["Turn", "Status", "Latency", "Tokens", "Cost", "Tools", "Trace"]);
  for (const turn of trial.turns || []) {
    const result = turn.result || {};
    const trace = traceIndex.get(traceKey(item.target_name, item.case_id, trial.trial, turn.index));
    const tools = (result.tool_calls || []).map((call) => call.tool_name).join(", ") || "—";
    const row = document.createElement("tr");
    appendCells(row, [turn.index, result.status || "unknown", seconds(turn.duration_seconds), integer(result.usage?.total_tokens), money(result.cost?.amount), tools]);
    const traceCell = document.createElement("td");
    traceCell.append(traceControl(trace));
    row.append(traceCell);
    turns.tBodies[0].append(row);
  }
  section.append(turns);

  const grades = table(["Grader", "Contract", "Score", "Result", "Reason", "Evidence"]);
  for (const grade of trial.grades || []) {
    const row = document.createElement("tr");
    const contract = `${grade.details?.grader_version || "1"}${grade.details?.required ? " · required" : " · info"}`;
    appendCells(row, [grade.grader, contract, Number(grade.score).toFixed(2), grade.error ? "error" : grade.passed ? "pass" : "fail", grade.reason]);
    const evidence = document.createElement("td");
    const evidenceDetails = node("details", "evidence");
    evidenceDetails.append(node("summary", "", "Inspect"), jsonEvidence({ details: grade.details, error: grade.error }));
    evidence.append(evidenceDetails);
    row.append(evidence);
    grades.tBodies[0].append(row);
  }
  section.append(grades);
  return section;
}

function traceControl(trace) {
  if (!trace || !trace.available) {
    return node("span", "trace-unavailable", trace ? `Unavailable · ${trace.reason.replaceAll("_", " ")}` : "Not recorded");
  }
  const wrap = node("div", "trace-control");
  const link = node("a", "trace-link", "Open trace");
  link.href = trace.href;
  link.addEventListener("click", async (event) => {
    event.preventDefault();
    link.textContent = "Loading…";
    try {
      const value = await api(trace.href);
      wrap.replaceChildren(jsonEvidence(value));
    } catch (error) {
      wrap.replaceChildren(node("span", "trace-unavailable", error.message));
    }
  });
  wrap.append(link);
  return wrap;
}

function metricStrip(values) {
  const strip = node("div", "metric-strip");
  for (const [label, value] of values) {
    const item = node("div", "metric");
    item.append(node("b", "", value), node("span", "", label));
    strip.append(item);
  }
  return strip;
}

function sectionHeading(title, description) {
  const head = node("header", "section-heading");
  head.append(node("h3", "", title), node("p", "", description));
  return head;
}

function table(headers) {
  const value = document.createElement("table");
  const head = document.createElement("thead");
  const row = document.createElement("tr");
  for (const header of headers) row.append(node("th", "", header));
  head.append(row);
  value.append(head, document.createElement("tbody"));
  return value;
}

function appendCells(row, values) {
  for (const value of values) row.append(node("td", "", String(value)));
}

function jsonEvidence(value) {
  const output = node("pre", "json-evidence");
  output.textContent = JSON.stringify(value, null, 2);
  return output;
}

function emptyState(title, description) {
  const value = node("div", "empty");
  value.append(node("h2", "", title), node("p", "", description));
  return value;
}

function node(tag, className = "", text = "") {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== "") value.textContent = String(text);
  return value;
}

function traceKey(target, caseId, trial, turn) {
  return [target, caseId, trial, turn].join("\u0000");
}

function pct(value) { return `${Math.round(Number(value || 0) * 100)}%`; }
function seconds(value) { return `${Number(value || 0).toFixed(2)}s`; }
function money(value) { return value == null ? "—" : `$${Number(value).toFixed(4)}`; }
function integer(value) { return value == null ? "—" : Math.round(Number(value)).toLocaleString(); }
function sign(value) { return Number(value || 0) > 0 ? "+" : ""; }
function signedPct(value) { return `${sign(value)}${(Number(value || 0) * 100).toFixed(1)}pp`; }
function signedSeconds(value) { return `${sign(value)}${Number(value || 0).toFixed(2)}s`; }
function signedInteger(value) { return `${sign(value)}${Math.round(Number(value || 0)).toLocaleString()}`; }
function signedMoney(value) { return `${sign(value)}$${Number(value || 0).toFixed(4)}`; }
function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value || "—") : date.toLocaleString();
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.error.textContent = "";
  try {
    await unlock();
  } catch (error) {
    elements.error.textContent = error.message;
    elements.connection.textContent = "Locked";
  }
});
elements.refresh.addEventListener("click", () => refresh().catch(showRefreshError));
elements.filterForm.addEventListener("submit", (event) => {
  event.preventDefault();
  state.offset = 0;
  refresh().catch(showRefreshError);
});
elements.clearFilters.addEventListener("click", () => {
  elements.filterForm.reset();
  state.offset = 0;
  refresh().catch(showRefreshError);
});
elements.previous.addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - state.limit);
  refresh().catch(showRefreshError);
});
elements.next.addEventListener("click", () => {
  if (!state.hasMore) return;
  state.offset += state.limit;
  refresh().catch(showRefreshError);
});

function showRefreshError(error) {
  elements.connection.textContent = "Error";
  elements.updated.textContent = error.message;
}

if (state.token) {
  elements.token.value = state.token;
  unlock().catch(() => sessionStorage.removeItem("chulk.controlToken"));
}
