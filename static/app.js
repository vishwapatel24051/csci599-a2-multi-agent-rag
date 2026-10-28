// Agent Lab frontend: vanilla JS, no build step.
// POST /api/chat returns text/event-stream; EventSource can't POST, so we read the body with fetch().

const $ = (sel) => document.querySelector(sel);
const WORKERS = new Set(["research", "code", "data", "summary"]);
const STALL_MS = 45000;        // 3 missed server heartbeats
const RECOVER_ATTEMPTS = 12;   // x 5 s: wait up to a minute for the server to finish and checkpoint the turn

const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

const app = {
  sessionId: store.get("agentlab.session") || newSessionId(),
  policy: store.get("agentlab.policy") || "balanced",
  controller: null,
  turn: null,
};
store.set("agentlab.session", app.sessionId);

function newSessionId() {
  return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/[^A-Za-z0-9]/g, "");
}

// ---------------------------------------------------------------- markdown + citations

function renderMarkdown(text) {
  // Turn [S3] / [W1] into citation markers, but never inside fenced code blocks.
  const parts = text.split(/(```[\s\S]*?(?:```|$))/g);
  const withCites = parts.map((part, i) => (i % 2 === 1 ? part : part.replace(
    /[[【]\s*((?:S|W)\d+)\s*[\]】]/g,  // 【W2】 too: Nemotron sometimes streams those brackets
    (_, label) => `<span class="cite${label[0] === "W" ? " web" : ""}" data-label="${label}">${label}</span>`,
  ))).join("");
  return DOMPurify.sanitize(marked.parse(withCites));
}

function linkCitations(root, citations) {
  root.querySelectorAll(".cite").forEach((el) => {
    const src = citations[el.dataset.label];
    if (!src) return;
    const where = [src.section, src.page_start ? `p.${src.page_start}` : ""].filter(Boolean).join(", ");
    el.title = `${src.title}${where ? " — " + where : ""}`;
    el.onclick = () => {
      showTab("citations");
      const item = document.getElementById(`cite-${src.label}`);
      if (item) { item.scrollIntoView({ block: "center" }); item.classList.add("flash"); setTimeout(() => item.classList.remove("flash"), 1200); }
      window.open(src.url, "_blank", "noopener");
    };
  });
}

// ---------------------------------------------------------------- trace summaries

const SUMMARY = {
  agent_route: (e) => `→ ${e.next} (${e.complexity}) [${e.decided_by}]: ${e.reason}`,
  llm_route: (e) => `${e.role} → ${e.model} (${e.dimension}${e.reasoning ? ", thinking" : ""}): ${e.reason}`,
  llm_call: (e) => `${e.model} ${e.latency_s}s · in ${e.input_tokens} / out ${e.output_tokens} · $${e.cost_usd}`,
  llm_error: (e) => `${e.slot}: ${e.error}`,
  rag_retrieval: (e) => `${e.agent} searched “${e.query}” → ${e.hits.map((h) => `${h.label} ${h.score ?? ""}`).join(", ")}`,
  mcp_tool_call: (e) => `${e.agent} → ${e.server}.${e.tool} ${e.status}${e.n_results != null ? `, ${e.n_results} results` : ""} (${e.latency_s}s)`,
  tool_call: (e) => `${e.agent} → ${e.tool} ${e.summary ?? e.error ?? ""}`,
  agent_start: (e) => `${e.agent} started: ${e.task}`,
  agent_result: (e) => `${e.agent} finished via ${e.model} in ${e.latency_s}s, cited ${e.cited.join(", ") || "nothing"}`,
  agent_error: (e) => `${e.agent} failed: ${e.error}`,
  turn_complete: (e) => `turn complete: ${e.mode}, agents ${e.agents.join(" → ") || "none"}, $${e.llm_cost_usd}`,
};

const STATUS = {
  rag_retrieval: (e) => `${e.agent}: searched the corpus`,
  mcp_tool_call: (e) => `${e.agent}: ${e.tool.replace("tavily_", "web ")} via MCP`,
  tool_call: (e) => `${e.agent}: ${e.tool}`,
  agent_start: (e) => `${e.agent} agent working…`,
  agent_route: (e) => (e.next === "FINISH" ? "composing the answer…" : `routing to ${e.next}…`),
};

// ---------------------------------------------------------------- turn rendering

function createTurn(query) {
  $("#empty")?.remove();
  addUserMessage(query);
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `
    <div class="card">
      <div class="route-strip"><span class="status">Supervisor is routing…</span></div>
      <div class="answer"></div>
      <details class="agent-outputs" hidden><summary>Agent outputs</summary></details>
      <div class="stats" hidden></div>
    </div>`;
  $("#messages").appendChild(wrap);
  resetInspector();
  return {
    el: wrap, strip: wrap.querySelector(".route-strip"), status: wrap.querySelector(".status"),
    answer: wrap.querySelector(".answer"), outputs: wrap.querySelector(".agent-outputs"),
    stats: wrap.querySelector(".stats"), t0: performance.now(), blocks: {}, streamingNode: null,
    chips: [], citations: {}, dirty: false, traceCount: 0, finished: false,
  };
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  $("#messages").appendChild(el);
  scrollToBottom(true);
}

function addAssistantMessage(text) {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="card"><div class="answer"></div></div>`;
  el.querySelector(".answer").innerHTML = renderMarkdown(text);
  $("#messages").appendChild(el);
}

function addChip(turn, { label, provider = "", title = "", cls = "" }) {
  turn.status.textContent = "";
  const chip = document.createElement("span");
  chip.className = `chip ${provider} ${cls}`.trim();
  chip.innerHTML = `<span class="dot"></span><span class="label"></span>`;
  chip.querySelector(".label").textContent = label;
  chip.title = title;
  turn.strip.insertBefore(chip, turn.status);
  turn.chips.push(chip);
  return chip;
}

function onTrace(turn, e) {
  appendTraceRow(turn, e);
  if (STATUS[e.kind]) turn.status.textContent = STATUS[e.kind](e);
  if (e.kind === "llm_route") addLlmRouteRow(e);

  if (e.kind === "llm_route" && e.role === "supervisor") {
    if (!turn.supervisorChip) {
      turn.supervisorChip = addChip(turn, { label: `supervisor · ${shortModel(e.model)} (${providerName(e.provider)}) · ${e.dimension}`, provider: e.provider,
        title: `Model routing (${e.dimension}): ${e.reason}` });
    }
  } else if (e.kind === "agent_route") {
    addAgentRouteRow(e);
    if (e.next !== "FINISH") {
      turn.pendingChip = addChip(turn, { label: `${e.next}`, cls: "pending",
        title: `Supervisor (${e.decided_by}): ${e.reason}\nTask: ${e.task}` });
      turn.pendingChip.dataset.agent = e.next;
    }
  } else if (e.kind === "llm_route") {
    const chip = [...turn.chips].reverse().find((c) => c.dataset.agent === e.role)
      || (e.role === "synthesizer" ? addChip(turn, { label: "synthesizer" }) : null);
    if (chip) {
      chip.dataset.agent = e.role;
      chip.className = `chip ${e.provider}`;
      chip.querySelector(".label").textContent = `${e.role} · ${shortModel(e.model)} (${providerName(e.provider)}) · ${e.dimension}${e.reasoning ? " · thinking" : ""}`;
      chip.title = `${chip.title ? chip.title + "\n" : ""}Model routing (${e.dimension}, rule ${e.rule}): ${e.reason}`;
    }
  } else if (e.kind === "agent_error") {
    const chip = [...turn.chips].reverse().find((c) => c.dataset.agent === e.agent);
    if (chip) { chip.classList.add("error"); chip.title += `\nFAILED: ${e.error}`; }
  }
}

function onToken(turn, { text, node }) {
  if (turn.streamingNode && turn.streamingNode !== node) parkBlock(turn, turn.streamingNode);
  turn.streamingNode = node;
  turn.blocks[node] = (turn.blocks[node] || "") + text;
  if (!turn.dirty) {
    turn.dirty = true;
    requestAnimationFrame(() => {
      turn.dirty = false;
      if (turn.streamingNode) turn.answer.innerHTML = renderMarkdown(turn.blocks[turn.streamingNode]);
      scrollToBottom();
    });
  }
}

function parkBlock(turn, node) {
  const block = document.createElement("div");
  block.className = "agent-block";
  block.innerHTML = `<h4>${node}</h4><div class="answer"></div>`;
  block.querySelector(".answer").innerHTML = renderMarkdown(turn.blocks[node] || "");
  turn.outputs.appendChild(block);
  turn.outputs.hidden = false;
  turn.outputs.querySelector("summary").textContent = `Agent outputs (${turn.outputs.querySelectorAll(".agent-block").length})`;
}

function onDone(turn, { final_answer, turn_stats: s, timing }) {
  turn.finished = true;
  turn.status.textContent = "";
  const last = turn.streamingNode;
  if (last && last !== "synthesize" && turn.blocks[last] && turn.outputs.querySelectorAll(".agent-block").length) {
    parkBlock(turn, last);  // keep the last agent's raw output next to the earlier ones
  }
  turn.answer.innerHTML = renderMarkdown(final_answer || "(no answer)");
  linkCitations(turn.answer, turn.citations);
  linkCitations(turn.outputs, turn.citations);
  const providers = (s.providers || []).join(" + ") || "none";
  turn.stats.hidden = false;
  turn.stats.innerHTML = `
    <span>first token <b>${timing.ttft_s != null ? timing.ttft_s.toFixed(1) + "s" : "—"}</b></span>
    <span>total <b>${timing.total_s.toFixed(1)}s</b></span>
    <span>tokens <b>${(s.input_tokens || 0).toLocaleString()} in / ${(s.output_tokens || 0).toLocaleString()} out</b></span>
    <span>cost <b>$${(s.llm_cost_usd || 0).toFixed(4)}</b></span>
    <span>providers <b>${providers}</b></span>
    <span>mode <b>${s.mode || "—"}</b></span>`;
  scrollToBottom();
}

function onCitation(turn, src) {
  turn.citations[src.label] = src;
  const li = document.createElement("li");
  li.id = `cite-${src.label}`;
  const where = [src.section, src.page_start ? `p.${src.page_start}${src.page_end && src.page_end !== src.page_start ? "–" + src.page_end : ""}` : ""]
    .filter(Boolean).join(" · ");
  li.innerHTML = `<div class="head"><span class="cite ${src.kind === "web" ? "web" : ""}"></span><a target="_blank" rel="noopener"></a></div>
    <div class="meta"></div><div class="snippet"></div>`;
  li.querySelector(".cite").textContent = src.label;
  const a = li.querySelector("a");
  a.href = src.url;
  a.textContent = src.title;
  li.querySelector(".meta").textContent = [where, src.chunk_id, src.score != null ? `relevance ${src.score}` : ""].filter(Boolean).join(" · ");
  li.querySelector(".snippet").textContent = src.snippet;
  $("#citations").appendChild(li);
  $("#cite-count").textContent = Object.keys(turn.citations).length;
}

function onError(turn, message) {
  turn.finished = true;
  turn.status.textContent = "";
  const box = document.createElement("div");
  box.className = "error-box";
  box.textContent = `Error: ${message}`;
  turn.el.querySelector(".card").appendChild(box);
}

// ---------------------------------------------------------------- inspector

function resetInspector() {
  $("#trace").innerHTML = "";
  $("#citations").innerHTML = "";
  $("#agent-routes tbody").innerHTML = "";
  $("#llm-routes tbody").innerHTML = "";
  $("#trace-count").textContent = "0";
  $("#cite-count").textContent = "0";
}

function appendTraceRow(turn, e) {
  const li = document.createElement("li");
  li.dataset.kind = e.kind;
  const t = ((performance.now() - turn.t0) / 1000).toFixed(1);
  const summary = SUMMARY[e.kind] ? SUMMARY[e.kind](e) : "";
  li.innerHTML = `<span class="t"></span><span class="k"></span><span class="s"></span>`;
  li.querySelector(".t").textContent = `${t}s`;
  li.querySelector(".k").textContent = e.kind;
  li.querySelector(".s").textContent = summary;
  li.onclick = () => {
    const open = li.querySelector("pre");
    if (open) { open.remove(); return; }
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(e, null, 2);
    li.appendChild(pre);
  };
  $("#trace").appendChild(li);
  turn.traceCount += 1;
  $("#trace-count").textContent = turn.traceCount;
}

function addRow(tbody, cells) {
  const tr = document.createElement("tr");
  cells.forEach((c) => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
  $(tbody).appendChild(tr);
}

const addAgentRouteRow = (e) => addRow("#agent-routes tbody", [e.hop + 1, e.next, e.complexity, e.decided_by, e.reason]);
const addLlmRouteRow = (e) => addRow("#llm-routes tbody", [e.role, `${e.model} (${e.provider})${e.reasoning ? " +thinking" : ""}`, e.dimension, e.reason]);
const shortModel = (m) => (m || "").replace(/^nvidia\//, "").replace(/-\d+b-a\d+b$/, "");
const providerName = (p) => ({ openai: "OpenAI", nvidia: "NVIDIA" })[p] || p;

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  ["trace", "routing", "citations"].forEach((t) => { $(`#panel-${t}`).hidden = t !== name; });
}

// ---------------------------------------------------------------- networking

async function send(query) {
  const turn = createTurn(query);
  app.turn = turn;
  setBusy(true);
  const controller = app.controller = new AbortController();
  // The server sends a heartbeat at least every 15 s, so this much silence means the stream is dead.
  let stalled = false, watchdog;
  const arm = () => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => { stalled = true; controller.abort(); }, STALL_MS);
  };
  let failure = "the stream ended before the answer arrived";
  try {
    arm();
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: app.sessionId, policy: app.policy }),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      arm();
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        try { dispatch(turn, frame); } catch (err) { console.error("SSE frame failed", err, frame.slice(0, 300)); }
      }
    }
  } catch (err) {
    if (err.name === "AbortError" && !stalled) { turn.status.textContent = "Stopped."; turn.finished = true; }
    else failure = stalled ? "the stream stalled" : err.message;
  } finally {
    clearTimeout(watchdog);
  }
  if (!turn.finished) await recoverTurn(turn, query, failure);
  setBusy(false);
  app.controller = null;
}

// A stream can drop (proxy, network) while the server still finishes the turn and checkpoints it. The saved
// conversation is the source of truth, so load the answer from it instead of leaving the turn hanging.
async function recoverTurn(turn, query, why) {
  turn.status.textContent = "Connection interrupted; loading the saved answer…";
  for (let attempt = 0; attempt < RECOVER_ATTEMPTS; attempt++) {
    try {
      const resp = await fetch(`/api/sessions/${app.sessionId}`);
      const { messages } = await resp.json();
      const [asked, answered] = messages.slice(-2);
      if (answered?.role === "ai" && asked?.content === query) {
        turn.finished = true;
        turn.status.textContent = "";
        turn.answer.innerHTML = renderMarkdown(answered.content);
        linkCitations(turn.answer, turn.citations);
        turn.stats.hidden = false;
        turn.stats.textContent = `The live stream was interrupted (${why}); this answer was loaded from the saved conversation.`;
        return;
      }
    } catch { /* retry */ }
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  onError(turn, `${why}. Reload the page to check the saved conversation.`);
}

function dispatch(turn, frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue;  // heartbeat comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "trace") onTrace(turn, payload);
  else if (event === "token") onToken(turn, payload);
  else if (event === "citation") onCitation(turn, payload);
  else if (event === "done") onDone(turn, payload);
  else if (event === "error") onError(turn, payload.message);
}

function setBusy(busy) {
  const btn = $("#send");
  btn.textContent = busy ? "Stop" : "Send";
  btn.classList.toggle("stop", busy);
  $("#input").disabled = busy;
  document.querySelectorAll(".examples button").forEach((b) => { b.disabled = busy; });
  if (!busy) $("#input").focus();
}

function scrollToBottom(force = false) {
  const box = $("#messages");
  if (force || box.scrollHeight - box.scrollTop - box.clientHeight < 160) box.scrollTop = box.scrollHeight;
}

async function restoreHistory() {
  try {
    const resp = await fetch(`/api/sessions/${app.sessionId}`);
    if (!resp.ok) return;
    const { messages } = await resp.json();
    if (!messages.length) return;
    $("#empty")?.remove();
    for (const m of messages) (m.role === "human" ? addUserMessage(m.content) : addAssistantMessage(m.content));
  } catch { /* first visit or offline: start empty */ }
}

// ---------------------------------------------------------------- wiring

function setPolicy(policy) {
  app.policy = policy;
  store.set("agentlab.policy", policy);
  document.querySelectorAll("#policy button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.policy === policy)));
}

document.addEventListener("DOMContentLoaded", () => {
  setPolicy(app.policy);
  $("#policy").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b && !app.controller) setPolicy(b.dataset.policy); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

  const input = $("#input");
  const submit = () => {
    if (app.controller) { app.controller.abort(); return; }
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    send(q);
  };
  $("#composer").addEventListener("submit", (ev) => { ev.preventDefault(); submit(); });
  input.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submit(); } });
  $("#send").addEventListener("click", (ev) => { if (app.controller) { ev.preventDefault(); app.controller.abort(); } });
  $("#examples")?.addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b && !app.controller) send(b.dataset.q); });

  $("#new-chat").addEventListener("click", async () => {
    if (app.controller) app.controller.abort();
    app.sessionId = newSessionId();
    store.set("agentlab.session", app.sessionId);
    location.reload();
  });

  restoreHistory();
  input.focus();
});
