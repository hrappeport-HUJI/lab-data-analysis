// chat: a small chat box answered by Claude through the claude.ai artifact `sample` capability.
// Works only when the page is published as a claude.ai artifact with capabilities {"sample": {}}; elsewhere it says so.
// The model computes through the page's data tools (TOOLS), never from memory.
const RULES = () => `You are the data assistant inside a scientific data viewer.
Dataset: ${S.title || ""}. ${S.subtitle || ""}
Rows: ${N}. Each row: ${S.row_meaning || "one record"}. Row label format: ${S.label || "id"}.
Definitions and context:
${S.notes || "(none)"}
Columns (name: label): ${Object.entries(COLS).map(([k, m]) => `${k}: ${m.label}${m.kind === "category" ? " [category: " + m.order.join(", ") + "]" : ""}`).join("; ")}.
The plot currently shows: ${JSON.stringify(viewOf())}.
Rules: compute every number with the tools; never estimate or invent values. Report n with every statistic. Keep answers short (a few sentences or a small table in plain text). If asked to show something, call set_view. Say plainly when the data cannot answer a question. Plain text only, no markdown headings.`;
let turns = [], chatCtl = null, sample = null;
function addMsg(cls, text) { const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text; const log = document.getElementById("chat-log"); log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
const COPY = { not_granted: "Claude isn't allowed for this page. You can allow it from the page's permissions.", sampling_disabled: "Claude isn't available for this account.",
  rate_limited: "Too many requests just now. Try again in a minute.", session_expired: "Please sign in again.", refused: "Claude declined that request. Try rephrasing it.",
  empty_completion: "No answer came back. Try a simpler question.", prompt_too_large: "The conversation got too long. Clear it and ask again.", tools_unavailable: "This view can't run the data tools.",
  upstream_error: "The connection dropped. Try again." };
async function ask() {
  const box = document.getElementById("chat-text"), q = box.value.trim(); if (!q || !sample) return;
  box.value = ""; addMsg("you", q); turns.push({ role: "user", content: q }); if (turns.length > 16) turns = turns.slice(-16);
  while (turns.length && turns[0].role !== "user") turns.shift();
  const bubble = addMsg("bot", "Thinking…"); const send = document.getElementById("chat-send"), stop = document.getElementById("chat-stop");
  send.disabled = true; stop.hidden = false; chatCtl = new AbortController();
  const tools = TOOLS.map(t => ({ ...t, execute: async (input, ctx) => { addMsg("tool", `→ ${t.name} ${JSON.stringify(input).slice(0, 160)}`); return t.execute(input || {}, ctx); } }));
  let permanent = false;
  try {
    const res = await sample([{ role: "user", content: RULES() }, ...turns], { tools, signal: chatCtl.signal, modelTier: document.getElementById("tier").value,
      onText: ({ text }) => { bubble.textContent = text; document.getElementById("chat-log").appendChild(bubble); } });
    bubble.textContent = res.text + (res.truncated ? "\n(cut short — ask for less at a time)" : ""); turns.push({ role: "assistant", content: res.text });
  } catch (e) {
    permanent = ["not_granted", "sampling_disabled", "not_declared", "capability_disabled", "capability_removed", "tools_unavailable"].includes(e.code);
    if (e.code === "cancelled") bubble.textContent = e.text ? e.text + "\n(stopped)" : "Stopped.";
    else if (e.code === "refused") bubble.textContent = COPY.refused;
    else if (permanent) bubble.textContent = COPY[e.code] || "Claude isn't available here.";
    else bubble.textContent = (e.text ? e.text + "\n\n" : "") + (COPY[e.code] || COPY.upstream_error);
  } finally { send.disabled = permanent; stop.hidden = true; }
}
async function initChat() {
  const hint = document.getElementById("chat-hint"), send = document.getElementById("chat-send");
  send.disabled = true;
  sample = window.claude ? await window.claude.use("sample").catch(() => null) : null;
  if (!sample) { hint.textContent = "The chat needs the claude.ai viewer: open this page from its claude.ai link."; return; }
  const lim = await sample.limits().catch(() => null);
  if (lim && !lim.tools) { hint.textContent = "This viewer can't run the data tools the chat needs. Try the claude.ai website."; return; }
  send.disabled = false;
  send.onclick = ask; document.getElementById("chat-stop").onclick = () => chatCtl?.abort();
  document.getElementById("chat-text").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
}
// features.chat = {"placeholder": "an example question"} (optional)
viewerModule("chat", { init(cfg) { if (cfg.placeholder) document.getElementById("chat-text").placeholder = cfg.placeholder; initChat(); } });
