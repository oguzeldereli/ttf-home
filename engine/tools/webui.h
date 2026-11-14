// The chat page `ttf-home MODEL --serve` sends at GET / : one self-contained HTML file (no network fetches), talking to
// this server's own POST /v1/chat/completions with stream on. Reasoning (reasoning_content) shows in a fold above the
// answer; Stop aborts the request (the server sees the send fail and ends the generation).
#pragma once

static const char * const TTF_WEBUI = R"TTFUI(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ttf-home chat</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc;
  --user: #ecebe6; --accent: #2f5d8a; --accent-ink: #ffffff; --code: #f0efea; --think: #8a877d;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --ink: #e8e7e2; --muted: #9a998f; --line: #34332f;
    --user: #2a2a27; --accent: #6f9fd0; --accent-ink: #0d1b28; --code: #2a2a27; --think: #8f8c82;
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
       display: flex; flex-direction: column; }
header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--line);
         background: var(--panel); flex-wrap: wrap; }
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
header .model { color: var(--muted); font-size: 13px; }
header .sp { flex: 1; }
button, select, input, textarea { font: inherit; color: inherit; }
button { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 6px 12px; cursor: pointer; }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button:disabled { opacity: .5; cursor: default; }
#settings { display: none; gap: 14px; padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--panel);
            flex-wrap: wrap; align-items: center; font-size: 13px; }
#settings.open { display: flex; }
#settings label { display: flex; gap: 6px; align-items: center; color: var(--muted); }
#settings input[type=number] { width: 80px; padding: 3px 6px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); }
#settings textarea { width: min(520px, 100%); height: 52px; padding: 6px; border: 1px solid var(--line); border-radius: 6px;
                     background: var(--bg); resize: vertical; }
#log { flex: 1; overflow-y: auto; padding: 20px 16px 8px; }
.wrap { max-width: 820px; margin: 0 auto; }
.msg { margin: 0 0 18px; }
.msg.user .body { background: var(--user); border-radius: 12px; padding: 10px 14px; display: inline-block; max-width: 100%;
                  white-space: pre-wrap; word-wrap: break-word; }
.msg.user { text-align: right; }
.msg.user .body { text-align: left; }
.msg.assistant .body { word-wrap: break-word; }
.msg .body p { margin: 0 0 10px; }
.msg .body p:last-child { margin-bottom: 0; }
pre { background: var(--code); padding: 10px 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }
:not(pre) > code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
details.think { color: var(--think); font-size: 13.5px; margin-bottom: 8px; border-left: 2px solid var(--line); padding-left: 10px; }
details.think summary { cursor: pointer; user-select: none; }
details.think .tb { white-space: pre-wrap; margin-top: 6px; }
.stats { color: var(--muted); font-size: 12px; margin-top: 6px; }
.err { color: #b3261e; }
footer { padding: 10px 16px 14px; border-top: 1px solid var(--line); background: var(--panel); }
footer .wrap { display: flex; gap: 8px; align-items: flex-end; }
#input { flex: 1; resize: none; overflow-y: hidden; min-height: 44px; max-height: 220px; padding: 10px 12px; border: 1px solid var(--line);
         border-radius: 10px; background: var(--bg); }
.dots { color: var(--muted); }
.msg h3, .msg h4, .msg h5, .msg h6 { margin: 12px 0 6px; }
.msg ul, .msg ol { margin: 0 0 10px; padding-left: 22px; }
.empty { color: var(--muted); text-align: center; margin-top: 18vh; }
</style>
</head>
<body>
<header>
  <h1>ttf-home</h1><span class="model" id="model"></span><span class="sp"></span>
  <label style="font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center">
    <input type="checkbox" id="think"> thinking</label>
  <button id="btnSettings">Settings</button>
  <button id="btnNew">New chat</button>
</header>
<div id="settings">
  <label>temperature <input type="number" id="temp" step="0.05" min="0" max="2" value="0.7"></label>
  <label>top-k <input type="number" id="topk" step="1" min="0" max="256" value="20"></label>
  <label>max tokens <input type="number" id="maxtok" step="64" min="1" value="2048"></label>
  <label style="flex-basis:100%;align-items:flex-start">system <textarea id="system" placeholder="(none)"></textarea></label>
</div>
<div id="log"><div class="wrap" id="msgs"><div class="empty" id="empty">Ask something.</div></div></div>
<footer><div class="wrap">
  <textarea id="input" rows="1" placeholder="Message (Enter to send, Shift+Enter for a new line)"></textarea>
  <button class="primary" id="btnSend">Send</button>
  <button id="btnStop" style="display:none">Stop</button>
</div></footer>
<script>
const $ = id => document.getElementById(id);
const KEY = 'ttf-home-chat-v1';
let history = [];          // {role, content, reasoning}
let ctrl = null;

function load() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) || '{}');
    history = s.history || [];
    for (const k of ['temp', 'topk', 'maxtok', 'system']) if (s[k] !== undefined) $(k).value = s[k];
    if (s.think !== undefined) $('think').checked = s.think;
  } catch (e) {}
}
function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify({ history, think: $('think').checked,
      temp: $('temp').value, topk: $('topk').value, maxtok: $('maxtok').value, system: $('system').value }));
  } catch (e) {}
}

function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
// small markdown: fenced code, inline code, bold, italics, headings, lists, paragraphs
function md(src) {
  const parts = src.split(/```/);
  let out = '';
  parts.forEach((p, i) => {
    if (i % 2) {
      const nl = p.indexOf('\n');
      const code = nl >= 0 && /^[\w+-]*$/.test(p.slice(0, nl).trim()) ? p.slice(nl + 1) : p;
      out += '<pre><code>' + esc(code.replace(/\n$/, '')) + '</code></pre>';
      return;
    }
    const blocks = p.split(/\n{2,}/);
    for (let b of blocks) {
      if (!b.trim()) continue;
      let h = esc(b)
        .replace(/`([^`\n]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
      const lines = h.split('\n');
      if (lines.every(l => /^\s*([-*]|\d+\.)\s+/.test(l))) {
        const ol = /^\s*\d+\./.test(lines[0]);
        out += (ol ? '<ol start="' + parseInt(lines[0]) + '">' : '<ul>') + lines.map(l => '<li>' + l.replace(/^\s*([-*]|\d+\.)\s+/, '') + '</li>').join('') + (ol ? '</ol>' : '</ul>');
      } else if (/^#{1,4}\s/.test(h)) {
        const n = h.match(/^#+/)[0].length;
        out += `<h${n + 2}>` + h.replace(/^#+\s*/, '').replace(/\n/g, '<br>') + `</h${n + 2}>`;
      } else {
        out += '<p>' + h.replace(/\n/g, '<br>') + '</p>';
      }
    }
  });
  return out;
}

function render() {
  const box = $('msgs');
  box.innerHTML = '';
  if (!history.length) { box.innerHTML = '<div class="empty">Ask something.</div>'; return; }
  for (const m of history) box.appendChild(node(m));
  $('log').scrollTop = $('log').scrollHeight;
}
function node(m) {
  const d = document.createElement('div');
  d.className = 'msg ' + m.role;
  if (m.role === 'user') {
    d.innerHTML = '<div class="body"></div>';
    d.querySelector('.body').textContent = m.content;
  } else {
    d.innerHTML = (m.reasoning ? '<details class="think"><summary>reasoning</summary><div class="tb"></div></details>' : '') +
      '<div class="body">' + md(m.content || '') + '</div>' + (m.error ? '<div class="err">error: ' + esc(m.error) + '</div>' : '') +
      (m.stats ? '<div class="stats">' + esc(m.stats) + '</div>' : '');
    if (m.reasoning) d.querySelector('.tb').textContent = m.reasoning;
  }
  return d;
}

function busy(b) {
  $('btnSend').style.display = b ? 'none' : '';
  $('btnStop').style.display = b ? '' : 'none';
  $('input').disabled = b;
}

async function send() {
  const text = $('input').value.trim();
  if (!text || ctrl) return;
  $('input').value = ''; grow();
  history.push({ role: 'user', content: text });
  const a = { role: 'assistant', content: '', reasoning: '' };
  history.push(a);
  render();
  const el = $('msgs').lastChild;
  const msgs = [];
  if ($('system').value.trim()) msgs.push({ role: 'system', content: $('system').value });
  for (const m of history.slice(0, -1)) {
    if (m.role === 'assistant' && !m.content) continue;  // a stopped or failed reply
    const o = { role: m.role, content: m.content };
    if (m.reasoning) o.reasoning_content = m.reasoning;
    msgs.push(o);
  }
  const req = { messages: msgs, stream: true, temperature: +$('temp').value, top_k: +$('topk').value,
                max_tokens: +$('maxtok').value, chat_template_kwargs: { enable_thinking: $('think').checked } };
  ctrl = new AbortController(); busy(true); save();
  let t0 = 0, err = '', tm = null;
  const t_start = performance.now();
  const upd = () => {
    const parts = [];
    if (a.reasoning) parts.push('<details class="think"' + (a.content ? '' : ' open') +
      '><summary>reasoning</summary><div class="tb"></div></details>');
    parts.push('<div class="body">' + (a.content || a.reasoning || a.stats ? md(a.content) : '<span class="dots">…</span>') + '</div>');
    if (a.error) parts.push('<div class="err">error: ' + esc(a.error) + '</div>');
    if (a.stats) parts.push('<div class="stats">' + esc(a.stats) + '</div>');
    el.innerHTML = parts.join('');
    if (a.reasoning) el.querySelector('.tb').textContent = a.reasoning;
    const L = $('log');
    if (L.scrollHeight - L.scrollTop - L.clientHeight < 160) L.scrollTop = L.scrollHeight;
  };
  try {
    const r = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify(req), signal: ctrl.signal });
    if (!r.ok) throw new Error((await r.text()) || ('HTTP ' + r.status));
    const rd = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const ev = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of ev.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          let j; try { j = JSON.parse(data); } catch (e) { continue; }
          if (j.error) { err = j.error.message || JSON.stringify(j.error); continue; }
          if (j.timings) tm = j.timings;
          const dl = j.choices && j.choices[0] && j.choices[0].delta || {};
          if (dl.reasoning_content) a.reasoning += dl.reasoning_content;
          if (dl.content) a.content += dl.content;
          if ((dl.reasoning_content || dl.content) && !t0) t0 = performance.now();
        }
      }
      upd();
    }
  } catch (e) {
    if (e.name !== 'AbortError') err = String(e.message || e);
  }
  const s = [];
  if (tm) {
    s.push(tm.prompt_n + ' prompt tokens' + (tm.cache_n ? ' (+' + tm.cache_n + ' reused)' : '') + ' in ' + (tm.prompt_ms / 1000).toFixed(2) + ' s');
    s.push(tm.predicted_n + ' tokens, ' + tm.predicted_per_second.toFixed(1) + ' tok/s');
  } else if (t0) s.push('first token ' + ((t0 - t_start) / 1000).toFixed(2) + ' s');
  if (ctrl.signal.aborted) s.push('stopped');
  a.stats = s.filter(Boolean).join(' · ');
  if (err) a.error = err;
  if (!a.content && !a.reasoning && !err) a.stats = a.stats || 'stopped';
  ctrl = null; busy(false); upd(); save(); $('input').focus();
}

function grow() { const t = $('input'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight + 2, 220) + 'px'; t.style.overflowY = t.scrollHeight > 220 ? 'auto' : 'hidden'; }
$('input').addEventListener('input', grow);
$('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
$('btnSend').onclick = send;
$('btnStop').onclick = () => ctrl && ctrl.abort();
$('btnNew').onclick = () => { if (ctrl) ctrl.abort(); history = []; save(); render(); $('input').focus(); };
$('btnSettings').onclick = () => $('settings').classList.toggle('open');
for (const k of ['think', 'temp', 'topk', 'maxtok', 'system']) $(k).addEventListener('change', save);
fetch('/v1/models').then(r => r.json()).then(j => { const m = j.data && j.data[0] && j.data[0].id; if (m) { $('model').textContent = m; document.title = m + ' · ttf-home'; } }).catch(() => {});
fetch('/props').then(r => r.json()).then(j => { if (j.think !== undefined && localStorage.getItem(KEY) === null) $('think').checked = j.think; }).catch(() => {});
load(); render(); $('input').focus();
</script>
</body>
</html>
)TTFUI";
