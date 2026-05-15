// Nova Retrieve web client — talks to FastAPI backend over SSE (POST /chat/stream).

const $chat = document.getElementById('chat');
const $form = document.getElementById('composer');
const $input = document.getElementById('input');
const $send = document.getElementById('sendBtn');
const $stop = document.getElementById('stopBtn');
const $empty = document.getElementById('emptyState');
const $healthDot = document.getElementById('healthDot');
const $healthText = document.getElementById('healthText');

let abortCtrl = null;

// ---------- health check ----------
async function checkHealth() {
  try {
    const r = await fetch('/health', { cache: 'no-store' });
    if (r.ok) {
      $healthDot.className = 'health-dot online';
      $healthText.textContent = '后端在线';
      return;
    }
  } catch (_) { /* ignore */ }
  $healthDot.className = 'health-dot offline';
  $healthText.textContent = '后端离线';
}
checkHealth();
setInterval(checkHealth, 15000);

// ---------- example chips ----------
document.querySelectorAll('.chip').forEach(c => {
  c.addEventListener('click', () => {
    $input.value = c.dataset.q;
    $input.focus();
    autoresize();
  });
});

// ---------- composer ----------
function autoresize() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 200) + 'px';
}
$input.addEventListener('input', autoresize);
$input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $form.requestSubmit();
  }
});

$form.addEventListener('submit', e => {
  e.preventDefault();
  const q = $input.value.trim();
  if (!q || abortCtrl) return;
  $input.value = '';
  autoresize();
  ask(q);
});

$stop.addEventListener('click', () => {
  if (abortCtrl) abortCtrl.abort();
});

// ---------- chat list helpers ----------
function ensureInner() {
  let inner = $chat.querySelector('.chat-inner');
  if (!inner) {
    inner = document.createElement('div');
    inner.className = 'chat-inner';
    $chat.appendChild(inner);
  }
  return inner;
}

function appendUserMsg(text) {
  if ($empty) $empty.remove();
  const inner = ensureInner();
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="bubble"></div>`;
  div.querySelector('.bubble').textContent = text;
  inner.appendChild(div);
  scrollToBottom();
}

function appendAssistantMsg() {
  const inner = ensureInner();
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = `
    <div class="bubble">
      <div class="steps"></div>
      <div class="answer"></div>
      <div class="meta"></div>
    </div>
  `;
  inner.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  $chat.scrollTop = $chat.scrollHeight;
}

// ---------- agent step UI ----------
// Map of node-name -> human label
const STEP_LABELS = {
  rewrite_query: '改写查询',
  route_question: '路由决策',
  retrieve: '向量检索',
  grade_documents: '相关性评分',
  transform_query: '重写查询(重试)',
  web_search: 'Web 检索',
  generate: '生成回答',
  hallucination_grader: '幻觉检测',
  answer_grader: '答案评估',
};

function upsertStep(stepsEl, node, status, elapsedMs) {
  const id = 'step-' + node;
  let row = stepsEl.querySelector(`[data-step="${node}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'step ' + status;
    row.dataset.step = node;
    row.innerHTML = `
      <span class="icon"></span>
      <span class="name"></span>
      <span class="elapsed"></span>
    `;
    row.querySelector('.name').textContent = STEP_LABELS[node] || node;
    stepsEl.appendChild(row);
  } else {
    row.className = 'step ' + status;
  }
  if (elapsedMs != null) {
    row.querySelector('.elapsed').textContent = `${elapsedMs.toFixed(0)} ms`;
  }
  return row;
}

// ---------- SSE parsing ----------
// sse-starlette emits `\r\n` line endings; standard says event terminator is a
// blank line. We normalize CRLF to LF then split on blank lines (\n\n).
async function* parseSSE(resp) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';

  const parseBlock = (block) => {
    let event = 'message';
    const dataLines = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return null;
    let data = null;
    try { data = JSON.parse(dataLines.join('\n')); } catch (e) {
      console.warn('[sse] JSON parse failed', e, dataLines.join('\n'));
    }
    return { event, data };
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      // flush any trailing event the server didn't terminate with \n\n
      buf += decoder.decode();
      const tail = buf.trim();
      if (tail) {
        const ev = parseBlock(tail);
        if (ev) yield ev;
      }
      break;
    }
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let i;
    while ((i = buf.indexOf('\n\n')) !== -1) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const ev = parseBlock(block);
      if (ev) yield ev;
    }
  }
}

// ---------- main ask flow ----------
async function ask(question) {
  appendUserMsg(question);
  const msgEl = appendAssistantMsg();
  const stepsEl = msgEl.querySelector('.steps');
  const answerEl = msgEl.querySelector('.answer');
  const metaEl = msgEl.querySelector('.meta');

  $send.hidden = true;
  $stop.hidden = false;
  $send.disabled = true;

  abortCtrl = new AbortController();
  let activeNode = null;
  let timings = [];

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: abortCtrl.signal,
    });

    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${detail}`);
    }

    for await (const ev of parseSSE(resp)) {
      if (ev.event === 'step') {
        const node = ev.data?.node;
        if (!node) continue;

        // mark previous as done (if any)
        if (activeNode && activeNode !== node) {
          const prev = stepsEl.querySelector(`[data-step="${activeNode}"]`);
          if (prev) prev.className = 'step done';
        }
        activeNode = node;

        // pull elapsed from the timings the node just emitted
        const ts = ev.data?.state_delta?.timings;
        const elapsed = Array.isArray(ts) && ts.length ? ts[ts.length - 1].elapsed_ms : null;
        upsertStep(stepsEl, node, elapsed != null ? 'done' : 'running', elapsed);

        // accumulate timing list as it flows in
        if (Array.isArray(ts)) timings = timings.concat(ts);
        scrollToBottom();
      } else if (ev.event === 'answer') {
        // mark final node done
        if (activeNode) {
          const prev = stepsEl.querySelector(`[data-step="${activeNode}"]`);
          if (prev) prev.className = 'step done';
        }
        const d = ev.data || {};
        console.log('[answer event]', d);
        const txt = (d.answer || '').trim();
        if (txt) {
          renderAnswer(answerEl, txt);
        } else {
          answerEl.innerHTML = '<em style="color:var(--muted)">（后端返回了空答案，请查看 DevTools Console 与服务端日志）</em>';
        }
        renderMeta(metaEl, d, timings, d.total_ms);
        scrollToBottom();
      } else if (ev.event === 'error') {
        renderError(answerEl, ev.data?.detail || '未知错误');
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      renderError(answerEl, '已停止生成。');
    } else {
      renderError(answerEl, err.message || String(err));
    }
  } finally {
    abortCtrl = null;
    $send.hidden = false;
    $stop.hidden = true;
    $send.disabled = false;
    $input.focus();
  }
}

// ---------- rendering ----------
function renderAnswer(el, md) {
  // marked.js CDN bundle has changed shape across versions; try every known
  // entry point, fall back to a minimal markdown→html if nothing loaded.
  try {
    const m = window.marked;
    let html = null;
    if (m && typeof m.parse === 'function') html = m.parse(md);
    else if (typeof m === 'function') html = m(md);
    if (html != null) {
      el.innerHTML = html;
      return;
    }
  } catch (e) {
    console.warn('[renderAnswer] marked failed, falling back to plain', e);
  }
  // Plain fallback — escape, then preserve paragraphs + line breaks.
  el.innerHTML = md
    .split(/\n{2,}/)
    .map((p) => `<p>${escapeHtml(p).replace(/\n/g, '<br>')}</p>`)
    .join('');
}

function renderError(el, msg) {
  el.innerHTML = `<div class="tag err">错误</div> <span>${escapeHtml(msg)}</span>`;
}

function renderMeta(el, data, timings, totalMs) {
  el.innerHTML = '';
  const tags = [];
  if (data.route) tags.push(tag(data.route === 'web_search' ? 'Web 检索' : '向量检索'));
  if (data.hallucinated === true) tags.push(tag('疑似幻觉', 'warn'));
  if (data.answer_relevant === false) tags.push(tag('答案相关性低', 'warn'));
  if (totalMs != null) tags.push(tag(`总耗时 ${totalMs.toFixed(0)} ms`));
  tags.forEach(t => el.appendChild(t));

  // citations
  if (Array.isArray(data.citations) && data.citations.length) {
    const det = document.createElement('details');
    det.className = 'citations';
    det.innerHTML = `<summary>引用来源 (${data.citations.length})</summary>`;
    data.citations.forEach(c => {
      const item = document.createElement('div');
      item.className = 'cite-item';
      const title = c.title || (c.source ? c.source.split('/').pop() : '来源');
      item.innerHTML = `
        <span class="cite-idx">${c.index}</span>
        <div class="cite-body">
          <div class="cite-title">${escapeHtml(title)}</div>
          <div class="cite-src">${linkOrText(c.source)}</div>
        </div>
      `;
      det.appendChild(item);
    });
    el.appendChild(det);
  }

  // timings
  if (Array.isArray(timings) && timings.length) {
    const det = document.createElement('details');
    det.className = 'timings';
    det.innerHTML = `<summary>各步骤耗时 (${timings.length} 步)</summary>`;
    const table = document.createElement('table');
    table.className = 'timing-table';
    const max = Math.max(...timings.map(t => t.elapsed_ms || 0), 1);
    const sorted = [...timings].sort((a, b) => (a.seq || 0) - (b.seq || 0));
    sorted.forEach(t => {
      const tr = document.createElement('tr');
      const w = Math.max(2, (t.elapsed_ms / max) * 120);
      tr.innerHTML = `
        <td>${STEP_LABELS[t.step] || t.step}</td>
        <td><span class="timing-bar" style="width:${w.toFixed(0)}px"></span>${t.elapsed_ms.toFixed(0)} ms</td>
      `;
      table.appendChild(tr);
    });
    det.appendChild(table);
    el.appendChild(det);
  }
}

function tag(text, kind = '') {
  const s = document.createElement('span');
  s.className = 'tag' + (kind ? ' ' + kind : '');
  s.textContent = text;
  return s;
}

function linkOrText(src) {
  if (!src) return '';
  const safe = escapeHtml(src);
  if (/^https?:\/\//.test(src)) {
    return `<a href="${safe}" target="_blank" rel="noopener">${safe}</a>`;
  }
  return safe;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
