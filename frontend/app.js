/**
 * Frontend controller for the deposition dashboard.
 *
 * Responsibilities:
 * - fetch API data
 * - maintain UI state (selected deposition, chat history, processing indicators)
 * - render timeline/list/detail/chat panels
 * - enforce LLM model readiness before invoking expensive operations
 */
const els = {
  caseId: document.getElementById('caseId'),
  directory: document.getElementById('directory'),
  llmSelect: document.getElementById('llmSelect'),
  skipReassess: document.getElementById('skipReassess'),
  refreshModelsBtn: document.getElementById('refreshModelsBtn'),
  ingestBtn: document.getElementById('ingestBtn'),
  refreshBtn: document.getElementById('refreshBtn'),
  status: document.getElementById('status'),
  statusIndicator: document.getElementById('statusIndicator'),
  statusClock: document.getElementById('statusClock'),
  list: document.getElementById('depositionList'),
  timeline: document.getElementById('timeline'),
  timelineBack: document.getElementById('timelineBack'),
  timelineForward: document.getElementById('timelineForward'),
  detailEmpty: document.getElementById('detailEmpty'),
  detailBody: document.getElementById('detailBody'),
  detailWitness: document.getElementById('detailWitness'),
  detailSummary: document.getElementById('detailSummary'),
  detailExplanation: document.getElementById('detailExplanation'),
  detailContradictions: document.getElementById('detailContradictions'),
  focusedReasoning: document.getElementById('focusedReasoning'),
  reasoningProgress: document.getElementById('reasoningProgress'),
  reasoningClock: document.getElementById('reasoningClock'),
  focusedReasoningBody: document.getElementById('focusedReasoningBody'),
  chatMessages: document.getElementById('chatMessages'),
  chatProgress: document.getElementById('chatProgress'),
  chatClock: document.getElementById('chatClock'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  chatSendBtn: document.getElementById('chatSendBtn'),
};

let depositions = [];
let selectedDepositionId = null;
let chatHistory = [];
let uiOpsInFlight = 0;
let llmOpsInFlight = 0;
const timerHandles = {
  llm: null,
  reasoning: null,
  chat: null,
};

async function api(path, options = {}) {
  /** Perform JSON API calls and normalize non-2xx errors. */
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(payload.detail || 'Request failed');
  }
  return response.json();
}

function setStatus(message) {
  /** Render a short status message in the control panel. */
  els.status.textContent = message;
}

function formatSeconds(startMs) {
  /** Format elapsed time from a start timestamp (performance.now). */
  const seconds = Math.max(0, (performance.now() - startMs) / 1000);
  return `${seconds.toFixed(1)}s`;
}

function startClock(key, el) {
  /** Start a requestAnimationFrame-driven elapsed-time clock. */
  stopClock(key);
  const startMs = performance.now();
  const tick = () => {
    el.textContent = formatSeconds(startMs);
    timerHandles[key] = window.requestAnimationFrame(tick);
  };
  tick();
}

function stopClock(key) {
  /** Stop a running clock by timer key. */
  if (timerHandles[key] !== null) {
    window.cancelAnimationFrame(timerHandles[key]);
    timerHandles[key] = null;
  }
}

async function nextPaint() {
  /** Wait one paint frame so status/progress UI is visible before async work. */
  await new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

function setControlsDisabled(disabled) {
  /** Toggle top-level controls while UI operations are in-flight. */
  els.ingestBtn.disabled = disabled;
  els.refreshBtn.disabled = disabled;
  els.refreshModelsBtn.disabled = disabled;
  els.llmSelect.disabled = disabled;
  els.skipReassess.disabled = disabled;
}

function startUiProcessing(message) {
  /** Enter UI-processing mode with optional status update. */
  uiOpsInFlight += 1;
  if (message) {
    setStatus(message);
  }
  setControlsDisabled(true);
}

function endUiProcessing() {
  /** Exit UI-processing mode and re-enable controls when queue is empty. */
  uiOpsInFlight = Math.max(0, uiOpsInFlight - 1);
  if (uiOpsInFlight === 0) {
    setControlsDisabled(false);
  }
}

function startLlmProcessing(message) {
  /** Enter LLM-processing mode and start global processing indicator clock. */
  llmOpsInFlight += 1;
  if (message) {
    setStatus(message);
  }
  if (llmOpsInFlight === 1) {
    els.statusIndicator.classList.remove('hidden');
    els.statusClock.classList.remove('hidden');
    startClock('llm', els.statusClock);
  }
}

function endLlmProcessing() {
  /** Exit LLM-processing mode and hide global indicator when idle. */
  llmOpsInFlight = Math.max(0, llmOpsInFlight - 1);
  if (llmOpsInFlight === 0) {
    stopClock('llm');
    els.statusIndicator.classList.add('hidden');
    els.statusClock.classList.add('hidden');
  }
}

function setReasoningProcessing(active, message = '') {
  /** Toggle focused contradiction reasoning spinner/clock state. */
  els.reasoningProgress.classList.toggle('hidden', !active);
  if (active) {
    if (message) {
      els.focusedReasoningBody.textContent = message;
    }
    startClock('reasoning', els.reasoningClock);
    return;
  }
  stopClock('reasoning');
}

function setChatProcessing(active) {
  /** Toggle chat in-flight state and disable chat input/send during generation. */
  els.chatProgress.classList.toggle('hidden', !active);
  els.chatInput.disabled = active;
  els.chatSendBtn.disabled = active;
  if (active) {
    startClock('chat', els.chatClock);
    return;
  }
  stopClock('chat');
}

function depositionSortByScore(a, b) {
  /** Sort helper for contradiction risk score (descending). */
  return (b.contradiction_score || 0) - (a.contradiction_score || 0);
}

function timelineSort(a, b) {
  /** Sort helper for timeline ordering by deposition date, then filename. */
  const aTime = parseDepositionDate(a.deposition_date);
  const bTime = parseDepositionDate(b.deposition_date);
  if (aTime !== null && bTime !== null) {
    return aTime - bTime;
  }
  if (aTime !== null) {
    return -1;
  }
  if (bTime !== null) {
    return 1;
  }
  return (a.file_name || '').localeCompare(b.file_name || '');
}

function parseDepositionDate(value) {
  /** Parse deposition date into epoch milliseconds; return null when invalid. */
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function displayDate(value) {
  /** Human-friendly date label for timeline cards. */
  const parsed = parseDepositionDate(value);
  if (parsed === null) {
    return 'Undated';
  }
  return new Date(parsed).toLocaleDateString();
}

function encodeLlmOption(provider, model) {
  /** Encode provider/model pair into select option value. */
  return `${provider}::${model}`;
}

function decodeLlmOption(value) {
  /** Decode select option value into provider/model pair. */
  const [provider, ...rest] = String(value || '').split('::');
  const model = rest.join('::');
  if (!provider || !model) {
    return { provider: 'openai', model: 'gpt-5.2' };
  }
  return { provider, model };
}

function getSelectedLlm() {
  /** Return currently selected LLM pair, with safe default fallback. */
  if (!els.llmSelect.value) {
    return { provider: 'openai', model: 'gpt-5.2' };
  }
  return decodeLlmOption(els.llmSelect.value);
}

function getSelectedLlmLabel() {
  /** Return user-facing label for current model selection. */
  const { provider, model } = getSelectedLlm();
  return `${provider === 'openai' ? 'ChatGPT' : 'Ollama'} (${model})`;
}

function getSelectedOptionMeta(options) {
  /** Lookup metadata for the currently selected option in an options payload. */
  const selectedValue = els.llmSelect.value;
  return options.find((item) => encodeLlmOption(item.provider, item.model) === selectedValue) || null;
}

async function loadLlmOptions({ silent = false, forceProbe = false } = {}) {
  /** Fetch model options and render readiness-aware dropdown entries. */
  const previous = els.llmSelect.value;
  if (!silent) {
    startUiProcessing(
      forceProbe ? 'Refreshing and validating LLMs...' : 'Loading available LLMs...'
    );
  }
  try {
    const query = forceProbe ? '?force_probe=true' : '';
    const payload = await api(`/api/llm-options${query}`);
    const options = payload.options || [];
    els.llmSelect.innerHTML = '';

    for (const item of options) {
      const option = document.createElement('option');
      option.value = encodeLlmOption(item.provider, item.model);
      option.textContent = item.operational ? item.label : `${item.label} (Unavailable)`;
      option.disabled = !item.operational;
      if (!item.operational) {
        const fix = item.possible_fix ? ` Fix: ${item.possible_fix}` : '';
        option.title = `${item.error || 'Model is not operational.'}${fix}`;
      }
      els.llmSelect.appendChild(option);
    }

    const defaultValue = encodeLlmOption(payload.selected_provider, payload.selected_model);
    const values = options.map((item) => encodeLlmOption(item.provider, item.model));
    const operationalValues = options
      .filter((item) => item.operational)
      .map((item) => encodeLlmOption(item.provider, item.model));
    const selected = values.includes(previous)
      ? previous
      : operationalValues.includes(defaultValue)
        ? defaultValue
        : operationalValues[0] || values[0] || defaultValue;

    if (!options.length) {
      const fallback = document.createElement('option');
      fallback.value = defaultValue;
      fallback.textContent = `ChatGPT - ${payload.selected_model}`;
      els.llmSelect.appendChild(fallback);
    }

    els.llmSelect.value = selected;
    const selectedMeta = getSelectedOptionMeta(options);
    if (selectedMeta && !selectedMeta.operational) {
      const fix = selectedMeta.possible_fix ? ` Possible fix: ${selectedMeta.possible_fix}` : '';
      setStatus(`Selected model unavailable: ${selectedMeta.error || 'Unknown error'}.${fix}`);
      return;
    }

    if (!silent) {
      setStatus(`LLM options loaded. Current: ${getSelectedLlmLabel()}.`);
    }
  } finally {
    if (!silent) {
      endUiProcessing();
    }
  }
}

function ensureLlmFallbackOption() {
  /** Ensure selector still has at least one option when options API fails. */
  if (els.llmSelect.options.length > 0) {
    return;
  }
  const fallback = document.createElement('option');
  fallback.value = 'openai::gpt-5.2';
  fallback.textContent = 'ChatGPT - gpt-5.2';
  els.llmSelect.appendChild(fallback);
  els.llmSelect.value = fallback.value;
}

function selectedLlmIsOperational() {
  /** Return true only when currently selected option is enabled/operational. */
  const selected = els.llmSelect.selectedOptions[0];
  return !!selected && !selected.disabled;
}

function renderTimeline() {
  /** Render chronological timeline strip for all loaded depositions. */
  els.timeline.innerHTML = '';
  if (!depositions.length) {
    els.timeline.innerHTML = '<div class="muted">Timeline will appear after loading depositions.</div>';
    return;
  }

  const ordered = [...depositions].sort(timelineSort);
  for (const dep of ordered) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `timeline-item ${dep._id === selectedDepositionId ? 'active' : ''}`;
    item.innerHTML = `
      <strong>${displayDate(dep.deposition_date)}</strong>
      <span>${dep.witness_name || 'Unknown Witness'}</span>
      <small>Score ${dep.contradiction_score || 0}</small>
    `;
    item.addEventListener('click', () => selectDeposition(dep._id));
    els.timeline.appendChild(item);
  }
}

function renderDepositions() {
  /** Render score-ranked deposition cards in the risk list panel. */
  const filtered = [...depositions].sort(depositionSortByScore);
  els.list.innerHTML = '';

  if (!filtered.length) {
    els.list.innerHTML = '<div class="muted">No depositions loaded for this case yet.</div>';
    return;
  }

  for (const dep of filtered) {
    const card = document.createElement('article');
    card.className = `dep-card ${dep._id === selectedDepositionId ? 'active' : ''}`;
    card.innerHTML = `
      <div class="dep-meta">
        <strong>${dep.witness_name || 'Unknown Witness'}</strong>
        <span class="badge ${dep.flagged ? 'flagged' : 'clear'}">${dep.flagged ? 'Flagged' : 'Clear'}</span>
      </div>
      <div class="muted">${dep.file_name || ''}</div>
      <div class="score-number">Risk Score: ${dep.contradiction_score || 0}</div>
    `;
    card.addEventListener('click', () => selectDeposition(dep._id));
    els.list.appendChild(card);
  }
}

function getOverallShortAnswer(dep) {
  /** Build concise summary line from contradiction explanation and score. */
  const raw = (dep.contradiction_explanation || '').trim();
  if (!raw) {
    return dep.flagged
      ? `Likely conflict detected with score ${dep.contradiction_score || 0}.`
      : `No material contradiction flagged (score ${dep.contradiction_score || 0}).`;
  }
  const firstSentence = raw.split(/(?<=[.!?])\s+/)[0];
  return firstSentence || raw;
}

function clearFocusedReasoning() {
  /** Reset focused contradiction reasoning panel state. */
  els.focusedReasoning.classList.add('hidden');
  setReasoningProcessing(false);
  els.focusedReasoningBody.textContent = '';
}

function renderDetail(dep) {
  /** Render selected deposition details and clickable contradiction items. */
  if (!dep) {
    els.detailEmpty.classList.remove('hidden');
    els.detailBody.classList.add('hidden');
    clearFocusedReasoning();
    return;
  }

  els.detailEmpty.classList.add('hidden');
  els.detailBody.classList.remove('hidden');

  els.detailWitness.textContent = `${dep.witness_name || 'Unknown'} (${dep.witness_role || 'Unknown role'})`;
  els.detailSummary.textContent = dep.summary || 'No summary available.';
  els.detailExplanation.textContent = getOverallShortAnswer(dep);

  const contradictions = dep.contradictions || [];
  els.detailContradictions.innerHTML = '';
  clearFocusedReasoning();

  if (!contradictions.length) {
    els.detailContradictions.innerHTML = '<li>No contradictions identified.</li>';
    return;
  }

  for (const contradiction of contradictions) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'contradiction-item-btn';
    button.textContent = `${contradiction.topic}: ${contradiction.rationale} (vs ${contradiction.other_witness_name}, severity ${contradiction.severity})`;
    button.addEventListener('click', () =>
      reasonAboutContradiction({
        caseId: els.caseId.value.trim(),
        depositionId: dep._id,
        contradiction,
      }).catch((err) => setStatus(err.message))
    );
    li.appendChild(button);
    els.detailContradictions.appendChild(li);
  }
}

async function reasonAboutContradiction({ caseId, depositionId, contradiction }) {
  /** Call focused reasoning endpoint for one contradiction list item. */
  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }
  const llm = getSelectedLlm();
  els.focusedReasoning.classList.remove('hidden');
  startLlmProcessing('Attorney is processing the selected contradiction...');
  setReasoningProcessing(true, 'Re-analyzing this detail item...');
  await nextPaint();

  try {
    const payload = await api('/api/reason-contradiction', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        deposition_id: depositionId,
        contradiction,
        llm_provider: llm.provider,
        llm_model: llm.model,
      }),
    });

    els.focusedReasoningBody.textContent = payload.response;
  } finally {
    setReasoningProcessing(false);
    endLlmProcessing();
  }
}

function addMessage(role, content) {
  /** Append a chat bubble and keep scroll anchored at latest message. */
  const item = document.createElement('div');
  item.className = `msg ${role}`;
  item.textContent = content;
  els.chatMessages.appendChild(item);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function loadDepositions() {
  /** Load and render all depositions for the active case id. */
  const caseId = els.caseId.value.trim();
  if (!caseId) {
    setStatus('Enter a case ID first.');
    return;
  }

  startUiProcessing(`Loading case ${caseId}...`);
  try {
    depositions = await api(`/api/depositions/${encodeURIComponent(caseId)}`);

    if (selectedDepositionId) {
      const stillExists = depositions.some((item) => item._id === selectedDepositionId);
      if (!stillExists) {
        selectedDepositionId = null;
      }
    }

    renderTimeline();
    renderDepositions();

    if (selectedDepositionId) {
      const selected = await api(`/api/deposition/${encodeURIComponent(selectedDepositionId)}`);
      renderDetail(selected);
    } else {
      renderDetail(null);
    }

    setStatus(`Loaded ${depositions.length} depositions.`);
  } finally {
    endUiProcessing();
  }
}

async function selectDeposition(depositionId) {
  /** Select a deposition and load full detail payload into detail panel. */
  selectedDepositionId = depositionId;
  renderTimeline();
  renderDepositions();

  startUiProcessing('Loading deposition detail...');
  try {
    const dep = await api(`/api/deposition/${encodeURIComponent(depositionId)}`);
    renderDetail(dep);
    setStatus(`Viewing ${dep.witness_name || 'selected witness'} deposition.`);
  } finally {
    endUiProcessing();
  }

  chatHistory = [];
  els.chatMessages.innerHTML = '';
  addMessage('assistant', 'Short answer mode is active. Ask for next legal actions or request a deeper bullet breakdown.');
}

async function ingestCase() {
  /** Start full-case ingestion using selected provider/model selection. */
  const caseId = els.caseId.value.trim();
  const directory = els.directory.value.trim();
  const llm = getSelectedLlm();
  const skipReassess = !!els.skipReassess.checked;

  if (!caseId || !directory) {
    setStatus('Case ID and folder path are required.');
    return;
  }

  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }

  startUiProcessing(
    skipReassess
      ? 'Running fast ingest (skip full case reassess)...'
      : 'Running LangGraph workflow over all .txt depositions...'
  );
  startLlmProcessing('Legal Clerk is processing depositions...');
  await nextPaint();
  try {
    await api('/api/ingest-case', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        directory,
        llm_provider: llm.provider,
        llm_model: llm.model,
        skip_reassess: skipReassess,
      }),
    });
  } finally {
    endLlmProcessing();
    endUiProcessing();
  }

  await loadDepositions();
}

async function sendChat(event) {
  /** Submit chat prompt and render assistant response into chat panel. */
  event.preventDefault();

  if (!selectedDepositionId) {
    setStatus('Select a deposition before chatting.');
    return;
  }

  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }

  const message = els.chatInput.value.trim();
  if (!message) {
    return;
  }

  els.chatInput.value = '';
  addMessage('user', message);
  chatHistory.push({ role: 'user', content: message });

  const caseId = els.caseId.value.trim();
  const llm = getSelectedLlm();
  startLlmProcessing('Attorney is processing your question...');
  setChatProcessing(true);
  await nextPaint();
  try {
    const payload = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        deposition_id: selectedDepositionId,
        message,
        history: chatHistory,
        llm_provider: llm.provider,
        llm_model: llm.model,
      }),
    });

    addMessage('assistant', payload.response);
    chatHistory.push({ role: 'assistant', content: payload.response });
  } finally {
    setChatProcessing(false);
    endLlmProcessing();
  }
}

els.ingestBtn.addEventListener('click', () => ingestCase().catch((err) => setStatus(err.message)));
els.refreshBtn.addEventListener('click', () => loadDepositions().catch((err) => setStatus(err.message)));
els.refreshModelsBtn.addEventListener('click', () =>
  loadLlmOptions({ forceProbe: true }).catch((err) => setStatus(err.message))
);
els.llmSelect.addEventListener('change', () => {
  const selected = els.llmSelect.selectedOptions[0];
  if (selected && selected.disabled) {
    setStatus('Selected model is unavailable. Choose an operational model or click Refresh Models.');
    return;
  }
  setStatus(`LLM selected: ${getSelectedLlmLabel()}.`);
});
els.timelineBack.addEventListener('click', () => {
  els.timeline.scrollBy({ left: -320, behavior: 'smooth' });
});
els.timelineForward.addEventListener('click', () => {
  els.timeline.scrollBy({ left: 320, behavior: 'smooth' });
});
els.chatForm.addEventListener('submit', (event) => sendChat(event).catch((err) => setStatus(err.message)));

renderDetail(null);
renderTimeline();
renderDepositions();
loadLlmOptions({ silent: true }).catch((err) => {
  ensureLlmFallbackOption();
  setStatus(`Failed to load LLM options: ${err.message}`);
});
