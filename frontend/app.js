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
  directoryOptions: document.getElementById('directoryOptions'),
  caseList: document.getElementById('caseList'),
  llmSelect: document.getElementById('llmSelect'),
  skipReassess: document.getElementById('skipReassess'),
  newCaseBtn: document.getElementById('newCaseBtn'),
  saveCaseBtn: document.getElementById('saveCaseBtn'),
  stopInferenceBtn: document.getElementById('stopInferenceBtn'),
  refreshCasesBtn: document.getElementById('refreshCasesBtn'),
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
let cases = [];
let selectedDepositionId = null;
let chatHistory = [];
let loadedCaseId = '';
let uiOpsInFlight = 0;
let llmOpsInFlight = 0;
const inferencingControllers = new Set();
const timerHandles = {
  llm: null,
  reasoning: null,
  chat: null,
};

function escapeHtml(value) {
  /** Escape dynamic text before inserting into innerHTML snippets. */
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function api(path, options = {}) {
  /** Perform JSON API calls and normalize non-2xx errors. */
  try {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(payload.detail || 'Request failed');
    }
    return response.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('Inference stopped by user.');
    }
    const message = err instanceof Error ? err.message : String(err || '');
    if (message.toLowerCase().includes('abort')) {
      throw new Error('Inference stopped by user.');
    }
    if (message.toLowerCase().includes('failed to fetch')) {
      throw new Error(
        'API is unreachable. Start/restart the stack with `docker compose up -d` and refresh the page.'
      );
    }
    throw err instanceof Error ? err : new Error('Request failed');
  }
}

function setStatus(message) {
  /** Render a short status message in the control panel. */
  els.status.textContent = message;
}

function syncCaseActionState() {
  /** Keep case-index actions aligned with current case id/name values. */
  const hasCaseId = !!els.caseId.value.trim();
  const selectedModel = els.llmSelect.selectedOptions[0];
  const selectedModelAvailable = !!selectedModel && !selectedModel.disabled;
  els.saveCaseBtn.disabled = uiOpsInFlight > 0 || !hasCaseId || !selectedModelAvailable;
  els.stopInferenceBtn.disabled = inferencingControllers.size === 0;
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
  els.newCaseBtn.disabled = disabled;
  els.ingestBtn.disabled = disabled;
  els.refreshBtn.disabled = disabled;
  els.saveCaseBtn.disabled = disabled || !els.caseId.value.trim();
  els.refreshCasesBtn.disabled = disabled;
  els.directory.disabled = disabled;
  els.refreshModelsBtn.disabled = disabled;
  els.llmSelect.disabled = disabled;
  els.skipReassess.disabled = disabled;
  els.stopInferenceBtn.disabled = inferencingControllers.size === 0;
}

function beginInferencingRequest() {
  /** Register one cancellable inferencing request and return its abort controller. */
  const controller = new AbortController();
  inferencingControllers.add(controller);
  syncCaseActionState();
  return controller;
}

function endInferencingRequest(controller) {
  /** Unregister one inferencing request controller. */
  inferencingControllers.delete(controller);
  syncCaseActionState();
}

async function inferencingApi(path, options = {}) {
  /** Execute API call with cancellation support for inferencing operations. */
  const controller = beginInferencingRequest();
  try {
    return await api(path, { ...options, signal: controller.signal });
  } finally {
    endInferencingRequest(controller);
  }
}

function stopInferencing() {
  /** Abort all active inferencing requests (ingest + analysis). */
  if (inferencingControllers.size === 0) {
    setStatus('No active inferencing request to stop.');
    return;
  }
  for (const controller of Array.from(inferencingControllers)) {
    controller.abort();
  }
  setReasoningProcessing(false);
  setChatProcessing(false);
  while (llmOpsInFlight > 0) {
    endLlmProcessing();
  }
  setStatus('Stop requested. Active inferencing calls were cancelled.');
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

function clearCurrentCaseView() {
  /** Clear loaded deposition/chat/detail state for starting a new blank case. */
  depositions = [];
  selectedDepositionId = null;
  chatHistory = [];
  loadedCaseId = '';
  els.chatMessages.innerHTML = '';
  els.focusedReasoningBody.textContent = '';
  els.chatInput.value = '';
  renderTimeline();
  renderDepositions();
  renderDetail(null);
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

function timelineVisibleSlots(itemCount) {
  /** Return timeline cards visible at once (desktop=5, mobile=2). */
  const maxVisible = window.matchMedia('(max-width: 760px)').matches ? 2 : 5;
  return Math.max(1, Math.min(maxVisible, itemCount));
}

function updateTimelineSlots() {
  /** Apply visible-slot count CSS variable for the current timeline item count. */
  if (!depositions.length) {
    els.timeline.style.removeProperty('--timeline-node-slots');
    return;
  }
  els.timeline.style.setProperty('--timeline-node-slots', String(timelineVisibleSlots(depositions.length)));
}

function timelineStepPixels() {
  /** Return one page-step width for timeline nav buttons. */
  const firstItem = els.timeline.querySelector('.timeline-item');
  if (!firstItem) {
    return 0;
  }
  const itemWidth = firstItem.getBoundingClientRect().width;
  const timelineStyle = window.getComputedStyle(els.timeline);
  const gap = Number.parseFloat(timelineStyle.columnGap || timelineStyle.gap || '0') || 0;
  const slots = timelineVisibleSlots(depositions.length);
  return Math.max(itemWidth + gap, (itemWidth + gap) * slots);
}

function syncTimelineNavButtons() {
  /** Disable timeline nav buttons at edges and when no overflow exists. */
  const maxScrollLeft = Math.max(0, els.timeline.scrollWidth - els.timeline.clientWidth);
  if (maxScrollLeft <= 1) {
    els.timelineBack.disabled = true;
    els.timelineForward.disabled = true;
    return;
  }
  els.timelineBack.disabled = els.timeline.scrollLeft <= 1;
  els.timelineForward.disabled = els.timeline.scrollLeft >= maxScrollLeft - 1;
}

function scrollTimeline(direction) {
  /** Scroll timeline by one visible page in requested direction (-1 or +1). */
  const step = timelineStepPixels();
  if (step <= 0) {
    return;
  }
  const maxScrollLeft = Math.max(0, els.timeline.scrollWidth - els.timeline.clientWidth);
  const delta = direction < 0 ? -step : step;
  const target = Math.min(maxScrollLeft, Math.max(0, els.timeline.scrollLeft + delta));
  els.timeline.scrollTo({ left: target, behavior: 'smooth' });
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
    syncCaseActionState();
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
  syncCaseActionState();
}

function renderDirectoryDropdown(payload) {
  /** Render server-discovered ingestion folder suggestions for free-text input. */
  const options = payload.options || [];
  const current = els.directory.value.trim();
  els.directoryOptions.innerHTML = '';

  const availablePaths = new Set();
  for (const item of options) {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = item.label || `${item.path} (${item.file_count || 0} files)`;
    els.directoryOptions.appendChild(option);
    availablePaths.add(item.path);
  }

  if (!current) {
    const fallback = availablePaths.has(payload.suggested)
      ? payload.suggested
      : options[0]?.path || '';
    els.directory.value = fallback;
  }
  syncCaseActionState();
  return options.length > 0;
}

async function loadDirectoryOptions({ silent = false } = {}) {
  /** Fetch valid ingestion folders and populate the directory dropdown. */
  if (!silent) {
    startUiProcessing('Loading deposition folders...');
  }
  try {
    const payload = await api('/api/deposition-directories');
    const hasOptions = renderDirectoryDropdown(payload);
    const base = (payload.base_directory || '').trim();
    if (!silent) {
      if (hasOptions) {
        setStatus(
          base
            ? `Deposition base: ${base}. Selected: ${els.directory.value.trim()}.`
            : `Deposition folders loaded. Selected: ${els.directory.value.trim()}.`
        );
      } else {
        setStatus(
          base
            ? `No .txt folders found under base: ${base}.`
            : 'No ingestion folders available yet. Add .txt files and reload.'
        );
      }
    }
  } finally {
    if (!silent) {
      endUiProcessing();
    }
  }
}

function getCaseUpdatedLabel(item) {
  /** Convert a case summary updated timestamp into a short UI label. */
  if (!item.updated_at) {
    return 'No updates';
  }
  const parsed = Date.parse(item.updated_at);
  if (Number.isNaN(parsed)) {
    return item.updated_at;
  }
  return new Date(parsed).toLocaleString();
}

function normalizeCaseDirectory(caseSummary) {
  /** Return saved folder path for a case summary, or empty string when unavailable. */
  return String(caseSummary?.last_directory || '').trim();
}

function applyCaseSelection(caseSummary) {
  /** Apply selected case id and saved folder path into form controls. */
  if (!caseSummary) {
    els.caseId.value = '';
    els.directory.value = '';
    syncCaseActionState();
    return;
  }
  els.caseId.value = caseSummary.case_id || '';
  els.directory.value = normalizeCaseDirectory(caseSummary);
  syncCaseActionState();
}

function renderCaseIndex() {
  /** Render vertical case index list and highlight the active case id. */
  const activeCaseId = els.caseId.value.trim();
  els.caseList.innerHTML = '';
  syncCaseActionState();

  if (!cases.length) {
    els.caseList.innerHTML = '<div class="muted">No saved cases yet.</div>';
    return;
  }

  for (const item of cases) {
    const row = document.createElement('div');
    row.className = 'case-item-row';

    const loadButton = document.createElement('button');
    loadButton.type = 'button';
    loadButton.className = `case-item ${item.case_id === activeCaseId ? 'active' : ''}`;
    loadButton.innerHTML = `
      <strong>${escapeHtml(item.case_id || '')}</strong>
      <small>Depositions: ${item.deposition_count || 0}</small>
      <small>LangGraph memory: ${item.memory_entries || 0}</small>
      <small>Updated: ${escapeHtml(getCaseUpdatedLabel(item))}</small>
    `;
    loadButton.addEventListener('click', () => {
      applyCaseSelection(item);
      renderCaseIndex();
      loadDepositions().catch((err) => setStatus(err.message));
    });

    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'secondary danger case-item-delete';
    deleteButton.textContent = 'Delete';
    deleteButton.title = `Delete case ${item.case_id}`;
    deleteButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      deleteCaseById(item.case_id).catch((err) => setStatus(err.message));
    });

    row.appendChild(loadButton);
    row.appendChild(deleteButton);
    els.caseList.appendChild(row);
  }
}

async function loadCases({ silent = false } = {}) {
  /** Load all cases from CouchDB-backed API and refresh the vertical index. */
  if (!silent) {
    startUiProcessing('Loading saved cases...');
  }
  try {
    const payload = await api('/api/cases');
    // Case Index should only show persisted/saved cases, not folder placeholders.
    cases = payload.cases || [];

    const previousCaseId = els.caseId.value.trim();
    const activeCaseId = previousCaseId;
    let nextCaseSummary = cases.find((item) => item.case_id === activeCaseId) || null;
    if (!nextCaseSummary && cases.length) {
      nextCaseSummary = cases[0];
    }

    if (!nextCaseSummary) {
      applyCaseSelection(null);
    } else if (nextCaseSummary.case_id !== previousCaseId) {
      applyCaseSelection(nextCaseSummary);
    }
    renderCaseIndex();
    if (!silent) {
      setStatus(`Loaded ${cases.length} saved cases.`);
    }
  } finally {
    if (!silent) {
      endUiProcessing();
    }
  }
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
    updateTimelineSlots();
    syncTimelineNavButtons();
    els.timeline.innerHTML = '<div class="muted">Timeline will appear after loading depositions.</div>';
    return;
  }

  const ordered = [...depositions].sort(timelineSort);
  updateTimelineSlots();
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
  syncTimelineNavButtons();
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
  startLlmProcessing('Persona:Attorney is processing the selected contradiction...');
  setReasoningProcessing(true, 'Re-analyzing this detail item...');
  await nextPaint();

  try {
    const payload = await inferencingApi('/api/reason-contradiction', {
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
    await loadCases({ silent: true });
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
  renderCaseIndex();
  syncCaseActionState();
  if (!caseId) {
    setStatus('Enter a case ID first.');
    return;
  }

  startUiProcessing(`Loading case ${caseId}...`);
  try {
    depositions = await api(`/api/depositions/${encodeURIComponent(caseId)}`);
    loadedCaseId = caseId;

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

async function refreshCase() {
  /** Clear all current deposition docs for the active case id. */
  const caseId = els.caseId.value.trim();
  if (!caseId) {
    setStatus('Enter a case ID first.');
    return;
  }

  startUiProcessing(`Refreshing case ${caseId} (clearing prior depositions)...`);
  try {
    const payload = await api(`/api/cases/${encodeURIComponent(caseId)}/depositions`, {
      method: 'DELETE',
    });
    clearCurrentCaseView();

    await loadCases({ silent: true });

    setStatus(`Refresh complete. Removed ${payload.deleted_depositions} depositions from ${caseId}.`);
  } finally {
    endUiProcessing();
  }
}

function createBlankCase() {
  /** Reset form + panes so user can create a brand-new empty case. */
  els.caseId.value = '';
  els.directory.value = '';
  els.skipReassess.checked = false;
  clearCurrentCaseView();
  renderCaseIndex();
  syncCaseActionState();
  setStatus('Blank case ready. Enter Case ID and select a deposition folder.');
}

async function saveCase() {
  /** Persist case metadata so it appears in the Case Index list. */
  const caseId = els.caseId.value.trim();
  const directory = els.directory.value.trim();
  const llm = getSelectedLlm();
  if (!caseId) {
    setStatus('Enter a Case ID before saving.');
    return;
  }
  if (!directory) {
    setStatus('Select a deposition folder before saving.');
    return;
  }
  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Choose an operational model before saving.');
    return;
  }

  startUiProcessing(`Saving case ${caseId}...`);
  try {
    await api('/api/cases', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        directory,
        llm_provider: llm.provider,
        llm_model: llm.model,
      }),
    });
    loadedCaseId = caseId;
    syncCaseActionState();
    await loadCases({ silent: true });
    renderCaseIndex();
    setStatus(`Saved case '${caseId}'.`);
  } finally {
    endUiProcessing();
  }
}

async function deleteCaseById(caseId) {
  /** Delete one case id and refresh dependent UI state and data. */
  const targetCaseId = String(caseId || '').trim();
  if (!targetCaseId) {
    setStatus('Select a case ID to delete.');
    return;
  }
  if (!window.confirm(`Delete case '${targetCaseId}' and all related records?`)) {
    return;
  }

  startUiProcessing(`Deleting case ${targetCaseId}...`);
  try {
    const payload = await api(`/api/cases/${encodeURIComponent(targetCaseId)}`, { method: 'DELETE' });
    depositions = [];
    selectedDepositionId = null;
    chatHistory = [];
    els.chatMessages.innerHTML = '';
    renderTimeline();
    renderDepositions();
    renderDetail(null);

    await loadCases({ silent: true });
    const fallbackSummary = cases[0] || null;
    const fallbackCase = fallbackSummary?.case_id || '';
    applyCaseSelection(fallbackSummary);
    renderCaseIndex();
    if (fallbackCase) {
      await loadDepositions();
    }
    setStatus(`Deleted ${payload.deleted_docs} documents for case ${targetCaseId}.`);
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
  startLlmProcessing('Persona:Legal Clerk is processing depositions...');
  await nextPaint();
  try {
    await inferencingApi('/api/ingest-case', {
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
  await loadCases({ silent: true });
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
  startLlmProcessing('Persona:Attorney is processing your question...');
  setChatProcessing(true);
  await nextPaint();
  try {
    const payload = await inferencingApi('/api/chat', {
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
    await loadCases({ silent: true });
  } finally {
    setChatProcessing(false);
    endLlmProcessing();
  }
}

els.newCaseBtn.addEventListener('click', () => createBlankCase());
els.ingestBtn.addEventListener('click', () => ingestCase().catch((err) => setStatus(err.message)));
els.refreshBtn.addEventListener('click', () => refreshCase().catch((err) => setStatus(err.message)));
els.saveCaseBtn.addEventListener('click', () => saveCase().catch((err) => setStatus(err.message)));
els.refreshCasesBtn.addEventListener('click', () => loadCases().catch((err) => setStatus(err.message)));
els.refreshModelsBtn.addEventListener('click', () =>
  loadLlmOptions({ forceProbe: true }).catch((err) => setStatus(err.message))
);
els.stopInferenceBtn.addEventListener('click', () => stopInferencing());
els.caseId.addEventListener('change', () => {
  syncCaseActionState();
  renderCaseIndex();
});
els.directory.addEventListener('change', () => {
  const selected = els.directory.value.trim();
  if (!selected) {
    setStatus('Select a deposition folder.');
    return;
  }
  setStatus(`Deposition folder selected: ${selected}`);
});
els.llmSelect.addEventListener('change', () => {
  const selected = els.llmSelect.selectedOptions[0];
  if (selected && selected.disabled) {
    setStatus('Selected model is unavailable. Choose an operational model or click Refresh Models.');
    return;
  }
  syncCaseActionState();
  setStatus(`LLM selected: ${getSelectedLlmLabel()}.`);
});
els.timelineBack.addEventListener('click', () => scrollTimeline(-1));
els.timelineForward.addEventListener('click', () => scrollTimeline(1));
els.timeline.addEventListener('scroll', () => syncTimelineNavButtons());
els.chatForm.addEventListener('submit', (event) => sendChat(event).catch((err) => setStatus(err.message)));
window.addEventListener('resize', () => {
  updateTimelineSlots();
  syncTimelineNavButtons();
});

renderDetail(null);
renderTimeline();
renderDepositions();
loadedCaseId = els.caseId.value.trim();
syncCaseActionState();
loadLlmOptions({ silent: true }).catch((err) => {
  ensureLlmFallbackOption();
  setStatus(`Failed to load LLM options: ${err.message}`);
});
loadDirectoryOptions({ silent: true })
  .then(async () => {
    await loadCases({ silent: true });
  })
  .catch((err) => {
    const message = err instanceof Error ? err.message : String(err || 'Unknown error');
    if (message.toLowerCase().includes('folder')) {
      setStatus(`Failed to load deposition folders: ${message}`);
      return;
    }
    setStatus(`Failed to load case index: ${message}`);
  });
