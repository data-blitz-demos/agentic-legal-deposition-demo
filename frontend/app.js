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
  traceStreamToggle: document.getElementById('traceStreamToggle'),
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
  traceStreamPanel: document.getElementById('traceStreamPanel'),
  conflictDetailPanel: document.getElementById('conflictDetailPanel'),
  traceMeta: document.getElementById('traceMeta'),
  traceLive: document.getElementById('traceLive'),
  traceOlderBtn: document.getElementById('traceOlderBtn'),
  traceNewerBtn: document.getElementById('traceNewerBtn'),
  traceWindowMeta: document.getElementById('traceWindowMeta'),
  metricsTextToggle: document.getElementById('metricsTextToggle'),
  metricsBody: document.getElementById('metricsBody'),
  refreshMetricsBtn: document.getElementById('refreshMetricsBtn'),
  metricsSampleMeta: document.getElementById('metricsSampleMeta'),
  metricsStorageMeta: document.getElementById('metricsStorageMeta'),
  metricsGrid: document.getElementById('metricsGrid'),
  correctnessGrid: document.getElementById('correctnessGrid'),
  metricDetailPanel: document.getElementById('metricDetailPanel'),
  metricDetailTitle: document.getElementById('metricDetailTitle'),
  metricDetailBody: document.getElementById('metricDetailBody'),
  ontologyPath: document.getElementById('ontologyPath'),
  ontologyOptions: document.getElementById('ontologyOptions'),
  ontologyBrowseBtn: document.getElementById('ontologyBrowseBtn'),
  loadOntologyBtn: document.getElementById('loadOntologyBtn'),
  openGraphBrowserBtn: document.getElementById('openGraphBrowserBtn'),
  graphRagQuestion: document.getElementById('graphRagQuestion'),
  graphRagToggle: document.getElementById('graphRagToggle'),
  graphRagStreamToggle: document.getElementById('graphRagStreamToggle'),
  graphRagAskBtn: document.getElementById('graphRagAskBtn'),
  graphRagClearBtn: document.getElementById('graphRagClearBtn'),
  graphRagAnswer: document.getElementById('graphRagAnswer'),
  graphRagMonitor: document.getElementById('graphRagMonitor'),
  ontologyBrowserModal: document.getElementById('ontologyBrowserModal'),
  ontologyBrowserTitle: document.getElementById('ontologyBrowserTitle'),
  ontologyBrowserPath: document.getElementById('ontologyBrowserPath'),
  ontologyBrowserList: document.getElementById('ontologyBrowserList'),
  ontologyBrowserCloseBtn: document.getElementById('ontologyBrowserCloseBtn'),
  ontologyBrowserUpBtn: document.getElementById('ontologyBrowserUpBtn'),
  ontologyBrowserUseFolderBtn: document.getElementById('ontologyBrowserUseFolderBtn'),
  ontologyBrowserRefreshBtn: document.getElementById('ontologyBrowserRefreshBtn'),
};

let depositions = [];
let cases = [];
let selectedDepositionId = null;
let chatHistory = [];
let loadedCaseId = '';
let uiOpsInFlight = 0;
let llmOpsInFlight = 0;
const inferencingControllers = new Set();
let activeTraceId = '';
let activeTraceChannel = 'ingest';
let tracePollHandle = null;
let lastTraceSnapshot = null;
let traceStreamEnabled = false;
let thoughtStreamStorageReady = false;
let traceWindowStart = 0;
const TRACE_WINDOW_SIZE = 6;
let traceWindowPinnedToLatest = true;
let metricsPollHandle = null;
const METRICS_POLL_MS = 15000;
let metricsPanelOpen = false;
let metricsLoaded = false;
let ontologyBrowserCurrentDirectory = '';
let ontologyBrowserParentDirectory = '';
let ontologyBrowserWildcardPath = '';
let graphRagCycles = [];
const RUNTIME_METRIC_DETAILS = {
  task_success_rate_pct:
    'Measures how often full runs end in completed status instead of failed status. A sustained drop usually indicates provider outages, schema drift, or parser instability in one workflow step.',
  run_failure_rate_pct:
    'Tracks failure share across finished runs. Rising failure rate is an early warning for reliability regressions, prompt/schema issues, or new edge-case inputs entering your workload.',
  p95_end_to_end_latency_sec:
    'Captures tail latency of the slowest 5% of runs. It is the best signal for user-perceived slowness and should be watched during model swaps, prompt expansions, and high-load periods.',
  p95_time_to_first_event_sec:
    'Shows startup responsiveness from run creation to first trace event. Spikes usually point to cold model loads, routing delays, or stalled upstream dependencies.',
  avg_steps_per_finished_run:
    'Represents average reasoning/event depth per finished run. Higher values can reflect heavier work, but sudden jumps often signal redundant loops or prompt over-expansion.',
  loop_risk_rate_pct:
    'Share of runs with unusually high step counts (20+). This is a direct control for runaway reasoning loops and token-cost blowups.',
  in_flight_runs:
    'Count of runs still in running state within the lookback window. Persistently high values can indicate queue pressure, stuck workers, or backend bottlenecks.',
  finished_runs_per_hour:
    'Operational throughput of finished runs over time. Use with latency/failure metrics to distinguish healthy scale-up from overloaded degradation.',
  rag_toggle_comparison_pairs:
    'Number of same-question A/B comparisons where the system has both RAG ON and RAG OFF completed answers in the lookback window.',
  rag_answer_change_rate_pct:
    'How often answer text changed between paired RAG ON and OFF runs. This quantifies direct retrieval influence on final responses.',
  rag_context_hit_rate_pct:
    'For RAG ON completed graph queries, percentage where retrieval returned at least one context row. Low values indicate sparse graph grounding.',
  rag_avg_context_rows_on:
    'Average number of retrieved graph context rows per completed query when RAG is enabled. Helps track retrieval depth trends.',
  rag_avg_answer_word_delta_on_minus_off:
    'Average answer length difference for paired queries computed as (RAG ON words - RAG OFF words). Positive values suggest retrieval-expanded responses.',
  rag_completed_queries_split:
    'Completed Graph RAG query count split shown as ON/OFF. Use to validate whether enough A/B traffic exists for reliable influence metrics.',
};
const CORRECTNESS_DRIFT_OBSERVABLES = [
  {
    label: 'Golden Set Accuracy',
    display: 'Track',
    target: '>= 95%',
    formula: 'correct / total over fixed eval set',
    description: 'Primary correctness monitor over a stable benchmark set.',
    detail:
      'Use a fixed, versioned benchmark set with known expected outputs. Re-baseline only after deliberate model or rubric changes, otherwise drops are true quality regressions.',
  },
  {
    label: 'Schema Adherence Rate',
    display: 'Track',
    target: '>= 99%',
    formula: 'valid_structured_outputs / total_structured_outputs',
    description: 'Catches parser/format regressions before they hit production workflows.',
    detail:
      'Track by endpoint and by model version. A drop here predicts downstream ingest failures before users see them.',
  },
  {
    label: 'Unsupported Claim Rate',
    display: 'Track',
    target: '<= 2%',
    formula: 'unsupported_claims / total_claims',
    description: 'Groundedness signal for hallucination-style drift.',
    detail:
      'Compute on audited samples where each claim is verified against source evidence. Rising values indicate model drift toward speculation or weak retrieval grounding.',
  },
  {
    label: 'Repeat Prompt Inconsistency',
    display: 'Track',
    target: '<= 10%',
    formula: 'inconsistent_repeats / repeat_prompt_groups',
    description: 'Compares outputs for near-identical prompts to detect stability drift.',
    detail:
      'Run repeated prompt cohorts under fixed settings (temperature, tools, context). Increasing inconsistency means behavior is less predictable and harder to trust operationally.',
  },
  {
    label: 'Model Mix Drift (JSD)',
    display: 'Track',
    target: '<= 0.12',
    formula: 'Jensen-Shannon divergence(recent_model_mix, baseline_model_mix)',
    description: 'Detects unexpected provider/model routing changes over time.',
    detail:
      'Watch this after routing-rule changes and incident failovers. Large divergence can explain abrupt behavior shifts even when prompts stayed constant.',
  },
  {
    label: 'Judge-Human Disagreement',
    display: 'Track',
    target: '<= 5%',
    formula: 'judge_human_disagreements / adjudicated_samples',
    description: 'Validates whether auto-eval quality still aligns with human review.',
    detail:
      'Use this as guardrail for automated evaluations. If disagreement rises, retrain judge prompts/rubrics and increase human spot checks before trusting score trends.',
  },
];
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

function clearMetricSelection() {
  /** Clear selected state from all metric cards in runtime + correctness grids. */
  document.querySelectorAll('.metric-card.selected').forEach((item) => item.classList.remove('selected'));
}

function showMetricDetail(metric, source) {
  /** Render expanded selected-metric details in the metrics detail panel. */
  if (!els.metricDetailPanel || !els.metricDetailTitle || !els.metricDetailBody) {
    return;
  }
  const label = String(metric?.label || 'Metric').trim() || 'Metric';
  const key = String(metric?.key || '').trim();
  const formula = String(metric?.formula || '').trim();
  const target = String(metric?.target || '').trim();
  const detailFromCatalog =
    source === 'runtime' && key ? String(RUNTIME_METRIC_DETAILS[key] || '').trim() : '';
  const explicitDetail = String(metric?.detail || '').trim();
  const baseDescription = String(metric?.description || '').trim();
  const detail = detailFromCatalog || explicitDetail || baseDescription || 'No additional details available yet.';
  const relation = describeMetricValueImpact(metric, source);

  const lines = [detail];
  if (relation) {
    lines.push(relation);
  }
  if (target) {
    lines.push(`Target: ${target}.`);
  }
  if (formula) {
    lines.push(`Formula: ${formula}.`);
  }

  els.metricDetailTitle.textContent = `${label} Detail`;
  els.metricDetailBody.textContent = lines.join(' ');
  els.metricDetailPanel.classList.remove('hidden');
}

function describeMetricValueImpact(metric, source) {
  /** Explain how a metric's current value relates to present system health/behavior. */
  const display = String(metric?.display || '').trim();
  const target = String(metric?.target || '').trim();
  const status = String(metric?.status || 'info').toLowerCase();

  if (source === 'correctness' && (!display || display.toLowerCase() === 'track')) {
    return 'Current live value is not shown here yet. Treat this as a system-level observable that should be instrumented and tracked over time for drift/correctness.';
  }
  if (!display || display === 'N/A') {
    return 'No reliable value is available in this lookback window, so this metric cannot currently characterize system health.';
  }

  const statusPhrase = {
    good: 'is in a healthy range for the system',
    warn: 'is outside the ideal range and signals elevated system risk',
    bad: 'is in an unhealthy range and indicates active system degradation',
    info: 'is informational and should be interpreted with related runtime metrics',
  }[status] || 'should be interpreted in context with the full metric set';

  if (target) {
    return `Current value (${display}) ${statusPhrase} relative to target ${target}.`;
  }
  return `Current value (${display}) ${statusPhrase}.`;
}

function resetMetricDetail() {
  /** Hide expanded metric-detail panel and clear previous selection copy. */
  if (!els.metricDetailPanel || !els.metricDetailTitle || !els.metricDetailBody) {
    return;
  }
  els.metricDetailTitle.textContent = 'Metric Detail';
  els.metricDetailBody.textContent = '';
  els.metricDetailPanel.classList.add('hidden');
}

function renderMetricCards(container, metrics, source) {
  /** Render generic metric cards into a target grid container. */
  if (!container) {
    return;
  }
  if (!metrics.length) {
    container.innerHTML = '<div class="muted">No metrics available yet.</div>';
    return;
  }
  container.innerHTML = metrics
    .map((metric) => {
      const statusClass = ['good', 'warn', 'bad', 'info'].includes(metric.status) ? metric.status : 'info';
      const formula = String(metric.formula || '').trim();
      return `
        <article class="metric-card ${statusClass}" role="button" tabindex="0">
          <div class="metric-label">${escapeHtml(metric.label || '')}</div>
          <div class="metric-value">${escapeHtml(metric.display || '')}</div>
          <div class="metric-target">Target: ${escapeHtml(metric.target || '')}</div>
          ${formula ? `<div class="metric-formula">Formula: ${escapeHtml(formula)}</div>` : ''}
          <div class="metric-desc">${escapeHtml(metric.description || '')}</div>
        </article>
      `;
    })
    .join('');

  const cards = Array.from(container.querySelectorAll('.metric-card'));
  cards.forEach((card, index) => {
    const metric = metrics[index];
    if (!metric) {
      return;
    }
    const select = () => {
      clearMetricSelection();
      card.classList.add('selected');
      showMetricDetail(metric, source);
    };
    card.addEventListener('click', select);
    card.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    });
  });
}

function renderCorrectnessDriftObservables() {
  /** Render static correctness/drift KPI definitions with formulas and thresholds. */
  renderMetricCards(
    els.correctnessGrid,
    CORRECTNESS_DRIFT_OBSERVABLES.map((item) => ({ ...item, status: 'info' })),
    'correctness'
  );
}

function renderAgentMetrics(payload) {
  /** Render runtime KPI cards and metadata into the Agent Runtime Metrics panel. */
  const metrics = Array.isArray(payload?.metrics) ? payload.metrics : [];
  const sampledRuns = Number(payload?.sampled_runs || 0);
  const ragSampledQueries = Number(payload?.rag_sampled_queries || 0);
  const ragPairedComparisons = Number(payload?.rag_paired_comparisons || 0);
  const lookbackHours = Number(payload?.lookback_hours || 24);
  const generatedAtRaw = String(payload?.generated_at || '').trim();
  const generatedAt = generatedAtRaw ? new Date(generatedAtRaw) : null;
  const generatedLabel =
    generatedAt && !Number.isNaN(generatedAt.getTime()) ? generatedAt.toLocaleTimeString() : 'N/A';

  els.metricsSampleMeta.textContent = `Sample: ${sampledRuns} runs + ${ragSampledQueries} graph queries (${lookbackHours}h), pairs ${ragPairedComparisons}, updated ${generatedLabel}`;
  const thoughtStorageText = payload?.storage_connected
    ? 'Thought Stream DB: connected'
    : 'Thought Stream DB: degraded (using in-memory sessions)';
  const ragStorageText = payload?.rag_storage_connected
    ? 'RAG Stream DB: connected'
    : 'RAG Stream DB: degraded (using runtime payload only)';
  els.metricsStorageMeta.textContent = `${thoughtStorageText} | ${ragStorageText}`;

  renderMetricCards(els.metricsGrid, metrics, 'runtime');
}

async function loadAgentMetrics({ silent = false } = {}) {
  /** Fetch runtime KPI payload from backend and refresh dashboard cards. */
  const payload = await api('/api/agent-metrics?lookback_hours=24');
  renderAgentMetrics(payload);
  metricsLoaded = true;
  if (!silent) {
    setStatus('Agent observables refreshed.');
  }
}

function stopMetricsPolling() {
  /** Stop periodic runtime metrics polling loop. */
  if (metricsPollHandle !== null) {
    window.clearInterval(metricsPollHandle);
    metricsPollHandle = null;
  }
}

function startMetricsPolling() {
  /** Start periodic runtime metrics polling loop. */
  stopMetricsPolling();
  metricsPollHandle = window.setInterval(() => {
    loadAgentMetrics({ silent: true }).catch(() => {});
  }, METRICS_POLL_MS);
}

function setMetricsPanelOpen(open) {
  /** Toggle runtime metrics body visibility from text-trigger click state. */
  metricsPanelOpen = !!open;
  els.metricsBody.classList.toggle('hidden', !metricsPanelOpen);
  els.metricsTextToggle.setAttribute('aria-expanded', metricsPanelOpen ? 'true' : 'false');
  if (metricsPanelOpen) {
    startMetricsPolling();
    return;
  }
  clearMetricSelection();
  resetMetricDetail();
  stopMetricsPolling();
}

async function toggleMetricsPanel() {
  /** Expand/collapse metrics body; load dashboard metrics on first expansion. */
  const next = !metricsPanelOpen;
  setMetricsPanelOpen(next);
  if (next && !metricsLoaded) {
    renderCorrectnessDriftObservables();
    await loadAgentMetrics({ silent: true });
    setStatus('Agent observables loaded.');
  }
  if (next && metricsLoaded) {
    renderCorrectnessDriftObservables();
  }
}

function createTraceId() {
  /** Create client-side thought-stream id for backend stream correlation. */
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  return `thought-stream-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function syncTraceActionState() {
  /** Keep thought-stream navigation controls aligned with toggle state. */
  els.traceOlderBtn.disabled = !traceStreamEnabled;
  els.traceNewerBtn.disabled = !traceStreamEnabled;
}

function setTraceStreamEnabled(enabled) {
  /** Toggle trace stream panel and reset stream state when disabled. */
  traceStreamEnabled = !!enabled;
  els.traceStreamPanel.classList.toggle('hidden', !traceStreamEnabled);
  els.conflictDetailPanel.classList.toggle('trace-on', traceStreamEnabled);
  if (!traceStreamEnabled) {
    stopTracePolling();
    els.traceMeta.textContent = activeTraceId
      ? 'Thought stream viewer is off (capture continues).'
      : 'Thought stream is off.';
    els.traceWindowMeta.textContent = 'Window: 0-0';
    els.traceLive.value = 'Thought stream is off. Enable Thought Stream to view live events.';
  } else if (activeTraceId) {
    startTracePolling();
  } else if (lastTraceSnapshot) {
    renderTraceSnapshot(lastTraceSnapshot);
  } else {
    els.traceMeta.textContent = 'No active thought-stream session.';
    els.traceWindowMeta.textContent = 'Window: 0-0';
    els.traceLive.value = 'Thought stream enabled. Run ingest or chat to start streaming.';
  }
  syncTraceActionState();
}

async function verifyThoughtStreamStorage() {
  /** Ensure Thought Stream CouchDB plumbing is reachable before live streaming UI is enabled. */
  const payload = await api('/api/thought-streams/health');
  if (!payload || payload.connected !== true) {
    throw new Error('Thought Stream storage is not connected.');
  }
  thoughtStreamStorageReady = true;
  return payload;
}

async function handleTraceStreamToggle() {
  /** Validate Thought Stream storage connectivity before turning live viewer on. */
  if (!els.traceStreamToggle.checked) {
    thoughtStreamStorageReady = false;
    setTraceStreamEnabled(false);
    return;
  }
  try {
    const payload = await verifyThoughtStreamStorage();
    setTraceStreamEnabled(true);
    setStatus(`Thought Stream connected to CouchDB database '${payload.database}'.`);
  } catch (err) {
    thoughtStreamStorageReady = false;
    els.traceStreamToggle.checked = false;
    setTraceStreamEnabled(false);
    throw err;
  }
}

function formatTraceEvents(events = []) {
  /** Convert thought-stream events into readable plain text. */
  if (!events.length) {
    return 'No thought-stream events available yet.';
  }
  return events
    .map((event, index) => {
      const lines = [
        `${index + 1}. ${event.persona || 'Agent'} :: ${event.phase || 'step'}`,
        `Provider/Model: ${event.llm_provider || '-'} / ${event.llm_model || '-'}`,
      ];
      if (event.file_name) {
        lines.push(`File: ${event.file_name}`);
      }
      if (event.notes) {
        lines.push(`Notes: ${event.notes}`);
      }
      if (event.input_preview) {
        lines.push(`Input Preview:\n${event.input_preview}`);
      }
      if (event.system_prompt) {
        lines.push(`System Prompt:\n${event.system_prompt}`);
      }
      if (event.user_prompt) {
        lines.push(`User Prompt:\n${event.user_prompt}`);
      }
      if (event.output_preview) {
        lines.push(`Output Preview:\n${event.output_preview}`);
      }
      return lines.join('\n');
    })
    .join('\n\n------------------------------\n\n');
}

function flattenTraceEvents(snapshot) {
  /** Combine legal-clerk and attorney traces into one ordered stream. */
  const trace = snapshot?.thought_stream || snapshot?.trace || {};
  const legalClerk = Array.isArray(trace.legal_clerk) ? trace.legal_clerk : [];
  const attorney = Array.isArray(trace.attorney) ? trace.attorney : [];
  const combined = [...legalClerk, ...attorney];
  combined.sort((a, b) => {
    const seqA = Number(a.sequence || 0);
    const seqB = Number(b.sequence || 0);
    if (seqA && seqB) {
      return seqA - seqB;
    }
    const atA = Date.parse(String(a.at || ''));
    const atB = Date.parse(String(b.at || ''));
    if (!Number.isNaN(atA) && !Number.isNaN(atB) && atA !== atB) {
      return atA - atB;
    }
    return 0;
  });
  return combined;
}

function applyTraceWindow(events, status) {
  /** Render sliding window over flattened trace events. */
  const total = events.length;
  const maxStart = Math.max(0, total - TRACE_WINDOW_SIZE);
  if (traceWindowPinnedToLatest || status !== 'running') {
    traceWindowStart = maxStart;
  } else {
    traceWindowStart = Math.min(traceWindowStart, maxStart);
  }

  const start = Math.max(0, traceWindowStart);
  const end = Math.min(total, start + TRACE_WINDOW_SIZE);
  const windowItems = events.slice(start, end);
  els.traceLive.value = formatTraceEvents(windowItems);
  els.traceWindowMeta.textContent = `Window: ${total ? start + 1 : 0}-${end} of ${total}`;
  els.traceOlderBtn.disabled = !traceStreamEnabled || start <= 0;
  els.traceNewerBtn.disabled = !traceStreamEnabled || end >= total;
}

function shiftTraceWindow(direction) {
  /** Move thought-stream sliding window older/newer and re-render current snapshot. */
  if (!traceStreamEnabled || !lastTraceSnapshot) {
    return;
  }
  const events = flattenTraceEvents(lastTraceSnapshot);
  if (!events.length) {
    return;
  }
  const delta = direction < 0 ? -TRACE_WINDOW_SIZE : TRACE_WINDOW_SIZE;
  const maxStart = Math.max(0, events.length - TRACE_WINDOW_SIZE);
  traceWindowPinnedToLatest = false;
  traceWindowStart = Math.min(maxStart, Math.max(0, traceWindowStart + delta));
  if (traceWindowStart >= maxStart) {
    traceWindowPinnedToLatest = true;
  }
  applyTraceWindow(events, lastTraceSnapshot.status || 'running');
}

function renderTraceSnapshot(snapshot) {
  /** Render one thought-stream snapshot into sliding window viewer. */
  if (!traceStreamEnabled) {
    return;
  }
  lastTraceSnapshot = snapshot || null;
  const events = flattenTraceEvents(snapshot);
  applyTraceWindow(events, snapshot?.status || 'running');
  const channelLabel = activeTraceChannel === 'chat' ? 'Attorney chat' : 'Ingest';
  els.traceMeta.textContent = `${channelLabel} thought stream ${snapshot?.status || 'running'} (${snapshot?.thought_stream_id || activeTraceId})`;
}

async function pollTraceOnce() {
  /** Fetch and render latest thought-stream session state. */
  if (!activeTraceId) {
    return;
  }
  if (!traceStreamEnabled) {
    return;
  }
  try {
    const payload = await api(`/api/thought-streams/${encodeURIComponent(activeTraceId)}`);
    renderTraceSnapshot(payload);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Unknown thought-stream poll error');
    els.traceMeta.textContent = `Thought stream unavailable: ${message}`;
  }
}

function stopTracePolling() {
  /** Stop live trace polling loop. */
  if (tracePollHandle !== null) {
    window.clearInterval(tracePollHandle);
    tracePollHandle = null;
  }
}

function startTracePolling() {
  /** Start periodic polling for active thought-stream session updates. */
  if (!traceStreamEnabled) {
    return;
  }
  stopTracePolling();
  pollTraceOnce().catch(() => {});
  tracePollHandle = window.setInterval(() => {
    pollTraceOnce().catch(() => {});
  }, 700);
}

function beginTraceSession(channel) {
  /** Initialize live thought-stream UI and state for one operation. */
  activeTraceId = createTraceId();
  activeTraceChannel = channel === 'chat' ? 'chat' : 'ingest';
  lastTraceSnapshot = null;
  traceWindowStart = 0;
  traceWindowPinnedToLatest = true;
  els.traceLive.value = `Starting ${activeTraceChannel} thought stream...\nWaiting for events...`;
  els.traceWindowMeta.textContent = 'Window: 0-0';
  els.traceMeta.textContent = `Starting ${activeTraceChannel} thought stream (${activeTraceId})`;
  syncTraceActionState();
  if (traceStreamEnabled && thoughtStreamStorageReady) {
    startTracePolling();
  }
  return activeTraceId;
}

async function finalizeTraceSession() {
  /** Stop polling and fetch one last thought-stream snapshot. */
  if (!traceStreamEnabled) {
    return;
  }
  stopTracePolling();
  await pollTraceOnce();
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
  els.ontologyPath.disabled = disabled;
  els.ontologyBrowseBtn.disabled = disabled;
  els.loadOntologyBtn.disabled = disabled;
  els.openGraphBrowserBtn.disabled = disabled;
  els.graphRagQuestion.disabled = disabled;
  els.graphRagToggle.disabled = disabled;
  els.graphRagStreamToggle.disabled = disabled;
  els.graphRagAskBtn.disabled = disabled;
  els.graphRagClearBtn.disabled = disabled;
  // Thought Stream toggle stays interactive during inferencing.
  els.traceStreamToggle.disabled = false;
  // Runtime metrics refresh stays interactive during inferencing.
  els.refreshMetricsBtn.disabled = false;
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

function renderOntologyDropdown(payload) {
  /** Render server-discovered OWL file choices for ontology path input. */
  const options = payload.options || [];
  const current = els.ontologyPath.value.trim();
  els.ontologyOptions.innerHTML = '';

  const availablePaths = new Set();
  for (const item of options) {
    const option = document.createElement('option');
    option.value = item.path;
    option.label = item.label || item.path;
    els.ontologyOptions.appendChild(option);
    availablePaths.add(item.path);
  }

  if (!current) {
    const fallback = availablePaths.has(payload.suggested)
      ? payload.suggested
      : options[0]?.path || '';
    els.ontologyPath.value = fallback;
  }
  return options.length > 0;
}

async function loadOntologyOptions({ silent = false } = {}) {
  /** Fetch available OWL paths and populate ontology dropdown options. */
  if (!silent) {
    startUiProcessing('Loading ontology file options...');
  }
  try {
    const payload = await api('/api/graph-rag/owl-options');
    const hasOptions = renderOntologyDropdown(payload);
    const base = (payload.base_directory || '').trim();
    if (!silent) {
      if (hasOptions) {
        setStatus(
          base
            ? `Ontology base: ${base}. Selected: ${els.ontologyPath.value.trim()}.`
            : `Ontology options loaded. Selected: ${els.ontologyPath.value.trim()}.`
        );
      } else {
        setStatus(
          base
            ? `No .owl files found under base: ${base}.`
            : 'No ontology options available yet. Add .owl files and reload.'
        );
      }
    }
  } finally {
    if (!silent) {
      endUiProcessing();
    }
  }
}

function setOntologyBrowserOpen(open) {
  /** Toggle ontology file browser modal visibility. */
  els.ontologyBrowserModal.classList.toggle('hidden', !open);
}

function renderOntologyBrowserRows(title, items, kind) {
  /** Build one directory/file section markup for ontology browser modal. */
  const rows = [];
  rows.push(`<div class="browser-section-title">${escapeHtml(title)}</div>`);
  if (!items.length) {
    rows.push('<div class="browser-empty">No items in this section.</div>');
    return rows.join('');
  }
  for (const item of items) {
    rows.push(`
      <button
        class="browser-item ${escapeHtml(kind)}"
        type="button"
        data-path="${escapeHtml(item.path || '')}"
        data-kind="${escapeHtml(kind)}"
      >
        <strong>${escapeHtml(item.name || item.path || '')}</strong>
        <small>${escapeHtml(item.path || '')}</small>
      </button>
    `);
  }
  return rows.join('');
}

function renderOntologyBrowser(payload) {
  /** Render ontology file-browser directory and file rows from API payload. */
  ontologyBrowserCurrentDirectory = String(payload?.current_directory || '').trim();
  ontologyBrowserParentDirectory = String(payload?.parent_directory || '').trim();
  ontologyBrowserWildcardPath = String(payload?.wildcard_path || '').trim();

  const directories = Array.isArray(payload?.directories) ? payload.directories : [];
  const files = Array.isArray(payload?.files) ? payload.files : [];
  const base = String(payload?.base_directory || '').trim();
  els.ontologyBrowserTitle.textContent = 'Ontology File Browser';
  els.ontologyBrowserPath.textContent = `Current folder: ${ontologyBrowserCurrentDirectory || base || '(none)'}`;
  els.ontologyBrowserUpBtn.disabled = !ontologyBrowserParentDirectory;
  els.ontologyBrowserUseFolderBtn.disabled = !ontologyBrowserWildcardPath;
  els.ontologyBrowserList.innerHTML = [
    renderOntologyBrowserRows('Folders', directories, 'directory'),
    renderOntologyBrowserRows('OWL Files', files, 'file'),
  ].join('');
}

async function browseOntologyDirectory(path = '') {
  /** Fetch one ontology directory level for modal browser navigation. */
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  const payload = await api(`/api/graph-rag/owl-browser${query}`);
  renderOntologyBrowser(payload);
}

async function openOntologyBrowser() {
  /** Open ontology file browser modal and load initial directory rows. */
  setOntologyBrowserOpen(true);
  await browseOntologyDirectory(ontologyBrowserCurrentDirectory || '');
}

async function handleOntologyBrowserListClick(event) {
  /** Handle click actions on ontology browser rows (navigate/select). */
  const target = event.target instanceof HTMLElement ? event.target.closest('.browser-item') : null;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const path = String(target.dataset.path || '').trim();
  const kind = String(target.dataset.kind || '').trim();
  if (!path) {
    return;
  }
  if (kind === 'directory') {
    await browseOntologyDirectory(path);
    return;
  }
  if (kind === 'file') {
    els.ontologyPath.value = path;
    setOntologyBrowserOpen(false);
    setStatus(`Ontology file selected: ${path}`);
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

function resetChatForSelectedDeposition() {
  /** Start a fresh assistant conversation whenever the active deposition changes. */
  chatHistory = [];
  els.chatMessages.innerHTML = '';
  addMessage('assistant', 'Short answer mode is active. Ask for next legal actions or request a deeper bullet breakdown.');
}

function getDefaultDepositionId() {
  /** Pick default deposition id (highest score first) for chat/detail fallback selection. */
  if (!depositions.length) {
    return null;
  }
  const ranked = [...depositions].sort(depositionSortByScore);
  return ranked[0]?._id || null;
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

    let autoSelected = false;
    if (!selectedDepositionId && depositions.length) {
      selectedDepositionId = getDefaultDepositionId();
      if (selectedDepositionId) {
        autoSelected = true;
        resetChatForSelectedDeposition();
      }
    }

    renderTimeline();
    renderDepositions();

    if (selectedDepositionId) {
      const selected = await api(`/api/deposition/${encodeURIComponent(selectedDepositionId)}`);
      renderDetail(selected);
      if (autoSelected) {
        setStatus(
          `Loaded ${depositions.length} depositions. Auto-selected ${selected.witness_name || 'a witness'} for Persona:Attorney chat.`
        );
        return;
      }
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

async function loadOntologyGraph() {
  /** Import OWL ontology files into Neo4j Graph RAG store. */
  const ontologyPath = els.ontologyPath.value.trim();
  if (!ontologyPath) {
    setStatus('Ontology path is required.');
    return;
  }

  startUiProcessing('Loading OWL ontology into Neo4j graph...');
  try {
    const payload = await api('/api/graph-rag/load-owl', {
      method: 'POST',
      body: JSON.stringify({
        path: ontologyPath,
        clear_existing: false,
        batch_size: 500,
      }),
    });
    setStatus(
      `Graph import complete: ${payload.loaded_files} file(s), ${payload.triples} triples, `
      + `${payload.resource_relationships} resource links, ${payload.literal_relationships} literal links.`
    );
  } finally {
    endUiProcessing();
  }
}

async function askGraphRag() {
  /** Query ontology graph via backend Graph RAG endpoint and render answer/sources. */
  const question = els.graphRagQuestion.value.trim();
  if (!question) {
    setStatus('Graph RAG question is required.');
    return;
  }
  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }

  const llm = getSelectedLlm();
  const useRag = Boolean(els.graphRagToggle?.checked);
  const streamRag = Boolean(els.graphRagStreamToggle?.checked);
  const traceId = beginTraceSession('chat');
  startUiProcessing('Running Graph RAG query...');
  startLlmProcessing('Graph RAG is retrieving ontology context and generating an answer...');
  await nextPaint();
  try {
    const payload = await inferencingApi('/api/graph-rag/query', {
      method: 'POST',
      body: JSON.stringify({
        question,
        top_k: 8,
        use_rag: useRag,
        stream_rag: streamRag,
        llm_provider: llm.provider,
        llm_model: llm.model,
        thought_stream_id: traceId,
      }),
    });
    const sources = Array.isArray(payload.sources) ? payload.sources : [];
    const sourceLines = sources.length
      ? sources.slice(0, 8).map((item) => `- ${item.label} (${item.iri})`).join('\n')
      : '- No matching source nodes returned.';
    els.graphRagAnswer.textContent = `${payload.answer}\n\nSources (${payload.context_rows}):\n${sourceLines}`;
    renderGraphRagCycle(payload);
    setStatus(
      useRag
        ? `Graph RAG answer generated from ${payload.context_rows} context row(s).`
        : 'Graph answer generated with RAG retrieval disabled.'
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Graph RAG query failed.');
    els.graphRagAnswer.textContent = `Graph RAG query failed.\n${message}`;
    throw err;
  } finally {
    await finalizeTraceSession();
    endLlmProcessing();
    endUiProcessing();
  }
}

function clearGraphRag() {
  /** Reset Graph RAG question/input, answer text, and monitor cycle history. */
  graphRagCycles = [];
  els.graphRagQuestion.value = '';
  els.graphRagAnswer.textContent =
    'Ask a question to run retrieval from Neo4j ontology and generate an answer.';
  els.graphRagMonitor.textContent = 'No graph inference cycles captured yet.';
  setStatus('Cleared Graph RAG query and results.');
}

function renderGraphRagCycle(payload) {
  /** Append and render one Graph RAG inference-cycle monitor record. */
  const monitor = payload?.monitor || {};
  const terms = Array.isArray(monitor.retrieval_terms) ? monitor.retrieval_terms : [];
  const retrievedResources = Array.isArray(monitor.retrieved_resources) ? monitor.retrieved_resources : [];
  const retrievedLines = [];
  retrievedResources.slice(0, 8).forEach((resource, index) => {
    const label = String(resource?.label || resource?.iri || '').trim() || '(unknown)';
    const iri = String(resource?.iri || '').trim();
    retrievedLines.push(`[${index + 1}] ${label}${iri ? ` (${iri})` : ''}`);

    const relations = Array.isArray(resource?.relations) ? resource.relations : [];
    relations.slice(0, 3).forEach((rel) => {
      const predicate = String(rel?.predicate || 'related_to').trim();
      const targetLabel = String(rel?.object_label || rel?.object_iri || '').trim() || '(unknown)';
      const targetIri = String(rel?.object_iri || '').trim();
      retrievedLines.push(`  - REL ${predicate} -> ${targetLabel}${targetIri ? ` (${targetIri})` : ''}`);
    });

    const literals = Array.isArray(resource?.literals) ? resource.literals : [];
    literals.slice(0, 4).forEach((literal) => {
      const predicate = String(literal?.predicate || 'value').trim();
      const value = String(literal?.value || '').trim();
      if (!value) {
        return;
      }
      retrievedLines.push(`  - LIT ${predicate}: ${value}`);
    });
  });
  const cycle = {
    at: new Date().toLocaleTimeString(),
    ragEnabled: monitor.rag_enabled !== false,
    ragStreamEnabled: monitor.rag_stream_enabled !== false,
    question: String(payload?.question || '').trim(),
    contextRows: Number(payload?.context_rows || 0),
    terms: terms.join(', ') || '(none)',
    retrievedResources: retrievedLines.join('\n') || '(none)',
    contextPreview: String(monitor.context_preview || '').trim() || '(empty context)',
    llmSystemPrompt: String(monitor.llm_system_prompt || '').trim() || '(missing)',
    llmUserPrompt: String(monitor.llm_user_prompt || '').trim() || '(missing)',
  };
  graphRagCycles = [cycle, ...graphRagCycles].slice(0, 8);
  const lines = graphRagCycles.map(
    (item, index) =>
      [
        `Cycle ${graphRagCycles.length - index} @ ${item.at}`,
        `RAG enabled: ${item.ragEnabled ? 'yes' : 'no'}`,
        `RAG stream logging: ${item.ragStreamEnabled ? 'yes' : 'no'}`,
        `Question: ${item.question}`,
        `Context rows: ${item.contextRows}`,
        `Retrieval terms: ${item.terms}`,
        'Retrieved resources:',
        item.retrievedResources,
        'Context -> LLM:',
        item.contextPreview,
        'System prompt -> LLM:',
        item.llmSystemPrompt,
        'User prompt -> LLM:',
        item.llmUserPrompt,
      ].join('\n')
  );
  els.graphRagMonitor.textContent = lines.join('\n\n==============================\n\n');
}

async function openGraphBrowser() {
  /** Open Neo4j Browser URL for graph inspection. */
  const payload = await api('/api/graph-rag/browser');
  const target = String(payload.launch_url || payload.browser_url || '').trim();
  if (!target) {
    throw new Error('Graph browser URL is not configured.');
  }
  const handle = window.open(target, '_blank', 'noopener,noreferrer');
  if (!handle) {
    setStatus(`Graph Browser URL: ${target}`);
    return;
  }
  setStatus(`Opened Graph Browser with node graph starter query at ${target}.`);
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

  resetChatForSelectedDeposition();
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
  const traceId = beginTraceSession('ingest');
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
        thought_stream_id: traceId,
      }),
    });
  } finally {
    await finalizeTraceSession();
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
    const fallbackDepositionId = getDefaultDepositionId();
    if (fallbackDepositionId) {
      await selectDeposition(fallbackDepositionId);
    }
  }

  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }

  if (!selectedDepositionId) {
    setStatus('No deposition is loaded for this case. Run Load.Depositions first.');
    addMessage('assistant', 'No deposition selected. Load.Depositions, then select a witness to start Persona:Attorney chat.');
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
  const traceId = beginTraceSession('chat');
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
        thought_stream_id: traceId,
      }),
    });

    addMessage('assistant', payload.response);
    chatHistory.push({ role: 'assistant', content: payload.response });
    await loadCases({ silent: true });
  } finally {
    await finalizeTraceSession();
    setChatProcessing(false);
    endLlmProcessing();
  }
}

els.newCaseBtn.addEventListener('click', () => createBlankCase());
els.ingestBtn.addEventListener('click', () => ingestCase().catch((err) => setStatus(err.message)));
els.refreshBtn.addEventListener('click', () => refreshCase().catch((err) => setStatus(err.message)));
els.saveCaseBtn.addEventListener('click', () => saveCase().catch((err) => setStatus(err.message)));
els.loadOntologyBtn.addEventListener('click', () => loadOntologyGraph().catch((err) => setStatus(err.message)));
els.graphRagAskBtn.addEventListener('click', () => askGraphRag().catch((err) => setStatus(err.message)));
els.graphRagClearBtn.addEventListener('click', () => clearGraphRag());
els.openGraphBrowserBtn.addEventListener('click', () => openGraphBrowser().catch((err) => setStatus(err.message)));
els.refreshCasesBtn.addEventListener('click', () => loadCases().catch((err) => setStatus(err.message)));
els.refreshModelsBtn.addEventListener('click', () =>
  loadLlmOptions({ forceProbe: true }).catch((err) => setStatus(err.message))
);
els.traceStreamToggle.addEventListener('change', () =>
  handleTraceStreamToggle().catch((err) => setStatus(err.message))
);
els.traceOlderBtn.addEventListener('click', () => shiftTraceWindow(-1));
els.traceNewerBtn.addEventListener('click', () => shiftTraceWindow(1));
els.metricsTextToggle.addEventListener('click', () => toggleMetricsPanel().catch((err) => setStatus(err.message)));
els.refreshMetricsBtn.addEventListener('click', () => loadAgentMetrics().catch((err) => setStatus(err.message)));
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
els.ontologyBrowseBtn.addEventListener('click', () =>
  openOntologyBrowser().catch((err) => setStatus(err.message))
);
els.ontologyBrowserCloseBtn.addEventListener('click', () => setOntologyBrowserOpen(false));
els.ontologyBrowserUpBtn.addEventListener('click', () =>
  browseOntologyDirectory(ontologyBrowserParentDirectory).catch((err) => setStatus(err.message))
);
els.ontologyBrowserRefreshBtn.addEventListener('click', () =>
  browseOntologyDirectory(ontologyBrowserCurrentDirectory).catch((err) => setStatus(err.message))
);
els.ontologyBrowserUseFolderBtn.addEventListener('click', () => {
  if (!ontologyBrowserWildcardPath) {
    return;
  }
  els.ontologyPath.value = ontologyBrowserWildcardPath;
  setOntologyBrowserOpen(false);
  setStatus(`Ontology folder selected: ${ontologyBrowserWildcardPath}`);
});
els.ontologyBrowserList.addEventListener('click', (event) =>
  handleOntologyBrowserListClick(event).catch((err) => setStatus(err.message))
);
els.ontologyBrowserModal.addEventListener('click', (event) => {
  if (event.target === els.ontologyBrowserModal) {
    setOntologyBrowserOpen(false);
  }
});
els.ontologyPath.addEventListener('change', () => {
  const selected = els.ontologyPath.value.trim();
  if (!selected) {
    setStatus('Select an ontology path.');
    return;
  }
  setStatus(`Ontology path selected: ${selected}`);
});
els.graphRagQuestion.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') {
    return;
  }
  event.preventDefault();
  askGraphRag().catch((err) => setStatus(err.message));
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
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !els.ontologyBrowserModal.classList.contains('hidden')) {
    setOntologyBrowserOpen(false);
  }
});

renderDetail(null);
renderTimeline();
renderDepositions();
loadedCaseId = els.caseId.value.trim();
syncCaseActionState();
setTraceStreamEnabled(false);
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
loadOntologyOptions({ silent: true }).catch((err) => {
  const message = err instanceof Error ? err.message : String(err || 'Unknown error');
  setStatus(`Failed to load ontology options: ${message}`);
});
setMetricsPanelOpen(false);
