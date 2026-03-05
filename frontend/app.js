/*
 * Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
 * License: Proprietary. See NOTICE.md.
 * Author: Paul Harvener.
 */

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
  tabLandingBtn: document.getElementById('tabLandingBtn'),
  tabIntelligenceBtn: document.getElementById('tabIntelligenceBtn'),
  tabProvisioningBtn: document.getElementById('tabProvisioningBtn'),
  tabObservablesBtn: document.getElementById('tabObservablesBtn'),
  tabAdminBtn: document.getElementById('tabAdminBtn'),
  tabPageLanding: document.getElementById('tabPageLanding'),
  tabPageIntelligence: document.getElementById('tabPageIntelligence'),
  intelligenceLanding: document.getElementById('intelligenceLanding'),
  intelligenceWorkspace: document.getElementById('intelligenceWorkspace'),
  tabPageProvisioning: document.getElementById('tabPageProvisioning'),
  tabPageObservables: document.getElementById('tabPageObservables'),
  tabPageAdmin: document.getElementById('tabPageAdmin'),
  caseId: document.getElementById('caseId'),
  directory: document.getElementById('directory'),
  directoryOptions: document.getElementById('directoryOptions'),
  browseDepositionBtn: document.getElementById('browseDepositionBtn'),
  caseList: document.getElementById('caseList'),
  llmSelect: document.getElementById('llmSelect'),
  ingestSchemaSelect: document.getElementById('ingestSchemaSelect'),
  ingestSchemaKey: document.getElementById('ingestSchemaKey'),
  ingestSchemaJson: document.getElementById('ingestSchemaJson'),
  newIngestSchemaBtn: document.getElementById('newIngestSchemaBtn'),
  saveIngestSchemaBtn: document.getElementById('saveIngestSchemaBtn'),
  removeIngestSchemaBtn: document.getElementById('removeIngestSchemaBtn'),
  ingestSchemaStatus: document.getElementById('ingestSchemaStatus'),
  skipReassess: document.getElementById('skipReassess'),
  traceStreamToggle: document.getElementById('traceStreamToggle'),
  newCaseBtn: document.getElementById('newCaseBtn'),
  saveCaseBtn: document.getElementById('saveCaseBtn'),
  importDepositionBtn: document.getElementById('importDepositionBtn'),
  importDepositionFolderBtn: document.getElementById('importDepositionFolderBtn'),
  importDepositionInput: document.getElementById('importDepositionInput'),
  importDepositionFolderInput: document.getElementById('importDepositionFolderInput'),
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
  timelineScale: document.getElementById('timelineScale'),
  timelineBack: document.getElementById('timelineBack'),
  timelineForward: document.getElementById('timelineForward'),
  saveIntelligenceBtn: document.getElementById('saveIntelligenceBtn'),
  duplicateCaseBtn: document.getElementById('duplicateCaseBtn'),
  detailEmpty: document.getElementById('detailEmpty'),
  detailBody: document.getElementById('detailBody'),
  detailWitness: document.getElementById('detailWitness'),
  detailSummary: document.getElementById('detailSummary'),
  detailExplanation: document.getElementById('detailExplanation'),
  computeSentimentBtn: document.getElementById('computeSentimentBtn'),
  detailSentiment: document.getElementById('detailSentiment'),
  detailContradictions: document.getElementById('detailContradictions'),
  focusedReasoning: document.getElementById('focusedReasoning'),
  reasoningProgress: document.getElementById('reasoningProgress'),
  reasoningClock: document.getElementById('reasoningClock'),
  summarizeFocusedReasoningBtn: document.getElementById('summarizeFocusedReasoningBtn'),
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
  openGrafanaObservablesBtn: document.getElementById('openGrafanaObservablesBtn'),
  metricsSampleMeta: document.getElementById('metricsSampleMeta'),
  metricsStorageMeta: document.getElementById('metricsStorageMeta'),
  metricsGrid: document.getElementById('metricsGrid'),
  correctnessGrid: document.getElementById('correctnessGrid'),
  metricDetailPanel: document.getElementById('metricDetailPanel'),
  metricDetailTitle: document.getElementById('metricDetailTitle'),
  metricDetailBody: document.getElementById('metricDetailBody'),
  metricDetailCloseBtn: document.getElementById('metricDetailCloseBtn'),
  metricTrendPanel: document.getElementById('metricTrendPanel'),
  metricTrendTitle: document.getElementById('metricTrendTitle'),
  metricTrendMeta: document.getElementById('metricTrendMeta'),
  metricTrendBody: document.getElementById('metricTrendBody'),
  metricTrendSvg: document.getElementById('metricTrendSvg'),
  metricTrendCloseBtn: document.getElementById('metricTrendCloseBtn'),
  ontologyPath: document.getElementById('ontologyPath'),
  ontologyOptions: document.getElementById('ontologyOptions'),
  ontologyBrowseBtn: document.getElementById('ontologyBrowseBtn'),
  loadOntologyBtn: document.getElementById('loadOntologyBtn'),
  openGraphBrowserBtn: document.getElementById('openGraphBrowserBtn'),
  graphRagEmbeddingEnabled: document.getElementById('graphRagEmbeddingEnabled'),
  graphRagEmbeddingProvider: document.getElementById('graphRagEmbeddingProvider'),
  graphRagEmbeddingModel: document.getElementById('graphRagEmbeddingModel'),
  graphRagEmbeddingDimensions: document.getElementById('graphRagEmbeddingDimensions'),
  graphRagEmbeddingIndex: document.getElementById('graphRagEmbeddingIndex'),
  graphRagEmbeddingNodeLabel: document.getElementById('graphRagEmbeddingNodeLabel'),
  graphRagEmbeddingProperty: document.getElementById('graphRagEmbeddingProperty'),
  saveGraphRagEmbeddingBtn: document.getElementById('saveGraphRagEmbeddingBtn'),
  reloadGraphRagEmbeddingBtn: document.getElementById('reloadGraphRagEmbeddingBtn'),
  graphRagQuestion: document.getElementById('graphRagQuestion'),
  graphRagToggle: document.getElementById('graphRagToggle'),
  graphRagStreamToggle: document.getElementById('graphRagStreamToggle'),
  graphRagAskBtn: document.getElementById('graphRagAskBtn'),
  graphRagClearBtn: document.getElementById('graphRagClearBtn'),
  graphRagAnswer: document.getElementById('graphRagAnswer'),
  graphRagMonitor: document.getElementById('graphRagMonitor'),
  adminTabTestBtn: document.getElementById('adminTabTestBtn'),
  adminTabUsersBtn: document.getElementById('adminTabUsersBtn'),
  adminTabPersonasBtn: document.getElementById('adminTabPersonasBtn'),
  adminTabMlopsBtn: document.getElementById('adminTabMlopsBtn'),
  adminTabPageTest: document.getElementById('adminTabPageTest'),
  adminTabPageUsers: document.getElementById('adminTabPageUsers'),
  adminTabPagePersonas: document.getElementById('adminTabPagePersonas'),
  adminTabPageMlops: document.getElementById('adminTabPageMlops'),
  adminUserCreatePanel: document.getElementById('adminUserCreatePanel'),
  adminUserFirstName: document.getElementById('adminUserFirstName'),
  adminUserLastName: document.getElementById('adminUserLastName'),
  adminUserAuthorization: document.getElementById('adminUserAuthorization'),
  adminAddUserBtn: document.getElementById('adminAddUserBtn'),
  adminSaveUserBtn: document.getElementById('adminSaveUserBtn'),
  adminCancelUserBtn: document.getElementById('adminCancelUserBtn'),
  adminGetUsersBtn: document.getElementById('adminGetUsersBtn'),
  adminRemoveUserBtn: document.getElementById('adminRemoveUserBtn'),
  adminUserFeedback: document.getElementById('adminUserFeedback'),
  adminUserSelect: document.getElementById('adminUserSelect'),
  adminUserSelectedMeta: document.getElementById('adminUserSelectedMeta'),
  adminUserList: document.getElementById('adminUserList'),
  adminRefreshUsersBtn: document.getElementById('adminRefreshUsersBtn'),
  adminUserDetailPanel: document.getElementById('adminUserDetailPanel'),
  adminUserDetailTitle: document.getElementById('adminUserDetailTitle'),
  adminUserDetailMeta: document.getElementById('adminUserDetailMeta'),
  adminUserDetailBody: document.getElementById('adminUserDetailBody'),
  adminUserDetailCloseBtn: document.getElementById('adminUserDetailCloseBtn'),
  adminPersonaCreatePanel: document.getElementById('adminPersonaCreatePanel'),
  adminPersonaName: document.getElementById('adminPersonaName'),
  adminPersonaLlm: document.getElementById('adminPersonaLlm'),
  adminPersonaSystemPromptTemplateSelect: document.getElementById('adminPersonaSystemPromptTemplateSelect'),
  adminPersonaAssistantPromptTemplateSelect: document.getElementById('adminPersonaAssistantPromptTemplateSelect'),
  adminPersonaContextPromptTemplateSelect: document.getElementById('adminPersonaContextPromptTemplateSelect'),
  adminPersonaSystemChoosePromptBtn: document.getElementById('adminPersonaSystemChoosePromptBtn'),
  adminPersonaAssistantChoosePromptBtn: document.getElementById('adminPersonaAssistantChoosePromptBtn'),
  adminPersonaContextChoosePromptBtn: document.getElementById('adminPersonaContextChoosePromptBtn'),
  adminPersonaSystemObservableBtn: document.getElementById('adminPersonaSystemObservableBtn'),
  adminPersonaAssistantObservableBtn: document.getElementById('adminPersonaAssistantObservableBtn'),
  adminPersonaContextObservableBtn: document.getElementById('adminPersonaContextObservableBtn'),
  adminPersonaSystemSavePromptBtn: document.getElementById('adminPersonaSystemSavePromptBtn'),
  adminPersonaAssistantSavePromptBtn: document.getElementById('adminPersonaAssistantSavePromptBtn'),
  adminPersonaContextSavePromptBtn: document.getElementById('adminPersonaContextSavePromptBtn'),
  adminPersonaOpenPromptModalBtn: document.getElementById('adminPersonaOpenPromptModalBtn'),
  adminPersonaToggleRagBtn: document.getElementById('adminPersonaToggleRagBtn'),
  adminPersonaTogglePromptObservablesBtn: document.getElementById('adminPersonaTogglePromptObservablesBtn'),
  adminPersonaToggleToolsBtn: document.getElementById('adminPersonaToggleToolsBtn'),
  adminPersonaPromptPanel: document.getElementById('adminPersonaPromptPanel'),
  adminPersonaRagPanel: document.getElementById('adminPersonaRagPanel'),
  adminPersonaPromptObservablesPanel: document.getElementById('adminPersonaPromptObservablesPanel'),
  adminPersonaToolsPanel: document.getElementById('adminPersonaToolsPanel'),
  adminPersonaSystemPrompt: document.getElementById('adminPersonaSystemPrompt'),
  adminPersonaAssistantPrompt: document.getElementById('adminPersonaAssistantPrompt'),
  adminPersonaContextPrompt: document.getElementById('adminPersonaContextPrompt'),
  adminPersonaRagSelect: document.getElementById('adminPersonaRagSelect'),
  adminPersonaLoadRagsBtn: document.getElementById('adminPersonaLoadRagsBtn'),
  adminPersonaRagAddBtn: document.getElementById('adminPersonaRagAddBtn'),
  adminPersonaRagList: document.getElementById('adminPersonaRagList'),
  adminPersonaRefreshPromptObservablesBtn: document.getElementById('adminPersonaRefreshPromptObservablesBtn'),
  adminPersonaPromptObservablesList: document.getElementById('adminPersonaPromptObservablesList'),
  adminPersonaPromptObservablesDetail: document.getElementById('adminPersonaPromptObservablesDetail'),
  adminPersonaToolSelect: document.getElementById('adminPersonaToolSelect'),
  adminPersonaLoadToolsBtn: document.getElementById('adminPersonaLoadToolsBtn'),
  adminPersonaToolAddBtn: document.getElementById('adminPersonaToolAddBtn'),
  adminPersonaToolList: document.getElementById('adminPersonaToolList'),
  adminAddPersonaBtn: document.getElementById('adminAddPersonaBtn'),
  adminSavePersonaBtn: document.getElementById('adminSavePersonaBtn'),
  adminPersonaFormPromptSentimentBtn: document.getElementById('adminPersonaFormPromptSentimentBtn'),
  adminPersonaFormPromptSentiment: document.getElementById('adminPersonaFormPromptSentiment'),
  adminCancelPersonaBtn: document.getElementById('adminCancelPersonaBtn'),
  adminPersonaSmokeTestBtn: document.getElementById('adminPersonaSmokeTestBtn'),
  adminPersonaFeedback: document.getElementById('adminPersonaFeedback'),
  adminPersonaSelect: document.getElementById('adminPersonaSelect'),
  adminPersonaSelectedMeta: document.getElementById('adminPersonaSelectedMeta'),
  adminPersonaList: document.getElementById('adminPersonaList'),
  adminPersonaGraphMeta: document.getElementById('adminPersonaGraphMeta'),
  adminPersonaGraphProgress: document.getElementById('adminPersonaGraphProgress'),
  adminPersonaGraphClock: document.getElementById('adminPersonaGraphClock'),
  adminPersonaGraphQuestion: document.getElementById('adminPersonaGraphQuestion'),
  adminPersonaGraphAskBtn: document.getElementById('adminPersonaGraphAskBtn'),
  adminPersonaGraphClearBtn: document.getElementById('adminPersonaGraphClearBtn'),
  adminPersonaGraphAnswer: document.getElementById('adminPersonaGraphAnswer'),
  adminRunTestsBtn: document.getElementById('adminRunTestsBtn'),
  adminRefreshTestLogBtn: document.getElementById('adminRefreshTestLogBtn'),
  adminTestRunClock: document.getElementById('adminTestRunClock'),
  adminTestLogSummary: document.getElementById('adminTestLogSummary'),
  adminTestLogOutput: document.getElementById('adminTestLogOutput'),
  adminTestReportFrame: document.getElementById('adminTestReportFrame'),
  adminMlopsRefreshMetricsBtn: document.getElementById('adminMlopsRefreshMetricsBtn'),
  adminMlopsRefreshModelsBtn: document.getElementById('adminMlopsRefreshModelsBtn'),
  adminMlopsOpenGrafanaBtn: document.getElementById('adminMlopsOpenGrafanaBtn'),
  adminMlopsPromptVersionsBtn: document.getElementById('adminMlopsPromptVersionsBtn'),
  adminMlopsModelRoutingBtn: document.getElementById('adminMlopsModelRoutingBtn'),
  adminMlopsRagBehaviorBtn: document.getElementById('adminMlopsRagBehaviorBtn'),
  adminMlopsTokenContextBtn: document.getElementById('adminMlopsTokenContextBtn'),
  adminMlopsCorrectnessBtn: document.getElementById('adminMlopsCorrectnessBtn'),
  adminMlopsTraceQualityBtn: document.getElementById('adminMlopsTraceQualityBtn'),
  adminMlopsThoughtHealthBtn: document.getElementById('adminMlopsThoughtHealthBtn'),
  adminMlopsRagHealthBtn: document.getElementById('adminMlopsRagHealthBtn'),
  adminMlopsOpenGithubActionsBtn: document.getElementById('adminMlopsOpenGithubActionsBtn'),
  adminMlopsOpenCiWorkflowBtn: document.getElementById('adminMlopsOpenCiWorkflowBtn'),
  adminMlopsOpenDeployWorkflowBtn: document.getElementById('adminMlopsOpenDeployWorkflowBtn'),
  adminMlopsRunTestsBtn: document.getElementById('adminMlopsRunTestsBtn'),
  adminMlopsRefreshReportBtn: document.getElementById('adminMlopsRefreshReportBtn'),
  adminMlopsTabLlmoopsBtn: document.getElementById('adminMlopsTabLlmoopsBtn'),
  adminMlopsTabFineTuningBtn: document.getElementById('adminMlopsTabFineTuningBtn'),
  adminMlopsTabDeploymentBtn: document.getElementById('adminMlopsTabDeploymentBtn'),
  adminMlopsTabCicdBtn: document.getElementById('adminMlopsTabCicdBtn'),
  adminMlopsTabPageLlmoops: document.getElementById('adminMlopsTabPageLlmoops'),
  adminMlopsTabPageFineTuning: document.getElementById('adminMlopsTabPageFineTuning'),
  adminMlopsTabPageDeployment: document.getElementById('adminMlopsTabPageDeployment'),
  adminMlopsTabPageCicd: document.getElementById('adminMlopsTabPageCicd'),
  adminMlopsOpenFineTuningBtn: document.getElementById('adminMlopsOpenFineTuningBtn'),
  adminMlopsFineTuningRefreshModelsBtn: document.getElementById('adminMlopsFineTuningRefreshModelsBtn'),
  adminMlopsDeploymentThoughtBtn: document.getElementById('adminMlopsDeploymentThoughtBtn'),
  adminMlopsDeploymentRagBtn: document.getElementById('adminMlopsDeploymentRagBtn'),
  adminMlopsDeploymentObservablesBtn: document.getElementById('adminMlopsDeploymentObservablesBtn'),
  adminMlopsOpenAdminTestBtn: document.getElementById('adminMlopsOpenAdminTestBtn'),
  adminMlopsStatus: document.getElementById('adminMlopsStatus'),
  depositionBrowserModal: document.getElementById('depositionBrowserModal'),
  depositionBrowserTitle: document.getElementById('depositionBrowserTitle'),
  depositionBrowserPath: document.getElementById('depositionBrowserPath'),
  depositionBrowserList: document.getElementById('depositionBrowserList'),
  depositionBrowserCloseBtn: document.getElementById('depositionBrowserCloseBtn'),
  depositionBrowserUpBtn: document.getElementById('depositionBrowserUpBtn'),
  depositionBrowserUseFolderBtn: document.getElementById('depositionBrowserUseFolderBtn'),
  depositionBrowserRefreshBtn: document.getElementById('depositionBrowserRefreshBtn'),
  ontologyBrowserModal: document.getElementById('ontologyBrowserModal'),
  ontologyBrowserTitle: document.getElementById('ontologyBrowserTitle'),
  ontologyBrowserPath: document.getElementById('ontologyBrowserPath'),
  ontologyBrowserList: document.getElementById('ontologyBrowserList'),
  ontologyBrowserCloseBtn: document.getElementById('ontologyBrowserCloseBtn'),
  ontologyBrowserUpBtn: document.getElementById('ontologyBrowserUpBtn'),
  ontologyBrowserUseFolderBtn: document.getElementById('ontologyBrowserUseFolderBtn'),
  ontologyBrowserRefreshBtn: document.getElementById('ontologyBrowserRefreshBtn'),
  adminPersonaPromptModal: document.getElementById('adminPersonaPromptModal'),
  adminPersonaPromptModalTitle: document.getElementById('adminPersonaPromptModalTitle'),
  adminPersonaPromptModalMeta: document.getElementById('adminPersonaPromptModalMeta'),
  adminPersonaPromptModalSystem: document.getElementById('adminPersonaPromptModalSystem'),
  adminPersonaPromptModalAssistant: document.getElementById('adminPersonaPromptModalAssistant'),
  adminPersonaPromptModalContext: document.getElementById('adminPersonaPromptModalContext'),
  adminPersonaPromptSentimentMeta: document.getElementById('adminPersonaPromptSentimentMeta'),
  adminPersonaPromptSentimentBtn: document.getElementById('adminPersonaPromptSentimentBtn'),
  adminPersonaPromptApplyBtn: document.getElementById('adminPersonaPromptApplyBtn'),
  adminPersonaPromptResaveBtn: document.getElementById('adminPersonaPromptResaveBtn'),
  adminPersonaPromptModalCloseBtn: document.getElementById('adminPersonaPromptModalCloseBtn'),
  adminPersonaPromptObservableModal: document.getElementById('adminPersonaPromptObservableModal'),
  adminPersonaPromptObservableModalTitle: document.getElementById('adminPersonaPromptObservableModalTitle'),
  adminPersonaPromptObservableModalMeta: document.getElementById('adminPersonaPromptObservableModalMeta'),
  adminPersonaPromptObservableModalBody: document.getElementById('adminPersonaPromptObservableModalBody'),
  adminPersonaPromptObservableModalCloseBtn: document.getElementById('adminPersonaPromptObservableModalCloseBtn'),
};

const GITHUB_ACTIONS_URLS = Object.freeze({
  actions: 'https://github.com/data-blitz-demos/agentic-legal-deposition-demo/actions',
  ciWorkflow:
    'https://github.com/data-blitz-demos/agentic-legal-deposition-demo/actions/workflows/ci-cd.yml',
  deployWorkflow:
    'https://github.com/data-blitz-demos/agentic-legal-deposition-demo/actions/workflows/deploy.yml',
});

let depositions = [];
let cases = [];
let selectedDepositionId = null;
let chatHistory = [];
let loadedCaseId = '';
let currentDepositionSentiment = null;
let depositionSentimentDetailOpen = false;
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
let depositionBrowserCurrentDirectory = '';
let depositionBrowserParentDirectory = '';
let depositionBrowserWildcardPath = '';
let ontologyBrowserCurrentDirectory = '';
let ontologyBrowserParentDirectory = '';
let ontologyBrowserWildcardPath = '';
let graphRagCycles = [];
let activeTab = 'landing';
let activeAdminSubtab = 'users';
let activeMlopsSubtab = 'llmops';
let adminUsersCache = [];
let adminPersonasCache = [];
let adminPersonaPromptTemplatesCache = [];
let adminPersonaRagOptionsCache = [];
let adminPersonaToolOptionsCache = [];
let selectedAdminUserId = '';
let editingAdminUserId = '';
let selectedAdminPersonaId = '';
let editingAdminPersonaId = '';
let adminPersonaRagSequence = [];
let adminPersonaToolSequence = [];
let adminPersonaSelectedPromptTemplateKey = '';
let adminPersonaPromptModalPersonaId = '';
let adminPersonaPromptModalPersonaName = '';
let adminPersonaPromptPanelActive = false;
let adminPersonaRagPanelActive = false;
let adminPersonaPromptObservablesPanelActive = false;
let adminPersonaToolsPanelActive = false;
let selectedAdminPersonaPromptObservableKey = '';
let adminPersonaPromptObservablesScope = 'all';
let adminPersonaPromptObservableModalScope = '';
let currentAdminPersonaPromptSentiment = null;
let adminPersonaPromptSentimentDetailOpen = false;
let adminTestRunClockHandle = null;
let adminTestRunStartedAtMs = 0;
const LAST_USED_CASE_STORAGE_KEY = 'deposition-demo:last-used-case-id';
const METRICS_CACHE_STORAGE_KEY = 'deposition-demo:last-agent-metrics';
let metricInteractionLockUntil = 0;
let focusedReasoningSourceText = '';
let focusedReasoningIsSummary = false;
let pendingSavedLlmValue = '';
let ingestSchemaOptionsCache = [];
let editingNewIngestSchema = false;
const ADMIN_TEST_REPORT_URL = '/admin/test-report';
const ADMIN_AUTHORIZATION_LABELS = {
  open: 'Open',
  admin: 'Admin',
  expert_user: 'Expert User',
  user: 'User',
  read_only: 'Read Only',
};
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
  rag_avg_context_bytes_on:
    'Average UTF-8 byte size of the retrieved RAG context that is passed toward the LLM on completed RAG-enabled calls. Use this to track retrieval payload growth and prompt budget pressure.',
  rag_avg_answer_word_delta_on_minus_off:
    'Average answer length difference for paired queries computed as (RAG ON words - RAG OFF words). Positive values suggest retrieval-expanded responses.',
  rag_completed_queries_split:
    'Completed Graph RAG query count split shown as ON/OFF. Use to validate whether enough A/B traffic exists for reliable influence metrics.',
};
const RUNTIME_METRIC_TRACEABILITY = {
  task_success_rate_pct:
    'Derived from thought-stream session statuses in the current lookback window using completed runs divided by all finished runs.',
  run_failure_rate_pct:
    'Derived from the same finished thought-stream session set using failed runs divided by all finished runs.',
  p95_end_to_end_latency_sec:
    'Computed from thought-stream session timestamps using created_at to updated_at duration for each sampled run, then taking the 95th percentile.',
  p95_time_to_first_event_sec:
    'Computed from thought-stream trace timestamps using the delay between session start and the first recorded trace event, then taking the 95th percentile.',
  avg_steps_per_finished_run:
    'Computed from sampled thought-stream traces by counting recorded events on completed and failed runs, then averaging those counts.',
  loop_risk_rate_pct:
    'Computed from sampled thought-stream traces as the share of finished runs with 20 or more recorded trace events.',
  in_flight_runs:
    'Counted directly from thought-stream sessions still marked running inside the current lookback window.',
  finished_runs_per_hour:
    'Computed from finished thought-stream sessions in the lookback window and normalized by the selected hour span.',
  rag_toggle_comparison_pairs:
    'Built from rag-stream completed answer events by pairing the latest RAG ON and RAG OFF result for the same normalized question text.',
  rag_answer_change_rate_pct:
    'Computed from those paired rag-stream comparisons as the share where normalized answer text differs between RAG ON and RAG OFF.',
  rag_context_hit_rate_pct:
    'Computed from completed rag-stream RAG ON events as the share where retrieval returned one or more context rows.',
  rag_avg_context_rows_on:
    'Computed from completed rag-stream RAG ON events by averaging each event\'s retrieved context_rows count.',
  rag_avg_context_bytes_on:
    'Computed from completed rag-stream RAG ON events by averaging each event\'s context_bytes payload sent toward the LLM.',
  rag_avg_answer_word_delta_on_minus_off:
    'Computed from paired rag-stream RAG ON and RAG OFF answers by subtracting OFF answer word count from ON answer word count, then averaging the differences.',
  rag_completed_queries_split:
    'Counted directly from completed rag-stream answer events and displayed as RAG ON count versus RAG OFF count.',
};
const CORRECTNESS_DRIFT_OBSERVABLES = [
  {
    key: 'golden_set_accuracy',
    label: 'Golden Set Accuracy',
    display: 'Track',
    target: '>= 95%',
    formula: 'correct / total over fixed eval set',
    description: 'Primary correctness monitor over a stable benchmark set.',
    tracking:
      'Track with a versioned benchmark harness and store each scored run by model, prompt bundle, and release so regression deltas remain attributable.',
    detail:
      'Use a fixed, versioned benchmark set with known expected outputs. Re-baseline only after deliberate model or rubric changes, otherwise drops are true quality regressions.',
  },
  {
    key: 'schema_adherence_rate',
    label: 'Schema Adherence Rate',
    display: 'Track',
    target: '>= 99%',
    formula: 'valid_structured_outputs / total_structured_outputs',
    description: 'Catches parser/format regressions before they hit production workflows.',
    tracking:
      'Track per endpoint and per model version from production responses by validating every structured payload against its schema and storing pass/fail counts.',
    detail:
      'Track by endpoint and by model version. A drop here predicts downstream ingest failures before users see them.',
  },
  {
    key: 'unsupported_claim_rate',
    label: 'Unsupported Claim Rate',
    display: 'Track',
    target: '<= 2%',
    formula: 'unsupported_claims / total_claims',
    description: 'Groundedness signal for hallucination-style drift.',
    tracking:
      'Track from audited samples where claims are checked against source evidence, then store unsupported-claim counts by model, task, and retrieval mode.',
    detail:
      'Compute on audited samples where each claim is verified against source evidence. Rising values indicate model drift toward speculation or weak retrieval grounding.',
  },
  {
    key: 'repeat_prompt_inconsistency',
    label: 'Repeat Prompt Inconsistency',
    display: 'Track',
    target: '<= 10%',
    formula: 'inconsistent_repeats / repeat_prompt_groups',
    description: 'Compares outputs for near-identical prompts to detect stability drift.',
    tracking:
      'Track by replaying fixed prompt cohorts under identical settings, grouping repeated runs together, and counting materially different outputs within each cohort.',
    detail:
      'Run repeated prompt cohorts under fixed settings (temperature, tools, context). Increasing inconsistency means behavior is less predictable and harder to trust operationally.',
  },
  {
    key: 'model_mix_drift_jsd',
    label: 'Model Mix Drift (JSD)',
    display: 'Track',
    target: '<= 0.12',
    formula: 'Jensen-Shannon divergence(recent_model_mix, baseline_model_mix)',
    description: 'Detects unexpected provider/model routing changes over time.',
    tracking:
      'Track from routing telemetry by snapshotting recent provider-model usage distribution and comparing it to a pinned baseline distribution.',
    detail:
      'Watch this after routing-rule changes and incident failovers. Large divergence can explain abrupt behavior shifts even when prompts stayed constant.',
  },
  {
    key: 'judge_human_disagreement',
    label: 'Judge-Human Disagreement',
    display: 'Track',
    target: '<= 5%',
    formula: 'judge_human_disagreements / adjudicated_samples',
    description: 'Validates whether auto-eval quality still aligns with human review.',
    tracking:
      'Track from adjudicated review samples where automated judge scores are compared with human labels, then record disagreement rate by rubric version.',
    detail:
      'Use this as guardrail for automated evaluations. If disagreement rises, retrain judge prompts/rubrics and increase human spot checks before trusting score trends.',
  },
];
const timerHandles = {
  llm: null,
  reasoning: null,
  chat: null,
  adminPersonaGraph: null,
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

function readLastUsedCaseId() {
  /** Read last-used case id from local storage (best-effort). */
  try {
    const value = window.localStorage.getItem(LAST_USED_CASE_STORAGE_KEY);
    return String(value || '').trim();
  } catch (_err) {
    return '';
  }
}

function writeLastUsedCaseId(caseId) {
  /** Persist last-used case id to local storage (best-effort). */
  const normalized = String(caseId || '').trim();
  try {
    if (!normalized) {
      window.localStorage.removeItem(LAST_USED_CASE_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(LAST_USED_CASE_STORAGE_KEY, normalized);
  } catch (_err) {
    // Ignore storage errors to avoid blocking core workflow.
  }
}

function readCachedMetricsPayload() {
  /** Read the last successful observables payload from local storage (best-effort). */
  try {
    const raw = window.localStorage.getItem(METRICS_CACHE_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const payload = JSON.parse(raw);
    return payload && typeof payload === 'object' ? payload : null;
  } catch (_err) {
    return null;
  }
}

function writeCachedMetricsPayload(payload) {
  /** Persist the last successful observables payload so the dashboard can restore between runs. */
  if (!payload || typeof payload !== 'object') {
    return;
  }
  try {
    window.localStorage.setItem(METRICS_CACHE_STORAGE_KEY, JSON.stringify(payload));
  } catch (_err) {
    // Ignore storage errors to avoid blocking observability rendering.
  }
}

async function api(path, options = {}) {
  /** Perform JSON API calls and normalize non-2xx errors. */
  try {
    const isMultipart = options?.body instanceof FormData;
    const providedHeaders = options && typeof options.headers === 'object' ? options.headers : {};
    const response = await fetch(path, {
      headers: isMultipart ? { ...providedHeaders } : { 'Content-Type': 'application/json', ...providedHeaders },
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

function lockMetricInteractions(durationMs = 320) {
  /** Suppress metric-card click handling briefly to avoid close-click fallthrough. */
  metricInteractionLockUntil = Date.now() + Math.max(0, Number(durationMs) || 0);
}

function metricInteractionsLocked() {
  /** Check whether metric-card interactions are temporarily suppressed. */
  return Date.now() < metricInteractionLockUntil;
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
  const traceability = resolveMetricTraceability(metric, source);
  const relation = describeMetricValueImpact(metric, source);

  const lines = [detail];
  if (traceability) {
    lines.push(`Traceability: ${traceability}`);
  }
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

function resolveMetricTraceability(metric, source) {
  /** Describe exactly how the selected observable is measured or where it should be collected from. */
  const explicitTraceability = String(metric?.tracking || metric?.traceability || '').trim();
  if (explicitTraceability) {
    return explicitTraceability;
  }
  const key = String(metric?.key || '').trim();
  if (source === 'runtime' && key) {
    const runtimeTraceability = String(RUNTIME_METRIC_TRACEABILITY[key] || '').trim();
    if (runtimeTraceability) {
      return runtimeTraceability;
    }
    return 'Computed from live thought-stream and rag-stream telemetry captured inside the current lookback window.';
  }
  if (source === 'correctness') {
    return 'This observable should be populated by an evaluation pipeline and trended across releases, prompts, and model revisions.';
  }
  return '';
}

function describeMetricValueImpact(metric, source) {
  /** Explain how a metric's current value relates to present system health/behavior. */
  const display = String(metric?.display || '').trim();
  const target = String(metric?.target || '').trim();
  const status = String(metric?.status || 'info').toLowerCase();

  if (source === 'correctness' && (!display || display.toLowerCase() === 'track')) {
    return 'No live dashboard value is wired for this observable yet. Use the traceability guidance above to capture it in an evaluation pipeline and trend it over time for drift and correctness.';
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
  els.metricDetailTitle.textContent = 'Observable Detail';
  els.metricDetailBody.textContent = '';
  els.metricDetailPanel.classList.add('hidden');
  clearMetricSelection();
}

function resetMetricTrend() {
  /** Hide trend modal and clear previously rendered chart content. */
  if (!els.metricTrendPanel || !els.metricTrendTitle || !els.metricTrendBody || !els.metricTrendSvg) {
    return;
  }
  els.metricTrendTitle.textContent = 'Observable Trend';
  if (els.metricTrendMeta) {
    els.metricTrendMeta.textContent = 'Double-click a card to graph it over time.';
  }
  els.metricTrendBody.textContent = '';
  els.metricTrendSvg.innerHTML = '';
  els.metricTrendPanel.classList.add('hidden');
  clearMetricSelection();
}

function revealMetricTrendPanel() {
  /** Unhide and scroll the inline trend panel into view so graph actions feel immediate. */
  if (!els.metricTrendPanel) {
    return;
  }
  els.metricTrendPanel.classList.remove('hidden');
  window.requestAnimationFrame(() => {
    els.metricTrendPanel.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
      inline: 'nearest',
    });
  });
}

function chooseMetricHistoryBucketHours(lookbackHours) {
  /** Pick a stable bucket size so trend charts stay readable. */
  if (lookbackHours <= 12) {
    return 1;
  }
  if (lookbackHours <= 48) {
    return 2;
  }
  if (lookbackHours <= 96) {
    return 4;
  }
  return 6;
}

function formatMetricTrendTime(value) {
  /** Format ISO timestamps for compact chart labels. */
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value || '');
  }
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function renderMetricTrendChart(payload) {
  /** Render a lightweight SVG line chart for one observable over time. */
  if (!els.metricTrendSvg) {
    return;
  }
  const width = 860;
  const height = 360;
  const left = 64;
  const right = 26;
  const top = 26;
  const bottom = 54;
  const points = Array.isArray(payload?.points) ? payload.points : [];
  const validPoints = points.filter((point) => Number.isFinite(Number(point?.value)));

  if (!validPoints.length) {
    els.metricTrendSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    els.metricTrendSvg.innerHTML = `
      <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="rgba(7, 16, 28, 0.72)"></rect>
      <rect x="${left}" y="${top}" width="${width - left - right}" height="${height - top - bottom}" rx="8"
        fill="rgba(8, 18, 30, 0.9)" stroke="rgba(136, 168, 196, 0.22)" stroke-dasharray="6 5"></rect>
      <text x="${width / 2}" y="${height / 2 - 8}" text-anchor="middle" fill="#fef3c7" font-size="22" font-weight="700">
        No live series yet
      </text>
      <text x="${width / 2}" y="${height / 2 + 20}" text-anchor="middle" fill="#9bb2c8" font-size="15">
        This observable is defined, but a stored time series is not available yet.
      </text>
    `;
    return;
  }

  const minValue = Math.min(...validPoints.map((point) => Number(point.value)));
  const maxValue = Math.max(...validPoints.map((point) => Number(point.value)));
  const paddedMin = minValue === maxValue ? minValue - 1 : minValue;
  const paddedMax = minValue === maxValue ? maxValue + 1 : maxValue;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const xStep = points.length > 1 ? chartWidth / (points.length - 1) : 0;
  const yFor = (value) => {
    const normalized = (Number(value) - paddedMin) / (paddedMax - paddedMin || 1);
    return top + chartHeight - normalized * chartHeight;
  };
  const xForIndex = (index) => left + xStep * index;

  const lineCommands = [];
  const circles = [];
  points.forEach((point, index) => {
    const value = Number(point?.value);
    if (!Number.isFinite(value)) {
      return;
    }
    const x = xForIndex(index);
    const y = yFor(value);
    lineCommands.push(`${lineCommands.length ? 'L' : 'M'} ${x} ${y}`);
    circles.push(
      `<circle cx="${x}" cy="${y}" r="4.5" fill="#f97316" stroke="#fef3c7" stroke-width="1.5"></circle>`
    );
  });

  const tickValues = [paddedMin, (paddedMin + paddedMax) / 2, paddedMax];
  const yTicks = tickValues
    .map((value) => {
      const y = yFor(value);
      return `
        <line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="rgba(136, 168, 196, 0.18)"></line>
        <text x="${left - 10}" y="${y + 4}" text-anchor="end" fill="#9bb2c8" font-size="13">${value.toFixed(1)}</text>
      `;
    })
    .join('');

  const labelIndexes = Array.from(
    new Set([0, Math.floor((points.length - 1) / 2), Math.max(points.length - 1, 0)])
  );
  const xLabels = labelIndexes
    .map((index) => {
      const point = points[index];
      if (!point) {
        return '';
      }
      const x = xForIndex(index);
      return `<text x="${x}" y="${height - 18}" text-anchor="middle" fill="#9bb2c8" font-size="13">${escapeHtml(
        formatMetricTrendTime(point.at)
      )}</text>`;
    })
    .join('');

  els.metricTrendSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  els.metricTrendSvg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="rgba(7, 16, 28, 0.72)"></rect>
    ${yTicks}
    <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="rgba(136, 168, 196, 0.22)"></line>
    <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"
      stroke="rgba(136, 168, 196, 0.22)"></line>
    <path d="${lineCommands.join(' ')}" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round"
      stroke-linejoin="round"></path>
    ${circles.join('')}
    ${xLabels}
  `;
}

function showMetricTrendFromPayload(metric, payload, source) {
  /** Render trend modal copy and chart content from a history payload. */
  if (
    !els.metricTrendPanel ||
    !els.metricTrendTitle ||
    !els.metricTrendBody ||
    !els.metricTrendMeta ||
    !els.metricTrendSvg
  ) {
    return;
  }

  const label = String(metric?.label || payload?.label || 'Observable').trim() || 'Observable';
  const lookbackHours = Number(payload?.lookback_hours || 24);
  const bucketHours = Number(payload?.bucket_hours || chooseMetricHistoryBucketHours(lookbackHours));
  const points = Array.isArray(payload?.points) ? payload.points : [];
  const livePoints = points.filter((point) => Number.isFinite(Number(point?.value)));
  const usesNumericFallback = livePoints.some((point) => String(point?.display || '').trim() === 'N/A');
  const latestPoint = livePoints.length ? livePoints[livePoints.length - 1] : null;
  const latestDisplay = latestPoint ? String(latestPoint.display || metric?.display || '') : 'N/A';
  const relation = describeMetricValueImpact(
    latestPoint && latestDisplay !== 'N/A' ? { ...metric, display: latestDisplay } : metric,
    source
  );

  els.metricTrendTitle.textContent = `${label} Trend`;
  els.metricTrendMeta.textContent = `${lookbackHours}h lookback in ${bucketHours}h buckets. Drag the lower-right corner to resize.`;
  els.metricTrendBody.textContent = livePoints.length
    ? usesNumericFallback
      ? `Current display is unavailable in this window, so the chart is using the stored numeric fallback behind the metric. ${relation}`
      : `Current: ${latestDisplay}. ${relation}`
    : `No stored live series is available for this observable yet. ${relation}`;
  renderMetricTrendChart(payload);
  revealMetricTrendPanel();
}

function showMetricTrendLoading(metric) {
  /** Show immediate loading state so trend actions feel responsive before history returns. */
  if (
    !els.metricTrendPanel ||
    !els.metricTrendTitle ||
    !els.metricTrendBody ||
    !els.metricTrendMeta ||
    !els.metricTrendSvg
  ) {
    return;
  }
  const label = String(metric?.label || 'Observable').trim() || 'Observable';
  els.metricTrendTitle.textContent = `${label} Trend`;
  els.metricTrendMeta.textContent = 'Loading trend history...';
  els.metricTrendBody.textContent = 'Fetching recent metric history for this observable.';
  els.metricTrendSvg.setAttribute('viewBox', '0 0 860 360');
  els.metricTrendSvg.innerHTML = `
    <rect x="0" y="0" width="860" height="360" rx="8" fill="rgba(7, 16, 28, 0.72)"></rect>
    <rect x="64" y="26" width="770" height="280" rx="8"
      fill="rgba(8, 18, 30, 0.9)" stroke="rgba(136, 168, 196, 0.22)" stroke-dasharray="6 5"></rect>
    <text x="430" y="170" text-anchor="middle" fill="#fef3c7" font-size="22" font-weight="700">Loading…</text>
  `;
  revealMetricTrendPanel();
}

function showMetricTrendError(metric, message) {
  /** Keep the trend shell open and render a clear failure state when history retrieval fails. */
  if (
    !els.metricTrendPanel ||
    !els.metricTrendTitle ||
    !els.metricTrendBody ||
    !els.metricTrendMeta ||
    !els.metricTrendSvg
  ) {
    return;
  }
  const label = String(metric?.label || 'Observable').trim() || 'Observable';
  const errorText = String(message || 'Failed to load trend history.').trim() || 'Failed to load trend history.';
  els.metricTrendTitle.textContent = `${label} Trend`;
  els.metricTrendMeta.textContent = 'Trend history unavailable.';
  els.metricTrendBody.textContent = errorText;
  els.metricTrendSvg.setAttribute('viewBox', '0 0 860 360');
  els.metricTrendSvg.innerHTML = `
    <rect x="0" y="0" width="860" height="360" rx="8" fill="rgba(7, 16, 28, 0.72)"></rect>
    <rect x="64" y="26" width="770" height="280" rx="8"
      fill="rgba(8, 18, 30, 0.9)" stroke="rgba(239, 68, 68, 0.28)" stroke-dasharray="6 5"></rect>
    <text x="430" y="158" text-anchor="middle" fill="#fecaca" font-size="22" font-weight="700">
      Trend unavailable
    </text>
    <text x="430" y="188" text-anchor="middle" fill="#cbd5e1" font-size="15">
      ${escapeHtml(errorText.slice(0, 96))}
    </text>
  `;
  revealMetricTrendPanel();
}

async function showMetricTrend(metric, source) {
  /** Fetch and render a trend graph for the selected metric card. */
  const lookbackHours = 24;
  const bucketHours = chooseMetricHistoryBucketHours(lookbackHours);
  resetMetricDetail();
  showMetricTrendLoading(metric);

  const key = String(metric?.key || '').trim();
  if (!key) {
    showMetricTrendFromPayload(metric, { lookback_hours: lookbackHours, bucket_hours: bucketHours, points: [] }, source);
    return;
  }
  const payload = await api(
    `/api/agent-metrics/history?metric_key=${encodeURIComponent(key)}&lookback_hours=${lookbackHours}&bucket_hours=${bucketHours}`
  );
  showMetricTrendFromPayload(metric, payload, source);
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
        <article class="metric-card ${statusClass}" role="button" tabindex="0" title="Click for detail. Double-click or use Trend for graph.">
          <div class="metric-card-head">
            <div class="metric-label">${escapeHtml(metric.label || '')}</div>
            <button class="metric-trend-btn" type="button" aria-label="Open ${escapeHtml(
              metric.label || 'metric'
            )} trend">
              Trend
            </button>
          </div>
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
    const trendButton = card.querySelector('.metric-trend-btn');
    let clickHandle = null;
    const CLICK_DELAY_MS = 340;
    const select = () => {
      clearMetricSelection();
      card.classList.add('selected');
      showMetricDetail(metric, source);
    };
    const showTrend = () => {
      if (clickHandle !== null) {
        window.clearTimeout(clickHandle);
        clickHandle = null;
      }
      clearMetricSelection();
      card.classList.add('selected');
      showMetricTrend(metric, source).catch((err) => {
        const message = err instanceof Error ? err.message : String(err || 'Failed to load trend history.');
        showMetricTrendError(metric, message);
        setStatus(message);
      });
    };
    card.addEventListener('click', (event) => {
      if (metricInteractionsLocked()) {
        return;
      }
      if (event.detail >= 2) {
        if (clickHandle !== null) {
          window.clearTimeout(clickHandle);
          clickHandle = null;
        }
        showTrend();
        return;
      }
      if (clickHandle !== null) {
        window.clearTimeout(clickHandle);
      }
      clickHandle = window.setTimeout(() => {
        clickHandle = null;
        select();
      }, CLICK_DELAY_MS);
    });
    card.addEventListener('keydown', (event) => {
      if (event.target !== card) {
        return;
      }
      if (metricInteractionsLocked()) {
        event.preventDefault();
        return;
      }
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
        return;
      }
      if (event.key.toLowerCase() === 'g') {
        event.preventDefault();
        showTrend();
      }
    });
    if (trendButton) {
      trendButton.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
      });
      trendButton.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (metricInteractionsLocked()) {
          return;
        }
        showTrend();
      });
      trendButton.addEventListener('keydown', (event) => {
        event.stopPropagation();
      });
    }
  });
}

function renderCorrectnessDriftObservables(payload = null) {
  /** Render correctness/drift KPI cards from backend payload with static fallback definitions. */
  const liveMetrics = Array.isArray(payload?.correctness_metrics) ? payload.correctness_metrics : [];
  const metrics = CORRECTNESS_DRIFT_OBSERVABLES.map((item) => {
    const liveMetric = liveMetrics.find((candidate) => String(candidate?.key || '').trim() === item.key);
    return liveMetric ? { ...item, ...liveMetric } : { ...item, status: 'info' };
  });
  renderMetricCards(els.correctnessGrid, metrics, 'correctness');
}

function renderAgentMetrics(payload, { cached = false } = {}) {
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

  const cacheSuffix = cached ? ', restored from last session' : '';
  els.metricsSampleMeta.textContent = `Sample: ${sampledRuns} runs + ${ragSampledQueries} graph queries (${lookbackHours}h), pairs ${ragPairedComparisons}, updated ${generatedLabel}${cacheSuffix}`;
  const thoughtStorageText = payload?.storage_connected
    ? 'Thought Stream DB: connected'
    : 'Thought Stream DB: degraded (using in-memory sessions)';
  const ragStorageText = payload?.rag_storage_connected
    ? 'RAG Stream DB: connected'
    : 'RAG Stream DB: degraded (using runtime payload only)';
  els.metricsStorageMeta.textContent = `${thoughtStorageText} | ${ragStorageText}`;

  renderMetricCards(els.metricsGrid, metrics, 'runtime');
  renderCorrectnessDriftObservables(payload);
}

async function loadAgentMetrics({ silent = false } = {}) {
  /** Fetch runtime KPI payload from backend and refresh dashboard cards. */
  const payload = await api('/api/agent-metrics?lookback_hours=24');
  writeCachedMetricsPayload(payload);
  renderAgentMetrics(payload);
  metricsLoaded = true;
  if (!silent) {
    setStatus('Observables refreshed.');
  }
}

function hydrateCachedMetrics() {
  /** Restore last successful observables snapshot so the UI keeps prior metric state between runs. */
  const cachedPayload = readCachedMetricsPayload();
  if (!cachedPayload) {
    renderCorrectnessDriftObservables();
    return;
  }
  renderAgentMetrics(cachedPayload, { cached: true });
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

function refreshAdminTestReportFrame() {
  /** Force-refresh the embedded pytest HTML report when Admin opens. */
  if (!els.adminTestReportFrame) {
    return;
  }
  els.adminTestReportFrame.src = `${ADMIN_TEST_REPORT_URL}?ts=${Date.now()}`;
}

async function loadAdminUsers() {
  /** Load saved users for the Admin/Users panel. */
  const payload = await api('/api/admin/users');
  adminUsersCache = Array.isArray(payload?.users) ? payload.users : [];
  renderAdminUsers(adminUsersCache);
}

function renderAdminUsers(users) {
  /** Render the saved users list with clear immediate feedback in the Users subtab. */
  if (!els.adminUserList) {
    return;
  }
  const normalizedUsers = Array.isArray(users) ? users : [];
  syncAdminUserSelection(normalizedUsers);
  if (!normalizedUsers.length) {
    els.adminUserList.classList.add('muted');
    els.adminUserList.textContent = 'No users added yet.';
    return;
  }
  els.adminUserList.classList.remove('muted');
  const items = normalizedUsers.map((item) => {
    const row = document.createElement('div');
    row.className = 'admin-user-row';
    row.dataset.userId = String(item.user_id || '');
    row.setAttribute('role', 'button');
    row.setAttribute('tabindex', '0');
    row.classList.toggle('selected', row.dataset.userId === selectedAdminUserId);
    const label = ADMIN_AUTHORIZATION_LABELS[String(item.authorization_level || '').trim()] || 'User';
    row.textContent = `${item.name} | ${label} | ${item.created_at}`;
    row.addEventListener('click', () => selectAdminUserById(item.user_id, { openDetail: true }));
    row.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') {
        return;
      }
      event.preventDefault();
      selectAdminUserById(item.user_id, { openDetail: true });
    });
    return row;
  });
  els.adminUserList.replaceChildren(...items);
}

function syncAdminUserSelection(users) {
  /** Keep the current-user selector and selected-user text in sync with the user cache. */
  const normalizedUsers = Array.isArray(users) ? users : [];
  if (els.adminUserSelect) {
    els.adminUserSelect.innerHTML = '';
    if (!normalizedUsers.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No users available';
      els.adminUserSelect.appendChild(option);
      els.adminUserSelect.disabled = true;
      selectedAdminUserId = '';
    } else {
      els.adminUserSelect.disabled = false;
      const availableIds = new Set(normalizedUsers.map((item) => String(item.user_id || '').trim()));
      if (!availableIds.has(selectedAdminUserId)) {
        selectedAdminUserId = String(normalizedUsers[0]?.user_id || '').trim();
      }
      for (const item of normalizedUsers) {
        const option = document.createElement('option');
        option.value = String(item.user_id || '').trim();
        option.textContent = item.name || 'Unnamed User';
        els.adminUserSelect.appendChild(option);
      }
      els.adminUserSelect.value = selectedAdminUserId;
    }
  }
  const selectedUser = normalizedUsers.find((item) => String(item.user_id || '').trim() === selectedAdminUserId);
  if (els.adminUserSelectedMeta) {
    if (!selectedUser) {
      els.adminUserSelectedMeta.textContent = 'No user selected.';
    } else {
      const label =
        ADMIN_AUTHORIZATION_LABELS[String(selectedUser.authorization_level || '').trim()] || 'User';
      els.adminUserSelectedMeta.textContent = `Selected: ${selectedUser.name} (${label})`;
    }
  }
}

function selectAdminUserById(userId, { openDetail = false } = {}) {
  /** Select one user from the current list and optionally open the user-detail pop-out. */
  const normalizedUserId = String(userId || '').trim();
  const selectedUser = adminUsersCache.find((item) => String(item.user_id || '').trim() === normalizedUserId);
  if (!selectedUser) {
    return;
  }
  selectedAdminUserId = normalizedUserId;
  renderAdminUsers(adminUsersCache);
  loadAdminUserIntoForm(selectedUser);
  if (openDetail) {
    showAdminUserDetail(selectedUser);
  }
}

function buildAdminUserDetailText(selectedUser) {
  /** Build text-box content listing all current users, with the selected one marked. */
  const normalizedUsers = Array.isArray(adminUsersCache) ? adminUsersCache : [];
  if (!normalizedUsers.length) {
    return 'No users are currently available.';
  }
  const selectedUserId = String(selectedUser?.user_id || '').trim();
  return normalizedUsers
    .map((item, index) => {
      const label = ADMIN_AUTHORIZATION_LABELS[String(item.authorization_level || '').trim()] || 'User';
      const marker = String(item.user_id || '').trim() === selectedUserId ? '>>' : '  ';
      return `${marker} ${index + 1}. ${item.name} | ${label} | ${item.created_at}`;
    })
    .join('\n');
}

function showAdminUserDetail(selectedUser) {
  /** Open a pop-out text box listing all current users and identify the selected user. */
  if (
    !els.adminUserDetailPanel ||
    !els.adminUserDetailTitle ||
    !els.adminUserDetailMeta ||
    !els.adminUserDetailBody
  ) {
    return;
  }
  const selectedName = String(selectedUser?.name || '').trim() || 'Unknown User';
  const selectedLabel =
    ADMIN_AUTHORIZATION_LABELS[String(selectedUser?.authorization_level || '').trim()] || 'User';
  els.adminUserDetailTitle.textContent = 'Current Users';
  els.adminUserDetailMeta.textContent = `Selected user: ${selectedName} (${selectedLabel})`;
  els.adminUserDetailBody.value = buildAdminUserDetailText(selectedUser);
  els.adminUserDetailPanel.classList.remove('hidden');
}

function hideAdminUserDetail() {
  /** Close the admin-user detail pop-out. */
  if (
    !els.adminUserDetailPanel ||
    !els.adminUserDetailTitle ||
    !els.adminUserDetailMeta ||
    !els.adminUserDetailBody
  ) {
    return;
  }
  els.adminUserDetailTitle.textContent = 'Current Users';
  els.adminUserDetailMeta.textContent = 'Selected user:';
  els.adminUserDetailBody.value = '';
  els.adminUserDetailPanel.classList.add('hidden');
}

function setAdminUserFeedback(message, tone = 'info') {
  /** Render local feedback in the Admin/Users panel so the button never appears inert. */
  if (!els.adminUserFeedback) {
    return;
  }
  const nextMessage = String(message || '').trim();
  els.adminUserFeedback.textContent = nextMessage || ' ';
  els.adminUserFeedback.classList.remove('muted', 'error', 'success');
  if (!nextMessage) {
    els.adminUserFeedback.classList.add('muted');
    return;
  }
  if (tone === 'error') {
    els.adminUserFeedback.classList.add('error');
    return;
  }
  if (tone === 'success') {
    els.adminUserFeedback.classList.add('success');
    return;
  }
  els.adminUserFeedback.classList.add('muted');
}

function setAdminUserCreateOpen(open) {
  /** Toggle the expandable add-user form. */
  const nextOpen = !!open;
  if (els.adminUserCreatePanel) {
    els.adminUserCreatePanel.classList.toggle('hidden', !nextOpen);
  }
  if (els.adminAddUserBtn) {
    els.adminAddUserBtn.textContent = nextOpen ? 'Close Add User' : 'Add User';
  }
  if (nextOpen && els.adminUserFirstName) {
    els.adminUserFirstName.focus();
  }
  syncAdminUserSaveButton();
}

function resetAdminUserForm() {
  /** Reset add-user form fields back to defaults. */
  editingAdminUserId = '';
  if (els.adminUserFirstName) {
    els.adminUserFirstName.value = '';
  }
  if (els.adminUserLastName) {
    els.adminUserLastName.value = '';
  }
  if (els.adminUserAuthorization) {
    els.adminUserAuthorization.value = 'user';
  }
  syncAdminUserSaveButton();
}

function syncAdminUserSaveButton() {
  /** Keep the save button text aligned to create vs edit mode. */
  if (!els.adminSaveUserBtn) {
    return;
  }
  els.adminSaveUserBtn.textContent = editingAdminUserId ? 'Save Changes' : 'Save User';
}

function loadAdminUserIntoForm(user) {
  /** Populate the add-user form with the selected user's current values. */
  if (!user) {
    return;
  }
  editingAdminUserId = String(user.user_id || '').trim();
  if (els.adminUserFirstName) {
    els.adminUserFirstName.value = String(user.first_name || '').trim();
  }
  if (els.adminUserLastName) {
    els.adminUserLastName.value = String(user.last_name || '').trim();
  }
  if (els.adminUserAuthorization) {
    els.adminUserAuthorization.value = String(user.authorization_level || 'user').trim() || 'user';
  }
  setAdminUserCreateOpen(true);
  setAdminUserFeedback(`Editing ${user.name}. Update the fields and save changes.`, 'info');
}

function highlightAdminUserRow(userId) {
  /** Highlight the saved user row so the add action is obvious to the operator. */
  const normalizedUserId = String(userId || '').trim();
  if (!normalizedUserId || !els.adminUserList) {
    return;
  }
  const rows = Array.from(els.adminUserList.querySelectorAll('.admin-user-row'));
  rows.forEach((row) => row.classList.remove('fresh'));
  const match = rows.find((row) => String(row.dataset.userId || '').trim() === normalizedUserId);
  if (!match) {
    return;
  }
  match.classList.add('fresh');
  match.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  window.setTimeout(() => match.classList.remove('fresh'), 2200);
}

async function addAdminUser() {
  /** Create or update one user and refresh the visible list. */
  const firstName = String(els.adminUserFirstName?.value || '').trim();
  const lastName = String(els.adminUserLastName?.value || '').trim();
  const authorizationLevel = String(els.adminUserAuthorization?.value || '').trim() || 'user';
  const isEditing = !!editingAdminUserId;
  if (!firstName || !lastName) {
    if (!firstName && els.adminUserFirstName) {
      els.adminUserFirstName.focus();
    } else if (els.adminUserLastName) {
      els.adminUserLastName.focus();
    }
    setAdminUserFeedback('Enter both a first name and a last name before saving a user.', 'error');
    setStatus('Enter a user name before adding a user.');
    return;
  }
  if (els.adminSaveUserBtn) {
    els.adminSaveUserBtn.disabled = true;
    els.adminSaveUserBtn.textContent = isEditing ? 'Saving Changes...' : 'Saving...';
  }
  if (els.adminCancelUserBtn) {
    els.adminCancelUserBtn.disabled = true;
  }
  if (els.adminAddUserBtn) {
    els.adminAddUserBtn.disabled = true;
  }
  setActiveTab('admin');
  setActiveAdminSubtab('users');
  setAdminUserFeedback(isEditing ? 'Saving user changes...' : 'Saving user...', 'info');
  try {
    const createdUser = await api('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify({
        user_id: editingAdminUserId || null,
        first_name: firstName,
        last_name: lastName,
        authorization_level: authorizationLevel,
      }),
    });
    adminUsersCache = [createdUser, ...adminUsersCache.filter((item) => item.user_id !== createdUser.user_id)];
    renderAdminUsers(adminUsersCache);
    highlightAdminUserRow(createdUser.user_id);
    resetAdminUserForm();
    await loadAdminUsers();
    selectedAdminUserId = String(createdUser.user_id || '').trim();
    renderAdminUsers(adminUsersCache);
    highlightAdminUserRow(createdUser.user_id);
    setAdminUserFeedback(
      `${createdUser.name} ${isEditing ? 'updated' : 'saved'} with ${ADMIN_AUTHORIZATION_LABELS[authorizationLevel] || 'User'} authorization.`,
      'success'
    );
    setAdminUserCreateOpen(false);
    setStatus(
      `${isEditing ? 'User updated' : 'User added'} with ${ADMIN_AUTHORIZATION_LABELS[authorizationLevel] || 'User'} authorization.`
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to save user.');
    setAdminUserFeedback(message, 'error');
    throw err;
  } finally {
    if (els.adminSaveUserBtn) {
      els.adminSaveUserBtn.disabled = false;
      syncAdminUserSaveButton();
    }
    if (els.adminCancelUserBtn) {
      els.adminCancelUserBtn.disabled = false;
    }
    if (els.adminAddUserBtn) {
      els.adminAddUserBtn.disabled = false;
    }
  }
}

async function removeAdminUser() {
  /** Permanently remove the currently selected user and refresh the visible list. */
  const userId = String(selectedAdminUserId || '').trim();
  const selectedUser = adminUsersCache.find((item) => String(item.user_id || '').trim() === userId);
  if (!userId || !selectedUser) {
    setAdminUserFeedback('Select a user before removing one.', 'error');
    setStatus('Select a user before removing one.');
    return;
  }
  const confirmed = window.confirm(`Permanently remove ${selectedUser.name}? This cannot be undone.`);
  if (!confirmed) {
    setAdminUserFeedback(`Removal canceled for ${selectedUser.name}.`, 'info');
    return;
  }
  if (els.adminRemoveUserBtn) {
    els.adminRemoveUserBtn.disabled = true;
    els.adminRemoveUserBtn.textContent = 'Removing...';
  }
  if (els.adminGetUsersBtn) {
    els.adminGetUsersBtn.disabled = true;
  }
  if (els.adminRefreshUsersBtn) {
    els.adminRefreshUsersBtn.disabled = true;
  }
  setAdminUserFeedback(`Removing ${selectedUser.name}...`, 'info');
  try {
    await api(`/api/admin/users/${encodeURIComponent(userId)}`, { method: 'DELETE' });
    adminUsersCache = adminUsersCache.filter((item) => String(item.user_id || '').trim() !== userId);
    if (editingAdminUserId === userId) {
      resetAdminUserForm();
      setAdminUserCreateOpen(false);
    }
    hideAdminUserDetail();
    selectedAdminUserId = '';
    renderAdminUsers(adminUsersCache);
    await loadAdminUsers();
    setAdminUserFeedback(`${selectedUser.name} was permanently removed.`, 'success');
    setStatus('User removed.');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to remove user.');
    setAdminUserFeedback(message, 'error');
    throw err;
  } finally {
    if (els.adminRemoveUserBtn) {
      els.adminRemoveUserBtn.disabled = false;
      els.adminRemoveUserBtn.textContent = 'Remove User';
    }
    if (els.adminGetUsersBtn) {
      els.adminGetUsersBtn.disabled = false;
    }
    if (els.adminRefreshUsersBtn) {
      els.adminRefreshUsersBtn.disabled = false;
    }
  }
}

function syncAdminPersonaLlmOptions() {
  /** Mirror the main LLM selector options into the persona editor. */
  if (!els.adminPersonaLlm) {
    return;
  }
  const currentValue = String(els.adminPersonaLlm.value || '').trim();
  const fallbackValue = encodeLlmOption('openai', 'gpt-5.2');
  const sourceOptions = Array.from(els.llmSelect?.options || []);
  els.adminPersonaLlm.innerHTML = '';
  if (!sourceOptions.length) {
    const fallback = document.createElement('option');
    fallback.value = fallbackValue;
    fallback.textContent = 'ChatGPT - gpt-5.2';
    els.adminPersonaLlm.appendChild(fallback);
    els.adminPersonaLlm.value = currentValue || fallbackValue;
    return;
  }
  for (const item of sourceOptions) {
    const option = document.createElement('option');
    option.value = item.value;
    option.textContent = item.textContent || item.value;
    option.disabled = !!item.disabled;
    option.title = item.title || '';
    els.adminPersonaLlm.appendChild(option);
  }
  const values = Array.from(els.adminPersonaLlm.options).map((item) => item.value);
  els.adminPersonaLlm.value = values.includes(currentValue)
    ? currentValue
    : values.find((value, index) => !els.adminPersonaLlm.options[index]?.disabled) || values[0] || fallbackValue;
}

function syncAdminPersonaPromptTemplateOptions() {
  /** Populate section prompt-template dropdowns and preserve current selection when possible. */
  const selectors = [
    { section: 'system', element: els.adminPersonaSystemPromptTemplateSelect },
    { section: 'assistant', element: els.adminPersonaAssistantPromptTemplateSelect },
    { section: 'context', element: els.adminPersonaContextPromptTemplateSelect },
  ].filter((item) => item.element instanceof HTMLSelectElement);
  if (!selectors.length) {
    return;
  }
  for (const selector of selectors) {
    const currentValue = String(selector.element.value || '').trim();
    selector.element.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose Prompt';
    placeholder.dataset.tone = 'neutral';
    selector.element.appendChild(placeholder);

    const scopedTemplates = adminPersonaPromptTemplatesCache.filter(
      (item) => resolveAdminPersonaPromptTarget(item?.key, item?.file_name) === selector.section
    );
    const candidates = scopedTemplates.length ? scopedTemplates : adminPersonaPromptTemplatesCache;
    for (const item of candidates) {
      const key = String(item?.key || '').trim();
      if (!key) {
        continue;
      }
      const tone = adminPersonaPromptTemplateTone(item, selector.section);
      const option = document.createElement('option');
      option.value = key;
      option.textContent = adminPersonaPromptTemplateLabel(item, tone);
      option.dataset.tone = tone;
      option.style.color = adminPersonaPromptTemplateColor(tone);
      selector.element.appendChild(option);
    }
    const values = Array.from(selector.element.options).map((item) => String(item.value || '').trim());
    selector.element.value = values.includes(currentValue) ? currentValue : '';
    syncAdminPersonaPromptTemplateSelectColor(selector.element);
  }
  syncAdminPersonaSelectedPromptTemplateKey();
}

function adminPersonaPromptTemplateTone(template, sectionFallback = 'assistant') {
  /** Classify one built-in prompt template for dropdown coloring and suffixing. */
  const key = String(template?.key || '').trim().toLowerCase();
  const fileName = String(template?.file_name || '').trim().toLowerCase();
  const section = String(sectionFallback || '').trim().toLowerCase();
  const combined = `${key} ${fileName}`;
  if (combined.endsWith('_sys') || combined.includes('_system') || combined.includes(' system')) {
    return 'system';
  }
  if (combined.endsWith('_user') || combined.includes('_user') || combined.includes(' user')) {
    return 'user';
  }
  if (combined.includes('context')) {
    return 'context';
  }
  if (section === 'context') {
    return 'context';
  }
  if (section === 'system') {
    return 'system';
  }
  return 'user';
}

function adminPersonaPromptTemplateLabel(template, tone = 'user') {
  /** Render one dropdown option label with expected suffix convention when applicable. */
  const key = String(template?.key || '').trim();
  if (!key) {
    return '';
  }
  if (tone === 'system' && !key.toLowerCase().endsWith('_sys')) {
    return `${key}_sys`;
  }
  if (tone === 'user' && !key.toLowerCase().endsWith('_user')) {
    return `${key}_user`;
  }
  return key;
}

function adminPersonaPromptTemplateColor(tone = 'neutral') {
  /** Return one color for prompt-template tone rendering. */
  const normalizedTone = String(tone || '').trim().toLowerCase();
  if (normalizedTone === 'system') {
    return '#ff4040';
  }
  if (normalizedTone === 'user') {
    return '#22c55e';
  }
  if (normalizedTone === 'context') {
    return '#b58900';
  }
  return '#dbeafe';
}

function syncAdminPersonaPromptTemplateSelectColor(selectElement) {
  /** Color the selected dropdown text using the template tone. */
  if (!(selectElement instanceof HTMLSelectElement)) {
    return;
  }
  const selectedOption = selectElement.options[selectElement.selectedIndex] || null;
  const tone = String(selectedOption?.dataset?.tone || 'neutral').trim().toLowerCase();
  selectElement.style.color = adminPersonaPromptTemplateColor(tone);
}

function setAdminPersonaPromptTemplateSelectionForKey(promptTemplateKey = '') {
  /** Select one tracked prompt-template key in the matching section dropdown. */
  const normalizedKey = String(promptTemplateKey || '').trim();
  adminPersonaSelectedPromptTemplateKey = normalizedKey;
  const map = {
    system: els.adminPersonaSystemPromptTemplateSelect,
    assistant: els.adminPersonaAssistantPromptTemplateSelect,
    context: els.adminPersonaContextPromptTemplateSelect,
  };
  const selectors = Object.values(map).filter((item) => item instanceof HTMLSelectElement);
  for (const item of selectors) {
    item.value = '';
  }
  if (!normalizedKey) {
    for (const item of selectors) {
      syncAdminPersonaPromptTemplateSelectColor(item);
    }
    return;
  }
  const targetSection = resolveAdminPersonaPromptTarget(normalizedKey);
  const targetSelect = map[targetSection];
  if (targetSelect && Array.from(targetSelect.options).some((item) => String(item.value || '').trim() === normalizedKey)) {
    targetSelect.value = normalizedKey;
  } else {
    const fallback = selectors.find((item) =>
      Array.from(item.options).some((option) => String(option.value || '').trim() === normalizedKey)
    );
    if (fallback) {
      fallback.value = normalizedKey;
    }
  }
  for (const item of selectors) {
    syncAdminPersonaPromptTemplateSelectColor(item);
  }
}

function syncAdminPersonaSelectedPromptTemplateKey() {
  /** Sync selected prompt-template key from section dropdown selections. */
  const values = [
    String(els.adminPersonaSystemPromptTemplateSelect?.value || '').trim(),
    String(els.adminPersonaAssistantPromptTemplateSelect?.value || '').trim(),
    String(els.adminPersonaContextPromptTemplateSelect?.value || '').trim(),
  ];
  adminPersonaSelectedPromptTemplateKey = values.find((item) => !!item) || '';
}

function getAdminPersonaPromptTemplateSelect(targetSection = 'assistant') {
  /** Resolve the prompt-template select element for one Persona prompt section. */
  const section = String(targetSection || '').trim().toLowerCase();
  const map = {
    system: els.adminPersonaSystemPromptTemplateSelect,
    assistant: els.adminPersonaAssistantPromptTemplateSelect,
    context: els.adminPersonaContextPromptTemplateSelect,
  };
  return map[section] || map.assistant || null;
}

function hideAdminPersonaPromptTemplateDropdowns() {
  /** Hide all Persona prompt-template dropdown controls. */
  const selectors = [
    els.adminPersonaSystemPromptTemplateSelect,
    els.adminPersonaAssistantPromptTemplateSelect,
    els.adminPersonaContextPromptTemplateSelect,
  ];
  for (const selector of selectors) {
    if (!(selector instanceof HTMLSelectElement)) {
      continue;
    }
    selector.classList.add('hidden');
  }
}

async function toggleAdminPersonaPromptTemplateDropdown(targetSection = 'assistant') {
  /** Toggle one Persona prompt-template dropdown and keep others closed. */
  const selectElement = getAdminPersonaPromptTemplateSelect(targetSection);
  if (!(selectElement instanceof HTMLSelectElement)) {
    return;
  }
  if (!adminPersonaPromptPanelActive) {
    setAdminPersonaFeedback('Open Prompt first before choosing a template.', 'error');
    return;
  }
  if (!adminPersonaPromptTemplatesCache.length) {
    await loadAdminPersonaPromptTemplates();
  } else {
    syncAdminPersonaPromptTemplateOptions();
  }
  const willShow = selectElement.classList.contains('hidden');
  hideAdminPersonaPromptTemplateDropdowns();
  if (!willShow) {
    return;
  }
  selectElement.value = '';
  syncAdminPersonaPromptTemplateSelectColor(selectElement);
  selectElement.classList.remove('hidden');
  selectElement.focus();
}

async function loadAdminPersonaPromptTemplates() {
  /** Load the built-in runtime prompts so the Persona editor can seed from them. */
  const payload = await api('/api/admin/personas/prompts');
  adminPersonaPromptTemplatesCache = Array.isArray(payload?.prompts) ? payload.prompts : [];
  syncAdminPersonaPromptTemplateOptions();
}

function parseLegacyPersonaPromptSections(legacyPrompts) {
  /** Parse legacy persona prompt text into system/assistant/context sections. */
  const text = String(legacyPrompts || '').trim();
  if (!text) {
    return { system: '', assistant: '', context: '' };
  }
  const buckets = { system: [], assistant: [], context: [] };
  let currentKey = 'system';
  let sawMarker = false;
  for (const rawLine of text.split('\n')) {
    const line = String(rawLine || '').replace(/\r$/, '');
    const match = line.match(/^\s*(system|assistant|context)\s*:\s*(.*)$/i);
    if (match) {
      currentKey = String(match[1] || '').trim().toLowerCase();
      if (!Object.prototype.hasOwnProperty.call(buckets, currentKey)) {
        currentKey = 'system';
      }
      sawMarker = true;
      const remainder = String(match[2] || '').trim();
      if (remainder) {
        buckets[currentKey].push(remainder);
      }
      continue;
    }
    buckets[currentKey].push(line);
  }
  if (!sawMarker) {
    return { system: text, assistant: '', context: '' };
  }
  return {
    system: buckets.system.join('\n').trim(),
    assistant: buckets.assistant.join('\n').trim(),
    context: buckets.context.join('\n').trim(),
  };
}

function resolveAdminPersonaPromptTarget(promptTemplateKey = '', promptFileName = '') {
  /** Resolve which prompt section a built-in prompt template should map into. */
  const combined = `${String(promptTemplateKey || '')} ${String(promptFileName || '')}`.toLowerCase();
  if (combined.includes('context')) {
    return 'context';
  }
  if (combined.includes('_system') || combined.endsWith('system') || combined.startsWith('system_')) {
    return 'system';
  }
  return 'assistant';
}

function normalizeAdminPersonaPromptSections(promptSections, legacyPrompts = '', promptTemplateKey = '', promptFileName = '') {
  /** Normalize persona prompt sections with fallback support for legacy prompt strings + template-key routing. */
  const source = promptSections && typeof promptSections === 'object' ? promptSections : {};
  const normalized = {
    system: String(source.system || '').trim(),
    assistant: String(source.assistant || '').trim(),
    context: String(source.context || '').trim(),
  };
  if (normalized.system || normalized.assistant || normalized.context) {
    return normalized;
  }
  const parsedLegacy = parseLegacyPersonaPromptSections(legacyPrompts);
  if (parsedLegacy.assistant || parsedLegacy.context) {
    return parsedLegacy;
  }
  const legacyText = String(legacyPrompts || '').trim();
  if (!legacyText) {
    return parsedLegacy;
  }
  const target = resolveAdminPersonaPromptTarget(promptTemplateKey, promptFileName);
  if (target === 'context') {
    return { system: '', assistant: '', context: legacyText };
  }
  if (target === 'assistant') {
    return { system: '', assistant: legacyText, context: '' };
  }
  return { system: legacyText, assistant: '', context: '' };
}

function composeLegacyPersonaPromptSections(promptSections) {
  /** Compose deterministic legacy prompt text from system/assistant/context sections. */
  const normalized = normalizeAdminPersonaPromptSections(promptSections, '');
  const parts = [];
  if (normalized.system) {
    parts.push(`System:\n${normalized.system}`);
  }
  if (normalized.assistant) {
    parts.push(`Assistant:\n${normalized.assistant}`);
  }
  if (normalized.context) {
    parts.push(`Context:\n${normalized.context}`);
  }
  return parts.join('\n\n').trim();
}

function setAdminPersonaPromptSectionsInForm(promptSections) {
  /** Write normalized prompt sections into the Persona form fields. */
  const normalized = normalizeAdminPersonaPromptSections(promptSections, '');
  if (els.adminPersonaSystemPrompt) {
    els.adminPersonaSystemPrompt.value = normalized.system;
  }
  if (els.adminPersonaAssistantPrompt) {
    els.adminPersonaAssistantPrompt.value = normalized.assistant;
  }
  if (els.adminPersonaContextPrompt) {
    els.adminPersonaContextPrompt.value = normalized.context;
  }
}

function getAdminPersonaPromptSectionsFromForm() {
  /** Read prompt sections from the Persona form fields. */
  return normalizeAdminPersonaPromptSections({
    system: String(els.adminPersonaSystemPrompt?.value || '').trim(),
    assistant: String(els.adminPersonaAssistantPrompt?.value || '').trim(),
    context: String(els.adminPersonaContextPrompt?.value || '').trim(),
  });
}

function setAdminPersonaPromptSectionsInModal(promptSections) {
  /** Write normalized prompt sections into the Persona prompt modal fields. */
  const normalized = normalizeAdminPersonaPromptSections(promptSections, '');
  if (els.adminPersonaPromptModalSystem) {
    els.adminPersonaPromptModalSystem.value = normalized.system;
  }
  if (els.adminPersonaPromptModalAssistant) {
    els.adminPersonaPromptModalAssistant.value = normalized.assistant;
  }
  if (els.adminPersonaPromptModalContext) {
    els.adminPersonaPromptModalContext.value = normalized.context;
  }
}

function getAdminPersonaPromptSectionsFromModal() {
  /** Read prompt sections from the Persona prompt modal fields. */
  return normalizeAdminPersonaPromptSections({
    system: String(els.adminPersonaPromptModalSystem?.value || '').trim(),
    assistant: String(els.adminPersonaPromptModalAssistant?.value || '').trim(),
    context: String(els.adminPersonaPromptModalContext?.value || '').trim(),
  });
}

function adminPersonaPromptSectionsHaveContent(promptSections) {
  /** Return whether any persona prompt section has content. */
  const normalized = normalizeAdminPersonaPromptSections(promptSections, '');
  return Boolean(normalized.system || normalized.assistant || normalized.context);
}

function loadSelectedAdminPersonaPromptTemplate(targetSection = 'assistant') {
  /** Add one selected built-in prompt template into one specific Persona prompt section field. */
  const section = String(targetSection || '').trim().toLowerCase();
  const selectBySection = {
    system: els.adminPersonaSystemPromptTemplateSelect,
    assistant: els.adminPersonaAssistantPromptTemplateSelect,
    context: els.adminPersonaContextPromptTemplateSelect,
  };
  const fieldBySection = {
    system: els.adminPersonaSystemPrompt,
    assistant: els.adminPersonaAssistantPrompt,
    context: els.adminPersonaContextPrompt,
  };
  const labelBySection = {
    system: 'System Prompt',
    assistant: 'Assistant Prompt',
    context: 'Context Prompt',
  };
  const selectElement = selectBySection[section] || selectBySection.assistant;
  const field = fieldBySection[section] || fieldBySection.assistant;
  const sectionLabel = labelBySection[section] || labelBySection.assistant;
  const selectedKey = String(selectElement?.value || '').trim();
  if (!selectedKey) {
    setAdminPersonaFeedback(`Select a ${sectionLabel} template from the dropdown first.`, 'error');
    return;
  }
  const selectedTemplate = adminPersonaPromptTemplatesCache.find(
    (item) => String(item?.key || '').trim() === selectedKey
  );
  if (!selectedTemplate) {
    setAdminPersonaFeedback('The selected prompt template was not found.', 'error');
    return;
  }
  const content = String(selectedTemplate.content || '').trim();
  if (field) {
    const current = String(field.value || '').trim();
    field.value = current ? `${current}\n\n${content}` : content;
  }
  setAdminPersonaPromptTemplateSelectionForKey(selectedKey);
  setAdminPersonaFeedback(`Added built-in prompt ${selectedKey} into ${sectionLabel}.`, 'info');
}

async function saveAdminPersonaPromptSection(targetSection = 'assistant') {
  /** Persist Persona prompt edits from one prompt window while keeping the Persona editor open. */
  const section = String(targetSection || '').trim().toLowerCase();
  const fieldBySection = {
    system: els.adminPersonaSystemPrompt,
    assistant: els.adminPersonaAssistantPrompt,
    context: els.adminPersonaContextPrompt,
  };
  const labelBySection = {
    system: 'System Prompt',
    assistant: 'Assistant Prompt',
    context: 'Context Prompt',
  };
  const field = fieldBySection[section] || fieldBySection.assistant;
  const label = labelBySection[section] || labelBySection.assistant;
  const text = String(field?.value || '').trim();
  if (!text) {
    field?.focus();
    setAdminPersonaFeedback(`Enter ${label} text before saving.`, 'error');
    setStatus(`Enter ${label} text before saving.`);
    return;
  }
  setAdminPersonaFeedback(`Saving ${label}...`, 'info');
  await addAdminPersona({ closeOnSuccess: false });
  setAdminPersonaFeedback(`${label} saved.`, 'success');
  setStatus(`${label} saved.`);
}

function getAdminPersonaRagLabel(ragKey) {
  /** Resolve one persona RAG key to a human-friendly label. */
  const normalizedKey = String(
    (ragKey && typeof ragKey === 'object' ? ragKey.key : ragKey) || ''
  ).trim();
  const match = adminPersonaRagOptionsCache.find((item) => String(item?.key || '').trim() === normalizedKey);
  return String(match?.label || normalizedKey || 'Unknown RAG');
}

function getAdminPersonaToolLabel(toolKey) {
  /** Resolve one persona MCP tool key to a human-friendly label. */
  const normalizedKey = String(
    (toolKey && typeof toolKey === 'object' ? toolKey.key : toolKey) || ''
  ).trim();
  const match = adminPersonaToolOptionsCache.find((item) => String(item?.key || '').trim() === normalizedKey);
  return String(match?.label || normalizedKey || 'Unknown Tool');
}

function normalizeAdminPersonaRagBinding(ragValue) {
  /** Normalize a saved RAG binding into the frontend shape used by the persona editor. */
  if (ragValue && typeof ragValue === 'object' && !Array.isArray(ragValue)) {
    const key = String(ragValue.key || '').trim();
    if (!key) {
      return null;
    }
    return {
      key,
      enabled: ragValue.enabled !== false,
    };
  }
  const key = String(ragValue || '').trim();
  if (!key) {
    return null;
  }
  return {
    key,
    enabled: true,
  };
}

function normalizeAdminPersonaToolBinding(toolValue) {
  /** Normalize a saved MCP tool binding into the frontend shape used by the persona editor. */
  if (toolValue && typeof toolValue === 'object' && !Array.isArray(toolValue)) {
    const key = String(toolValue.key || '').trim();
    if (!key) {
      return null;
    }
    return {
      key,
      enabled: toolValue.enabled !== false,
    };
  }
  const key = String(toolValue || '').trim();
  if (!key) {
    return null;
  }
  return {
    key,
    enabled: true,
  };
}

function getActiveAdminPersonaGraphConfig() {
  /** Resolve the active persona graph configuration from the open form or current saved selection. */
  const formOpen = !els.adminPersonaCreatePanel?.classList.contains('hidden');
  const selectedPersona = adminPersonasCache.find(
    (item) => String(item?.persona_id || '').trim() === selectedAdminPersonaId
  );

  if (formOpen) {
    const draftName = String(els.adminPersonaName?.value || '').trim();
    const { provider, model } = decodeLlmOption(els.adminPersonaLlm?.value || '');
    const draftSequence = Array.isArray(adminPersonaRagSequence)
      ? adminPersonaRagSequence
          .map((item) => normalizeAdminPersonaRagBinding(item))
          .filter((item) => !!item)
      : [];
    const draftToolSequence = Array.isArray(adminPersonaToolSequence)
      ? adminPersonaToolSequence
          .map((item) => normalizeAdminPersonaToolBinding(item))
          .filter((item) => !!item)
      : [];
    if (draftName || editingAdminPersonaId || draftSequence.length || draftToolSequence.length) {
      return {
        source: 'draft',
        name: draftName || (selectedPersona?.name ? `${selectedPersona.name} (draft)` : 'Unsaved Persona'),
        llmProvider: provider,
        llmModel: model,
        ragSequence: draftSequence,
        toolSequence: draftToolSequence,
      };
    }
  }

  if (!selectedPersona) {
    return null;
  }

  return {
    source: 'saved',
    name: String(selectedPersona.name || '').trim() || 'Saved Persona',
    llmProvider: String(selectedPersona.llm_provider || '').trim().toLowerCase(),
    llmModel: String(selectedPersona.llm_model || '').trim(),
    ragSequence: Array.isArray(selectedPersona.rag_sequence)
      ? selectedPersona.rag_sequence
          .map((item) => normalizeAdminPersonaRagBinding(item))
          .filter((item) => !!item)
      : [],
    toolSequence: Array.isArray(selectedPersona.tool_sequence)
      ? selectedPersona.tool_sequence
          .map((item) => normalizeAdminPersonaToolBinding(item))
          .filter((item) => !!item)
      : [],
  };
}

function renderAdminPersonaGraphMeta() {
  /** Show which persona and enabled RAG chain will be used for graph-only questions. */
  if (!els.adminPersonaGraphMeta) {
    return;
  }
  const config = getActiveAdminPersonaGraphConfig();
  if (!config) {
    els.adminPersonaGraphMeta.textContent = 'No active persona graph configuration.';
    return;
  }
  const enabledRags = config.ragSequence.filter((item) => item.enabled);
  const enabledSummary = enabledRags.length
    ? enabledRags.map((item) => getAdminPersonaRagLabel(item)).join(' -> ')
    : 'No enabled RAG steps';
  const enabledTools = (Array.isArray(config.toolSequence) ? config.toolSequence : []).filter((item) => item.enabled);
  const enabledToolsSummary = enabledTools.length
    ? enabledTools.map((item) => getAdminPersonaToolLabel(item)).join(' -> ')
    : 'No enabled MCP tools';
  const providerLabel = config.llmProvider === 'ollama' ? 'Ollama' : 'ChatGPT';
  els.adminPersonaGraphMeta.textContent =
    `${config.name} [${config.source}] | ${providerLabel} (${config.llmModel || 'unconfigured'}) | `
    + `${enabledSummary} | ${enabledToolsSummary}`;
}

function renderAdminPersonaStoredGraphSession(persona) {
  /** Restore the last saved graph-only persona question and answer from persistence. */
  if (!els.adminPersonaGraphAnswer) {
    return;
  }
  const question = String(persona?.last_graph_question || '').trim();
  const answer = String(persona?.last_graph_answer || '').trim();
  if (!question || !answer) {
    els.adminPersonaGraphAnswer.value =
      "Ask a graph question to test the active persona's Graph RAG configuration.";
    return;
  }
  if (els.adminPersonaGraphQuestion) {
    els.adminPersonaGraphQuestion.value = question;
  }
  els.adminPersonaGraphAnswer.value = answer;
}

function setAdminPersonaGraphProcessing(active) {
  /** Toggle the local progress indicator and clock for persona graph-only questions. */
  const isActive = !!active;
  if (els.adminPersonaGraphProgress) {
    els.adminPersonaGraphProgress.classList.toggle('hidden', !isActive);
  }
  const controlsEnabled = !isActive && adminPersonaRagPanelActive;
  if (els.adminPersonaGraphAskBtn) {
    els.adminPersonaGraphAskBtn.disabled = !controlsEnabled;
    els.adminPersonaGraphAskBtn.textContent = isActive ? 'Asking...' : 'Ask Graph';
  }
  if (els.adminPersonaGraphClearBtn) {
    els.adminPersonaGraphClearBtn.disabled = !controlsEnabled;
  }
  if (els.adminPersonaGraphQuestion) {
    els.adminPersonaGraphQuestion.disabled = !controlsEnabled;
  }
  if (!els.adminPersonaGraphClock) {
    return;
  }
  if (isActive) {
    startClock('adminPersonaGraph', els.adminPersonaGraphClock);
    return;
  }
  stopClock('adminPersonaGraph');
  els.adminPersonaGraphClock.textContent = '0.0s';
}

function syncAdminPersonaRagOptions() {
  /** Mirror current backend RAG chain options into the persona editor selector. */
  if (!els.adminPersonaRagSelect) {
    return;
  }
  const currentValue = String(els.adminPersonaRagSelect.value || '').trim();
  els.adminPersonaRagSelect.innerHTML = '';
  if (!adminPersonaRagOptionsCache.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No RAG steps available';
    els.adminPersonaRagSelect.appendChild(option);
    els.adminPersonaRagSelect.disabled = true;
    return;
  }
  els.adminPersonaRagSelect.disabled = !adminPersonaRagPanelActive;
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Choose one RAG step';
  els.adminPersonaRagSelect.appendChild(placeholder);
  for (const item of adminPersonaRagOptionsCache) {
    const option = document.createElement('option');
    option.value = String(item.key || '').trim();
    option.textContent = item.label || item.key || 'Unknown RAG';
    option.title = item.description || '';
    option.disabled = item.available === false;
    els.adminPersonaRagSelect.appendChild(option);
  }
  const values = Array.from(els.adminPersonaRagSelect.options).map((item) => item.value);
  els.adminPersonaRagSelect.value = values.includes(currentValue) ? currentValue : '';
}

function renderAdminPersonaRagSequence() {
  /** Render the ordered RAG chain attached to the persona currently being edited. */
  if (!els.adminPersonaRagList) {
    return;
  }
  if (!adminPersonaRagSequence.length) {
    els.adminPersonaRagList.classList.add('muted');
    els.adminPersonaRagList.textContent = 'No RAG steps selected.';
    return;
  }
  els.adminPersonaRagList.classList.remove('muted');
  const rows = adminPersonaRagSequence.map((ragBinding, index) => {
    const row = document.createElement('div');
    row.className = 'admin-persona-rag-row';

    const label = document.createElement('span');
    label.className = 'admin-persona-rag-label';
    const stateLabel = ragBinding.enabled ? 'Enabled' : 'Disabled';
    label.textContent = `${index + 1}. ${getAdminPersonaRagLabel(ragBinding)} [${stateLabel}]`;
    row.appendChild(label);

    const actions = document.createElement('div');
    actions.className = 'admin-persona-rag-actions';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'secondary';
    toggle.textContent = ragBinding.enabled ? 'Disable' : 'Enable';
    toggle.disabled = !adminPersonaRagPanelActive;
    toggle.addEventListener('click', () => toggleAdminPersonaRag(index));
    actions.appendChild(toggle);

    const up = document.createElement('button');
    up.type = 'button';
    up.className = 'secondary';
    up.textContent = 'Up';
    up.disabled = !adminPersonaRagPanelActive || index === 0;
    up.addEventListener('click', () => moveAdminPersonaRag(index, -1));
    actions.appendChild(up);

    const down = document.createElement('button');
    down.type = 'button';
    down.className = 'secondary';
    down.textContent = 'Down';
    down.disabled = !adminPersonaRagPanelActive || index >= adminPersonaRagSequence.length - 1;
    down.addEventListener('click', () => moveAdminPersonaRag(index, 1));
    actions.appendChild(down);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'secondary';
    remove.textContent = 'Remove';
    remove.disabled = !adminPersonaRagPanelActive;
    remove.addEventListener('click', () => removeAdminPersonaRag(index));
    actions.appendChild(remove);

    row.appendChild(actions);
    return row;
  });
  els.adminPersonaRagList.replaceChildren(...rows);
}

function moveAdminPersonaRag(index, direction) {
  /** Move one persona RAG step earlier or later in the configured execution order. */
  const currentIndex = Number(index);
  const nextIndex = currentIndex + Number(direction);
  if (
    !Number.isInteger(currentIndex) ||
    currentIndex < 0 ||
    currentIndex >= adminPersonaRagSequence.length ||
    nextIndex < 0 ||
    nextIndex >= adminPersonaRagSequence.length
  ) {
    return;
  }
  const nextSequence = [...adminPersonaRagSequence];
  const [item] = nextSequence.splice(currentIndex, 1);
  nextSequence.splice(nextIndex, 0, item);
  adminPersonaRagSequence = nextSequence;
  renderAdminPersonaRagSequence();
  renderAdminPersonaGraphMeta();
}

function removeAdminPersonaRag(index) {
  /** Remove one RAG step from the current persona chain definition. */
  const currentIndex = Number(index);
  if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= adminPersonaRagSequence.length) {
    return;
  }
  adminPersonaRagSequence = adminPersonaRagSequence.filter((_, itemIndex) => itemIndex !== currentIndex);
  renderAdminPersonaRagSequence();
  renderAdminPersonaGraphMeta();
}

function toggleAdminPersonaRag(index) {
  /** Toggle whether one persona RAG step is enabled without changing its position. */
  const currentIndex = Number(index);
  if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= adminPersonaRagSequence.length) {
    return;
  }
  adminPersonaRagSequence = adminPersonaRagSequence.map((item, itemIndex) =>
    itemIndex === currentIndex ? { ...item, enabled: !item.enabled } : item
  );
  renderAdminPersonaRagSequence();
  renderAdminPersonaGraphMeta();
}

function addAdminPersonaRag() {
  /** Append the currently selected RAG step to the persona chain if it is not already present. */
  const selectedRagKey = String(els.adminPersonaRagSelect?.value || '').trim();
  if (!selectedRagKey) {
    setAdminPersonaFeedback('Choose a RAG step before adding it to the persona chain.', 'error');
    return;
  }
  if (adminPersonaRagSequence.some((item) => item.key === selectedRagKey)) {
    setAdminPersonaFeedback(`${getAdminPersonaRagLabel(selectedRagKey)} is already in this persona chain.`, 'error');
    return;
  }
  adminPersonaRagSequence = [...adminPersonaRagSequence, { key: selectedRagKey, enabled: true }];
  renderAdminPersonaRagSequence();
  renderAdminPersonaGraphMeta();
  setAdminPersonaFeedback(`${getAdminPersonaRagLabel(selectedRagKey)} added to the persona chain.`, 'info');
}

async function loadAdminPersonaRagOptions() {
  /** Load the currently available backend RAG steps that personas can execute in sequence. */
  const payload = await api('/api/admin/personas/rags');
  adminPersonaRagOptionsCache = Array.isArray(payload?.rags) ? payload.rags : [];
  syncAdminPersonaRagOptions();
  renderAdminPersonaRagSequence();
  renderAdminPersonaGraphMeta();
}

function syncAdminPersonaToolOptions() {
  /** Mirror current backend MCP tool options into the persona editor selector. */
  if (!els.adminPersonaToolSelect) {
    return;
  }
  const currentValue = String(els.adminPersonaToolSelect.value || '').trim();
  els.adminPersonaToolSelect.innerHTML = '';
  if (!adminPersonaToolOptionsCache.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No MCP tools available';
    els.adminPersonaToolSelect.appendChild(option);
    els.adminPersonaToolSelect.disabled = true;
    return;
  }
  els.adminPersonaToolSelect.disabled = !adminPersonaToolsPanelActive;
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Choose one MCP tool';
  els.adminPersonaToolSelect.appendChild(placeholder);
  for (const item of adminPersonaToolOptionsCache) {
    const option = document.createElement('option');
    option.value = String(item.key || '').trim();
    option.textContent = item.label || item.key || 'Unknown MCP tool';
    option.title = item.description || '';
    option.disabled = item.available === false;
    els.adminPersonaToolSelect.appendChild(option);
  }
  const values = Array.from(els.adminPersonaToolSelect.options).map((item) => item.value);
  els.adminPersonaToolSelect.value = values.includes(currentValue) ? currentValue : '';
}

function renderAdminPersonaToolSequence() {
  /** Render the ordered MCP tools chain attached to the persona currently being edited. */
  if (!els.adminPersonaToolList) {
    return;
  }
  if (!adminPersonaToolSequence.length) {
    els.adminPersonaToolList.classList.add('muted');
    els.adminPersonaToolList.textContent = 'No MCP tools selected.';
    return;
  }
  els.adminPersonaToolList.classList.remove('muted');
  const rows = adminPersonaToolSequence.map((toolBinding, index) => {
    const row = document.createElement('div');
    row.className = 'admin-persona-rag-row';

    const label = document.createElement('span');
    label.className = 'admin-persona-rag-label';
    const stateLabel = toolBinding.enabled ? 'Enabled' : 'Disabled';
    label.textContent = `${index + 1}. ${getAdminPersonaToolLabel(toolBinding)} [${stateLabel}]`;
    row.appendChild(label);

    const actions = document.createElement('div');
    actions.className = 'admin-persona-rag-actions';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'secondary';
    toggle.textContent = toolBinding.enabled ? 'Disable' : 'Enable';
    toggle.disabled = !adminPersonaToolsPanelActive;
    toggle.addEventListener('click', () => toggleAdminPersonaTool(index));
    actions.appendChild(toggle);

    const up = document.createElement('button');
    up.type = 'button';
    up.className = 'secondary';
    up.textContent = 'Up';
    up.disabled = !adminPersonaToolsPanelActive || index === 0;
    up.addEventListener('click', () => moveAdminPersonaTool(index, -1));
    actions.appendChild(up);

    const down = document.createElement('button');
    down.type = 'button';
    down.className = 'secondary';
    down.textContent = 'Down';
    down.disabled = !adminPersonaToolsPanelActive || index >= adminPersonaToolSequence.length - 1;
    down.addEventListener('click', () => moveAdminPersonaTool(index, 1));
    actions.appendChild(down);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'secondary';
    remove.textContent = 'Remove';
    remove.disabled = !adminPersonaToolsPanelActive;
    remove.addEventListener('click', () => removeAdminPersonaTool(index));
    actions.appendChild(remove);

    row.appendChild(actions);
    return row;
  });
  els.adminPersonaToolList.replaceChildren(...rows);
}

function moveAdminPersonaTool(index, direction) {
  /** Move one persona MCP tool earlier or later in the configured execution order. */
  const currentIndex = Number(index);
  const nextIndex = currentIndex + Number(direction);
  if (
    !Number.isInteger(currentIndex)
    || currentIndex < 0
    || currentIndex >= adminPersonaToolSequence.length
    || nextIndex < 0
    || nextIndex >= adminPersonaToolSequence.length
  ) {
    return;
  }
  const nextSequence = [...adminPersonaToolSequence];
  const [item] = nextSequence.splice(currentIndex, 1);
  nextSequence.splice(nextIndex, 0, item);
  adminPersonaToolSequence = nextSequence;
  renderAdminPersonaToolSequence();
  renderAdminPersonaGraphMeta();
}

function removeAdminPersonaTool(index) {
  /** Remove one MCP tool from the current persona chain definition. */
  const currentIndex = Number(index);
  if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= adminPersonaToolSequence.length) {
    return;
  }
  adminPersonaToolSequence = adminPersonaToolSequence.filter((_, itemIndex) => itemIndex !== currentIndex);
  renderAdminPersonaToolSequence();
  renderAdminPersonaGraphMeta();
}

function toggleAdminPersonaTool(index) {
  /** Toggle whether one persona MCP tool is enabled without changing its position. */
  const currentIndex = Number(index);
  if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= adminPersonaToolSequence.length) {
    return;
  }
  adminPersonaToolSequence = adminPersonaToolSequence.map((item, itemIndex) =>
    itemIndex === currentIndex ? { ...item, enabled: !item.enabled } : item
  );
  renderAdminPersonaToolSequence();
  renderAdminPersonaGraphMeta();
}

function addAdminPersonaTool() {
  /** Append the currently selected MCP tool to the persona chain if it is not already present. */
  const selectedToolKey = String(els.adminPersonaToolSelect?.value || '').trim();
  if (!selectedToolKey) {
    setAdminPersonaFeedback('Choose an MCP tool before adding it to the persona chain.', 'error');
    return;
  }
  if (adminPersonaToolSequence.some((item) => item.key === selectedToolKey)) {
    setAdminPersonaFeedback(`${getAdminPersonaToolLabel(selectedToolKey)} is already in this persona chain.`, 'error');
    return;
  }
  adminPersonaToolSequence = [...adminPersonaToolSequence, { key: selectedToolKey, enabled: true }];
  renderAdminPersonaToolSequence();
  renderAdminPersonaGraphMeta();
  setAdminPersonaFeedback(`${getAdminPersonaToolLabel(selectedToolKey)} added to the persona chain.`, 'info');
}

async function loadAdminPersonaToolOptions() {
  /** Load the currently available backend MCP tool steps that personas can execute in sequence. */
  const payload = await api('/api/admin/personas/tools');
  adminPersonaToolOptionsCache = Array.isArray(payload?.tools) ? payload.tools : [];
  syncAdminPersonaToolOptions();
  renderAdminPersonaToolSequence();
  renderAdminPersonaGraphMeta();
}

async function loadAdminPersonas() {
  /** Load saved personas for the Admin/Personas panel. */
  const payload = await api('/api/admin/personas');
  adminPersonasCache = Array.isArray(payload?.personas) ? payload.personas : [];
  renderAdminPersonas(adminPersonasCache);
}

function renderAdminPersonas(personas) {
  /** Render the saved persona list for the Personas subtab. */
  if (!els.adminPersonaList) {
    return;
  }
  const normalizedPersonas = Array.isArray(personas) ? personas : [];
  syncAdminPersonaSelection(normalizedPersonas);
  if (!normalizedPersonas.length) {
    els.adminPersonaList.classList.add('muted');
    els.adminPersonaList.textContent = 'No personas added yet.';
    return;
  }
  els.adminPersonaList.classList.remove('muted');
  const items = normalizedPersonas.map((item) => {
    const row = document.createElement('div');
    row.className = 'admin-user-row';
    row.dataset.personaId = String(item.persona_id || '');
    row.setAttribute('role', 'button');
    row.setAttribute('tabindex', '0');
    row.classList.toggle('selected', row.dataset.personaId === selectedAdminPersonaId);
    const llmLabel = `${item.llm_provider === 'openai' ? 'ChatGPT' : 'Ollama'} (${item.llm_model})`;
    const ragCount = Array.isArray(item.rag_sequence) ? item.rag_sequence.length : 0;
    const toolCount = Array.isArray(item.tool_sequence) ? item.tool_sequence.length : 0;
    const shell = document.createElement('div');
    shell.className = 'admin-user-row-shell';

    const copy = document.createElement('span');
    copy.className = 'admin-user-row-copy';
    copy.textContent = `${item.name} | ${llmLabel} | RAG steps: ${ragCount} | MCP tools: ${toolCount} | ${item.created_at}`;
    shell.appendChild(copy);

    row.appendChild(shell);

    const select = () => selectAdminPersonaById(item.persona_id);
    row.addEventListener('click', select);
    row.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') {
        return;
      }
      event.preventDefault();
      select();
    });
    return row;
  });
  els.adminPersonaList.replaceChildren(...items);
}

function syncAdminPersonaSelection(personas) {
  /** Keep the current-persona selector and selected-persona text in sync. */
  const normalizedPersonas = Array.isArray(personas) ? personas : [];
  if (els.adminPersonaSelect) {
    els.adminPersonaSelect.innerHTML = '';
    if (!normalizedPersonas.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No personas available';
      els.adminPersonaSelect.appendChild(option);
      els.adminPersonaSelect.disabled = true;
      selectedAdminPersonaId = '';
    } else {
      els.adminPersonaSelect.disabled = false;
      const availableIds = new Set(normalizedPersonas.map((item) => String(item.persona_id || '').trim()));
      if (!availableIds.has(selectedAdminPersonaId)) {
        selectedAdminPersonaId = String(normalizedPersonas[0]?.persona_id || '').trim();
      }
      for (const item of normalizedPersonas) {
        const option = document.createElement('option');
        option.value = String(item.persona_id || '').trim();
        option.textContent = item.name || 'Unnamed Persona';
        els.adminPersonaSelect.appendChild(option);
      }
      els.adminPersonaSelect.value = selectedAdminPersonaId;
    }
  }
  const selectedPersona = normalizedPersonas.find(
    (item) => String(item.persona_id || '').trim() === selectedAdminPersonaId
  );
  if (els.adminPersonaSelectedMeta) {
    if (!selectedPersona) {
      els.adminPersonaSelectedMeta.textContent = 'No persona selected.';
    } else {
      const llmLabel = `${selectedPersona.llm_provider === 'openai' ? 'ChatGPT' : 'Ollama'} (${selectedPersona.llm_model})`;
      const ragSummary = Array.isArray(selectedPersona.rag_sequence) && selectedPersona.rag_sequence.length
        ? selectedPersona.rag_sequence
            .map((item) => {
              const binding = normalizeAdminPersonaRagBinding(item);
              if (!binding) {
                return null;
              }
              return `${getAdminPersonaRagLabel(binding)}:${binding.enabled ? 'on' : 'off'}`;
            })
            .filter(Boolean)
            .join(' -> ')
        : 'No RAG steps';
      const toolSummary = Array.isArray(selectedPersona.tool_sequence) && selectedPersona.tool_sequence.length
        ? selectedPersona.tool_sequence
            .map((item) => {
              const binding = normalizeAdminPersonaToolBinding(item);
              if (!binding) {
                return null;
              }
              return `${getAdminPersonaToolLabel(binding)}:${binding.enabled ? 'on' : 'off'}`;
            })
            .filter(Boolean)
            .join(' -> ')
        : 'No MCP tools';
      els.adminPersonaSelectedMeta.textContent =
        `Selected: ${selectedPersona.name} (${llmLabel}) | ${ragSummary} | ${toolSummary}`;
    }
  }
  renderAdminPersonaGraphMeta();
}

function setAdminPersonaFeedback(message, tone = 'info') {
  /** Render local feedback in the Admin/Personas panel. */
  if (!els.adminPersonaFeedback) {
    return;
  }
  const nextMessage = String(message || '').trim();
  els.adminPersonaFeedback.textContent = nextMessage || ' ';
  els.adminPersonaFeedback.classList.remove('muted', 'error', 'success');
  if (!nextMessage) {
    els.adminPersonaFeedback.classList.add('muted');
    return;
  }
  if (tone === 'error') {
    els.adminPersonaFeedback.classList.add('error');
    return;
  }
  if (tone === 'success') {
    els.adminPersonaFeedback.classList.add('success');
    return;
  }
  els.adminPersonaFeedback.classList.add('muted');
}

function setAdminPersonaPromptModalSentiment(message, tone = 'info') {
  /** Render prompt-sentiment feedback inside the persona prompt pop-out. */
  if (!els.adminPersonaPromptSentimentMeta) {
    return;
  }
  const nextMessage = String(message || '').trim();
  els.adminPersonaPromptSentimentMeta.textContent = nextMessage || 'Prompt sentiment has not been scored.';
  els.adminPersonaPromptSentimentMeta.classList.remove('muted', 'error', 'success');
  if (!nextMessage || tone === 'info') {
    els.adminPersonaPromptSentimentMeta.classList.add('muted');
    return;
  }
  if (tone === 'error') {
    els.adminPersonaPromptSentimentMeta.classList.add('error');
    return;
  }
  els.adminPersonaPromptSentimentMeta.classList.add('success');
}

function showAdminPersonaPromptModal({
  personaId = '',
  personaName = '',
  promptSections = null,
  promptText = '',
  promptTemplateKey = '',
} = {}) {
  /** Open the persona prompt pop-out for one saved persona or the active draft prompt sections. */
  if (
    !els.adminPersonaPromptModal ||
    !els.adminPersonaPromptModalSystem ||
    !els.adminPersonaPromptModalAssistant ||
    !els.adminPersonaPromptModalContext
  ) {
    return;
  }
  adminPersonaPromptModalPersonaId = String(personaId || '').trim();
  adminPersonaPromptModalPersonaName = String(personaName || '').trim();

  if (els.adminPersonaPromptModalTitle) {
    els.adminPersonaPromptModalTitle.textContent = 'Persona Prompt';
  }
  if (els.adminPersonaPromptModalMeta) {
    if (adminPersonaPromptModalPersonaId) {
      const name = adminPersonaPromptModalPersonaName || 'Saved Persona';
      els.adminPersonaPromptModalMeta.textContent = `${name} | id: ${adminPersonaPromptModalPersonaId}`;
    } else {
      els.adminPersonaPromptModalMeta.textContent = 'Active draft prompt';
    }
  }
  const normalizedPromptSections = normalizeAdminPersonaPromptSections(
    promptSections,
    promptText,
    String(promptTemplateKey || '').trim()
  );
  setAdminPersonaPromptSectionsInModal(normalizedPromptSections);
  setAdminPersonaPromptModalSentiment('', 'info');
  if (els.adminPersonaPromptResaveBtn) {
    els.adminPersonaPromptResaveBtn.disabled = !adminPersonaPromptModalPersonaId;
  }
  els.adminPersonaPromptModal.classList.remove('hidden');
  els.adminPersonaPromptModalSystem.focus();
}

function hideAdminPersonaPromptModal() {
  /** Close the persona prompt pop-out and clear transient modal state. */
  if (
    !els.adminPersonaPromptModal ||
    !els.adminPersonaPromptModalSystem ||
    !els.adminPersonaPromptModalAssistant ||
    !els.adminPersonaPromptModalContext
  ) {
    return;
  }
  els.adminPersonaPromptModal.classList.add('hidden');
  els.adminPersonaPromptModalSystem.value = '';
  els.adminPersonaPromptModalAssistant.value = '';
  els.adminPersonaPromptModalContext.value = '';
  if (els.adminPersonaPromptModalMeta) {
    els.adminPersonaPromptModalMeta.textContent = 'No prompt selected.';
  }
  setAdminPersonaPromptModalSentiment('', 'info');
  adminPersonaPromptModalPersonaId = '';
  adminPersonaPromptModalPersonaName = '';
}

function openAdminPersonaPromptFromForm() {
  /** Open the prompt pop-out using the current draft/editor prompt sections. */
  const promptSections = getAdminPersonaPromptSectionsFromForm();
  const draftName = String(els.adminPersonaName?.value || '').trim() || 'Draft Persona';
  const draftPersonaId = String(editingAdminPersonaId || '').trim();
  const promptTemplateKey = String(adminPersonaSelectedPromptTemplateKey || '').trim();
  showAdminPersonaPromptModal({
    personaId: draftPersonaId,
    personaName: draftName,
    promptSections,
    promptTemplateKey,
  });
}

function syncAdminPersonaPromptToggleButtonLabel() {
  /** Keep the Prompt toggle button label fixed. */
  if (!els.adminPersonaOpenPromptModalBtn) {
    return;
  }
  els.adminPersonaOpenPromptModalBtn.textContent = 'Prompt';
}

function syncAdminPersonaRagToggleButtonLabel() {
  /** Keep the RAG toggle button label fixed. */
  if (!els.adminPersonaToggleRagBtn) {
    return;
  }
  els.adminPersonaToggleRagBtn.textContent = 'RAG';
}

function syncAdminPersonaToolsToggleButtonLabel() {
  /** Keep the Tools toggle button label fixed. */
  if (!els.adminPersonaToggleToolsBtn) {
    return;
  }
  els.adminPersonaToggleToolsBtn.textContent = 'Tools';
}

function syncAdminPersonaPromptObservablesToggleButtonLabel() {
  /** Keep the Prompt Observables toggle button label fixed. */
  if (!els.adminPersonaTogglePromptObservablesBtn) {
    return;
  }
  els.adminPersonaTogglePromptObservablesBtn.textContent = 'All Prompts Observables';
}

function normalizeAdminPersonaPromptObservablesScope(scope = 'all') {
  /** Normalize requested prompt-observables scope to one supported value. */
  const normalized = String(scope || '')
    .trim()
    .toLowerCase();
  if (normalized === 'system' || normalized === 'assistant' || normalized === 'context') {
    return normalized;
  }
  return 'all';
}

function adminPersonaPromptObservablesScopeLabel(scope = 'all') {
  /** Return a display label for the active prompt-observable scope. */
  const normalized = normalizeAdminPersonaPromptObservablesScope(scope);
  if (normalized === 'system') {
    return 'System Prompt';
  }
  if (normalized === 'assistant') {
    return 'Assistant Prompt';
  }
  if (normalized === 'context') {
    return 'Context Prompt';
  }
  return 'All Prompts';
}

function computeAdminPersonaPromptObservableMetrics(scope = 'all') {
  /** Compute prompt-quality observables from current Persona prompt section content. */
  const normalizedScope = normalizeAdminPersonaPromptObservablesScope(scope);
  const sections = getAdminPersonaPromptSectionsFromForm();
  const scopedSections = {
    system: normalizedScope === 'all' || normalizedScope === 'system' ? sections.system : '',
    assistant: normalizedScope === 'all' || normalizedScope === 'assistant' ? sections.assistant : '',
    context: normalizedScope === 'all' || normalizedScope === 'context' ? sections.context : '',
  };
  const scopeLabel = adminPersonaPromptObservablesScopeLabel(normalizedScope);
  const scopeDescriptor =
    normalizedScope === 'all'
      ? 'across system, assistant, and context prompts'
      : `in the ${scopeLabel.toLowerCase()}`;
  const combined = `${scopedSections.system}\n${scopedSections.assistant}\n${scopedSections.context}`.trim();
  if (!combined) {
    return [];
  }
  const tokenize = (text = '') =>
    String(text || '')
      .toLowerCase()
      .match(/[a-z0-9_'-]+/g) || [];
  const sectionWords = {
    system: tokenize(scopedSections.system).length,
    assistant: tokenize(scopedSections.assistant).length,
    context: tokenize(scopedSections.context).length,
  };
  const allWords = tokenize(combined);
  const totalWords = allWords.length;
  const totalChars = combined.length;
  const estimatedTokens = Math.max(1, Math.round(totalWords * 1.33));
  const uniqueWordRatio = totalWords ? (new Set(allWords).size / totalWords) : 0;

  const constraintPattern = /\b(must|shall|required|always|never|only|exactly|do\s+not|cannot)\b/gi;
  const ambiguityPattern = /\b(maybe|might|can|could|approximately|roughly|some|various|etc|possibly|generally|usually)\b/gi;
  const constraintMatches = (combined.match(constraintPattern) || []).length;
  const ambiguityMatches = (combined.match(ambiguityPattern) || []).length;
  const constraintDensity = totalWords ? (constraintMatches / totalWords) * 100 : 0;
  const ambiguityDensity = totalWords ? (ambiguityMatches / totalWords) * 100 : 0;

  const targetShare = 1 / 3;
  const shares = [
    sectionWords.system / totalWords,
    sectionWords.assistant / totalWords,
    sectionWords.context / totalWords,
  ];
  const meanDeviation = shares.reduce((sum, value) => sum + Math.abs(value - targetShare), 0) / shares.length;
  const sectionBalance = Math.max(0, Math.min(100, 100 - Math.round((meanDeviation / targetShare) * 100)));
  const clarityScore = Math.max(0, Math.min(100, Math.round(65 + (constraintDensity * 2.2) - (ambiguityDensity * 3.1))));
  const systemSharePct = totalWords ? (sectionWords.system / totalWords) * 100 : 0;
  const contextSharePct = totalWords ? (sectionWords.context / totalWords) * 100 : 0;
  const baseMetrics = [
    {
      key: 'prompt_total_words',
      label: 'Prompt Total Words',
      value: `${totalWords}`,
      explanation:
        `Total words ${scopeDescriptor}. ` +
        `Higher values usually increase model steering depth, but also cost and latency.`,
    },
    {
      key: 'estimated_prompt_tokens',
      label: 'Estimated Prompt Tokens',
      value: `${estimatedTokens}`,
      explanation:
        `Approximate token footprint before runtime context is added. ` +
        `Track this to avoid prompt bloat and context-window pressure.`,
    },
    {
      key: 'system_instruction_share_pct',
      label: 'System Instruction Share',
      value: `${systemSharePct.toFixed(1)}%`,
      explanation:
        `Portion of prompt words located in the System section. ` +
        `Too low can weaken policy/control; too high can over-constrain assistant behavior.`,
    },
    {
      key: 'context_instruction_share_pct',
      label: 'Context Share',
      value: `${contextSharePct.toFixed(1)}%`,
      explanation:
        `Portion of prompt words in Context guidance. ` +
        `Higher values can improve retrieval grounding, but may crowd out task instructions.`,
    },
    {
      key: 'constraint_density_pct',
      label: 'Constraint Density',
      value: `${constraintDensity.toFixed(2)}%`,
      explanation:
        `Density of explicit constraint terms (must/never/only/etc). ` +
        `Higher density usually improves adherence, but excessive constraints can reduce flexibility.`,
    },
    {
      key: 'ambiguity_risk_pct',
      label: 'Ambiguity Risk',
      value: `${ambiguityDensity.toFixed(2)}%`,
      explanation:
        `Density of ambiguity terms (maybe/could/approximately/etc). ` +
        `Higher risk often correlates with inconsistent outputs and lower first-pass success.`,
    },
    {
      key: 'section_balance_score',
      label: 'Section Balance Score',
      value: `${sectionBalance}/100`,
      explanation:
        `How evenly prompt content is distributed across System/Assistant/Context sections. ` +
        `Low balance can signal one section dominating behavior unexpectedly.`,
    },
    {
      key: 'directive_clarity_score',
      label: 'Directive Clarity Score',
      value: `${clarityScore}/100`,
      explanation:
        `Composite estimate from explicit constraints minus ambiguity markers. ` +
        `Higher scores tend to produce more deterministic and policy-aligned responses.`,
    },
    {
      key: 'lexical_diversity_ratio',
      label: 'Lexical Diversity',
      value: `${(uniqueWordRatio * 100).toFixed(1)}%`,
      explanation:
        `Unique-word ratio in prompt text. ` +
        `Very low diversity can indicate repetitive prompts; very high diversity can imply diffuse instructions.`,
    },
    {
      key: 'prompt_total_characters',
      label: 'Prompt Total Characters',
      value: `${totalChars}`,
      explanation:
        `Raw character count ${scopeDescriptor}. ` +
        `Useful for quick growth tracking and pre-tokenization budget checks.`,
    },
  ];
  if (normalizedScope !== 'all') {
    return baseMetrics.filter(
      (metric) =>
        metric.key !== 'system_instruction_share_pct' &&
        metric.key !== 'context_instruction_share_pct' &&
        metric.key !== 'section_balance_score'
    );
  }
  return baseMetrics;
}

function setAdminPersonaPromptObservableDetail(metric) {
  /** Render one prompt-observable explanation in the Persona panel detail area. */
  if (!els.adminPersonaPromptObservablesDetail) {
    return;
  }
  if (!metric || typeof metric !== 'object') {
    els.adminPersonaPromptObservablesDetail.textContent = 'Select a metric to view what it means.';
    els.adminPersonaPromptObservablesDetail.classList.add('muted');
    return;
  }
  els.adminPersonaPromptObservablesDetail.classList.remove('muted');
  els.adminPersonaPromptObservablesDetail.textContent =
    `${metric.label}: ${metric.value}\n\n${metric.explanation}`;
}

function renderAdminPersonaPromptObservables() {
  /** Render prompt observables list and detail for the Persona prompt content. */
  if (!els.adminPersonaPromptObservablesList) {
    return;
  }
  const scopeLabel = adminPersonaPromptObservablesScopeLabel(adminPersonaPromptObservablesScope);
  if (!adminPersonaPromptObservablesPanelActive) {
    els.adminPersonaPromptObservablesList.classList.add('muted');
    els.adminPersonaPromptObservablesList.textContent = 'Open All Prompts Observables to compute prompt metrics.';
    setAdminPersonaPromptObservableDetail(null);
    return;
  }
  const metrics = computeAdminPersonaPromptObservableMetrics(adminPersonaPromptObservablesScope);
  if (!metrics.length) {
    selectedAdminPersonaPromptObservableKey = '';
    els.adminPersonaPromptObservablesList.classList.add('muted');
    els.adminPersonaPromptObservablesList.textContent = `Enter content in ${scopeLabel} to compute prompt observables.`;
    setAdminPersonaPromptObservableDetail({
      label: `All Prompts Observables | ${scopeLabel}`,
      value: 'No Data',
      explanation: `Add prompt content for ${scopeLabel} and refresh metrics.`,
    });
    return;
  }

  const metricByKey = new Map(metrics.map((item) => [item.key, item]));
  if (!metricByKey.has(selectedAdminPersonaPromptObservableKey)) {
    selectedAdminPersonaPromptObservableKey = metrics[0].key;
  }

  els.adminPersonaPromptObservablesList.classList.remove('muted');
  const scopeMeta = document.createElement('div');
  scopeMeta.className = 'muted';
  scopeMeta.textContent = `Scope: ${scopeLabel}`;
  const rows = metrics.map((metric) => {
    const row = document.createElement('div');
    row.className = 'admin-persona-rag-row';

    const label = document.createElement('div');
    label.className = 'admin-persona-rag-label';
    label.textContent = `${metric.label}: ${metric.value}`;
    row.appendChild(label);

    const actions = document.createElement('div');
    actions.className = 'admin-persona-rag-actions';
    const explain = document.createElement('button');
    explain.type = 'button';
    explain.className = 'secondary prompt-mini-btn';
    explain.textContent = selectedAdminPersonaPromptObservableKey === metric.key ? 'Selected' : 'Explain';
    explain.disabled = selectedAdminPersonaPromptObservableKey === metric.key;
    explain.addEventListener('click', () => {
      selectedAdminPersonaPromptObservableKey = metric.key;
      renderAdminPersonaPromptObservables();
    });
    actions.appendChild(explain);
    row.appendChild(actions);
    return row;
  });
  els.adminPersonaPromptObservablesList.replaceChildren(scopeMeta, ...rows);
  setAdminPersonaPromptObservableDetail(metricByKey.get(selectedAdminPersonaPromptObservableKey));
}

function setAdminPersonaPromptPanelActive(active) {
  /** Enable or disable all controls contained in the grouped Prompt panel. */
  adminPersonaPromptPanelActive = !!active;
  const controls = [
    els.adminPersonaSystemPromptTemplateSelect,
    els.adminPersonaAssistantPromptTemplateSelect,
    els.adminPersonaContextPromptTemplateSelect,
    els.adminPersonaSystemChoosePromptBtn,
    els.adminPersonaAssistantChoosePromptBtn,
    els.adminPersonaContextChoosePromptBtn,
    els.adminPersonaSystemObservableBtn,
    els.adminPersonaAssistantObservableBtn,
    els.adminPersonaContextObservableBtn,
    els.adminPersonaSystemSavePromptBtn,
    els.adminPersonaAssistantSavePromptBtn,
    els.adminPersonaContextSavePromptBtn,
    els.adminPersonaSystemPrompt,
    els.adminPersonaAssistantPrompt,
    els.adminPersonaContextPrompt,
  ];
  for (const control of controls) {
    if (!control) {
      continue;
    }
    control.disabled = !adminPersonaPromptPanelActive;
  }
  if (!adminPersonaPromptPanelActive) {
    hideAdminPersonaPromptTemplateDropdowns();
  }
}

function setAdminPersonaRagPanelActive(active) {
  /** Enable or disable all controls contained in the grouped RAG panel. */
  adminPersonaRagPanelActive = !!active;
  if (els.adminPersonaLoadRagsBtn) {
    els.adminPersonaLoadRagsBtn.disabled = !adminPersonaRagPanelActive;
  }
  if (els.adminPersonaRagAddBtn) {
    els.adminPersonaRagAddBtn.disabled = !adminPersonaRagPanelActive;
  }
  if (!adminPersonaRagPanelActive) {
    if (els.adminPersonaGraphQuestion) {
      els.adminPersonaGraphQuestion.disabled = true;
    }
    if (els.adminPersonaGraphAskBtn) {
      els.adminPersonaGraphAskBtn.disabled = true;
      els.adminPersonaGraphAskBtn.textContent = 'Ask Graph';
    }
    if (els.adminPersonaGraphClearBtn) {
      els.adminPersonaGraphClearBtn.disabled = true;
    }
    if (els.adminPersonaGraphProgress) {
      els.adminPersonaGraphProgress.classList.add('hidden');
    }
    stopClock('adminPersonaGraph');
    if (els.adminPersonaGraphClock) {
      els.adminPersonaGraphClock.textContent = '0.0s';
    }
  }
  syncAdminPersonaRagOptions();
  renderAdminPersonaRagSequence();
  if (adminPersonaRagPanelActive) {
    setAdminPersonaGraphProcessing(false);
  }
}

function setAdminPersonaPromptObservablesPanelActive(active) {
  /** Enable or disable Prompt Observables controls and refresh visible metrics. */
  adminPersonaPromptObservablesPanelActive = !!active;
  if (els.adminPersonaRefreshPromptObservablesBtn) {
    els.adminPersonaRefreshPromptObservablesBtn.disabled = !adminPersonaPromptObservablesPanelActive;
  }
  if (!adminPersonaPromptObservablesPanelActive) {
    selectedAdminPersonaPromptObservableKey = '';
    adminPersonaPromptObservablesScope = 'all';
  }
  renderAdminPersonaPromptObservables();
}

function setAdminPersonaToolsPanelActive(active) {
  /** Enable or disable all controls contained in the grouped Tools panel. */
  adminPersonaToolsPanelActive = !!active;
  if (els.adminPersonaLoadToolsBtn) {
    els.adminPersonaLoadToolsBtn.disabled = !adminPersonaToolsPanelActive;
  }
  if (els.adminPersonaToolAddBtn) {
    els.adminPersonaToolAddBtn.disabled = !adminPersonaToolsPanelActive;
  }
  syncAdminPersonaToolOptions();
  renderAdminPersonaToolSequence();
  if (adminPersonaToolsPanelActive && !adminPersonaToolOptionsCache.length) {
    loadAdminPersonaToolOptions().catch((err) => setStatus(err.message));
  }
}

function toggleAdminPersonaPromptPanel() {
  /** Toggle all prompt-related content as one grouped panel. */
  if (!els.adminPersonaPromptPanel) {
    return;
  }
  const nextActive = els.adminPersonaPromptPanel.classList.contains('hidden');
  els.adminPersonaPromptPanel.classList.toggle('hidden', !nextActive);
  setAdminPersonaPromptPanelActive(nextActive);
  syncAdminPersonaPromptToggleButtonLabel();
}

function toggleAdminPersonaRagPanel() {
  /** Toggle all RAG-related content as one grouped panel. */
  if (!els.adminPersonaRagPanel) {
    return;
  }
  const nextActive = els.adminPersonaRagPanel.classList.contains('hidden');
  els.adminPersonaRagPanel.classList.toggle('hidden', !nextActive);
  setAdminPersonaRagPanelActive(nextActive);
  syncAdminPersonaRagToggleButtonLabel();
}

function toggleAdminPersonaPromptObservablesPanel() {
  /** Toggle prompt-quality observables panel for the active Persona prompt content. */
  if (!els.adminPersonaPromptObservablesPanel) {
    return;
  }
  const nextActive = els.adminPersonaPromptObservablesPanel.classList.contains('hidden');
  if (nextActive) {
    adminPersonaPromptObservablesScope = 'all';
  }
  els.adminPersonaPromptObservablesPanel.classList.toggle('hidden', !nextActive);
  setAdminPersonaPromptObservablesPanelActive(nextActive);
  syncAdminPersonaPromptObservablesToggleButtonLabel();
}

function openAdminPersonaPromptObservablesForScope(scope = 'all') {
  /** Open prompt observables and render metrics for one prompt section or all prompt sections. */
  if (!els.adminPersonaPromptObservablesPanel) {
    return;
  }
  adminPersonaPromptObservablesScope = normalizeAdminPersonaPromptObservablesScope(scope);
  els.adminPersonaPromptObservablesPanel.classList.remove('hidden');
  setAdminPersonaPromptObservablesPanelActive(true);
  syncAdminPersonaPromptObservablesToggleButtonLabel();
}

function hideAdminPersonaPromptObservableModal() {
  /** Close section-scoped prompt observables popup and clear transient modal state. */
  if (
    !els.adminPersonaPromptObservableModal ||
    !els.adminPersonaPromptObservableModalTitle ||
    !els.adminPersonaPromptObservableModalMeta ||
    !els.adminPersonaPromptObservableModalBody
  ) {
    return;
  }
  els.adminPersonaPromptObservableModal.classList.add('hidden');
  els.adminPersonaPromptObservableModalTitle.textContent = 'Prompt Observables';
  els.adminPersonaPromptObservableModalMeta.textContent = 'No prompt scope selected.';
  els.adminPersonaPromptObservableModalBody.classList.add('muted');
  els.adminPersonaPromptObservableModalBody.textContent =
    'Click a prompt-level Observable button to inspect section-specific observables.';
  adminPersonaPromptObservableModalScope = '';
}

function showAdminPersonaPromptObservableModal(scope = 'system') {
  /** Open popup with all observables computed for one specific prompt section. */
  if (
    !els.adminPersonaPromptObservableModal ||
    !els.adminPersonaPromptObservableModalTitle ||
    !els.adminPersonaPromptObservableModalMeta ||
    !els.adminPersonaPromptObservableModalBody
  ) {
    return;
  }
  const normalizedScope = normalizeAdminPersonaPromptObservablesScope(scope);
  const sectionScope = normalizedScope === 'all' ? 'system' : normalizedScope;
  const scopeLabel = adminPersonaPromptObservablesScopeLabel(sectionScope);
  adminPersonaPromptObservableModalScope = sectionScope;

  els.adminPersonaPromptObservableModalTitle.textContent = `${scopeLabel} Observables`;
  els.adminPersonaPromptObservableModalMeta.textContent = `Scoped metrics for ${scopeLabel}.`;

  const metrics = computeAdminPersonaPromptObservableMetrics(sectionScope);
  if (!metrics.length) {
    els.adminPersonaPromptObservableModalBody.classList.add('muted');
    els.adminPersonaPromptObservableModalBody.textContent =
      `No prompt content found in ${scopeLabel}. Add text and click Observable again.`;
  } else {
    const rows = metrics.map((metric) => {
      const row = document.createElement('article');
      row.className = 'admin-persona-observable-row';

      const title = document.createElement('div');
      title.className = 'admin-persona-observable-row-title';
      const label = document.createElement('strong');
      label.textContent = metric.label;
      const value = document.createElement('span');
      value.className = 'admin-persona-observable-row-value';
      value.textContent = String(metric.value || 'N/A');
      title.append(label, value);

      const body = document.createElement('p');
      body.textContent = String(metric.explanation || 'No explanation available.');

      row.append(title, body);
      return row;
    });
    els.adminPersonaPromptObservableModalBody.classList.remove('muted');
    els.adminPersonaPromptObservableModalBody.replaceChildren(...rows);
  }

  els.adminPersonaPromptObservableModal.classList.remove('hidden');
  els.adminPersonaPromptObservableModalCloseBtn?.focus();
}

function toggleAdminPersonaToolsPanel() {
  /** Toggle all MCP tool-related content as one grouped panel. */
  if (!els.adminPersonaToolsPanel) {
    return;
  }
  const nextActive = els.adminPersonaToolsPanel.classList.contains('hidden');
  els.adminPersonaToolsPanel.classList.toggle('hidden', !nextActive);
  if (nextActive) {
    els.adminPersonaToolsPanel.open = true;
  }
  setAdminPersonaToolsPanelActive(nextActive);
  syncAdminPersonaToolsToggleButtonLabel();
}

function openAdminPersonaPromptFromPersona(persona) {
  /** Open the prompt pop-out from one saved persona row. */
  if (!persona) {
    return;
  }
  showAdminPersonaPromptModal({
    personaId: String(persona.persona_id || '').trim(),
    personaName: String(persona.name || '').trim() || 'Saved Persona',
    promptSections: normalizeAdminPersonaPromptSections(
      persona.prompt_sections,
      persona.prompts,
      String(persona.prompt_template_key || '').trim()
    ),
    promptTemplateKey: String(persona.prompt_template_key || '').trim(),
  });
}

function applyAdminPersonaPromptFromModal() {
  /** Apply prompt-section edits from pop-out back into the Persona form without saving to CouchDB. */
  const promptSections = getAdminPersonaPromptSectionsFromModal();
  setAdminPersonaPromptSectionsInForm(promptSections);
  if (els.adminPersonaCreatePanel?.classList.contains('hidden')) {
    setAdminPersonaCreateOpen(true);
  }
  setAdminPersonaFeedback('Applied prompt changes to the Persona editor. Click Save Persona to persist.', 'info');
  setStatus('Applied prompt changes to the Persona editor.');
}

async function resaveAdminPersonaPromptFromModal() {
  /** Persist the edited prompt text for the selected persona directly from the pop-out. */
  const personaId = String(adminPersonaPromptModalPersonaId || '').trim();
  if (!personaId) {
    setStatus('Select and save a persona before re-saving prompts from the pop-out.');
    return;
  }
  const persona = adminPersonasCache.find((item) => String(item?.persona_id || '').trim() === personaId);
  if (!persona) {
    setStatus('The selected persona is unavailable. Refresh personas and try again.');
    return;
  }
  const promptSections = getAdminPersonaPromptSectionsFromModal();
  if (!adminPersonaPromptSectionsHaveContent(promptSections)) {
    setStatus('At least one prompt section is required before re-saving the persona.');
    return;
  }
  const prompts = composeLegacyPersonaPromptSections(promptSections);
  if (els.adminPersonaPromptResaveBtn) {
    els.adminPersonaPromptResaveBtn.disabled = true;
    els.adminPersonaPromptResaveBtn.textContent = 'Resaving...';
  }
  try {
    const savedPersona = await api('/api/admin/personas', {
      method: 'POST',
      body: JSON.stringify({
        persona_id: personaId,
        name: String(persona.name || '').trim(),
        llm_provider: String(persona.llm_provider || '').trim(),
        llm_model: String(persona.llm_model || '').trim(),
        prompt_template_key: String(persona.prompt_template_key || '').trim() || null,
        prompt_sections: promptSections,
        prompts,
        rag_sequence: Array.isArray(persona.rag_sequence)
          ? persona.rag_sequence.map((item) => {
              const binding = normalizeAdminPersonaRagBinding(item);
              return binding ? { key: binding.key, enabled: binding.enabled !== false } : null;
            }).filter((item) => !!item)
          : [],
        tool_sequence: Array.isArray(persona.tool_sequence)
          ? persona.tool_sequence.map((item) => {
              const binding = normalizeAdminPersonaToolBinding(item);
              return binding ? { key: binding.key, enabled: binding.enabled !== false } : null;
            }).filter((item) => !!item)
          : [],
      }),
    });
    adminPersonasCache = [
      savedPersona,
      ...adminPersonasCache.filter((item) => String(item.persona_id || '').trim() !== personaId),
    ];
    selectedAdminPersonaId = String(savedPersona.persona_id || '').trim();
    renderAdminPersonas(adminPersonasCache);
    if (editingAdminPersonaId && String(editingAdminPersonaId).trim() === personaId) {
      setAdminPersonaPromptSectionsInForm(promptSections);
    }
    setAdminPersonaFeedback(`${savedPersona.name} prompt updated from pop-out.`, 'success');
    setStatus('Persona prompt saved.');
    hideAdminPersonaPromptModal();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to save persona prompt.');
    setStatus(message);
    setAdminPersonaPromptModalSentiment(`Save failed: ${message}`, 'error');
    throw err;
  } finally {
    if (els.adminPersonaPromptResaveBtn) {
      els.adminPersonaPromptResaveBtn.disabled = false;
      els.adminPersonaPromptResaveBtn.textContent = 'Resave Persona';
    }
  }
}

async function scoreAdminPersonaPromptSentiment() {
  /** Score sentiment for all prompt sections currently shown in the prompt pop-out editor. */
  const promptText = composeLegacyPersonaPromptSections(getAdminPersonaPromptSectionsFromModal());
  if (!promptText) {
    setAdminPersonaPromptModalSentiment('Enter prompt section text before scoring sentiment.', 'error');
    return;
  }
  if (els.adminPersonaPromptSentimentBtn) {
    els.adminPersonaPromptSentimentBtn.disabled = true;
    els.adminPersonaPromptSentimentBtn.textContent = 'Scoring...';
  }
  try {
    const response = await api('/api/text-sentiment', {
      method: 'POST',
      body: JSON.stringify({ text: promptText }),
    });
    setAdminPersonaPromptModalSentiment(
      `${String(response.label || '').toUpperCase()} ${Number(response.score || 0).toFixed(2)} | `
      + `+${Number(response.positive_matches || 0)} / -${Number(response.negative_matches || 0)} `
      + `across ${Number(response.word_count || 0)} words`,
      'success'
    );
    setStatus('Prompt sentiment scored.');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to score prompt sentiment.');
    setAdminPersonaPromptModalSentiment(`Sentiment failed: ${message}`, 'error');
    throw err;
  } finally {
    if (els.adminPersonaPromptSentimentBtn) {
      els.adminPersonaPromptSentimentBtn.disabled = false;
      els.adminPersonaPromptSentimentBtn.textContent = 'Score Sentiment';
    }
  }
}

async function askAdminPersonaGraphQuestion() {
  /** Run a graph-only question using the active persona's LLM and enabled Graph RAG chain. */
  const question = String(els.adminPersonaGraphQuestion?.value || '').trim();
  if (!question) {
    setStatus('Persona graph question is required.');
    return;
  }
  const config = getActiveAdminPersonaGraphConfig();
  if (!config) {
    setStatus('Create or select a persona before asking graph questions.');
    return;
  }
  const enabledRags = config.ragSequence.filter((item) => item.enabled);
  const enabledTools = (Array.isArray(config.toolSequence) ? config.toolSequence : []).filter((item) => item.enabled);
  const hasGraphRag = enabledRags.some((item) => item.key === 'graph_rag_neo4j');
  if (!hasGraphRag) {
    setStatus('Enable Graph RAG (Neo4j) in the active persona before asking graph questions.');
    return;
  }
  if (!config.llmProvider || !config.llmModel) {
    setStatus('The active persona must have an LLM configured before asking graph questions.');
    return;
  }

  const traceId = beginTraceSession('chat');
  startUiProcessing('Running persona graph question...');
  startLlmProcessing('Persona graph question is retrieving graph context and generating an answer...');
  setAdminPersonaGraphProcessing(true);
  await nextPaint();
  try {
    const payload = await inferencingApi('/api/graph-rag/query', {
      method: 'POST',
      body: JSON.stringify({
        question,
        top_k: 8,
        use_rag: true,
        stream_rag: true,
        llm_provider: config.llmProvider,
        llm_model: config.llmModel,
        thought_stream_id: traceId,
      }),
    });
    const sources = Array.isArray(payload?.sources) ? payload.sources : [];
    const sourceLines = sources.length
      ? sources.slice(0, 8).map((item) => `- ${item.label} (${item.iri})`).join('\n')
      : '- No matching source nodes returned.';
    const renderedAnswer =
      `Persona: ${config.name}\n`
      + `LLM: ${config.llmProvider}/${config.llmModel}\n`
      + `Enabled RAGs: ${enabledRags.map((item) => getAdminPersonaRagLabel(item)).join(' -> ')}\n\n`
      + `Enabled MCP Tools: ${enabledTools.length ? enabledTools.map((item) => getAdminPersonaToolLabel(item)).join(' -> ') : 'None'}\n\n`
      + `${payload.answer}\n\nSources (${payload.context_rows}):\n${sourceLines}`;
    if (els.adminPersonaGraphAnswer) {
      els.adminPersonaGraphAnswer.value = renderedAnswer;
    }
    const savedPersonaId =
      config.source === 'saved'
        ? selectedAdminPersonaId
        : String(editingAdminPersonaId || '').trim();
    if (savedPersonaId) {
      const persistedPersona = await api(`/api/admin/personas/${encodeURIComponent(savedPersonaId)}/graph-session`, {
        method: 'POST',
        body: JSON.stringify({
          question,
          answer: renderedAnswer,
        }),
      });
      adminPersonasCache = [
        persistedPersona,
        ...adminPersonasCache.filter((item) => item.persona_id !== persistedPersona.persona_id),
      ];
      selectedAdminPersonaId = String(persistedPersona.persona_id || '').trim();
      renderAdminPersonas(adminPersonasCache);
    }
    setStatus(`Persona graph answer generated from ${payload.context_rows} context row(s).`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Persona graph query failed.');
    if (els.adminPersonaGraphAnswer) {
      els.adminPersonaGraphAnswer.value = `Persona graph query failed.\n\n${message}`;
    }
    throw err;
  } finally {
    setAdminPersonaGraphProcessing(false);
    await finalizeTraceSession();
    endLlmProcessing();
    endUiProcessing();
  }
}

function clearAdminPersonaGraphQuestion() {
  /** Reset the persona graph question input and answer area. */
  if (els.adminPersonaGraphQuestion) {
    els.adminPersonaGraphQuestion.value = '';
  }
  if (els.adminPersonaGraphAnswer) {
    els.adminPersonaGraphAnswer.value =
      "Ask a graph question to test the active persona's Graph RAG configuration.";
  }
  renderAdminPersonaGraphMeta();
  setStatus('Cleared the persona graph question and result.');
}

function runAdminPersonaSmokeTestStub() {
  /** Placeholder for Persona smoke testing; implementation intentionally deferred. */
  setAdminPersonaFeedback('Smoke Test is stubbed and not implemented yet.', 'info');
  setStatus('Persona Smoke Test is stubbed and will be implemented later.');
}

function syncAdminPersonaFormPromptSentimentActionState() {
  /** Keep the Persona prompt sentiment action label aligned with computed/visible state. */
  if (!els.adminPersonaFormPromptSentimentBtn) {
    return;
  }
  if (!currentAdminPersonaPromptSentiment) {
    els.adminPersonaFormPromptSentimentBtn.textContent = 'Compute Prompt Sentiment';
    return;
  }
  const hidden = !!els.adminPersonaFormPromptSentiment?.classList.contains('hidden');
  els.adminPersonaFormPromptSentimentBtn.textContent = hidden ? 'Show Prompt Sentiment' : 'Hide Prompt Sentiment';
}

function clearAdminPersonaFormPromptSentiment() {
  /** Reset Persona prompt sentiment panel state for new prompt content or form reset. */
  currentAdminPersonaPromptSentiment = null;
  adminPersonaPromptSentimentDetailOpen = false;
  if (els.adminPersonaFormPromptSentiment) {
    els.adminPersonaFormPromptSentiment.replaceChildren();
    els.adminPersonaFormPromptSentiment.classList.add('hidden');
    els.adminPersonaFormPromptSentiment.classList.remove('detail-sentiment-detailed');
    els.adminPersonaFormPromptSentiment.removeAttribute('role');
    els.adminPersonaFormPromptSentiment.removeAttribute('tabindex');
    els.adminPersonaFormPromptSentiment.removeAttribute('aria-expanded');
  }
  syncAdminPersonaFormPromptSentimentActionState();
}

function hideAdminPersonaFormPromptSentiment({ preserveData = true } = {}) {
  /** Hide Persona prompt sentiment panel while optionally keeping computed data for re-show. */
  if (!preserveData) {
    clearAdminPersonaFormPromptSentiment();
    return;
  }
  adminPersonaPromptSentimentDetailOpen = false;
  if (els.adminPersonaFormPromptSentiment) {
    els.adminPersonaFormPromptSentiment.classList.add('hidden');
    els.adminPersonaFormPromptSentiment.classList.remove('detail-sentiment-detailed');
    els.adminPersonaFormPromptSentiment.removeAttribute('aria-expanded');
  }
  syncAdminPersonaFormPromptSentimentActionState();
}

function buildAdminPersonaPromptSentimentDetailText(sentiment) {
  /** Explain Persona prompt sentiment score and what it implies for runtime behavior. */
  const positive = Number(sentiment?.positive_matches || 0);
  const negative = Number(sentiment?.negative_matches || 0);
  const words = Number(sentiment?.word_count || 0);
  const score = Number(sentiment?.score || 0);
  const label = String(sentiment?.label || 'neutral').trim() || 'neutral';
  const totalMarkers = positive + negative;

  let scoringReason =
    'The prompt tone is near neutral because positive and negative lexical markers are sparse or balanced.';
  if (label === 'positive') {
    scoringReason =
      `The prompt tone trends positive because ${positive} positive markers outweighed ${negative} negative markers across ${words} words.`;
  } else if (label === 'negative') {
    scoringReason =
      `The prompt tone trends negative because ${negative} negative markers outweighed ${positive} positive markers across ${words} words.`;
  } else if (totalMarkers > 0) {
    scoringReason =
      `The prompt tone remains neutral because ${positive} positive and ${negative} negative markers offset each other across ${words} words.`;
  }

  let implication =
    'A neutral prompt tone usually reduces emotional bias and keeps responses more controlled. This score does not evaluate factual correctness.';
  if (label === 'positive') {
    implication =
      'A positive prompt tone may increase cooperative or optimistic phrasing in responses. Monitor for over-assurance in legal narratives.';
  } else if (label === 'negative') {
    implication =
      'A negative prompt tone may increase adversarial or caution-heavy phrasing. Monitor for excessive alarm language or defensive output style.';
  }

  return (
    `Why this score: ${scoringReason}\n` +
    `Signal strength: ${totalMarkers} matched tone markers in ${words} words produced a normalized score of ${formatSentimentScore(score)}.\n` +
    `What this implies: ${implication}`
  );
}

function paintAdminPersonaFormPromptSentimentPanel() {
  /** Paint Persona prompt sentiment panel in summary or detailed explanation mode. */
  if (!els.adminPersonaFormPromptSentiment || !currentAdminPersonaPromptSentiment) {
    clearAdminPersonaFormPromptSentiment();
    return;
  }
  const label =
    currentAdminPersonaPromptSentiment.label.charAt(0).toUpperCase()
    + currentAdminPersonaPromptSentiment.label.slice(1);
  const title = document.createElement('div');
  title.className = 'detail-sentiment-title';
  title.textContent = `Prompt Sentiment: ${label} (${formatSentimentScore(currentAdminPersonaPromptSentiment.score)})`;

  const body = document.createElement('div');
  body.className = 'detail-sentiment-body';
  body.textContent = adminPersonaPromptSentimentDetailOpen
    ? buildAdminPersonaPromptSentimentDetailText(currentAdminPersonaPromptSentiment)
    : (
        `${currentAdminPersonaPromptSentiment.summary}\n`
        + `Positive markers: ${currentAdminPersonaPromptSentiment.positive_matches} | `
        + `Negative markers: ${currentAdminPersonaPromptSentiment.negative_matches} | `
        + `Words: ${currentAdminPersonaPromptSentiment.word_count}`
      );

  const toggle = document.createElement('div');
  toggle.className = 'detail-sentiment-toggle';
  toggle.textContent = adminPersonaPromptSentimentDetailOpen
    ? 'Click to return to the normal prompt sentiment summary.'
    : 'Click to see why this prompt score was assigned and what it implies.';

  els.adminPersonaFormPromptSentiment.replaceChildren(title, body, toggle);
  els.adminPersonaFormPromptSentiment.classList.remove('hidden');
  els.adminPersonaFormPromptSentiment.classList.toggle('detail-sentiment-detailed', adminPersonaPromptSentimentDetailOpen);
  els.adminPersonaFormPromptSentiment.setAttribute('role', 'button');
  els.adminPersonaFormPromptSentiment.setAttribute('tabindex', '0');
  els.adminPersonaFormPromptSentiment.setAttribute('aria-expanded', adminPersonaPromptSentimentDetailOpen ? 'true' : 'false');
  syncAdminPersonaFormPromptSentimentActionState();
}

function renderAdminPersonaFormPromptSentiment(sentiment, { detailOpen = false } = {}) {
  /** Render computed Persona prompt sentiment summary and optional detailed explanation. */
  if (!sentiment || typeof sentiment !== 'object') {
    clearAdminPersonaFormPromptSentiment();
    return;
  }
  currentAdminPersonaPromptSentiment = {
    score: Number(sentiment.score || 0),
    label: String(sentiment.label || 'neutral').trim() || 'neutral',
    summary: String(sentiment.summary || '').trim(),
    positive_matches: Number(sentiment.positive_matches || 0),
    negative_matches: Number(sentiment.negative_matches || 0),
    word_count: Number(sentiment.word_count || 0),
  };
  adminPersonaPromptSentimentDetailOpen = !!detailOpen;
  paintAdminPersonaFormPromptSentimentPanel();
}

function toggleAdminPersonaFormPromptSentimentDetail() {
  /** Toggle Persona prompt sentiment panel between summary and detailed explanation. */
  if (!currentAdminPersonaPromptSentiment || !els.adminPersonaFormPromptSentiment) {
    return;
  }
  if (els.adminPersonaFormPromptSentiment.classList.contains('hidden')) {
    return;
  }
  adminPersonaPromptSentimentDetailOpen = !adminPersonaPromptSentimentDetailOpen;
  paintAdminPersonaFormPromptSentimentPanel();
}

async function scoreAdminPersonaFormPromptSentiment() {
  /** Compute prompt sentiment from current Persona form prompt content and render drill-down panel. */
  if (currentAdminPersonaPromptSentiment) {
    if (els.adminPersonaFormPromptSentiment?.classList.contains('hidden')) {
      renderAdminPersonaFormPromptSentiment(currentAdminPersonaPromptSentiment, {
        detailOpen: adminPersonaPromptSentimentDetailOpen,
      });
      setStatus('Prompt sentiment shown.');
      return;
    }
    hideAdminPersonaFormPromptSentiment({ preserveData: true });
    setStatus('Prompt sentiment hidden.');
    return;
  }
  const promptText = composeLegacyPersonaPromptSections(getAdminPersonaPromptSectionsFromForm());
  if (!promptText) {
    setAdminPersonaFeedback('Enter prompt content before scoring sentiment.', 'error');
    setStatus('Enter prompt content before scoring sentiment.');
    return;
  }
  if (els.adminPersonaFormPromptSentimentBtn) {
    els.adminPersonaFormPromptSentimentBtn.disabled = true;
    els.adminPersonaFormPromptSentimentBtn.textContent = 'Scoring...';
  }
  try {
    const response = await api('/api/text-sentiment', {
      method: 'POST',
      body: JSON.stringify({ text: promptText }),
    });
    renderAdminPersonaFormPromptSentiment(response, { detailOpen: false });
    setStatus('Prompt sentiment computed.');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to score prompt sentiment.');
    setAdminPersonaFeedback(`Prompt sentiment failed: ${message}`, 'error');
    throw err;
  } finally {
    if (els.adminPersonaFormPromptSentimentBtn) {
      els.adminPersonaFormPromptSentimentBtn.disabled = false;
    }
    syncAdminPersonaFormPromptSentimentActionState();
  }
}

function captureAdminPersonaEditorUiState() {
  /** Snapshot Persona editor visibility state so Save Changes can preserve current UI context. */
  return {
    promptPanelOpen: !els.adminPersonaPromptPanel?.classList.contains('hidden'),
    ragPanelOpen: !els.adminPersonaRagPanel?.classList.contains('hidden'),
    promptObservablesPanelOpen: !els.adminPersonaPromptObservablesPanel?.classList.contains('hidden'),
    toolsPanelOpen: !els.adminPersonaToolsPanel?.classList.contains('hidden'),
    systemTemplateOpen: !els.adminPersonaSystemPromptTemplateSelect?.classList.contains('hidden'),
    assistantTemplateOpen: !els.adminPersonaAssistantPromptTemplateSelect?.classList.contains('hidden'),
    contextTemplateOpen: !els.adminPersonaContextPromptTemplateSelect?.classList.contains('hidden'),
  };
}

function restoreAdminPersonaEditorUiState(snapshot) {
  /** Restore Persona editor visibility state after Save Changes refreshes persona data. */
  if (!snapshot || typeof snapshot !== 'object') {
    return;
  }
  const promptOpen = !!snapshot.promptPanelOpen;
  const ragOpen = !!snapshot.ragPanelOpen;
  const promptObservablesOpen = !!snapshot.promptObservablesPanelOpen;
  const toolsOpen = !!snapshot.toolsPanelOpen;

  if (els.adminPersonaPromptPanel) {
    els.adminPersonaPromptPanel.classList.toggle('hidden', !promptOpen);
  }
  if (els.adminPersonaRagPanel) {
    els.adminPersonaRagPanel.classList.toggle('hidden', !ragOpen);
  }
  if (els.adminPersonaPromptObservablesPanel) {
    els.adminPersonaPromptObservablesPanel.classList.toggle('hidden', !promptObservablesOpen);
  }
  if (els.adminPersonaToolsPanel) {
    els.adminPersonaToolsPanel.classList.toggle('hidden', !toolsOpen);
  }
  setAdminPersonaPromptPanelActive(promptOpen);
  setAdminPersonaRagPanelActive(ragOpen);
  setAdminPersonaPromptObservablesPanelActive(promptObservablesOpen);
  setAdminPersonaToolsPanelActive(toolsOpen);

  if (promptOpen) {
    if (els.adminPersonaSystemPromptTemplateSelect) {
      els.adminPersonaSystemPromptTemplateSelect.classList.toggle('hidden', !snapshot.systemTemplateOpen);
    }
    if (els.adminPersonaAssistantPromptTemplateSelect) {
      els.adminPersonaAssistantPromptTemplateSelect.classList.toggle('hidden', !snapshot.assistantTemplateOpen);
    }
    if (els.adminPersonaContextPromptTemplateSelect) {
      els.adminPersonaContextPromptTemplateSelect.classList.toggle('hidden', !snapshot.contextTemplateOpen);
    }
  }
}

function syncAdminPersonaSaveButton() {
  /** Keep the persona save button text aligned to create vs edit mode. */
  if (!els.adminSavePersonaBtn) {
    return;
  }
  els.adminSavePersonaBtn.textContent = editingAdminPersonaId ? 'Save Changes' : 'Save Persona';
}

function setAdminPersonaCreateOpen(open) {
  /** Toggle the expandable add-persona form. */
  const nextOpen = !!open;
  if (els.adminPersonaCreatePanel) {
    els.adminPersonaCreatePanel.classList.toggle('hidden', !nextOpen);
  }
  if (els.adminAddPersonaBtn) {
    els.adminAddPersonaBtn.textContent = nextOpen ? 'Close Persona Details' : 'Add Persona';
  }
  if (nextOpen) {
    syncAdminPersonaLlmOptions();
    syncAdminPersonaPromptTemplateOptions();
    els.adminPersonaName?.focus();
  }
  if (els.adminPersonaPromptPanel) {
    els.adminPersonaPromptPanel.classList.add('hidden');
  }
  if (els.adminPersonaRagPanel) {
    els.adminPersonaRagPanel.classList.add('hidden');
  }
  if (els.adminPersonaPromptObservablesPanel) {
    els.adminPersonaPromptObservablesPanel.classList.add('hidden');
  }
  if (els.adminPersonaToolsPanel) {
    els.adminPersonaToolsPanel.classList.add('hidden');
  }
  setAdminPersonaPromptPanelActive(false);
  setAdminPersonaRagPanelActive(false);
  setAdminPersonaPromptObservablesPanelActive(false);
  setAdminPersonaToolsPanelActive(false);
  if (!nextOpen) {
    clearAdminPersonaFormPromptSentiment();
  }
  renderAdminPersonaGraphMeta();
  syncAdminPersonaSaveButton();
  syncAdminPersonaPromptToggleButtonLabel();
  syncAdminPersonaRagToggleButtonLabel();
  syncAdminPersonaPromptObservablesToggleButtonLabel();
  syncAdminPersonaToolsToggleButtonLabel();
}

function resetAdminPersonaForm() {
  /** Reset persona form fields back to defaults. */
  editingAdminPersonaId = '';
  adminPersonaSelectedPromptTemplateKey = '';
  if (els.adminPersonaName) {
    els.adminPersonaName.value = '';
  }
  syncAdminPersonaLlmOptions();
  syncAdminPersonaPromptTemplateOptions();
  setAdminPersonaPromptTemplateSelectionForKey('');
  setAdminPersonaPromptSectionsInForm({ system: '', assistant: '', context: '' });
  adminPersonaRagSequence = [];
  adminPersonaToolSequence = [];
  syncAdminPersonaRagOptions();
  syncAdminPersonaToolOptions();
  renderAdminPersonaRagSequence();
  renderAdminPersonaToolSequence();
  if (els.adminPersonaPromptPanel) {
    els.adminPersonaPromptPanel.classList.add('hidden');
  }
  if (els.adminPersonaRagPanel) {
    els.adminPersonaRagPanel.classList.add('hidden');
  }
  if (els.adminPersonaPromptObservablesPanel) {
    els.adminPersonaPromptObservablesPanel.classList.add('hidden');
  }
  if (els.adminPersonaToolsPanel) {
    els.adminPersonaToolsPanel.classList.add('hidden');
  }
  setAdminPersonaPromptPanelActive(false);
  setAdminPersonaRagPanelActive(false);
  setAdminPersonaPromptObservablesPanelActive(false);
  setAdminPersonaToolsPanelActive(false);
  clearAdminPersonaFormPromptSentiment();
  renderAdminPersonaGraphMeta();
  syncAdminPersonaSaveButton();
  syncAdminPersonaPromptToggleButtonLabel();
  syncAdminPersonaRagToggleButtonLabel();
  syncAdminPersonaPromptObservablesToggleButtonLabel();
  syncAdminPersonaToolsToggleButtonLabel();
}

function loadAdminPersonaIntoForm(persona) {
  /** Populate the persona form with the selected persona's current values. */
  if (!persona) {
    return;
  }
  editingAdminPersonaId = String(persona.persona_id || '').trim();
  if (els.adminPersonaName) {
    els.adminPersonaName.value = String(persona.name || '').trim();
  }
  syncAdminPersonaLlmOptions();
  if (els.adminPersonaLlm) {
    els.adminPersonaLlm.value = encodeLlmOption(persona.llm_provider, persona.llm_model);
  }
  syncAdminPersonaPromptTemplateOptions();
  adminPersonaSelectedPromptTemplateKey = String(persona.prompt_template_key || '').trim();
  setAdminPersonaPromptTemplateSelectionForKey(adminPersonaSelectedPromptTemplateKey);
  setAdminPersonaPromptSectionsInForm(
    normalizeAdminPersonaPromptSections(
      persona.prompt_sections,
      persona.prompts,
      String(persona.prompt_template_key || '').trim()
    )
  );
  adminPersonaRagSequence = Array.isArray(persona.rag_sequence)
    ? persona.rag_sequence
        .map((item) => normalizeAdminPersonaRagBinding(item))
        .filter((item) => !!item)
    : [];
  adminPersonaToolSequence = Array.isArray(persona.tool_sequence)
    ? persona.tool_sequence
        .map((item) => normalizeAdminPersonaToolBinding(item))
        .filter((item) => !!item)
    : [];
  renderAdminPersonaRagSequence();
  renderAdminPersonaToolSequence();
  clearAdminPersonaFormPromptSentiment();
  renderAdminPersonaGraphMeta();
  renderAdminPersonaStoredGraphSession(persona);
  setAdminPersonaCreateOpen(true);
  setAdminPersonaFeedback(`Editing ${persona.name}. Update the fields and save changes.`, 'info');
  syncAdminPersonaPromptToggleButtonLabel();
  syncAdminPersonaToolsToggleButtonLabel();
}

function highlightAdminPersonaRow(personaId) {
  /** Highlight the saved persona row so the save action is obvious. */
  const normalizedPersonaId = String(personaId || '').trim();
  if (!normalizedPersonaId || !els.adminPersonaList) {
    return;
  }
  const rows = Array.from(els.adminPersonaList.querySelectorAll('.admin-user-row'));
  rows.forEach((row) => row.classList.remove('fresh'));
  const match = rows.find((row) => String(row.dataset.personaId || '').trim() === normalizedPersonaId);
  if (!match) {
    return;
  }
  match.classList.add('fresh');
  match.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  window.setTimeout(() => match.classList.remove('fresh'), 2200);
}

function selectAdminPersonaById(personaId) {
  /** Select one persona from the current list and load it into the form. */
  const normalizedPersonaId = String(personaId || '').trim();
  const selectedPersona = adminPersonasCache.find(
    (item) => String(item.persona_id || '').trim() === normalizedPersonaId
  );
  if (!selectedPersona) {
    return;
  }
  selectedAdminPersonaId = normalizedPersonaId;
  renderAdminPersonas(adminPersonasCache);
  loadAdminPersonaIntoForm(selectedPersona);
}

async function addAdminPersona(options = {}) {
  /** Create or update one persona and refresh the visible list. */
  const closeOnSuccess = options?.closeOnSuccess !== false;
  const uiStateSnapshot = closeOnSuccess ? null : captureAdminPersonaEditorUiState();
  const name = String(els.adminPersonaName?.value || '').trim();
  const promptTemplateKey = String(adminPersonaSelectedPromptTemplateKey || '').trim() || null;
  const promptSections = getAdminPersonaPromptSectionsFromForm();
  const prompts = composeLegacyPersonaPromptSections(promptSections);
  const { provider, model } = decodeLlmOption(els.adminPersonaLlm?.value || '');
  const isEditing = !!editingAdminPersonaId;
  if (!name) {
    els.adminPersonaName?.focus();
    setAdminPersonaFeedback('Enter a persona name before saving.', 'error');
    setStatus('Enter a persona name before saving.');
    return;
  }
  if (!adminPersonaPromptSectionsHaveContent(promptSections)) {
    els.adminPersonaSystemPrompt?.focus();
    setAdminPersonaFeedback('Enter at least one prompt section before saving a persona.', 'error');
    setStatus('Enter at least one prompt section before saving a persona.');
    return;
  }
  if (els.adminSavePersonaBtn) {
    els.adminSavePersonaBtn.disabled = true;
    els.adminSavePersonaBtn.textContent = isEditing ? 'Saving Changes...' : 'Saving...';
  }
  if (els.adminCancelPersonaBtn) {
    els.adminCancelPersonaBtn.disabled = true;
  }
  if (els.adminPersonaSmokeTestBtn) {
    els.adminPersonaSmokeTestBtn.disabled = true;
  }
  if (els.adminAddPersonaBtn) {
    els.adminAddPersonaBtn.disabled = true;
  }
  setAdminPersonaFeedback(isEditing ? 'Saving persona changes...' : 'Saving persona...', 'info');
  try {
    const savedPersona = await api('/api/admin/personas', {
      method: 'POST',
      body: JSON.stringify({
        persona_id: editingAdminPersonaId || null,
        name,
        llm_provider: provider,
        llm_model: model,
        prompt_template_key: promptTemplateKey,
        prompt_sections: promptSections,
        prompts,
        rag_sequence: adminPersonaRagSequence.map((item) => ({ key: item.key, enabled: item.enabled !== false })),
        tool_sequence: adminPersonaToolSequence.map((item) => ({ key: item.key, enabled: item.enabled !== false })),
      }),
    });
    const savedPersonaId = String(savedPersona.persona_id || '').trim();
    selectedAdminPersonaId = savedPersonaId;
    adminPersonasCache = [
      savedPersona,
      ...adminPersonasCache.filter((item) => item.persona_id !== savedPersona.persona_id),
    ];
    renderAdminPersonas(adminPersonasCache);
    highlightAdminPersonaRow(savedPersona.persona_id);
    if (closeOnSuccess) {
      resetAdminPersonaForm();
    } else {
      editingAdminPersonaId = String(savedPersona.persona_id || '').trim();
    }
    await loadAdminPersonas();
    selectedAdminPersonaId = savedPersonaId;
    renderAdminPersonas(adminPersonasCache);
    highlightAdminPersonaRow(savedPersona.persona_id);
    if (!closeOnSuccess) {
      const refreshedPersona = adminPersonasCache.find(
        (item) => String(item.persona_id || '').trim() === selectedAdminPersonaId
      );
      if (refreshedPersona) {
        if (els.adminPersonaName) {
          els.adminPersonaName.value = String(refreshedPersona.name || '').trim();
        }
        if (els.adminPersonaLlm) {
          els.adminPersonaLlm.value = encodeLlmOption(refreshedPersona.llm_provider, refreshedPersona.llm_model);
        }
        adminPersonaSelectedPromptTemplateKey = String(refreshedPersona.prompt_template_key || '').trim();
        setAdminPersonaPromptTemplateSelectionForKey(adminPersonaSelectedPromptTemplateKey);
        setAdminPersonaPromptSectionsInForm(
          normalizeAdminPersonaPromptSections(
            refreshedPersona.prompt_sections,
            refreshedPersona.prompts,
            String(refreshedPersona.prompt_template_key || '').trim()
          )
        );
        adminPersonaRagSequence = Array.isArray(refreshedPersona.rag_sequence)
          ? refreshedPersona.rag_sequence
              .map((item) => normalizeAdminPersonaRagBinding(item))
              .filter((item) => !!item)
          : [];
        adminPersonaToolSequence = Array.isArray(refreshedPersona.tool_sequence)
          ? refreshedPersona.tool_sequence
              .map((item) => normalizeAdminPersonaToolBinding(item))
              .filter((item) => !!item)
          : [];
        renderAdminPersonaRagSequence();
        renderAdminPersonaToolSequence();
      }
      restoreAdminPersonaEditorUiState(uiStateSnapshot);
    }
    setAdminPersonaFeedback(`${savedPersona.name} ${isEditing ? 'updated' : 'saved'}.`, 'success');
    if (closeOnSuccess) {
      setAdminPersonaCreateOpen(false);
    }
    setStatus(`Persona ${isEditing ? 'updated' : 'saved'}.`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to save persona.');
    setAdminPersonaFeedback(message, 'error');
    throw err;
  } finally {
    if (els.adminSavePersonaBtn) {
      els.adminSavePersonaBtn.disabled = false;
      syncAdminPersonaSaveButton();
    }
    if (els.adminCancelPersonaBtn) {
      els.adminCancelPersonaBtn.disabled = false;
    }
    if (els.adminPersonaSmokeTestBtn) {
      els.adminPersonaSmokeTestBtn.disabled = false;
    }
    if (els.adminAddPersonaBtn) {
      els.adminAddPersonaBtn.disabled = false;
    }
  }
}

async function loadAdminTestLog() {
  /** Load the parsed test-log output for the Admin/Test panel. */
  if (!els.adminTestLogSummary || !els.adminTestLogOutput) {
    return;
  }
  const payload = await api('/api/admin/test-log');
  els.adminTestLogSummary.textContent = String(payload?.summary || 'No test log summary available.');
  els.adminTestLogOutput.value = String(payload?.log_output || 'No test log output loaded.');
}

function formatAdminTestDuration(durationMs) {
  /** Format an elapsed duration as MM:SS.hh so sub-second test runs stay visible. */
  const totalCentiseconds = Math.max(0, Math.floor(Number(durationMs || 0) / 10));
  const minutes = Math.floor(totalCentiseconds / 6000);
  const seconds = Math.floor((totalCentiseconds % 6000) / 100);
  const centiseconds = totalCentiseconds % 100;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
}

function setAdminTestRunClockText(label) {
  /** Render one runtime label in the Admin/Test clock. */
  if (!els.adminTestRunClock) {
    return;
  }
  els.adminTestRunClock.textContent = String(label || 'Test runtime: 00:00.00');
}

function stopAdminTestRunClock(finalDurationMs = null) {
  /** Stop the live Admin/Test runtime clock and optionally lock in a final duration. */
  if (adminTestRunClockHandle) {
    window.clearInterval(adminTestRunClockHandle);
    adminTestRunClockHandle = null;
  }
  if (finalDurationMs !== null && finalDurationMs !== undefined) {
    setAdminTestRunClockText(`Last run: ${formatAdminTestDuration(finalDurationMs)}`);
  } else {
    setAdminTestRunClockText('Test runtime: 00:00.00');
  }
  adminTestRunStartedAtMs = 0;
}

function startAdminTestRunClock() {
  /** Start the live Admin/Test runtime clock while pytest is running. */
  if (adminTestRunClockHandle) {
    window.clearInterval(adminTestRunClockHandle);
  }
  adminTestRunStartedAtMs = Date.now();
  setAdminTestRunClockText('Running: 00:00.00');
  adminTestRunClockHandle = window.setInterval(() => {
    const elapsedMs = Date.now() - adminTestRunStartedAtMs;
    setAdminTestRunClockText(`Running: ${formatAdminTestDuration(elapsedMs)}`);
  }, 50);
}

async function runAdminTests() {
  /** Execute the full pytest suite from the Admin/Test panel and refresh the report artifacts. */
  if (!els.adminRunTestsBtn || !els.adminTestLogSummary || !els.adminTestLogOutput) {
    return;
  }
  els.adminRunTestsBtn.disabled = true;
  els.adminRunTestsBtn.textContent = 'Running...';
  startAdminTestRunClock();
  els.adminTestLogSummary.textContent = 'Running the full pytest suite...';
  els.adminTestLogOutput.value = 'Pytest is running. This can take a moment.';
  try {
    const payload = await api('/api/admin/run-tests', {
      method: 'POST',
    });
    els.adminTestLogSummary.textContent = String(payload?.summary || 'Test run completed.');
    els.adminTestLogOutput.value = String(payload?.output || 'Pytest completed without additional console output.');
    stopAdminTestRunClock(Number(payload?.duration_seconds || 0) * 1000);
    await refreshAdminTestView();
    setStatus(String(payload?.summary || 'All tests finished.'));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Failed to run tests.');
    const elapsedMs = adminTestRunStartedAtMs > 0 ? Date.now() - adminTestRunStartedAtMs : 0;
    stopAdminTestRunClock(elapsedMs);
    els.adminTestLogSummary.textContent = `Failed to run tests: ${message}`;
    els.adminTestLogOutput.value = `Run All Tests failed.\n\n${message}`;
    setStatus(`Failed to run tests: ${message}`);
  } finally {
    if (adminTestRunStartedAtMs > 0) {
      stopAdminTestRunClock();
    }
    els.adminRunTestsBtn.disabled = false;
    els.adminRunTestsBtn.textContent = 'Run All Tests';
  }
}

async function refreshAdminTestView() {
  /** Refresh the Admin/Test content and reset the embedded report to its collapsed default state. */
  refreshAdminTestReportFrame();
  await loadAdminTestLog();
}

async function refreshAdminUsersView() {
  /** Refresh the Admin/Users content. */
  await loadAdminUsers();
}

async function refreshAdminPersonasView() {
  /** Refresh the Admin/Personas content. */
  syncAdminPersonaLlmOptions();
  await loadAdminPersonaPromptTemplates();
  await loadAdminPersonaRagOptions();
  await loadAdminPersonaToolOptions();
  await loadAdminPersonas();
}

function setAdminMlopsStatus(message) {
  /** Render one operational status line for the Admin/MLOps panel. */
  if (!els.adminMlopsStatus) {
    return;
  }
  const timestamp = new Date().toLocaleTimeString();
  els.adminMlopsStatus.value = `[${timestamp}] ${String(message || 'MLOps operations are ready.')}`;
}

async function refreshAdminMlopsView() {
  /** Refresh the Admin/MLOps status surface without forcing a backend call. */
  setActiveMlopsSubtab(activeMlopsSubtab);
  setAdminMlopsStatus('MLOps operations are ready.');
}

async function runAdminMlopsAction(actionLabel, fn) {
  /** Execute one MLOps action and write the outcome into the MLOps status box. */
  setAdminMlopsStatus(`${actionLabel}...`);
  try {
    const detail = await fn();
    const suffix = detail ? ` ${detail}` : '';
    setAdminMlopsStatus(`${actionLabel} completed.${suffix}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err || 'Operation failed.');
    setAdminMlopsStatus(`${actionLabel} failed. ${message}`);
    setStatus(message);
  }
}

function openGithubActionsUrl(url) {
  /** Open one GitHub Actions destination in a separate tab. */
  window.open(url, '_blank', 'noopener,noreferrer');
}

function setActiveMlopsSubtab(subtabName) {
  /** Toggle nested MLOps subtabs and keep the selected tool surface visible. */
  const normalized = String(subtabName || '').trim().toLowerCase();
  const nextSubtab = ['llmops', 'fine_tuning', 'deployment', 'cicd'].includes(normalized)
    ? normalized
    : 'llmops';
  activeMlopsSubtab = nextSubtab;

  const isLlmoops = nextSubtab === 'llmops';
  const isFineTuning = nextSubtab === 'fine_tuning';
  const isDeployment = nextSubtab === 'deployment';
  const isCicd = nextSubtab === 'cicd';

  if (els.adminMlopsTabLlmoopsBtn) {
    els.adminMlopsTabLlmoopsBtn.classList.toggle('active', isLlmoops);
    els.adminMlopsTabLlmoopsBtn.setAttribute('aria-selected', isLlmoops ? 'true' : 'false');
  }
  if (els.adminMlopsTabFineTuningBtn) {
    els.adminMlopsTabFineTuningBtn.classList.toggle('active', isFineTuning);
    els.adminMlopsTabFineTuningBtn.setAttribute('aria-selected', isFineTuning ? 'true' : 'false');
  }
  if (els.adminMlopsTabDeploymentBtn) {
    els.adminMlopsTabDeploymentBtn.classList.toggle('active', isDeployment);
    els.adminMlopsTabDeploymentBtn.setAttribute('aria-selected', isDeployment ? 'true' : 'false');
  }
  if (els.adminMlopsTabCicdBtn) {
    els.adminMlopsTabCicdBtn.classList.toggle('active', isCicd);
    els.adminMlopsTabCicdBtn.setAttribute('aria-selected', isCicd ? 'true' : 'false');
  }

  if (els.adminMlopsTabPageLlmoops) {
    els.adminMlopsTabPageLlmoops.classList.toggle('hidden', !isLlmoops);
  }
  if (els.adminMlopsTabPageFineTuning) {
    els.adminMlopsTabPageFineTuning.classList.toggle('hidden', !isFineTuning);
  }
  if (els.adminMlopsTabPageDeployment) {
    els.adminMlopsTabPageDeployment.classList.toggle('hidden', !isDeployment);
  }
  if (els.adminMlopsTabPageCicd) {
    els.adminMlopsTabPageCicd.classList.toggle('hidden', !isCicd);
  }
}

function setActiveAdminSubtab(subtabName) {
  /** Toggle Admin subtabs and refresh the selected panel content. */
  const normalized = String(subtabName || '').trim().toLowerCase();
  const nextSubtab = ['users', 'personas', 'test', 'mlops'].includes(normalized) ? normalized : 'users';
  activeAdminSubtab = nextSubtab;
  const isUsers = nextSubtab === 'users';
  const isPersonas = nextSubtab === 'personas';
  const isTest = nextSubtab === 'test';
  const isMlops = nextSubtab === 'mlops';

  if (els.adminTabTestBtn) {
    els.adminTabTestBtn.classList.toggle('active', isTest);
    els.adminTabTestBtn.setAttribute('aria-selected', isTest ? 'true' : 'false');
  }
  if (els.adminTabUsersBtn) {
    els.adminTabUsersBtn.classList.toggle('active', isUsers);
    els.adminTabUsersBtn.setAttribute('aria-selected', isUsers ? 'true' : 'false');
  }
  if (els.adminTabPersonasBtn) {
    els.adminTabPersonasBtn.classList.toggle('active', isPersonas);
    els.adminTabPersonasBtn.setAttribute('aria-selected', isPersonas ? 'true' : 'false');
  }
  if (els.adminTabMlopsBtn) {
    els.adminTabMlopsBtn.classList.toggle('active', isMlops);
    els.adminTabMlopsBtn.setAttribute('aria-selected', isMlops ? 'true' : 'false');
  }
  if (els.adminTabPageTest) {
    els.adminTabPageTest.classList.toggle('hidden', !isTest);
  }
  if (els.adminTabPageUsers) {
    els.adminTabPageUsers.classList.toggle('hidden', !isUsers);
  }
  if (els.adminTabPagePersonas) {
    els.adminTabPagePersonas.classList.toggle('hidden', !isPersonas);
  }
  if (els.adminTabPageMlops) {
    els.adminTabPageMlops.classList.toggle('hidden', !isMlops);
  }

  const refresh = isTest
    ? refreshAdminTestView
    : isMlops
      ? refreshAdminMlopsView
      : isPersonas
        ? refreshAdminPersonasView
        : refreshAdminUsersView;
  refresh().catch((err) => setStatus(err.message));
}

function setActiveTab(tabName) {
  /** Toggle application top-level pages and maintain tab button active state. */
  const normalized = String(tabName || '').trim().toLowerCase();
  const nextTab = ['landing', 'intelligence', 'provisioning', 'observables', 'admin'].includes(normalized)
    ? normalized
    : 'landing';
  const previousTab = activeTab;
  activeTab = nextTab;

  const isLanding = nextTab === 'landing';
  const isIntelligence = nextTab === 'intelligence';
  const isProvisioning = nextTab === 'provisioning';
  const isObservables = nextTab === 'observables';
  const isAdmin = nextTab === 'admin';

  els.tabPageLanding.classList.toggle('hidden', !isLanding);
  els.tabPageIntelligence.classList.toggle('hidden', !isIntelligence);
  els.tabPageProvisioning.classList.toggle('hidden', !isProvisioning);
  els.tabPageObservables.classList.toggle('hidden', !isObservables);
  els.tabPageAdmin.classList.toggle('hidden', !isAdmin);

  els.tabLandingBtn.classList.toggle('active', isLanding);
  els.tabIntelligenceBtn.classList.toggle('active', isIntelligence);
  els.tabProvisioningBtn.classList.toggle('active', isProvisioning);
  els.tabObservablesBtn.classList.toggle('active', isObservables);
  els.tabAdminBtn.classList.toggle('active', isAdmin);

  els.tabLandingBtn.setAttribute('aria-selected', isLanding ? 'true' : 'false');
  els.tabIntelligenceBtn.setAttribute('aria-selected', isIntelligence ? 'true' : 'false');
  els.tabProvisioningBtn.setAttribute('aria-selected', isProvisioning ? 'true' : 'false');
  els.tabObservablesBtn.setAttribute('aria-selected', isObservables ? 'true' : 'false');
  els.tabAdminBtn.setAttribute('aria-selected', isAdmin ? 'true' : 'false');

  if (previousTab === 'observables' && nextTab !== 'observables') {
    setMetricsPanelOpen(false);
  }

  if (isObservables && previousTab !== 'observables') {
    setMetricsPanelOpen(true);
    renderCorrectnessDriftObservables();
    const hadMetricsLoaded = metricsLoaded;
    loadAgentMetrics({ silent: true })
      .then(() => {
        setStatus(hadMetricsLoaded ? 'Observables refreshed.' : 'Observables loaded.');
      })
      .catch((err) => setStatus(err.message));
  }

  if (isIntelligence) {
    updateTimelineSlots();
    syncTimelineNavButtons();
  }

  if (isAdmin && previousTab !== 'admin') {
    setActiveAdminSubtab(activeAdminSubtab);
  }
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
    setStatus('Observables loaded.');
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

function looksLikeTxtPath(value) {
  /** Return true when one typed deposition source looks like a single text-file path. */
  return String(value || '').trim().toLowerCase().endsWith('.txt');
}

function syncCaseActionState() {
  /** Keep case-index actions aligned with current case id/name values. */
  const hasCaseId = !!els.caseId.value.trim();
  const selectedSource = els.directory.value.trim();
  const hasDirectory = !!selectedSource;
  const canImportIntoDirectory = hasDirectory && !looksLikeTxtPath(selectedSource);
  const selectedModel = els.llmSelect.selectedOptions[0];
  const selectedModelAvailable = !!selectedModel && !selectedModel.disabled;
  els.saveCaseBtn.disabled = uiOpsInFlight > 0 || !hasCaseId || !selectedModelAvailable;
  els.saveIntelligenceBtn.disabled = uiOpsInFlight > 0 || !hasCaseId || !selectedModelAvailable;
  els.duplicateCaseBtn.disabled = uiOpsInFlight > 0 || !hasCaseId;
  els.importDepositionBtn.disabled = uiOpsInFlight > 0 || !canImportIntoDirectory;
  els.importDepositionFolderBtn.disabled = uiOpsInFlight > 0 || !canImportIntoDirectory;
  els.computeSentimentBtn.disabled = uiOpsInFlight > 0 || !selectedDepositionId;
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
  els.importDepositionBtn.disabled = disabled;
  els.importDepositionFolderBtn.disabled = disabled;
  els.ingestBtn.disabled = disabled;
  els.refreshBtn.disabled = disabled;
  els.saveCaseBtn.disabled = disabled || !els.caseId.value.trim();
  els.saveIntelligenceBtn.disabled = disabled || !els.caseId.value.trim();
  els.duplicateCaseBtn.disabled = disabled || !els.caseId.value.trim();
  els.computeSentimentBtn.disabled = disabled || !selectedDepositionId;
  els.refreshCasesBtn.disabled = disabled;
  els.directory.disabled = disabled;
  els.refreshModelsBtn.disabled = disabled;
  els.llmSelect.disabled = disabled;
  els.skipReassess.disabled = disabled;
  els.ontologyPath.disabled = disabled;
  els.ontologyBrowseBtn.disabled = disabled;
  els.loadOntologyBtn.disabled = disabled;
  els.openGraphBrowserBtn.disabled = disabled;
  els.graphRagEmbeddingEnabled.disabled = disabled;
  els.graphRagEmbeddingProvider.disabled = disabled;
  els.graphRagEmbeddingModel.disabled = disabled;
  els.graphRagEmbeddingDimensions.disabled = disabled;
  els.graphRagEmbeddingIndex.disabled = disabled;
  els.graphRagEmbeddingNodeLabel.disabled = disabled;
  els.graphRagEmbeddingProperty.disabled = disabled;
  els.saveGraphRagEmbeddingBtn.disabled = disabled;
  els.reloadGraphRagEmbeddingBtn.disabled = disabled;
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

function syncFocusedReasoningActionState() {
  /** Enable focused re-analysis actions only when source text exists and no reasoning call is active. */
  if (!els.summarizeFocusedReasoningBtn) {
    return;
  }
  const hasSource = !!String(focusedReasoningSourceText || '').trim();
  const reasoningBusy = !els.reasoningProgress.classList.contains('hidden');
  els.summarizeFocusedReasoningBtn.textContent = focusedReasoningIsSummary
    ? 'Full Re-Analysis'
    : 'Summarize';
  els.summarizeFocusedReasoningBtn.disabled = !hasSource || reasoningBusy;
}

function setReasoningProcessing(active, message = '') {
  /** Toggle focused contradiction reasoning spinner/clock state. */
  els.reasoningProgress.classList.toggle('hidden', !active);
  syncFocusedReasoningActionState();
  if (active) {
    if (message) {
      els.focusedReasoningBody.textContent = message;
    }
    startClock('reasoning', els.reasoningClock);
    return;
  }
  stopClock('reasoning');
}

function setFocusedReasoningContent(displayText, { sourceText = displayText, summarized = false } = {}) {
  /** Apply focused reasoning display text while preserving the full unsummarized source payload. */
  focusedReasoningSourceText = String(sourceText || '').trim();
  focusedReasoningIsSummary = !!summarized;
  const visibleText = String(displayText || '').trim();
  if (!visibleText) {
    clearFocusedReasoning();
    return;
  }
  els.focusedReasoning.classList.remove('hidden');
  els.focusedReasoningBody.textContent = visibleText;
  syncFocusedReasoningActionState();
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

function syncIntelligenceLanding() {
  /** Keep the intelligence workspace visible; landing is now a separate tab. */
  if (els.intelligenceWorkspace) {
    els.intelligenceWorkspace.classList.remove('hidden');
  }
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
  syncIntelligenceLanding();
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
    els.timelineScale?.style.removeProperty('--timeline-node-slots');
    return;
  }
  const nextSlotCount = String(timelineVisibleSlots(depositions.length));
  els.timeline.style.setProperty('--timeline-node-slots', nextSlotCount);
  els.timelineScale?.style.setProperty('--timeline-node-slots', nextSlotCount);
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
  const desired = pendingSavedLlmValue;
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
    const selected = operationalValues.includes(desired)
      ? desired
      : values.includes(desired)
        ? operationalValues[0] || desired
        : operationalValues.includes(previous)
          ? previous
          : values.includes(previous)
            ? operationalValues[0] || previous
          : operationalValues.includes(defaultValue)
            ? defaultValue
            : operationalValues[0] || values[0] || defaultValue;
    if (desired) {
      pendingSavedLlmValue = '';
    }

    if (!options.length) {
      const fallback = document.createElement('option');
      fallback.value = defaultValue;
      fallback.textContent = `ChatGPT - ${payload.selected_model}`;
      els.llmSelect.appendChild(fallback);
    }

    els.llmSelect.value = selected;
    syncAdminPersonaLlmOptions();
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

function ensureIngestSchemaFallbackOption() {
  /** Ensure selector still has at least one schema option when the API is unavailable. */
  if (els.ingestSchemaSelect.options.length > 0) {
    return;
  }
  const fallback = document.createElement('option');
  fallback.value = 'deposition_schema';
  fallback.textContent = 'DepositionSchema';
  els.ingestSchemaSelect.appendChild(fallback);
  els.ingestSchemaSelect.value = fallback.value;
}

async function loadIngestSchemaOptions({ silent = false } = {}) {
  /** Fetch available ingest schemas and populate the schema selector. */
  if (!silent) {
    startUiProcessing('Loading ingest schemas...');
  }
  try {
    const previous = String(els.ingestSchemaSelect.value || '').trim();
    const payload = await api('/api/ingest-schemas');
    const options = Array.isArray(payload) ? payload : [];
    ingestSchemaOptionsCache = options;
    els.ingestSchemaSelect.innerHTML = '';

    for (const item of options) {
      const option = document.createElement('option');
      option.value = String(item.key || '').trim();
      option.textContent = String(item.key || 'Unnamed schema').trim();
      option.title = `${String(item.file_name || '').trim()} (${String(item.mode || 'native').trim()})`;
      els.ingestSchemaSelect.appendChild(option);
    }

    ensureIngestSchemaFallbackOption();
    const values = Array.from(els.ingestSchemaSelect.options || []).map((option) =>
      String(option.value || '').trim()
    );
    els.ingestSchemaSelect.value = values.includes(previous)
      ? previous
      : values.includes('deposition_schema')
        ? 'deposition_schema'
        : values[0] || 'deposition_schema';
    syncSelectedIngestSchemaEditor();
    if (!silent) {
      setStatus(`Loaded ${values.length} ingest schema option${values.length === 1 ? '' : 's'}.`);
    }
  } finally {
    if (!silent) {
      endUiProcessing();
    }
  }
}

function setIngestSchemaStatus(message, isError = false) {
  /** Render one local status message for ingest schema management. */
  if (!els.ingestSchemaStatus) {
    return;
  }
  els.ingestSchemaStatus.textContent = String(message || '').trim();
  els.ingestSchemaStatus.classList.toggle('error', !!isError);
  els.ingestSchemaStatus.classList.toggle('muted', !isError);
}

function getSelectedIngestSchemaOption() {
  /** Return metadata for the currently selected ingest schema option. */
  const selectedKey = String(els.ingestSchemaSelect.value || '').trim();
  return (
    ingestSchemaOptionsCache.find((item) => String(item?.key || '').trim() === selectedKey) || null
  );
}

function syncSelectedIngestSchemaEditor() {
  /** Sync the schema editor fields with the currently selected schema option. */
  if (editingNewIngestSchema) {
    return;
  }
  const option = getSelectedIngestSchemaOption();
  if (!option) {
    els.ingestSchemaKey.value = '';
    els.ingestSchemaJson.value = '';
    els.saveIngestSchemaBtn.disabled = false;
    els.removeIngestSchemaBtn.disabled = true;
    setIngestSchemaStatus('Choose a schema path to inspect or create a new custom schema.');
    return;
  }

  els.ingestSchemaKey.value = String(option.key || '').trim();
  els.ingestSchemaJson.value = JSON.stringify(option.schema || option.schema_payload || {}, null, 2);
  const builtin = !!option.builtin;
  els.saveIngestSchemaBtn.disabled = builtin;
  els.removeIngestSchemaBtn.disabled = !option.removable;
  setIngestSchemaStatus(
    builtin
      ? 'Built-in schemas are read-only. Click New Schema to create a custom schema.'
      : 'Custom schema loaded. Save changes or remove it.',
    false
  );
}

function startNewIngestSchemaDraft() {
  /** Reset the schema editor into a blank draft state for a new custom schema. */
  editingNewIngestSchema = true;
  els.ingestSchemaKey.value = '';
  els.ingestSchemaJson.value = JSON.stringify(
    {
      title: 'Custom Schema',
      type: 'object',
      properties: {},
    },
    null,
    2
  );
  els.saveIngestSchemaBtn.disabled = false;
  els.removeIngestSchemaBtn.disabled = true;
  setIngestSchemaStatus('Enter a schema path and JSON payload, then save.');
}

async function saveIngestSchema() {
  /** Persist one custom ingest schema to the backend and reload selector options. */
  const key = String(els.ingestSchemaKey.value || '').trim();
  const rawJson = String(els.ingestSchemaJson.value || '').trim();
  if (!key || !rawJson) {
    setIngestSchemaStatus('Schema key and schema JSON are required.', true);
    return;
  }

  let schemaPayload;
  try {
    schemaPayload = JSON.parse(rawJson);
  } catch (err) {
    setIngestSchemaStatus(`Schema JSON is invalid: ${err.message}`, true);
    return;
  }
  if (!schemaPayload || typeof schemaPayload !== 'object' || Array.isArray(schemaPayload)) {
    setIngestSchemaStatus('Schema JSON must be a JSON object.', true);
    return;
  }

  startUiProcessing('Saving ingest schema...');
  try {
    const payload = await api('/api/ingest-schemas', {
      method: 'POST',
      body: JSON.stringify({
        key,
        schema: schemaPayload,
      }),
    });
    editingNewIngestSchema = false;
    await loadIngestSchemaOptions({ silent: true });
    const savedKey = String(payload?.key || '').trim();
    if (savedKey) {
      els.ingestSchemaSelect.value = savedKey;
    }
    syncSelectedIngestSchemaEditor();
    setIngestSchemaStatus(`Saved custom ingest schema '${savedKey || key}'.`);
    setStatus(`Saved ingest schema ${savedKey || key}.`);
  } finally {
    endUiProcessing();
  }
}

async function removeIngestSchema() {
  /** Permanently delete the currently selected custom ingest schema. */
  const option = getSelectedIngestSchemaOption();
  if (!option) {
    setIngestSchemaStatus('Select a schema first.', true);
    return;
  }
  if (option.builtin || !option.removable) {
    setIngestSchemaStatus('Built-in schemas cannot be removed.', true);
    return;
  }
  const confirmed = window.confirm(`Permanently remove ingest schema "${option.key}"?`);
  if (!confirmed) {
    return;
  }

  startUiProcessing('Removing ingest schema...');
  try {
    await api(`/api/ingest-schemas/${encodeURIComponent(option.key)}`, {
      method: 'DELETE',
    });
    editingNewIngestSchema = false;
    await loadIngestSchemaOptions({ silent: true });
    ensureIngestSchemaFallbackOption();
    if (Array.from(els.ingestSchemaSelect.options || []).some((item) => item.value === 'deposition_schema')) {
      els.ingestSchemaSelect.value = 'deposition_schema';
    }
    syncSelectedIngestSchemaEditor();
    setIngestSchemaStatus(`Removed ingest schema '${option.key}'.`);
    setStatus(`Removed ingest schema ${option.key}.`);
  } finally {
    endUiProcessing();
  }
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

function setDepositionBrowserOpen(open) {
  /** Toggle deposition file-browser modal visibility. */
  els.depositionBrowserModal.classList.toggle('hidden', !open);
}

function renderDepositionBrowserRows(title, items, kind) {
  /** Build one directory/file section markup for the deposition browser. */
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

function renderDepositionBrowser(payload) {
  /** Render deposition browser directory and file rows from API payload. */
  depositionBrowserCurrentDirectory = String(payload?.current_directory || '').trim();
  depositionBrowserParentDirectory = String(payload?.parent_directory || '').trim();
  depositionBrowserWildcardPath = String(payload?.wildcard_path || '').trim();

  const directories = Array.isArray(payload?.directories) ? payload.directories : [];
  const files = Array.isArray(payload?.files) ? payload.files : [];
  const base = String(payload?.base_directory || '').trim();
  els.depositionBrowserTitle.textContent = 'Deposition Browser';
  els.depositionBrowserPath.textContent = `Current folder: ${depositionBrowserCurrentDirectory || base || '(none)'}`;
  els.depositionBrowserUpBtn.disabled = !depositionBrowserParentDirectory;
  els.depositionBrowserUseFolderBtn.disabled = !depositionBrowserCurrentDirectory;
  els.depositionBrowserList.innerHTML = [
    renderDepositionBrowserRows('Folders', directories, 'directory'),
    renderDepositionBrowserRows('TXT Files', files, 'file'),
  ].join('');
}

async function browseDepositionDirectory(path = '') {
  /** Fetch one deposition directory level for modal browser navigation. */
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  const payload = await api(`/api/deposition-browser${query}`);
  renderDepositionBrowser(payload);
}

async function openGrafanaWithCredentials() {
  /** Open Grafana login and surface the configured runtime credentials to the user. */
  const payload = await api('/api/observability/grafana');
  const grafanaUrl = String(
    payload?.dashboard_url || payload?.login_url || payload?.url || 'http://localhost:3000'
  ).trim();
  const username = String(payload?.username || '').trim();
  const password = String(payload?.password || '').trim();
  const credentialText = username || password ? ` Username: ${username} Password: ${password}` : '';
  window.open(grafanaUrl, '_blank', 'noopener,noreferrer');
  setStatus(`Opened Grafana.${credentialText}`.trim());
}

function resolveDepositionBrowserStartPath() {
  /** Resolve the best starting point for browsing from the current deposition input. */
  const raw = String(els.directory?.value || '').trim();
  if (!raw) {
    return depositionBrowserCurrentDirectory || '';
  }

  const wildcardIndex = raw.search(/[*?\[]/);
  if (wildcardIndex !== -1) {
    const prefix = raw.slice(0, wildcardIndex).replace(/[\\/]+$/, '');
    if (prefix) {
      return prefix;
    }
    if (raw.startsWith('/')) {
      return '/';
    }
    return depositionBrowserCurrentDirectory || '';
  }

  return raw;
}

async function openDepositionBrowser() {
  /** Open deposition browser modal and load its starting directory rows. */
  setDepositionBrowserOpen(true);
  await browseDepositionDirectory(resolveDepositionBrowserStartPath());
}

async function handleDepositionBrowserListClick(event) {
  /** Handle click actions on deposition browser rows (navigate/select). */
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
    await browseDepositionDirectory(path);
    return;
  }
  els.directory.value = path;
  setDepositionBrowserOpen(false);
  syncCaseActionState();
  setStatus(`Deposition file selected: ${path}`);
}

function promptImportDeposition() {
  /** Open the file picker for uploading new deposition text files into the selected folder. */
  const selected = els.directory.value.trim();
  if (!selected) {
    setStatus('Select a deposition directory before importing a deposition.');
    return;
  }
  if (looksLikeTxtPath(selected)) {
    setStatus('Import Deposition writes into a directory. Select a deposition directory, not a single .txt file.');
    return;
  }
  els.importDepositionInput.click();
}

function promptImportDepositionFolder() {
  /** Open the folder picker for uploading every ``.txt`` file inside one selected folder. */
  const selected = els.directory.value.trim();
  if (!selected) {
    setStatus('Select a deposition directory before importing a folder.');
    return;
  }
  if (looksLikeTxtPath(selected)) {
    setStatus('Import Folder writes into a directory. Select a deposition directory, not a single .txt file.');
    return;
  }
  els.importDepositionFolderInput.click();
}

function normalizeImportedDepositionFiles(fileList, fromFolder = false) {
  /** Keep only ``.txt`` uploads and preserve folder-relative names when importing a folder. */
  const files = Array.isArray(fileList) ? fileList : [];
  return files
    .filter((file) => String(file?.name || '').toLowerCase().endsWith('.txt'))
    .map((file) => {
      if (!fromFolder) {
        return { file, uploadName: file.name };
      }
      const relativePath = String(file.webkitRelativePath || file.name || '').trim();
      const uploadName = relativePath ? relativePath.replace(/[\\/]+/g, '__') : file.name;
      return { file, uploadName };
    });
}

async function importSelectedDepositionFilesFromInput(inputEl, { fromFolder = false } = {}) {
  /** Upload chosen deposition files into the current folder and re-ingest the current case. */
  const directory = els.directory.value.trim();
  const rawFiles = Array.from(inputEl?.files || []);
  const files = normalizeImportedDepositionFiles(rawFiles, fromFolder);
  if (!directory) {
    setStatus(
      fromFolder
        ? 'Select a deposition directory before importing a folder.'
        : 'Select a deposition directory before importing a deposition.'
    );
    inputEl.value = '';
    return;
  }
  if (looksLikeTxtPath(directory)) {
    setStatus(
      fromFolder
        ? 'Import Folder requires a deposition directory target, not a single .txt file.'
        : 'Import Deposition requires a deposition directory target, not a single .txt file.'
    );
    inputEl.value = '';
    return;
  }
  if (!rawFiles.length) {
    return;
  }
  if (!files.length) {
    setStatus(fromFolder ? 'No .txt files were found in the selected folder.' : 'Choose at least one .txt file.');
    inputEl.value = '';
    return;
  }

  const formData = new FormData();
  formData.append('directory', directory);
  for (const item of files) {
    formData.append('files', item.file, item.uploadName);
  }

  startUiProcessing(fromFolder ? 'Importing deposition folder...' : 'Importing deposition files...');
  try {
    const payload = await api('/api/depositions/upload', {
      method: 'POST',
      body: formData,
    });
    await loadDirectoryOptions({ silent: true });
    if (els.caseId.value.trim()) {
      setStatus(
        `Imported ${payload.file_count} deposition ${payload.file_count === 1 ? 'file' : 'files'}. Re-ingesting current folder...`
      );
      await ingestCase();
      return;
    }
    setStatus(`Imported ${payload.file_count} deposition file(s) into ${payload.directory}.`);
  } finally {
    inputEl.value = '';
    endUiProcessing();
  }
}

async function importSelectedDepositionFiles() {
  /** Upload selected ``.txt`` files from the file picker. */
  await importSelectedDepositionFilesFromInput(els.importDepositionInput, { fromFolder: false });
}

async function importSelectedDepositionFolder() {
  /** Upload every ``.txt`` file found in the chosen folder selection. */
  await importSelectedDepositionFilesFromInput(els.importDepositionFolderInput, { fromFolder: true });
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

function resolveOntologyBrowserStartPath() {
  /** Resolve the best starting location for the ontology browser from the current input value. */
  const raw = String(els.ontologyPath?.value || '').trim();
  if (!raw) {
    return ontologyBrowserCurrentDirectory || '';
  }

  const wildcardIndex = raw.search(/[*?\[]/);
  if (wildcardIndex !== -1) {
    const prefix = raw.slice(0, wildcardIndex).replace(/[\\/]+$/, '');
    if (prefix) {
      return prefix;
    }
    if (raw.startsWith('/')) {
      return '/';
    }
    return ontologyBrowserCurrentDirectory || '';
  }

  return raw;
}

async function openOntologyBrowser() {
  /** Open ontology file browser modal and load initial directory rows. */
  setOntologyBrowserOpen(true);
  await browseOntologyDirectory(resolveOntologyBrowserStartPath());
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

function resolveSavedLlmValue(caseSummary, snapshot = null) {
  /** Resolve the best saved LLM selector value from case summary and persisted snapshot. */
  const snapshotValue = String(snapshot?.dropdowns?.llm_selected || '').trim();
  if (snapshotValue) {
    return snapshotValue;
  }
  const provider = String(caseSummary?.last_llm_provider || '').trim().toLowerCase();
  const model = String(caseSummary?.last_llm_model || '').trim();
  if (provider && model) {
    return encodeLlmOption(provider, model);
  }
  return '';
}

function resolveSavedIngestSchemaValue(snapshot = null) {
  /** Resolve the best saved ingest schema selector value from a persisted snapshot. */
  const snapshotValue = String(
    snapshot?.dropdowns?.ingest_schema_selected || snapshot?.controls?.ingest_schema || ''
  ).trim();
  return snapshotValue || '';
}

function firstOperationalLlmValue() {
  /** Return the first enabled LLM selector option value, or empty string when unavailable. */
  const option = Array.from(els.llmSelect.options || []).find((item) => !item.disabled);
  return option ? String(option.value || '').trim() : '';
}

function applySavedLlmSelection(caseSummary, snapshot = null) {
  /** Apply one saved LLM choice into the selector when that option is available. */
  const desiredValue = resolveSavedLlmValue(caseSummary, snapshot);
  if (!desiredValue) {
    return false;
  }
  pendingSavedLlmValue = desiredValue;
  const options = Array.from(els.llmSelect.options || []);
  if (!options.length) {
    return false;
  }
  const desiredOption = options.find((option) => option.value === desiredValue) || null;
  if (desiredOption && !desiredOption.disabled) {
    els.llmSelect.value = desiredValue;
    pendingSavedLlmValue = '';
    syncCaseActionState();
    return true;
  }
  const fallbackValue = firstOperationalLlmValue();
  if (fallbackValue) {
    els.llmSelect.value = fallbackValue;
    pendingSavedLlmValue = '';
    syncCaseActionState();
    return false;
  }
  if (desiredOption) {
    els.llmSelect.value = desiredValue;
    pendingSavedLlmValue = '';
  }
  syncCaseActionState();
  return false;
}

function applySavedIngestSchemaSelection(snapshot = null) {
  /** Apply one saved ingest schema choice into the selector when that option exists. */
  const desiredValue = resolveSavedIngestSchemaValue(snapshot);
  if (!desiredValue) {
    return false;
  }
  const option = Array.from(els.ingestSchemaSelect.options || []).find(
    (item) => item.value === desiredValue
  );
  if (!option) {
    return false;
  }
  els.ingestSchemaSelect.value = desiredValue;
  syncSelectedIngestSchemaEditor();
  return true;
}

function applyCaseSelection(caseSummary) {
  /** Apply selected case id and saved folder path into form controls. */
  if (!caseSummary) {
    els.caseId.value = '';
    els.directory.value = '';
    ensureIngestSchemaFallbackOption();
    els.ingestSchemaSelect.value = 'deposition_schema';
    writeLastUsedCaseId('');
    syncCaseActionState();
    return;
  }
  els.caseId.value = caseSummary.case_id || '';
  els.directory.value = normalizeCaseDirectory(caseSummary);
  applySavedLlmSelection(caseSummary);
  writeLastUsedCaseId(caseSummary.case_id || '');
  syncCaseActionState();
  syncIntelligenceLanding();
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
    const storedCaseId = readLastUsedCaseId();
    const activeCaseId = previousCaseId || storedCaseId;
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
  if (els.timelineScale) {
    els.timelineScale.innerHTML = '';
  }
  if (!depositions.length) {
    updateTimelineSlots();
    syncTimelineNavButtons();
    els.timeline.innerHTML = '<div class="muted">Timeline will appear after loading depositions.</div>';
    if (els.timelineScale) {
      els.timelineScale.innerHTML = '<div class="muted">Time scale will appear after loading depositions.</div>';
    }
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
    if (els.timelineScale) {
      const scaleItem = document.createElement('div');
      scaleItem.className = 'timeline-scale-item';
      scaleItem.textContent = displayDate(dep.deposition_date);
      els.timelineScale.appendChild(scaleItem);
    }
  }
  if (els.timelineScale) {
    els.timelineScale.scrollLeft = els.timeline.scrollLeft;
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

function clearDepositionSentiment() {
  /** Reset deposition-wide sentiment panel state for a new or cleared selection. */
  currentDepositionSentiment = null;
  depositionSentimentDetailOpen = false;
  els.detailSentiment.replaceChildren();
  els.detailSentiment.classList.add('hidden');
  els.detailSentiment.classList.remove('detail-sentiment-detailed');
  els.detailSentiment.removeAttribute('role');
  els.detailSentiment.removeAttribute('tabindex');
  els.detailSentiment.removeAttribute('aria-expanded');
  els.computeSentimentBtn.disabled = true;
  els.computeSentimentBtn.textContent = 'Compute Sentiment';
}

function syncDepositionSentimentActionState() {
  /** Keep the sentiment action label aligned with whether sentiment is computed and visible. */
  if (!els.computeSentimentBtn) {
    return;
  }
  if (!currentDepositionSentiment) {
    els.computeSentimentBtn.textContent = 'Compute Sentiment';
    return;
  }
  els.computeSentimentBtn.textContent = els.detailSentiment.classList.contains('hidden')
    ? 'Show Sentiment'
    : 'Hide Sentiment';
}

function hideDepositionSentiment({ preserveData = true } = {}) {
  /** Hide the rendered sentiment panel, optionally retaining the computed result for re-show. */
  if (!preserveData) {
    clearDepositionSentiment();
    return;
  }
  depositionSentimentDetailOpen = false;
  els.detailSentiment.classList.add('hidden');
  els.detailSentiment.classList.remove('detail-sentiment-detailed');
  els.detailSentiment.removeAttribute('aria-expanded');
  els.computeSentimentBtn.disabled = false;
  syncDepositionSentimentActionState();
}

function formatSentimentScore(score) {
  /** Format one sentiment score with an explicit sign for readability. */
  const numeric = Number(score || 0);
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(2)}`;
}

function buildDepositionSentimentDetailText(sentiment) {
  /** Explain why the current sentiment score was assigned and what it implies. */
  const positive = Number(sentiment?.positive_matches || 0);
  const negative = Number(sentiment?.negative_matches || 0);
  const words = Number(sentiment?.word_count || 0);
  const score = Number(sentiment?.score || 0);
  const label = String(sentiment?.label || 'neutral').trim() || 'neutral';
  const totalMarkers = positive + negative;

  let scoringReason =
    'The score is near zero because the lexical tone markers were either sparse or nearly balanced.';
  if (label === 'positive') {
    scoringReason =
      `The score trends positive because ${positive} positive tone markers outweighed ${negative} negative markers across ${words} words.`;
  } else if (label === 'negative') {
    scoringReason =
      `The score trends negative because ${negative} negative tone markers outweighed ${positive} positive markers across ${words} words.`;
  } else if (totalMarkers > 0) {
    scoringReason =
      `The score remains neutral because ${positive} positive and ${negative} negative markers were close enough to offset each other across ${words} words.`;
  }

  let implication =
    'This implies the deposition tone is broadly balanced. That does not prove the testimony is accurate; it only means the language reads more even than emotionally loaded.';
  if (label === 'positive') {
    implication =
      'This implies the testimony reads relatively composed, cooperative, or stable in tone. It can support perceived clarity, but it is not evidence that the substance is true.';
  } else if (label === 'negative') {
    implication =
      'This implies the testimony carries more adversarial, distressed, or conflict-heavy language. That can flag emotional pressure or dispute intensity, but it is not a legal conclusion by itself.';
  }

  return (
    `Why this score: ${scoringReason}\n` +
    `Signal strength: ${totalMarkers} matched tone markers were found in ${words} words, producing a normalized score of ${formatSentimentScore(score)}.\n` +
    `What this implies: ${implication}`
  );
}

function paintDepositionSentimentPanel() {
  /** Paint the current sentiment card in summary or detailed mode. */
  if (!currentDepositionSentiment || typeof currentDepositionSentiment !== 'object') {
    clearDepositionSentiment();
    return;
  }

  const label = currentDepositionSentiment.label.charAt(0).toUpperCase() + currentDepositionSentiment.label.slice(1);
  const title = document.createElement('div');
  title.className = 'detail-sentiment-title';
  title.textContent = `Sentiment: ${label} (${formatSentimentScore(currentDepositionSentiment.score)})`;

  const body = document.createElement('div');
  body.className = 'detail-sentiment-body';
  body.textContent = depositionSentimentDetailOpen
    ? buildDepositionSentimentDetailText(currentDepositionSentiment)
    : (
        `${currentDepositionSentiment.summary}\n` +
        `Positive markers: ${currentDepositionSentiment.positive_matches} | ` +
        `Negative markers: ${currentDepositionSentiment.negative_matches} | ` +
        `Words: ${currentDepositionSentiment.word_count}`
      );

  const toggle = document.createElement('div');
  toggle.className = 'detail-sentiment-toggle';
  toggle.textContent = depositionSentimentDetailOpen
    ? 'Click to return to the normal sentiment summary.'
    : 'Click to see why this score was assigned and what it implies.';

  els.detailSentiment.replaceChildren(title, body, toggle);
  els.detailSentiment.classList.remove('hidden');
  els.detailSentiment.classList.toggle('detail-sentiment-detailed', depositionSentimentDetailOpen);
  els.detailSentiment.setAttribute('role', 'button');
  els.detailSentiment.setAttribute('tabindex', '0');
  els.detailSentiment.setAttribute('aria-expanded', depositionSentimentDetailOpen ? 'true' : 'false');
  els.computeSentimentBtn.disabled = false;
  syncDepositionSentimentActionState();
}

function renderDepositionSentiment(sentiment, { detailOpen = false } = {}) {
  /** Render deposition-wide sentiment summary inside the conflict detail panel. */
  if (!sentiment || typeof sentiment !== 'object') {
    clearDepositionSentiment();
    return;
  }
  depositionSentimentDetailOpen = !!detailOpen;
  currentDepositionSentiment = {
    score: Number(sentiment.score || 0),
    label: String(sentiment.label || 'neutral').trim() || 'neutral',
    summary: String(sentiment.summary || '').trim(),
    positive_matches: Number(sentiment.positive_matches || 0),
    negative_matches: Number(sentiment.negative_matches || 0),
    word_count: Number(sentiment.word_count || 0),
  };
  paintDepositionSentimentPanel();
}

function restoreHiddenDepositionSentiment(sentiment, { detailOpen = false } = {}) {
  /** Restore a previously computed sentiment result without reopening the visible panel. */
  if (!sentiment || typeof sentiment !== 'object') {
    clearDepositionSentiment();
    return;
  }
  depositionSentimentDetailOpen = !!detailOpen;
  currentDepositionSentiment = {
    score: Number(sentiment.score || 0),
    label: String(sentiment.label || 'neutral').trim() || 'neutral',
    summary: String(sentiment.summary || '').trim(),
    positive_matches: Number(sentiment.positive_matches || 0),
    negative_matches: Number(sentiment.negative_matches || 0),
    word_count: Number(sentiment.word_count || 0),
  };
  hideDepositionSentiment({ preserveData: true });
}

async function toggleDepositionSentimentDetail() {
  /** Toggle the sentiment panel between summary and detailed explanation. */
  if (!currentDepositionSentiment || els.detailSentiment.classList.contains('hidden')) {
    return;
  }
  depositionSentimentDetailOpen = !depositionSentimentDetailOpen;
  paintDepositionSentimentPanel();
  await persistCurrentCaseSnapshot();
}

function clearFocusedReasoning() {
  /** Reset focused contradiction reasoning panel state. */
  els.focusedReasoning.classList.add('hidden');
  focusedReasoningSourceText = '';
  focusedReasoningIsSummary = false;
  setReasoningProcessing(false);
  els.focusedReasoningBody.textContent = '';
  syncFocusedReasoningActionState();
}

function restoreDepositionSentimentFromCaseSnapshot(snapshot) {
  /** Restore saved deposition sentiment state for the currently selected deposition. */
  const conflictDetail =
    snapshot && typeof snapshot.conflict_detail === 'object' ? snapshot.conflict_detail : null;
  const sentiment =
    conflictDetail && conflictDetail.sentiment && typeof conflictDetail.sentiment === 'object'
      ? conflictDetail.sentiment
      : null;
  if (!sentiment) {
    return false;
  }
  const renderState = {
    detailOpen: !!conflictDetail.sentiment_detail_view,
  };
  if (conflictDetail.sentiment_visible ?? true) {
    renderDepositionSentiment(sentiment, renderState);
  } else {
    restoreHiddenDepositionSentiment(sentiment, renderState);
  }
  return true;
}

function renderDetail(dep) {
  /** Render selected deposition details and clickable contradiction items. */
  if (!dep) {
    els.detailEmpty.classList.remove('hidden');
    els.detailBody.classList.add('hidden');
    clearDepositionSentiment();
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
  clearDepositionSentiment();
  clearFocusedReasoning();
  els.computeSentimentBtn.disabled = false;
  syncDepositionSentimentActionState();

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

    setFocusedReasoningContent(payload.response, {
      sourceText: payload.response,
      summarized: false,
    });
    await persistCurrentCaseSnapshot();
  } finally {
    setReasoningProcessing(false);
    endLlmProcessing();
  }
}

async function computeDepositionSentiment() {
  /** Compute a deterministic sentiment summary across the full selected deposition text. */
  if (!selectedDepositionId || !loadedCaseId) {
    setStatus('Load a case and select a deposition first.');
    return;
  }
  if (currentDepositionSentiment) {
    if (els.detailSentiment.classList.contains('hidden')) {
      paintDepositionSentimentPanel();
      await persistCurrentCaseSnapshot();
      setStatus('Deposition sentiment shown.');
      return;
    }
    hideDepositionSentiment({ preserveData: true });
    await persistCurrentCaseSnapshot();
    setStatus('Deposition sentiment hidden.');
    return;
  }

  startUiProcessing('Computing deposition sentiment...');
  try {
    const payload = await api('/api/deposition-sentiment', {
      method: 'POST',
      body: JSON.stringify({
        case_id: loadedCaseId,
        deposition_id: selectedDepositionId,
      }),
    });
    renderDepositionSentiment(payload);
    await persistCurrentCaseSnapshot();
    setStatus('Deposition sentiment computed.');
  } finally {
    endUiProcessing();
  }
}

async function summarizeFocusedReasoning() {
  /** Toggle focused re-analysis between full text and summarized text. */
  const sourceText = String(focusedReasoningSourceText || '').trim();
  if (!sourceText) {
    setStatus('Run a focused re-analysis first.');
    return;
  }
  if (focusedReasoningIsSummary) {
    setFocusedReasoningContent(sourceText, {
      sourceText,
      summarized: false,
    });
    await persistCurrentCaseSnapshot();
    setStatus('Restored full focused re-analysis.');
    return;
  }
  if (!selectedDepositionId || !loadedCaseId) {
    setStatus('Load a case and select a deposition first.');
    return;
  }
  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Click Refresh Models and choose an operational model.');
    return;
  }

  const llm = getSelectedLlm();
  els.focusedReasoning.classList.remove('hidden');
  startLlmProcessing('Persona:Attorney is summarizing the focused re-analysis...');
  setReasoningProcessing(true, 'Summarizing the focused re-analysis...');
  await nextPaint();

  try {
    const payload = await inferencingApi('/api/summarize-focused-reasoning', {
      method: 'POST',
      body: JSON.stringify({
        case_id: loadedCaseId,
        deposition_id: selectedDepositionId,
        reasoning_text: sourceText,
        llm_provider: llm.provider,
        llm_model: llm.model,
      }),
    });
    setFocusedReasoningContent(payload.summary, {
      sourceText,
      summarized: true,
    });
    await persistCurrentCaseSnapshot();
    setStatus('Focused re-analysis summarized.');
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

async function loadSavedCaseSnapshot(caseId) {
  /** Load persisted case snapshot state for one case id (best-effort). */
  const normalized = String(caseId || '').trim();
  if (!normalized) {
    return null;
  }
  try {
    const payload = await api(`/api/cases/${encodeURIComponent(normalized)}`);
    return payload && typeof payload.snapshot === 'object' && payload.snapshot ? payload.snapshot : null;
  } catch (_err) {
    return null;
  }
}

function restoreChatFromCaseSnapshot(snapshot) {
  /** Restore saved Persona:Attorney chat transcript and draft input from a case snapshot. */
  const chat = snapshot && typeof snapshot.chat === 'object' ? snapshot.chat : null;
  if (!chat) {
    return false;
  }

  const history = Array.isArray(chat.history)
    ? chat.history
        .filter((item) => item && typeof item === 'object')
        .map((item) => ({
          role: item.role === 'user' ? 'user' : 'assistant',
          content: String(item.content || '').trim(),
        }))
        .filter((item) => item.content)
    : [];
  const visibleMessages = Array.isArray(chat.visible_messages)
    ? chat.visible_messages
        .filter((item) => item && typeof item === 'object')
        .map((item) => ({
          role: item.role === 'user' ? 'user' : 'assistant',
          content: String(item.content || '').trim(),
        }))
        .filter((item) => item.content)
    : [];

  chatHistory = history;
  els.chatMessages.innerHTML = '';
  const messagesToRender = visibleMessages.length ? visibleMessages : history;
  messagesToRender.forEach((item) => addMessage(item.role, item.content));
  els.chatInput.value = String(chat.draft_input || '');
  return messagesToRender.length > 0 || history.length > 0 || !!els.chatInput.value;
}

function restoreFocusedReasoningFromCaseSnapshot(snapshot) {
  /** Restore saved focused re-analysis display/source state for the active selected deposition. */
  const conflictDetail =
    snapshot && typeof snapshot.conflict_detail === 'object' ? snapshot.conflict_detail : null;
  if (!conflictDetail) {
    return false;
  }

  const displayText = String(
    conflictDetail.focused_reasoning_display || conflictDetail.focused_reasoning || ''
  ).trim();
  if (!displayText) {
    return false;
  }

  const sourceText = String(conflictDetail.focused_reasoning_source || displayText).trim() || displayText;
  const visible = Boolean(conflictDetail.focused_reasoning_visible ?? true);
  if (!visible) {
    focusedReasoningSourceText = sourceText;
    focusedReasoningIsSummary = !!conflictDetail.focused_reasoning_is_summary;
    syncFocusedReasoningActionState();
    return false;
  }

  setFocusedReasoningContent(displayText, {
    sourceText,
    summarized: !!conflictDetail.focused_reasoning_is_summary,
  });
  return true;
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
  const previousLoadedCaseId = loadedCaseId;
  const currentCaseSummary = cases.find((item) => item.case_id === caseId) || null;
  renderCaseIndex();
  syncCaseActionState();
  if (!caseId) {
    setStatus('Enter a case ID first.');
    return;
  }

  startUiProcessing(`Loading case ${caseId}...`);
  try {
    depositions = await api(`/api/depositions/${encodeURIComponent(caseId)}`);
    const shouldRestoreSavedState = caseId !== previousLoadedCaseId;
    const savedSnapshot = shouldRestoreSavedState ? await loadSavedCaseSnapshot(caseId) : null;
    if (shouldRestoreSavedState) {
      applySavedLlmSelection(currentCaseSummary, savedSnapshot);
      applySavedIngestSchemaSelection(savedSnapshot);
    }
    loadedCaseId = caseId;
    writeLastUsedCaseId(caseId);

    if (selectedDepositionId) {
      const stillExists = depositions.some((item) => item._id === selectedDepositionId);
      if (!stillExists) {
        selectedDepositionId = null;
      }
    }

    const snapshotSelectedDepositionId = String(savedSnapshot?.selected_deposition_id || '').trim();
    if (!selectedDepositionId && snapshotSelectedDepositionId) {
      const snapshotExists = depositions.some((item) => item._id === snapshotSelectedDepositionId);
      if (snapshotExists) {
        selectedDepositionId = snapshotSelectedDepositionId;
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
      if (shouldRestoreSavedState) {
        const canRestoreSelectedState =
          !snapshotSelectedDepositionId || snapshotSelectedDepositionId === selectedDepositionId;
        if (canRestoreSelectedState) {
          restoreDepositionSentimentFromCaseSnapshot(savedSnapshot);
          restoreFocusedReasoningFromCaseSnapshot(savedSnapshot);
        }
        const restoredChat = canRestoreSelectedState ? restoreChatFromCaseSnapshot(savedSnapshot) : false;
        if (!restoredChat) {
          resetChatForSelectedDeposition();
        }
      }
      if (autoSelected) {
        setStatus(
          `Loaded ${depositions.length} depositions. Auto-selected ${selected.witness_name || 'a witness'} for Persona:Attorney chat.`
        );
        return;
      }
    } else {
      renderDetail(null);
      if (shouldRestoreSavedState) {
        restoreChatFromCaseSnapshot(savedSnapshot);
      }
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

function buildCaseSaveSnapshot() {
  /** Capture the current case-related UI state so Save Case persists the active working context. */
  const llm = getSelectedLlm();
  const selectedDeposition = depositions.find((item) => item._id === selectedDepositionId) || null;
  const dropdownOptions = {
    llm_selected: els.llmSelect.value,
    ingest_schema_selected: String(els.ingestSchemaSelect.value || '').trim() || 'deposition_schema',
    llm_options: Array.from(els.llmSelect.options || []).map((option) => ({
      value: String(option.value || ''),
      label: String(option.textContent || '').trim(),
      disabled: !!option.disabled,
      selected: !!option.selected,
    })),
    directory_options: Array.from(els.directoryOptions?.options || [])
      .map((option) => String(option.value || '').trim())
      .filter(Boolean),
    ontology_options: Array.from(els.ontologyOptions?.options || [])
      .map((option) => String(option.value || '').trim())
      .filter(Boolean),
  };
  const contradictionItems = Array.from(els.detailContradictions?.querySelectorAll('button') || [])
    .map((item) => String(item.textContent || '').trim())
    .filter(Boolean);
  const chatMessages = Array.from(els.chatMessages?.querySelectorAll('.msg') || []).map((item) => ({
    role: item.classList.contains('user') ? 'user' : 'assistant',
    content: String(item.textContent || '').trim(),
  }));

  return {
    saved_at_client: new Date().toISOString(),
    loaded_case_id: loadedCaseId || '',
    selected_deposition_id: selectedDepositionId || null,
    deposition_count: depositions.length,
    depositions: depositions.map((item) => ({
      deposition_id: item._id,
      file_name: item.file_name,
      witness_name: item.witness_name,
      witness_role: item.witness_role,
      contradiction_score: item.contradiction_score,
      flagged: item.flagged,
    })),
    selected_deposition: selectedDeposition
      ? {
          deposition_id: selectedDeposition._id,
          file_name: selectedDeposition.file_name,
          witness_name: selectedDeposition.witness_name,
          witness_role: selectedDeposition.witness_role,
          contradiction_score: selectedDeposition.contradiction_score,
          flagged: selectedDeposition.flagged,
        }
      : null,
    controls: {
      directory: els.directory.value.trim(),
      skip_reassess: !!els.skipReassess.checked,
      thought_stream_enabled: !!els.traceStreamToggle.checked,
      llm_provider: llm.provider,
      llm_model: llm.model,
      llm_label: getSelectedLlmLabel(),
      ingest_schema: String(els.ingestSchemaSelect.value || '').trim() || 'deposition_schema',
    },
    dropdowns: dropdownOptions,
    conflict_detail: {
      witness: String(els.detailWitness?.textContent || '').trim(),
      summary: String(els.detailSummary?.textContent || '').trim(),
      short_answer: String(els.detailExplanation?.textContent || '').trim(),
      sentiment: currentDepositionSentiment
        ? {
            score: currentDepositionSentiment.score,
            label: currentDepositionSentiment.label,
            summary: currentDepositionSentiment.summary,
            positive_matches: currentDepositionSentiment.positive_matches,
            negative_matches: currentDepositionSentiment.negative_matches,
            word_count: currentDepositionSentiment.word_count,
          }
        : null,
      sentiment_visible: !els.detailSentiment.classList.contains('hidden'),
      sentiment_detail_view: !!depositionSentimentDetailOpen,
      contradictions: contradictionItems,
      focused_reasoning: String(els.focusedReasoningBody?.textContent || '').trim(),
      focused_reasoning_display: String(els.focusedReasoningBody?.textContent || '').trim(),
      focused_reasoning_source: String(focusedReasoningSourceText || '').trim(),
      focused_reasoning_is_summary: !!focusedReasoningIsSummary,
      focused_reasoning_visible: !els.focusedReasoning.classList.contains('hidden'),
    },
    chat: {
      history: chatHistory.map((item) => ({ role: item.role, content: item.content })),
      visible_messages: chatMessages,
      message_count: chatMessages.length,
      draft_input: String(els.chatInput?.value || ''),
    },
    thought_stream: {
      enabled: !!traceStreamEnabled,
      panel_visible: !els.traceStreamPanel.classList.contains('hidden'),
      meta: String(els.traceMeta?.textContent || '').trim(),
      window_meta: String(els.traceWindowMeta?.textContent || '').trim(),
      visible_text: String(els.traceLive?.value || ''),
      window_start: traceWindowStart,
      pinned_to_latest: !!traceWindowPinnedToLatest,
    },
    ontology: {
      path: els.ontologyPath.value.trim(),
      browser_current_directory: ontologyBrowserCurrentDirectory || '',
      browser_parent_directory: ontologyBrowserParentDirectory || '',
      browser_wildcard_path: ontologyBrowserWildcardPath || '',
    },
    graph_rag: {
      question: els.graphRagQuestion.value.trim(),
      rag_enabled: !!els.graphRagToggle.checked,
      rag_stream_enabled: !!els.graphRagStreamToggle.checked,
      embedding_enabled: !!els.graphRagEmbeddingEnabled.checked,
      embedding_provider: String(els.graphRagEmbeddingProvider?.value || '').trim(),
      embedding_model: String(els.graphRagEmbeddingModel?.value || '').trim(),
      embedding_dimensions: Number.parseInt(String(els.graphRagEmbeddingDimensions?.value || '0'), 10) || 0,
      embedding_index_name: String(els.graphRagEmbeddingIndex?.value || '').trim(),
      embedding_node_label: String(els.graphRagEmbeddingNodeLabel?.value || '').trim(),
      embedding_property_name: String(els.graphRagEmbeddingProperty?.value || '').trim(),
      answer: String(els.graphRagAnswer?.textContent || '').trim(),
      monitor: String(els.graphRagMonitor?.textContent || '').trim(),
      cycles: graphRagCycles.length,
    },
    ui: {
      active_tab: activeTab,
      metrics_panel_open: !!metricsPanelOpen,
    },
  };
}

async function persistCurrentCaseSnapshot() {
  /** Save the current working case snapshot without changing the visible status message. */
  const caseId = els.caseId.value.trim();
  const directory = els.directory.value.trim();
  if (!caseId || !directory) {
    return false;
  }

  const llm = getSelectedLlm();
  await api('/api/cases', {
    method: 'POST',
    body: JSON.stringify({
      case_id: caseId,
      directory,
      llm_provider: llm.provider,
      llm_model: llm.model,
      snapshot: buildCaseSaveSnapshot(),
    }),
  });
  loadedCaseId = caseId;
  writeLastUsedCaseId(caseId);
  syncCaseActionState();
  await loadCases({ silent: true });
  renderCaseIndex();
  return true;
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
    setStatus('Select a deposition folder or .txt file before saving.');
    return;
  }
  if (!selectedLlmIsOperational()) {
    setStatus('Selected model is unavailable. Choose an operational model before saving.');
    return;
  }

  startUiProcessing(`Saving case ${caseId}...`);
  try {
    await persistCurrentCaseSnapshot();
    setStatus(`Saved case '${caseId}'.`);
  } finally {
    endUiProcessing();
  }
}

async function duplicateCaseToNew() {
  /** Duplicate the active case into a new case id using case version clone flow. */
  const sourceCaseId = els.caseId.value.trim();
  if (!sourceCaseId) {
    setStatus('Enter a Case ID before duplicating.');
    return;
  }

  const suggested = `${sourceCaseId}-COPY`;
  const targetCaseId = String(window.prompt('Duplicate into new Case ID:', suggested) || '').trim();
  if (!targetCaseId) {
    setStatus('Case duplication canceled.');
    return;
  }
  if (targetCaseId === sourceCaseId) {
    setStatus('New Case ID must be different from the current Case ID.');
    return;
  }

  const directory = els.directory.value.trim();
  if (!directory) {
    setStatus('Select a deposition folder or .txt file before duplicating.');
    return;
  }

  const llm = getSelectedLlm();
  startUiProcessing(`Duplicating case ${sourceCaseId} to ${targetCaseId}...`);
  try {
    const payload = await api('/api/cases/version', {
      method: 'POST',
      body: JSON.stringify({
        case_id: targetCaseId,
        source_case_id: sourceCaseId,
        directory,
        llm_provider: llm.provider,
        llm_model: llm.model,
        snapshot: {
          duplicated_from: sourceCaseId,
          selected_deposition_id: selectedDepositionId || null,
          deposition_count: depositions.length,
        },
      }),
    });

    await loadCases({ silent: true });
    const targetSummary = cases.find((item) => item.case_id === targetCaseId) || null;
    applyCaseSelection(targetSummary);
    renderCaseIndex();
    await loadDepositions();
    setStatus(`Duplicated case '${sourceCaseId}' into '${targetCaseId}' (version ${payload.version}).`);
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

function applyGraphRagEmbeddingConfig(payload) {
  /** Apply one backend Graph RAG embedding configuration payload into the UI controls. */
  const config = payload && typeof payload === 'object' ? payload : {};
  els.graphRagEmbeddingEnabled.checked = !!config.enabled;
  els.graphRagEmbeddingProvider.value = String(config.provider || 'openai').trim() || 'openai';
  els.graphRagEmbeddingModel.value = String(config.model || 'text-embedding-3-small').trim()
    || 'text-embedding-3-small';
  els.graphRagEmbeddingDimensions.value = String(config.dimensions || 1536);
  els.graphRagEmbeddingIndex.value = String(config.index_name || 'resource_embeddings').trim()
    || 'resource_embeddings';
  els.graphRagEmbeddingNodeLabel.value = String(config.node_label || 'Resource').trim() || 'Resource';
  els.graphRagEmbeddingProperty.value = String(config.property_name || 'embedding').trim() || 'embedding';
}

async function loadGraphRagEmbeddingConfig({ silent = false } = {}) {
  /** Load the persisted Graph RAG embedding configuration from the backend. */
  const payload = await api('/api/graph-rag/embedding-config');
  applyGraphRagEmbeddingConfig(payload);
  if (!silent) {
    const mode = payload.enabled ? 'enabled' : 'disabled';
    setStatus(`Loaded Graph RAG embedding configuration (${mode}).`);
  }
  return payload;
}

async function saveGraphRagEmbeddingConfig() {
  /** Persist the current Graph RAG embedding configuration to the backend. */
  const model = String(els.graphRagEmbeddingModel.value || '').trim();
  const indexName = String(els.graphRagEmbeddingIndex.value || '').trim();
  const nodeLabel = String(els.graphRagEmbeddingNodeLabel.value || '').trim();
  const propertyName = String(els.graphRagEmbeddingProperty.value || '').trim();
  const dimensionsText = String(els.graphRagEmbeddingDimensions.value || '').trim();
  const dimensions = Number.parseInt(dimensionsText || '1536', 10);

  if (!model) {
    setStatus('Embedding model is required.');
    return;
  }
  if (!indexName) {
    setStatus('Vector index name is required.');
    return;
  }
  if (!nodeLabel) {
    setStatus('Embedding node label is required.');
    return;
  }
  if (!propertyName) {
    setStatus('Embedding property name is required.');
    return;
  }
  if (!Number.isFinite(dimensions) || dimensions <= 0) {
    setStatus('Embedding dimensions must be a positive number.');
    return;
  }

  startUiProcessing('Saving Graph RAG embedding configuration...');
  try {
    const payload = await api('/api/graph-rag/embedding-config', {
      method: 'POST',
      body: JSON.stringify({
        enabled: !!els.graphRagEmbeddingEnabled.checked,
        provider: String(els.graphRagEmbeddingProvider.value || 'openai').trim() || 'openai',
        model,
        dimensions,
        index_name: indexName,
        node_label: nodeLabel,
        property_name: propertyName,
      }),
    });
    applyGraphRagEmbeddingConfig(payload);
    const mode = payload.enabled ? 'enabled' : 'disabled';
    setStatus(`Saved Graph RAG embedding configuration (${mode}, ${payload.provider}/${payload.model}).`);
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
        embedding_config: {
          enabled: !!els.graphRagEmbeddingEnabled.checked,
          provider: String(els.graphRagEmbeddingProvider.value || 'openai').trim() || 'openai',
          model: String(els.graphRagEmbeddingModel.value || '').trim() || 'text-embedding-3-small',
          dimensions: Number.parseInt(String(els.graphRagEmbeddingDimensions.value || '1536'), 10) || 1536,
          index_name: String(els.graphRagEmbeddingIndex.value || '').trim() || 'resource_embeddings',
          node_label: String(els.graphRagEmbeddingNodeLabel.value || '').trim() || 'Resource',
          property_name: String(els.graphRagEmbeddingProperty.value || '').trim() || 'embedding',
        },
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
        ? `Graph RAG answer generated from ${payload.context_rows} context row(s) via ${payload.monitor?.retrieval_mode || 'keyword'} retrieval.`
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
  const retrievalMode = String(monitor?.retrieval_mode || 'keyword').trim() || 'keyword';
  const embeddingMode = monitor?.embedding_enabled
    ? `${String(monitor?.embedding_provider || '').trim() || 'embedding'}:${String(monitor?.embedding_model || '').trim() || 'default'}`
    : 'disabled';
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
    retrievalMode,
    embeddingMode,
    queryEmbeddingUsed: !!monitor?.query_embedding_used,
    embeddingError: String(monitor?.embedding_error || '').trim(),
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
        `Retrieval mode: ${item.retrievalMode}`,
        `Embeddings: ${item.embeddingMode} | Query vector used: ${item.queryEmbeddingUsed ? 'yes' : 'no'}`,
        item.embeddingError ? `Embedding fallback: ${item.embeddingError}` : null,
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
      ]
        .filter(Boolean)
        .join('\n')
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
  const schemaName = String(els.ingestSchemaSelect.value || '').trim() || 'deposition_schema';
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
        schema_name: schemaName,
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

els.tabLandingBtn.addEventListener('click', () => setActiveTab('landing'));
els.tabIntelligenceBtn.addEventListener('click', () => setActiveTab('intelligence'));
els.tabProvisioningBtn.addEventListener('click', () => setActiveTab('provisioning'));
els.tabObservablesBtn.addEventListener('click', () => setActiveTab('observables'));
els.tabAdminBtn.addEventListener('click', () => setActiveTab('admin'));
els.adminTabTestBtn.addEventListener('click', () => setActiveAdminSubtab('test'));
els.adminTabUsersBtn.addEventListener('click', () => setActiveAdminSubtab('users'));
if (els.adminTabPersonasBtn) {
  els.adminTabPersonasBtn.addEventListener('click', () => setActiveAdminSubtab('personas'));
}
if (els.adminTabMlopsBtn) {
  els.adminTabMlopsBtn.addEventListener('click', () => setActiveAdminSubtab('mlops'));
}
if (els.adminMlopsTabLlmoopsBtn) {
  els.adminMlopsTabLlmoopsBtn.addEventListener('click', () => setActiveMlopsSubtab('llmops'));
}
if (els.adminMlopsTabFineTuningBtn) {
  els.adminMlopsTabFineTuningBtn.addEventListener('click', () => setActiveMlopsSubtab('fine_tuning'));
}
if (els.adminMlopsTabDeploymentBtn) {
  els.adminMlopsTabDeploymentBtn.addEventListener('click', () => setActiveMlopsSubtab('deployment'));
}
if (els.adminMlopsTabCicdBtn) {
  els.adminMlopsTabCicdBtn.addEventListener('click', () => setActiveMlopsSubtab('cicd'));
}
els.adminAddUserBtn.addEventListener('click', () => {
  const nextOpen = !!els.adminUserCreatePanel?.classList.contains('hidden');
  if (nextOpen) {
    resetAdminUserForm();
  }
  setAdminUserCreateOpen(nextOpen);
  if (nextOpen) {
    setAdminUserFeedback('Enter a first name, last name, and authorization level, then save.', 'info');
    return;
  }
  setAdminUserFeedback('Click Add User to enter a first name, last name, and authorization level.', 'info');
});
els.adminSaveUserBtn.addEventListener('click', () => addAdminUser().catch((err) => setStatus(err.message)));
els.adminCancelUserBtn.addEventListener('click', () => {
  resetAdminUserForm();
  setAdminUserCreateOpen(false);
  setAdminUserFeedback('Click Add User to enter a first name, last name, and authorization level.', 'info');
});
els.adminUserSelect.addEventListener('change', () => {
  const selectedId = String(els.adminUserSelect?.value || '').trim();
  if (!selectedId) {
    return;
  }
  selectAdminUserById(selectedId, { openDetail: false });
});
els.adminGetUsersBtn.addEventListener('click', () => loadAdminUsers().catch((err) => setStatus(err.message)));
if (els.adminRemoveUserBtn) {
  els.adminRemoveUserBtn.addEventListener('click', () =>
    removeAdminUser().catch((err) => setStatus(err.message))
  );
}
els.adminRefreshUsersBtn.addEventListener('click', () => loadAdminUsers().catch((err) => setStatus(err.message)));
if (els.adminAddPersonaBtn) {
  els.adminAddPersonaBtn.addEventListener('click', () => {
    const nextOpen = !!els.adminPersonaCreatePanel?.classList.contains('hidden');
    if (nextOpen) {
      resetAdminPersonaForm();
    }
    setAdminPersonaCreateOpen(nextOpen);
    if (nextOpen) {
      setAdminPersonaFeedback('Enter a persona name, choose an LLM, and define prompts, then save.', 'info');
      return;
    }
    setAdminPersonaFeedback('Click Add Persona to enter a name, choose an LLM, and define prompts.', 'info');
  });
}
if (els.adminSavePersonaBtn) {
  els.adminSavePersonaBtn.addEventListener('click', () =>
    addAdminPersona({ closeOnSuccess: false }).catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaRagAddBtn) {
  els.adminPersonaRagAddBtn.addEventListener('click', () => addAdminPersonaRag());
}
if (els.adminPersonaLoadRagsBtn) {
  els.adminPersonaLoadRagsBtn.addEventListener('click', () => {
    loadAdminPersonaRagOptions()
      .then(() => setAdminPersonaFeedback('Loaded the currently available RAG steps.', 'info'))
      .catch((err) => setStatus(err.message));
  });
}
if (els.adminPersonaToolAddBtn) {
  els.adminPersonaToolAddBtn.addEventListener('click', () => addAdminPersonaTool());
}
if (els.adminPersonaLoadToolsBtn) {
  els.adminPersonaLoadToolsBtn.addEventListener('click', () => {
    loadAdminPersonaToolOptions()
      .then(() => setAdminPersonaFeedback('Loaded the currently available MCP tools.', 'info'))
      .catch((err) => setStatus(err.message));
  });
}
if (els.adminPersonaOpenPromptModalBtn) {
  els.adminPersonaOpenPromptModalBtn.addEventListener('click', () => toggleAdminPersonaPromptPanel());
}
if (els.adminPersonaToggleRagBtn) {
  els.adminPersonaToggleRagBtn.addEventListener('click', () => toggleAdminPersonaRagPanel());
}
if (els.adminPersonaTogglePromptObservablesBtn) {
  els.adminPersonaTogglePromptObservablesBtn.addEventListener('click', () => toggleAdminPersonaPromptObservablesPanel());
}
if (els.adminPersonaToggleToolsBtn) {
  els.adminPersonaToggleToolsBtn.addEventListener('click', () => toggleAdminPersonaToolsPanel());
}
if (els.adminPersonaRefreshPromptObservablesBtn) {
  els.adminPersonaRefreshPromptObservablesBtn.addEventListener('click', () => renderAdminPersonaPromptObservables());
}
if (els.adminPersonaSystemChoosePromptBtn) {
  els.adminPersonaSystemChoosePromptBtn.addEventListener('click', () =>
    toggleAdminPersonaPromptTemplateDropdown('system').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaSystemObservableBtn) {
  els.adminPersonaSystemObservableBtn.addEventListener('click', () => showAdminPersonaPromptObservableModal('system'));
}
if (els.adminPersonaSystemSavePromptBtn) {
  els.adminPersonaSystemSavePromptBtn.addEventListener('click', () =>
    saveAdminPersonaPromptSection('system').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaAssistantChoosePromptBtn) {
  els.adminPersonaAssistantChoosePromptBtn.addEventListener('click', () =>
    toggleAdminPersonaPromptTemplateDropdown('assistant').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaAssistantObservableBtn) {
  els.adminPersonaAssistantObservableBtn.addEventListener('click', () => showAdminPersonaPromptObservableModal('assistant'));
}
if (els.adminPersonaAssistantSavePromptBtn) {
  els.adminPersonaAssistantSavePromptBtn.addEventListener('click', () =>
    saveAdminPersonaPromptSection('assistant').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaContextChoosePromptBtn) {
  els.adminPersonaContextChoosePromptBtn.addEventListener('click', () =>
    toggleAdminPersonaPromptTemplateDropdown('context').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaContextObservableBtn) {
  els.adminPersonaContextObservableBtn.addEventListener('click', () => showAdminPersonaPromptObservableModal('context'));
}
if (els.adminPersonaContextSavePromptBtn) {
  els.adminPersonaContextSavePromptBtn.addEventListener('click', () =>
    saveAdminPersonaPromptSection('context').catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaSystemPromptTemplateSelect) {
  els.adminPersonaSystemPromptTemplateSelect.addEventListener('change', () => {
    syncAdminPersonaPromptTemplateSelectColor(els.adminPersonaSystemPromptTemplateSelect);
    syncAdminPersonaSelectedPromptTemplateKey();
    if (String(els.adminPersonaSystemPromptTemplateSelect.value || '').trim()) {
      loadSelectedAdminPersonaPromptTemplate('system');
      els.adminPersonaSystemPromptTemplateSelect.classList.add('hidden');
      if (adminPersonaPromptObservablesPanelActive) {
        renderAdminPersonaPromptObservables();
      }
    }
  });
}
if (els.adminPersonaAssistantPromptTemplateSelect) {
  els.adminPersonaAssistantPromptTemplateSelect.addEventListener('change', () => {
    syncAdminPersonaPromptTemplateSelectColor(els.adminPersonaAssistantPromptTemplateSelect);
    syncAdminPersonaSelectedPromptTemplateKey();
    if (String(els.adminPersonaAssistantPromptTemplateSelect.value || '').trim()) {
      loadSelectedAdminPersonaPromptTemplate('assistant');
      els.adminPersonaAssistantPromptTemplateSelect.classList.add('hidden');
      if (adminPersonaPromptObservablesPanelActive) {
        renderAdminPersonaPromptObservables();
      }
    }
  });
}
if (els.adminPersonaContextPromptTemplateSelect) {
  els.adminPersonaContextPromptTemplateSelect.addEventListener('change', () => {
    syncAdminPersonaPromptTemplateSelectColor(els.adminPersonaContextPromptTemplateSelect);
    syncAdminPersonaSelectedPromptTemplateKey();
    if (String(els.adminPersonaContextPromptTemplateSelect.value || '').trim()) {
      loadSelectedAdminPersonaPromptTemplate('context');
      els.adminPersonaContextPromptTemplateSelect.classList.add('hidden');
      if (adminPersonaPromptObservablesPanelActive) {
        renderAdminPersonaPromptObservables();
      }
    }
  });
}
if (els.adminPersonaSystemPrompt) {
  els.adminPersonaSystemPrompt.addEventListener('input', () => {
    if (adminPersonaPromptObservablesPanelActive) {
      renderAdminPersonaPromptObservables();
    }
  });
}
if (els.adminPersonaAssistantPrompt) {
  els.adminPersonaAssistantPrompt.addEventListener('input', () => {
    if (adminPersonaPromptObservablesPanelActive) {
      renderAdminPersonaPromptObservables();
    }
  });
}
if (els.adminPersonaContextPrompt) {
  els.adminPersonaContextPrompt.addEventListener('input', () => {
    if (adminPersonaPromptObservablesPanelActive) {
      renderAdminPersonaPromptObservables();
    }
  });
}
if (els.adminCancelPersonaBtn) {
  els.adminCancelPersonaBtn.addEventListener('click', () => {
    resetAdminPersonaForm();
    setAdminPersonaCreateOpen(false);
    setAdminPersonaFeedback('Click Add Persona to enter a name, choose an LLM, and define prompts.', 'info');
  });
}
if (els.adminPersonaSmokeTestBtn) {
  els.adminPersonaSmokeTestBtn.addEventListener('click', () => runAdminPersonaSmokeTestStub());
}
if (els.adminPersonaFormPromptSentimentBtn) {
  els.adminPersonaFormPromptSentimentBtn.addEventListener('click', () =>
    scoreAdminPersonaFormPromptSentiment().catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaFormPromptSentiment) {
  els.adminPersonaFormPromptSentiment.addEventListener('click', () => toggleAdminPersonaFormPromptSentimentDetail());
  els.adminPersonaFormPromptSentiment.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return;
    }
    event.preventDefault();
    toggleAdminPersonaFormPromptSentimentDetail();
  });
}
if (els.adminPersonaSelect) {
  els.adminPersonaSelect.addEventListener('change', () => {
    const selectedId = String(els.adminPersonaSelect?.value || '').trim();
    if (!selectedId) {
      return;
    }
    selectAdminPersonaById(selectedId);
  });
}
if (els.adminPersonaLlm) {
  els.adminPersonaLlm.addEventListener('change', () => renderAdminPersonaGraphMeta());
}
if (els.adminPersonaName) {
  els.adminPersonaName.addEventListener('input', () => renderAdminPersonaGraphMeta());
}
if (els.adminPersonaGraphAskBtn) {
  els.adminPersonaGraphAskBtn.addEventListener('click', () =>
    askAdminPersonaGraphQuestion().catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaGraphClearBtn) {
  els.adminPersonaGraphClearBtn.addEventListener('click', () => clearAdminPersonaGraphQuestion());
}
if (els.adminPersonaGraphQuestion) {
  els.adminPersonaGraphQuestion.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') {
      return;
    }
    event.preventDefault();
    askAdminPersonaGraphQuestion().catch((err) => setStatus(err.message));
  });
}
els.adminRunTestsBtn.addEventListener('click', () => runAdminTests().catch((err) => setStatus(err.message)));
els.adminRefreshTestLogBtn.addEventListener('click', () =>
  loadAdminTestLog().catch((err) => setStatus(err.message))
);
if (els.adminMlopsRefreshMetricsBtn) {
  els.adminMlopsRefreshMetricsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Refresh observables', async () => {
      await loadAgentMetrics({ silent: true });
      return 'Observables snapshot refreshed from the backend.';
    });
  });
}
if (els.adminMlopsRefreshModelsBtn) {
  els.adminMlopsRefreshModelsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Refresh models', async () => {
      await loadLlmOptions({ forceProbe: true });
      return 'Model availability was re-probed.';
    });
  });
}
if (els.adminMlopsOpenGrafanaBtn) {
  els.adminMlopsOpenGrafanaBtn.addEventListener('click', () => {
    runAdminMlopsAction('Open Grafana', async () => {
      const payload = await api('/api/observability/grafana');
      const grafanaUrl = String(
        payload?.dashboard_url || payload?.login_url || payload?.url || 'http://localhost:3000'
      ).trim();
      window.open(grafanaUrl, '_blank', 'noopener,noreferrer');
      return `Opened Grafana. Username: ${payload.username} Password: ${payload.password}`;
    });
  });
}
if (els.adminMlopsPromptVersionsBtn) {
  els.adminMlopsPromptVersionsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Prompt versions', async () =>
      'Stub only. Wire prompt registry versioning and prompt-diff history here.'
    );
  });
}
if (els.adminMlopsModelRoutingBtn) {
  els.adminMlopsModelRoutingBtn.addEventListener('click', () => {
    runAdminMlopsAction('Model selection / routing', async () =>
      'Stub only. Wire routing policy inspection and provider/model decision logs here.'
    );
  });
}
if (els.adminMlopsRagBehaviorBtn) {
  els.adminMlopsRagBehaviorBtn.addEventListener('click', () => {
    runAdminMlopsAction('RAGs', async () =>
      'Stub only. Wire retrieval coverage, toggle behavior, and RAG impact summaries here.'
    );
  });
}
if (els.adminMlopsTokenContextBtn) {
  els.adminMlopsTokenContextBtn.addEventListener('click', () => {
    runAdminMlopsAction('Token / context size', async () => {
      const payload = await api('/api/agent-metrics?lookback_hours=24');
      const byKey = Object.fromEntries((payload.metrics || []).map((item) => [item.key, item]));
      const callCount = String(byKey.llm_calls_sampled?.display || '0');
      const promptBytes = String(byKey.avg_prompt_context_bytes_per_llm_call?.display || '0 B');
      const promptTokens = String(byKey.avg_estimated_prompt_tokens_per_llm_call?.display || '0.0');
      const outputTokens = String(byKey.avg_estimated_output_tokens_per_llm_call?.display || '0.0');
      const ragContextBytes = String(byKey.rag_avg_context_bytes_on?.display || '0 B');
      return [
        'Live 24h LLMOps size telemetry:',
        `- Sampled LLM calls: ${callCount}`,
        `- Avg prompt context size / call: ${promptBytes}`,
        `- Avg estimated prompt tokens / call: ${promptTokens}`,
        `- Avg estimated output tokens / call: ${outputTokens}`,
        `- Avg RAG context size / ON call: ${ragContextBytes}`,
      ].join('\n');
    });
  });
}
if (els.adminMlopsCorrectnessBtn) {
  els.adminMlopsCorrectnessBtn.addEventListener('click', () => {
    runAdminMlopsAction(
      'Correctness / drift',
      async () => `LLM drift remediation checklist:
1. Confirm drift on the golden evaluation set before changing anything.
2. Isolate the changed layer: model version, prompt version, routing policy, or RAG index/settings.
3. Roll back the changed layer first if the regression started after a known release.
4. Re-run A/B checks with RAG on and off to separate retrieval drift from generation drift.
5. Tighten output constraints: schema validation, retries on invalid structure, and explicit grounding rules.
6. Check retrieval quality: expected source hit rate, noisy chunk inflation, and context size growth.
7. Pin model routing for critical flows so fallback models do not silently change behavior.
8. Expand the eval set with recent production failures and keep a stable core golden set.
9. Fine-tune only after prompt and retrieval fixes fail on a narrow, repeatable error pattern.
10. Monitor post-fix metrics: golden set accuracy, schema adherence, unsupported claims, repeat inconsistency, and RAG answer deltas.`
    );
  });
}
if (els.adminMlopsTraceQualityBtn) {
  els.adminMlopsTraceQualityBtn.addEventListener('click', () => {
    runAdminMlopsAction('Thought stream / trace quality', async () =>
      'Stub only. Wire trace completeness, prompt capture coverage, and event-sequence integrity checks here.'
    );
  });
}
if (els.adminMlopsThoughtHealthBtn) {
  els.adminMlopsThoughtHealthBtn.addEventListener('click', () => {
    runAdminMlopsAction('Check Thought Stream DB', async () => {
      const payload = await api('/api/thought-streams/health');
      return `Database ${payload.database} is connected.`;
    });
  });
}
if (els.adminMlopsRagHealthBtn) {
  els.adminMlopsRagHealthBtn.addEventListener('click', () => {
    runAdminMlopsAction('Check RAG Stream DB', async () => {
      const payload = await api('/api/rag-streams/health');
      return `Database ${payload.database} is connected.`;
    });
  });
}
if (els.adminMlopsOpenGithubActionsBtn) {
  els.adminMlopsOpenGithubActionsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Open GitHub Actions', async () => {
      openGithubActionsUrl(GITHUB_ACTIONS_URLS.actions);
      return 'Opened the GitHub Actions overview.';
    });
  });
}
if (els.adminMlopsOpenCiWorkflowBtn) {
  els.adminMlopsOpenCiWorkflowBtn.addEventListener('click', () => {
    runAdminMlopsAction('Open CI/CD workflow', async () => {
      openGithubActionsUrl(GITHUB_ACTIONS_URLS.ciWorkflow);
      return 'Opened the CI/CD workflow definition.';
    });
  });
}
if (els.adminMlopsOpenDeployWorkflowBtn) {
  els.adminMlopsOpenDeployWorkflowBtn.addEventListener('click', () => {
    runAdminMlopsAction('Open deploy workflow', async () => {
      openGithubActionsUrl(GITHUB_ACTIONS_URLS.deployWorkflow);
      return 'Opened the deploy workflow definition.';
    });
  });
}
if (els.adminMlopsRunTestsBtn) {
  els.adminMlopsRunTestsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Run all tests', async () => {
      await runAdminTests();
      return 'Pytest completed and the Admin/Test view was refreshed.';
    });
  });
}
if (els.adminMlopsRefreshReportBtn) {
  els.adminMlopsRefreshReportBtn.addEventListener('click', () => {
    runAdminMlopsAction('Refresh test view', async () => {
      await refreshAdminTestView();
      return 'Test log and pytest HTML report were refreshed.';
    });
  });
}
if (els.adminMlopsOpenFineTuningBtn) {
  els.adminMlopsOpenFineTuningBtn.addEventListener('click', () => {
    runAdminMlopsAction('Focus Fine Tuning', async () => {
      setActiveTab('admin');
      setActiveAdminSubtab('mlops');
      setActiveMlopsSubtab('fine_tuning');
      return 'Focused the Fine Tuning tools inside Admin -> MLOps.';
    });
  });
}
if (els.adminMlopsFineTuningRefreshModelsBtn) {
  els.adminMlopsFineTuningRefreshModelsBtn.addEventListener('click', () => {
    runAdminMlopsAction('Refresh fine tuning models', async () => {
      await loadLlmOptions({ forceProbe: true });
      return 'Model availability was refreshed for fine tuning workflows.';
    });
  });
}
if (els.adminMlopsDeploymentThoughtBtn) {
  els.adminMlopsDeploymentThoughtBtn.addEventListener('click', () => {
    runAdminMlopsAction('Verify deployment thought stream', async () => {
      const payload = await api('/api/thought-streams/health');
      return `Thought Stream database ${payload.database} is connected.`;
    });
  });
}
if (els.adminMlopsDeploymentRagBtn) {
  els.adminMlopsDeploymentRagBtn.addEventListener('click', () => {
    runAdminMlopsAction('Verify deployment RAG stream', async () => {
      const payload = await api('/api/rag-streams/health');
      return `RAG Stream database ${payload.database} is connected.`;
    });
  });
}
if (els.adminMlopsDeploymentObservablesBtn) {
  els.adminMlopsDeploymentObservablesBtn.addEventListener('click', () => {
    runAdminMlopsAction('Pull deployment observables snapshot', async () => {
      await loadAgentMetrics({ silent: true });
      return 'Deployment observables snapshot refreshed from the backend.';
    });
  });
}
if (els.adminMlopsOpenAdminTestBtn) {
  els.adminMlopsOpenAdminTestBtn.addEventListener('click', () => {
    runAdminMlopsAction('Open Test subtab', async () => {
      setActiveAdminSubtab('test');
      return 'Switched to the Admin Test subtab.';
    });
  });
}
for (const field of [els.adminUserFirstName, els.adminUserLastName].filter(Boolean)) {
  field.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') {
      return;
    }
    event.preventDefault();
    addAdminUser().catch((err) => setStatus(err.message));
  });
}
els.saveIntelligenceBtn.addEventListener('click', () => saveCase().catch((err) => setStatus(err.message)));
els.duplicateCaseBtn.addEventListener('click', () => duplicateCaseToNew().catch((err) => setStatus(err.message)));
els.newCaseBtn.addEventListener('click', () => createBlankCase());
if (els.ingestSchemaSelect) {
  els.ingestSchemaSelect.addEventListener('change', () => {
    editingNewIngestSchema = false;
    syncSelectedIngestSchemaEditor();
  });
}
if (els.newIngestSchemaBtn) {
  els.newIngestSchemaBtn.addEventListener('click', () => startNewIngestSchemaDraft());
}
if (els.saveIngestSchemaBtn) {
  els.saveIngestSchemaBtn.addEventListener('click', () =>
    saveIngestSchema().catch((err) => setStatus(err.message))
  );
}
if (els.removeIngestSchemaBtn) {
  els.removeIngestSchemaBtn.addEventListener('click', () =>
    removeIngestSchema().catch((err) => setStatus(err.message))
  );
}
els.browseDepositionBtn.addEventListener('click', () =>
  openDepositionBrowser().catch((err) => setStatus(err.message))
);
els.importDepositionBtn.addEventListener('click', () => promptImportDeposition());
els.importDepositionFolderBtn.addEventListener('click', () => promptImportDepositionFolder());
els.importDepositionInput.addEventListener('change', () =>
  importSelectedDepositionFiles().catch((err) => setStatus(err.message))
);
els.importDepositionFolderInput.addEventListener('change', () =>
  importSelectedDepositionFolder().catch((err) => setStatus(err.message))
);
els.ingestBtn.addEventListener('click', () => ingestCase().catch((err) => setStatus(err.message)));
els.refreshBtn.addEventListener('click', () => refreshCase().catch((err) => setStatus(err.message)));
els.saveCaseBtn.addEventListener('click', () => saveCase().catch((err) => setStatus(err.message)));
els.loadOntologyBtn.addEventListener('click', () => loadOntologyGraph().catch((err) => setStatus(err.message)));
els.saveGraphRagEmbeddingBtn.addEventListener(
  'click',
  () => saveGraphRagEmbeddingConfig().catch((err) => setStatus(err.message))
);
els.reloadGraphRagEmbeddingBtn.addEventListener(
  'click',
  () => loadGraphRagEmbeddingConfig().catch((err) => setStatus(err.message))
);
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
els.openGrafanaObservablesBtn.addEventListener('click', () =>
  openGrafanaWithCredentials().catch((err) => setStatus(err.message))
);
els.computeSentimentBtn.addEventListener('click', () =>
  computeDepositionSentiment().catch((err) => setStatus(err.message))
);
els.detailSentiment.addEventListener('click', () =>
  toggleDepositionSentimentDetail().catch((err) => setStatus(err.message))
);
els.detailSentiment.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') {
    return;
  }
  event.preventDefault();
  toggleDepositionSentimentDetail().catch((err) => setStatus(err.message));
});
els.metricDetailCloseBtn.addEventListener('click', () => resetMetricDetail());
els.metricTrendCloseBtn.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  event.stopPropagation();
  lockMetricInteractions();
});
els.metricTrendCloseBtn.addEventListener('click', (event) => {
  event.preventDefault();
  event.stopPropagation();
  lockMetricInteractions();
  resetMetricTrend();
});
els.summarizeFocusedReasoningBtn.addEventListener('click', () =>
  summarizeFocusedReasoning().catch((err) => setStatus(err.message))
);
els.stopInferenceBtn.addEventListener('click', () => stopInferencing());
els.caseId.addEventListener('change', () => {
  syncCaseActionState();
  renderCaseIndex();
});
els.caseId.addEventListener('input', () => {
  syncCaseActionState();
  renderCaseIndex();
});
els.directory.addEventListener('input', () => {
  syncCaseActionState();
});
els.directory.addEventListener('change', () => {
  const selected = els.directory.value.trim();
  if (!selected) {
    setStatus('Select a deposition folder or .txt file.');
    return;
  }
  setStatus(
    looksLikeTxtPath(selected)
      ? `Deposition file selected: ${selected}`
      : `Deposition folder selected: ${selected}`
  );
});
els.depositionBrowserCloseBtn.addEventListener('click', () => setDepositionBrowserOpen(false));
els.depositionBrowserUpBtn.addEventListener('click', () =>
  browseDepositionDirectory(depositionBrowserParentDirectory).catch((err) => setStatus(err.message))
);
els.depositionBrowserRefreshBtn.addEventListener('click', () =>
  browseDepositionDirectory(depositionBrowserCurrentDirectory).catch((err) => setStatus(err.message))
);
els.depositionBrowserUseFolderBtn.addEventListener('click', () => {
  if (!depositionBrowserCurrentDirectory) {
    return;
  }
  els.directory.value = depositionBrowserCurrentDirectory;
  setDepositionBrowserOpen(false);
  syncCaseActionState();
  setStatus(`Deposition folder selected: ${depositionBrowserCurrentDirectory}`);
});
els.depositionBrowserList.addEventListener('click', (event) =>
  handleDepositionBrowserListClick(event).catch((err) => setStatus(err.message))
);
els.depositionBrowserModal.addEventListener('click', (event) => {
  if (event.target === els.depositionBrowserModal) {
    setDepositionBrowserOpen(false);
  }
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
els.adminUserDetailCloseBtn.addEventListener('click', () => hideAdminUserDetail());
els.adminUserDetailPanel.addEventListener('click', (event) => {
  if (event.target === els.adminUserDetailPanel) {
    hideAdminUserDetail();
  }
});
if (els.adminPersonaPromptModalCloseBtn) {
  els.adminPersonaPromptModalCloseBtn.addEventListener('click', () => hideAdminPersonaPromptModal());
}
if (els.adminPersonaPromptModal) {
  els.adminPersonaPromptModal.addEventListener('click', (event) => {
    if (event.target === els.adminPersonaPromptModal) {
      hideAdminPersonaPromptModal();
    }
  });
}
if (els.adminPersonaPromptObservableModalCloseBtn) {
  els.adminPersonaPromptObservableModalCloseBtn.addEventListener('click', () => hideAdminPersonaPromptObservableModal());
}
if (els.adminPersonaPromptObservableModal) {
  els.adminPersonaPromptObservableModal.addEventListener('click', (event) => {
    if (event.target === els.adminPersonaPromptObservableModal) {
      hideAdminPersonaPromptObservableModal();
    }
  });
}
if (els.adminPersonaPromptApplyBtn) {
  els.adminPersonaPromptApplyBtn.addEventListener('click', () => applyAdminPersonaPromptFromModal());
}
if (els.adminPersonaPromptResaveBtn) {
  els.adminPersonaPromptResaveBtn.addEventListener('click', () =>
    resaveAdminPersonaPromptFromModal().catch((err) => setStatus(err.message))
  );
}
if (els.adminPersonaPromptSentimentBtn) {
  els.adminPersonaPromptSentimentBtn.addEventListener('click', () =>
    scoreAdminPersonaPromptSentiment().catch((err) => setStatus(err.message))
  );
}
els.metricDetailPanel.addEventListener('click', (event) => {
  if (event.target === els.metricDetailPanel) {
    resetMetricDetail();
  }
});
els.metricTrendPanel.addEventListener('click', (event) => {
  if (els.metricTrendPanel.classList.contains('overlay-modal') && event.target === els.metricTrendPanel) {
    resetMetricTrend();
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
  pendingSavedLlmValue = '';
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
els.timeline.addEventListener('scroll', () => {
  if (els.timelineScale) {
    els.timelineScale.scrollLeft = els.timeline.scrollLeft;
  }
  syncTimelineNavButtons();
});
els.chatForm.addEventListener('submit', (event) => sendChat(event).catch((err) => setStatus(err.message)));
window.addEventListener('resize', () => {
  updateTimelineSlots();
  syncTimelineNavButtons();
});
window.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') {
    return;
  }
  if (els.adminPersonaPromptObservableModal && !els.adminPersonaPromptObservableModal.classList.contains('hidden')) {
    hideAdminPersonaPromptObservableModal();
    return;
  }
  if (els.adminPersonaPromptModal && !els.adminPersonaPromptModal.classList.contains('hidden')) {
    hideAdminPersonaPromptModal();
    return;
  }
  if (!els.metricDetailPanel.classList.contains('hidden')) {
    resetMetricDetail();
    return;
  }
  if (!els.adminUserDetailPanel.classList.contains('hidden')) {
    hideAdminUserDetail();
    return;
  }
  if (!els.metricTrendPanel.classList.contains('hidden')) {
    resetMetricTrend();
    return;
  }
  if (!els.ontologyBrowserModal.classList.contains('hidden')) {
    setOntologyBrowserOpen(false);
  }
});

renderDetail(null);
renderTimeline();
renderDepositions();
hydrateCachedMetrics();
syncFocusedReasoningActionState();
loadedCaseId = els.caseId.value.trim();
syncCaseActionState();
setTraceStreamEnabled(false);
loadLlmOptions({ silent: true }).catch((err) => {
  ensureLlmFallbackOption();
  setStatus(`Failed to load LLM options: ${err.message}`);
});
Promise.all([
  loadIngestSchemaOptions({ silent: true }).catch((err) => {
    ensureIngestSchemaFallbackOption();
    syncSelectedIngestSchemaEditor();
    throw new Error(`Failed to load ingest schemas: ${err.message}`);
  }),
  loadDirectoryOptions({ silent: true }),
])
  .then(async () => {
    await loadCases({ silent: true });
    if (els.caseId.value.trim()) {
      await loadDepositions();
    }
  })
  .catch((err) => {
    const message = err instanceof Error ? err.message : String(err || 'Unknown error');
    if (message.startsWith('Failed to load ingest schemas:')) {
      setStatus(message);
      return;
    }
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
loadGraphRagEmbeddingConfig({ silent: true }).catch((err) => {
  const message = err instanceof Error ? err.message : String(err || 'Unknown error');
  setStatus(`Failed to load Graph RAG embedding config: ${message}`);
});
setActiveTab('landing');
setMetricsPanelOpen(false);
