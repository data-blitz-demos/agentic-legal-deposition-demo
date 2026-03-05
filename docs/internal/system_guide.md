# Internal System Guide

This document is the human-written internal companion to the generated function reference in `docs/internal/code_function_reference.md`.

Use this guide for:

- high-level architecture
- major file ownership
- UI tab and overlay behavior
- persistence layout
- runtime flows
- observability and deployment surfaces

Use the generated function reference for:

- per-function signatures
- per-module function summaries
- source line anchors

## 1. Application Shape

The application is a single FastAPI service that:

- serves the browser UI
- exposes JSON APIs used by the browser
- orchestrates LangGraph ingestion workflows
- reads and writes CouchDB, Neo4j, Prometheus metrics, Loki/Grafana support files, and the local filesystem

Core runtime path:

1. Browser UI loads `/`
2. Browser JavaScript from `/static/app.js` calls `/api/...`
3. FastAPI routes in `backend/app/main.py` perform validation and orchestration
4. Route handlers call service modules:
   - `graph.py`
   - `chat.py`
   - `neo4j_graph.py`
   - `couchdb.py`
   - `llm.py`
5. Results are persisted and returned to the UI

## 2. Maintained File Map

This section documents the major maintained files that define the running system. Binary assets, generated screenshots, and rendered PDFs are intentionally excluded unless they are operationally important.

### 2.1 Top-Level Product and Ops Files

| File | Purpose |
| --- | --- |
| `README.md` | Operator-facing project guide, setup, runtime, observability, AWS/Terraform notes. |
| `AGENT.md` | Chronological implementation work log for the workspace. |
| `NOTICE.md` | Ownership and notice text. |
| `docker-compose.yml` | Local stack orchestration for API, CouchDB, Neo4j, Prometheus, Grafana, Loki, and Promtail. |
| `requirements.txt` | Python runtime and development dependencies used in the container image and local runs. |

### 2.2 Backend Service Layer

| File | Purpose |
| --- | --- |
| `backend/app/main.py` | FastAPI app, route layer, orchestration, persistence glue, trace/session handling, observability endpoints. |
| `backend/app/models.py` | Pydantic request/response contracts and domain models. |
| `backend/app/config.py` | Cached environment-driven settings. |
| `backend/app/couchdb.py` | CouchDB client for CRUD, Mango queries, and view access. |
| `backend/app/graph.py` | LangGraph ingestion workflow for deposition mapping and contradiction assessment. |
| `backend/app/chat.py` | Persona:Attorney chat, contradiction reasoning, focused summary logic. |
| `backend/app/llm.py` | LLM provider selection, readiness checks, model inventory, failure wrapping. |
| `backend/app/neo4j_graph.py` | Ontology load, graph retrieval, keyword RAG, vector-enabled Graph RAG retrieval. |
| `backend/app/prompts.py` | Prompt file loading, rendering, and prompt-template inventory. |
| `backend/app/schemas.py` | Built-in ingest schema loading and dynamic schema discovery. |
| `backend/app/logging_config.py` | Python logging setup for stdout and file-backed logs. |
| `backend/app/metrics.py` | Prometheus counters, histograms, gauges, and instrumentation helpers. |

### 2.3 Frontend UI Layer

| File | Purpose |
| --- | --- |
| `frontend/index.html` | All application markup, top tabs, admin subtabs, overlays, panel structure, and wired element IDs. |
| `frontend/styles.css` | All layout, branding, animation, responsive behavior, overlay styling, and control styling. |
| `frontend/app.js` | Entire browser controller: state, rendering, fetch calls, timers, modals, event binding, snapshot persistence. |

### 2.4 Infrastructure and Deployment

| File or Folder | Purpose |
| --- | --- |
| `.github/workflows/ci-cd.yml` | CI validation and image publication workflow. |
| `.github/workflows/deploy.yml` | Manual deploy workflow targeting Kubernetes. |
| `terraform/` | Terraform scaffold for AWS VPC, ECR, EKS, CPU/GPU node groups, and EFS. |
| `deploy/observability/` | Prometheus, Loki, Promtail, and Grafana provisioning files. |
| `deploy/k8s/aws/` | Kubernetes manifests and AWS deployment runbook. |
| `scripts/bounce` | Local restart helper for Docker Compose + Chrome with cache-busting behavior. |

### 2.5 Test Suite

The `tests/` folder is the executable specification for the system. The key groups are:

| File Group | Purpose |
| --- | --- |
| `tests/test_api_integration.py` | FastAPI integration coverage across routes and cross-component flows. |
| `tests/test_main.py` | Direct route/helper coverage for the orchestration layer. |
| `tests/test_graph.py` | LangGraph workflow behavior and fallback logic. |
| `tests/test_chat.py` | Persona:Attorney formatting and fallback behavior. |
| `tests/test_neo4j_graph.py` | Graph retrieval, vector fallback, and ontology graph helpers. |
| `tests/test_internal_docs.py` | Generated documentation coverage. |

## 3. UI Tab Reference

### 3.1 Home

Purpose:

- landing screen
- branding / animation
- zero-data first view

Key elements:

- `#tabPageLanding`
- `.landing-logo-burst`

Notes:

- The landing page is visual only. It does not perform ingestion or API actions itself.

### 3.2 Deposition

Purpose:

- primary case review workspace

Major panels:

- `Deposition Timeline`
- `Risk Scores`
- `Conflict Detail`
- `Persona:Attorney Chat`

Key behaviors:

- timeline selection drives the active deposition
- contradiction clicks trigger focused re-analysis
- sentiment can be computed, shown/hidden, and expanded
- chat is case/deposition scoped

### 3.3 Case

Purpose:

- case lifecycle management
- ingestion source selection
- schema management
- ontology and Graph RAG controls

Major controls:

- `Case ID`
- `Deposition Source`
- `LLM`
- `Ingest Schema`
- `Load.Depositions`
- import deposition / import folder
- graph ontology load / graph browser / embedding config

Important behavior:

- `Deposition Source` accepts either:
  - a directory
  - a single `.txt`
- recursive ingestion is supported for nested deposition trees
- custom ingest schemas are created, updated, and removed from the UI

### 3.4 Observables

Purpose:

- operator-facing runtime telemetry
- thought stream inspection
- Grafana handoff

Major panels:

- `Thought Stream`
- `Observables`

Key behaviors:

- current metrics are polled from `/api/agent-metrics`
- metric cards support single-click detail and trend rendering
- Grafana opens from the tab header

### 3.5 Admin

Purpose:

- operational administration
- user/persona management
- test controls
- MLOps/LLMOps access

Admin subtabs:

- `Users`
- `Personas`
- `Test`
- `MLOps`

## 4. Overlay and Modal Reference

The UI uses explicit overlay sections in `frontend/index.html`. These are the true modal surfaces that need to be documented and kept stable.

### 4.1 Observable Detail Overlay

- Root: `#metricDetailPanel`
- Purpose: display the descriptive text for one metric/observable
- Opened by:
  - single-clicking a metric card
- Closed by:
  - `#metricDetailCloseBtn`
  - backdrop click
  - `Esc`

### 4.2 Deposition Browser Overlay

- Root: `#depositionBrowserModal`
- Purpose: browse directories and choose either a folder or a single deposition file source for the Case tab
- Opened by:
  - `#browseDepositionBtn`
- Key controls:
  - `#depositionBrowserUpBtn`
  - `#depositionBrowserUseFolderBtn`
  - `#depositionBrowserRefreshBtn`

### 4.3 Ontology Browser Overlay

- Root: `#ontologyBrowserModal`
- Purpose: browse ontology folders and `.owl` files for Neo4j load operations
- Opened by:
  - `#ontologyBrowseBtn`
- Key controls:
  - `#ontologyBrowserUpBtn`
  - `#ontologyBrowserUseFolderBtn`
  - `#ontologyBrowserRefreshBtn`

### 4.4 Admin User Detail Overlay

- Root: `#adminUserDetailPanel`
- Purpose: show a text-box summary of current users with the selected user highlighted
- Opened by:
  - clicking a user row in `Admin -> Users`
- Closed by:
  - `#adminUserDetailCloseBtn`
  - backdrop click
  - `Esc`

## 5. Non-Modal Expand/Collapse Surfaces

These are not overlays, but they behave like expandable control surfaces and are operationally important.

### 5.1 Ingest Schema Manager

- Root element: `.ingest-schema-manager`
- Location: `Case`
- Purpose: create, edit, persist, and delete custom ingest schemas

### 5.2 Ontology and Graph RAG Controls

- Root element: `.graph-rag-dropdown`
- Location: `Case`
- Purpose:
  - ontology load
  - graph browser
  - graph querying
  - embedding configuration

### 5.3 Focused Re-Analysis

- Root element: `#focusedReasoning`
- Location: `Conflict Detail`
- Purpose:
  - show item-specific re-analysis
  - summarize or return to full re-analysis text

This is a panel, not an overlay.

## 6. Persistence Model

### 6.1 Primary Databases

| Store | Role |
| --- | --- |
| `depositions` (CouchDB) | Main operational data: cases, depositions, users, personas, schema docs, snapshots, graph config, admin metadata. |
| `memory` (CouchDB) | Case memory records and event-like persisted memory artifacts. |
| `thought_stream` (CouchDB) | Trace sessions and thought-stream persistence. |
| `rag-stream` (CouchDB) | RAG trace payloads and query monitoring records. |
| `Neo4j` | Ontology graph and Graph RAG retrieval surface. |

### 6.2 Key Document Families in CouchDB

| `type` | Purpose |
| --- | --- |
| `case` | Case metadata and UI snapshot state. |
| `deposition` | One deposition record per ingested file. |
| `case_memory` | Saved memory/event entries tied to a case. |
| `user` | Admin-managed user record with authorization level. |
| `persona` | Saved persona configuration, including prompts and ordered RAG bindings. |
| `ingest_schema` | Custom user-managed ingest schema payload. |
| `observable_snapshot` | Persisted metrics snapshots for trend history. |

### 6.3 Persona Persistence

A persona currently stores:

- identity
- model choice
- prompt text
- prompt template seed key
- ordered RAG sequence with enabled state
- last graph-only question
- last graph-only answer
- last graph query timestamp

### 6.4 Case Snapshot Persistence

Case save captures UI/runtime state including:

- selected case
- selected deposition
- chat history and visible transcript
- conflict detail content
- thought stream visible state
- graph/rag controls
- ingest schema selection
- dropdown options used by the UI

## 7. Runtime Flow Reference

### 7.1 Ingest Case

Primary files:

- `backend/app/main.py`
- `backend/app/graph.py`

Flow:

1. Case tab submits `POST /api/ingest-case`
2. `main.py` resolves file(s), schema, and LLM
3. `DepositionWorkflow.run(...)` invokes LangGraph
4. Workflow state moves through:
   - `read_file`
   - `map_deposition`
   - `save_deposition`
   - `load_other_depositions`
   - `evaluate_contradictions`
   - `persist_assessment`
5. UI refreshes case list and deposition list

### 7.2 Persona:Attorney Chat

Primary files:

- `backend/app/main.py`
- `backend/app/chat.py`

Flow:

1. Deposition tab submits `POST /api/chat`
2. Route validates case/deposition consistency
3. `AttorneyChatService.respond_with_trace(...)` builds prompt context
4. Response is normalized into stable descriptive text
5. Trace and memory are persisted
6. UI appends transcript and updates status/timers

### 7.3 Graph RAG

Primary files:

- `backend/app/main.py`
- `backend/app/neo4j_graph.py`

Flow:

1. UI submits `POST /api/graph-rag/query`
2. Backend loads active embedding config
3. If vector retrieval is enabled, the API attempts query embedding generation
4. Neo4j retrieval runs:
   - vector first when configured
   - keyword fallback when vector is unavailable or empty
5. Retrieved rows are passed into the Graph RAG prompt
6. Output, trace, and RAG monitoring record are persisted

### 7.4 Observability

Primary files:

- `backend/app/metrics.py`
- `backend/app/logging_config.py`
- `backend/app/main.py`

Outputs:

- `/metrics` for Prometheus
- log file for Promtail/Loki
- Grafana dashboards for metrics and log stream
- in-app `Observables` panel for operator-facing KPIs

## 8. Prompt and Schema Source of Truth

### 8.1 Prompt Source of Truth

Built-in runtime prompt files live in:

- `backend/prompts`

Loaded by:

- `backend/app/prompts.py`

Important distinction:

- Built-in prompt files are still the runtime source of truth for most flows.
- Persona prompts are persisted configuration and editor content, but they are not yet the universal runtime prompt source for every route.

### 8.2 Ingest Schema Source of Truth

Built-in schema files live in:

- `backend/schemas`

Custom schemas are persisted in CouchDB as `type: "ingest_schema"`.

The UI now manages schemas by path/key only, not a separate display label.

## 9. LangGraph State Model

LangGraph state is defined in:

- `backend/app/graph.py`

State model:

- `GraphState(TypedDict, total=False)`

This is the shared mutable workflow state used by `StateGraph(GraphState)`.

Current fields:

- `case_id`
- `file_path`
- `llm_provider`
- `llm_model`
- `schema_name`
- `selected_schema`
- `selected_schema_mode`
- `raw_text`
- `deposition`
- `deposition_doc`
- `ingest_schema_mode`
- `ingest_schema_payload`
- `other_depositions`
- `assessment`
- `legal_clerk_trace`
- `attorney_trace`

## 10. How to Refresh Internal Documentation

### 10.1 Generated Function Reference

Regenerate with:

```bash
cd .
python scripts/generate_internal_docs.py
```

Output:

- `docs/internal/code_function_reference.md`

### 10.2 This System Guide

This file is hand-maintained and should be updated whenever any of these change:

- top-level tabs or admin subtabs
- overlay/modal IDs
- persistence document shapes
- core runtime flow boundaries
- observability stack
- deployment topology

### 10.3 Recommended Documentation Discipline

When changing code:

1. update function docstrings in the source
2. regenerate `code_function_reference.md`
3. update this system guide if the change affects:
   - UI structure
   - persistence
   - routing
   - infrastructure
4. update `README.md` when the operator-facing workflow changes
5. append `AGENT.md` with the milestone
