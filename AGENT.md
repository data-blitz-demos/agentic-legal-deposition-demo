# AGENT Work Log

This file records the implementation steps performed in this workspace, in chronological order.

1. Initialized project structure (`backend/app`, `frontend`).
2. Added backend configuration (`backend/app/config.py`) with environment-driven settings for OpenAI model, CouchDB, and context limits.
3. Added domain/data models (`backend/app/models.py`) for claims, deposition schema, contradiction findings/assessment, ingest API contracts, and chat contracts.
4. Implemented CouchDB client (`backend/app/couchdb.py`) for database creation, insert/update/get/find/list operations.
5. Implemented LangGraph workflow (`backend/app/graph.py`) with nodes:
   - Read deposition text file.
   - Map unstructured deposition into schema with persona prompt: seasoned law clerk.
   - Persist deposition document.
   - Load peer depositions in same case.
   - Evaluate contradictions with persona prompt: seasoned attorney.
   - Persist contradiction score/findings.
6. Added attorney chat service (`backend/app/chat.py`) backed by ChatGPT model calls.
7. Added FastAPI app and routes (`backend/app/main.py`):
   - `POST /api/ingest-case`
   - `GET /api/depositions/{case_id}`
   - `GET /api/deposition/{deposition_id}`
   - `POST /api/chat`
8. Added initial frontend UI (`frontend/index.html`, `frontend/styles.css`, `frontend/app.js`) with ingest controls, contradiction score list, detail panel, and chat panel.
9. Added project dependencies (`requirements.txt`), sample deposition files (`sample_depositions/*.txt`), compose file for CouchDB, `.env.example`, `.gitignore`, and README.
10. Fixed ingestion/reassessment behavior:
    - Reassessed all depositions in a case after ingest so earlier records are re-evaluated against newer records.
    - Made deposition document IDs deterministic by case/file stem to support idempotent updates.
    - Improved `update_doc` to handle create-or-update flows.
11. Improved API correctness:
    - Added case/deposition consistency check in chat endpoint.
    - Adjusted CORS credentials policy for wildcard origin compatibility.
12. Added full Docker support:
    - Created `Dockerfile` for API service.
    - Created `.dockerignore`.
    - Expanded `docker-compose.yml` to include `api` + `couchdb` services.
    - Mounted host deposition directory into container at `/data/depositions`.
13. Added startup resilience by implementing CouchDB connection retry loop in `ensure_db`.
14. Added configurable host ports (`API_PORT`, `COUCHDB_PORT`) and mapped them in compose.
15. Updated docs for docker-only run flow and in-container ingestion path usage.
16. Diagnosed/verified runtime with compose and curl checks:
    - Confirmed API root and static assets served.
    - Confirmed openapi exposure and endpoint availability.
17. Added timeline and contradiction UX enhancements:
    - Added horizontal deposition timeline with back/forward controls.
    - Switched score presentation to numeric score display on cards.
    - Added "Overall Short Answer" section.
    - Added clickable contradiction bullet items.
18. Added focused contradiction re-analysis capability:
    - Added models for focused reasoning request/response.
    - Added `POST /api/reason-contradiction` endpoint.
    - Added chat service method to reason about a single contradiction item.
19. Updated attorney response formatting requirements to always return:
    - `Short answer:`
    - `Details:` with bullet points.
20. Refined contradiction assessor prompt to keep explanation short.
21. Updated frontend to call focused-reasoning endpoint when bullet items are clicked.
22. Removed minimum risk score filter per request:
    - Removed slider/filter controls from UI markup.
    - Removed filtering logic and event handlers from frontend JS.
    - Simplified styles to remove filter-specific CSS.
23. Updated README UI documentation to reflect current UI layout and behavior.
24. Added this `AGENT.md` file documenting all performed steps.
25. Added CouchDB MCP server implementation at `mcp_servers/couchdb_server.py` with tools:
    - `list_case_depositions`
    - `get_deposition`
    - `list_flagged_depositions`
    - `search_claims`
26. Added MCP client demo at `scripts/use_couchdb_mcp.py` to start a stdio MCP session and call server tools.
27. Added `mcp` dependency to `requirements.txt` and documented MCP usage in `README.md`.
28. Updated `Dockerfile` to copy `mcp_servers/` and `scripts/` into the API container.
29. Added robust path handling so MCP server can import project modules when executed as a script.
30. Validated MCP usage with `docker compose exec api python scripts/use_couchdb_mcp.py` and confirmed successful CouchDB tool calls.
31. Added live LLM processing timers with animated icons in the UI for:
    - Global LLM operations (ingest/chat/reasoning status row)
    - Focused contradiction re-analysis
    - Attorney chat response generation
32. Added Data-Blitz branding and attribution updates:
    - Downloaded and included Data-Blitz logo in `frontend/assets/data-blitz-logo.png`
    - Rendered logo in UI header strip
    - Added ownership/footer notice in UI
    - Added formal ownership text to `README.md` and created `NOTICE.md`
33. Simplified top-of-UI branding to logo-only presentation and removed top header copy to reduce crowding.
34. Increased processing timer precision and update cadence to prevent \"stuck at zero\" perception:
    - Timer format changed to `mm:ss.cc`
    - Timer refresh interval changed to 50ms
35. Refactored timer implementation to explicit seconds clocks and reliable rendering:
    - Unified timer helpers (`startClock` / `stopClock`) for LLM, chat, and focused-reasoning clocks
    - Added paint synchronization before long LLM requests
    - Updated ingest status wording to: `Legal Clerk is processing depositions...`
36. Executed thorough validation suite:
    - Static checks: Node syntax + Python compile
    - API E2E: ingest, list, detail, chat, focused-reasoning endpoints
    - Browser E2E (Playwright): verified `chatClock` and `reasoningClock` values increase while requests are in-flight
    - Browser E2E (Playwright): verified legal clerk ingest status text and global status clock increments during processing

## Current State Summary

- Stack runs via `docker compose`.
- UI is available on configured API port (default `http://localhost:8000`).
- Deposition folder should be entered as `/data/depositions` in the UI.
- Contradiction list is score-ranked numerically, with timeline navigation and focused contradiction re-analysis.
