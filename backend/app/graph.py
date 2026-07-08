# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""LangGraph workflow for deposition mapping and contradiction analysis."""

import json
import re
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .config import Settings
from .couchdb import CouchDBClient
from .langfuse_integration import (
    build_langchain_config,
    propagate_trace_attributes,
    score_prompt_operation,
    score_skill_operation,
)
from .llm import build_chat_model, llm_failure_message, resolve_llm_selection
from .models import Claim, ContradictionAssessment, ContradictionFinding, DepositionSchema
from .prompts import render_prompt
from .schemas import load_schema, schema_mode


class GraphState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph workflow nodes."""

    case_id: str
    file_path: str
    llm_provider: str
    llm_model: str
    schema_name: str
    selected_schema: dict | None
    selected_schema_mode: str
    trace_session_id: str
    raw_text: str
    deposition: DepositionSchema
    deposition_doc: dict
    ingest_schema_mode: str
    ingest_schema_payload: dict | None
    other_depositions: list[dict]
    assessment: ContradictionAssessment
    legal_clerk_trace: list[dict]
    attorney_trace: list[dict]


class DepositionWorkflow:
    """Coordinates ingestion, LLM extraction, assessment, and persistence."""

    def __init__(self, settings: Settings, couchdb: CouchDBClient) -> None:
        """Initialize workflow dependencies and compile the graph."""

        self.settings = settings
        self.couchdb = couchdb
        self.default_llm_provider, self.default_llm_model = resolve_llm_selection(
            settings,
            None,
            None,
        )
        self.llm = build_chat_model(
            settings,
            self.default_llm_provider,
            self.default_llm_model,
            temperature=0,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph node/edge topology."""

        builder = StateGraph(GraphState)
        builder.add_node("read_file", self._read_file)
        builder.add_node("map_deposition", self._map_deposition)
        builder.add_node("save_deposition", self._save_deposition)
        builder.add_node("load_other_depositions", self._load_other_depositions)
        builder.add_node("evaluate_contradictions", self._evaluate_contradictions)
        builder.add_node("persist_assessment", self._persist_assessment)

        builder.set_entry_point("read_file")
        builder.add_edge("read_file", "map_deposition")
        builder.add_edge("map_deposition", "save_deposition")
        builder.add_edge("save_deposition", "load_other_depositions")
        builder.add_edge("load_other_depositions", "evaluate_contradictions")
        builder.add_edge("evaluate_contradictions", "persist_assessment")
        builder.add_edge("persist_assessment", END)
        return builder.compile()

    def run(
        self,
        case_id: str,
        file_path: str,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        schema_name: str | None = None,
        selected_schema: dict | None = None,
        selected_schema_mode: str | None = None,
        trace_session_id: str | None = None,
    ) -> dict:
        """Run the workflow for a single deposition file."""

        return self.graph.invoke(
            {
                "case_id": case_id,
                "file_path": file_path,
                "llm_provider": llm_provider or "",
                "llm_model": llm_model or "",
                "schema_name": schema_name or "deposition_schema",
                "selected_schema": selected_schema,
                "selected_schema_mode": selected_schema_mode or "",
                "trace_session_id": trace_session_id or case_id,
            }
        )

    def reassess_case(
        self,
        case_id: str,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        trace_session_id: str | None = None,
    ) -> list[dict]:
        """Recompute contradiction assessments for every deposition in a case."""

        docs = self.couchdb.list_depositions(case_id)
        updated_docs: list[dict] = []
        for doc in docs:
            peers = [peer for peer in docs if peer.get("_id") != doc.get("_id")]
            assessment = self._assess_deposition_against_peers(
                doc,
                peers,
                llm_provider=llm_provider,
                llm_model=llm_model,
                trace_session_id=trace_session_id or case_id,
            )
            doc["contradiction_score"] = assessment.contradiction_score
            doc["flagged"] = assessment.flagged
            doc["contradiction_explanation"] = assessment.explanation
            doc["contradictions"] = [item.model_dump() for item in assessment.contradictions]
            updated_docs.append(self.couchdb.update_doc(doc))
        return updated_docs

    def _read_file(self, state: GraphState) -> GraphState:
        """Read and decode deposition text with tolerant encoding fallbacks."""

        file_path = Path(state["file_path"])
        raw_bytes = file_path.read_bytes()
        raw_text: str | None = None
        encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
        for encoding in encodings:
            try:
                raw_text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if raw_text is None:
            raise ValueError(
                f"Could not decode deposition file {file_path.name} with supported encodings: {encodings}"
            )
        return {"raw_text": raw_text}

    def _map_deposition(self, state: GraphState) -> GraphState:
        """Map raw deposition text into a structured ``DepositionSchema``."""

        file_name = Path(state["file_path"]).name
        selected_schema_name = str(state.get("schema_name") or "deposition_schema").strip() or "deposition_schema"
        selected_schema = state.get("selected_schema")
        if not isinstance(selected_schema, dict):
            selected_schema = load_schema(selected_schema_name)
        selected_schema_mode = (
            str(state.get("selected_schema_mode") or "").strip() or schema_mode(selected_schema_name)
        )
        system_prompt = render_prompt("map_deposition_system")
        schema_json = json.dumps(selected_schema, indent=2)
        human_prompt = render_prompt(
            "map_deposition_user",
            case_id=state["case_id"],
            file_name=file_name,
            schema_json=schema_json,
            raw_text=state["raw_text"],
        )
        start = perf_counter()
        fallback_used = False
        parse_recovery_used = False
        ingest_schema_payload: dict | None = None

        try:
            llm, effective_provider, effective_model = self._get_llm(
                state.get("llm_provider"),
                state.get("llm_model"),
                temperature=0,
            )
            parser_llm = self._with_structured_output(
                llm,
                selected_schema,
                llm_provider=effective_provider,
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            invoke_config = self._langchain_config(
                operation="map_deposition",
                session_id=state.get("trace_session_id") or state.get("case_id"),
                llm_provider=effective_provider,
                llm_model=effective_model,
                metadata={
                    "case_id": str(state.get("case_id") or ""),
                    "file_name": file_name,
                    "schema_name": selected_schema_name,
                },
                tags=(
                    "attorneyos",
                    "category-prompt",
                    "category-skill",
                    "prompt-map-deposition",
                    "skill-deposition-mapping",
                    "ingest",
                    effective_provider,
                ),
            )
            with propagate_trace_attributes(
                self.settings,
                session_id=state.get("trace_session_id") or state.get("case_id"),
                tags=(
                    "attorneyos",
                    "category-prompt",
                    "category-skill",
                    "prompt-map-deposition",
                    "skill-deposition-mapping",
                    "ingest",
                    effective_provider,
                ),
                metadata={
                    "operation": "map_deposition",
                    "case_id": str(state.get("case_id") or ""),
                    "file_name": file_name,
                    "schema_name": selected_schema_name,
                },
            ):
                if invoke_config:
                    parsed = parser_llm.invoke(messages, config=invoke_config)
                else:
                    parsed = parser_llm.invoke(messages)
            ingest_schema_payload = self._coerce_schema_payload(parsed)
            deposition = self._normalize_ingest_deposition(
                selected_schema_name,
                ingest_schema_payload,
                state,
                file_name,
            )
            if selected_schema_mode != "native":
                fallback_used = True
        except Exception as exc:
            if self._is_structured_parse_failure(exc):
                deposition = self._fallback_map_deposition(state, file_name)
                fallback_used = True
                parse_recovery_used = True
            else:
                raise RuntimeError(
                    llm_failure_message(
                        self.settings,
                        state.get("llm_provider"),
                        state.get("llm_model"),
                        exc,
                    )
                ) from exc

        if parse_recovery_used:
            deposition.case_id = state["case_id"]
            deposition.file_name = file_name
            ingest_schema_payload = self._finalize_ingest_schema_payload(
                selected_schema_mode=selected_schema_mode,
                deposition=deposition,
                ingest_schema_payload=ingest_schema_payload,
            )
            score_prompt_operation(
                self.settings,
                operation="map_deposition",
                prompt_text=state["raw_text"],
                response_text=json.dumps(deposition.model_dump(), indent=2),
                system_prompt=system_prompt,
                user_prompt=human_prompt,
                prompt_template_keys=["map_deposition_system", "map_deposition_user"],
                extra_metrics={
                    "raw_text_chars": len(state["raw_text"]),
                    "claim_count": len(deposition.claims),
                    "schema_mode_native": selected_schema_mode == "native",
                    "parse_recovery_used": True,
                    "fallback_used": True,
                },
            )
            score_skill_operation(
                self.settings,
                skill_name="deposition_mapping",
                metrics={
                    "success": True,
                    "claim_count": len(deposition.claims),
                    "raw_text_chars": len(state["raw_text"]),
                    "parse_recovery_used": True,
                    "fallback_used": True,
                },
            )
            legal_clerk_trace = [
                {
                    "persona": "Persona:Legal Clerk",
                    "phase": "map_deposition",
                    "file_name": file_name,
                    "llm_provider": effective_provider,
                    "llm_model": effective_model,
                    "schema_name": selected_schema_name,
                    "input_preview": self._preview_text(state["raw_text"], 1200),
                    "system_prompt": self._preview_text(system_prompt, 2500),
                    "user_prompt": self._preview_text(human_prompt, 6000),
                    "output_preview": self._preview_text(
                        json.dumps(deposition.model_dump(), indent=2),
                        5000,
                    ),
                    "notes": (
                        f"Structured mapping recovered via fallback in {int((perf_counter() - start) * 1000)}ms "
                        f"after parse failure using schema '{selected_schema_name}'."
                    ),
                }
            ]
            return {
                "deposition": deposition,
                "legal_clerk_trace": legal_clerk_trace,
                "ingest_schema_mode": selected_schema_mode,
                "ingest_schema_payload": ingest_schema_payload,
            }

        # Some local models under-produce structured claims; backfill with deterministic parsing
        # so downstream contradiction assessment has usable evidence.
        if not deposition.claims:
            fallback = self._fallback_map_deposition(state, file_name)
            deposition.claims = fallback.claims
            fallback_used = True

        deposition.case_id = state["case_id"]
        deposition.file_name = file_name
        ingest_schema_payload = self._finalize_ingest_schema_payload(
            selected_schema_mode=selected_schema_mode,
            deposition=deposition,
            ingest_schema_payload=ingest_schema_payload,
        )
        score_prompt_operation(
            self.settings,
            operation="map_deposition",
            prompt_text=state["raw_text"],
            response_text=json.dumps(deposition.model_dump(), indent=2),
            system_prompt=system_prompt,
            user_prompt=human_prompt,
            prompt_template_keys=["map_deposition_system", "map_deposition_user"],
            extra_metrics={
                "raw_text_chars": len(state["raw_text"]),
                "claim_count": len(deposition.claims),
                "schema_mode_native": selected_schema_mode == "native",
                "parse_recovery_used": False,
                "fallback_used": fallback_used,
            },
        )
        score_skill_operation(
            self.settings,
            skill_name="deposition_mapping",
            metrics={
                "success": True,
                "claim_count": len(deposition.claims),
                "raw_text_chars": len(state["raw_text"]),
                "parse_recovery_used": False,
                "fallback_used": fallback_used,
            },
        )
        legal_clerk_trace = [
            {
                "persona": "Persona:Legal Clerk",
                "phase": "map_deposition",
                "file_name": file_name,
                "llm_provider": effective_provider,
                "llm_model": effective_model,
                "schema_name": selected_schema_name,
                "input_preview": self._preview_text(state["raw_text"], 1200),
                "system_prompt": self._preview_text(system_prompt, 2500),
                "user_prompt": self._preview_text(human_prompt, 6000),
                "output_preview": self._preview_text(
                    json.dumps(deposition.model_dump(), indent=2),
                    5000,
                ),
                "notes": (
                    f"Structured mapping completed in {int((perf_counter() - start) * 1000)}ms."
                    + f" Schema={selected_schema_name}."
                    + (" Fallback claim extraction was used." if fallback_used else "")
                ),
            }
        ]
        return {
            "deposition": deposition,
            "legal_clerk_trace": legal_clerk_trace,
            "ingest_schema_mode": selected_schema_mode,
            "ingest_schema_payload": ingest_schema_payload,
        }

    def _coerce_schema_payload(self, parsed: object) -> dict | None:
        """Normalize a structured LLM response into a plain ``dict`` when possible."""

        if isinstance(parsed, DepositionSchema):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            payload = model_dump()
            if isinstance(payload, dict):
                return payload
        return None

    def _finalize_ingest_schema_payload(
        self,
        *,
        selected_schema_mode: str,
        deposition: DepositionSchema,
        ingest_schema_payload: dict | None,
    ) -> dict | None:
        """Persist a stable payload shape for the selected schema mode."""

        if selected_schema_mode == "native":
            return deposition.model_dump()
        return ingest_schema_payload

    def _normalize_ingest_deposition(
        self,
        selected_schema_name: str,
        payload: dict | None,
        state: GraphState,
        file_name: str,
    ) -> DepositionSchema:
        """Convert arbitrary schema output into the app's runtime deposition model."""

        if selected_schema_name == "deposition_schema" and payload is not None:
            return DepositionSchema.model_validate(payload)
        return self._fallback_map_deposition(state, file_name)

    def _fallback_map_deposition(self, state: GraphState, file_name: str) -> DepositionSchema:
        """Heuristic non-LLM mapping helper retained for testability/backstop."""

        raw_text = state["raw_text"]
        lines = [line.strip() for line in raw_text.splitlines()]
        non_empty = [line for line in lines if line]

        witness_name = self._extract_first_group(raw_text, r"^\s*Deposition\s+of\s+(.+?)\s*$")
        if not witness_name:
            witness_name = self._extract_first_group(raw_text, r"^\s*Witness\s*:\s*(.+?)\s*$")
        if not witness_name:
            witness_name = Path(file_name).stem.replace("_", " ").replace("-", " ").title()

        deposition_date = self._extract_first_group(raw_text, r"^\s*Date\s*:\s*(.+?)\s*$")
        witness_role = self._extract_first_group(raw_text, r"^\s*Role\s*:\s*(.+?)\s*$") or "Unknown role"

        body_lines = [
            line
            for line in non_empty
            if not line.lower().startswith(("deposition of", "date:", "role:"))
        ]
        body_text = " ".join(body_lines)
        sentences = self._split_sentences(body_text)

        claims: list[Claim] = []
        for sentence in sentences[:8]:
            topic = self._infer_topic(sentence)
            claims.append(
                Claim(
                    topic=topic,
                    statement=sentence,
                    confidence=0.55,
                    source_quote=sentence,
                )
            )

        summary = " ".join(sentences[:2]).strip()
        if not summary:
            summary = body_lines[0] if body_lines else "No narrative testimony was available."

        return DepositionSchema(
            case_id=state["case_id"],
            file_name=file_name,
            witness_name=witness_name,
            witness_role=witness_role,
            deposition_date=deposition_date,
            summary=summary,
            claims=claims,
        )

    def _extract_first_group(self, text: str, pattern: str) -> str | None:
        """Extract first regex capture group or ``None`` when absent."""

        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            return None
        value = match.group(1).strip()
        return value or None

    def _split_sentences(self, text: str) -> list[str]:
        """Split free-form text into rough sentence units."""

        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _infer_topic(self, sentence: str) -> str:
        """Infer a short topic label from a claim sentence."""

        lowered = sentence.lower()
        topic_keywords: list[tuple[str, tuple[str, ...]]] = [
            ("Arrival time", ("arrived", "arrival", "a.m", "p.m", "morning", "evening", "time")),
            ("Camera status", ("camera", "video", "recording", "footage")),
            ("Forklift incident", ("forklift", "strike", "hit", "rack", "collision")),
            ("Safety report", ("safety", "supervisor", "reported", "told", "notified")),
            ("Spill details", ("leak", "drum", "solvent", "spill", "damaged")),
        ]
        for topic, keywords in topic_keywords:
            if any(keyword in lowered for keyword in keywords):
                return topic

        compact = re.sub(r"[^a-zA-Z0-9\s]", "", sentence).strip()
        if not compact:
            return "General testimony"
        words = compact.split()
        return " ".join(words[:6]).title()

    def _save_deposition(self, state: GraphState) -> GraphState:
        """Persist mapped deposition content into CouchDB."""

        deposition = state["deposition"]
        file_stem = Path(deposition.file_name).stem
        normalized_case_id = re.sub(r"[^a-zA-Z0-9]+", "-", state["case_id"]).strip("-").lower() or "case"
        normalized_stem = re.sub(r"[^a-zA-Z0-9]+", "-", file_stem).strip("-").lower() or "deposition"
        doc = {
            "_id": f"dep:{normalized_case_id}:{normalized_stem}",
            "type": "deposition",
            "case_id": deposition.case_id,
            "file_name": deposition.file_name,
            "witness_name": deposition.witness_name,
            "witness_role": deposition.witness_role,
            "deposition_date": deposition.deposition_date,
            "summary": deposition.summary,
            "claims": [claim.model_dump() for claim in deposition.claims],
            "ingest_schema_name": str(state.get("schema_name") or "deposition_schema"),
            "ingest_schema_mode": str(state.get("ingest_schema_mode") or "native"),
            "ingest_schema_payload": state.get("ingest_schema_payload"),
            "raw_text": state["raw_text"],
            "contradiction_score": 0,
            "flagged": False,
            "contradiction_explanation": "",
            "contradictions": [],
        }
        saved = self.couchdb.save_doc(doc)
        return {"deposition_doc": saved}

    def _load_other_depositions(self, state: GraphState) -> GraphState:
        """Load peer depositions in the same case (bounded context window)."""

        current_id = state["deposition_doc"]["_id"]
        all_depositions = self.couchdb.list_depositions(state["case_id"])
        others = [doc for doc in all_depositions if doc.get("_id") != current_id]
        return {"other_depositions": others[: self.settings.max_context_depositions]}

    def _evaluate_contradictions(self, state: GraphState) -> GraphState:
        """Evaluate target deposition against peers and attach assessment."""

        attorney_trace: list[dict] = []
        assessment = self._assess_deposition_against_peers(
            state["deposition_doc"],
            state.get("other_depositions", []),
            llm_provider=state.get("llm_provider"),
            llm_model=state.get("llm_model"),
            trace_events=attorney_trace,
            trace_session_id=state.get("trace_session_id"),
        )
        return {"assessment": assessment, "attorney_trace": attorney_trace}

    def _assess_deposition_against_peers(
        self,
        target_doc: dict,
        other_depositions: list[dict],
        llm_provider: str | None = None,
        llm_model: str | None = None,
        trace_events: list[dict] | None = None,
        trace_session_id: str | None = None,
    ) -> ContradictionAssessment:
        """Run contradiction assessment using structured-output LLM calls."""

        start = perf_counter()
        if not other_depositions:
            if trace_events is not None:
                trace_events.append(
                    {
                        "persona": "Persona:Attorney",
                        "phase": "assess_contradictions",
                        "file_name": str(target_doc.get("file_name") or ""),
                        "llm_provider": self._normalize_provider(llm_provider),
                        "llm_model": self._trace_model_name(llm_model),
                        "input_preview": self._preview_text(
                            json.dumps(
                                {
                                    "target_id": target_doc.get("_id"),
                                    "target_witness": target_doc.get("witness_name"),
                                    "peer_count": 0,
                                },
                                indent=2,
                            ),
                            1200,
                        ),
                        "output_preview": "No contradiction assessment was needed because there were no peer depositions.",
                        "notes": f"Skipped in {int((perf_counter() - start) * 1000)}ms.",
                    }
                )
            return ContradictionAssessment(
                contradiction_score=0,
                flagged=False,
                explanation="No peer depositions were available for contradiction analysis.",
                contradictions=[],
            )

        target = {
            "id": target_doc["_id"],
            "witness_name": target_doc["witness_name"],
            "witness_role": target_doc["witness_role"],
            "summary": target_doc["summary"],
            "claims": target_doc["claims"],
        }
        others = [
            {
                "id": item.get("_id"),
                "witness_name": item.get("witness_name"),
                "witness_role": item.get("witness_role"),
                "summary": item.get("summary"),
                "claims": item.get("claims", []),
            }
            for item in other_depositions
        ]

        system_prompt = render_prompt("assess_contradictions_system")
        human_prompt = render_prompt(
            "assess_contradictions_user",
            target_json=json.dumps(target, indent=2),
            others_json=json.dumps(others, indent=2),
        )

        result_source = "llm"
        try:
            llm, effective_provider, effective_model = self._get_llm(
                llm_provider,
                llm_model,
                temperature=0,
            )
            assessor_llm = self._with_structured_output(
                llm,
                ContradictionAssessment,
                llm_provider=effective_provider,
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            invoke_config = self._langchain_config(
                operation="assess_contradictions",
                session_id=trace_session_id or str(target_doc.get("case_id") or ""),
                llm_provider=effective_provider,
                llm_model=effective_model,
                metadata={
                    "case_id": str(target_doc.get("case_id") or ""),
                    "deposition_id": str(target_doc.get("_id") or ""),
                    "file_name": str(target_doc.get("file_name") or ""),
                    "peer_count": len(other_depositions),
                },
                tags=(
                    "attorneyos",
                    "category-prompt",
                    "category-skill",
                    "prompt-assess-contradictions",
                    "skill-contradiction-assessment",
                    "ingest",
                    effective_provider,
                ),
            )
            with propagate_trace_attributes(
                self.settings,
                session_id=trace_session_id or str(target_doc.get("case_id") or ""),
                tags=(
                    "attorneyos",
                    "category-prompt",
                    "category-skill",
                    "prompt-assess-contradictions",
                    "skill-contradiction-assessment",
                    "ingest",
                    effective_provider,
                ),
                metadata={
                    "operation": "assess_contradictions",
                    "case_id": str(target_doc.get("case_id") or ""),
                    "deposition_id": str(target_doc.get("_id") or ""),
                    "file_name": str(target_doc.get("file_name") or ""),
                },
            ):
                if invoke_config:
                    assessment = assessor_llm.invoke(messages, config=invoke_config)
                else:
                    assessment = assessor_llm.invoke(messages)
            if assessment.contradiction_score > 0 and assessment.contradictions:
                score_prompt_operation(
                    self.settings,
                    operation="assess_contradictions",
                    prompt_text=json.dumps(target, indent=2),
                    response_text=json.dumps(assessment.model_dump(), indent=2),
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    prompt_template_keys=["assess_contradictions_system", "assess_contradictions_user"],
                    extra_metrics={
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(assessment.contradictions),
                        "result_source": result_source,
                        "parse_recovery_used": False,
                    },
                )
                score_skill_operation(
                    self.settings,
                    skill_name="contradiction_assessment",
                    metrics={
                        "success": True,
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(assessment.contradictions),
                        "result_source": result_source,
                    },
                )
                self._append_assessment_trace(
                    trace_events,
                    target_doc=target_doc,
                    llm_provider=effective_provider,
                    llm_model=effective_model,
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    assessment=assessment,
                    elapsed_ms=int((perf_counter() - start) * 1000),
                    result_source=result_source,
                )
                return assessment

            # If model output is overly conservative, use deterministic cross-claim checks.
            fallback = self._fallback_assess_deposition(target_doc, other_depositions)
            if fallback.contradiction_score > assessment.contradiction_score and fallback.contradictions:
                result_source = "fallback"
                score_prompt_operation(
                    self.settings,
                    operation="assess_contradictions",
                    prompt_text=json.dumps(target, indent=2),
                    response_text=json.dumps(fallback.model_dump(), indent=2),
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    prompt_template_keys=["assess_contradictions_system", "assess_contradictions_user"],
                    extra_metrics={
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(fallback.contradictions),
                        "result_source": result_source,
                        "parse_recovery_used": False,
                    },
                )
                score_skill_operation(
                    self.settings,
                    skill_name="contradiction_assessment",
                    metrics={
                        "success": True,
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(fallback.contradictions),
                        "result_source": result_source,
                    },
                )
                self._append_assessment_trace(
                    trace_events,
                    target_doc=target_doc,
                    llm_provider=effective_provider,
                    llm_model=effective_model,
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    assessment=fallback,
                    elapsed_ms=int((perf_counter() - start) * 1000),
                    result_source=result_source,
                )
                return fallback
            score_prompt_operation(
                self.settings,
                operation="assess_contradictions",
                prompt_text=json.dumps(target, indent=2),
                response_text=json.dumps(assessment.model_dump(), indent=2),
                system_prompt=system_prompt,
                user_prompt=human_prompt,
                prompt_template_keys=["assess_contradictions_system", "assess_contradictions_user"],
                extra_metrics={
                    "peer_count": len(other_depositions),
                    "contradiction_count": len(assessment.contradictions),
                    "result_source": result_source,
                    "parse_recovery_used": False,
                },
            )
            score_skill_operation(
                self.settings,
                skill_name="contradiction_assessment",
                metrics={
                    "success": True,
                    "peer_count": len(other_depositions),
                    "contradiction_count": len(assessment.contradictions),
                    "result_source": result_source,
                },
            )
            self._append_assessment_trace(
                trace_events,
                target_doc=target_doc,
                llm_provider=effective_provider,
                llm_model=effective_model,
                system_prompt=system_prompt,
                user_prompt=human_prompt,
                assessment=assessment,
                elapsed_ms=int((perf_counter() - start) * 1000),
                result_source=result_source,
            )
            return assessment
        except Exception as exc:
            if self._is_structured_parse_failure(exc):
                fallback = self._fallback_assess_deposition(target_doc, other_depositions)
                score_prompt_operation(
                    self.settings,
                    operation="assess_contradictions",
                    prompt_text=json.dumps(target, indent=2),
                    response_text=json.dumps(fallback.model_dump(), indent=2),
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    prompt_template_keys=["assess_contradictions_system", "assess_contradictions_user"],
                    extra_metrics={
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(fallback.contradictions),
                        "result_source": "fallback_parse_recovery",
                        "parse_recovery_used": True,
                    },
                )
                score_skill_operation(
                    self.settings,
                    skill_name="contradiction_assessment",
                    metrics={
                        "success": True,
                        "peer_count": len(other_depositions),
                        "contradiction_count": len(fallback.contradictions),
                        "result_source": "fallback_parse_recovery",
                    },
                )
                self._append_assessment_trace(
                    trace_events,
                    target_doc=target_doc,
                    llm_provider=effective_provider,
                    llm_model=effective_model,
                    system_prompt=system_prompt,
                    user_prompt=human_prompt,
                    assessment=fallback,
                    elapsed_ms=int((perf_counter() - start) * 1000),
                    result_source="fallback_parse_recovery",
                )
                return fallback
            score_skill_operation(
                self.settings,
                skill_name="contradiction_assessment",
                metrics={
                    "success": False,
                    "peer_count": len(other_depositions),
                    "error_kind": type(exc).__name__,
                },
            )
            raise RuntimeError(
                llm_failure_message(
                    self.settings,
                    llm_provider,
                    llm_model,
                    exc,
                )
            ) from exc

    def _is_structured_parse_failure(self, exc: Exception) -> bool:
        """Return ``True`` when an exception indicates structured-output parsing failure."""

        cursor: Exception | None = exc
        while cursor is not None:
            text = str(cursor).lower()
            if (
                "failed to parse" in text
                or "output_parsing_failure" in text
                or "validation error" in text
                or "jsondecodeerror" in text
            ):
                return True
            cursor = cursor.__cause__ if isinstance(cursor.__cause__, Exception) else None
        return False

    def _append_assessment_trace(
        self,
        trace_events: list[dict] | None,
        *,
        target_doc: dict,
        llm_provider: str | None,
        llm_model: str | None,
        system_prompt: str,
        user_prompt: str,
        assessment: ContradictionAssessment,
        elapsed_ms: int,
        result_source: str,
    ) -> None:
        """Append one attorney assessment trace event."""

        if trace_events is None:
            return
        trace_events.append(
            {
                "persona": "Persona:Attorney",
                "phase": "assess_contradictions",
                "file_name": str(target_doc.get("file_name") or ""),
                "llm_provider": self._normalize_provider(llm_provider),
                "llm_model": self._trace_model_name(llm_model),
                "input_preview": self._preview_text(
                    json.dumps(
                        {
                            "target_id": target_doc.get("_id"),
                            "target_witness": target_doc.get("witness_name"),
                            "target_claims": len(target_doc.get("claims", [])),
                            "result_source": result_source,
                        },
                        indent=2,
                    ),
                    1400,
                ),
                "system_prompt": self._preview_text(system_prompt, 2500),
                "user_prompt": self._preview_text(user_prompt, 6000),
                "output_preview": self._preview_text(
                    json.dumps(assessment.model_dump(), indent=2),
                    5000,
                ),
                "notes": f"Assessment completed in {elapsed_ms}ms via {result_source}.",
            }
        )

    def _trace_model_name(self, requested_model: str | None) -> str:
        """Resolve effective model name for trace metadata."""

        return resolve_llm_selection(self.settings, None, requested_model)[1]

    def _normalize_provider(self, provider: str | None):
        """Normalize provider string for trace payload typing."""

        return resolve_llm_selection(self.settings, provider, None)[0]

    def _preview_text(self, value: str, limit: int) -> str:
        """Trim large text blocks for UI trace display."""

        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...(truncated)"

    def _with_structured_output(self, llm, schema: object, llm_provider: str | None):
        """Wrap structured-output setup with Ollama-specific compatibility mode."""

        if self._normalize_provider(llm_provider) == "ollama":
            # Avoid Ollama json_schema grammar crashes on certain schema shapes.
            return llm.with_structured_output(schema, method="json_mode")
        return llm.with_structured_output(schema)

    def _get_llm(self, llm_provider: str | None, llm_model: str | None, *, temperature: float):
        """Return the effective workflow model together with its resolved provider/model."""

        provider, model = resolve_llm_selection(self.settings, llm_provider, llm_model)
        if provider == self.default_llm_provider and model == self.default_llm_model:
            return self.llm, provider, model
        return (
            build_chat_model(
                self.settings,
                provider,
                model,
                temperature=temperature,
            ),
            provider,
            model,
        )

    def _langchain_config(
        self,
        *,
        operation: str,
        session_id: str | None,
        llm_provider: str | None,
        llm_model: str | None,
        metadata: dict | None = None,
        tags: tuple[str, ...] | list[str] = (),
    ) -> dict:
        """Build one LangChain config payload enriched for Langfuse tracing."""

        provider, model = resolve_llm_selection(self.settings, llm_provider, llm_model)
        payload = {
            "llm_provider": provider,
            "llm_model": model,
        }
        if isinstance(metadata, dict):
            payload.update(metadata)
        return build_langchain_config(
            self.settings,
            operation=operation,
            session_id=session_id,
            tags=list(tags),
            metadata=payload,
        )

    def _fallback_assess_deposition(
        self, target_doc: dict, other_depositions: list[dict]
    ) -> ContradictionAssessment:
        """Heuristic contradiction assessor used in utility/test scenarios."""

        findings: list[ContradictionFinding] = []
        seen: set[str] = set()

        target_claims = target_doc.get("claims", [])
        for peer in other_depositions:
            peer_claims = peer.get("claims", [])
            for claim in target_claims[:12]:
                claim_topic = str(claim.get("topic", ""))
                claim_statement = str(claim.get("statement", ""))
                if not claim_statement:
                    continue

                for peer_claim in peer_claims[:12]:
                    peer_topic = str(peer_claim.get("topic", ""))
                    peer_statement = str(peer_claim.get("statement", ""))
                    if not peer_statement:
                        continue

                    if not self._topics_overlap(claim_topic, peer_topic, claim_statement, peer_statement):
                        continue

                    conflict, rationale, severity = self._claims_conflict(claim_statement, peer_statement)
                    if not conflict:
                        continue

                    key = f"{peer.get('_id')}::{claim_topic.lower()}::{rationale.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)

                    findings.append(
                        ContradictionFinding(
                            other_deposition_id=str(peer.get("_id", "unknown")),
                            other_witness_name=str(peer.get("witness_name", "Unknown witness")),
                            topic=claim_topic or "Potential contradiction",
                            rationale=rationale,
                            severity=severity,
                        )
                    )
                    if len(findings) >= 8:
                        break
                if len(findings) >= 8:
                    break
            if len(findings) >= 8:
                break

        if not findings:
            return ContradictionAssessment(
                contradiction_score=0,
                flagged=False,
                explanation="No clear direct contradictions were found in fallback analysis.",
                contradictions=[],
            )

        peak = max(item.severity for item in findings)
        contradiction_score = min(100, peak + min(35, (len(findings) - 1) * 10))
        flagged = contradiction_score >= 35
        explanation = (
            f"Fallback analysis found {len(findings)} potential contradiction(s) across peer depositions."
        )
        return ContradictionAssessment(
            contradiction_score=contradiction_score,
            flagged=flagged,
            explanation=explanation,
            contradictions=findings,
        )

    def _topics_overlap(
        self, left_topic: str, right_topic: str, left_statement: str, right_statement: str
    ) -> bool:
        """Determine whether two claims likely refer to the same subject."""

        left_tokens = self._keyword_tokens(left_topic)
        right_tokens = self._keyword_tokens(right_topic)
        if left_tokens and right_tokens and left_tokens.intersection(right_tokens):
            return True

        left_statement_tokens = self._keyword_tokens(left_statement)
        right_statement_tokens = self._keyword_tokens(right_statement)
        shared = left_statement_tokens.intersection(right_statement_tokens)
        return len(shared) >= 3

    def _keyword_tokens(self, value: str) -> set[str]:
        """Extract informative keyword tokens, excluding common stopwords."""

        words = {token for token in re.findall(r"[a-zA-Z]+", value.lower()) if len(token) >= 3}
        stopwords = {
            "the",
            "and",
            "that",
            "with",
            "from",
            "this",
            "were",
            "been",
            "about",
            "around",
            "witness",
            "deposition",
            "said",
            "they",
            "their",
        }
        return {token for token in words if token not in stopwords}

    def _claims_conflict(self, left_statement: str, right_statement: str) -> tuple[bool, str, int]:
        """Detect conflict patterns between two claim statements."""

        left_text = left_statement.lower()
        right_text = right_statement.lower()

        if left_text == right_text:
            return False, "", 0

        left_neg = self._contains_negation(left_text)
        right_neg = self._contains_negation(right_text)
        shared_keywords = self._keyword_tokens(left_statement).intersection(self._keyword_tokens(right_statement))

        if left_neg != right_neg and shared_keywords:
            return (
                True,
                "One witness denies a fact that another witness affirms on the same topic.",
                70,
            )

        left_time = self._extract_time_minutes(left_text)
        right_time = self._extract_time_minutes(right_text)
        if left_time is not None and right_time is not None and abs(left_time - right_time) >= 15:
            return (
                True,
                "Witnesses provide materially different timelines for the same event.",
                65,
            )

        left_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", left_text))
        right_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", right_text))
        if left_numbers and right_numbers and left_numbers != right_numbers and shared_keywords:
            return (
                True,
                "Witnesses provide conflicting quantitative details.",
                55,
            )

        return False, "", 0

    def _contains_negation(self, value: str) -> bool:
        """Check whether statement text contains negation semantics."""

        negation_tokens = (
            " not ",
            " no ",
            " never ",
            "none",
            "didn't",
            "did not",
            "wasn't",
            "was not",
            "weren't",
            "were not",
            "cannot",
            "can't",
            "without",
        )
        normalized = f" {value.strip()} "
        return any(token in normalized for token in negation_tokens)

    def _extract_time_minutes(self, value: str) -> int | None:
        """Parse a clock timestamp from text and convert to minutes since midnight."""

        match = re.search(r"\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am|pm)?(?:\b|$)", value, flags=re.IGNORECASE)
        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2))
        meridiem = (match.group(3) or "").lower()

        if meridiem in {"p.m.", "pm"} and hour < 12:
            hour += 12
        if meridiem in {"a.m.", "am"} and hour == 12:
            hour = 0

        return hour * 60 + minute

    def _persist_assessment(self, state: GraphState) -> GraphState:
        """Persist contradiction assessment fields to the deposition document."""

        doc = state["deposition_doc"]
        assessment = state["assessment"]

        doc["contradiction_score"] = assessment.contradiction_score
        doc["flagged"] = assessment.flagged
        doc["contradiction_explanation"] = assessment.explanation
        doc["contradictions"] = [item.model_dump() for item in assessment.contradictions]

        saved = self.couchdb.update_doc(doc)
        return {"deposition_doc": saved}
