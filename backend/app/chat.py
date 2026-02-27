from __future__ import annotations

"""Attorney-oriented LLM chat and focused contradiction reasoning service."""

import json
import re
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Settings
from .llm import build_chat_model, llm_failure_message
from .prompts import render_prompt


class AttorneyChatService:
    """Builds and executes attorney prompts over deposition context."""

    def __init__(self, settings: Settings) -> None:
        """Initialize default OpenAI chat model and retain settings."""

        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

    def respond(
        self,
        deposition: dict,
        peers: list[dict],
        user_message: str,
        history: list[dict],
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> str:
        """Generate a general attorney response for the selected deposition."""

        response, _trace = self.respond_with_trace(
            deposition=deposition,
            peers=peers,
            user_message=user_message,
            history=history,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        return response

    def respond_with_trace(
        self,
        deposition: dict,
        peers: list[dict],
        user_message: str,
        history: list[dict],
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Generate attorney chat response plus visible process trace."""

        system_prompt = render_prompt("chat_system")

        context_payload = self._build_context_payload(deposition, peers)
        context_prompt = render_prompt(
            "chat_user_context",
            context_json=json.dumps(context_payload, indent=2),
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_prompt),
        ]

        for item in history[-8:]:
            role = item.get("role")
            content = item.get("content", "")
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=user_message))

        llm = self._get_llm(llm_provider, llm_model, temperature=0.2)
        start = perf_counter()

        try:
            result = llm.invoke(messages)
            response_text = str(result.content or "")
            normalized_response = self._normalize_chat_output(response_text, deposition, user_message)
            trace = [
                {
                    "persona": "Persona:Attorney",
                    "phase": "chat_response",
                    "file_name": str(deposition.get("file_name") or ""),
                    "llm_provider": self._normalize_provider(llm_provider),
                    "llm_model": self._trace_model_name(llm_model),
                    "input_preview": self._preview_text(
                        json.dumps(
                            {
                                "deposition_id": deposition.get("_id"),
                                "peer_count": len(peers),
                                "history_items": len(history),
                                "user_message": user_message,
                            },
                            indent=2,
                        ),
                        1800,
                    ),
                    "system_prompt": self._preview_text(system_prompt, 2500),
                    "user_prompt": self._preview_text(context_prompt, 5000),
                    "output_preview": self._preview_text(normalized_response, 5000),
                    "notes": f"Chat response generated in {int((perf_counter() - start) * 1000)}ms.",
                }
            ]
            return normalized_response, trace
        except Exception as exc:
            raise RuntimeError(
                llm_failure_message(self.settings, llm_provider, llm_model, exc)
            ) from exc

    def reason_about_contradiction(
        self,
        deposition: dict,
        peers: list[dict],
        contradiction: dict,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> str:
        """Generate focused reasoning for one contradiction finding."""

        system_prompt = render_prompt("reason_contradiction_system")
        context_payload = self._build_context_payload(deposition, peers)
        user_prompt = render_prompt(
            "reason_contradiction_user",
            contradiction_json=json.dumps(contradiction, indent=2),
            context_json=json.dumps(context_payload, indent=2),
        )

        llm = self._get_llm(llm_provider, llm_model, temperature=0.2)

        try:
            result = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return self._normalize_reasoning_output(
                str(result.content or ""),
                contradiction,
                context_payload,
            )
        except Exception as exc:
            raise RuntimeError(
                llm_failure_message(self.settings, llm_provider, llm_model, exc)
            ) from exc

    def _get_llm(self, llm_provider: str | None, llm_model: str | None, *, temperature: float):
        """Return default model or provider-specific override model."""

        if not llm_provider and not llm_model:
            return self.llm
        return build_chat_model(
            self.settings,
            llm_provider,
            llm_model,
            temperature=temperature,
        )

    def _trace_model_name(self, requested_model: str | None) -> str:
        """Resolve effective model name for trace metadata."""

        return (requested_model or self.settings.model_name or "").strip()

    def _normalize_provider(self, provider: str | None):
        """Normalize provider string for trace payload typing."""

        normalized = (provider or self.settings.default_llm_provider or "openai").strip().lower()
        return normalized if normalized in {"openai", "ollama"} else "openai"

    def _preview_text(self, value: str, limit: int) -> str:
        """Trim large text blocks for UI trace display."""

        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...(truncated)"

    def _normalize_chat_output(self, raw: str, deposition: dict, user_message: str) -> str:
        """Normalize attorney chat output into plain-language short answer + details."""

        defaults = self._default_chat_sections(deposition, user_message)
        parsed = self._parse_json_payload(raw)
        if parsed is not None:
            return self._render_chat_sections(self._merge_json_chat(parsed, defaults))

        short_answer = self._extract_short_answer(raw) or defaults["short_answer"]
        if self._contains_placeholder(short_answer):
            short_answer = defaults["short_answer"]
        details = self._extract_bullets(raw)
        if not details:
            details = list(defaults["details"])
        elif len(details) == 1 and len(defaults["details"]) > 1:
            details = [details[0], defaults["details"][1]]

        normalized = dict(defaults)
        normalized["short_answer"] = short_answer
        normalized["details"] = details[:4]
        return self._render_chat_sections(normalized)

    def _default_chat_sections(self, deposition: dict, user_message: str) -> dict:
        """Build deterministic fallback chat sections used when output formatting drifts."""

        witness_name = str(deposition.get("witness_name") or "this witness")
        score = int(deposition.get("contradiction_score") or 0)
        contradictions = deposition.get("contradictions") or []

        if contradictions:
            top = contradictions[0]
            topic = str(top.get("topic") or "key facts")
            other_witness = str(top.get("other_witness_name") or "another witness")
            rationale = str(top.get("rationale") or "stored contradictions")
            short_answer = f"{witness_name}'s testimony currently shows contradiction risk at {score}/100."
            details = [
                f"Primary conflict is on {topic} versus {other_witness}: {rationale}",
                "Next step: compare source quotes and timeline evidence for that disputed event.",
            ]
        else:
            short_answer = f"{witness_name}'s testimony is currently low-risk with a contradiction score of {score}/100."
            details = [
                "No direct contradictions are stored for this deposition yet.",
                "Next step: corroborate key facts with documents and additional witness statements.",
            ]

        message = str(user_message or "").strip()
        if message:
            details[1] = f"Requested focus: {message[:150]}"

        return {
            "short_answer": short_answer,
            "details": details,
        }

    def _merge_json_chat(self, payload: dict, defaults: dict) -> dict:
        """Merge JSON-style chat output into the normalized short-answer format."""

        merged = dict(defaults)

        short_answer = str(
            payload.get("short_answer")
            or payload.get("answer")
            or payload.get("summary")
            or payload.get("response")
            or ""
        ).strip()
        if short_answer:
            merged["short_answer"] = short_answer

        details = (
            payload.get("details")
            or payload.get("bullets")
            or payload.get("insights")
            or payload.get("analysis")
            or payload.get("next_steps")
            or []
        )
        if isinstance(details, str):
            details = [details]
        if isinstance(details, list):
            cleaned = [
                str(item).strip()
                for item in details
                if str(item).strip() and not self._contains_placeholder(str(item))
            ]
            if cleaned:
                merged["details"] = cleaned[:4]

        return merged

    def _render_chat_sections(self, sections: dict) -> str:
        """Render normalized attorney chat sections into one stable plain-text block."""

        details = sections.get("details") or []
        if not details:
            details = ["No additional details are available yet."]
        bullets = "\n".join(f"- {item}" for item in details[:4])
        return f"Short answer: {sections['short_answer']}\nDetails:\n{bullets}"

    def _fallback_chat_response(self, deposition: dict, user_message: str) -> str:
        """Legacy deterministic chat fallback (kept for test and utility use)."""

        witness_name = str(deposition.get("witness_name") or "this witness")
        score = int(deposition.get("contradiction_score") or 0)
        contradictions = deposition.get("contradictions") or []

        if contradictions:
            top = contradictions[0]
            topic = str(top.get("topic") or "key facts")
            other_witness = str(top.get("other_witness_name") or "another witness")
            rationale = str(top.get("rationale") or "stored contradictions")
            short_answer = f"{witness_name}'s testimony currently shows contradiction risk at {score}/100."
            bullets = [
                f"Primary conflict is on {topic} versus {other_witness}: {rationale}",
                "Next step: compare source quotes and timeline evidence for that disputed event.",
            ]
        else:
            short_answer = f"{witness_name}'s testimony is currently low-risk with a contradiction score of {score}/100."
            bullets = [
                "No direct contradictions are stored for this deposition yet.",
                "Next step: corroborate key facts with documents and additional witness statements.",
            ]

        if user_message.strip():
            bullets[1] = f"Requested focus: {user_message.strip()[:150]}"

        return (
            f"Short answer: {short_answer}\n"
            "Details:\n"
            f"- {bullets[0]}\n"
            f"- {bullets[1]}"
        )

    def _fallback_contradiction_reasoning(self, contradiction: dict) -> str:
        """Legacy deterministic contradiction fallback (utility/test coverage)."""

        severity = int(contradiction.get("severity") or 0)
        if severity >= 70:
            strength = "strong"
        elif severity >= 40:
            strength = "moderate"
        else:
            strength = "weak"

        topic = str(contradiction.get("topic") or "this topic")
        other_witness = str(contradiction.get("other_witness_name") or "the peer witness")
        rationale = str(contradiction.get("rationale") or "stored evidence")

        return (
            f"Short answer: This contradiction appears {strength} based on the currently stored evidence.\n"
            "Details:\n"
            f"- Topic in dispute: {topic} against testimony from {other_witness}.\n"
            f"- Current rationale: {rationale}\n"
            "- Next step: verify transcript excerpts and timeline artifacts tied to this point."
        )

    def _normalize_reasoning_output(
        self,
        raw: str,
        contradiction: dict,
        context_payload: dict,
    ) -> str:
        """Normalize focused reasoning into a stable, detailed plain-text format."""

        defaults = self._default_reasoning_sections(contradiction, context_payload)
        parsed = self._parse_json_payload(raw)
        if parsed is not None:
            return self._render_reasoning_sections(self._merge_json_reasoning(parsed, defaults))

        short_answer = self._extract_short_answer(raw) or defaults["short_answer"]
        if self._contains_placeholder(short_answer):
            short_answer = defaults["short_answer"]
        insight_lines = self._extract_bullets(raw)
        if not insight_lines:
            insight_lines = list(defaults["insights"])
        elif len(insight_lines) == 1:
            insight_lines = [insight_lines[0], *defaults["insights"][:1]]

        normalized = dict(defaults)
        normalized["short_answer"] = short_answer
        normalized["insights"] = insight_lines[:4]
        return self._render_reasoning_sections(normalized)

    def _parse_json_payload(self, raw: str) -> dict | None:
        """Parse direct JSON or fenced JSON payloads returned by less-consistent models."""

        text = raw.strip()
        if not text:
            return None

        for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return None

    def _contains_placeholder(self, raw: str) -> bool:
        """Detect placeholder/template output that should be replaced with concrete content."""

        text = raw.lower()
        if any(marker in text for marker in ("<1 sentence>", "<bullet>", "short answer: <")):
            return True
        return re.search(r"<[^>\n]{3,}>", raw) is not None

    def _extract_short_answer(self, raw: str) -> str:
        """Extract short-answer line from free-form text responses."""

        match = re.search(r"short answer:\s*(.+)", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines[0] if lines else ""

    def _extract_bullets(self, raw: str) -> list[str]:
        """Extract bullet-like lines from text responses."""

        bullets: list[str] = []
        for line in raw.splitlines():
            item = line.strip()
            if not item:
                continue
            if item.startswith(("- ", "* ")):
                content = item[2:].strip()
                if self._contains_placeholder(content):
                    continue
                bullets.append(content)
        return bullets

    def _default_reasoning_sections(self, contradiction: dict, context_payload: dict) -> dict:
        """Build a detailed deterministic scaffold for focused contradiction explanations."""

        severity = int(contradiction.get("severity") or 0)
        if severity >= 70:
            strength = "strong"
        elif severity >= 40:
            strength = "moderate"
        else:
            strength = "weak"

        topic = str(contradiction.get("topic") or "the disputed topic")
        other_witness = str(contradiction.get("other_witness_name") or "the peer witness")
        rationale = str(contradiction.get("rationale") or "stored contradiction evidence")
        target_dep = context_payload.get("target_deposition", {})
        peer = self._pick_peer_for_contradiction(context_payload, contradiction)

        target_claim = self._first_claim_statement(target_dep)
        peer_claim = self._first_claim_statement(peer)
        target_summary = str(target_dep.get("summary") or "").strip()
        peer_summary = str(peer.get("summary") or "").strip()

        return {
            "short_answer": (
                f"The contradiction on {topic} appears {strength} and should be tested directly "
                "before relying on this witness for a key fact."
            ),
            "strength": strength,
            "conflict_focus": f"{topic}: {rationale} (contrasted against {other_witness}).",
            "target_evidence": target_claim or target_summary or "Target testimony has limited detail in stored context.",
            "peer_evidence": peer_claim or peer_summary or "Peer testimony has limited detail in stored context.",
            "insights": [
                (
                    "The conflict goes to witness credibility on a concrete factual point "
                    "rather than a minor wording difference."
                ),
                "Timeline, event attribution, and corroborating records should be compared side-by-side.",
            ],
            "next_steps": [
                f"Pin both witnesses to precise facts on {topic} (who, what, and when).",
                "Confront each witness with the opposing account and ask for a specific reconciliation.",
                "Tie each answer to exhibits (logs, camera records, incident reports, transcript lines).",
            ],
        }

    def _pick_peer_for_contradiction(self, context_payload: dict, contradiction: dict) -> dict:
        """Pick the most relevant peer deposition for this contradiction context."""

        peers = context_payload.get("peer_depositions", [])
        if not peers:
            return {}

        other_witness = str(contradiction.get("other_witness_name") or "").strip().lower()
        for peer in peers:
            if str(peer.get("witness_name") or "").strip().lower() == other_witness:
                return peer
        return peers[0]

    def _first_claim_statement(self, deposition: dict) -> str:
        """Extract first claim statement from a deposition payload."""

        claims = deposition.get("claims", [])
        if not claims:
            return ""
        statement = str(claims[0].get("statement") or "").strip()
        return statement

    def _merge_json_reasoning(self, payload: dict, defaults: dict) -> dict:
        """Merge JSON-style model output into normalized reasoning sections."""

        merged = dict(defaults)

        short_answer = str(payload.get("short_answer") or payload.get("answer") or payload.get("summary") or "").strip()
        if short_answer:
            merged["short_answer"] = short_answer

        strength = str(payload.get("strength") or "").strip().lower()
        if strength in {"strong", "moderate", "weak"}:
            merged["strength"] = strength

        conflict_focus = str(payload.get("conflict_focus") or payload.get("focus") or "").strip()
        if conflict_focus:
            merged["conflict_focus"] = conflict_focus

        target_evidence = str(payload.get("target_evidence") or "").strip()
        if target_evidence:
            merged["target_evidence"] = target_evidence

        peer_evidence = str(payload.get("peer_evidence") or "").strip()
        if peer_evidence:
            merged["peer_evidence"] = peer_evidence

        insights = payload.get("insights") or payload.get("details") or []
        if isinstance(insights, str):
            insights = [insights]
        if isinstance(insights, list):
            cleaned = [str(item).strip() for item in insights if str(item).strip()]
            if cleaned:
                merged["insights"] = cleaned[:4]

        next_steps = payload.get("next_steps") or payload.get("recommended_actions") or []
        if isinstance(next_steps, str):
            next_steps = [next_steps]
        if isinstance(next_steps, list):
            cleaned_steps = [str(item).strip() for item in next_steps if str(item).strip()]
            if cleaned_steps:
                merged["next_steps"] = cleaned_steps[:3]

        return merged

    def _render_reasoning_sections(self, sections: dict) -> str:
        """Render normalized focused reasoning sections into one consistent text block."""

        insight_lines = sections.get("insights", [])
        if not insight_lines:
            insight_lines = ["The current record indicates material inconsistency that warrants direct examination."]

        steps = sections.get("next_steps", [])
        if not steps:
            steps = ["Run targeted follow-up questioning and reconcile with documentary evidence."]

        numbered_steps = "\n".join(f"{idx}. {item}" for idx, item in enumerate(steps, start=1))
        insight_bullets = "\n".join(f"- {item}" for item in insight_lines)

        return (
            f"Short answer: {sections['short_answer']}\n"
            f"Strength: {sections['strength'].title()}\n"
            "Conflict focus:\n"
            f"- {sections['conflict_focus']}\n"
            "Evidence comparison:\n"
            f"- Target testimony: {sections['target_evidence']}\n"
            f"- Peer testimony: {sections['peer_evidence']}\n"
            "Why this matters:\n"
            f"{insight_bullets}\n"
            "Recommended next steps:\n"
            f"{numbered_steps}"
        )

    def _build_context_payload(self, deposition: dict, peers: list[dict]) -> dict:
        """Assemble compact JSON context payload sent to chat prompts."""

        return {
            "target_deposition": {
                "id": deposition.get("_id"),
                "witness_name": deposition.get("witness_name"),
                "witness_role": deposition.get("witness_role"),
                "summary": deposition.get("summary"),
                "contradiction_score": deposition.get("contradiction_score"),
                "flagged": deposition.get("flagged"),
                "contradiction_explanation": deposition.get("contradiction_explanation"),
                "contradictions": deposition.get("contradictions", []),
                "claims": deposition.get("claims", []),
            },
            "peer_depositions": [
                {
                    "id": peer.get("_id"),
                    "witness_name": peer.get("witness_name"),
                    "summary": peer.get("summary"),
                    "claims": peer.get("claims", []),
                }
                for peer in peers
            ],
        }
