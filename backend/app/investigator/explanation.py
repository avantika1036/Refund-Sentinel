"""Investigation explanation service with provider-aware LLM synthesis.

The deterministic explanation path is always available and remains the safe
fallback when no provider is configured or an external request fails.
"""

from __future__ import annotations

import json
import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from backend.app.config import settings
from backend.app.investigator.evidence import EvidenceBundle


# A small process-local cache prevents repeated page refreshes from repeatedly
# calling the external LLM for the same evidence bundle. Only successful LLM
# responses are cached. Deterministic fallbacks are deliberately NOT cached so
# a transient 429/network failure does not permanently mask a newly available
# provider key or later successful request.
_EXPLANATION_CACHE: dict[str, InvestigationExplanation] = {}
_EXPLANATION_CACHE_MAX = 256


@dataclass(frozen=True)
class InvestigationExplanation:
    headline: str
    narrative_summary: str
    key_risk_drivers: list[str]
    suggested_action_rationale: str
    is_llm_generated: bool


class InvestigationExplanationService:
    """Produces concise investigator explanations from an EvidenceBundle."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or settings.llm_api_key
            or settings.gemini_api_key
            or settings.openai_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        self.provider = (provider or settings.llm_provider or "").strip().lower()
        self.model = (model or settings.llm_model or "").strip()
        self.timeout_seconds = float(timeout_seconds or settings.llm_timeout_seconds)
        
        print(f"[LLM] Initialized - Provider: {self.provider}, Model: {self.model}, API Key configured: {bool(self.api_key)}")

    def explain(self, bundle: EvidenceBundle) -> InvestigationExplanation:
        cache_key = self._cache_key(bundle)
        cached = _EXPLANATION_CACHE.get(cache_key)
        if cached is not None:
            print(f"[LLM] Using cached LLM explanation for refund {bundle.refund_id}")
            return cached

        if not self.api_key or not self.provider:
            print("[LLM] No API key or provider configured, using heuristic fallback")
            return self._generate_heuristic_narrative(bundle)

        try:
            print(f"[LLM] Attempting {self.provider} API call with model {self.model}")
            result = self._call_llm(bundle)
            print(
                f"[LLM] Successfully generated explanation "
                f"(is_llm_generated: {result.is_llm_generated})"
            )
            # Only successful provider-generated results are cached. A fallback
            # result must never prevent a later retry with a fresh provider key.
            if result.is_llm_generated:
                self._cache_result(cache_key, result)
            return result
        except urllib.error.HTTPError as error:
            if error.code == 429:
                print(
                    "[LLM] Provider rate limit/quota reached (HTTP 429); "
                    "using heuristic fallback without caching"
                )
            else:
                print(
                    f"[LLM] API call failed with HTTP {error.code}, "
                    "using heuristic fallback without caching"
                )
            return self._generate_heuristic_narrative(bundle)
        except Exception as error:
            print(
                f"[LLM] API call failed: {error}, "
                "using heuristic fallback without caching"
            )
            return self._generate_heuristic_narrative(bundle)

    def _cache_key(self, bundle: EvidenceBundle) -> str:
        """Cache by evidence + provider configuration, never by refund ID alone."""
        evidence = json.dumps(bundle.to_dict(), sort_keys=True, ensure_ascii=False)
        api_fingerprint = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]
        evidence_fingerprint = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
        return (
            f"{self.provider}:{self.model}:{api_fingerprint}:"
            f"{bundle.refund_id}:{evidence_fingerprint}"
        )

    @staticmethod
    def _cache_result(cache_key: str, result: InvestigationExplanation) -> None:
        if len(_EXPLANATION_CACHE) >= _EXPLANATION_CACHE_MAX:
            oldest_key = next(iter(_EXPLANATION_CACHE))
            _EXPLANATION_CACHE.pop(oldest_key, None)
        _EXPLANATION_CACHE[cache_key] = result

    def _call_llm(self, bundle: EvidenceBundle) -> InvestigationExplanation:
        prompt = self._build_prompt(bundle)
        if self.provider == "gemini":
            payload = self._call_gemini(prompt)
        elif self.provider == "openai":
            payload = self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return self._parse_llm_payload(payload)

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        model = self.model or "gemini-2.5-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        candidates = response_body.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts)
        
        # Try to parse as JSON, if fails return a structured response from text
        try:
            return self._parse_json_object(text)
        except json.JSONDecodeError:
            # Fallback: create structured response from plain text
            return {
                "headline": "Investigation Summary",
                "narrative_summary": text[:500],
                "key_risk_drivers": ["LLM-generated analysis"],
                "suggested_action_rationale": "Review the full evidence bundle.",
            }

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": self.model or "gpt-4o-mini",
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a careful fraud investigator. Never invent evidence."},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        choices = response_body.get("choices") or []
        if not choices:
            raise ValueError("OpenAI returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        return self._parse_json_object(str(content))

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return parsed

    def _build_prompt(self, bundle: EvidenceBundle) -> str:
        return (
            "Write a concise investigator explanation using ONLY this evidence. "
            "Do not infer missing facts. Return JSON with exactly these keys: "
            "headline, narrative_summary, key_risk_drivers, suggested_action_rationale. "
            "narrative_summary must be 2-3 sentences. key_risk_drivers must contain at most 4 short items.\n\n"
            + json.dumps(bundle.to_dict(), ensure_ascii=False)
        )

    def _parse_llm_payload(self, payload: dict[str, Any]) -> InvestigationExplanation:
        required = (
            "headline",
            "narrative_summary",
            "key_risk_drivers",
            "suggested_action_rationale",
        )
        if any(key not in payload for key in required):
            raise ValueError("LLM response is missing required explanation fields")
        drivers = payload["key_risk_drivers"]
        if not isinstance(drivers, list) or not all(isinstance(item, str) for item in drivers):
            raise ValueError("key_risk_drivers must be a list of strings")
        return InvestigationExplanation(
            headline=str(payload["headline"]).strip(),
            narrative_summary=str(payload["narrative_summary"]).strip(),
            key_risk_drivers=[item.strip() for item in drivers[:4] if item.strip()],
            suggested_action_rationale=str(payload["suggested_action_rationale"]).strip(),
            is_llm_generated=True,
        )

    def _generate_heuristic_narrative(self, bundle: EvidenceBundle) -> InvestigationExplanation:
        triggered_rules = [r for r in bundle.rule_violations if r.triggered]
        drivers: list[str] = []

        if bundle.graph_topology.cluster_size > 1:
            drivers.append(
                f"Connected component contains {bundle.graph_topology.cluster_size} refunds across "
                f"{len(bundle.graph_topology.connected_customer_ids)} customer accounts."
            )
        if bundle.graph_topology.shared_device_fingerprints:
            drivers.append(
                f"Shared devices link the connected accounts: "
                f"{len(bundle.graph_topology.shared_device_fingerprints)} shared device identifier(s)."
            )
        for rule in triggered_rules[:3]:
            drivers.append(f"{rule.rule_id}: {rule.notes}")
        if bundle.customer_profile.refund_rate_by_amount > 0.6:
            drivers.append(
                f"{bundle.customer_profile.refund_rate_by_amount * 100:.1f}% of the customer's order value is already refunded."
            )
        if not drivers:
            drivers.append("No deterministic rule breach or structural coordination indicator was found.")

        score = bundle.final_risk_score
        pending = bundle.financial_exposure.pending_refund_exposure_paise / 100
        if bundle.risk_level == "high":
            headline = f"HIGH RISK: Refund {bundle.refund_id} requires investigation"
            rationale = "Hold or investigate before further refund processing."
        elif bundle.risk_level == "medium":
            headline = f"ELEVATED RISK: Refund {bundle.refund_id} requires manual review"
            rationale = "Verify the supporting evidence before refund processing."
        else:
            headline = f"LOW RISK: No material anomaly confirmed for refund {bundle.refund_id}"
            rationale = "Standard processing is appropriate unless new evidence arrives."

        narrative = (
            f"The refund has a deterministic risk score of {score:.2f} and {len(triggered_rules)} triggered rule(s). "
            f"The current pending refund exposure associated with the investigated component is ₹{pending:,.2f}. "
            f"The evidence bundle should be interpreted with the graph topology and customer history shown for this case."
        )
        return InvestigationExplanation(
            headline=headline,
            narrative_summary=narrative,
            key_risk_drivers=drivers,
            suggested_action_rationale=rationale,
            is_llm_generated=False,
        )
