"""Regression tests for safe investigation explanation caching."""

from __future__ import annotations

import urllib.error

import backend.app.investigator.explanation as explanation_module
from backend.app.investigator.explanation import (
    InvestigationExplanation,
    InvestigationExplanationService,
)


class _DummyBundle:
    refund_id = "refund-1"

    def to_dict(self):
        return {"refund_id": self.refund_id, "evidence_version": 1}


def test_fallback_is_not_cached_after_provider_failure(monkeypatch):
    explanation_module._EXPLANATION_CACHE.clear()
    service = InvestigationExplanationService(
        api_key="test-key",
        provider="gemini",
        model="gemini-2.5-flash",
    )
    calls = {"count": 0}

    def rate_limited(_bundle):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            {},
            None,
        )

    fallback = InvestigationExplanation(
        "fallback", "fallback", ["fallback"], "fallback", False
    )
    monkeypatch.setattr(service, "_call_llm", rate_limited)
    monkeypatch.setattr(service, "_generate_heuristic_narrative", lambda _bundle: fallback)

    service.explain(_DummyBundle())
    service.explain(_DummyBundle())

    assert calls["count"] == 2
    assert not explanation_module._EXPLANATION_CACHE


def test_successful_llm_result_is_cached(monkeypatch):
    explanation_module._EXPLANATION_CACHE.clear()
    service = InvestigationExplanationService(
        api_key="test-key",
        provider="gemini",
        model="gemini-2.5-flash",
    )
    calls = {"count": 0}
    success = InvestigationExplanation("ok", "ok", ["ok"], "ok", True)

    def generate(_bundle):
        calls["count"] += 1
        return success

    monkeypatch.setattr(service, "_call_llm", generate)

    assert service.explain(_DummyBundle()) is success
    assert service.explain(_DummyBundle()) is success
    assert calls["count"] == 1
    assert len(explanation_module._EXPLANATION_CACHE) == 1


def test_gemini_native_key_from_settings_is_supported(monkeypatch):
    monkeypatch.setattr(explanation_module.settings, "llm_api_key", "")
    monkeypatch.setattr(explanation_module.settings, "gemini_api_key", "settings-gemini-key")
    monkeypatch.setattr(explanation_module.settings, "openai_api_key", "")

    service = InvestigationExplanationService(provider="gemini")

    assert service.api_key == "settings-gemini-key"
