"""Costruzione esplicita dei provider semantici, senza default di rete."""

from __future__ import annotations

from collections.abc import Mapping

from .client import FakeSemanticDecisionProvider, RestSemanticDecisionProvider
from .replay import ReplaySemanticDecisionProvider


def build_semantic_provider(config: Mapping, telemetry=None):
    mode = str(config.get("provider", "")).strip().lower()
    if mode == "fake":
        decisions = config.get("decisions") or {}
        if not isinstance(decisions, Mapping):
            raise ValueError("governors.semantic.decisions must be a mapping")
        return FakeSemanticDecisionProvider(decisions)
    if mode == "replay":
        path = str(config.get("replay_from", "") or "").strip()
        if not path:
            raise ValueError("governors.semantic.replay_from is required for replay")
        return ReplaySemanticDecisionProvider.from_jsonl(path, telemetry=telemetry)
    if mode == "rest":
        endpoint = str(config.get("endpoint", "") or "").strip()
        if not endpoint:
            raise ValueError(
                "governors.semantic.endpoint is required for REST; no network "
                "endpoint is selected implicitly"
            )
        return RestSemanticDecisionProvider(
            endpoint,
            api_key_env=str(config.get("api_key_env", "") or ""),
            timeout_seconds=float(config.get("timeout_seconds", 10.0)),
            max_retries=int(config.get("max_retries", 1)),
            circuit_failure_threshold=int(config.get("circuit_failure_threshold", 3)),
            circuit_cooldown_seconds=float(config.get("circuit_cooldown_seconds", 30.0)),
            min_confidence=float(config.get("min_confidence", 0.65)),
            telemetry=telemetry,
            temperature=config.get("temperature"),
        )
    if mode == "typesafe":
        from pathlib import Path

        from .budget import SpendingLedger
        from .typesafe import TypeSafeSemanticDecisionProvider

        budget = float(config.get("budget_usd", 0.0) or 0.0)
        if budget <= 0.0:
            raise ValueError(
                "governors.semantic.budget_usd must be > 0 for typesafe; "
                "no paid call is made without an explicit cap"
            )
        # Assoluto sotto la radice del repository: relativo alla cartella di
        # lavoro, una run lanciata altrove ripartirebbe da un registro vuoto e il
        # tetto cumulativo si azzererebbe in silenzio.
        ledger_path = str(config.get("ledger_path", "") or "").strip() or str(
            Path(__file__).resolve().parents[2]
            / "runs" / "jev_semif_experiments" / "_typesafe_spend.json"
        )
        return TypeSafeSemanticDecisionProvider(
            ledger=SpendingLedger(ledger_path, cap_usd=budget),
            endpoint=str(config.get("endpoint", "") or "https://api.typesafe.ai"),
            api_key_env=str(config.get("api_key_env", "") or "TYPESAFE_AI"),
            model=str(config.get("model", "") or "jev-latest"),
            question_type=str(config.get("question_type", "") or "choice"),
            timeout_seconds=float(config.get("timeout_seconds", 15.0)),
            min_confidence=float(config.get("min_confidence", 0.65)),
        )
    if mode == "laya":
        from .typesafe import LayaSemanticDecisionProvider

        endpoint = str(config.get("endpoint", "") or "").strip()
        if not endpoint:
            raise ValueError(
                "governors.semantic.endpoint is required for laya; no network "
                "endpoint is selected implicitly"
            )
        return LayaSemanticDecisionProvider(
            endpoint=endpoint,
            api_key_env=str(config.get("api_key_env", "") or ""),
            model=str(config.get("model", "") or "laya"),
            question_type=str(config.get("question_type", "") or "choice"),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            min_confidence=float(config.get("min_confidence", 0.65)),
        )
    raise ValueError(
        "governors.semantic.provider must be one of: fake, replay, rest, typesafe, laya"
    )
