"""Artifact riproducibili per le sole run SemIf/JEV sperimentali."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.governors.policy_candidates import (
    ADMIN_CANDIDATE_PROFILE_VERSION,
    CANDIDATE_PROFILE_VERSION,
)

from .schemas import SCHEMA_VERSION
from .telemetry import redact_secrets


SEMANTIC_ARMS = frozenset({"semif", "hybrid"})
SEMANTIC_MANIFEST_VERSION = "1.0"


def semantic_agents_enabled(config: dict) -> bool:
    """Lo strato SemIf dei singoli agenti e' attivo in questa configurazione."""
    section = config.get("semantic_agents")
    return isinstance(section, dict) and (
        str(section.get("mode", "off") or "off").strip().lower() != "off"
    )


def semantic_governance_enabled(config: dict) -> bool:
    if semantic_agents_enabled(config):
        return True
    governors = config.get("governors")
    if not isinstance(governors, dict):
        return False
    governor_arm = str(governors.get("arm", "") or "").strip().lower()
    administrators = governors.get("administrators")
    admin_arm = ""
    if isinstance(administrators, dict) and administrators.get("enabled", False):
        if administrators.get("follow_governor_arm", True):
            admin_arm = governor_arm
        else:
            admin_arm = str(administrators.get("arm", "") or "").strip().lower()
    return governor_arm in SEMANTIC_ARMS or admin_arm in SEMANTIC_ARMS


def _safe_endpoint(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def sanitized_semantic_config(config: dict) -> dict:
    """Configurazione risolta senza credenziali, query URL o userinfo."""
    redacted = redact_secrets(config)

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    _safe_endpoint(str(item))
                    if str(key).lower() == "endpoint"
                    else visit(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(redacted)


def semantic_config_fingerprint(config: dict) -> str:
    governors = sanitized_semantic_config(config).get("governors", {})
    semantic = dict(governors.get("semantic") or {}) if isinstance(governors, dict) else {}
    # Il run_id identifica l'esecuzione, non il trattamento: escluderlo permette
    # al resume di riconoscere la stessa configurazione in una nuova cartella.
    semantic.pop("run_id", None)
    administrators = governors.get("administrators") if isinstance(governors, dict) else None
    administrators = dict(administrators) if isinstance(administrators, dict) else {}
    admin_semantic = administrators.get("semantic")
    if isinstance(admin_semantic, dict):
        admin_semantic = dict(admin_semantic)
        admin_semantic.pop("run_id", None)
        administrators["semantic"] = admin_semantic
    payload = {
        "arm": governors.get("arm", "") if isinstance(governors, dict) else "",
        "semantic": semantic,
        "administrators": administrators,
        "assignments": governors.get("assignments", []) if isinstance(governors, dict) else [],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if isinstance(value, dict):
            rows.append(value)
    return rows


def collect_semantic_decisions(output_dir: str | Path) -> list[dict]:
    root = Path(output_dir)
    decisions: list[dict] = []
    def tutte(record) -> list:
        # Profilo a due livelli: la lista completa; altrimenti la sola decisione.
        if not isinstance(record, dict):
            return []
        lista = record.get("semantic_decisions")
        if isinstance(lista, list) and lista:
            return [d for d in lista if isinstance(d, dict)]
        decision = record.get("semantic_decision")
        return [decision] if isinstance(decision, dict) else []

    for row in _jsonl(root / "governor_decisions.jsonl"):
        for decision in tutte(row.get("governor")):
            decisions.append(redact_secrets(decision))
    for row in _jsonl(root / "administrator_decisions.jsonl"):
        for district in row.get("last_round", []) or []:
            for decision in tutte(district):
                decisions.append(redact_secrets(decision))
    return decisions


def _git_commit(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def governance_summary(output_dir: str | Path) -> dict:
    """Quanto e come ha governato la run: il governo che non agisce mai non si
    puo' studiare, e uno che riafferma la stessa legge non la sta cambiando.

    Governatore: tornate, cambi di legge (una legge non vuota diversa dalla
    precedente), riaffermazioni (la stessa legge di nuovo), attese (nessuna
    nuova legge). Amministratori: tornate di distretto, riscritture, accettazioni
    deliberate, e distretti fermi perche' il governo non aveva una legge.
    """
    root = Path(output_dir)
    rounds = changes = reaffirmations = holds = abrogations = 0
    in_force = None  # regole della legge in vigore, serializzate

    def regole(policy):
        rules = policy.get("rules") if isinstance(policy, dict) else policy
        return json.dumps(rules or [], sort_keys=True, ensure_ascii=False)

    for row in _jsonl(root / "governor_decisions.jsonl"):
        rounds += 1
        governor = row.get("governor") if isinstance(row.get("governor"), dict) else {}
        # Si legge la PROPOSTA: il registro scrive la legge in vigore anche
        # quando il governo lascia com'e' (pilota del 2026-09-23). Registri che
        # non hanno il campo ricadono sulla legge in vigore.
        proposal = governor.get("proposal") if "proposal" in governor else row.get("policy")
        if proposal is None:
            holds += 1
            continue
        corrente = regole(proposal)
        if corrente == "[]":
            abrogations += 1
        elif corrente == in_force:
            reaffirmations += 1
        else:
            changes += 1
        in_force = corrente
    district_rounds = rewrites = accepts = no_law = 0
    for row in _jsonl(root / "administrator_decisions.jsonl"):
        for district in row.get("last_round", []) or []:
            if not isinstance(district, dict):
                continue
            district_rounds += 1
            rationale = str(district.get("rationale", "") or "")
            if rationale.startswith("no governor law in force"):
                no_law += 1
            elif district.get("accepted_government_policy"):
                accepts += 1
            else:
                rewrites += 1
    return {
        "governor": {
            "rounds": rounds, "law_changes": changes,
            "reaffirmations": reaffirmations, "holds": holds,
            "abrogations": abrogations,
        },
        "administrators": {
            "district_rounds": district_rounds, "rewrites": rewrites,
            "accepts": accepts, "no_governor_law": no_law,
        },
    }


def _configured_profiles(semantic: dict) -> dict:
    """Le versioni di profilo davvero in uso: `v1` se la sezione non ne indica."""
    from src.governors.policy_candidates import PROFILE_VERSIONS

    profilo = str((semantic or {}).get("candidate_profile", "v1") or "v1").strip().lower()
    governatore, amministratore = PROFILE_VERSIONS.get(
        profilo, (CANDIDATE_PROFILE_VERSION, ADMIN_CANDIDATE_PROFILE_VERSION)
    )
    return {"governor": governatore, "administrator": amministratore}


def write_semantic_artifacts(
    output_dir: str | Path,
    config: dict,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    """Scrive decisioni, riepilogo e manifest solo per una run semantica."""
    if not semantic_governance_enabled(config):
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    decisions = collect_semantic_decisions(root)
    decision_path = root / "semantic_decisions.jsonl"
    decision_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in decisions
        ),
        encoding="utf-8",
    )
    fallback_reasons: dict[str, int] = {}
    routes: dict[str, int] = {}
    deep = 0
    latency_ms = 0.0
    for row in decisions:
        route = str(row.get("route", "") or "")
        routes[route] = routes.get(route, 0) + 1
        reason = str(row.get("fallback_reason", "") or "")
        if reason:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        deep += int(row.get("selected_option") == "deep_reasoning")
        latency_ms += float(row.get("latency_ms", 0.0) or 0.0)
    telemetry = {
        "schema_version": SCHEMA_VERSION,
        "decisions": len(decisions),
        "bounded_decisions": len(decisions) - deep,
        "deep_escalations": deep,
        "avoided_deep_generations": len(decisions) - deep,
        "routes": dict(sorted(routes.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "semantic_latency_ms_total": round(latency_ms, 6),
        "simulation_elapsed_seconds": (
            None if elapsed_seconds is None else round(float(elapsed_seconds), 6)
        ),
        "governance": governance_summary(root),
    }
    (root / "semantic_telemetry.json").write_text(
        json.dumps(telemetry, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[2]
    safe_config = sanitized_semantic_config(config)
    governors = safe_config.get("governors", {})
    semantic = governors.get("semantic", {}) if isinstance(governors, dict) else {}
    if not semantic and isinstance(governors, dict):
        administrators = governors.get("administrators")
        if isinstance(administrators, dict):
            semantic = administrators.get("semantic", {}) or {}
    agent_layer = {}
    if semantic_agents_enabled(config):
        agent_layer = safe_config.get("semantic_agents", {}) or {}
        if not semantic:
            semantic = agent_layer.get("semantic", {}) or {}
    manifest = {
        "manifest_version": SEMANTIC_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repository),
        "seed": int(config.get("seed", 0)),
        "schema_version": SCHEMA_VERSION,
        "candidate_profiles": _configured_profiles(semantic),
        "semantic_config_hash": semantic_config_fingerprint(config),
        "semantic_provider": str(semantic.get("provider", "") or ""),
        "logical_endpoint": _safe_endpoint(str(semantic.get("endpoint", "") or "")),
        "resolved_config": safe_config,
        "artifacts": {
            "decisions": decision_path.name,
            "telemetry": "semantic_telemetry.json",
            "governor_decisions": "governor_decisions.jsonl",
            "administrator_decisions": "administrator_decisions.jsonl",
            **({
                "agent_decisions": "semantic_agent_decisions.jsonl",
                "agent_telemetry": "semantic_agent_telemetry.json",
            } if agent_layer else {}),
        },
        "agent_layer": agent_layer,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    (root / "semantic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
