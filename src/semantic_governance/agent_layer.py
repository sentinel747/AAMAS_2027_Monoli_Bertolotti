"""Strato SemIf dei singoli agenti: chiede, traduce in fattori, deposita.

Vive nella shell. Il kernel non lo conosce: riceve soltanto i dizionari di
fattori su `CoreState`, esattamente come riceve la policy del governatore.
Le decisioni vengono scritte in streaming, una riga per decisione, perche' a
trecento agenti il volume e' di ordini di grandezza sopra quello del governo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import agent_questions as aq
from .batch import decide_many
from .factory import build_semantic_provider
from .schemas import MAX_OPTIONS, MAX_OPTIONS_TYPESAFE

_MODES = ("all", "hard")


@dataclass(frozen=True)
class AgentLayerSettings:
    mode: str
    cadence_steps: int
    levels: int
    workers: int
    hard_margin: float
    semantic: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "AgentLayerSettings | None":
        section = config.get("semantic_agents")
        if not isinstance(section, dict):
            return None
        mode = str(section.get("mode", "off") or "off").strip().lower()
        if mode == "off":
            return None
        if mode not in _MODES:
            raise ValueError(f"semantic_agents.mode must be off, all or hard, got {mode!r}")
        levels = int(section.get("levels", 1))
        if levels not in (1, 2):
            raise ValueError("semantic_agents.levels must be 1 or 2")
        cadence = int(section.get("cadence_steps", 4))
        if cadence < 1:
            raise ValueError("semantic_agents.cadence_steps must be >= 1")
        semantic = section.get("semantic")
        if not isinstance(semantic, dict) or not semantic.get("provider"):
            raise ValueError("semantic_agents.semantic.provider is required")
        return cls(
            mode=mode,
            cadence_steps=cadence,
            levels=levels,
            workers=max(1, int(section.get("workers", 64))),
            hard_margin=float(section.get("hard_margin", 0.15)),
            # Nessuna soglia a livello di agente (spec, misura dello spike del
            # 2026-09-22): la distribuzione intera diventa il fattore e
            # l'incertezza si attenua da se'. Una soglia ereditata dal governo
            # (0,65) scartava 249 decisioni su 250 sulla farm.
            semantic={**dict(semantic), "min_confidence": 0.0},
        )


class SemanticAgentLayer:
    def __init__(self, settings: AgentLayerSettings, *, run_id: str, provider=None) -> None:
        self.settings = settings
        self.run_id = str(run_id)
        self.provider = provider if provider is not None else build_semantic_provider(settings.semantic)
        self._max_options = (
            MAX_OPTIONS_TYPESAFE
            if str(settings.semantic.get("provider", "")).lower() == "typesafe"
            else MAX_OPTIONS
        )
        self._out: Path | None = None
        self._stream = None
        self._pillar: dict = {}
        self._action: dict = {}
        self.telemetry = {
            "profile": aq.AGENT_PROFILE_VERSION,
            "mode": settings.mode,
            "levels": settings.levels,
            "cadence_steps": settings.cadence_steps,
            "hard_margin": settings.hard_margin,
            "boundaries": 0,
            "decisions": 0,
            "level2_decisions": 0,
            "fallbacks": 0,
            "fallback_reasons": {},
            "semantic_seconds": 0.0,
        }

    def attach_output(self, path: Path) -> None:
        self._out = Path(path)
        self._out.mkdir(parents=True, exist_ok=True)
        # Svuotato all'avvio come gli altri registri della run: una run rieseguita
        # nella stessa cartella non deve accodare le decisioni di quella che sostituisce.
        self._stream = (self._out / "semantic_agent_decisions.jsonl").open("w", encoding="utf-8")

    def _is_boundary(self, step: int) -> bool:
        return (int(step) - 1) % self.settings.cadence_steps == 0

    def _write(self, decisions) -> None:
        for decision in decisions:
            self.telemetry["decisions"] += 1
            if decision.route == "fallback":
                self.telemetry["fallbacks"] += 1
                reasons = self.telemetry["fallback_reasons"]
                key = decision.fallback_reason or "unspecified"
                reasons[key] = reasons.get(key, 0) + 1
            if self._stream is not None:
                self._stream.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
        if self._stream is not None:
            self._stream.flush()

    def advance(self, step: int, core) -> None:
        if self.settings.mode == "hard" and core.pillar_margin_out is None:
            core.pillar_margin_out = {}
        if core.semantic_counters is None:
            core.semantic_counters = {}
        agents = core.agents
        alive = agents.alive_rows()
        alive_ids = {agents.ids[int(r)] for r in alive}
        # i morti escono dai dizionari; fuori confine i fattori restano quelli
        self._pillar = {k: v for k, v in self._pillar.items() if k in alive_ids}
        self._action = {k: v for k, v in self._action.items() if k in alive_ids}
        if self._is_boundary(step):
            self._ask(int(step), core, alive)
        core.semantic_pillar_factors = self._pillar
        core.semantic_action_factors = self._action if self.settings.levels == 2 else None

    def _ask(self, step: int, core, alive) -> None:
        agents, cells = core.agents, core.cells
        rows = list(alive)
        if self.settings.mode == "hard":
            margins = core.pillar_margin_out or {}
            rows = [r for r in rows
                    if margins.get(agents.ids[int(r)], 1.0) < self.settings.hard_margin]
            core.pillar_margin_out = {}
            # Chi non e' piu' ambiguo torna rule-based: l'ablation misura
            # soltanto l'effetto sui casi ambigui di QUESTO confine.
            chiesti = {agents.ids[int(r)] for r in rows}
            self._pillar = {k: v for k, v in self._pillar.items() if k in chiesti}
            self._action = {k: v for k, v in self._action.items() if k in chiesti}
        # Un agente richiesto a questo confine non conserva il livello 2 del
        # confine precedente: se il nuovo pilastro non lo ammette, torna neutro.
        for r in rows:
            self._action.pop(agents.ids[int(r)], None)
        self.telemetry["boundaries"] += 1
        if not rows:
            return
        colony = aq.colony_summary(agents, alive)
        records = {int(r): aq.agent_record(agents, cells, int(r)) for r in rows}
        requests = [
            aq.level1_request(self.run_id, step, agents.ids[int(r)], colony,
                              records[int(r)], self._max_options)
            for r in rows
        ]
        started = time.perf_counter()
        decisions = decide_many(self.provider, requests, max_workers=self.settings.workers)
        self._write(decisions)
        winners: dict[int, str] = {}
        for r, decision in zip(rows, decisions):
            agent_id = agents.ids[int(r)]
            self._pillar[agent_id] = aq.pillar_factors(decision)
            if decision.route != "fallback":
                winners[int(r)] = decision.selected_option
        if self.settings.levels == 2:
            eligible = [r for r in rows if winners.get(int(r)) in aq.LEVEL2_PILLARS]
            level2 = [
                aq.level2_request(self.run_id, step, agents.ids[int(r)], winners[int(r)],
                                  colony, records[int(r)], self._max_options)
                for r in eligible
            ]
            answers = decide_many(self.provider, level2, max_workers=self.settings.workers)
            self._write(answers)
            self.telemetry["level2_decisions"] += len(answers)
            for r, decision in zip(eligible, answers):
                self._action[agents.ids[int(r)]] = aq.action_factors(decision, winners[int(r)])
        self.telemetry["semantic_seconds"] += time.perf_counter() - started

    def close(self) -> dict:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        summary = dict(self.telemetry)
        summary["semantic_seconds"] = round(summary["semantic_seconds"], 6)
        if self._out is not None:
            (self._out / "semantic_agent_telemetry.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return summary
