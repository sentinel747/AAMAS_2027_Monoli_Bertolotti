"""Amministratori SemIf bounded e ibridi, uno per distretto non vuoto."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy
from src.governors.policy_candidates import (
    ADMIN_CANDIDATE_PROFILE_VERSION,
    CandidatePolicyFactory,
    is_decisive,
    is_hierarchical,
)
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption


@dataclass(frozen=True)
class AdminSemanticContext:
    district: int
    picture: ColonyPicture
    governor_policy: Policy | None
    colony_indicators: dict
    prompt: str
    #: Morti del distretto dall'ultima tornata, per causa (v4). Vuoto per
    #: default: i profili precedenti non lo leggono.
    deaths: dict = field(default_factory=dict)


def _governor_policy_reach(policy: Policy | None, picture: ColonyPicture) -> list[dict]:
    """Per ogni regola del governo: dove scatta adesso nel distretto (v4)."""
    from src.governors.policy import PILLAR_BY_NAME
    from src.governors.policy_candidates import _v3_reach

    if policy is None:
        return []
    nomi = {indice: nome for nome, indice in PILLAR_BY_NAME.items()}
    out = []
    for rule in policy.rules:
        pesi = " ".join(
            f"{nomi.get(k, k)} x{float(v):g}" for k, v in sorted(dict(rule.weights).items())
        )
        cond = rule.condition
        if cond is None:
            out.append({"rule": f"always -> {pesi}",
                        "fires": f"fires now on all {max(1, picture.n_cells)} cells (unconditional)"})
            continue
        stats = picture.indicators.get(cond.indicator)
        fires = (
            _v3_reach(stats, cond.op, float(cond.value), picture.n_cells)
            if isinstance(stats, dict) and "min" in stats else "unknown"
        )
        out.append({"rule": f"{cond.indicator} {cond.op} {float(cond.value):g} -> {pesi}", "fires": fires})
    return out


def _policy_state(policy: Policy | None) -> list[dict]:
    if policy is None:
        return []
    result = []
    for rule in policy.rules:
        condition = None
        if rule.condition is not None:
            condition = {
                "indicator": rule.condition.indicator,
                "op": rule.condition.op,
                "value": rule.condition.value,
            }
        result.append({"if": condition, "weights": dict(rule.weights)})
    return result


class AmministratoreSemIf:
    def __init__(
        self,
        provider,
        bounds: Bounds,
        *,
        run_id: str,
        district: int,
        deep_reasoning_proposer=None,
        candidate_profile: str = ADMIN_CANDIDATE_PROFILE_VERSION,
    ) -> None:
        self._provider = provider
        self._bounds = bounds
        self._run_id = run_id
        self._district = int(district)
        self._deep_reasoning_proposer = deep_reasoning_proposer
        self._factory = CandidatePolicyFactory(candidate_profile)

    def _state(self, context: AdminSemanticContext) -> dict:
        return {
            "district": self._district,
            "population": context.picture.population,
            "indicators": context.picture.indicators,
            "population_stats": context.picture.population_stats,
            "structures": context.picture.structures,
            "colony_indicators": context.colony_indicators,
            "governor_policy": _policy_state(context.governor_policy),
        }

    async def _propose_hierarchical(self, context: AdminSemanticContext, candidates) -> dict:
        """Due domande strette per distretto: tenere il governo o quale area,
        poi quale candidata. Un secondo livello incerto accetta il governo."""
        profilo = candidates.profile_version
        legge = context.governor_policy
        if legge is None or not legge.rules:
            # Scelta dell'utente (2026-09-23): il governo decide SE e su cosa
            # legiferare; gli amministratori adattano la sua legge, accettandola
            # o riscrivendola. Senza una legge di governo in vigore il distretto
            # non delibera, e nessuna chiamata parte.
            return {
                "raw": {
                    "accept": True,
                    "rationale": "no governor law in force; district follows the governor",
                },
                "provider": "semantic",
                "model": "bounded",
                "latency_s": 0.0,
                "candidate_profile": profilo,
                "semantic_escalated": False,
            }
        state = self._state(context)
        gruppi = candidates.by_pillar()
        opzioni = [
            SemanticOption("accept_governor", "Keep the governor policy for this district"),
            SemanticOption("abstain", "Abstain and leave this district to the governor"),
        ]
        opzioni.extend(
            SemanticOption(
                f"area:{area}",
                f"Rewrite this district to target {area} with one bounded rule: "
                + "; ".join(c.description for c in candidati),
            )
            for area, candidati in gruppi.items()
        )
        if self._deep_reasoning_proposer is not None:
            opzioni.append(SemanticOption(
                "deep_reasoning",
                "Explicitly delegate this district decision to the configured LLM proposer",
            ))
        opzioni.append(SemanticOption("unknown", "Insufficient confidence or evidence"))
        primo = await asyncio.to_thread(self._provider.decide, SemanticDecisionRequest(
            run_id=self._run_id,
            step=context.picture.step,
            actor=f"administrator:{self._district}",
            question_id=f"{profilo}:L1",
            state=state,
            question="Should this district keep the governor policy or target an area of its own?",
            options=tuple(opzioni),
        ))
        decisioni = [primo.to_dict()]

        def metadati(ultima) -> dict:
            return {
                "provider": ultima.runtime or "semantic",
                "model": ultima.model or "bounded",
                "latency_s": sum(float(d.get("latency_ms", 0.0)) for d in decisioni) / 1_000.0,
                "semantic_decision": decisioni[-1],
                "semantic_decisions": list(decisioni),
                "candidate_profile": profilo,
                "semantic_escalated": False,
            }

        scelta = primo.selected_option
        if scelta in {"accept_governor", "abstain"}:
            return {
                "raw": {"accept": True, "rationale": f"SemIf selected {scelta}"},
                **metadati(primo),
            }
        if scelta not in {"unknown", "none_of_the_above"} and not is_decisive(primo):
            return {
                "raw": {
                    "accept": True,
                    "rationale": f"SemIf level 1 not decisive ({scelta}); accept governor",
                },
                **metadati(primo),
            }
        if scelta == "deep_reasoning" and self._deep_reasoning_proposer is not None:
            response = await self._deep_reasoning_proposer.propose_text_async(context.prompt)
            return {**metadati(primo), **response, "semantic_escalated": True}
        area = scelta.split(":", 1)[1] if scelta.startswith("area:") else ""
        if area not in gruppi:
            reason = primo.fallback_reason or f"semantic_selection_{scelta}"
            return {
                "raw": {"accept": True, "rationale": f"SemIf fallback: {reason}; accept governor"},
                **metadati(primo),
            }
        secondo = await asyncio.to_thread(self._provider.decide, SemanticDecisionRequest(
            run_id=self._run_id,
            step=context.picture.step,
            actor=f"administrator:{self._district}",
            question_id=f"{profilo}:L2:{area}",
            state=state,
            question=f"Which bounded rule should this district use to target {area}?",
            options=tuple(
                SemanticOption(f"rewrite_with_candidate:{c.id}", c.description)
                for c in gruppi[area]
            ) + (SemanticOption("unknown", "Insufficient confidence or evidence"),),
        ))
        decisioni.append(secondo.to_dict())
        candidate = None
        if secondo.selected_option.startswith("rewrite_with_candidate:") and is_decisive(secondo):
            candidate = candidates.by_id().get(secondo.selected_option.split(":", 1)[1])
        if candidate is None:
            reason = secondo.fallback_reason or f"semantic_selection_{secondo.selected_option}"
            return {
                "raw": {
                    "accept": True,
                    "rationale": f"SemIf level 2 fallback: {reason}; accept governor",
                },
                **metadati(secondo),
            }
        return {
            "raw": {
                "accept": False,
                "policy": candidate.raw_policy["policy"],
                "rationale": candidate.raw_policy["rationale"],
            },
            **metadati(secondo),
        }

    async def _propose_v3(self, context: AdminSemanticContext, candidates) -> dict:
        """Come v2h (niente delibera senza legge del governo), ma vince la piu'
        probabile, tenere la legge e' esplicito anche da indecisi, e al secondo
        livello c'e' «nessuna di queste»."""
        from src.governors.policy_candidates import AREA_ORDER, is_v4, plurality_choice
        from src.governors.semif_arm import v3_area_description, v3_indicators

        profilo = candidates.profile_version
        legge = context.governor_policy
        base = {
            "provider": "semantic", "model": "bounded", "latency_s": 0.0,
            "candidate_profile": profilo, "semantic_escalated": False,
        }
        if legge is None or not legge.rules:
            return {
                "raw": {"accept": True, "rationale": "no governor law in force; district follows the governor"},
                **base,
            }
        picture = context.picture
        v4 = is_v4(profilo)
        state = {
            "district": self._district,
            "population": picture.population,
            "n_cells": picture.n_cells,
            "deaths_since_last_round": (
                dict(sorted((str(k), int(v)) for k, v in context.deaths.items())) if v4
                else int(sum(int(n) for cause in picture.deaths.values() for n in cause.values()))
            ),
            "indicators": v3_indicators(picture),
            "population_stats": picture.population_stats,
            "governor_policy": _policy_state(legge),
        }
        if v4:
            state["governor_policy_reach"] = _governor_policy_reach(legge, picture)
        gruppi = candidates.by_pillar()
        criterio = (
            "Rewrite only if an indicator in this district shows a problem the governor "
            "policy does not address; otherwise keep it. Rule weights are relative: "
            "raising one activity lowers the share of all the others."
        )
        if v4:
            criterio += (
                " A problem is what kills colonists here (deaths_since_last_round): "
                "starvation means food, dehydration means water (sustenance), hypoxia "
                "means oxygen (life); if nobody dies, keep the governor policy."
            )
        opzioni = [
            SemanticOption("accept_governor", "Keep the governor policy: no local problem a rule would fix"),
            SemanticOption("unknown", "Undecided: keep the governor policy and look again next round"),
        ]
        opzioni.extend(
            SemanticOption(f"area:{area}", "Rewrite: " + v3_area_description(area, gruppi[area]))
            for area in AREA_ORDER if area in gruppi
        )
        primo = await asyncio.to_thread(self._provider.decide, SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor=f"administrator:{self._district}",
            question_id=f"{profilo}:L1",
            state=state,
            question=f"Should this district keep the governor policy or target an area of its own? {criterio}",
            options=tuple(opzioni),
        ))
        decisioni = [primo.to_dict()]

        def metadati(ultima) -> dict:
            return {
                **base,
                "provider": ultima.runtime or "semantic",
                "model": ultima.model or "bounded",
                "latency_s": sum(float(d.get("latency_ms", 0.0)) for d in decisioni) / 1_000.0,
                "semantic_decision": decisioni[-1],
                "semantic_decisions": list(decisioni),
            }

        scelta = plurality_choice(primo)
        area = scelta.split(":", 1)[1] if scelta.startswith("area:") else ""
        if area not in gruppi:
            return {
                "raw": {"accept": True, "rationale": f"SemIf v3 selected {scelta}; accept governor"},
                **metadati(primo),
            }
        secondo = await asyncio.to_thread(self._provider.decide, SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor=f"administrator:{self._district}",
            question_id=f"{profilo}:L2:{area}",
            state=state,
            question=f"Which rule should this district use to target {area}? {criterio}",
            options=tuple(
                SemanticOption(f"rewrite_with_candidate:{c.id}", c.description) for c in gruppi[area]
            ) + (SemanticOption(
                "none_of_the_above", "None of these: no rule here fixes a real problem; keep the governor policy",
            ),),
        ))
        decisioni.append(secondo.to_dict())
        scelta2 = plurality_choice(secondo)
        candidate = None
        if scelta2.startswith("rewrite_with_candidate:"):
            candidate = candidates.by_id().get(scelta2.split(":", 1)[1])
        if candidate is None:
            return {
                "raw": {"accept": True, "rationale": f"SemIf v3 level 2 selected {scelta2}; accept governor"},
                **metadati(secondo),
            }
        return {
            "raw": {
                "accept": False,
                "policy": candidate.raw_policy["policy"],
                "rationale": candidate.raw_policy["rationale"],
            },
            **metadati(secondo),
        }

    async def propose_semantic_async(self, context: AdminSemanticContext) -> dict:
        candidates = self._factory.build(context.picture, self._bounds)
        from src.governors.policy_candidates import is_v3

        if is_v3(candidates.profile_version):
            return await self._propose_v3(context, candidates)
        if is_hierarchical(candidates.profile_version):
            return await self._propose_hierarchical(context, candidates)
        options = [
            SemanticOption("accept_governor", "Keep the governor policy for this district"),
            SemanticOption("abstain", "Abstain and leave this district to the governor"),
        ]
        options.extend(
            SemanticOption(
                f"rewrite_with_candidate:{candidate.id}", candidate.description
            )
            for candidate in candidates.candidates
        )
        if self._deep_reasoning_proposer is not None:
            options.append(SemanticOption(
                "deep_reasoning",
                "Explicitly delegate this district decision to the configured LLM proposer",
            ))
        options.append(SemanticOption("unknown", "Insufficient confidence or evidence"))
        request = SemanticDecisionRequest(
            run_id=self._run_id,
            step=context.picture.step,
            actor=f"administrator:{self._district}",
            question_id=candidates.profile_version,
            state={
                "district": self._district,
                "population": context.picture.population,
                "indicators": context.picture.indicators,
                "population_stats": context.picture.population_stats,
                "structures": context.picture.structures,
                "colony_indicators": context.colony_indicators,
                "governor_policy": _policy_state(context.governor_policy),
            },
            question="Should this district accept or rewrite the governor policy?",
            options=tuple(options),
        )
        decision = await asyncio.to_thread(self._provider.decide, request)
        metadata = {
            "provider": decision.runtime or "semantic",
            "model": decision.model or "bounded",
            "latency_s": decision.latency_ms / 1_000.0,
            "semantic_decision": decision.to_dict(),
            "candidate_profile": candidates.profile_version,
            "semantic_escalated": False,
        }
        selected = decision.selected_option
        if selected in {"accept_governor", "abstain"}:
            return {
                "raw": {"accept": True, "rationale": f"SemIf selected {selected}"},
                **metadata,
            }
        if selected == "deep_reasoning" and self._deep_reasoning_proposer is not None:
            response = await self._deep_reasoning_proposer.propose_text_async(context.prompt)
            # Provider/modello/token/latency del risultato profondo devono
            # restare quelli LLM; la decisione SemIf vive nei campi separati.
            return {**metadata, **response, "semantic_escalated": True}
        if selected.startswith("rewrite_with_candidate:"):
            candidate_id = selected.split(":", 1)[1]
            candidate = candidates.by_id().get(candidate_id)
            if candidate is not None:
                return {
                    "raw": {
                        "accept": False,
                        "policy": candidate.raw_policy["policy"],
                        "rationale": candidate.raw_policy["rationale"],
                    },
                    **metadata,
                }
        reason = decision.fallback_reason or f"semantic_selection_{selected}"
        return {
            "raw": {
                "accept": True,
                "rationale": f"SemIf fallback: {reason}; accept governor",
            },
            **metadata,
        }


class AmministratoreIbrido(AmministratoreSemIf):
    def __init__(
        self,
        provider,
        llm_proposer,
        bounds: Bounds,
        *,
        run_id: str,
        district: int,
        candidate_profile: str = ADMIN_CANDIDATE_PROFILE_VERSION,
    ) -> None:
        if llm_proposer is None:
            raise ValueError("hybrid administrator requires an explicit LLM proposer")
        super().__init__(
            provider,
            bounds,
            run_id=run_id,
            district=district,
            deep_reasoning_proposer=llm_proposer,
            candidate_profile=candidate_profile,
        )
