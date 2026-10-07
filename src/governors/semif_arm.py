"""Arm governatore SemIf bounded; nessuna escalation LLM."""

from __future__ import annotations

from src.governors.arms import GovernorProposal
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy, parse_policy
from src.governors.policy_candidates import (
    AREA_ORDER,
    CandidatePolicyFactory,
    is_decisive,
    is_hierarchical,
    is_v3,
    is_v4,
    plurality_choice,
)
from src.semantic_governance.client import SemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption

#: Il criterio che v3 mette nella domanda (scelta dell'utente, 2026-09-23): nella
#: ricostruzione di jev_s4 il governo legiferava su una colonia in salute
#: perche' nessuno gli diceva quando NON intervenire.
#: Va DOPO la domanda: Laya tronca l'intestazione allo spazio lasciato dalle
#: opzioni, e cio' che si perde deve essere il criterio, non la domanda.
V3_CRITERION = (
    "Intervene only if an indicator shows a problem the rule would fix; otherwise "
    "wait or choose no intervention. Rule weights are relative: raising one "
    "activity lowers the share of all the others."
)
#: v4: il metro del «problema» sono le cause di morte. Nel mondo della campagna
#: si muore solo di fame e di sete; dirlo con i numeri evita regole su
#: indicatori che non uccidono nessuno (pilota v3: «life x3 sull'ossigeno»).
V4_PROBLEM = (
    "A problem is what kills colonists (deaths_since_last_round): starvation means "
    "food, dehydration means water (sustenance), hypoxia means oxygen (life)."
)
V3_WAIT = "Wait: undecided or no change needed now; keep the policy in force, look again next round"
V3_NO_INTERVENTION = "No intervention: no indicator shows a problem a rule would fix; run with no policy"
V3_NONE_OF_THESE = "None of these: no rule here fixes a real problem; keep the policy in force"


def v3_indicators(picture: ColonyPicture) -> dict:
    """Indicatori compatti per v3: un valore con una cella, media/min/max con piu'.

    Niente deviazione standard e niente indicatori esclusi da v3: lo stato deve
    stare nei 512 token di Laya senza perdere cio' che serve a decidere.
    """
    from src.governors.policy_candidates import _V3_EXCLUDED_INDICATORS

    out = {}
    for nome, stats in picture.indicators.items():
        if nome in _V3_EXCLUDED_INDICATORS or not isinstance(stats, dict):
            continue
        if picture.n_cells <= 1 or float(stats.get("std", 0.0)) == 0.0:
            out[nome] = stats.get("mean")
        else:
            out[nome] = {k: stats.get(k) for k in ("mean", "min", "max")}
    return out


def deaths_by_cause(deaths: dict) -> dict:
    """{(y, x): {causa: n}} -> {causa: n}, in ordine alfabetico (hash stabile)."""
    totale: dict[str, int] = {}
    for cause in deaths.values():
        for causa, n in cause.items():
            totale[str(causa)] = totale.get(str(causa), 0) + int(n)
    return dict(sorted(totale.items()))


def v3_area_description(area: str, candidati) -> str:
    """Corta, con l'area in testa: il dettaglio sta nella domanda di livello 2."""
    indicatori = ", ".join(
        dict.fromkeys(c.raw_policy["policy"][0]["if"]["indicator"] for c in candidati)
    )
    return f"Act on {area}: rules on {indicatori}"


class SemifGovernorProposer:
    def __init__(
        self,
        provider: SemanticDecisionProvider,
        *,
        run_id: str,
        candidate_factory: CandidatePolicyFactory | None = None,
        deep_reasoning_proposer=None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        self._provider = provider
        self._run_id = run_id
        self._factory = candidate_factory or CandidatePolicyFactory()
        self._deep_reasoning_proposer = deep_reasoning_proposer

    def _state(self, picture: ColonyPicture, profile_version: str) -> dict:
        return {
            "profile_version": profile_version,
            "population": picture.population,
            "n_cells": picture.n_cells,
            "indicators": picture.indicators,
            "population_stats": picture.population_stats,
            "structures": picture.structures,
        }

    async def _propose_hierarchical(
        self, picture: ColonyPicture, bounds: Bounds, candidate_set
    ) -> GovernorProposal:
        """Due domande strette: quale area, poi quale candidata nell'area.

        Le candidate sono quelle di v2; cambia soltanto come si chiedono. Un
        secondo livello incerto o fallito conserva la policy in vigore: il
        governo non cambia legge su una scelta che non sa motivare.
        """
        profilo = candidate_set.profile_version
        state = self._state(picture, profilo)
        gruppi = candidate_set.by_pillar()
        opzioni = [
            SemanticOption("keep_previous", "Keep the policy already in force"),
            SemanticOption("no_intervention", "Adopt an empty policy"),
        ]
        opzioni.extend(
            SemanticOption(
                f"area:{area}",
                f"Target {area} with one bounded rule: "
                + "; ".join(c.description for c in candidati),
            )
            for area, candidati in gruppi.items()
        )
        if self._deep_reasoning_proposer is not None:
            opzioni.append(SemanticOption(
                "deep_reasoning",
                "Explicitly delegate this decision to the configured LLM proposer",
            ))
        opzioni.append(SemanticOption("unknown", "Insufficient confidence or evidence"))
        primo = self._provider.decide(SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor="governor",
            question_id=f"{profilo}:L1",
            state=state,
            question="Which area, if any, should the colony-wide policy target this round?",
            options=tuple(opzioni),
        ))
        decisioni = [primo.to_dict()]

        def comune(ultima) -> dict:
            return {
                "provider": ultima.runtime or "semantic",
                "model": ultima.model or "bounded",
                "latency_s": sum(float(d.get("latency_ms", 0.0)) for d in decisioni) / 1_000.0,
                "semantic_decision": decisioni[-1],
                "semantic_decisions": list(decisioni),
                "candidate_profile": profilo,
            }

        scelta = primo.selected_option
        if scelta == "keep_previous":
            return GovernorProposal(
                policy=None, rationale="SemIf selected keep_previous", **comune(primo)
            )
        if not is_decisive(primo):
            # Preferenza divisa (o astensione): il governo lascia com'e'.
            return GovernorProposal(
                policy=None,
                rationale=f"SemIf level 1 not decisive ({scelta}); keep previous policy",
                **comune(primo),
            )
        if scelta == "no_intervention":
            return GovernorProposal(
                policy=Policy((), "SemIf selected no_intervention"),
                rationale="SemIf selected no_intervention",
                **comune(primo),
            )
        if scelta == "deep_reasoning" and self._deep_reasoning_proposer is not None:
            proposal = await self._deep_reasoning_proposer.propose(picture, bounds)
            proposal.semantic_decision = decisioni[-1]
            proposal.semantic_decisions = list(decisioni)
            proposal.candidate_profile = profilo
            proposal.rationale = (
                "SemIf selected explicit deep_reasoning; " + str(proposal.rationale or "")
            ).rstrip()
            return proposal
        if not scelta.startswith("area:") or scelta.split(":", 1)[1] not in gruppi:
            return self._fallback(
                comune(primo), primo.fallback_reason or f"semantic_selection_{scelta}"
            )
        area = scelta.split(":", 1)[1]
        secondo = self._provider.decide(SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor="governor",
            question_id=f"{profilo}:L2:{area}",
            state=state,
            question=f"Which bounded rule should the colony-wide policy use to target {area}?",
            options=tuple(
                SemanticOption(f"select_candidate:{c.id}", c.description)
                for c in gruppi[area]
            ) + (SemanticOption("unknown", "Insufficient confidence or evidence"),),
        ))
        decisioni.append(secondo.to_dict())
        candidate = None
        if secondo.selected_option.startswith("select_candidate:") and is_decisive(secondo):
            candidate = candidate_set.by_id().get(secondo.selected_option.split(":", 1)[1])
        elif secondo.selected_option.startswith("select_candidate:"):
            return self._fallback(
                comune(secondo), f"level 2 not decisive ({secondo.selected_option})"
            )
        if candidate is None:
            motivo = secondo.fallback_reason or f"semantic_selection_{secondo.selected_option}"
            return self._fallback(comune(secondo), f"level 2 {motivo}")
        policy, drops = parse_policy(candidate.raw_policy, bounds)
        if policy is None:
            return self._fallback(comune(secondo), "level 2 selected_candidate_invalid")
        return GovernorProposal(
            policy=policy,
            rationale=policy.rationale,
            drops=drops,
            raw_policy=candidate.raw_policy,
            **comune(secondo),
        )

    def _state_v3(self, picture: ColonyPicture, profile_version: str) -> dict:
        """Stato compatto e con l'esito: morti dall'ultima tornata e legge in vigore.

        v4 da' i morti per causa (il metro del «problema»), v3 il solo totale.
        """
        if is_v4(profile_version):
            morti = deaths_by_cause(picture.deaths)
        else:
            morti = int(sum(
                int(n) for cause in picture.deaths.values() for n in cause.values()
            ))
        return {
            "profile_version": profile_version,
            "population": picture.population,
            "n_cells": picture.n_cells,
            "deaths_since_last_round": morti,
            "policy_in_force": [
                {"rule": str(regola), "cell_steps": int(n)} for regola, n in picture.rule_hits
            ],
            "indicators": v3_indicators(picture),
            "population_stats": picture.population_stats,
        }

    async def _propose_v3(
        self, picture: ColonyPicture, bounds: Bounds, candidate_set
    ) -> GovernorProposal:
        """Due livelli come v2h, ma vince la piu' probabile e l'attesa e' esplicita."""
        profilo = candidate_set.profile_version
        state = self._state_v3(picture, profilo)
        criterio = f"{V3_CRITERION} {V4_PROBLEM}" if is_v4(profilo) else V3_CRITERION
        gruppi = candidate_set.by_pillar()
        opzioni = [
            SemanticOption("unknown", V3_WAIT),
            SemanticOption("no_intervention", V3_NO_INTERVENTION),
        ]
        opzioni.extend(
            SemanticOption(f"area:{area}", v3_area_description(area, gruppi[area]))
            for area in AREA_ORDER if area in gruppi
        )
        primo = self._provider.decide(SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor="governor",
            question_id=f"{profilo}:L1",
            state=state,
            question=f"Which area, if any, should the colony-wide policy target this round? {criterio}",
            options=tuple(opzioni),
        ))
        decisioni = [primo.to_dict()]

        def comune(ultima) -> dict:
            return {
                "provider": ultima.runtime or "semantic",
                "model": ultima.model or "bounded",
                "latency_s": sum(float(d.get("latency_ms", 0.0)) for d in decisioni) / 1_000.0,
                "semantic_decision": decisioni[-1],
                "semantic_decisions": list(decisioni),
                "candidate_profile": profilo,
            }

        scelta = plurality_choice(primo)
        if scelta in {"unknown", "none_of_the_above"}:
            return GovernorProposal(
                policy=None,
                rationale=f"SemIf v3 waits ({primo.fallback_reason or 'most probable: wait'}); keep previous policy",
                **comune(primo),
            )
        if scelta == "no_intervention":
            return GovernorProposal(
                policy=Policy((), "SemIf selected no_intervention"),
                rationale="SemIf selected no_intervention",
                **comune(primo),
            )
        area = scelta.split(":", 1)[1] if scelta.startswith("area:") else ""
        if area not in gruppi:
            return self._fallback(comune(primo), f"semantic_selection_{scelta}")
        secondo = self._provider.decide(SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor="governor",
            question_id=f"{profilo}:L2:{area}",
            state=state,
            question=(
                f"Which rule should the colony-wide policy use to target {area}? {criterio}"
            ),
            options=tuple(
                SemanticOption(f"select_candidate:{c.id}", c.description) for c in gruppi[area]
            ) + (SemanticOption("none_of_the_above", V3_NONE_OF_THESE),),
        ))
        decisioni.append(secondo.to_dict())
        scelta2 = plurality_choice(secondo)
        if scelta2 in {"none_of_the_above", "unknown"}:
            return GovernorProposal(
                policy=None,
                rationale=f"SemIf v3 level 2 chose none of these ({secondo.fallback_reason or area}); keep previous policy",
                **comune(secondo),
            )
        candidate = None
        if scelta2.startswith("select_candidate:"):
            candidate = candidate_set.by_id().get(scelta2.split(":", 1)[1])
        if candidate is None:
            return self._fallback(comune(secondo), f"level 2 semantic_selection_{scelta2}")
        policy, drops = parse_policy(candidate.raw_policy, bounds)
        if policy is None:
            return self._fallback(comune(secondo), "level 2 selected_candidate_invalid")
        return GovernorProposal(
            policy=policy,
            rationale=policy.rationale,
            drops=drops,
            raw_policy=candidate.raw_policy,
            **comune(secondo),
        )

    async def propose(self, picture: ColonyPicture, bounds: Bounds) -> GovernorProposal:
        candidate_set = self._factory.build(picture, bounds)
        if is_v3(candidate_set.profile_version):
            return await self._propose_v3(picture, bounds, candidate_set)
        if is_hierarchical(candidate_set.profile_version):
            return await self._propose_hierarchical(picture, bounds, candidate_set)
        request = SemanticDecisionRequest(
            run_id=self._run_id,
            step=picture.step,
            actor="governor",
            question_id=candidate_set.profile_version,
            state={
                "profile_version": candidate_set.profile_version,
                "population": picture.population,
                "n_cells": picture.n_cells,
                "indicators": picture.indicators,
                "population_stats": picture.population_stats,
                "structures": picture.structures,
            },
            question="Which bounded governor policy should be applied?",
            options=candidate_set.semantic_options(
                include_deep_reasoning=self._deep_reasoning_proposer is not None
            ),
        )
        decision = self._provider.decide(request)
        decision_payload = decision.to_dict()
        common = {
            "provider": decision.runtime or "semantic",
            "model": decision.model or "bounded",
            "latency_s": decision.latency_ms / 1_000.0,
            "semantic_decision": decision_payload,
            "candidate_profile": candidate_set.profile_version,
        }

        selected = decision.selected_option
        if selected == "keep_previous":
            return GovernorProposal(
                policy=None,
                rationale="SemIf selected keep_previous",
                **common,
            )
        if selected == "no_intervention":
            return GovernorProposal(
                policy=Policy((), "SemIf selected no_intervention"),
                rationale="SemIf selected no_intervention",
                **common,
            )
        if selected == "deep_reasoning" and self._deep_reasoning_proposer is not None:
            proposal = await self._deep_reasoning_proposer.propose(picture, bounds)
            proposal.semantic_decision = decision_payload
            proposal.candidate_profile = candidate_set.profile_version
            proposal.rationale = (
                "SemIf selected explicit deep_reasoning; " + str(proposal.rationale or "")
            ).rstrip()
            return proposal
        if selected.startswith("select_candidate:"):
            candidate_id = selected.split(":", 1)[1]
            candidate = candidate_set.by_id().get(candidate_id)
            if candidate is None:
                return self._fallback(common, "selected_candidate_not_available")
            # Il parser corrente resta l'autorita' finale anche se la candidate
            # era gia' stata prevalidata dalla factory.
            policy, drops = parse_policy(candidate.raw_policy, bounds)
            if policy is None:
                return self._fallback(common, "selected_candidate_invalid")
            return GovernorProposal(
                policy=policy,
                rationale=policy.rationale,
                drops=drops,
                raw_policy=candidate.raw_policy,
                **common,
            )
        return self._fallback(
            common,
            decision.fallback_reason or f"semantic_selection_{selected}",
        )

    @staticmethod
    def _fallback(common: dict, reason: str) -> GovernorProposal:
        return GovernorProposal(
            policy=None,
            rationale=f"SemIf fallback: {reason}; keep previous policy",
            **common,
        )
