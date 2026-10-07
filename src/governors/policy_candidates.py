"""Alternative deterministiche e prevalidate per il governatore SemIf."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy, parse_policy
from src.semantic_governance.schemas import SemanticOption


CANDIDATE_PROFILE_VERSION = "jev-semif-governor-v1"
ADMIN_CANDIDATE_PROFILE_VERSION = "jev-semif-admin-v1"
MAX_POLICY_CANDIDATES = 13

CANDIDATE_PROFILE_VERSION_V2 = "jev-semif-governor-v2"
ADMIN_CANDIDATE_PROFILE_VERSION_V2 = "jev-semif-admin-v2"
#: v2h (2026-09-23): stesse candidate di v2, chieste in due domande strette
#: (area, poi candidata) invece che in una da 14 opzioni, dove la confidenza
#: restava a 0,25-0,35 e la soglia 0,65 scartava ogni decisione.
CANDIDATE_PROFILE_VERSION_V2H = "jev-semif-governor-v2h"
ADMIN_CANDIDATE_PROFILE_VERSION_V2H = "jev-semif-admin-v2h"
#: v3 (2026-09-23, scelte dell'utente dopo la campagna v2h): vince l'opzione
#: piu' probabile, senza soglia ne' distacco, perche' aspettare e non
#: intervenire sono opzioni esplicite a ogni livello; descrizioni oneste e
#: corte (condizione + su quante celle scatta adesso); niente `power_coverage`,
#: che il quadro legge dopo il consumo e vale 0 anche a corrente coperta.
CANDIDATE_PROFILE_VERSION_V3 = "jev-semif-governor-v3"
ADMIN_CANDIDATE_PROFILE_VERSION_V3 = "jev-semif-admin-v3"
#: v4 (2026-09-24): v3 piu' il metro del «problema». Nel pilota v3 gli
#: amministratori Jev riscrivevano il 99% delle volte con «life x3 sull'ossigeno»
#: in un mondo dove nessuno muore di ipossia: v4 da' i morti per causa (colonia
#: e distretto), dice all'amministratore dove scatta la legge del governo e lega
#: il criterio alle cause di morte. Regola, opzioni e candidate come v3.
CANDIDATE_PROFILE_VERSION_V4 = "jev-semif-governor-v4"
ADMIN_CANDIDATE_PROFILE_VERSION_V4 = "jev-semif-admin-v4"
PROFILE_VERSIONS: dict[str, tuple[str, str]] = {
    "v1": (CANDIDATE_PROFILE_VERSION, ADMIN_CANDIDATE_PROFILE_VERSION),
    "v2": (CANDIDATE_PROFILE_VERSION_V2, ADMIN_CANDIDATE_PROFILE_VERSION_V2),
    "v2h": (CANDIDATE_PROFILE_VERSION_V2H, ADMIN_CANDIDATE_PROFILE_VERSION_V2H),
    "v3": (CANDIDATE_PROFILE_VERSION_V3, ADMIN_CANDIDATE_PROFILE_VERSION_V3),
    "v4": (CANDIDATE_PROFILE_VERSION_V4, ADMIN_CANDIDATE_PROFILE_VERSION_V4),
}
_V4_PROFILES = frozenset({CANDIDATE_PROFILE_VERSION_V4, ADMIN_CANDIDATE_PROFILE_VERSION_V4})
_V3_PROFILES = frozenset({CANDIDATE_PROFILE_VERSION_V3, ADMIN_CANDIDATE_PROFILE_VERSION_V3}) | _V4_PROFILES
_V2_PROFILES = frozenset({
    CANDIDATE_PROFILE_VERSION_V2, ADMIN_CANDIDATE_PROFILE_VERSION_V2,
    CANDIDATE_PROFILE_VERSION_V2H, ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
}) | _V3_PROFILES
_HIERARCHICAL_PROFILES = frozenset({
    CANDIDATE_PROFILE_VERSION_V2H, ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
}) | _V3_PROFILES
#: Indicatori che v3 non offre: `power_coverage` e' l'energia AVANZATA dopo il
#: consumo e i prelievi dei coloni, divisa per il carico (vedi
#: `apply._power_coverage` e `kernel_biology`), quindi ~0 anche quando gli
#: impianti sono alimentati. Resta nel vocabolario della tesi, intatto.
_V3_EXCLUDED_INDICATORS = frozenset({"power_coverage"})


def is_v3(profile_version: str) -> bool:
    """Il profilo decide a maggioranza relativa con opzioni di attesa esplicite."""
    return profile_version in _V3_PROFILES


def is_v4(profile_version: str) -> bool:
    """v3 con morti per causa e portata della legge del governo nel distretto."""
    return profile_version in _V4_PROFILES


def plurality_choice(decision) -> str:
    """Regola di v3: vince l'opzione piu' probabile, qualunque sia il valore.

    Nessuna soglia e nessun distacco (scelta dell'utente, 2026-09-23): il
    modello ha sempre davanti anche «aspetta» e «non intervenire», quindi una
    politica vince solo se batte pure quelle. A parita' vale la scelta del
    provider; senza probabilita' vale la scelta del provider.
    """
    probabilities = dict(decision.probabilities or {})
    selected = decision.selected_option
    if not probabilities:
        return selected
    massimo = max(float(v) for v in probabilities.values())
    if selected in probabilities and float(probabilities[selected]) >= massimo:
        return selected
    return next(k for k, v in probabilities.items() if float(v) >= massimo)
#: Ordine delle aree al livello 1: quello dei pilastri pesabili della policy.
AREA_ORDER = ("sustenance", "resources", "build", "life", "explore")


def is_hierarchical(profile_version: str) -> bool:
    """Il profilo si chiede in due livelli (area, poi candidata)."""
    return profile_version in _HIERARCHICAL_PROFILES


#: Distacco minimo della prima scelta sulla seconda perche' v2h agisca.
DECISION_MARGIN = 0.10


def is_decisive(decision, *, margin: float = DECISION_MARGIN) -> bool:
    """Regola di decisione del profilo v2h, fissata prima della campagna.

    Un governo non deve emanare politiche a ogni tornata: agisce quando la
    preferenza e' netta, lascia com'e' quando e' divisa. Tre condizioni:

    1. la prima scelta non e' un'astensione (`unknown`/`none_of_the_above`);
    2. vale almeno il doppio del caso, `2/n`, con tetto a 1/2 (con due sole
       opzioni il doppio del caso sarebbe la certezza);
    3. supera la seconda di almeno `margin`.

    Sostituisce, per v2h, la soglia unica 0,65 sull'entropia normalizzata, che
    bocciava ogni caso con due risposte ugualmente fondate (misura dal vivo del
    2026-09-23: prima scelta 0,41 contro 0,40, confidenza 0,34).
    """
    return decisive_probabilities(
        decision.probabilities, decision.selected_option, margin=margin
    )


def decisive_probabilities(probabilities, selected: str, *, margin: float = DECISION_MARGIN) -> bool:
    """La regola di `is_decisive` sui soli numeri (usata anche in calibrazione)."""
    from src.semantic_governance.schemas import FALLBACK_OPTIONS

    if selected in FALLBACK_OPTIONS:
        return False
    probabilities = dict(probabilities)
    n = len(probabilities)
    if n < 2 or selected not in probabilities:
        return False
    primo = float(probabilities[selected])
    secondo = max(
        (float(v) for k, v in probabilities.items() if k != selected), default=0.0
    )
    return primo >= min(2.0 / n, 0.5) and primo - secondo >= margin
#: Margine con cui v2 porta la soglia oltre un valore comune a tutte le celle,
#: cosi' che la regola copra l'intera colonia invece di non scattare mai.
#: Misura del 2026-09-22: con v1 e una sola cella nessuna candidata poteva
#: scattare, e il modello sceglieva (correttamente) `keep_previous`.
_V2_UNIFORM_MARGIN = 0.05


def _v2_threshold(stats: dict, op: str) -> tuple[float, str]:
    """Soglia e ambito di una candidata v2.

    Con celle diverse resta la media (si colpiscono le celle sotto o sopra la
    media, come in v1). Con celle tutte uguali la media non separerebbe nulla:
    il parser ammette solo `<` e `>`, quindi la soglia si sposta appena oltre il
    valore comune e la regola copre tutta la colonia.
    """
    mean = float(stats["mean"])
    if float(stats.get("std", 0.0)) > 0.0:
        side = "below" if op == "<" else "above"
        return mean, f"cells {side} the colony average {mean:g}"
    step = max(abs(mean) * _V2_UNIFORM_MARGIN, 1e-6)
    threshold = mean + step if op == "<" else mean - step
    return threshold, f"every settled cell (current value {mean:g})"


def _v3_reach(stats: dict, op: str, threshold: float, n_cells: int) -> str:
    """Su quante celle la condizione scatta adesso: nessuna, una parte, tutte.

    Esatto dai soli aggregati: con `<`, tutte se il massimo e' sotto la soglia,
    nessuna se il minimo non lo e', altrimenti una parte (e viceversa con `>`).
    """
    minimo, massimo = float(stats["min"]), float(stats["max"])
    n = max(1, int(n_cells))
    if op == "<":
        tutte, nessuna = massimo < threshold, minimo >= threshold
    else:
        tutte, nessuna = minimo > threshold, massimo <= threshold
    if tutte:
        return f"fires now on all {n} cells (a constant)"
    if nessuna:
        return f"fires now on none of {n} cells"
    return f"fires now on some of {n} cells"


def _v3_description(pillar: str, weight: float, indicator: str, op: str,
                    threshold: float, reach: str) -> str:
    """Leva e condizione in testa, poi la portata: Laya tronca a ~22 token."""
    return f"{pillar} x{weight:g} where {indicator} {op} {threshold:.4g} | {reach}"

# Una leva sola per alternativa mantiene leggibile l'attribuzione causale.
# La soglia viene dal quadro; peso e direzione appartengono al profilo versionato.
_PROFILE: tuple[tuple[str, str, str, float], ...] = (
    ("food_per_occupant", "<", "sustenance", 3.0),
    ("water_per_occupant", "<", "sustenance", 3.0),
    ("oxygen_per_occupant", "<", "life", 3.0),
    ("ice_per_occupant", "<", "resources", 2.0),
    ("material_per_occupant", "<", "build", 2.0),
    ("minerals_per_occupant", "<", "resources", 2.0),
    ("power_coverage", "<", "build", 3.0),
    ("structure_integrity", "<", "build", 2.0),
    ("occupants", ">", "explore", 2.0),
)


@dataclass(frozen=True)
class PolicyCandidate:
    id: str
    description: str
    raw_policy: dict[str, Any]
    policy: Policy


@dataclass(frozen=True)
class CandidatePolicySet:
    profile_version: str
    candidates: tuple[PolicyCandidate, ...]

    def by_id(self) -> dict[str, PolicyCandidate]:
        return {candidate.id: candidate for candidate in self.candidates}

    def by_pillar(self) -> dict[str, tuple[PolicyCandidate, ...]]:
        """Candidate per pilastro pesato, nell'ordine di `AREA_ORDER`.

        Ogni candidata ha una sola regola con un solo peso (una leva per
        alternativa), quindi il pilastro e' univoco. Le aree senza candidate
        non compaiono.
        """
        gruppi: dict[str, list[PolicyCandidate]] = {}
        for candidate in self.candidates:
            pesi = candidate.raw_policy["policy"][0]["weights"]
            pilastro = next(iter(pesi))
            gruppi.setdefault(str(pilastro), []).append(candidate)
        return {
            area: tuple(gruppi[area]) for area in AREA_ORDER if area in gruppi
        }

    def semantic_options(self, *, include_deep_reasoning: bool = False) -> tuple[SemanticOption, ...]:
        options = [
            SemanticOption("keep_previous", "Keep the policy already in force"),
            SemanticOption("no_intervention", "Adopt an empty policy"),
        ]
        options.extend(
            SemanticOption(f"select_candidate:{candidate.id}", candidate.description)
            for candidate in self.candidates
        )
        if include_deep_reasoning:
            options.append(SemanticOption(
                "deep_reasoning",
                "Explicitly delegate this decision to the configured LLM proposer",
            ))
        options.append(SemanticOption("unknown", "Insufficient confidence or evidence"))
        return tuple(options)


class CandidatePolicyFactory:
    """Genera un insieme finito usando soltanto statistiche del quadro."""

    def __init__(self, profile_version: str = CANDIDATE_PROFILE_VERSION) -> None:
        if not profile_version.strip():
            raise ValueError("candidate profile version must not be empty")
        self.profile_version = profile_version

    def build(self, picture: ColonyPicture, bounds: Bounds) -> CandidatePolicySet:
        candidates: list[PolicyCandidate] = []
        v3 = self.profile_version in _V3_PROFILES
        for indicator, op, pillar, requested_weight in _PROFILE:
            if v3 and indicator in _V3_EXCLUDED_INDICATORS:
                continue
            stats = picture.indicators.get(indicator)
            if not isinstance(stats, dict) or "mean" not in stats:
                continue
            peso = min(max(requested_weight, bounds.weight_min), bounds.weight_max)
            if v3:
                threshold, scope = _v2_threshold(stats, op)
                rationale = (
                    f"profilo {self.profile_version}: {indicator} {op} "
                    f"{threshold:g} ({scope}), priorita {pillar}"
                )
                description = _v3_description(
                    pillar, peso, indicator, op, threshold,
                    _v3_reach(stats, op, threshold, picture.n_cells),
                )
            elif self.profile_version in _V2_PROFILES:
                threshold, scope = _v2_threshold(stats, op)
                rationale = (
                    f"profilo {self.profile_version}: {indicator} {op} "
                    f"{threshold:g} ({scope}), priorita {pillar}"
                )
                description = f"Weight {pillar} by {peso:g} in {scope} for {indicator}"
            else:
                threshold = float(stats["mean"])
                rationale = (
                    f"profilo {self.profile_version}: {indicator} {op} "
                    f"media {threshold:g}, priorita {pillar}"
                )
                description = (
                    f"When {indicator} {op} {threshold:g}, weight {pillar} "
                    f"by {peso:g}"
                )
            raw = {
                "policy": [{
                    "if": {"indicator": indicator, "op": op, "value": threshold},
                    "weights": {pillar: requested_weight},
                }],
                "rationale": rationale,
            }
            policy, drops = parse_policy(raw, bounds)
            if policy is None or drops.dropped_rules or drops.unknown_indicators \
                    or drops.unknown_pillars:
                raise ValueError(
                    f"candidate profile {self.profile_version} generated an invalid "
                    f"policy for {indicator}"
                )
            candidates.append(PolicyCandidate(
                id=f"mean_{indicator}",
                description=description,
                raw_policy=raw,
                policy=policy,
            ))
        if self.profile_version in _V2_PROFILES:
            # Import locale: `arms` e' piu' in alto nella catena d'import.
            from src.governors.arms import REFERENCE_RULES

            for indicator, op, threshold, pillar, weight in REFERENCE_RULES:
                raw = {
                    "policy": [{
                        "if": {"indicator": indicator, "op": op, "value": threshold},
                        "weights": {pillar: weight},
                    }],
                    "rationale": (
                        f"profilo {self.profile_version}: costituzione di riferimento"
                    ),
                }
                policy, drops = parse_policy(raw, bounds)
                if policy is None or drops.dropped_rules:
                    raise ValueError(f"reference rule {indicator} is invalid")
                peso = min(max(weight, bounds.weight_min), bounds.weight_max)
                stats = picture.indicators.get(indicator)
                if v3 and isinstance(stats, dict) and "min" in stats:
                    descrizione = _v3_description(
                        pillar, peso, indicator, op, threshold,
                        _v3_reach(stats, op, threshold, picture.n_cells),
                    )
                else:
                    descrizione = (
                        f"Reference constitution: when {indicator} {op} {threshold:g}, "
                        f"weight {pillar} by {peso:g}"
                    )
                candidates.append(PolicyCandidate(
                    id=f"reference_{indicator}",
                    description=descrizione,
                    raw_policy=raw,
                    policy=policy,
                ))
        if len(candidates) > MAX_POLICY_CANDIDATES:
            raise ValueError("candidate profile exceeds the bounded option budget")
        return CandidatePolicySet(self.profile_version, tuple(candidates))
