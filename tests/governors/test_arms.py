"""I bracci di controllo: la costituzione dello scripted, il caso del random.

Con la policy globale il braccio scripted non legge piu' il quadro: emette
sempre `REFERENCE_RULES`, e l'adattivita' sta nelle condizioni che il kernel
valuta cella per cella. I pericoli sono quindi: una costituzione che esca dai
limiti configurati, un braccio casuale non riproducibile fra processi, e un
prompt LLM che non riporti la stessa costituzione — perche' allora i due bracci
non riceverebbero lo stesso mandato operativo e il confronto misurerebbe anche
la differenza fra un obiettivo scritto e uno da indovinare.
"""

import asyncio

from src.governors.arms import (
    REFERENCE_RULES,
    RandomProposer,
    ScriptedProposer,
    reference_policy,
    reference_rules_text,
)
from src.governors.llm_arm import build_governor_prompt
from src.governors.observation import ColonyPicture
from src.governors.policy import (
    Bounds,
    INDICATORS,
    MAX_RULES,
    PILLAR_BY_NAME,
)

BOUNDS = Bounds(0.25, 4.0)
PICTURE = ColonyPicture(step=41, population=100, n_cells=2)


def _propose(proposer, picture=PICTURE, bounds=BOUNDS):
    return asyncio.run(proposer.propose(picture, bounds))


def test_the_scripted_arm_always_emits_the_reference_policy():
    first = _propose(ScriptedProposer())
    second = _propose(ScriptedProposer(), picture=ColonyPicture(step=99, population=3))
    assert first.policy == second.policy, (
        "la costituzione non guarda il quadro: le sue condizioni si'"
    )
    assert len(first.policy.rules) == len(REFERENCE_RULES)
    assert first.rationale.startswith("politica di riferimento")


def test_the_reference_rules_live_in_the_vocabulary():
    for indicator, op, _threshold, pillar, _weight in REFERENCE_RULES:
        assert indicator in INDICATORS
        assert op in ("<", ">")
        assert pillar in PILLAR_BY_NAME


def test_the_constitution_respects_custom_bounds():
    """I limiti sono configurabili: una costituzione che li scavalcasse darebbe
    allo scripted uno spazio che il braccio LLM non ha."""
    narrow = Bounds(0.5, 1.5)
    policy = reference_policy(narrow)
    for rule in policy.rules:
        for weight in rule.weights.values():
            assert narrow.weight_min <= weight <= narrow.weight_max


def test_the_llm_prompt_quotes_the_same_constitution():
    prompt = build_governor_prompt(PICTURE, BOUNDS)
    assert reference_rules_text() in prompt, (
        "i due bracci devono ricevere lo stesso mandato operativo"
    )


def test_the_random_arm_is_reproducible_from_seed_and_step():
    a = _propose(RandomProposer(seed=7))
    b = _propose(RandomProposer(seed=7))
    assert a.policy == b.policy, "stesso seme, stesso passo: stessa policy"
    c = _propose(RandomProposer(seed=8))
    d = _propose(RandomProposer(seed=7), picture=ColonyPicture(step=42, population=100))
    assert a.policy != c.policy or a.policy != d.policy, (
        "seme o passo diversi devono poter dare policy diverse"
    )


def test_random_policies_stay_inside_the_legal_space():
    for seed in range(10):
        proposal = _propose(RandomProposer(seed=seed))
        rules = proposal.policy.rules
        assert 1 <= len(rules) <= MAX_RULES
        for rule in rules:
            assert rule.condition is not None
            assert rule.condition.indicator in INDICATORS
            assert rule.condition.op in ("<", ">")
            for weight in rule.weights.values():
                assert BOUNDS.weight_min <= weight <= BOUNDS.weight_max


def test_ogni_indicatore_ha_una_banda_per_il_braccio_casuale():
    """Parita' espressiva: il braccio casuale deve poter dire cio' che dice l'LLM.

    Se il vocabolario cresce e la banda no, il braccio casuale non emette mai
    quella condizione, e ogni proprieta' di quella condizione si presenta come
    una proprieta' del modello linguistico. E' il criterio che la tesi enuncia
    come "disparita' di potere espressivo", verificato qui invece che
    diagnosticato dopo la campagna.
    """
    from src.governors.arms import _RANDOM_THRESHOLDS
    from src.governors.policy import INDICATORS

    assert set(_RANDOM_THRESHOLDS) == set(INDICATORS)
