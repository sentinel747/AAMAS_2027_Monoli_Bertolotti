"""I tre bracci del punteggio devono dare lo stesso numero, bit per bit.

Il rischio non e' che un braccio sia lento: e' che ne diverga uno all'ultimo bit
e la run resti plausibile mentre le decisioni cambiano. Il pericolo concreto e'
noto e misurato: l'originale eleva al quadrato con ``x ** 2`` su un ``float``
Python, cioe' attraverso la ``pow()`` di sistema, e su questa piattaforma
``x ** 2 != x * x`` per **556 valori su un milione** nel dominio vero. Il primo
test qui sotto usa proprio quei valori.

Il test finale non si fida di nessuna derivazione: fa girare i tre bracci dentro
una run reale, sugli stessi argomenti, e li confronta.
"""

import numpy as np
import pytest

from src.agents import pillars
from src.core import decision_scoring, native
from src.core.arrays import AgentArrays, MISSION_SCOUT_OUT
from src.core.views import AgentSideState, AgentView

_ARMS_AVAILABLE = ["numpy", "python"] + (["rust"] if native.is_available() else [])


def _agent(row: int = 0, count: int = 3, **overrides) -> AgentView:
    agents = AgentArrays(count)
    for index in range(count):
        agents.ids.append(f"a{index}")
        agents.index[f"a{index}"] = index
        agents.alive[index] = True
        agents.health[index] = agents.hydration[index] = 1.0
        agents.satiety[index] = agents.morale[index] = 1.0
    agents.n = count
    for name, value in overrides.items():
        getattr(agents, name)[row] = value
    side = AgentSideState()
    return AgentView(agents, row, side)


def _pow_divergent_bases(limit: int = 40) -> list[float]:
    """Valori per cui ``x ** 2`` e ``x * x`` divergono all'ultimo bit."""
    rng = np.random.default_rng(11)
    found: list[float] = []
    while len(found) < limit:
        for value in rng.uniform(0.0, 1.0, 200_000).tolist():
            if value**2 != value * value:
                found.append(value)
                if len(found) >= limit:
                    break
    return found


def test_every_arm_squares_through_pow_and_not_multiplication():
    """Il caso che una moltiplicazione romperebbe, su tutti i bracci disponibili.

    Se un braccio sostituisse ``x ** 2`` con ``x * x`` -- che sembra la stessa
    cosa e in Rust e' anche piu' veloce -- questo test cadrebbe. Senza di esso la
    divergenza comparirebbe come una decisione diversa ogni ~1800 agenti, cioe'
    come un risultato di simulazione leggermente altro.
    """
    bases = _pow_divergent_bases()
    assert bases, "nessun valore divergente trovato: il test perderebbe il suo scopo"
    mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.linspace(0.1, 1.0, pillars.N_ACTIONS)

    for base in bases:
        # `sustenance_base` e' max(1 - idratazione, 1 - sazieta'): fissando
        # l'idratazione a 1 - base la base dell'elevamento e' esattamente `base`.
        agent = _agent(hydration=1.0 - base, satiety=1.0)
        results = {
            arm: decision_scoring.score_decision(
                agent, mask, priority, 0.37, "softmax", True, backend=arm
            )
            for arm in _ARMS_AVAILABLE
        }
        reference_scores, reference_chosen = results["numpy"]
        for arm, (scores, chosen) in results.items():
            assert np.array_equal(scores, reference_scores), (
                f"braccio {arm!r} diverge da `numpy` con base {base!r}: "
                f"{scores.tolist()} contro {reference_scores.tolist()}"
            )
            assert chosen == reference_chosen, f"braccio {arm!r}, base {base!r}"


@pytest.mark.parametrize("sampling", ["softmax", "greedy"])
def test_the_arms_agree_on_adversarial_inputs(sampling):
    mask = np.zeros(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.zeros(pillars.N_ACTIONS, dtype=np.float64)
    cases = [
        ("maschera vuota", mask.copy(), priority.copy()),
        (
            "una sola azione, priorita' nulla",
            np.array([i == 0 for i in range(pillars.N_ACTIONS)], dtype=np.bool_),
            priority.copy(),
        ),
        (
            "priorita' negative ammissibili",
            np.ones(pillars.N_ACTIONS, dtype=np.bool_),
            np.full(pillars.N_ACTIONS, -1.5),
        ),
        (
            "tutto ammissibile, priorita' crescenti",
            np.ones(pillars.N_ACTIONS, dtype=np.bool_),
            np.linspace(0.0, 3.0, pillars.N_ACTIONS),
        ),
    ]
    for label, case_mask, case_priority in cases:
        for u01 in (0.0, 0.5, 0.999999, 1.0):
            agent = _agent(
                hydration=0.31, satiety=0.44, health=0.62, fatigue=0.77,
                stress=0.29, morale=0.13, curiosity=0.91,
                steps_without_water=3, steps_without_food=7,
                mission=MISSION_SCOUT_OUT,
            )
            agent.inventory.construction_material = 4.5
            agent.inventory.minerals = 2.25
            results = {
                arm: decision_scoring.score_decision(
                    agent, case_mask, case_priority, u01, sampling, True, backend=arm
                )
                for arm in _ARMS_AVAILABLE
            }
            reference = results["numpy"]
            for arm, (scores, chosen) in results.items():
                assert np.array_equal(scores, reference[0]), f"{label}, {arm}, u01={u01}"
                assert chosen == reference[1], f"{label}, {arm}, u01={u01}"


def test_survival_disabled_flattens_the_urgencies_on_every_arm():
    mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.linspace(0.2, 1.4, pillars.N_ACTIONS)
    agent = _agent(hydration=0.05, satiety=0.05, health=0.05)
    reference = decision_scoring.score_decision(
        agent, mask, priority, 0.5, "softmax", False, backend="numpy"
    )
    for arm in _ARMS_AVAILABLE:
        scores, chosen = decision_scoring.score_decision(
            agent, mask, priority, 0.5, "softmax", False, backend=arm
        )
        assert np.array_equal(scores, reference[0]), arm
        assert chosen == reference[1], arm


def test_an_unknown_sampling_is_refused_by_every_arm():
    mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.ones(pillars.N_ACTIONS, dtype=np.float64)
    for arm in _ARMS_AVAILABLE:
        with pytest.raises(ValueError, match="sampling"):
            decision_scoring.score_decision(
                _agent(), mask, priority, 0.5, "boltzmann", True, backend=arm
            )


@pytest.mark.skipif(
    not native.is_available(), reason="kernel nativo non compilato in questo checkout"
)
def test_the_rust_arm_falls_back_when_the_agent_has_no_columns():
    """Gli agenti del motore a oggetti non hanno colonne da passare.

    Degradare al braccio Python -- che e' l'equivalente esatto -- e' preferibile
    a fallire su un percorso che i test del motore a oggetti percorrono ancora.
    """

    class _Bare:
        hydration = satiety = health = 1.0
        fatigue = stress_index = 0.0
        morale = curiosity = 1.0
        steps_without_water = steps_without_food = 0
        scout_phase = settle_phase = None
        founder_kit_reserved = False
        pillar_preferences = None
        pillar_skills = None

        class inventory:  # noqa: N801 - facciata minima, non una classe pubblica
            construction_material = 1.0
            minerals = 1.0

    mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)
    priority = np.linspace(0.1, 1.0, pillars.N_ACTIONS)
    native_scores, native_chosen = decision_scoring.score_decision(
        _Bare(), mask, priority, 0.4, "softmax", True, backend="rust"
    )
    python_scores, python_chosen = decision_scoring.score_decision(
        _Bare(), mask, priority, 0.4, "softmax", True, backend="python"
    )
    assert np.array_equal(native_scores, python_scores)
    assert native_chosen == python_chosen


def test_all_arms_agree_across_a_real_run():
    """La prova che conta: gli stessi argomenti veri, i tre bracci confrontati.

    Un test sintetico non produce le combinazioni che una colonia genera davvero
    -- maschere ridotte dai claim, priorita' di cella, orologi di privazione.
    Qui i bracci girano tutti su ogni decisione di una run reale.
    """
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    real = decision_scoring.score_decision
    seen = {"decisions": 0, "mismatches": []}

    def comparing(agent, mask, action_priority, u01, sampling, survival, backend=None):
        results = {
            arm: real(agent, mask, action_priority, u01, sampling, survival, backend=arm)
            for arm in _ARMS_AVAILABLE
        }
        reference = results["numpy"]
        seen["decisions"] += 1
        for arm, (scores, chosen) in results.items():
            if not np.array_equal(scores, reference[0]) or chosen != reference[1]:
                seen["mismatches"].append(
                    f"{arm}: {scores.tolist()} / {chosen} contro "
                    f"{reference[0].tolist()} / {reference[1]}"
                )
        return reference

    decision_scoring.score_decision = comparing
    try:
        AgentCoupledRunner(realistic_config(40, 8, 0)).run(days=8, output_dir=None)
    finally:
        decision_scoring.score_decision = real

    assert seen["decisions"] > 0, "nessuna decisione osservata: la sonda non e' agganciata"
    assert not seen["mismatches"], (
        f"{len(seen['mismatches'])} disaccordi su {seen['decisions']} decisioni:\n  "
        + "\n  ".join(seen["mismatches"][:5])
    )
