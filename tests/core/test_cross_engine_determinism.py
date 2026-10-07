"""Task 13's decisive test: the GUI shell (`SimulationController`,
src/api/state_store.py, Task 12) and the CLI shell (`AgentCoupledRunner`,
src/simulation/agent_coupled_runner.py, Task 13) now BOTH build their
`CoreState` the same way (`src/core/shell_common.build_core_state`) and
drive every step through the exact same `src.core.kernel.step` call - this
test is what permanently replaces the manual "make sure both engines run the
same logic" discipline the project used to rely on. Same small config, same
seed, N steps on each engine, then a field-by-field comparison of everything
both shells expose: live population, agent positions, `struct_count`
per-cell-per-type totals, and the shared KPI metrics.

Global-random note: `rule_based_agent.py`'s scouting `should_move` roll is
the ONE place per-step decisions consume the stdlib `random` module's GLOBAL
state (not a locally-seeded `random.Random`, see `src/core/kernel.py`'s own
`step()` docstring) - `WorldGenerator`/`spawn_initial_agents`/
`seed_initial_colony_support` all use their OWN locally-seeded
`random.Random(seed)` instances instead (verified by reading each), so
neither engine's construction touches global state. Both engines are
therefore reseeded to the SAME value immediately before their own run so
this test's outcome does not depend on what other tests ran before it in the
same process (global `random` state otherwise carries over between tests).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.api.state_store import SimulationController
from src.simulation.agent_coupled_runner import AgentCoupledRunner


CONFIG = {
    "name": "cross_engine_determinism",
    "seed": 42,
    "days": 30,
    "world": {"width": 20, "height": 20, "map_profile": "balanced"},
    "agents": {"count": 10, "llm_count": 0},
    "simulation": {"days_per_step": 7, "max_days": 1_000_000},
    "population": {"enabled": False},
    "climate": {"enabled": False},
    "environmental_layer": {"enabled": False},
    "extreme_events": {"enabled": False, "chance_per_step": 0},
    "llm": {"max_calls_per_step": 0},
    "colony": {"initial_structures": {"habitat": 2, "greenhouse": 1, "solar_array": 1, "storage_depot": 1}},
}

STEPS = 30
RANDOM_SEED = 20260722

# The shared KPI surface both `_current_metrics` implementations emit -
# includes the exact v1 keys the task brief names by name.
SHARED_KPI_KEYS = (
    "population",
    "survival_rate",
    "average_agent_health",
    "average_habitability",
    "structures_built",
    "crew_stress_index",
    "power_margin",
    "colony_prosperity_index",
    "action_diversity",
)


def _agent_positions(agents: dict) -> list[tuple[str, int, int]]:
    return sorted((agent_id, int(agent.x), int(agent.y)) for agent_id, agent in agents.items())


#: I regimi di decisione da confrontare, e non solo quello storico.
#:
#: **Perche' la parametrizzazione e' stata aggiunta.** Il kernel legge
#: `decision_mode` con default `"tree"`, mentre `ManualConfigOptions` lo mette a
#: `"preferences"`: la config grezza qui sopra non lo specifica, quindi questo
#: test girava sull'albero storico — cioe' su un percorso che NESSUNA run reale
#: esegue, dato che frontend, headless runner e script d'esperimento passano
#: tutti da `build_manual_config`. La garanzia "le due shell fanno la stessa
#: cosa" copriva il motore sbagliato.
DECISION_REGIMES = {
    "tree": {"decision_mode": "tree"},
    "preferences-vectorized": {
        "decision_mode": "preferences",
        "decision_engine": "vectorized",
        "decision_sampling": "softmax",
    },
    "preferences-reference": {
        "decision_mode": "preferences",
        "decision_engine": "reference",
        "decision_sampling": "softmax",
    },
    "preferences-greedy": {
        "decision_mode": "preferences",
        "decision_engine": "vectorized",
        "decision_sampling": "greedy",
    },
}


@pytest.mark.parametrize("regime", sorted(DECISION_REGIMES))
def test_cross_engine_determinism(regime, tmp_path):
    config = dict(CONFIG)
    config["agents"] = {**CONFIG["agents"], **DECISION_REGIMES[regime]}

    random.seed(RANDOM_SEED)
    ctl = SimulationController(dict(config))
    # La shell GUI scrive sempre gli artefatti, e senza questa riga finirebbero
    # in `outputs/runs` — la stessa cartella da cui il pannello e il wizard
    # offrono le run da ripetere, che si riempirebbe di rifiuti dei test.
    ctl.output_dir = tmp_path / "gui"
    ctl.step(STEPS)

    random.seed(RANDOM_SEED)
    runner = AgentCoupledRunner(dict(config))
    # `output_dir=None` tiene gli artefatti in memoria. Senza, ogni esecuzione
    # della suite scrive una run vera in `outputs/runs`, che e' la cartella da
    # cui il pannello e il wizard offrono le run da ripetere: le voci piu'
    # recenti diventerebbero rifiuti dei test invece dei propri esperimenti.
    runner.run(days=STEPS, output_dir=None)

    # Sanity: the scenario actually ran and has a live population left (a
    # comparison between two empty populations would pass trivially without
    # proving anything).
    assert len(ctl.agents) > 0
    assert len(runner.agents) > 0

    # 1. Live population count.
    assert len(ctl.agents) == len(runner.agents)

    # 2. Agent positions (ordered set) - catches any divergence in movement/
    # scouting/settlement decisions between the two shells.
    assert _agent_positions(ctl.agents) == _agent_positions(runner.agents)

    # 3. struct_count totals per cell/type - catches any divergence in
    # structure-building actions or in how `seed_initial_colony_support`
    # seeded the initial colony.
    assert np.array_equal(ctl.core.cells.struct_count, runner.core.cells.struct_count)
    assert ctl.core.cells.struct_count.sum() == runner.core.cells.struct_count.sum() > 0

    # 4. Shared KPI values, within float tolerance (float32 arrays on both
    # sides, built via the identical `CellArrays.from_world`/
    # `AgentArrays.from_agents` conversion - expected to match far tighter
    # than this tolerance, but see test_kernel_step.py's own precedent for
    # why a small tolerance rather than bit-exact equality is the right
    # discipline for this kind of cross-engine numeric comparison).
    ctl_metrics = ctl._current_metrics()
    runner_metrics = runner._current_metrics(len(runner.global_metrics), runner.world.day)
    for key in SHARED_KPI_KEYS:
        assert key in ctl_metrics, key
        assert key in runner_metrics, key
        assert abs(float(ctl_metrics[key]) - float(runner_metrics[key])) < 1e-6, (
            f"{key}: gui={ctl_metrics[key]!r} cli={runner_metrics[key]!r}"
        )


def test_the_isru_conversion_keeps_the_two_shells_in_step(tmp_path):
    """La capacita' aggiunta non deve divergere fra le shell.

    Il ramo ISRU vive in `kernel_biology.update_cells`, che entrambe le shell
    raggiungono attraverso la stessa `kernel.step`; questo test lo verifica
    invece di dedurlo, perche' e' l'unica modifica recente che tocchi il
    bilancio delle risorse di cella e quindi tutto cio' che ne discende.
    """
    config = dict(CONFIG)
    config["agents"] = {**CONFIG["agents"], **DECISION_REGIMES["preferences-vectorized"]}
    config["colony"] = {**CONFIG["colony"], "isru_material_rate": 5.0}

    random.seed(RANDOM_SEED)
    ctl = SimulationController(dict(config))
    # La shell GUI scrive sempre gli artefatti, e senza questa riga finirebbero
    # in `outputs/runs` — la stessa cartella da cui il pannello e il wizard
    # offrono le run da ripetere, che si riempirebbe di rifiuti dei test.
    ctl.output_dir = tmp_path / "gui"
    ctl.step(STEPS)

    random.seed(RANDOM_SEED)
    runner = AgentCoupledRunner(dict(config))
    # `output_dir=None` tiene gli artefatti in memoria. Senza, ogni esecuzione
    # della suite scrive una run vera in `outputs/runs`, che e' la cartella da
    # cui il pannello e il wizard offrono le run da ripetere: le voci piu'
    # recenti diventerebbero rifiuti dei test invece dei propri esperimenti.
    runner.run(days=STEPS, output_dir=None)

    assert len(ctl.agents) == len(runner.agents) > 0
    assert _agent_positions(ctl.agents) == _agent_positions(runner.agents)
    assert np.array_equal(ctl.core.cells.struct_count, runner.core.cells.struct_count)
