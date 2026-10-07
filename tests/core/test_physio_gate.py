"""Il filtro puo' saltare una chiamata solo se quella chiamata non fa nulla.

Il rischio non e' che il filtro sia lento: e' che sia **ottimista**. Un salto di
troppo toglie un'azione di sopravvivenza a un agente, e la run resta plausibile
mentre il modello e' cambiato. Il test principale non verifica quindi la
derivazione scritta in ``physio_gate``: la mette alla prova contro la funzione
vera, su una run reale, confrontando ogni verdetto di salto con cio' che
``_physiological_override`` avrebbe effettivamente restituito.

Gli altri test tengono allineati i tre bracci, che devono produrre lo stesso
vettore di booleani perche' sono la stessa condizione scritta tre volte.
"""

import numpy as np
import pytest

from src.core import native, physio_gate
from src.core.arrays import AgentArrays, CellArrays


def _batch(count: int = 6):
    """Un batch sintetico con una cella supportata e una nuda."""
    agents = AgentArrays(count)
    cells = CellArrays(4, 4)
    # Cella (1, 1): abitat + serra + solare + ossigeno = capacita' positiva.
    cells.struct_count[1, 1, physio_gate._S_HABITAT] = 1
    cells.struct_count[1, 1, physio_gate._S_GREENHOUSE] = 1
    cells.struct_count[1, 1, physio_gate._S_SOLAR] = 1
    cells.struct_count[1, 1, physio_gate._S_OXYGEN] = 1
    for row in range(count):
        agents.ids.append(f"a{row}")
        agents.index[f"a{row}"] = row
        agents.alive[row] = True
        agents.x[row] = 1
        agents.y[row] = 1
        agents.hydration[row] = 1.0
        agents.satiety[row] = 1.0
        agents.health[row] = 1.0
        agents.inv[row, physio_gate._R_WATER] = 5.0
        agents.inv[row, physio_gate._R_FOOD] = 5.0
    agents.n = count
    return agents, cells


def test_a_supported_healthy_batch_skips_and_each_guard_denies_the_skip():
    agents, cells = _batch()
    rows = np.arange(6, dtype=np.int64)
    # Ogni riga oltre la prima spegne una guardia diversa: il filtro deve
    # concedere il salto solo alla riga sana.
    agents.hydration[1] = 0.74
    agents.satiety[2] = 0.69
    agents.health[3] = 0.54
    agents.steps_without_water[4] = 2
    agents.inv[5, physio_gate._R_FOOD] = 1.0

    verdicts = physio_gate.skip_override_numpy(agents, cells, rows)

    assert verdicts.tolist() == [True, False, False, False, False, False]


def test_an_unsupported_cell_denies_the_skip():
    agents, cells = _batch(1)
    # Senza serra la capacita' di supporto e' zero, e con essa cade la
    # dimostrazione che le due distanze di supporto valgono zero.
    cells.struct_count[1, 1, physio_gate._S_GREENHOUSE] = 0
    verdicts = physio_gate.skip_override_numpy(
        agents, cells, np.zeros(1, dtype=np.int64)
    )
    assert not bool(verdicts[0])


def test_the_scalar_oracle_and_the_numpy_arm_agree():
    agents, cells = _batch(40)
    rng = np.random.default_rng(20260807)
    agents.hydration[:40] = rng.uniform(0.5, 1.0, 40)
    agents.satiety[:40] = rng.uniform(0.5, 1.0, 40)
    agents.health[:40] = rng.uniform(0.4, 1.0, 40)
    agents.steps_without_water[:40] = rng.integers(0, 4, 40)
    agents.steps_without_food[:40] = rng.integers(0, 8, 40)
    agents.inv[:40, physio_gate._R_WATER] = rng.uniform(0.0, 6.0, 40)
    agents.inv[:40, physio_gate._R_ICE] = rng.uniform(0.0, 2.0, 40)
    agents.inv[:40, physio_gate._R_FOOD] = rng.uniform(0.0, 3.0, 40)
    rows = np.arange(40, dtype=np.int64)

    scalar = physio_gate.skip_override_scalar(agents, cells, rows)
    vectorized = physio_gate.skip_override_numpy(agents, cells, rows)

    # Il campione deve contenere entrambi gli esiti, altrimenti l'accordo e'
    # vacuo: due funzioni che restituiscono sempre False sono d'accordo.
    assert scalar.any() and not scalar.all()
    assert scalar.tolist() == vectorized.tolist()


@pytest.mark.skipif(
    not native.is_available(), reason="kernel nativo non compilato in questo checkout"
)
def test_the_rust_arm_matches_the_other_two():
    agents, cells = _batch(40)
    rng = np.random.default_rng(20260807)
    agents.hydration[:40] = rng.uniform(0.5, 1.0, 40)
    agents.satiety[:40] = rng.uniform(0.5, 1.0, 40)
    agents.health[:40] = rng.uniform(0.4, 1.0, 40)
    agents.steps_without_water[:40] = rng.integers(0, 4, 40)
    agents.steps_without_food[:40] = rng.integers(0, 8, 40)
    agents.inv[:40, physio_gate._R_WATER] = rng.uniform(0.0, 6.0, 40)
    agents.inv[:40, physio_gate._R_ICE] = rng.uniform(0.0, 2.0, 40)
    agents.inv[:40, physio_gate._R_FOOD] = rng.uniform(0.0, 3.0, 40)
    # Righe fuori ordine e non tutte: il lato nativo indicizza da se',
    # e il risultato deve seguire `rows`.
    rows = np.array([7, 0, 39, 12], dtype=np.int64)

    scalar = physio_gate.skip_override_scalar(agents, cells, rows)
    rust = physio_gate.skip_override_rust(agents, cells, rows)

    assert rust.tolist() == scalar.tolist()


@pytest.mark.skipif(
    not native.is_available(), reason="kernel nativo non compilato in questo checkout"
)
def test_the_native_bridge_refuses_a_row_outside_the_columns():
    agents, cells = _batch(2)
    with pytest.raises(ValueError, match="fuori dagli"):
        physio_gate.skip_override_rust(
            agents, cells, np.array([999], dtype=np.int64)
        )


def test_every_skipped_call_would_have_returned_none_on_a_real_run():
    """La prova che conta: il filtro contro la funzione vera, su una run reale.

    Il filtro viene calcolato ma **non** applicato -- restituisce ``None`` al
    chiamante, quindi ogni riga chiama comunque ``_physiological_override``. Il
    test confronta poi, riga per riga, il verdetto con l'esito reale. Un solo
    salto concesso a una riga che invece produceva un'azione fa fallire il test:
    e' la differenza fra un filtro dimostrato e un filtro plausibile.
    """
    import src.agents.preference_agent as preference_agent
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    steps: list[tuple[np.ndarray, list]] = []
    current: dict = {"verdicts": None, "results": []}
    real_gate = physio_gate.skip_override
    real_override = preference_agent._physiological_override

    def recording_gate(agents, cells, rows, backend=None):
        if current["verdicts"] is not None:
            steps.append((current["verdicts"], current["results"]))
        current["verdicts"] = real_gate(agents, cells, rows, backend="numpy")
        current["results"] = []
        return None  # filtro spento: ogni riga chiama, cosi' c'e' cosa confrontare

    def recording_override(agent, world, cell, mask):
        result = real_override(agent, world, cell, mask)
        current["results"].append(result)
        return result

    physio_gate.skip_override = recording_gate
    preference_agent._physiological_override = recording_override
    try:
        runner = AgentCoupledRunner(realistic_config(40, 8, 0))
        runner.run(days=8, output_dir=None)
    finally:
        physio_gate.skip_override = real_gate
        preference_agent._physiological_override = real_override
    if current["verdicts"] is not None:
        steps.append((current["verdicts"], current["results"]))

    assert steps, "la run non ha prodotto nessun passo di decisione"
    skipped = 0
    for index, (verdicts, results) in enumerate(steps):
        assert len(results) == len(verdicts), (
            f"passo {index}: {len(results)} chiamate per {len(verdicts)} verdetti; "
            "l'accoppiamento posizione-riga non regge piu'"
        )
        for position, verdict in enumerate(verdicts):
            if not verdict:
                continue
            skipped += 1
            assert results[position] is None, (
                f"passo {index}, posizione {position}: il filtro concede il salto "
                f"ma `_physiological_override` restituisce {results[position]!r}. "
                "La condizione sufficiente in physio_gate non e' piu' sufficiente."
            )
    # Senza salti concessi il test passerebbe senza verificare nulla.
    assert skipped > 0, "nessun salto concesso: il confronto sarebbe vacuo"
