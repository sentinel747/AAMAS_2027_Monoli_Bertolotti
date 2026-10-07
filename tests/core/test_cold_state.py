"""Le colonne cold devono dire la verita', e coprire cio' che il ciclo legge.

Due rischi distinti, quindi due test distinti:

- le colonne potrebbero non corrispondere ai campi da cui provengono;
- l'elenco dei campi potrebbe restare indietro rispetto a cio' che la fase di
  decisione legge davvero, e allora il lato nativo leggerebbe un valore che non
  esiste. Il secondo test misura le letture invece di fidarsi dell'elenco.
"""

import numpy as np

from src.core import cold_state
from src.core.views import AgentSideState


def _side_for(count: int) -> tuple[dict, list]:
    side, ids = {}, []
    for index in range(count):
        agent_id = f"a{index}"
        ids.append(agent_id)
        side[agent_id] = AgentSideState(
            actions_taken=index * 3,
            founder_kit_reserved=bool(index % 2),
            local_x_m=index * 1.5,
            local_y_m=index * -2.5,
            movement_distance_m_per_step=59_000.0 + index,
        )
    return side, ids


def test_columns_match_their_source_and_follow_the_row_order():
    side, ids = _side_for(5)
    # Ordine sparso e non tutte le righe: il lato nativo indicizza per posizione,
    # quindi l'ordine del risultato deve seguire `rows`, non il dizionario.
    rows = np.array([3, 0, 4], dtype=np.int64)

    columns = cold_state.materialize(side, ids, rows)

    for position, row in enumerate(rows):
        entry = side[ids[int(row)]]
        assert columns["actions_taken"][position] == entry.actions_taken
        assert columns["founder_kit_reserved"][position] == entry.founder_kit_reserved
        assert columns["local_x_m"][position] == entry.local_x_m
        assert columns["local_y_m"][position] == entry.local_y_m
        assert (
            columns["movement_distance_m_per_step"][position]
            == entry.movement_distance_m_per_step
        )


def test_columns_carry_the_dtypes_the_native_side_expects():
    side, ids = _side_for(3)
    columns = cold_state.materialize(side, ids, np.arange(3, dtype=np.int64))

    assert columns["actions_taken"].dtype == np.int64
    assert columns["founder_kit_reserved"].dtype == np.bool_
    for name in ("local_x_m", "local_y_m", "movement_distance_m_per_step"):
        assert columns[name].dtype == np.float64
    assert all(array.flags["C_CONTIGUOUS"] for array in columns.values())


def test_empty_row_selection_produces_empty_columns():
    side, ids = _side_for(2)
    columns = cold_state.materialize(side, ids, np.empty(0, dtype=np.int64))
    assert all(array.size == 0 for array in columns.values())
    assert set(columns) == set(cold_state.COLD_FIELDS)


def test_the_field_list_covers_what_the_decision_phase_actually_reads():
    """Misura invece di fidarsi: se il ciclo comincia a leggere un altro campo
    cold, il lato nativo leggerebbe un valore che nessuno gli ha passato.

    Il proxy avvolge le istanze DOPO la costruzione, perche' sostituire la classe
    prima non intercetta gli oggetti gia' creati -- errore commesso e corretto
    durante la misura originale.
    """
    from collections import Counter

    import src.core.kernel as kernel
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    reads: Counter = Counter()
    recording = {"on": False}

    class _SideProxy:
        __slots__ = ("_wrapped",)

        def __init__(self, wrapped):
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name):
            if recording["on"]:
                reads[name] += 1
            return getattr(object.__getattribute__(self, "_wrapped"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_wrapped"), name, value)

    runner = AgentCoupledRunner(realistic_config(40, 6, 0))
    for agent_id in list(runner.core.side):
        runner.core.side[agent_id] = _SideProxy(runner.core.side[agent_id])

    inner = kernel.decide_batch

    def recording_batch(*args, **kwargs):
        recording["on"] = True
        try:
            return inner(*args, **kwargs)
        finally:
            recording["on"] = False

    kernel.decide_batch = recording_batch
    try:
        runner.run(days=6, output_dir=None)
    finally:
        kernel.decide_batch = inner

    observed = {name for name in reads if not name.startswith("_")}
    missing = observed - set(cold_state.COLD_FIELDS)
    assert not missing, (
        f"la fase di decisione legge campi cold non esposti come colonne: "
        f"{sorted(missing)}. Aggiungerli a COLD_FIELDS e a materialize(), "
        f"altrimenti il lato nativo leggera' un valore che non gli e' stato passato."
    )
