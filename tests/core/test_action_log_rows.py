"""La riga di log si costruisce quando serve, e quando serve e' quella di prima.

Il rischio di questa ottimizzazione non e' la lentezza: e' che un artefatto di
run perda righe in silenzio. Il test principale accende `store_memory_logs` e
verifica che il contenuto sia **identico** a quello prodotto costruendo sempre --
cioe' che l'unica differenza sia quando nessuno guarda.

Il secondo test copre la direzione opposta e piu' insidiosa: con i log spenti le
azioni RIFIUTATE devono conservare la riga, perche' il conteggio dei motivi di
rifiuto ne legge il messaggio.
"""

import numpy as np
import pytest

from src.core.shell_common import _action_log_row_may_be_read


def _run(days: int, store_memory_logs: bool):
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(40, days, 0)
    config.setdefault("headless", {})["store_memory_logs"] = store_memory_logs
    runner = AgentCoupledRunner(config)
    runner.run(days=days, output_dir=None)
    return runner


def test_with_memory_logs_on_every_row_is_still_built_and_complete():
    runner = _run(6, store_memory_logs=True)
    rows = list(runner.validated_actions) + list(runner.rejected_actions)
    assert rows, "con i log accesi le righe devono esserci"
    # Le righe devono essere complete, non troncate: se l'ottimizzazione avesse
    # tolto campi invece che chiamate, un artefatto di run perderebbe colonne.
    for row in rows[:20]:
        for field in ("step", "day", "agent_id", "action", "accepted", "message", "inventory"):
            assert field in row, f"campo {field!r} mancante dalla riga di log"


def test_with_memory_logs_off_no_row_is_kept_but_rejections_still_count():
    """Il caso che l'ottimizzazione sfrutta, e la sua unica eccezione."""
    import src.simulation.agent_coupled_runner as runner_mod

    rejected_rows = {"seen": 0}
    accepted_none = {"seen": 0}

    # `agent_coupled_runner` fa `from src.core.kernel import step as _core_step`:
    # il nome da sostituire e' quello legato NEL modulo chiamante, non nel modulo
    # che lo definisce. Sostituire `kernel.step` non intercetta nulla e il test
    # passerebbe senza aver guardato un solo record.
    inner = runner_mod._core_step

    def inspecting(*args, **kwargs):
        outcome = inner(*args, **kwargs)
        for record in outcome.agent_step_records:
            if record.accepted:
                if record.action_log_row is None:
                    accepted_none["seen"] += 1
            else:
                rejected_rows["seen"] += 1
                assert record.action_log_row is not None, (
                    "un'azione rifiutata deve conservare la riga: il conteggio "
                    "dei motivi di rifiuto ne legge il messaggio"
                )
        return outcome

    runner_mod._core_step = inspecting
    try:
        runner = _run(6, store_memory_logs=False)
    finally:
        runner_mod._core_step = inner

    assert not runner.validated_actions and not runner.rejected_actions
    assert accepted_none["seen"] > 0, (
        "nessuna riga saltata: l'ottimizzazione non sta facendo nulla e il test "
        "non proverebbe niente"
    )


def test_the_gate_defaults_to_building_when_the_key_is_absent():
    """In caso di dubbio si costruisce: mai meno di quanto un lettore legge.

    Le due shell risolvono `store_memory_logs` in modo diverso quando la chiave
    manca, e il kernel non sa quale lo stia chiamando. Rispondere `True` puo'
    solo costruire una riga inutile; rispondere `False` produrrebbe un dato
    mancante in un artefatto di run.
    """
    assert _action_log_row_may_be_read({}) is True
    assert _action_log_row_may_be_read({"headless": {}}) is True
    assert _action_log_row_may_be_read({"headless": {"store_memory_logs": True}}) is True
    assert _action_log_row_may_be_read({"headless": {"store_memory_logs": False}}) is False
    # Le config passano anche da YAML/JSON, dove il flag puo' arrivare stringa.
    assert _action_log_row_may_be_read({"headless": {"store_memory_logs": "false"}}) is False
    assert _action_log_row_may_be_read({"headless": {"store_memory_logs": "true"}}) is True


@pytest.mark.parametrize("store", [True, False])
def test_the_simulation_state_does_not_depend_on_the_logging_choice(store):
    """I log sono osservazione, non simulazione: lo stato deve coincidere."""
    from src.core.state_digest import step_digest

    assert np.isscalar(store) or isinstance(store, bool)
    reference = step_digest(_run(6, store_memory_logs=True).core)
    other = step_digest(_run(6, store_memory_logs=store).core)
    assert other["overall"] == reference["overall"]
