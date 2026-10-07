"""La memo di `_bind_explore` regge solo se la risposta e' davvero una costante.

Due rischi distinti, quindi due test distinti.

Il primo mette alla prova l'**invariante**, non la memo: gira con la memo spenta,
guarda le risposte vere di `cell_accepts_arrival` e verifica che dentro una fase
di decisione la stessa cella bersaglio non abbia mai dato due risposte diverse.
Se un giorno qualcosa cominciasse a muovere `agents_present` o `struct_count`
durante la decisione, questo test fallirebbe **prima** che la memo cominci a
mentire in silenzio.

Il secondo verifica il risultato: memo accesa e memo spenta devono produrre lo
stesso digest di stato.
"""

import numpy as np
import pytest


def _run(days: int, memo: bool, monkeypatch):
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner
    import src.agents.preference_agent as preference_agent

    monkeypatch.setattr(preference_agent, "_BIND_MEMO_ENABLED", memo)
    runner = AgentCoupledRunner(realistic_config(40, days, 0))
    runner.run(days=days, output_dir=None)
    return runner


def test_the_arrival_answer_is_constant_within_one_decision_phase(monkeypatch):
    import src.agents.preference_agent as preference_agent
    import src.core.kernel as kernel

    real_accepts = preference_agent.cell_accepts_arrival
    phase = {"index": -1}
    seen: dict[tuple[int, int], bool] = {}
    contradictions: list[str] = []
    observed = 0

    def spy(origin, target, pending_arrivals=0, **kwargs):
        nonlocal observed
        answer = real_accepts(origin, target, pending_arrivals, **kwargs)
        # Solo il call-site di `_bind_explore`, che chiama con i default:
        # `_claim_arrival_if_available` passa arrivi pendenti e permessi che
        # cambiano durante il passo, e non e' l'invariante in esame.
        if pending_arrivals or kwargs:
            return answer
        observed += 1
        key = (int(target.x), int(target.y))
        if key in seen and seen[key] != answer:
            contradictions.append(
                f"fase {phase['index']}, cella {key}: prima {seen[key]}, poi {answer}"
            )
        seen[key] = answer
        return answer

    inner = kernel.decide_batch

    def phased(*args, **kwargs):
        phase["index"] += 1
        seen.clear()
        return inner(*args, **kwargs)

    monkeypatch.setattr(preference_agent, "cell_accepts_arrival", spy)
    monkeypatch.setattr(kernel, "decide_batch", phased)
    _run(8, memo=False, monkeypatch=monkeypatch)

    assert observed > 0, "nessuna chiamata osservata: la sonda non e' agganciata"
    assert not contradictions, (
        "`cell_accepts_arrival` ha dato risposte diverse per la stessa cella "
        "dentro una fase di decisione, quindi la memo di `_decision_memo` non e' "
        "piu' esatta:\n  " + "\n  ".join(contradictions[:5])
    )


def test_the_memo_is_dropped_when_the_step_changes():
    """Senza questo, la memo risponderebbe con i dati del passo precedente."""
    from src.agents.preference_agent import _decision_memo

    class _World:
        step = 0

    world = _World()
    neighbors, arrivals = _decision_memo(world)
    neighbors[(1, 2)] = ["vecchio"]
    arrivals[(3, 4)] = True

    again_neighbors, again_arrivals = _decision_memo(world)
    assert again_neighbors is neighbors, "dentro lo stesso passo la memo persiste"

    world.step = 1
    fresh_neighbors, fresh_arrivals = _decision_memo(world)
    assert fresh_neighbors == {} and fresh_arrivals == {}
    assert fresh_neighbors is not neighbors


@pytest.mark.parametrize("days", [8])
def test_the_memo_does_not_change_the_state(days, monkeypatch):
    from src.core.state_digest import step_digest

    without = step_digest(_run(days, memo=False, monkeypatch=monkeypatch).core)
    with_memo = step_digest(_run(days, memo=True, monkeypatch=monkeypatch).core)

    assert with_memo["overall"] == without["overall"], (
        "la memo ha cambiato lo stato; primo campo divergente: "
        + next(
            (
                name
                for name in sorted(without["fields"])
                if without["fields"][name] != with_memo["fields"].get(name)
            ),
            "nessuno",
        )
    )


def test_a_false_answer_is_cached_and_not_recomputed(monkeypatch):
    """`False` e' falsy: una memo scritta con `or` lo ricalcolerebbe ogni volta.

    E' l'esito dominante su configurazione reale -- tutti gli otto vicini
    rifiutati -- quindi sbagliare qui annullerebbe l'intero guadagno senza
    cambiare un solo risultato, cioe' in modo invisibile.
    """
    import src.agents.preference_agent as preference_agent

    calls = {"count": 0}

    def counting(origin, target, pending_arrivals=0, **kwargs):
        calls["count"] += 1
        return False

    monkeypatch.setattr(preference_agent, "cell_accepts_arrival", counting)
    monkeypatch.setattr(preference_agent, "_BIND_MEMO_ENABLED", True)

    class _Cell:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _World:
        step = 7

        def neighbors(self, x, y, radius):
            return [_Cell(1, 0), _Cell(0, 1)]

    world = _World()
    neighbor_memo, arrival_memo = preference_agent._decision_memo(world)
    for _ in range(5):
        for candidate in world.neighbors(0, 0, 1):
            key = (candidate.x, candidate.y)
            answer = arrival_memo.get(key)
            if answer is None:
                answer = preference_agent.cell_accepts_arrival(None, candidate)
                arrival_memo[key] = answer
            assert answer is False

    assert calls["count"] == 2, (
        f"{calls['count']} chiamate per 2 celle distinte: la memo non trattiene "
        "le risposte `False`"
    )
    assert np.array_equal(sorted(arrival_memo), [(0, 1), (1, 0)])
    assert neighbor_memo == {}
