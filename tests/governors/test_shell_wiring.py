"""La shell interroga il governatore, e senza governatore non cambia nulla."""

import contextlib
import pytest

#: Le quattro voci non numeriche che lo strato planetario mette DAVVERO fra le
#: metriche di colonia: `unlocked_techs` arriva da `PlanetaryCoupler.
#: get_active_bonuses`, le tre stringhe da `MarsClimate.to_metrics`. Sono qui,
#: letterali, invece di accendere `environmental_layer`+`climate` nei test
#: perche' quella strada costa il runtime Fortran del MCD e una run quattro
#: volte piu' lenta, per provare esattamente la stessa cosa: `_current_metrics`
#: puo' restituire valori che non sono numeri.
_PLANETARY_NON_NUMERIC = {
    "unlocked_techs": ["Advanced Materials"],
    "climate_source": "mcd_call_mcd",
    "climate_scenario": "climatology",
    "climate_data_path": "data\\mcd_runtime\\MCD_6.1\\data",
}


def _gui_config(**governors) -> dict:
    """Config minima per la shell GUI, senza strato planetario ne' eventi."""
    return {
        "name": "gui_governors",
        "seed": 0,
        "days": 6,
        "world": {"width": 16, "height": 16, "map_profile": "balanced"},
        "agents": {"count": 12, "llm_count": 0},
        "simulation": {"days_per_step": 7, "max_days": 100_000},
        "population": {"enabled": False},
        "climate": {"enabled": False},
        "environmental_layer": {"enabled": False},
        "extreme_events": {"enabled": False, "chance_per_step": 0},
        "llm": {"max_calls_per_step": 0},
        "governors": dict(governors) if governors else {},
    }


def test_without_governors_the_state_is_the_same_as_before():
    """Il cancello della parita', in forma di test rapido.

    La verifica completa e' l'harness su 120 passi; questa e' la rete che
    scatta durante lo sviluppo. `count: 0` e' l'interruttore spento dell'era
    del consiglio e deve continuare a significare "spento".
    """
    from scripts.parity_harness import realistic_config
    from src.core.state_digest import step_digest
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    runner = AgentCoupledRunner(realistic_config(40, 6, 0))
    runner.run(days=6, output_dir=None)
    without = step_digest(runner.core)["overall"]

    config = realistic_config(40, 6, 0)
    config["governors"] = {"count": 0}
    other = AgentCoupledRunner(config)
    other.run(days=6, output_dir=None)
    assert step_digest(other.core)["overall"] == without


@contextlib.contextmanager
def _costituzione_di_prova():
    """Una regola incondizionata: scatta su ogni cella occupata, sempre.

    Serve ai test di cablaggio, che devono poter distinguere "la leva non e'
    attaccata" da "la costituzione in vigore non scatta in questo scenario".
    """
    from src.governors.policy import Condition, PILLAR_BY_NAME, Policy, Rule
    import src.governors.arms as arms

    originale = arms.reference_policy
    # Condizione vera su ogni cella occupata, non `None`: il testo leggibile
    # della regola comincia allora con "se ", come per ogni regola reale, e i
    # contatori restano confrontabili con quelli di una run vera.
    arms.reference_policy = lambda bounds: Policy(
        rules=(
            Rule(
                condition=Condition("occupants", ">", 0.0),
                weights={PILLAR_BY_NAME["explore"]: 3.0},
            ),
        ),
        rationale="politica di riferimento: di prova, se occupants > 0 -> explore x3",
    )
    try:
        yield
    finally:
        arms.reference_policy = originale


def test_an_active_governor_changes_the_run_and_writes_its_record(tmp_path):
    """Il test che dice se la leva e' attaccata.

    **Le due run devono salvare entrambe.** Due run IDENTICHE, entrambe senza
    governatore, danno digest diversi con e senza cartella di output — salvare
    gli artefatti materializza voci che `state_digest` include. Due cartelle
    diverse, stesso trattamento, e la sola differenza che resta e' la policy.

    **La policy e' scritta QUI, non presa da `REFERENCE_RULES`.** Fino al
    2026-08-25 il test usava la costituzione in vigore, e la taglia della
    popolazione era scelta perche' quella costituzione scattasse. Riscegliendo
    la costituzione sui dati il test e' diventato rosso — giustamente: la nuova
    non scatta in otto passi su una colonia appena fondata. Ma cio' che questo
    test verifica e' il CABLAGGIO (governatore -> policy -> la run cambia), non
    il merito della costituzione: legarlo a quale politica sia in vigore lo
    rende fragile a ogni ritaratura, e — peggio — potrebbe renderlo verde a
    leva staccata se un domani la costituzione tornasse a scattare per caso.
    Una regola incondizionata scritta qui scatta sempre per costruzione.
    """
    from scripts.parity_harness import realistic_config
    from src.core.state_digest import step_digest
    from src.governors.record import load_records
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    baseline = AgentCoupledRunner(realistic_config(200, 8, 0))
    baseline.run(days=8, output_dir=tmp_path / "baseline")
    plain = step_digest(baseline.core)["overall"]

    config = realistic_config(200, 8, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2}
    governed = AgentCoupledRunner(config)
    with _costituzione_di_prova():
        governed.run(days=8, output_dir=tmp_path / "governed")

    assert step_digest(governed.core)["overall"] != plain, (
        "un governatore attivo deve cambiare la run: se non la cambia, la leva "
        "non e' attaccata e ogni risultato nullo dell'esperimento sarebbe ambiguo"
    )
    rows = load_records(tmp_path / "governed" / "governor_decisions.jsonl")
    assert rows, "un governatore attivo con cartella di output lascia un registro"
    assert rows[0]["governor"]["rationale"].startswith("politica di riferimento")
    assert rows[0]["policy"]["rules"], "la costituzione ha regole, e vanno registrate"


def test_a_governed_run_writes_the_rule_hit_counters(tmp_path):
    """L'ombra fra regole diventa un artefatto: `governor_policy_hits.json`.

    Con prima-regola-vince una regola larga scritta per prima affama le
    successive, e prima di questo file lo si scopriva solo strumentando a
    mano. Le chiavi sono il testo delle regole della costituzione piu' l'else,
    i valori (cella, passo) catturate: ordinato per conteggio, la prima riga
    e' la politica che ha governato davvero.

    La policy e' quella di prova, per la stessa ragione del test qui sopra:
    cio' che si verifica e' che i contatori vengano scritti e abbiano il testo
    leggibile delle regole per chiave, non che la costituzione in vigore scatti
    in questo scenario.
    """
    import json

    from scripts.parity_harness import realistic_config
    from src.governors.apply import ELSE_KEY
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(200, 8, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2}
    with _costituzione_di_prova():
        AgentCoupledRunner(config).run(days=8, output_dir=tmp_path)

    path = tmp_path / "governor_policy_hits.json"
    assert path.exists(), "una run governata con cartella deve lasciare i contatori"
    hits = json.loads(path.read_text(encoding="utf-8"))
    assert hits, "otto passi governati non possono aver contato niente"
    assert all(isinstance(v, int) and v > 0 for v in hits.values())
    regole = [k for k in hits if k != ELSE_KEY]
    assert any(k.startswith("se ") for k in regole), (
        "le chiavi devono essere il testo leggibile delle regole"
    )


def test_the_picture_carries_the_metrics_of_the_previous_step(tmp_path, monkeypatch):
    """Il quadro che il governatore osserva viene DAVVERO dalla shell.

    `_last_metrics` e' l'unico ponte fra le metriche di colonia e il
    governatore: se restasse vuoto — mai popolato, popolato dopo l'uso,
    azzerato a ogni passo — il braccio LLM scriverebbe policy su un quadro
    senza metriche, in silenzio. Lo scripted non le legge piu' (la costituzione
    e' fissa), quindi il ponte va sorvegliato direttamente: si intercetta
    `build_picture` e si guarda che cosa riceve.

    Il ritardo di un passo e' voluto: al passo 1 le metriche non esistono
    ancora, dal secondo confine devono esserci.
    """
    from scripts.parity_harness import realistic_config
    import src.simulation.agent_coupled_runner as runner_module
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    seen: list[dict] = []
    original = runner_module.build_picture

    def spying_build_picture(step, metrics, cells, agents, rows, rule_hits=(), **kwargs):
        seen.append(dict(metrics or {}))
        return original(step, metrics, cells, agents, rows, rule_hits=rule_hits, **kwargs)

    monkeypatch.setattr(runner_module, "build_picture", spying_build_picture)

    config = realistic_config(20, 6, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2}
    AgentCoupledRunner(config).run(days=6, output_dir=None)

    assert seen, "la shell non ha mai costruito un quadro"
    assert seen[0] == {}, "al passo 1 le metriche del passo precedente non esistono"
    assert any("food_margin" in metrics for metrics in seen[1:]), (
        "le metriche di colonia non arrivano mai al quadro: il ponte "
        "`_last_metrics` e' rotto"
    )


def test_the_picture_is_built_only_at_tick_boundaries(monkeypatch):
    """Il cancello delle prestazioni del quadro, in forma strutturale.

    Fra un confine e l'altro `advance` non guarderebbe nemmeno il quadro, e
    costruirlo comunque costava lo 0,58%% del passo (1,29 ms su 300 agenti) per
    buttarne via diciannove su venti a cadenza 20. La garanzia deve valere per
    costruzione: si conta quante volte la shell chiama `build_picture`, che a
    cadenza 2 su 6 passi sono i tre confini 1, 3 e 5 — non sei.
    """
    from scripts.parity_harness import realistic_config
    import src.simulation.agent_coupled_runner as runner_module
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    calls = []
    original = runner_module.build_picture

    def counting_build_picture(step, metrics, cells, agents, rows, rule_hits=(), **kwargs):
        calls.append(int(step))
        return original(step, metrics, cells, agents, rows, rule_hits=rule_hits, **kwargs)

    monkeypatch.setattr(runner_module, "build_picture", counting_build_picture)

    config = realistic_config(20, 6, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2}
    AgentCoupledRunner(config).run(days=6, output_dir=None)

    assert calls == [1, 3, 5], (
        f"il quadro va costruito ai soli confini di tick, non a ogni passo: {calls}"
    )


def test_a_non_numeric_metric_does_not_kill_the_governed_step(tmp_path, monkeypatch):
    """Il quadro riceve numeri, qualunque cosa la shell tenga fra le metriche.

    Con lo strato planetario acceso le metriche contengono una lista e tre
    stringhe: passare il dizionario grezzo faceva uscire un `TypeError` da
    dentro il passo. Il filtro dev'essere l'esatto inverso di cio' che rompe, e
    non deve svuotare il quadro: `food_margin` deve continuare ad arrivare.
    """
    from scripts.parity_harness import realistic_config
    import src.simulation.agent_coupled_runner as runner_module
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(20, 6, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2}
    runner = AgentCoupledRunner(config)

    numeric_only = runner._current_metrics

    def with_planetary_strings(step, day):
        return dict(numeric_only(step, day), **_PLANETARY_NON_NUMERIC)

    monkeypatch.setattr(runner, "_current_metrics", with_planetary_strings)

    seen: list[dict] = []
    original = runner_module.build_picture

    def spying_build_picture(step, metrics, cells, agents, rows, rule_hits=(), **kwargs):
        seen.append(dict(metrics or {}))
        return original(step, metrics, cells, agents, rows, rule_hits=rule_hits, **kwargs)

    monkeypatch.setattr(runner_module, "build_picture", spying_build_picture)
    runner.run(days=6, output_dir=tmp_path)

    with_metrics = [metrics for metrics in seen if metrics]
    assert with_metrics, "il filtro ha svuotato il quadro invece di ripulirlo"
    assert all("unlocked_techs" not in metrics for metrics in seen)
    assert any("food_margin" in metrics for metrics in with_metrics)


def test_the_gui_shell_also_keeps_non_numeric_metrics_out_of_the_picture(
    tmp_path, monkeypatch
):
    """Lo stesso pericolo nella shell GUI, che ha il proprio loop e il proprio
    `_current_metrics`: correggerne una sola lascerebbe l'altra a morire al
    secondo passo, ed e' quella che un utente lancia dall'interfaccia."""
    from src.api.state_store import SimulationController

    monkeypatch.chdir(tmp_path)
    controller = SimulationController(
        _gui_config(arm="scripted", cadence_steps=2)
    )
    numeric_only = controller._current_metrics

    def with_planetary_strings(step=None, day=None):
        return dict(numeric_only(step, day), **_PLANETARY_NON_NUMERIC)

    monkeypatch.setattr(controller, "_current_metrics", with_planetary_strings)
    try:
        controller.step(5)
        assert controller.step_index == 5, (
            "la run si e' fermata prima dei cinque passi: il quadro ha sollevato "
            "dentro il passo"
        )
        assert controller.core.governor_policy is not None
    finally:
        controller._close_governors()


def test_the_governor_thread_is_closed_when_the_run_ends_even_on_failure(tmp_path):
    """Il governatore possiede un thread: la run deve chiuderlo comunque vada.

    Senza `close()` il thread e il suo event loop restano vivi dopo la run, e
    con un provider vero una chiamata in volo resta aperta con la sua
    connessione. Con un'eccezione a meta' passo la garanzia vale lo stesso: e'
    il `finally` di `run_async`.
    """
    from scripts.parity_harness import realistic_config
    import src.simulation.agent_coupled_runner as runner_module
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    def governed_config():
        config = realistic_config(10, 4, 0)
        config["governors"] = {"arm": "scripted", "cadence_steps": 2}
        return config

    runner = AgentCoupledRunner(governed_config())
    thread = runner._governor._thread
    runner.run(days=4, output_dir=tmp_path / "ok")
    assert not thread.is_alive()

    crashing = AgentCoupledRunner(governed_config())
    crashing_thread = crashing._governor._thread
    original_step = runner_module._core_step

    def exploding_step(*args, **kwargs):
        raise RuntimeError("passo esploso a meta' run")

    runner_module._core_step = exploding_step
    try:
        with pytest.raises(RuntimeError, match="esploso"):
            crashing.run(days=4, output_dir=tmp_path / "boom")
    finally:
        runner_module._core_step = original_step
    assert not crashing_thread.is_alive()


def test_the_gui_shell_wires_the_governor_too(tmp_path, monkeypatch):
    """L'innesto nella shell GUI, che ha il proprio loop.

    Senza questo test l'innesto in `SimulationController` potrebbe non essere
    collegato affatto — nessuna policy, nessun registro, nessun blocco
    `governors` nello stato live — e la GUI mostrerebbe una run "governata" che
    governata non e'.
    """
    from src.api.state_store import SimulationController
    from src.governors.record import load_records

    # Le run della GUI scrivono sotto `outputs/runs` relativo alla directory
    # corrente: spostarla in `tmp_path` tiene il repository pulito.
    monkeypatch.chdir(tmp_path)
    controller = SimulationController(_gui_config(arm="scripted", cadence_steps=2))
    try:
        controller.step(5)
        output_dir = controller.output_dir
        state = controller.current_state()
        assert state["governors"], "lo stato live deve dichiarare il governatore"
        assert state["governors"]["in_force"] is not None, (
            "dopo cinque passi con cadenza 2 una policy deve essere in vigore"
        )
        assert state["governors"]["last_tick"]["rules_in_force"] >= 1
        assert controller.core.governor_policy is not None
        rows = load_records(output_dir / "governor_decisions.jsonl")
        assert rows and rows[0]["policy"]["rules"]
    finally:
        controller._close_governors()
