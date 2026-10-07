"""Drives the terminal wizard through its numbered plain-mode fallback.

MARS_WIZARD_PLAIN forces line-based menus, so the whole flow is scriptable
via stdin exactly like a user typing choices.
"""

import io

import pytest

from src.cli.config_wizard import run_wizard, summary_lines
from src.experiments.manual_config import ManualConfigOptions, build_manual_config


@pytest.fixture(autouse=True)
def plain_mode(monkeypatch):
    monkeypatch.setenv("MARS_WIZARD_PLAIN", "1")


def _feed(monkeypatch, lines: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))


def _menu_index(value: str) -> str:
    """Il numero da digitare per la voce `value` del menu principale.

    Le sequenze scritte a mano si sono gia' rotte due volte per l'inserimento di
    una voce in mezzo, e ogni volta il sintomo era illeggibile: il wizard
    consumava l'input sbagliato e moriva con EOF, senza indicare quale numero
    fosse cambiato. Chiedere l'indice al menu stesso rende i test indifferenti
    all'ordine, che e' una scelta di presentazione e non un contratto.
    """
    from src.cli.config_wizard import _main_menu

    selezionabili = [item for item in _main_menu(ManualConfigOptions()).items if item.kind not in {"separator"}]
    for numero, item in enumerate(selezionabili, start=1):
        if item.value == value:
            return str(numero)
    raise AssertionError(f"voce di menu assente: {value}")


#: Le voci usate dalle sequenze qui sotto, risolte una volta sola.
RUN = _menu_index("run")
TEMPO = _menu_index("time")
COLONIA = _menu_index("colony")
AGENTI = _menu_index("agents")
CONSIGLIO = _menu_index("governors")
IMPORTA = _menu_index("import")
AVVIA = _menu_index("launch")


def test_wizard_sets_run_name_and_launches(monkeypatch):
    # 1=Run e scenario, 1=Nome run, testo, 0=indietro, 9=riepilogo, s=avvia
    _feed(monkeypatch, f"{RUN}\n1\nwizard_smoke\n0\n{AVVIA}\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.run_name == "wizard_smoke"
    assert options.max_steps == 500  # defaults preserved unless edited
    assert options.days_per_step == 7
    assert options.agent_count == 50
    assert options.max_llm_calls_per_step == 0
    assert options.fast_observation_enabled is False


def test_wizard_manual_colony_site_flows_into_config(monkeypatch):
    # 4=Colonia, 1=Sito colonia, 2=Manuale, X, Y, 0=indietro,
    # 9=riepilogo (nome mancante -> prompt), s=avvia
    _feed(monkeypatch, f"{COLONIA}\n1\n2\n210\n95\n0\n{AVVIA}\nwizard_colony\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.run_name == "wizard_colony"
    assert options.colony_start_x == 210
    assert options.colony_start_y == 95

    config = build_manual_config(options)
    assert config["colony"]["start_x"] == 210
    assert config["colony"]["start_y"] == 95


def test_wizard_llm_assignments_menu_sets_provider_and_model(monkeypatch):
    # 5=Agenti, 1=Modalita', 2=mix, 5=Provider e modelli LLM,
    # 1=Imposta tutti, 1=gpt, 1=gpt-4o-mini, 0=indietro, 0=indietro,
    # 9=riepilogo (nome mancante -> prompt), s=avvia
    _feed(monkeypatch, f"{AGENTI}\n1\n2\n5\n1\n1\n1\n0\n0\n{AVVIA}\nwizard_llm\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.mode == "mix"
    assert options.assignments == [{"provider": "gpt", "model": "gpt-4o-mini"}] * 3

    config = build_manual_config(options)
    assert config["agents"]["llm_count"] == 3
    assert config["agents"]["llm_assignments"] == [{"provider": "gpt", "model": "gpt-4o-mini"}] * 3


def test_wizard_exit_without_launch_returns_none(monkeypatch):
    # 0=esci dal menu principale, conferma vuota (default = esci)
    _feed(monkeypatch, "0\n\n")
    assert run_wizard() is None


def test_wizard_time_section_supports_unlimited_steps(monkeypatch):
    # 2=Tempo, 1=step massimi, '-'=nessun limite, 2=giorni per step, 3650,
    # 0=indietro, 9=riepilogo, nome, s=avvia
    _feed(monkeypatch, f"{TEMPO}\n1\n-\n2\n3650\n0\n{AVVIA}\nwizard_deeptime\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.max_steps is None
    assert options.days_per_step == 3650
    config = build_manual_config(options)
    assert "days" not in config
    assert config["simulation"]["days_per_step"] == 3650


def test_wizard_prefilled_options_are_editable(monkeypatch):
    # Prefill (come da flag CLI), poi 9=riepilogo e avvio diretto.
    _feed(monkeypatch, f"{AVVIA}\ns\n")
    initial = ManualConfigOptions(run_name="prefilled", seed=7, colony_start_x=12, colony_start_y=88)
    options = run_wizard(initial)

    assert options is not None
    assert options.run_name == "prefilled"
    assert options.seed == 7
    assert options.colony_start_x == 12
    assert options.colony_start_y == 88


def test_summary_lines_cover_all_sections():
    lines = "\n".join(summary_lines(ManualConfigOptions(run_name="recap", colony_start_x=10, colony_start_y=20)))
    for token in ("recap", "manuale (10, 20)", "Tempo:", "Mondo:", "Agenti:", "Eventi:", "Avanzate:"):
        assert token in lines


def test_wizard_configures_the_governor(monkeypatch):
    """Il governatore si imposta dal terminale come dal pannello della GUI.

    Nel menu del governatore (senza piu' conteggio e mandati): 1=Braccio ->
    4=llm, 2=Cadenza -> 10, 4=Provider, 5=Modello, 7=Temperatura -> 0. Senza
    questa sezione una run governata si poteva avviare solo passando argomenti
    allo script, e il wizard non era piu' la stessa cosa del frontend.
    """
    _feed(monkeypatch, f"{CONSIGLIO}\n1\n4\n2\n10\n4\ngpu_farm\n5\ngpt-oss:20b\n7\n0\n0\n{AVVIA}\nwizard_gov\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.governors_arm == "llm"
    assert options.governors_cadence_steps == 10
    assert options.governors_provider == "gpu_farm"
    assert options.governors_model == "gpt-oss:20b"
    assert options.governors_temperature == 0.0

    config = build_manual_config(options)
    assert config["governors"]["arm"] == "llm"
    assert config["governors"]["cadence_steps"] == 10
    # Le assegnazioni ci sono solo con un provider reale: senza,
    # `build_governor` costruirebbe il fallback e nessun modello parlerebbe.
    assert config["governors"]["assignments"][0]["provider"] == "gpu_farm"
    assert config["governors"]["assignments"][0]["temperature"] == 0.0


def test_wizard_without_a_governor_writes_no_block(monkeypatch):
    """`none` non scrive il blocco, e non e' una sfumatura.

    `build_governor` legge il blocco assente come "nessun governatore" e
    restituisce `None`; col `None` il kernel non invoca nemmeno la funzione che
    applica la policy, quindi la baseline resta bit-exact per costruzione
    invece che per lettura di una stringa.
    """
    _feed(monkeypatch, f"{RUN}\n1\nwizard_nogov\n0\n{AVVIA}\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.governors_arm == "none"
    assert "governors" not in build_manual_config(options)


def test_wizard_repeats_a_previous_run(monkeypatch, tmp_path):
    """Importare una run gia' eseguita ne riporta tutte le impostazioni.

    E' la scorciatoia che evita di ricomporre a mano una configurazione per
    cambiarne un campo solo, che e' il caso di gran lunga piu' frequente.
    """
    import json

    from src.experiments import run_import

    sorgente = build_manual_config(ManualConfigOptions(
        run_name="run_da_ripetere", seed=17, agent_count=123,
        map_profile="ice_rich", governors_arm="scripted", governors_cadence_steps=42,
    ))
    cartella = tmp_path / "vecchia"
    cartella.mkdir()
    (cartella / run_import.CONFIG_FILENAME).write_text(json.dumps(sorgente), encoding="utf-8")
    monkeypatch.setattr(
        "src.api.routes_runs.discover_runs", lambda: {"vecchia": cartella}
    )

    # IMPORTA, 1=la sola run offerta, poi riepilogo, nome nuovo, avvio.
    _feed(monkeypatch, f"{IMPORTA}\n1\n{AVVIA}\nrun_ripetuta\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.seed == 17
    assert options.agent_count == 123
    assert options.map_profile == "ice_rich"
    assert options.governors_arm == "scripted"
    assert options.governors_cadence_steps == 42
    # Il nome NON si eredita: due run indistinguibili sono esattamente la
    # confusione che questa scorciatoia dovrebbe risparmiare.
    assert options.run_name == "run_ripetuta"


def test_wizard_import_with_no_saved_runs_changes_nothing(monkeypatch):
    """Nessuna run da importare non deve azzerare quanto gia' impostato."""
    monkeypatch.setattr("src.api.routes_runs.discover_runs", lambda: {})

    _feed(monkeypatch, f"{IMPORTA}\n{RUN}\n1\nwizard_vuoto\n0\n{AVVIA}\ns\n")
    options = run_wizard()

    assert options is not None
    assert options.run_name == "wizard_vuoto"
    assert options.governors_arm == "none"
