"""Il confronto fra bracci, e soprattutto il suo primo controllo.

Uno strumento di confronto che presenta tabelle ordinate su bracci identici e'
peggio di nessuno strumento: mette in bella copia un non-risultato e lo fa
leggere come "governare non serve". Qui si prova che il caso nullo viene
dichiarato per primo, con la sua diagnosi, e che il comando esce con codice 1.
"""

import json

from scripts.compare_arms import (
    build_report,
    discover_runs,
    first_divergence,
    governor_activity,
    load_series,
)

_COLONNE = "step,population,structures_built,greenhouses,colony_prosperity_index,food_margin,material_margin,average_habitability"


def _scrivi_run(root, arm: str, seed: int, strutture: list[int], record: list[dict] | None = None):
    run = root / f"{arm}_seed{seed}"
    run.mkdir(parents=True)
    righe = [_COLONNE]
    for indice, valore in enumerate(strutture, start=1):
        righe.append(f"{indice},{100 + indice},{valore},{valore // 10},0.5,1.2,0.9,0.004")
    (run / "state_timeseries.csv").write_text("\n".join(righe) + "\n", encoding="utf-8")
    (run / "action_summary.json").write_text(
        json.dumps({"by_action": {"build_greenhouse": {"accepted": valore}}}), encoding="utf-8"
    )
    if record is not None:
        (run / "governor_decisions.jsonl").write_text(
            "\n".join(json.dumps(riga) for riga in record) + ("\n" if record else ""),
            encoding="utf-8",
        )
    return run


def _tick(tick: int, regole: list[dict], motivazione: str = "perche' si") -> dict:
    policy = {"rationale": motivazione, "rules": regole}
    return {
        "tick": tick,
        "application_step": 1 + (tick + 1) * 20,
        "picture_step": 1 + tick * 20,
        "picture_digest": "x",
        "missed": False,
        "governor": {
            "provider": "finto", "model": "m",
            "rationale": motivazione, "tokens_in": 10, "tokens_out": 5,
            "cost": 0.0, "latency_s": 1.5, "drops": {}, "proposal": policy,
        },
        "policy": policy,
    }


def _regola(indicatore="food_per_occupant", pilastro="sustenance", peso=2.0):
    return {
        "if": {"indicator": indicatore, "op": "<", "value": 2.0},
        "weights": {pilastro: peso},
    }


def test_two_identical_runs_never_diverge(tmp_path):
    a = load_series(_scrivi_run(tmp_path, "none", 0, [10, 20, 30]))
    b = load_series(_scrivi_run(tmp_path, "scripted", 0, [10, 20, 30]))
    assert first_divergence(a, b) is None


def test_the_divergence_step_is_the_first_one_that_differs(tmp_path):
    a = load_series(_scrivi_run(tmp_path, "none", 0, [10, 20, 30, 40]))
    b = load_series(_scrivi_run(tmp_path, "scripted", 0, [10, 20, 31, 45]))
    assert first_divergence(a, b) == 3


def test_an_empty_record_is_reported_as_empty(tmp_path):
    """Registro vuoto: la firma della cadenza piu' lunga della run."""
    run = _scrivi_run(tmp_path, "llm", 0, [10, 20], record=[])
    assert governor_activity(run)["registro"] == "VUOTO"


def test_a_missing_record_is_not_confused_with_an_empty_one(tmp_path):
    run = _scrivi_run(tmp_path, "llm", 1, [10, 20])
    assert governor_activity(run)["registro"] == "assente"


def test_the_report_declares_a_void_experiment_before_anything_else(tmp_path):
    _scrivi_run(tmp_path, "none", 0, [10, 20, 30])
    _scrivi_run(tmp_path, "scripted", 0, [10, 20, 30], record=[])
    report, payload = build_report(tmp_path, "none", rationales=5)

    assert payload["identical_to_baseline"] == ["scripted_seed0"]
    testa = report.split("## 2.")[0]
    assert "IDENTICHE" in testa, "il verdetto di vacuita' deve stare prima delle tabelle"
    assert "cadenza piu' lunga della run" in testa, "deve portare anche la diagnosi"


def test_a_real_difference_is_not_flagged_as_void(tmp_path):
    _scrivi_run(tmp_path, "none", 0, [10, 20, 30])
    _scrivi_run(tmp_path, "scripted", 0, [10, 22, 36], record=[_tick(0, [_regola()])])
    report, payload = build_report(tmp_path, "none", rationales=5)

    assert payload["identical_to_baseline"] == []
    assert payload["first_divergence"]["scripted_seed0"] == 2
    assert "Nessun braccio coincide" in report
    assert "sustenance x1" in report, "va detto QUALE pilastro e' stato pesato, per nome"
    assert "food_per_occupant x1" in report, "e su quale condizione"
    assert "perche' si" in report, "la motivazione del governatore va riportata"


def test_the_sign_column_says_when_the_seeds_disagree(tmp_path):
    """Con pochi semi la concordanza di segno e' l'unica lettura difendibile."""
    for seed, (base, trattato) in enumerate([(30, 40), (30, 45), (30, 25)]):
        _scrivi_run(tmp_path, "none", seed, [10, base])
        _scrivi_run(tmp_path, "scripted", seed, [10, trattato])
    report, _ = build_report(tmp_path, "none", rationales=0)
    assert "| ++- |" in report, "due semi su tre in una direzione non e' un risultato"


def test_runs_are_discovered_by_directory_name(tmp_path):
    _scrivi_run(tmp_path, "none", 0, [1])
    _scrivi_run(tmp_path, "llm", 7, [1])
    (tmp_path / "results.json").write_text("[]", encoding="utf-8")
    assert set(discover_runs(tmp_path)) == {("none", 0), ("llm", 7)}


def test_pillars_travel_by_name_in_the_report(tmp_path):
    """Il registro v2 porta i pilastri per nome, e il rapporto li riporta
    cosi': "pilastro 2 x1" non sarebbe un rapporto."""
    _scrivi_run(tmp_path, "none", 0, [10, 20])
    _scrivi_run(tmp_path, "scripted", 0, [10, 25], record=[
        _tick(0, [_regola(pilastro="build")])
    ])
    report, _ = build_report(tmp_path, "none", rationales=0)
    assert "build x1" in report


def test_an_unconditional_rule_is_reported_as_such(tmp_path):
    """Una policy fatta della sola regola incondizionata non deve sparire dal
    rapporto come se non avesse condizioni da mostrare."""
    _scrivi_run(tmp_path, "none", 0, [10, 20])
    _scrivi_run(tmp_path, "scripted", 0, [10, 25], record=[
        _tick(0, [{"if": None, "weights": {"explore": 3.0}}])
    ])
    report, _ = build_report(tmp_path, "none", rationales=0)
    assert "explore x1" in report
    assert "nessuna condizione" in report


def test_a_council_era_record_is_declared_not_summed_to_zero(tmp_path):
    """Un registro del consiglio (pre-2026-08-24) va dichiarato per quello che
    e', non letto come un registro senza policy."""
    _scrivi_run(tmp_path, "none", 0, [10, 20])
    _scrivi_run(tmp_path, "scripted", 0, [10, 25], record=[{
        "tick": 0, "missed": False,
        "governors": [{"mandate": "life_support"}],
        "directive": {"cells": []},
    }])
    from scripts.compare_arms import governor_activity
    attivita = governor_activity(tmp_path / "scripted_seed0")
    assert "CONSIGLIO" in attivita["registro"]


def test_a_rationale_the_console_cannot_encode_does_not_kill_the_report(tmp_path):
    """Il testo di un modello e' unicode arbitrario, e la console Windows no.

    Una motivazione conteneva U+2011 (trattino insecabile): `print` sollevava
    `UnicodeEncodeError` in cp1252 DOPO aver calcolato tutto il confronto. Il
    rapporto piu' interessante -- quello di una run LLM vera -- e' anche l'unico
    che puo' contenere quei caratteri.
    """
    _scrivi_run(tmp_path, "none", 0, [10, 20])
    _scrivi_run(tmp_path, "llm", 0, [10, 25], record=[
        _tick(0, [_regola()],
              motivazione="margine non\u2011critico, \u2192 spingo le serre \u00e8 ok")
    ])
    report, _ = build_report(tmp_path, "none", rationales=5)
    assert "\u2011" in report, "il rapporto conserva il testo del modello"

    # La copia a schermo si degrada, non fallisce: e' cio' che fa
    # `_console_must_not_crash_on_model_text`.
    assert report.encode("cp1252", errors="replace"), "la console non deve poter sollevare"


def test_the_report_file_keeps_the_characters_the_console_would_lose(tmp_path):
    """Il degrado vale per lo schermo, non per il file: `--out` resta completo."""
    _scrivi_run(tmp_path, "none", 0, [10, 20])
    _scrivi_run(tmp_path, "llm", 0, [10, 25], record=[
        _tick(0, [_regola()], motivazione="non\u2011critico")
    ])
    report, _ = build_report(tmp_path, "none", rationales=5)
    destinazione = tmp_path / "confronto.md"
    destinazione.write_text(report, encoding="utf-8")
    assert "non\u2011critico" in destinazione.read_text(encoding="utf-8")


def test_many_seeds_switch_the_table_to_a_distribution(tmp_path):
    """Con molte ripetizioni conta la distribuzione, non il singolo seme.

    Dieci colonne per-seme sono illeggibili, ma il motivo vero e' un altro:
    con dieci semi la domanda non e' piu' "quanto ha fatto il seme 7" bensi'
    "quanto sposta il seme, e lo scarto e' piu' grande di quello".
    """
    for seed in range(10):
        # Sette semi in su, tre in giu': segnale reale ma non unanime.
        trattato = 40 if seed < 7 else 20
        _scrivi_run(tmp_path, "none", seed, [10, 30])
        _scrivi_run(tmp_path, "scripted", seed, [10, trattato])
    report, _ = build_report(tmp_path, "none", rationales=0)

    assert "scarto mediano" in report, "oltre sei semi la tabella diventa distribuzione"
    assert "seme 7" not in report, "le colonne per-seme non devono comparire"
    assert "7/10 +" in report, "la concordanza va riportata sulla direzione piu' numerosa"


def test_few_seeds_keep_the_per_seed_columns(tmp_path):
    for seed in range(3):
        _scrivi_run(tmp_path, "none", seed, [10, 30])
        _scrivi_run(tmp_path, "scripted", seed, [10, 40])
    report, _ = build_report(tmp_path, "none", rationales=0)
    assert "seme 2" in report
    assert "scarto mediano" not in report


def test_the_majority_direction_is_reported_even_when_it_is_negative(tmp_path):
    """Con 3 positivi su 10 il fatto notevole e' che sette vanno di sotto:
    leggere `3/10` inviterebbe a concludere che non succeda niente."""
    for seed in range(10):
        trattato = 40 if seed < 3 else 20
        _scrivi_run(tmp_path, "none", seed, [10, 30])
        _scrivi_run(tmp_path, "scripted", seed, [10, trattato])
    report, _ = build_report(tmp_path, "none", rationales=0)
    assert "7/10 -" in report


def test_a_target_never_reached_is_counted_apart_from_the_median(tmp_path):
    """Non raggiungere il bersaglio non e' una lentezza, e' un altro esito.

    Mescolare i "mai" nella mediana dei passi renderebbe un fallimento
    indistinguibile da un ritardo, che e' il modo in cui una tabella ordinata
    dice una cosa falsa.
    """
    for seed in range(10):
        _scrivi_run(tmp_path, "none", seed, [10, 20, 30])
        # Cinque semi superano il bersaglio presto, cinque restano sotto.
        _scrivi_run(tmp_path, "scripted", seed, [10, 30, 30] if seed < 5 else [10, 12, 14])
    report, _ = build_report(tmp_path, "none", rationales=0)

    assert "mai raggiunto" in report
    assert "5/10" in report, "i semi che non arrivano vanno contati, non mediati"


def test_le_run_si_trovano_anche_in_sottocartelle(tmp_path):
    """Una campagna lanciata in piu' processi scrive in cartelle diverse.

    I bracci si eseguono in parallelo --- uno per processo, cosi' l'attesa del
    modello si sovrappone invece di sommarsi --- e ognuno ha la sua cartella.
    Se il confronto guardasse un solo livello, i bracci mancanti non darebbero
    errore: darebbero un rapporto con meno bracci, che si legge come un
    esperimento piu' povero invece che come un difetto dello strumento.
    """
    _scrivi_run(tmp_path / "controlli", "none", 3, [1])
    _scrivi_run(tmp_path / "controlli", "scripted", 3, [1])
    _scrivi_run(tmp_path / "llm_cieco", "llm_cieco", 3, [1])
    assert set(discover_runs(tmp_path)) == {
        ("none", 3), ("scripted", 3), ("llm_cieco", 3),
    }


def test_lo_stesso_nome_sotto_flussi_diversi_resta_due_bracci(tmp_path):
    """Il modello non entra nel nome del braccio, ma il flusso si'.

    Due governatori che parlano a modelli diversi sono due trattamenti, e la
    cartella di flusso e' l'unica cosa che li distingue: il nome del braccio
    porta arm, gradino di contesto e strato amministrativo, non il modello.
    Sceglierne uno a caso falserebbe il confronto; scartarne uno perderebbe un
    braccio.
    """
    _scrivi_run(tmp_path / "modello_grande", "llm_completo", 3, [1])
    _scrivi_run(tmp_path / "modello_piccolo", "llm_completo", 3, [2])
    assert set(discover_runs(tmp_path)) == {
        ("modello_grande/llm_completo", 3),
        ("modello_piccolo/llm_completo", 3),
    }


def test_il_prefisso_si_aggiunge_solo_dove_serve(tmp_path):
    """Nel caso normale i nomi restano leggibili."""
    _scrivi_run(tmp_path / "controlli", "none", 3, [1])
    _scrivi_run(tmp_path / "llm", "llm_completo", 3, [1])
    assert set(discover_runs(tmp_path)) == {("none", 3), ("llm_completo", 3)}
