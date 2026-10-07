"""Le run devono essere trovabili dal frontend di analisi, ovunque siano.

Tre difetti misurati sulle run vere di questa campagna:

1. l'API guardava solo `outputs/runs`, mentre gli esperimenti da terminale
   scrivono in `runs/` — nessuna delle quaranta run di governatori era
   analizzabile;
2. le run di un esperimento stanno **annidate** (`runs/campagna/llm_seed0`), e
   l'elenco leggeva solo i figli diretti della radice;
3. `governor_decisions.jsonl` veniva scritto ma non compariva fra gli stream
   dell'analisi, cioe' proprio le decisioni del consiglio — razionali, proposte
   per mandato, token, latenza — restavano invisibili.
"""

import json

from fastapi.testclient import TestClient

from src.api.main import app
import src.api.routes_runs as routes_runs


def _scrivi_run(cartella, popolazione: int = 7):
    cartella.mkdir(parents=True, exist_ok=True)
    (cartella / "run_metadata.json").write_text('{"run_id":"x"}', encoding="utf-8")
    (cartella / "final_metrics.json").write_text(
        json.dumps({"population": popolazione}), encoding="utf-8"
    )
    return cartella


def test_a_run_written_by_the_terminal_is_listed(tmp_path, monkeypatch):
    _scrivi_run(tmp_path / "runs" / "prova_diretta")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    client = TestClient(app)
    elenco = client.get("/api/runs").json()
    assert [r["run_id"] for r in elenco] == ["prova_diretta"]


def test_nested_experiment_runs_are_listed_one_by_one(tmp_path, monkeypatch):
    """Un esperimento e' una cartella di run, non una run.

    `run_governor_experiment.py` scrive `<out>/<braccio>_seed<n>/`: senza questo,
    quaranta run di una campagna comparivano come zero.
    """
    campagna = tmp_path / "runs" / "campagna"
    _scrivi_run(campagna / "llm_seed0", 300)
    _scrivi_run(campagna / "none_seed0", 313)
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    client = TestClient(app)
    ids = sorted(r["run_id"] for r in client.get("/api/runs").json())
    assert ids == [
        f"campagna{routes_runs.RUN_SEPARATOR}llm_seed0",
        f"campagna{routes_runs.RUN_SEPARATOR}none_seed0",
    ]


def test_a_nested_run_can_be_analysed_and_its_files_read(tmp_path, monkeypatch):
    """Il separatore non e' `/` di proposito.

    Un identificatore con la barra dentro renderebbe ambigua la rotta dei file
    (`/api/runs/{id}/files/{path}`) e costringerebbe a fidarsi di come il server
    decodifica `%2F`. Con un separatore che nelle rotte non compare, l'ambiguita'
    non esiste.
    """
    run = _scrivi_run(tmp_path / "runs" / "campagna" / "llm_seed3", 321)
    (run / "governor_decisions.jsonl").write_text(
        '{"tick":0,"application_step":1,"governors":[{"mandate":"life_support",'
        '"rationale":"serre sotto soglia","tokens_out":204}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    client = TestClient(app)
    rid = f"campagna{routes_runs.RUN_SEPARATOR}llm_seed3"

    analisi = client.get(f"/api/runs/{rid}/analysis").json()
    assert analisi["final_metrics"]["population"] == 321
    assert analisi["overview"]["governor_ticks"] == 1
    assert analisi["streams"]["governor_decisions"][0]["tick"] == 0

    contenuto = client.get(f"/api/runs/{rid}/files/final_metrics.json").json()
    assert contenuto["population"] == 321


def test_the_council_rationales_reach_the_analysis(tmp_path, monkeypatch):
    """E' la voce che l'utente guarda per prima: che cosa ha detto il consiglio.

    Il file c'era gia' — 363 KB per run — ma nessun consumatore lo apriva.
    """
    run = _scrivi_run(tmp_path / "runs" / "solo")
    (run / "governor_decisions.jsonl").write_text(
        '{"tick":0,"missed":false,"governors":[{"mandate":"logistics","rationale":"a"}],'
        '"directive":{"cells":[{"y":1,"x":2}]}}\n'
        '{"tick":1,"missed":true,"governors":[],"directive":null}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    analisi = TestClient(app).get("/api/runs/solo/analysis").json()
    assert analisi["overview"]["governor_ticks"] == 2
    assert analisi["overview"]["governor_directives"] == 1
    assert analisi["overview"]["governor_misses"] == 1
    assert analisi["insights"]["governor_mandates"] == {"logistics": 1}


def test_a_directory_without_a_run_inside_is_not_a_run(tmp_path, monkeypatch):
    """Una cartella qualunque sotto la radice non deve diventare una run vuota."""
    (tmp_path / "runs" / "cartaccia").mkdir(parents=True)
    _scrivi_run(tmp_path / "runs" / "vera")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    elenco = TestClient(app).get("/api/runs").json()
    assert [r["run_id"] for r in elenco] == ["vera"]


def test_huge_streams_are_capped_but_counted(tmp_path, monkeypatch):
    """Con i log per azione accesi una run scrive centinaia di migliaia di righe.

    Misurato: 300 agenti per 200 passi producono 59,5 MB di
    `validated_actions.jsonl`, che a 800 passi diventano circa 260 MB. Se
    l'analisi le materializzasse tutte nel proprio payload, accendere i log
    renderebbe il frontend inutilizzabile — cioe' la correzione romperebbe la
    cosa che doveva servire.

    Il conteggio deve pero' restare **vero**: un totale tagliato al tetto
    direbbe che la run ha fatto meno azioni di quante ne ha fatte.
    """
    run = _scrivi_run(tmp_path / "runs" / "grande")
    righe = "\n".join(
        json.dumps({"step": i, "action": "observe"}) for i in range(1, 5001)
    )
    (run / "validated_actions.jsonl").write_text(righe + "\n", encoding="utf-8")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(routes_runs, "MAX_STREAM_ROWS", 100)

    analisi = TestClient(app).get("/api/runs/grande/analysis").json()
    assert analisi["overview"]["validated_actions"] == 5000, "il conteggio e' quello vero"
    assert len(analisi["streams"]["validated_actions"]) == 100, "le righe sono tagliate"
    voce = next(f for f in analisi["files"] if f["name"] == "validated_actions.jsonl")
    assert voce["row_count"] == 5000
    assert voce["truncated"] is True


def test_the_file_listing_does_not_carry_the_whole_run_twice(tmp_path, monkeypatch):
    """L'elenco dei file e' un indice, non una seconda copia.

    Misurato su una run vera: payload di 160 MB, di cui 80 erano
    `world_static_base.json` serializzato una seconda volta dentro `files` —
    la prima essendo `replay_static_base`. Il frontend, dal canto suo, di
    `files` usa soltanto i nomi.
    """
    run = _scrivi_run(tmp_path / "runs" / "indice")
    (run / "world_static_base.json").write_text(
        json.dumps({"width": 4, "height": 3, "cells": [{"x": i} for i in range(200)]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    analisi = TestClient(app).get("/api/runs/indice/analysis").json()

    voce = next(f for f in analisi["files"] if f["name"] == "world_static_base.json")
    assert "data" not in voce and "rows" not in voce and "text" not in voce
    assert voce["size_bytes"] > 0, "l'indice dice comunque quanto pesa"
    assert len(analisi["replay_static_base"]["cells"]) == 200, "il contenuto resta dov'e' letto"


def test_with_several_roots_the_identifier_carries_the_root_name(tmp_path, monkeypatch):
    """Tre campagne, stessi nomi di braccio e di seme: senza il nome della radice
    nell'identificatore vince la prima e le altre due non si possono aprire.
    Misurato il 2026-09-06 con `campagna_scarsa`, `_v2` e `_v3`, che hanno tutte
    `llm_completo_amm/llm_completo+amm_seed3`."""
    import os

    for campagna, pop in (("campagna_a", 1), ("campagna_b", 2)):
        _scrivi_run(tmp_path / campagna / "llm_completo_amm" / "llm_completo+amm_seed3", pop)
    monkeypatch.setenv(
        "MARSABM_RUNS_ROOT",
        os.pathsep.join([str(tmp_path / "campagna_a"), str(tmp_path / "campagna_b")]),
    )
    client = TestClient(app)
    ids = sorted(r["run_id"] for r in client.get("/api/runs").json())
    assert ids == [
        "campagna_a~llm_completo_amm~llm_completo+amm_seed3",
        "campagna_b~llm_completo_amm~llm_completo+amm_seed3",
    ]
    analisi = client.get(f"/api/runs/{ids[1]}/analysis").json()
    assert analisi["final_metrics"]["population"] == 2


def test_with_a_single_root_identifiers_stay_as_before(tmp_path, monkeypatch):
    """Le run aperte dalla GUI restano raggiungibili con l'identificatore di sempre."""
    _scrivi_run(tmp_path / "runs" / "campagna" / "llm_seed0")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    client = TestClient(app)
    assert [r["run_id"] for r in client.get("/api/runs").json()] == ["campagna~llm_seed0"]


def test_the_static_base_reaches_the_analysis_slimmed_to_what_the_replay_draws(tmp_path, monkeypatch):
    """`world_static_base.json` pesa 73 MB su 360x180 perche' ogni cella porta
    trenta campi; il replay ne disegna quattro. Allegato intero all'analisi
    faceva 92 MB per run e il proxy di Vite si arrendeva prima della risposta
    (misurato il 2026-09-06). Ridotta ai campi disegnati pesa 5 MB."""
    run = _scrivi_run(tmp_path / "runs" / "prova")
    (run / "world_static_base.json").write_text(json.dumps({
        "width": 2, "height": 1, "day": 0, "metadata": {"a": 1},
        "cells": [
            {"x": 0, "y": 0, "terrain": "crater", "habitability": 0.2, "radiation": 1.1,
             "resources": {"food": 3}, "geometry": {"area_m2": 9}, "structures": [{"type": "habitat"}],
             "agents_present": ["a1"], "explored": True},
            {"x": 1, "y": 0, "terrain": "regolith_plain", "habitability": 0.0, "radiation": 0.9,
             "resources": {}, "geometry": {}, "structures": [], "agents_present": [], "explored": False},
        ],
    }), encoding="utf-8")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(tmp_path / "runs"))
    base = TestClient(app).get("/api/runs/prova/analysis").json()["replay_static_base"]
    assert base["width"] == 2 and base["height"] == 1
    assert base["cells"] == [
        {"x": 0, "y": 0, "terrain": "crater", "habitability": 0.2, "structures": [{"type": "habitat"}],
         "agents_present": ["a1"], "explored": True},
        {"x": 1, "y": 0, "terrain": "regolith_plain", "habitability": 0.0, "structures": [],
         "agents_present": [], "explored": False},
    ]


def test_campaign_runs_three_levels_deep_are_listed_with_the_default_roots(tmp_path, monkeypatch):
    """`runs/<campagna>/<braccio>/<run>`: tre livelli, cioe' come scrive davvero
    `run_governor_experiment.py`. Fino al 2026-09-06 se ne leggevano due e la
    tendina, con le radici di default, non mostrava nessuna campagna."""
    radice = tmp_path / "runs"
    _scrivi_run(radice / "campagna_scarsa" / "ctrl_none" / "none_seed3")
    _scrivi_run(radice / "campagna_scarsa" / "llm_completo_amm" / "llm_completo+amm_seed3")
    _scrivi_run(radice / "prova_diretta")
    (radice / "campagna_scarsa" / "ctrl_none" / "none_seed3" / "world_snapshots").mkdir()
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(radice))
    ids = sorted(r["run_id"] for r in TestClient(app).get("/api/runs").json())
    S = routes_runs.RUN_SEPARATOR
    assert ids == [
        f"campagna_scarsa{S}ctrl_none{S}none_seed3",
        f"campagna_scarsa{S}llm_completo_amm{S}llm_completo+amm_seed3",
        "prova_diretta",
    ]


def test_runs_are_listed_from_the_most_recent_with_their_time(tmp_path, monkeypatch):
    """La tendina raggruppa per giorno e ordina per ora: il server deve dare
    `updated_at` (da `run_metadata.json`) e l'ordine dal piu' recente."""
    radice = tmp_path / "runs"
    vecchia = _scrivi_run(radice / "vecchia")
    nuova = _scrivi_run(radice / "nuova")
    (vecchia / "run_metadata.json").write_text(
        json.dumps({"run_id": "vecchia", "updated_at": "2026-09-01T08:00:00"}), encoding="utf-8"
    )
    (nuova / "run_metadata.json").write_text(
        json.dumps({"run_id": "nuova", "updated_at": "2026-09-06T15:30:00"}), encoding="utf-8"
    )
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(radice))
    elenco = TestClient(app).get("/api/runs").json()
    assert [r["run_id"] for r in elenco] == ["nuova", "vecchia"]
    assert elenco[0]["updated_at"] == "2026-09-06T15:30:00"


def test_a_run_without_updated_at_falls_back_to_the_file_time(tmp_path, monkeypatch):
    radice = tmp_path / "runs"
    _scrivi_run(radice / "senza_data")  # run_metadata senza updated_at
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(radice))
    [riga] = TestClient(app).get("/api/runs").json()
    assert riga["updated_at"].startswith("20")  # ISO 8601 dalla data del file


def _scrivi_run_confrontabile(cartella, passi: int, vivi: int, morti: int):
    """Una run con serie temporale, morti e due snapshot, come le campagne vere."""
    _scrivi_run(cartella, vivi)
    (cartella / "final_metrics.json").write_text(
        json.dumps({"population": vivi, "survival_rate": 0.9, "food_stock": 12.5, "nested": {"x": 1}}),
        encoding="utf-8",
    )
    righe = ["step,day,population,food_stock,colonna_ignota"]
    righe += [f"{i},{i * 7},{300 + i},{100 - i * 0.01},zzz" for i in range(passi + 1)]
    (cartella / "state_timeseries.csv").write_text("\n".join(righe) + "\n", encoding="utf-8")
    (cartella / "dead_agents.jsonl").write_text("".join('{"id": %d}\n' % i for i in range(morti)), encoding="utf-8")
    (cartella / "world_snapshots").mkdir()
    for passo in (25, 50):
        (cartella / "world_snapshots" / f"step_{passo:06d}_day_{passo * 7:06d}.json").write_text("{}", encoding="utf-8")
    (cartella / "world_static_base.json").write_text(
        json.dumps({"width": 2, "height": 1, "cells": [
            {"x": 0, "y": 0, "terrain": "plain", "habitability": 0.2, "structures": [], "agents_present": 0,
             "explored": False, "temperature": -60, "extra": "via"},
        ]}),
        encoding="utf-8",
    )


def test_the_compare_payload_is_compact_and_keeps_the_last_row(tmp_path, monkeypatch):
    radice = tmp_path / "runs"
    _scrivi_run_confrontabile(radice / "campagna" / "braccio" / "run_seed3", passi=1000, vivi=1300, morti=4)
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(radice))
    S = routes_runs.RUN_SEPARATOR
    dati = TestClient(app).get(f"/api/runs/campagna{S}braccio{S}run_seed3/compare").json()
    assert dati["label"] == {"campaign": "campagna", "arm": "braccio", "run": "run_seed3"}
    assert dati["steps"] == 1000
    serie = dati["timeseries"]
    assert len(serie) <= routes_runs.COMPARE_MAX_ROWS + 1
    assert serie[0]["step"] == 0 and serie[-1]["step"] == 1000       # l'ultima riga c'e' sempre
    assert "colonna_ignota" not in serie[0] and "food_stock" in serie[0]
    riassunto = dati["summary"]
    assert riassunto["population_final"] == 1300 and riassunto["deaths"] == 4
    assert riassunto["population_initial"] == 300 and riassunto["births"] == 1004
    assert riassunto["survival_rate"] == 0.9 and riassunto["snapshots"] == 2
    assert dati["final_metrics"] == {"population": 1300, "survival_rate": 0.9, "food_stock": 12.5}
    assert [s["step"] for s in dati["snapshots"]] == [25, 50]
    assert dati["snapshots"][0]["path"] == "world_snapshots/step_000025_day_000175.json"
    assert "state_timeseries.csv" in dati["signature"] and "world_snapshots/" in dati["signature"]


def test_the_static_base_route_is_slim_and_404_without_a_base(tmp_path, monkeypatch):
    radice = tmp_path / "runs"
    _scrivi_run_confrontabile(radice / "con_base", passi=3, vivi=10, morti=0)
    _scrivi_run(radice / "senza_base")
    monkeypatch.setenv("MARSABM_RUNS_ROOT", str(radice))
    client = TestClient(app)
    base = client.get("/api/runs/con_base/static_base").json()
    assert base["width"] == 2 and len(base["cells"]) == 1
    assert "temperature" not in base["cells"][0] and "extra" not in base["cells"][0]
    assert base["cells"][0]["terrain"] == "plain"
    assert client.get("/api/runs/senza_base/static_base").status_code == 404
