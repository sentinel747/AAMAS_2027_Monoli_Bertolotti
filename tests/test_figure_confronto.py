# -*- coding: utf-8 -*-
"""Le figure del confronto appaiato devono leggere i dati senza mediare cio' che non va mediato.

- un mondo entra con i soli semi che hanno baseline E coppia;
- le repliche dello stesso seme restano distinte e il loro intervallo e' lo scarto;
- la differenza appaiata delle traiettorie usa i soli passi comuni;
- la scoperta preferisce la coppia `_v2` (parser corretto) e ordina per D;
- le figure escono in PDF e PNG.
"""

import json

import pytest

from scripts.figure_confronto import (
    Mondo,
    carica_mondo,
    differenza_appaiata,
    disegna_effetto_vs_difficolta,
    disegna_esiti,
    disegna_traiettorie,
    riepilogo,
    scarto_repliche,
    scopri_mondi,
    serie,
)
from tests.test_score_difficolta import _base


def _braccio(radice, nome, prefisso, esiti, con_base=True, scarsita=1.0, profilo="balanced"):
    """esiti: {seme: (vivi, morti, celle)}; scrive results.jsonl, cartelle run e serie temporali."""
    b = radice / nome
    b.mkdir(parents=True, exist_ok=True)
    righe = []
    for seme, (vivi, morti, celle) in esiti.items():
        run = b / f"{prefisso}_seed{seme}"
        run.mkdir()
        righe.append(json.dumps({
            "seed": seme, "population": vivi, "deaths": morti, "agents": 300,
            "espansione": {"celle_totali": celle},
            "config": {"map_profile": profilo, "dotazione": 0.6},
        }))
        (run / "state_timeseries.csv").write_text(
            "step,population,occupied_cells\n" + "".join(
                f"{p},{300 + (vivi - 300) * p // 10},{1 + celle * p // 10}\n" for p in range(1, 11)),
            encoding="utf-8")
        if con_base:
            base = _base(width=20, height=10, water_ice=8.124 * scarsita,
                         resources={"minerals": 7.56 * scarsita, "construction_material": 12.61 * scarsita})
            base["metadata"]["map_profile"] = profilo
            (run / "world_static_base.json").write_text(json.dumps(base), encoding="utf-8")
    (b / "results.jsonl").write_text("\n".join(righe) + "\n", encoding="utf-8")
    return b


def test_a_world_keeps_only_seeds_present_on_both_sides(tmp_path):
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1200, 200, 60), 5: (900, 50, 40)})
    c1 = _braccio(tmp_path, "coppia", "llm", {3: (1100, 90, 55), 4: (1150, 210, 58)}, con_base=False)
    c2 = _braccio(tmp_path, "coppia_rep2", "llm", {3: (1300, 80, 70)}, con_base=False)
    m = carica_mondo("balanced", base, [c1, c2], raggio=2)
    assert m.semi == [3, 4]
    assert m.effetti("vivi") == {3: [100, 300], 4: [-50]}
    assert m.effetti("celle") == {3: [5, 20], 4: [-2]}
    assert m.baseline[3].nascite == 1000 + 100 - 300
    assert m.D == pytest.approx(0.1)  # solo la dotazione pesa: (1 - 0.6) / 4


def test_replica_spread_is_the_mean_range_over_repeated_seeds(tmp_path):
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 0, 1), 4: (1000, 0, 1)})
    c1 = _braccio(tmp_path, "c1", "llm", {3: (1000, 0, 1), 4: (1500, 0, 1)}, con_base=False)
    c2 = _braccio(tmp_path, "c2", "llm", {3: (1200, 0, 1), 4: (1100, 0, 1)}, con_base=False)
    m = carica_mondo("w", base, [c1, c2], raggio=2)
    assert scarto_repliche(m, "vivi") == pytest.approx((200 + 400) / 2)
    solo = carica_mondo("w", base, [c1], raggio=2)
    assert scarto_repliche(solo, "vivi") is None


def test_paired_difference_uses_common_steps_only():
    base = ([1, 2, 3, 4], [300.0, 310.0, 320.0, 330.0])
    tratt = ([2, 3, 5], [320.0, 320.0, 400.0])
    assert differenza_appaiata(base, tratt) == ([2, 3], [10.0, 0.0])


def test_series_reads_the_requested_column(tmp_path):
    b = _braccio(tmp_path, "ctrl_none", "none", {3: (1300, 0, 20)}, con_base=False)
    passi, valori = serie(b / "none_seed3", "occupied_cells")
    assert passi == list(range(1, 11)) and valori[-1] == 21.0
    assert serie(tmp_path / "manca") == ([], [])


def test_summary_reports_signs_per_seed_not_a_pooled_mean(tmp_path):
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 10), 4: (1000, 100, 10), 5: (1000, 100, 10)})
    c1 = _braccio(tmp_path, "c1", "llm", {3: (1100, 100, 12), 4: (900, 100, 8), 5: (1000, 100, 10)}, con_base=False)
    r = riepilogo([carica_mondo("w", base, [c1], raggio=2)])[0]
    assert r["segni_vivi"] == "+-0" and r["concordi_vivi"] == 1
    assert r["effetto_vivi"] == 0.0 and r["effetto_celle"] == 0.0
    assert r["morti_per_nascita_baseline"] == pytest.approx(100 / 800, abs=1e-3)


def test_discovery_prefers_v2_and_orders_by_difficulty(tmp_path):
    radice = tmp_path / "mondi"
    facile = radice / "ricco"
    _braccio(facile, "ctrl_none", "none", {3: (1000, 0, 1)}, scarsita=1.0, profilo="ricco")
    _braccio(facile, "llm_completo_amm", "llm", {3: (1000, 0, 1)}, con_base=False)
    duro = radice / "scarso"
    _braccio(duro, "ctrl_none", "none", {3: (800, 0, 1)}, scarsita=0.5, profilo="scarso")
    _braccio(duro, "llm_completo_amm", "llm", {3: (1, 0, 1)}, con_base=False)          # difetto del parser
    _braccio(duro, "llm_completo_amm_v2", "llm", {3: (900, 0, 1)}, con_base=False)     # corretta
    _braccio(radice / "senza_coppia", "ctrl_none", "none", {3: (1, 0, 1)})
    nessun_rif = ("x", tmp_path / "assente", [tmp_path / "assente2"])
    mondi = scopri_mondi(radice, riferimento=nessun_rif, raggio=2)
    assert [m.nome for m in mondi] == ["ricco", "scarso"]
    assert mondi[0].D < mondi[1].D
    assert mondi[1].coppia[3][0].vivi == 900


def test_a_world_keeps_the_replicas_of_its_pair(tmp_path):
    """Le ripetizioni sui mondi non di riferimento devono entrare nel confronto."""
    from scripts.figure_confronto import scopri_mondi, scarto_repliche
    radice = tmp_path / "mondi"
    mondo = radice / "ice_rich"
    _braccio(mondo, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1200, 200, 60)})
    _braccio(mondo, "llm_completo_amm", "llm", {3: (1100, 90, 55), 4: (1150, 210, 58)}, con_base=False)
    _braccio(mondo, "llm_completo_amm_rep2", "llm", {3: (1300, 80, 70), 4: (1000, 210, 40)}, con_base=False)
    base = _braccio(tmp_path, "rif_base", "none", {3: (1000, 100, 50)})
    rif = _braccio(tmp_path, "rif", "llm", {3: (1100, 90, 55)}, con_base=False)
    mondi = scopri_mondi(radice, riferimento=("balanced", base, [rif]), raggio=2)
    ice = next(m for m in mondi if m.nome == "ice_rich")
    assert [len(ice.coppia[s]) for s in ice.semi] == [2, 2]
    assert scarto_repliche(ice) == 175.0                      # (200 + 150) / 2


def test_a_series_shares_the_reference_baseline_and_keeps_the_declared_order(tmp_path):
    from scripts.figure_confronto import scopri_serie
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1200, 200, 60)})
    rif = _braccio(tmp_path, "rif", "llm", {3: (1100, 90, 55), 4: (1150, 210, 58)}, con_base=False)
    radice = tmp_path / "modelli"
    _braccio(radice / "qwen38_27b", "llm_completo_amm", "llm", {3: (900, 90, 40), 4: (950, 210, 45)}, con_base=False)
    _braccio(radice / "gptoss120b", "llm_completo_amm", "llm", {3: (1300, 80, 30)}, con_base=False)
    (radice / "vuota").mkdir()
    serie = scopri_serie(radice, "gptoss20b", riferimento=("balanced", base, [rif]), raggio=2)
    assert [m.nome for m in serie] == ["gptoss20b", "gptoss120b", "qwen38_27b"]   # ORDINE_SERIE, non alfabetico
    assert serie[1].semi == [3] and serie[2].semi == [3, 4]
    assert serie[2].baseline[3].vivi == 1000                                        # stessa baseline del riferimento
    assert serie[1].effetti("celle") == {3: [-20]}


def test_summary_averages_each_seed_before_averaging_the_seeds(tmp_path):
    """Un seme ripetuto non vale tre semi: le medie della coppia pesano per seme, come gli effetti."""
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1000, 100, 50)})
    c1 = _braccio(tmp_path, "c1", "llm", {3: (1000, 100, 100), 4: (1600, 400, 400)}, con_base=False)
    c2 = _braccio(tmp_path, "c2", "llm", {4: (1000, 100, 100)}, con_base=False)
    c3 = _braccio(tmp_path, "c3", "llm", {4: (1000, 100, 100)}, con_base=False)
    m = carica_mondo("balanced", base, [c1, c2, c3], raggio=2)
    r = riepilogo([m])[0]
    assert r["vivi_coppia"] == 1100.0           # (1000 + 1200) / 2, non (1000+1600+1000+1000)/4 = 1150
    assert r["celle_coppia"] == 150.0           # (100 + 200) / 2, non (100+400+100+100)/4 = 175
    assert r["effetto_vivi"] == 100.0           # le medie e gli effetti pesano allo stesso modo
    assert r["effetto_celle"] == 100.0


def test_a_series_keeps_the_replicas_of_its_arm(tmp_path):
    """Il seme 4 di «nomi veri» e' stato ripetuto tre volte: le repliche misurano lo scarto."""
    from scripts.figure_confronto import scopri_serie, scarto_repliche
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1200, 200, 60)})
    rif = _braccio(tmp_path, "rif", "llm", {3: (1100, 90, 55), 4: (1150, 210, 58)}, con_base=False)
    radice = tmp_path / "livelli"
    _braccio(radice / "nomi_veri", "llm_nomi_veri_amm", "llm", {3: (900, 90, 40), 4: (1500, 210, 45)}, con_base=False)
    _braccio(radice / "nomi_veri", "llm_nomi_veri_amm_rep2", "llm", {4: (1100, 210, 45)}, con_base=False)
    _braccio(radice / "nomi_veri", "llm_nomi_veri_amm_rep3", "llm", {4: (1300, 210, 45)}, con_base=False)
    serie = scopri_serie(radice, "completo", riferimento=("balanced", base, [rif]), raggio=2)
    nomi_veri = next(m for m in serie if m.nome == "nomi_veri")
    assert len(nomi_veri.coppia[4]) == 3 and len(nomi_veri.coppia[3]) == 1
    assert scarto_repliche(nomi_veri) == 400.0                       # 1500 - 1100, sul solo seme ripetuto


def test_a_series_accepts_arms_placed_directly_under_the_root(tmp_path):
    """runs/governo/<braccio>/results.jsonl: il braccio e' la cartella stessa, non una sottocartella."""
    from scripts.figure_confronto import scopri_serie
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50)})
    rif = _braccio(tmp_path, "rif", "llm", {3: (1100, 90, 55)}, con_base=False)
    radice = tmp_path / "governo"
    _braccio(radice, "amm_soli", "none", {3: (980, 120, 49)}, con_base=False)
    _braccio(radice, "llm_completo", "llm", {3: (1050, 95, 52)}, con_base=False)
    serie = scopri_serie(radice, "coppia", riferimento=("balanced", base, [rif]), raggio=2)
    assert [m.nome for m in serie] == ["llm_completo", "coppia", "amm_soli"]
    assert serie[0].coppia[3][0].vivi == 1050 and serie[2].effetti("vivi") == {3: [-20]}


def test_figures_are_written_as_pdf_and_png(tmp_path):
    base = _braccio(tmp_path, "ctrl_none", "none", {3: (1000, 100, 50), 4: (1200, 200, 60)})
    c1 = _braccio(tmp_path, "c1", "llm", {3: (1100, 90, 55), 4: (1150, 210, 58)}, con_base=False)
    c2 = _braccio(tmp_path, "c2", "llm", {3: (1300, 80, 70), 4: (1000, 210, 40)}, con_base=False)
    m = carica_mondo("balanced", base, [c1, c2], raggio=2)
    out = tmp_path / "fig"
    for f in (disegna_esiti([m], out / "a.pdf"), disegna_traiettorie([m], out / "b.pdf"),
              disegna_effetto_vs_difficolta([m], out / "c.pdf")):
        assert f.exists() and f.stat().st_size > 0
        assert f.with_suffix(".png").exists()
