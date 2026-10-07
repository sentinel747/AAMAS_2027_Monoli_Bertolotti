# -*- coding: utf-8 -*-
"""Uno snapshot che si possa davvero salvare a ogni tornata.

**Il numero.** Sulla griglia della campagna (360x180) uno snapshot completo del
mondo pesa 73 MB, perche' elenca tutte le 64.800 celle anche dove non c'e' e
non c'e' mai stato nessuno. Quaranta snapshot per run sono tre gigabyte, e la
richiesta e' di poter rivedere OGNI run nell'analysis frontend, distretti
compresi. La versione compatta tiene le sole celle che raccontano qualcosa
--- occupate, con strutture, o esplorate --- e dichiara di esserlo, cosi' che
un lettore sappia che le celle assenti sono vuote e non mancanti. Larghezza e
altezza restano, perche' il renderer ne ha bisogno per la scala.
"""

import json

from src.core.shell_common import compatta_snapshot


def _snapshot():
    return {
        "width": 4,
        "height": 3,
        "cells": [
            {"x": 0, "y": 0, "structures": [], "agents_present": [], "explored": False},
            {"x": 1, "y": 0, "structures": [{"type": "habitat"}], "agents_present": [], "explored": False},
            {"x": 2, "y": 0, "structures": [], "agents_present": ["agent_1"], "explored": False},
            {"x": 3, "y": 0, "structures": [], "agents_present": [], "explored": True},
            {"x": 0, "y": 1, "structures": [], "agents_present": [], "explored": False},
        ],
        "agents": [{"agent_id": "agent_1"}],
    }


def test_restano_solo_le_celle_che_raccontano_qualcosa():
    compatto = compatta_snapshot(_snapshot())
    assert [(c["x"], c["y"]) for c in compatto["cells"]] == [(1, 0), (2, 0), (3, 0)]


def test_il_compatto_si_dichiara_e_conserva_le_dimensioni():
    compatto = compatta_snapshot(_snapshot())
    assert compatto["compact"] is True
    assert compatto["width"] == 4 and compatto["height"] == 3
    assert compatto["agents"] == [{"agent_id": "agent_1"}]


def test_uno_snapshot_senza_celle_non_esplode():
    compatto = compatta_snapshot({"width": 1, "height": 1})
    assert compatto["cells"] == [] and compatto["compact"] is True


def test_la_shell_scrive_snapshot_compatti_quando_richiesto(tmp_path):
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(40, 4, 0)
    config.setdefault("headless", {})["snapshot_interval"] = 2
    config["headless"]["snapshot_compact"] = True
    AgentCoupledRunner(config).run(days=4, output_dir=tmp_path / "run")

    file = sorted((tmp_path / "run" / "world_snapshots").glob("*.json"))
    assert file, "con snapshot_interval > 0 la cartella si riempie"
    snapshot = json.loads(file[-1].read_text(encoding="utf-8"))
    assert snapshot["compact"] is True
    assert 0 < len(snapshot["cells"]) < snapshot["width"] * snapshot["height"]
    assert all(c["agents_present"] or c["structures"] or c["explored"] for c in snapshot["cells"])
