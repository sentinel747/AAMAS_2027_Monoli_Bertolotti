# -*- coding: utf-8 -*-
"""Cio' che serve per VEDERE i distretti, in diretta e nel replay.

Una run decentrata si legge sulla mappa solo se ogni cella dice a quale
amministratore appartiene e che cosa quell'amministratore ha deciso. Il
registro per tornata porta gia' le celle di ogni distretto e l'esito
(accetta / riscrive / a vuoto); mancava la cella madre, che per progetto sta
fuori da ogni distretto e va disegnata come tale, e mancava che lo snapshot
del mondo portasse il blocco --- il frontend live lo riceve nello stato, il
replay no.

Nessuno snapshot completo viene reso obbligatorio: su questa griglia pesa 73
MB, e il replay dell'analisi ricostruisce i distretti dal registro degli
amministratori, che ogni run scrive comunque.
"""

import numpy as np

from src.governors.administration import Amministrazione
from src.governors.districts import Distretti
from src.governors.policy import Bounds

BOUNDS = Bounds(0.25, 4.0)


class _Accetta:
    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": {"accept": True, "rationale": "va bene"}}


def test_il_riassunto_porta_la_cella_madre():
    distretti = Distretti()
    distretti.imposta_madre((2, 2))
    amministrazione = Amministrazione(lambda d: _Accetta(), distretti, BOUNDS)
    assert amministrazione.riassunto()["mother_cell"] == [2, 2]


def test_senza_madre_dichiarata_il_riassunto_dice_none():
    amministrazione = Amministrazione(lambda d: _Accetta(), Distretti(), BOUNDS)
    assert amministrazione.riassunto()["mother_cell"] is None


def test_lo_snapshot_del_mondo_porta_lo_strato_amministrativo(tmp_path):
    """Il replay per snapshot deve poter colorare i distretti senza il registro."""
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(60, 4, 0)
    config["governors"] = {
        "arm": "scripted",
        "cadence_steps": 2,
        "administrators": {"enabled": True, "arm": "none", "cells_per_district": 3},
    }
    runner = AgentCoupledRunner(config)
    runner.run(days=4, output_dir=tmp_path / "run")
    payload = runner._world_snapshot_payload(4, 4)
    strato = payload["administrators"]
    assert strato["enabled"] is True
    assert "last_round" in strato
    assert "mother_cell" in strato


def test_senza_amministratori_lo_snapshot_non_inventa_lo_strato(tmp_path):
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    runner = AgentCoupledRunner(realistic_config(60, 2, 0))
    runner.run(days=2, output_dir=tmp_path / "run")
    assert runner._world_snapshot_payload(2, 2).get("administrators") is None


def test_gli_amministratori_senza_governo_accendono_i_contatori_di_scatto(tmp_path):
    """Il braccio «amministratori soli» deve contare le regole che applica.

    Le catene dei distretti passano da `apply_layered` anche senza governatore,
    ma i contatori erano accesi solo quando un governatore esisteva: il braccio
    riportava zero regole applicate pur avendo riscritto, indistinguibile da un
    braccio inerte. Che `apply_layered` sappia contare e' gia' verificato in
    `test_distretti.py`; qui si verifica il cablaggio, cioe' che il dizionario
    esista quando lo strato amministrativo e' attivo.
    """
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = realistic_config(60, 4, 0)
    config["governors"] = {
        "arm": "none",
        "cadence_steps": 2,
        "administrators": {
            "enabled": True,
            "arm": "scripted",
            "follow_governor_arm": False,      # senza governo non c'e' un braccio da seguire
            "cells_per_district": 3,
        },
    }
    out = tmp_path / "run"
    runner = AgentCoupledRunner(config)
    runner.run(days=6, output_dir=out)

    assert runner.core.governor_policy_hits is not None, (
        "senza questo dizionario il kernel passa hits=None e non conta niente"
    )
    assert (out / "governor_policy_hits.json").exists()


def test_senza_governo_ne_amministratori_i_contatori_restano_spenti(tmp_path):
    """Il cancello della parita': la baseline non deve nemmeno aprire il file."""
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    out = tmp_path / "run"
    runner = AgentCoupledRunner(realistic_config(60, 2, 0))
    runner.run(days=4, output_dir=out)
    assert runner.core.governor_policy_hits is None
    assert not (out / "governor_policy_hits.json").exists()
