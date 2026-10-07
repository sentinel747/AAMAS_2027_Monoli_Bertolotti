# -*- coding: utf-8 -*-
"""Che cosa rende una run non comparabile, e che cosa invece si limita a dire.

**Perche' questi test esistono.** La differenza fra le due liste e' una scelta di
metodo, non un dettaglio: sposta quali run entrano nell'analisi. Il 15 settembre
le risposte non interpretabili degli amministratori sono passate da causa di
esclusione a esito da riportare, e senza un test che lo fissi la distinzione
tornerebbe indietro alla prima persona che, leggendo il codice, trovasse naturale
rimettere i due contatori nello stesso ciclo --- che e' esattamente come ci
erano finiti.

Il criterio: si scarta per i guasti del banco di prova, si riporta cio' che ha
fatto il modello.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cancello = importlib.import_module("scripts.qualita_run")


def _run(tmp_path: Path, **campi) -> Path:
    """Una cartella con dentro una sola run, descritta dai campi passati."""
    riga = {
        "seed": 3,
        "governor_misses": 0,
        "api": {"failed_calls": 0},
        "amministrazione": {
            "enabled": True,
            "provider_failures": 0,
            "malformed": 0,
            "interventions": 200,
            "abstentions": 450,
        },
    }
    for chiave, valore in campi.items():
        if isinstance(valore, dict) and isinstance(riga.get(chiave), dict):
            riga[chiave] = {**riga[chiave], **valore}
        else:
            riga[chiave] = valore
    io.open(tmp_path / "results.jsonl", "w", encoding="utf-8").write(
        json.dumps(riga) + "\n")
    return tmp_path


def test_una_run_pulita_passa(tmp_path):
    assert cancello.guasti(_run(tmp_path)) == []


def test_le_tornate_perse_del_governatore_bocciano(tmp_path):
    problemi = cancello.guasti(_run(tmp_path, governor_misses=7))
    assert len(problemi) == 1
    assert "tornate del governatore perse" in problemi[0]


def test_le_chiamate_fallite_bocciano(tmp_path):
    problemi = cancello.guasti(_run(tmp_path, api={"failed_calls": 2}))
    assert len(problemi) == 1
    assert "chiamate al modello fallite" in problemi[0]


def test_i_guasti_del_fornitore_agli_amministratori_bocciano(tmp_path):
    problemi = cancello.guasti(_run(tmp_path, amministrazione={"provider_failures": 1}))
    assert len(problemi) == 1
    assert "guasti del fornitore" in problemi[0]


def test_le_risposte_non_interpretabili_NON_bocciano(tmp_path):
    """Il punto di tutto il file.

    Una riscrittura che il parser non capisce e' una cosa che il modello ha
    fatto, non un guasto nostro. Scartare la run per questo toglierebbe il
    fenomeno insieme al rumore --- e in modo selettivo, perche' il modello che
    ne produce di piu' verrebbe giudicato sui suoi giorni migliori.
    """
    assert cancello.guasti(_run(tmp_path, amministrazione={"malformed": 3})) == []


def test_le_risposte_non_interpretabili_vengono_pero_dette(tmp_path):
    """Non bocciare non vuol dire tacere.

    Una riscrittura non capita lascia la cella al governo, quindi somiglia a
    un'accettazione: se nessuno la nomina, l'analisi legge obbedienza dove c'e'
    stato un tentativo fallito.
    """
    note = cancello.malformate(_run(tmp_path, amministrazione={"malformed": 3}))
    assert len(note) == 1
    assert "3 risposte" in note[0]
    # Il denominatore serve a dare la misura: tre su seicentocinquanta non e'
    # la stessa cosa che tre su cinque.
    assert "650 richieste" in note[0]


def test_niente_nota_se_non_ce_ne_sono_state(tmp_path):
    assert cancello.malformate(_run(tmp_path)) == []


def test_un_contatore_assente_non_vale_come_zero(tmp_path):
    """Ci sono stati amministratori ma il contatore non c'e': non e' «nessun guasto».

    Sulle campagne precedenti all'11 settembre il contatore non veniva scritto,
    e leggerne l'assenza come zero faceva certificare alla coda una condizione
    che non aveva modo di controllare.
    """
    senza = {"enabled": True, "malformed": 0, "interventions": 200, "abstentions": 450}
    riga = _run(tmp_path)
    dati = json.loads(io.open(riga / "results.jsonl", encoding="utf-8").read())
    dati["amministrazione"] = senza
    io.open(riga / "results.jsonl", "w", encoding="utf-8").write(json.dumps(dati) + "\n")
    problemi = cancello.guasti(riga)
    assert len(problemi) == 1
    assert "provider_failures" in problemi[0]


def test_una_run_che_non_c_e_non_e_un_guasto(tmp_path):
    """Non ancora eseguita e' diverso da eseguita male, e la coda conta su questo."""
    assert cancello.guasti(tmp_path) == []
    assert cancello.malformate(tmp_path) == []


def _uscita(cartella: Path, seme=None) -> int:
    """Il codice d'uscita del cancello chiamato da riga di comando."""
    argomenti = [sys.executable, str(Path(cancello.__file__)), str(cartella)]
    if seme is not None:
        argomenti += ['--seme', str(seme)]
    return subprocess.run(argomenti, capture_output=True).returncode


def test_un_seme_chiesto_per_nome_e_assente_e_un_guasto(tmp_path):
    """Il difetto che ha aperto otto buchi in una campagna.

    La coda esegue una run e poi chiede al cancello se quel seme va bene. La
    notte in cui il fornitore e' caduto, la run non arrivava a scrivere nulla,
    il cancello non trovava righe da controllare e rispondeva «in ordine»: la
    coda contava la run come fatta e tirava dritto. Nessun secondo tentativo e
    nessuna segnalazione, per otto run --- quattro delle quali nel braccio che
    misura il rumore, cioe' quello senza cui la campagna non si legge.
    """
    _run(tmp_path, seed=3)
    assert _uscita(tmp_path, seme=3) == 0
    assert _uscita(tmp_path, seme=4) == 1, 'un seme assente deve far fallire'


def test_senza_seme_una_cartella_incompleta_resta_in_ordine(tmp_path):
    """La distinzione dipende da chi domanda.

    Su una cartella, «manca il seme 4» significa «non ancora eseguito», e
    chiamarlo guasto fermerebbe una campagna appena partita. E' solo la domanda
    puntuale, subito dopo un tentativo, che rende l'assenza un guasto.
    """
    _run(tmp_path, seed=3)
    assert _uscita(tmp_path) == 0
