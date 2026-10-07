# -*- coding: utf-8 -*-
"""L'uscita forzata del runner vale SOLO dopo il Mars Climate Database.

Una run con `--strato-ambientale` non riesce a terminare: la libreria Fortran
del MCD, compilata con MinGW, blocca il distacco delle DLL, e nemmeno
`os._exit` basta perche' passa anch'esso da li'. Il runner esce percio' con
`TerminateProcess`.

E' un'uscita che salta la finalizzazione, quindi va tenuta stretta: se
diventasse incondizionata, ogni futuro `atexit` del progetto smetterebbe di
scattare senza che nessuno se ne accorga. Questo test e' il guardiano di quella
condizione, e costa niente perche' non esegue nessuna simulazione.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _runner():
    modulo = importlib.import_module("scripts.run_governor_experiment")
    return importlib.reload(modulo)


def test_senza_mcd_l_uscita_e_ordinata():
    runner = _runner()
    assert runner._MCD_TOCCATO is False, "il modulo appena caricato non ha toccato il MCD"
    with pytest.raises(SystemExit) as uscita:
        runner._esci(0)
    assert uscita.value.code == 0


def test_senza_mcd_il_codice_di_errore_arriva_intatto():
    runner = _runner()
    with pytest.raises(SystemExit) as uscita:
        runner._esci(3)
    assert uscita.value.code == 3


def test_con_mcd_l_uscita_non_passa_da_systemexit(monkeypatch):
    """Con il MCD toccato, `_esci` non solleva: termina il processo.

    Qui non si puo' lasciarlo terminare davvero, quindi si sostituiscono le due
    vie d'uscita e si verifica che ne venga presa una. Il punto del test e'
    che NON venga sollevato `SystemExit`: sarebbe la finalizzazione ordinata,
    cioe' esattamente il percorso che si blocca.
    """
    runner = _runner()
    monkeypatch.setattr(runner, "_MCD_TOCCATO", True)
    chiamate = []
    monkeypatch.setattr(runner.os, "_exit", lambda codice: chiamate.append(("_exit", codice)))
    if sys.platform == "win32":
        import ctypes

        monkeypatch.setattr(
            ctypes.windll.kernel32, "TerminateProcess",
            lambda handle, codice: chiamate.append(("terminate", codice)) or 1,
        )
    runner._esci(0)
    assert chiamate, "nessuna via d'uscita e' stata presa"
