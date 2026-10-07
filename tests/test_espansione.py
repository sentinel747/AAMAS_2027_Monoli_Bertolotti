"""L'espansione come esito, invece della popolazione.

Trenta run di controllo si dividono in due gruppi che non si sovrappongono:
diciotto con una sola cella insediata e popolazione 306-337, dodici con due o
piu' e popolazione 373-579. La popolazione e' una lettura indiretta di quel
bivio, e la sua mediana — 313 — non e' un valore centrale ma esattamente il
valore di uno dei due stati. Vedi
`docs/benchmarks/2026-08-20-biforcazione-espansione.md`.
"""

import json

from scripts.run_governor_experiment import _expansion


def _scrivi(tmp_path, celle):
    (tmp_path / "cell_infrastructure.json").write_text(
        json.dumps([{"x": i, "y": 0, "population": p} for i, p in enumerate(celle)]),
        encoding="utf-8",
    )
    return tmp_path


def test_the_satellites_below_the_threshold_do_not_count(tmp_path):
    """Il satellite da tre persone e' il caso da NON contare.

    E' proprio quello che una colonia arrestata produce a decine: il piano di
    costruzione fissa gli alloggi sulla popolazione presente, quindi un satellite
    tarato sui propri tre abitanti non ha mai capienza libera per una nascita e
    resta a tre per sempre. Contarlo come insediamento direbbe che la colonia si
    e' espansa proprio quando non ha potuto.
    """
    dati = _expansion(_scrivi(tmp_path, [301, 5, 4, 1, 1, 1]))
    assert dati["celle_totali"] == 6
    assert dati["celle_insediate"] == 1
    assert dati["espansa"] is False
    assert dati["popolazione_seconda_cella"] == 5


def test_a_second_real_settlement_counts_as_expansion(tmp_path):
    dati = _expansion(_scrivi(tmp_path, [297, 153, 57, 24, 7, 3]))
    assert dati["celle_insediate"] == 4
    assert dati["espansa"] is True
    assert dati["popolazione_seconda_cella"] == 153


def test_no_record_is_an_absence_and_not_a_zero(tmp_path):
    """Un artefatto mancante non deve diventare "non si e' espansa".

    Le due cose finiscono nella stessa tabella e la seconda e' un risultato: se
    una run senza registro riportasse `espansa: false`, il conteggio dei
    ribaltamenti includerebbe run che nessuno ha misurato.
    """
    assert _expansion(tmp_path) == {}
    assert _expansion(None) == {}
