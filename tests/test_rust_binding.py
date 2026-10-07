from __future__ import annotations

"""Test del binding nativo (Task 1 del piano di migrazione Rust).

I test che richiedono il modulo compilato si saltano quando non e' installato:
un checkout pulito senza toolchain Rust deve restare completamente verde, perche'
il piano impone che Python resti autorevole e autosufficiente per tutta la
migrazione. Cio' che NON si salta e' il comportamento del wrapper quando il
modulo manca: e' proprio quello il percorso che ogni utente senza Rust percorre.
"""

import pytest

from src.core import native

requires_native = pytest.mark.skipif(
    not native.is_available(),
    reason=f"kernel nativo non disponibile: {native.unavailable_reason()}",
)


def test_availability_and_reason_are_consistent() -> None:
    """Disponibile senza motivo di errore, oppure indisponibile con un motivo."""
    if native.is_available():
        assert native.unavailable_reason() is None
    else:
        assert native.unavailable_reason()


def test_describe_always_returns_a_manifest_row() -> None:
    """Il manifest della run deve poter descrivere anche l'assenza del backend."""
    described = native.describe()
    assert isinstance(described["available"], bool)
    if not described["available"]:
        assert described["reason"]
        assert described["expected_schema_version"] == native.SCHEMA_VERSION


def test_require_raises_actionable_error_when_missing() -> None:
    if native.is_available():
        pytest.skip("il kernel nativo e' installato: percorso non esercitabile qui")
    with pytest.raises(native.NativeKernelUnavailable, match="non disponibile"):
        native.require()


@requires_native
def test_native_module_reports_matching_schema() -> None:
    """Il disallineamento di schema e' la divergenza che va scoperta all'avvio.

    Se i due lati non concordano sulla versione dello schema, gli stessi byte
    verrebbero letti con indici di risorsa o codici azione diversi: una classe di
    errore che si manifesterebbe come divergenza numerica a centinaia di step di
    distanza dalla causa.
    """
    module = native.require()
    assert module.schema_version() == native.SCHEMA_VERSION
    assert module.SCHEMA_VERSION == native.SCHEMA_VERSION


@requires_native
def test_native_version_string_is_informative() -> None:
    module = native.require()
    reported = module.version()
    assert "mars_core" in reported
    assert module.KERNEL_VERSION in reported


@requires_native
def test_describe_reports_kernel_identity() -> None:
    described = native.describe()
    assert described["available"] is True
    assert described["schema_version"] == native.SCHEMA_VERSION
    assert described["kernel_version"]
    assert described["description"]


def test_crossing_probe_accepts_the_columns_the_decision_loop_reads():
    """Task A del port: il confine deve reggere il carico vero, non un campione.

    La sonda non calcola nulla di proprio, quindi il test non verifica un
    risultato di simulazione: verifica che la firma accetti esattamente le forme
    e i dtype che il ciclo decisionale legge davvero. Se un dtype cambia lato
    Python, questo test fallisce qui invece che con numeri sbagliati a valle.
    """
    import numpy as np

    mars_core_py = pytest.importorskip("mars_core_py")

    n, height, width, n_actions, n_pillars = 8, 5, 7, 27, 6
    n_res, n_struct = 11, 10
    total = mars_core_py.crossing_probe(
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int16),
        np.zeros(n, dtype=np.int16),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros((n, n_res), dtype=np.float64),
        np.zeros((n, n_pillars), dtype=np.float64),
        np.zeros((n, n_pillars), dtype=np.float64),
        np.zeros((n, n_actions), dtype=np.bool_),
        np.zeros((n, n_actions), dtype=np.float64),
        np.zeros((n, n_actions), dtype=np.int32),
        np.zeros((height, width, n_struct), dtype=np.int16),
        np.zeros((height, width, n_struct), dtype=np.float64),
        np.zeros((height, width, n_res), dtype=np.float64),
        np.zeros((height, width), dtype=np.int16),
    )
    # Somma delle prime dimensioni: sei array per-agente piu' quattro di cella.
    assert total == n * 7 + height * 4


def test_crossing_probe_rejects_mismatched_agent_columns():
    """Colonne di lunghezza diversa sono un errore di schema, non di valore."""
    import numpy as np

    mars_core_py = pytest.importorskip("mars_core_py")

    with pytest.raises(ValueError):
        mars_core_py.crossing_probe(
            np.arange(4, dtype=np.int64),
            np.zeros(3, dtype=np.int16),
            np.zeros(4, dtype=np.int16),
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros((4, 11), dtype=np.float64),
            np.zeros((4, 6), dtype=np.float64),
            np.zeros((4, 6), dtype=np.float64),
            np.zeros((4, 27), dtype=np.bool_),
            np.zeros((4, 27), dtype=np.float64),
            np.zeros((4, 27), dtype=np.int32),
            np.zeros((2, 2, 10), dtype=np.int16),
            np.zeros((2, 2, 10), dtype=np.float64),
            np.zeros((2, 2, 11), dtype=np.float64),
            np.zeros((2, 2), dtype=np.int16),
        )
