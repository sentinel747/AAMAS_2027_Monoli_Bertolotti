from __future__ import annotations

"""Selezione del backend di raggiungibilita' (Task 3 della migrazione Rust).

Il dispatcher decide quale formulazione della distanza di supporto viene usata dal
percorso decisionale. Tre proprieta' vanno protette:

1. il default e' ``scalar``, perche' e' la formulazione **misurata piu' veloce**
   end-to-end (i backend a campo vincono il microbenchmark ma regrediscono nel
   simulatore) e perche' un checkout senza toolchain Rust deve restare funzionante;
2. un valore non valido fallisce subito e con un messaggio utile, non piu' avanti
   e in modo silenzioso;
3. il backend viene risolto **una volta sola**: rileggere l'ambiente sul percorso
   caldo reintrodurrebbe il costo per-chiamata che la modifica elimina.
"""

import pytest

from src.core import native, reachability_batch


@pytest.fixture(autouse=True)
def _restore_backend_cache():
    """Isola i test: ognuno riparte da un backend non risolto."""
    reachability_batch.reset_backend_cache()
    yield
    reachability_batch.reset_backend_cache()


def test_default_backend_is_scalar(monkeypatch) -> None:
    monkeypatch.delenv("MARSABM_REACH_BACKEND", raising=False)
    assert reachability_batch.configured_backend() == "scalar"


def test_numpy_backend_is_selectable(monkeypatch) -> None:
    monkeypatch.setenv("MARSABM_REACH_BACKEND", "numpy")
    assert reachability_batch.configured_backend() == "numpy"


def test_backend_name_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("MARSABM_REACH_BACKEND", "  NumPy  ")
    assert reachability_batch.configured_backend() == "numpy"


def test_invalid_backend_fails_fast_and_names_the_alternatives(monkeypatch) -> None:
    monkeypatch.setenv("MARSABM_REACH_BACKEND", "fortran")
    with pytest.raises(ValueError, match="non valido"):
        reachability_batch.configured_backend()


def test_backend_is_resolved_once(monkeypatch) -> None:
    """Il percorso caldo non deve rileggere l'ambiente a ogni chiamata."""
    monkeypatch.setenv("MARSABM_REACH_BACKEND", "numpy")
    assert reachability_batch.configured_backend() == "numpy"

    monkeypatch.setenv("MARSABM_REACH_BACKEND", "scalar")
    assert reachability_batch.configured_backend() == "numpy", (
        "il backend risolto non deve cambiare a meta' run: lascerebbe campi di "
        "distanza calcolati da un backend e consultati da un altro"
    )

    reachability_batch.reset_backend_cache()
    assert reachability_batch.configured_backend() == "scalar"


def test_rust_backend_requires_the_native_kernel(monkeypatch) -> None:
    monkeypatch.setenv("MARSABM_REACH_BACKEND", "rust")
    if native.is_available():
        assert reachability_batch.configured_backend() == "rust"
    else:
        with pytest.raises(native.NativeKernelUnavailable, match="build_rust"):
            reachability_batch.configured_backend()


def test_support_distance_field_rejects_the_scalar_backend() -> None:
    """`scalar` non ha campo precomputato: chiederlo e' un errore di programmazione."""
    with pytest.raises(ValueError, match="senza campo precomputato"):
        reachability_batch.support_distance_field(4, 4, [0], [0], "scalar")
