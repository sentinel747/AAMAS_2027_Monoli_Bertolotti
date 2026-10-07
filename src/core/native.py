from __future__ import annotations

"""Accesso opzionale al kernel nativo Rust, senza renderlo obbligatorio.

Il piano di migrazione impone che Python resti autorevole e completamente
funzionante finche' Rust non supera i gate. Questo modulo e' il punto unico in cui
il resto del codice puo' chiedere "il kernel nativo e' disponibile?" senza che un
checkout privo del modulo compilato smetta di funzionare.

Regole:

- l'import fallito **non** e' un errore: e' lo stato normale di un checkout in cui
  nessuno ha ancora eseguito ``python scripts/build_rust.py``;
- un modulo presente ma con schema incompatibile **e'** un errore, e va segnalato
  forte: significa che i buffer scambiati verrebbero interpretati con indici di
  risorsa o codici azione diversi dai due lati, cioe' esattamente la classe di
  divergenza silenziosa che il piano vuole rendere impossibile.
"""

from typing import Any

# Versione dello schema di stato condiviso. Deve coincidere con
# `mars_core::SCHEMA_VERSION` (rust/mars_core/src/lib.rs). Al Task 2 questo valore
# verra' generato dallo schema versionato comune invece di essere scritto due
# volte; qui e' duplicato consapevolmente perche' il Task 1 non introduce ancora
# alcun scambio di buffer da proteggere.
SCHEMA_VERSION = 1


class NativeKernelUnavailable(RuntimeError):
    """Il kernel nativo e' stato richiesto ma non e' utilizzabile."""


def _load() -> tuple[Any | None, str | None]:
    try:
        import mars_core_py  # type: ignore[import-not-found]
    except ImportError as exc:
        return None, f"modulo non installato ({exc})"

    native_schema = getattr(mars_core_py, "SCHEMA_VERSION", None)
    if native_schema != SCHEMA_VERSION:
        return None, (
            f"schema incompatibile: Python attende v{SCHEMA_VERSION}, "
            f"il modulo nativo dichiara v{native_schema}. "
            "Ricostruire con `python scripts/build_rust.py`."
        )
    return mars_core_py, None


_MODULE, _UNAVAILABLE_REASON = _load()


def is_available() -> bool:
    """``True`` se il kernel nativo e' importabile e allineato allo schema."""
    return _MODULE is not None


def unavailable_reason() -> str | None:
    """Motivo dell'indisponibilita', o ``None`` se il kernel e' disponibile."""
    return _UNAVAILABLE_REASON


def require() -> Any:
    """Restituisce il modulo nativo o solleva con un messaggio azionabile."""
    if _MODULE is None:
        raise NativeKernelUnavailable(
            f"kernel nativo non disponibile: {_UNAVAILABLE_REASON}"
        )
    return _MODULE


def describe() -> dict[str, Any]:
    """Riga di manifest sul backend nativo, per gli artefatti di run.

    Il piano richiede che backend, schema hash e versione del kernel finiscano nel
    manifest della run: senza, un risultato archiviato non dice con quale motore
    e' stato prodotto e non e' riproducibile.
    """
    if _MODULE is None:
        return {
            "available": False,
            "reason": _UNAVAILABLE_REASON,
            "expected_schema_version": SCHEMA_VERSION,
        }
    return {
        "available": True,
        "schema_version": int(_MODULE.SCHEMA_VERSION),
        "kernel_version": str(_MODULE.KERNEL_VERSION),
        "description": str(_MODULE.version()),
    }
