"""Registro di spesa per il provider TypeSafe: persistente e con tetto.

Il tetto si controlla PRIMA della richiesta, sulla stima dei token, e la spesa
si registra DOPO, sui token dichiarati dalla risposta. Il file sopravvive fra
processi: il tetto vale per l'insieme delle run, non per la singola.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path


#: Un lock per file, condiviso fra tutte le istanze del processo: governatore,
#: amministratori e strato degli agenti possono costruire ciascuno il proprio
#: registro sullo stesso file, e un lock per istanza perderebbe aggiornamenti
#: (e su Windows `os.replace` concorrente sullo stesso file solleva).
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


@contextlib.contextmanager
def _lock_fra_processi(path: Path, timeout_s: float = 30.0):
    """Blocco esclusivo su `<registro>.lock`, valido fra processi.

    **Perche' (campagna v4, 2026-09-24).** Con quattro run in parallelo lo 0,34%
    delle decisioni Jev e' finito in `ledger_error`: il lock di `_lock_for` vale
    dentro un processo, e fra processi il file (e il suo `.tmp`) veniva letto e
    sostituito insieme, con `PermissionError` su Windows e aggiornamenti persi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path) + ".lock", "a+b")
    try:
        scadenza = time.monotonic() + timeout_s
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > scadenza:
                        raise
                    time.sleep(0.005)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _con_ripetizioni(operazione, tentativi: int = 50, attesa_s: float = 0.02):
    """Ripete un'operazione sul file finche' non smette di dare `PermissionError`.

    Anche col blocco fra processi restavano ~11 errori su ~10.000 decisioni
    (2026-09-24): su Windows un antivirus o l'indicizzatore tiene aperto per un
    attimo il file appena scritto. E' un errore transitorio; dopo ~1 s si
    rinuncia e l'errore risale come prima.
    """
    for tentativo in range(tentativi):
        try:
            return operazione()
        except PermissionError:
            if tentativo == tentativi - 1:
                raise
            time.sleep(attesa_s)


class BudgetExceeded(RuntimeError):
    """La richiesta porterebbe la spesa oltre il tetto dichiarato."""


class SpendingLedger:
    def __init__(
        self,
        path: str | Path,
        cap_usd: float,
        usd_per_million_input: float = 0.042,
    ) -> None:
        if not cap_usd > 0.0:
            raise ValueError("cap_usd must be positive")
        if not usd_per_million_input > 0.0:
            raise ValueError("usd_per_million_input must be positive")
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self.usd_per_million_input = float(usd_per_million_input)
        self._lock = _lock_for(self.path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"input_tokens": 0, "requests": 0}
        raw = json.loads(_con_ripetizioni(lambda: self.path.read_text(encoding="utf-8")))
        return {
            "input_tokens": int(raw.get("input_tokens", 0)),
            "requests": int(raw.get("requests", 0)),
        }

    def _cost(self, tokens: int) -> float:
        return tokens * self.usd_per_million_input / 1_000_000.0

    def spent_usd(self) -> float:
        with self._lock, _lock_fra_processi(self.path):
            return self._cost(self._read()["input_tokens"])

    def authorize(self, estimated_input_tokens: int) -> None:
        with self._lock, _lock_fra_processi(self.path):
            spent = self._cost(self._read()["input_tokens"])
            if spent + self._cost(max(0, int(estimated_input_tokens))) > self.cap_usd:
                raise BudgetExceeded(
                    f"spent {spent:.6f} USD, cap {self.cap_usd:.6f} USD"
                )

    def record(self, input_tokens: int) -> None:
        with self._lock, _lock_fra_processi(self.path):
            state = self._read()
            state["input_tokens"] += max(0, int(input_tokens))
            state["requests"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Un temporaneo per processo: anche col blocco, un processo
            # interrotto a meta' non deve lasciare il `.tmp` a un altro.
            temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
            _con_ripetizioni(lambda: temporary.write_text(json.dumps(state), encoding="utf-8"))
            _con_ripetizioni(lambda: os.replace(temporary, self.path))

    def summary(self) -> dict:
        with self._lock, _lock_fra_processi(self.path):
            state = self._read()
        return {
            **state,
            "spent_usd": round(self._cost(state["input_tokens"]), 6),
            "cap_usd": self.cap_usd,
            "usd_per_million_input": self.usd_per_million_input,
        }
