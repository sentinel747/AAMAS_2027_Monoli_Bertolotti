import json
import threading

import pytest

from src.semantic_governance.budget import BudgetExceeded, SpendingLedger


def test_new_ledger_has_spent_nothing(tmp_path):
    ledger = SpendingLedger(tmp_path / "l.json", cap_usd=0.5)
    assert ledger.spent_usd() == 0.0


def test_record_persists_across_instances(tmp_path):
    path = tmp_path / "l.json"
    SpendingLedger(path, cap_usd=0.5).record(1_000_000)
    again = SpendingLedger(path, cap_usd=0.5)
    assert again.spent_usd() == pytest.approx(0.042)
    assert json.loads(path.read_text())["input_tokens"] == 1_000_000


def test_authorize_refuses_a_request_that_would_cross_the_cap(tmp_path):
    ledger = SpendingLedger(tmp_path / "l.json", cap_usd=0.05)
    ledger.record(1_000_000)  # 0,042 USD
    ledger.authorize(100_000)  # +0,0042 -> 0,0462: ammesso
    with pytest.raises(BudgetExceeded):
        ledger.authorize(500_000)  # +0,021 -> oltre 0,05


def test_ledger_already_over_cap_refuses_everything(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"input_tokens": 50_000_000, "requests": 1}))
    with pytest.raises(BudgetExceeded):
        SpendingLedger(path, cap_usd=0.5).authorize(1)


def test_cap_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        SpendingLedger(tmp_path / "l.json", cap_usd=0.0)


def test_concurrent_records_are_not_lost(tmp_path):
    ledger = SpendingLedger(tmp_path / "l.json", cap_usd=10.0)
    threads = [threading.Thread(target=ledger.record, args=(1000,)) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.summary()["input_tokens"] == 50_000
    assert ledger.summary()["requests"] == 50


def test_two_ledgers_on_the_same_file_do_not_lose_records(tmp_path):
    # Governatore e strato agenti costruiscono ciascuno il proprio registro
    # sullo stesso file: il lock deve valere per file, non per istanza.
    path = tmp_path / "shared.json"
    first = SpendingLedger(path, cap_usd=10.0)
    second = SpendingLedger(path, cap_usd=10.0)
    threads = [
        threading.Thread(target=(first if i % 2 else second).record, args=(1000,))
        for i in range(400)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert first.summary()["requests"] == 400


def _registra_molte_volte(path: str, volte: int) -> None:
    ledger = SpendingLedger(path, cap_usd=100.0)
    for _ in range(volte):
        ledger.record(10)
        ledger.spent_usd()


def test_processes_sharing_the_file_neither_fail_nor_lose_records(tmp_path):
    """Campagna v4 (2026-09-24): con quattro run in parallelo lo 0,34% delle
    decisioni Jev e' finito in `ledger_error`, perche' il lock era per processo
    e i processi si contendevano lo stesso file (e lo stesso `.tmp`) su Windows.
    Il registro deve reggere processi concorrenti senza errori e senza perdere
    aggiornamenti."""
    import multiprocessing

    path = str(tmp_path / "l.json")
    ctx = multiprocessing.get_context("spawn")
    processi = [ctx.Process(target=_registra_molte_volte, args=(path, 150)) for _ in range(4)]
    for p in processi:
        p.start()
    for p in processi:
        p.join(120)
    assert [p.exitcode for p in processi] == [0, 0, 0, 0]
    stato = json.loads((tmp_path / "l.json").read_text())
    assert stato == {"input_tokens": 4 * 150 * 10, "requests": 4 * 150}


def test_transient_permission_errors_are_retried(tmp_path, monkeypatch):
    """Dopo il blocco fra processi restavano ~11 `ledger_error` su ~10.000
    decisioni (2026-09-24): su Windows un antivirus o l'indicizzatore puo'
    tenere aperto per un attimo il file appena scritto, e `os.replace` o la
    lettura falliscono con `PermissionError`. Errori transitori: si riprova."""
    import os as _os

    import src.semantic_governance.budget as budget

    path = tmp_path / "l.json"
    ledger = SpendingLedger(path, cap_usd=1.0)
    ledger.record(5)
    vero = _os.replace
    fallimenti = {"n": 0}

    def replace_capriccioso(src, dst):
        if fallimenti["n"] < 3:
            fallimenti["n"] += 1
            raise PermissionError(13, "file in uso")
        return vero(src, dst)

    monkeypatch.setattr(budget.os, "replace", replace_capriccioso)
    ledger.record(7)
    assert fallimenti["n"] == 3
    assert json.loads(path.read_text()) == {"input_tokens": 12, "requests": 2}
