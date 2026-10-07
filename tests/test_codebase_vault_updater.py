from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import update_codebase_vault


def test_offline_graphify_environment_removes_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in update_codebase_vault.EXTERNAL_API_ENV_VARS:
        monkeypatch.setenv(variable, "must-not-reach-graphify")

    environment = update_codebase_vault.offline_graphify_environment()

    assert update_codebase_vault.EXTERNAL_API_ENV_VARS.isdisjoint(environment)
    assert environment["HTTPS_PROXY"] == update_codebase_vault.BLOCKED_NETWORK_PROXY
    assert environment["HTTP_PROXY"] == update_codebase_vault.BLOCKED_NETWORK_PROXY
    assert environment["ALL_PROXY"] == update_codebase_vault.BLOCKED_NETWORK_PROXY
    assert environment["GRAPHIFY_MAX_WORKERS"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"


def test_legacy_deep_mode_is_a_hard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["update_codebase_vault.py", "--deep", "--backend", "gemini"],
    )

    with pytest.raises(SystemExit, match="External semantic extraction is disabled"):
        update_codebase_vault.main()


def test_unlink_with_retry_handles_transient_permission_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated_note = tmp_path / "generated.md"
    generated_note.write_text("generated", encoding="utf-8")
    original_unlink = Path.unlink
    attempts = 0

    def flaky_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal attempts
        if path == generated_note:
            attempts += 1
            if attempts < 3:
                raise PermissionError("transient editor lock")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr(update_codebase_vault.time, "sleep", lambda _seconds: None)

    update_codebase_vault.unlink_with_retry(generated_note)

    assert attempts == 3
    assert not generated_note.exists()
