"""Selezione del profilo candidate v1/v2 dalla configurazione (piano 2026-09-22, Task 4)."""

import asyncio

import pytest

from src.governors.config import build_administration, build_governor
from src.governors.observation import ColonyPicture
from src.semantic_governance.client import FakeSemanticDecisionProvider

PICTURE = ColonyPicture(
    step=1, population=8, n_cells=1,
    indicators={"food_per_occupant": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}},
)


def _governor_config(profile=None):
    semantic = {"run_id": "wiring"}
    if profile is not None:
        semantic["candidate_profile"] = profile
    return {"days": 2, "governors": {"arm": "semif", "cadence_steps": 1,
                                     "wait_seconds": 1, "semantic": semantic}}


def _asked_question(profile):
    provider = FakeSemanticDecisionProvider()
    governor = build_governor(_governor_config(profile), seed=1, semantic_provider=provider)
    try:
        governor.advance(1, PICTURE)
    finally:
        governor.close()
    return provider.requests[0].question_id


def test_default_profile_is_v1():
    assert _asked_question(None) == "jev-semif-governor-v1"


def test_v2_profile_is_selectable_for_the_governor():
    assert _asked_question("v2") == "jev-semif-governor-v2"


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="candidate_profile"):
        build_governor(_governor_config("v9"), seed=1,
                       semantic_provider=FakeSemanticDecisionProvider())


def test_v2_profile_reaches_the_administrators():
    config = _governor_config("v2")
    config["governors"]["administrators"] = {"enabled": True}
    provider = FakeSemanticDecisionProvider()
    administration = build_administration(
        config, seed=1, semantic_provider_factory=lambda _d: provider,
    )
    admin = administration._proponente(0)
    assert admin._factory.profile_version == "jev-semif-admin-v2"


def test_manifest_reports_the_configured_profile(tmp_path):
    import json

    from src.semantic_governance.artifacts import write_semantic_artifacts

    config = {"seed": 3, "governors": {"arm": "semif", "semantic": {
        "provider": "fake", "run_id": "m", "candidate_profile": "v2"}}}
    write_semantic_artifacts(tmp_path, config)
    manifest = json.loads((tmp_path / "semantic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_profiles"] == {
        "governor": "jev-semif-governor-v2", "administrator": "jev-semif-admin-v2",
    }


def test_cli_section_writes_the_profile_only_when_not_v1():
    from scripts.run_governor_experiment import _semantic_section

    assert "candidate_profile" not in _semantic_section("fake", run_id="x")
    v2 = _semantic_section("fake", run_id="x", candidate_profile="v2")
    assert v2["candidate_profile"] == "v2"
    assert v2["decisions"]["jev-semif-governor-v2"] == "keep_previous"
    assert v2["decisions"]["jev-semif-admin-v2"] == "accept_governor"


def test_run_arm_carries_the_profile_into_the_run_config(tmp_path):
    import yaml

    from scripts.run_governor_experiment import run_arm

    out = tmp_path / "run"
    run_arm("semif", 101, 2, 8, out, cadence_steps=1, wait_seconds=1,
            semantic_provider="fake", semantic_candidate_profile="v2")
    config = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert config["governors"]["semantic"]["candidate_profile"] == "v2"
