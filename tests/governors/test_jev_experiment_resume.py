import argparse
from pathlib import Path

import pytest

from scripts.run_governor_experiment import (
    _add_semantic_arguments,
    _default_output_root,
    _etichetta,
    _resume_key,
    _semantic_section,
    _semantic_treatment_hash,
)


def test_semantic_cli_is_offline_by_default_and_never_invents_an_endpoint():
    parser = argparse.ArgumentParser()
    _add_semantic_arguments(parser)
    args = parser.parse_args([])
    assert args.semantic_provider == "fake"
    section = _semantic_section(args.semantic_provider, run_id="offline")
    assert section["provider"] == "fake"
    assert "endpoint" not in section


def test_rest_and_replay_require_explicit_inputs():
    with pytest.raises(ValueError, match="endpoint"):
        _semantic_section("rest", run_id="x")
    with pytest.raises(ValueError, match="replay"):
        _semantic_section("replay", run_id="x")


def test_resume_distinguishes_semantic_treatments_but_ignores_run_identity():
    first = _semantic_section(
        "fake", run_id="run-a", governor_decision="keep_previous"
    )
    same_treatment = _semantic_section(
        "fake", run_id="run-b", governor_decision="keep_previous"
    )
    different = _semantic_section(
        "fake", run_id="run-a", governor_decision="no_intervention"
    )
    first_hash = _semantic_treatment_hash("semif", "", first)
    assert first_hash == _semantic_treatment_hash("semif", "", same_treatment)
    assert first_hash != _semantic_treatment_hash("semif", "", different)
    assert _resume_key("semif:fake", 1, first_hash) != _resume_key(
        "semif:fake", 1, _semantic_treatment_hash("semif", "", different)
    )
    assert _resume_key("scripted", 1) == ("scripted", 1, "")


def test_semantic_provider_is_visible_in_new_labels_only():
    assert _etichetta("semif", "", False, semantic_provider="fake") == "semif:fake"
    assert _etichetta("hybrid", "completo", False, semantic_provider="replay") == (
        "hybrid:completo:replay"
    )
    assert _etichetta("scripted", "", False, semantic_provider="fake") == "scripted"


def test_semantic_arms_default_to_the_isolated_results_root():
    assert _default_output_root(["semif"]) == Path("runs/jev_semif_experiments")
    assert _default_output_root(["none"], "hybrid") == Path("runs/jev_semif_experiments")
    assert _default_output_root(["none", "scripted"]) == Path("runs/governor_experiment")
