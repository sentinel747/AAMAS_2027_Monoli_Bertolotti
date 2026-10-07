"""La configurazione: default sicuro, errori dichiarati, nessuna sorpresa."""

import pytest

from src.governors.config import build_governor, directives_applied, read_settings


def test_the_default_configuration_has_no_governor():
    settings = read_settings({})
    assert settings.enabled is False
    assert build_governor({}, seed=0) is None


def test_arm_none_disables_the_path_entirely():
    config = {"governors": {"arm": "none", "cadence_steps": 5}}
    assert build_governor(config, seed=0) is None


def test_an_unknown_arm_is_a_configuration_error():
    with pytest.raises(ValueError, match="arm"):
        read_settings({"governors": {"arm": "presidente"}})


def test_a_council_era_count_zero_still_means_off():
    """`count: 0` era l'interruttore spento del consiglio: una config importata
    da una run vecchia non deve accendere un governatore che quella run non
    aveva."""
    config = {"governors": {"count": 0, "arm": "scripted"}}
    assert read_settings(config).enabled is False
    assert build_governor(config, seed=0) is None


def test_council_era_keys_warn_but_do_not_stop(capsys):
    """La funzione "ripeti una run" importa configurazioni intere: una run del
    consiglio deve poter ripartire — con UN governatore — e dirlo."""
    config = {
        "governors": {
            "count": 3,
            "arm": "scripted",
            "mandates": ["life_support", "expansion", "logistics"],
            "cadence_steps": 5,
        }
    }
    settings = read_settings(config)
    assert settings.enabled is True
    out = capsys.readouterr().out
    assert "consiglio non esiste piu'" in out
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
    finally:
        governor.close()


def test_the_llm_arm_without_consent_falls_back_and_says_so(monkeypatch):
    from src.llm.provider_registry import LLM_CONSENT_ENV

    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    config = {"governors": {"arm": "llm", "cadence_steps": 20}}
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
        # Il proposer esiste ma il provider sotto e' il fallback deterministico.
        from src.governors.llm_arm import LLMProposer

        assert isinstance(governor._proposer, LLMProposer)
    finally:
        governor.close()


def test_the_number_of_policies_a_run_applies():
    """L'aritmetica dei confini di tick, che decide se la run misura qualcosa.

    I confini cadono ai passi 1, 1+c, 1+2c...; la policy del tick `t` entra in
    vigore al passo `1 + (t+1)*c`. Con cadenza 20 su 1500 passi sono 74; con
    una cadenza pari o superiore alla run sono zero.
    """
    assert directives_applied(1500, 20) == 74
    assert directives_applied(1500, 1499) == 1
    assert directives_applied(1500, 1500) == 0
    assert directives_applied(1500, 2644) == 0


def test_the_blocking_regime_counts_the_boundary_itself():
    """Nel regime bloccante la policy entra in vigore al confine che l'ha
    prodotta, quindi ogni confine dentro la run ne produce una."""
    assert directives_applied(1500, 20, blocking=True) == 75
    assert directives_applied(1500, 1500, blocking=True) == 1
    assert directives_applied(1, 20, blocking=True) == 1
    assert directives_applied(0, 20, blocking=True) == 0


def test_a_cadence_longer_than_the_run_is_a_configuration_error():
    config = {
        "days": 10,
        "governors": {"arm": "scripted", "cadence_steps": 10},
    }
    with pytest.raises(ValueError, match="cadence_steps"):
        build_governor(config, seed=0)


def test_a_long_cadence_is_accepted_when_the_governor_waits(capsys):
    config = {
        "days": 10,
        "governors": {"arm": "scripted", "cadence_steps": 10, "wait_seconds": 5.0},
    }
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
        assert "bloccante" in capsys.readouterr().out
    finally:
        governor.close()


def test_a_cadence_that_fits_the_run_is_accepted():
    config = {"days": 10, "governors": {"arm": "scripted", "cadence_steps": 5}}
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
    finally:
        governor.close()


def test_without_a_declared_length_the_cadence_is_not_judged():
    config = {"governors": {"arm": "scripted", "cadence_steps": 10_000}}
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
    finally:
        governor.close()


def test_the_bounds_come_from_the_configuration():
    config = {
        "governors": {
            "arm": "scripted",
            "priority_multiplier_range": [0.5, 2.0],
        }
    }
    settings = read_settings(config)
    assert settings.bounds.weight_min == 0.5
    assert settings.bounds.weight_max == 2.0


def test_the_wait_is_read_from_the_configuration():
    settings = read_settings(
        {"governors": {"arm": "scripted", "wait_seconds": 12.5}}
    )
    assert settings.wait_seconds == 12.5
    assert read_settings({"governors": {"arm": "scripted"}}).wait_seconds == 0.0


def test_the_reasoning_effort_reaches_the_provider(monkeypatch):
    """Il difetto che ha reso nulla la prima run vera a quattro bracci.

    Il preflight accetta `--effort` e misura una latenza; se quello sforzo non
    arriva alla run, la run gira con lo sforzo del registro e il limite di
    attesa e' stato scelto su un numero che non le appartiene.
    """
    class _ProviderFinto:
        provider_id = "gpu_farm"
        model = "gpt-oss:20b"

        def __init__(self) -> None:
            self.options = {"extra_body": {"reasoning_effort": "medium"}}

    finto = _ProviderFinto()
    monkeypatch.setattr(
        "src.llm.provider_registry.load_provider_registry",
        lambda *a, **k: {"gpu_farm": {"available_models": ["gpt-oss:20b"], "missing_env": ""}},
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.resolve_offered_model",
        lambda provider_id, spec, model: model,
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.create_llm_provider",
        lambda provider_id, model=None, registry=None: finto,
    )
    config = {
        "governors": {
            "arm": "llm",
            "cadence_steps": 20,
            "assignments": [{"provider": "gpu_farm", "model": "gpt-oss:20b", "effort": "low"}],
        }
    }
    governor = build_governor(config, seed=0)
    try:
        assert finto.options["extra_body"]["reasoning_effort"] == "low"
    finally:
        governor.close()


def test_the_temperature_reaches_the_provider_and_the_attribute_that_is_sent(monkeypatch):
    """Ripetere lo stesso seme non misura niente se il modello ricampiona.

    **L'attributo, non solo il dizionario.** `ConfiguredLLMProvider` legge
    `options["temperature"]` una volta sola nel costruttore e poi manda
    `self.temperature`: scrivere nel dizionario dopo la costruzione non
    cambierebbe una singola richiesta, e il test passerebbe lo stesso.
    """
    class _ProviderFinto:
        provider_id = "gpu_farm"
        model = "gpt-oss:20b"

        def __init__(self) -> None:
            self.options = {"temperature": 1.0}
            self.temperature = 1.0

    finto = _ProviderFinto()
    monkeypatch.setattr(
        "src.llm.provider_registry.load_provider_registry",
        lambda *a, **k: {"gpu_farm": {"available_models": ["gpt-oss:20b"], "missing_env": ""}},
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.resolve_offered_model",
        lambda provider_id, spec, model: model,
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.create_llm_provider",
        lambda provider_id, model=None, registry=None: finto,
    )
    config = {
        "governors": {
            "arm": "llm",
            "cadence_steps": 20,
            "assignments": [
                {"provider": "gpu_farm", "model": "gpt-oss:20b", "temperature": 0.0}
            ],
        }
    }
    governor = build_governor(config, seed=0)
    try:
        assert finto.temperature == 0.0
        assert finto.options["temperature"] == 0.0
    finally:
        governor.close()


def test_no_temperature_in_the_assignment_leaves_the_registry_alone(monkeypatch):
    """Zero e' un valore, non un'assenza: la distinzione va tenuta."""
    class _ProviderFinto:
        provider_id = "gpu_farm"
        model = "gpt-oss:20b"

        def __init__(self) -> None:
            self.options = {"temperature": 1.0}
            self.temperature = 1.0

    finto = _ProviderFinto()
    monkeypatch.setattr(
        "src.llm.provider_registry.load_provider_registry",
        lambda *a, **k: {"gpu_farm": {"available_models": ["gpt-oss:20b"], "missing_env": ""}},
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.resolve_offered_model",
        lambda provider_id, spec, model: model,
    )
    monkeypatch.setattr(
        "src.llm.provider_registry.create_llm_provider",
        lambda provider_id, model=None, registry=None: finto,
    )
    config = {
        "governors": {
            "arm": "llm",
            "assignments": [{"provider": "gpu_farm", "model": "gpt-oss:20b"}],
        }
    }
    governor = build_governor(config, seed=0)
    try:
        assert finto.temperature == 1.0
    finally:
        governor.close()
