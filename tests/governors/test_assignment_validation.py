"""Un'assegnazione sbagliata deve fermare la run, non renderla muta.

Il braccio `llm` ha una proprieta' scomoda: **nessuno dei suoi modi di essere
mal configurato produce un errore**. `LLMProposer` non solleva mai per progetto
(un problema di rete non deve fermare una simulazione), e
`ConfiguredLLMProvider` risponde con il fallback quando la chiave manca. Un
provider inesistente, un modello non offerto o una chiave assente danno quindi
1500 passi di direttive vuote e risultati identici alla baseline — e la
conclusione naturale, "il modello non cambia niente", sarebbe tratta da un
esperimento mai eseguito.

Questi test provano che quei tre casi costano un errore alla costruzione.
"""

import pytest

from src.governors.config import build_governor
from src.llm.provider_registry import LLM_CONSENT_ENV


_REGISTRY = {
    "gpu_farm": {
        "provider_type": "openai",
        "api_key_env": "CHIAVE_DI_PROVA",
        "available_models": ["qwen3.6:27b"],
        "missing_env": "",
    }
}


def _config(**governors) -> dict:
    base = {
        "arm": "llm",
        "cadence_steps": 20,
    }
    base.update(governors)
    return {"governors": base}


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch):
    """Il registro vero legge chiavi API attive: qui non deve entrarci."""
    monkeypatch.setattr(
        "src.llm.provider_registry.load_provider_registry",
        lambda *args, **kwargs: {k: dict(v) for k, v in _REGISTRY.items()},
    )


def test_an_unknown_provider_stops_the_construction(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    config = _config(assignments=[{"provider": "farm_che_non_esiste", "model": "x"}])
    with pytest.raises(ValueError) as error:
        build_governor(config, seed=0)
    assert "gpu_farm" in str(error.value), "l'errore deve elencare i provider veri"


def test_an_assignment_without_a_model_stops_the_construction(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    config = _config(assignments=[{"provider": "gpu_farm"}])
    with pytest.raises(ValueError) as error:
        build_governor(config, seed=0)
    assert "modello" in str(error.value).lower()


def test_an_unoffered_model_stops_the_construction(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    config = _config(assignments=[{"provider": "gpu_farm", "model": "modello-inventato"}])
    with pytest.raises(ValueError):
        build_governor(config, seed=0)


def test_consent_without_the_key_stops_the_construction(monkeypatch):
    """Il caso piu' insidioso: consenso dato, chiave assente, nessun errore a run.

    `ConfiguredLLMProvider` restituisce il fallback quando `api_key` e' vuota, e
    lo fa senza popolare il campo `error`: la run sembrerebbe governata.
    """
    monkeypatch.setenv(LLM_CONSENT_ENV, "1")
    monkeypatch.setattr(
        "src.llm.provider_registry.load_provider_registry",
        lambda *args, **kwargs: {
            "gpu_farm": dict(_REGISTRY["gpu_farm"], missing_env="CHIAVE_DI_PROVA")
        },
    )
    config = _config(assignments=[{"provider": "gpu_farm", "model": "qwen3.6:27b"}])
    with pytest.raises(ValueError) as error:
        build_governor(config, seed=0)
    assert "CHIAVE_DI_PROVA" in str(error.value)


def test_without_consent_the_governor_is_built_but_says_so(monkeypatch, capsys):
    """Senza consenso il braccio `llm` gira sul fallback: legittimo, ma visibile.

    E' il modo in cui questo repository esegue il protocollo a quattro bracci
    senza consumare credito. Silenzioso pero' sarebbe indistinguibile da una run
    LLM riuscita, ed e' esattamente l'errore che questo file esiste per impedire.
    """
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    config = _config(assignments=[{"provider": "gpu_farm", "model": "qwen3.6:27b"}])
    governor = build_governor(config, seed=0)
    try:
        assert governor is not None
        avviso = capsys.readouterr().out
        assert LLM_CONSENT_ENV in avviso
        assert "fallback" in avviso
    finally:
        governor.close()


def test_a_deliberate_fallback_assignment_needs_no_model(monkeypatch):
    """`provider: fallback` significa "nessun provider", e resta legittimo."""
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    governor = build_governor(_config(assignments=[{"provider": "fallback"}]), seed=0)
    try:
        assert governor is not None
    finally:
        governor.close()
