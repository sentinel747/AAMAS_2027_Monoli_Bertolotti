"""Un modello non offerto e' un errore, non una sostituzione silenziosa.

Il codice prendeva il primo modello elencato quando quello chiesto non c'era.
Una run configurata su `qwen3.6:27b` girava allora su `gpt-oss:120b` senza una
riga di avviso: non un risultato peggiore, un risultato **falso**, perche'
l'esperimento attribuisce a un modello cio' che ha detto un altro.
"""

import pytest

from src.llm.provider_registry import normalize_llm_assignments, resolve_offered_model


_REGISTRY = {
    "gpu_farm": {
        "provider_type": "openai",
        "api_key_env": "CHIAVE_DI_PROVA",
        "available_models": ["qwen3.6:27b", "gpt-oss:120b"],
    }
}


def test_an_offered_model_passes_through():
    assert resolve_offered_model("gpu_farm", _REGISTRY["gpu_farm"], "qwen3.6:27b") == "qwen3.6:27b"


def test_an_unoffered_model_raises_and_names_the_alternatives():
    with pytest.raises(ValueError) as error:
        resolve_offered_model("gpu_farm", _REGISTRY["gpu_farm"], "qwen9:999b")
    messaggio = str(error.value)
    assert "qwen9:999b" in messaggio
    assert "qwen3.6:27b" in messaggio, "l'errore deve dire cosa si puo' chiedere"


def test_an_empty_catalogue_accepts_any_name():
    """`available_models` vuoto significa "non dichiaro nulla", non "niente e' valido"."""
    assert resolve_offered_model("x", {"available_models": []}, "qualsiasi") == "qualsiasi"
    assert resolve_offered_model("x", {}, "qualsiasi") == "qualsiasi"


def test_the_assignment_keeps_the_model_it_names():
    config = {
        "agents": {"llm_assignments": [{"provider": "gpu_farm", "model": "gpt-oss:120b"}]},
    }
    assert normalize_llm_assignments(config, 1, _REGISTRY) == [
        {"provider": "gpu_farm", "model": "gpt-oss:120b"}
    ]


def test_an_assignment_on_an_unoffered_model_stops_the_run():
    config = {"agents": {"llm_assignments": [{"provider": "gpu_farm", "model": "inesistente"}]}}
    with pytest.raises(ValueError):
        normalize_llm_assignments(config, 1, _REGISTRY)


def test_no_llm_agents_resolves_nothing_even_without_a_registry():
    """Una run rule-based non deve dipendere dal registro dei provider.

    Senza questa uscita anticipata la risoluzione del provider di default
    verrebbe eseguita comunque, e la validazione stretta fermerebbe una run che
    di LLM non usa niente.
    """
    assert normalize_llm_assignments({"llm": {"provider": "inesistente"}}, 0, {}) == []
