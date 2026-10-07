"""Nessuna chiamata API senza consenso esplicito.

Le chiavi di `gpt`, `gemini`, `gpu_farm`, `gpu_farm2`, `claude` e `ollama` sono
ATTIVE nell'ambiente di sviluppo. Qualunque test o run che risolva un provider dal
registro farebbe chiamate vere e consumerebbe credito. La disciplina non e' una
garanzia: questo interlock lo e'.

Il test verifica solo il TIPO restituito. Non chiama mai il provider.
"""

from src.llm.provider import FallbackProvider
from src.llm.provider_registry import (
    LLM_CONSENT_ENV,
    create_llm_provider,
    real_calls_allowed,
)

_REGISTRY = {
    "gpt": {
        "provider_type": "openai",
        "api_key_env": "GPT_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "available_models": ["gpt-4o-mini"],
    }
}


def test_without_consent_a_real_provider_is_never_built(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    monkeypatch.setenv("GPT_API_KEY", "una-chiave-che-sembra-vera")
    provider = create_llm_provider("gpt", "gpt-4o-mini", _REGISTRY)
    assert isinstance(provider, FallbackProvider), (
        "senza consenso la fabbrica deve restituire il fallback anche con la "
        "chiave presente: e' l'unica cosa che impedisce di consumare credito"
    )


def test_with_consent_a_real_provider_is_built_but_not_called(monkeypatch):
    monkeypatch.setenv(LLM_CONSENT_ENV, "1")
    monkeypatch.setenv("GPT_API_KEY", "una-chiave-che-sembra-vera")
    provider = create_llm_provider("gpt", "gpt-4o-mini", _REGISTRY)
    assert not isinstance(provider, FallbackProvider)
    assert provider.provider_id == "gpt"


def test_the_consent_flag_reads_the_environment(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    assert real_calls_allowed() is False
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv(LLM_CONSENT_ENV, value)
        assert real_calls_allowed() is False, f"{value!r} non e' un consenso"
    for value in ("1", "true", "yes"):
        monkeypatch.setenv(LLM_CONSENT_ENV, value)
        assert real_calls_allowed() is True


def test_the_fallback_provider_is_still_reachable_by_name(monkeypatch):
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    assert isinstance(create_llm_provider("fallback", None, _REGISTRY), FallbackProvider)


def test_a_value_that_is_not_an_explicit_yes_denies_the_consent(monkeypatch):
    """Un valore inatteso non e' un consenso: l'interruttore chiude, non apre.

    Aggiunto oltre ai test del piano. Una lista di NEGAZIONI note lascia passare
    tutto il resto: `MARSABM_ALLOW_LLM_CALLS=flase` (refuso), `=disabled`,
    `=fallback` o l'`=None` che una shell scrive quando interpola una variabile
    vuota accenderebbero le chiamate vere. In un repository a policy zero-API il
    caso ambiguo deve costare un fallback, non del credito.
    """
    for value in ("flase", "disabled", "None", "nope", "2", "-1", "si"):
        monkeypatch.setenv(LLM_CONSENT_ENV, value)
        assert real_calls_allowed() is False, f"{value!r} non e' un consenso esplicito"


def test_without_consent_no_provider_id_in_the_registry_builds_a_real_client(monkeypatch):
    """L'interlock copre ogni provider del registro, non solo `gpt`.

    Aggiunto oltre ai test del piano. Il test del piano prova una sola voce; le
    chiavi presenti nell'ambiente sono sei, e l'alias `openai` -> `gpt` mostra che
    il nome chiesto non e' quello risolto. Qui si prova che nessuna delle vie
    d'ingresso costruisce un client reale senza consenso.
    """
    registry = {
        "gpt": {"provider_type": "openai", "api_key_env": "GPT_API_KEY", "available_models": ["gpt-4o-mini"]},
        "gemini": {"provider_type": "google", "api_key_env": "GEMINI_API_KEY", "available_models": ["gemini-2.0-flash"]},
        "claude": {"provider_type": "anthropic", "api_key_env": "CLAUDE_API_KEY", "available_models": ["claude-sonnet-4"]},
        "ollama": {"provider_type": "ollama", "base_url": "http://localhost:11434", "available_models": ["llama3"]},
    }
    monkeypatch.delenv(LLM_CONSENT_ENV, raising=False)
    for provider_id in (*registry, "openai"):
        provider = create_llm_provider(provider_id, None, registry)
        assert isinstance(provider, FallbackProvider), f"{provider_id!r} ha costruito un client reale senza consenso"
