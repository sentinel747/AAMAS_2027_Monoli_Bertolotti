"""Le opzioni per modello arrivano davvero nel corpo della richiesta.

Nessun test qui fa una chiamata: costruiscono il provider e leggono il payload
che manderebbe. E' la sola cosa che serve provare, perche' i tre valori che
contano — tetto dei token, timeout, temperatura — non danno errore quando sono
sbagliati: danno una risposta vuota o troncata, che il resto del sistema
interpreta come "il governatore non ha proposto niente".
"""

import asyncio

import pytest

from src.llm.configured_provider import ConfiguredLLMProvider, resolve_model_options


_FARM = {
    "base_url": "http://esempio.invalido:3000/ollama/v1",
    "api_key_env": "CHIAVE_DI_PROVA",
    "provider_type": "openai",
    "temperature": 0.6,
    "max_tokens": 8192,
    "timeout_seconds": 600,
    "extra_body": {"top_k": 20},
    "available_models": ["qwen3.6:27b", "gpt-oss:20b"],
    "model_options": {
        "qwen3.6:27b": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "timeout_seconds": 900,
            "extra_body": {"reasoning_effort": "high"},
        }
    },
}


def _provider(model: str, config: dict | None = None) -> ConfiguredLLMProvider:
    return ConfiguredLLMProvider("gpu_farm", dict(config or _FARM), model)


def test_model_options_override_the_provider_entry():
    resolved = resolve_model_options(_FARM, "qwen3.6:27b")
    assert resolved["max_tokens"] == 32768
    assert resolved["timeout_seconds"] == 900
    assert resolved["top_p"] == 0.95


def test_a_model_without_its_own_options_inherits_the_provider_entry():
    resolved = resolve_model_options(_FARM, "gpt-oss:20b")
    assert resolved["max_tokens"] == 8192
    assert resolved["timeout_seconds"] == 600
    assert "top_p" not in resolved, "top_p non e' dichiarato a livello di provider"


def test_a_provider_without_options_keeps_the_historical_defaults():
    """La voce che non dichiara nulla si comporta come prima del cambiamento."""
    provider = _provider("gpt-4o-mini", {"provider_type": "openai", "base_url": "https://api.openai.com/v1"})
    assert provider.max_tokens == 700
    assert provider.timeout == 45
    assert provider.temperature == 0.2
    assert provider.top_p is None


def test_the_payload_carries_the_calibrated_options():
    provider = _provider("qwen3.6:27b")
    payload = provider._openai_payload("ciao")
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 32768
    assert provider.timeout == 900


def test_the_reasoning_family_gets_no_temperature_and_another_token_field():
    provider = _provider("gpt-5-mini", {"provider_type": "openai", "temperature": 0.6, "top_p": 0.9})
    payload = provider._openai_payload("ciao")
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload


def test_extra_body_merges_the_two_levels():
    """Fusione e non sostituzione: il modello aggiunge, non cancella."""
    payload = _provider("qwen3.6:27b")._openai_payload("ciao")
    assert payload["top_k"] == 20, "il campo del provider deve sopravvivere"
    assert payload["reasoning_effort"] == "high", "il campo del modello deve arrivare"


def test_response_format_can_be_turned_off():
    """L'unico campo che il codice manda di sua iniziativa deve essere spegnibile.

    Un endpoint OpenAI-compatibile non ufficiale puo' rifiutare
    `response_format`, e la diagnosi si fa con `check_provider`, non riscrivendo
    il codice a meta' di un esperimento.
    """
    assert "response_format" in _provider("qwen3.6:27b")._openai_payload("ciao")
    spento = dict(_FARM, disable_response_format=True)
    assert "response_format" not in _provider("qwen3.6:27b", spento)._openai_payload("ciao")


def test_the_sync_and_async_paths_send_the_same_body(monkeypatch):
    """I due percorsi devono restare indistinguibili nel corpo che mandano.

    Il preflight usa l'asincrono come i governatori, ma la GUI e gli agenti
    usano il sincrono: se i due divergessero, un controllo superato non direbbe
    piu' niente sulla run. Componevano il payload ciascuno per conto suo con le
    stesse dodici righe; ora la sorgente e' una, e questo test lo mantiene vero.
    """
    provider = _provider("qwen3.6:27b")
    monkeypatch.setenv("CHIAVE_DI_PROVA", "finta")
    provider.api_key = "finta"
    visti: list[dict] = []

    def _sync(url, payload, headers):
        visti.append(payload)
        return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    async def _async(url, payload, headers):
        visti.append(payload)
        return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    monkeypatch.setattr(provider, "_request_json", _sync)
    monkeypatch.setattr(provider, "_async_request_json", _async)

    provider.complete_json("ciao")
    asyncio.run(provider.async_complete_json("ciao"))

    assert len(visti) == 2
    assert visti[0] == visti[1]


_GEMINI = {
    "provider_type": "google",
    "api_key_env": "CHIAVE_DI_PROVA",
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 4096,
    "extra_body": {"top_k": 20},
    "available_models": ["gemini-2.5-flash"],
    "model_options": {"gemini-2.5-flash": {"max_tokens": 8192, "timeout_seconds": 90}},
}


def test_gemini_receives_temperature_and_top_p():
    """Su questo provider le opzioni erano lettera morta.

    `_google` costruiva `generationConfig` con il solo `maxOutputTokens` (piu'
    `topK` da `extra_body`): temperatura e `topP` non venivano mandate affatto,
    quindi ogni chiamata usava i default del servizio qualunque cosa dicesse la
    configurazione.
    """
    provider = ConfiguredLLMProvider("gemini", dict(_GEMINI), "gemini-2.5-flash")
    provider.api_key = "finta"
    url, payload = provider._google_request("ciao")
    config = payload["generationConfig"]
    assert config["temperature"] == 0.2
    assert config["topP"] == 0.95
    assert config["topK"] == 20
    assert config["maxOutputTokens"] == 8192, "l'opzione per modello deve vincere"
    assert "gemini-2.5-flash" in url


def test_the_two_gemini_paths_send_the_same_body(monkeypatch):
    provider = ConfiguredLLMProvider("gemini", dict(_GEMINI), "gemini-2.5-flash")
    provider.api_key = "finta"
    visti: list[dict] = []

    def _sync(url, payload, headers):
        visti.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}

    async def _async(url, payload, headers):
        visti.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}

    monkeypatch.setattr(provider, "_request_json", _sync)
    monkeypatch.setattr(provider, "_async_request_json", _async)
    provider.complete_json("ciao")
    asyncio.run(provider.async_complete_json("ciao"))
    assert len(visti) == 2
    assert visti[0] == visti[1]


def test_the_real_registry_gives_governor_sized_ceilings():
    """700 token non bastano: la farm ne ha prodotti 526-627 di sola risposta.

    Il default storico avrebbe troncato una direttiva vera su un modello che
    ragiona, e il troncamento non da' errore -- da' "risposta non interpretabile",
    cioe' un governatore muto.
    """
    from src.llm.provider_registry import load_provider_registry

    registry = load_provider_registry()
    for provider_id, model in (
        ("gpu_farm", "qwen3.6:27b"),
        ("gpt", "gpt-4o-mini"),
        ("gemini", "gemini-2.5-flash"),
    ):
        opzioni = resolve_model_options(registry[provider_id], model)
        assert opzioni.get("max_tokens", 700) >= 4096, f"{provider_id}:{model}"
        assert opzioni.get("timeout_seconds", 45) >= 60, f"{provider_id}:{model}"


@pytest.mark.parametrize("valore", [0.0, 1.0])
def test_a_temperature_of_zero_is_sent_and_not_confused_with_absence(valore):
    """`0.0` e' falso in Python: un controllo di verita' lo cancellerebbe."""
    provider = _provider("gpt-oss:20b", dict(_FARM, temperature=valore))
    assert provider._openai_payload("ciao")["temperature"] == valore
