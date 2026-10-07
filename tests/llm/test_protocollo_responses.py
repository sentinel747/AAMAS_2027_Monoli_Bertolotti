# -*- coding: utf-8 -*-
"""Il secondo protocollo OpenAI: `/responses` invece di `/chat/completions`.

Non sono due indirizzi, sono due API. Cambiano il campo che porta il prompt
(`input` invece di `messages`), il tetto di token (`max_output_tokens`), la
forma della risposta (una lista `output` di elementi tipizzati invece di
`choices`) e i nomi nell'uso (`input_tokens` invece di `prompt_tokens`).

**Perche' merita dei test anche prima di poterlo provare sul campo.** In questo
esperimento una chiamata che fallisce non da' errore: restituisce una risposta
vuota, che diventa una politica senza regole, che e' indistinguibile da un
governo che sceglie di non intervenire. Un campo sbagliato nel corpo non si
manifesterebbe come 400 a schermo ma come una campagna identica alla baseline
--- e' gia' successo due volte.
"""

import json

import pytest

from src.llm.configured_provider import ConfiguredLLMProvider


def _provider(**extra):
    config = {
        "base_url": "http://esempio/v1",
        "api_key_env": "PROVA_CHIAVE_INESISTENTE",
        "provider_type": "openai",
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        **extra,
    }
    return ConfiguredLLMProvider("gpu_farm", config, "Qwen/Qwen3.8-27B-FP8")


# ------------------------------------------------------------- la richiesta


def test_il_protocollo_di_serie_resta_quello_di_chat():
    url, payload = _provider()._openai_request("ciao")
    assert url.endswith("/chat/completions")
    assert payload["messages"][0]["content"] == "ciao"


def test_responses_cambia_percorso_e_nomi_dei_campi():
    url, payload = _provider(api_style="responses")._openai_request("ciao")
    assert url.endswith("/responses")
    assert payload["input"] == "ciao"
    assert payload["max_output_tokens"] == 8192
    # I nomi della chat non devono sopravvivere: su un endpoint severo un campo
    # sconosciuto e' un 400, e un 400 qui e' una run che sembra una baseline.
    assert "messages" not in payload
    assert "max_tokens" not in payload
    assert "response_format" not in payload


def test_il_json_obbligatorio_viaggia_annidato_sotto_text():
    _, payload = _provider(api_style="responses")._openai_request("ciao")
    assert payload["text"] == {"format": {"type": "json_object"}}


def test_lo_stesso_interruttore_spegne_il_formato_nei_due_protocolli():
    """`disable_response_format` esiste perche' i gateway non ufficiali lo rifiutano."""
    _, chat = _provider(disable_response_format=True)._openai_request("ciao")
    _, resp = _provider(api_style="responses", disable_response_format=True)._openai_request("ciao")
    assert "response_format" not in chat
    assert "text" not in resp


def test_il_ragionamento_diventa_un_oggetto_e_l_estensione_di_chat_sparisce():
    p = _provider(api_style="responses", thinking_style="ollama_effort")
    p._campi_thinking = {"reasoning_effort": "low",
                         "chat_template_kwargs": {"enable_thinking": False}}
    _, payload = p._openai_request("ciao")
    assert payload["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


# -------------------------------------------------------------- la risposta


def test_legge_la_lista_output_saltando_il_ragionamento():
    """Un modello che ragiona mette PRIMA un elemento `reasoning`.

    Prendere `output[0]` restituirebbe il ragionamento al posto della risposta.
    """
    dati = {
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": '{"ok": true}'}]},
        ],
        "usage": {"input_tokens": 120, "output_tokens": 14, "total_tokens": 134},
    }
    r = _provider(api_style="responses")._openai_response(dati)
    assert json.loads(r.text) == {"ok": True}
    assert (r.tokens_in, r.tokens_out, r.tokens_total) == (120, 14, 134)


def test_usa_output_text_quando_il_gateway_lo_offre():
    dati = {"output_text": '{"ok": 1}', "usage": {"input_tokens": 5, "output_tokens": 2}}
    r = _provider(api_style="responses")._openai_response(dati)
    assert r.text == '{"ok": 1}'
    assert r.tokens_total == 7


def test_una_risposta_in_forma_di_chat_viene_letta_lo_stesso():
    """Tolleranza deliberata: l'endpoint e' un gateway di terze parti.

    Se rispondesse ancora nella forma vecchia, il fallback e' quella forma e non
    una stringa vuota --- che si travestirebbe da governo silenzioso.
    """
    dati = {"choices": [{"message": {"content": '{"ok": 2}'}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}}
    r = _provider(api_style="responses")._openai_response(dati)
    assert r.text == '{"ok": 2}'
    assert (r.tokens_in, r.tokens_out) == (9, 3)


def test_una_risposta_incomprensibile_non_diventa_una_stringa_vuota():
    r = _provider(api_style="responses")._openai_response({"qualcosa": "altro"})
    assert r.text == "{}"


# ------------------------------------------------------- il registro reale


def test_le_due_farm_sono_distinte_e_sono_le_uniche():
    """Due farm, due chiavi, due insiemi di modelli disgiunti.

    Sono macchine diverse: mandare a una il modello dell'altra da' un 404, e un
    404 in questo esperimento non e' un errore visibile ma una run identica alla
    baseline. Le farm dismesse restano fuori dal registro per non poter essere
    scelte per sbaglio da una riga di comando.
    """
    from src.llm.provider_registry import load_provider_registry

    registro = load_provider_registry()
    farm = {n for n in registro if n.startswith("gpu_farm")}
    assert farm == {"gpu_farm", "gpu_farm4"}

    tre, quattro = registro["gpu_farm"], registro["gpu_farm4"]
    assert tre["api_key_env"] == "GPU_FARM_KEY3"
    assert quattro["api_key_env"] == "GPU_FARM_KEY4"
    assert quattro["base_url"] == "http://llm-server.example:8000/v1"
    assert quattro["available_models"] == ["Qwen/Qwen3.8-27B-FP8"]
    assert not set(tre["available_models"]) & set(quattro["available_models"])


def test_nessuna_delle_due_farm_dichiara_il_protocollo_responses():
    """Entrambe parlano l'API di chat. Il supporto a `/responses` resta nel
    codice --- provato dai test qui sopra --- e si accende dal registro se un
    endpoint futuro lo richiede."""
    from src.llm.provider_registry import load_provider_registry

    registro = load_provider_registry()
    for nome in ("gpu_farm", "gpu_farm4"):
        assert registro[nome].get("api_style", "chat") == "chat"


def test_le_farm_non_costano_niente_anche_col_nome_di_un_modello_a_pagamento():
    """`Qwen3.8-27b-fp8` gira su una GPU nostra, non su DashScope.

    Il listino intercetta i nomi che cominciano per `qwen`: senza la
    precedenza al ramo delle farm, una campagna gratuita riporterebbe un costo
    inventato --- l'errore speculare a quello che rendeva Qwen gratis.
    """
    from src.llm.pricing import price_for_model

    prezzo = price_for_model("openai", "gpu_farm4", "Qwen/Qwen3.8-27B-FP8")
    assert prezzo is not None
    assert (prezzo.input_per_million, prezzo.output_per_million) == (0.0, 0.0)


# --------------------------------- il ragionamento «spento» sui modelli o*


@pytest.mark.parametrize("modello", ["o1", "o3-mini", "o4-mini"])
def test_off_non_manda_minimal_ai_modelli_che_non_lo_conoscono(modello):
    """`minimal` esiste solo dalla famiglia gpt-5 in poi.

    Misurato il 2026-09-03: o1, o3-mini e o4-mini rispondono HTTP 400
    ("does not support 'minimal' with this model"), quindi tre modelli su sei
    del fornitore erano inutilizzabili come governatori — e il sintomo non era
    un errore ma una policy vuota, cioe' una run indistinguibile dalla
    baseline.
    """
    from src.llm.thinking import campi

    valori, avviso = campi("openai_effort", modello, "off")
    assert valori == {"reasoning_effort": "low"}
    assert "minimal" in avviso


def test_off_resta_minimal_dove_e_supportato():
    from src.llm.thinking import campi

    valori, avviso = campi("openai_effort", "gpt-5-mini", "off")
    assert valori == {"reasoning_effort": "minimal"}
    assert avviso == ""


def test_i_livelli_espliciti_non_sono_toccati():
    from src.llm.thinking import campi

    for modello in ("o4-mini", "gpt-5-mini"):
        assert campi("openai_effort", modello, "high")[0] == {"reasoning_effort": "high"}
