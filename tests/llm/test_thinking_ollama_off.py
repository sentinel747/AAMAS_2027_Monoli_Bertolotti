# -*- coding: utf-8 -*-
"""`thinking: off` via Ollama e' `reasoning_effort: none`, non `low`.

Misurato il 2026-09-07 sulla farm (Ollama 0.32.15): con `low` qwen3.6:27b
ragionava comunque, 216 s a chiamata e nessun JSON nei 600 token; con `none`
risponde in 31 s con la policy. gpt-oss non sa spegnere il ragionamento: per
lui `off` resta `low`, il minimo, ed e' la forma con cui tutte le campagne
gpt-oss sono state eseguite.
"""

from src.llm.thinking import campi


def test_off_on_ollama_is_none_for_qwen():
    valori, avviso = campi("ollama_effort", "qwen3.6:27b", "off")
    assert valori["reasoning_effort"] == "none"
    assert valori["chat_template_kwargs"] == {"enable_thinking": False}
    assert avviso == ""


def test_off_on_ollama_stays_low_for_gpt_oss_which_cannot_switch_off():
    valori, _ = campi("ollama_effort", "gpt-oss:20b", "off")
    assert valori["reasoning_effort"] == "low"
    valori, _ = campi("ollama_effort", "gpt-oss:120b", "off")
    assert valori["reasoning_effort"] == "low"


def test_off_on_vllm_qwen_keeps_the_template_switch():
    valori, _ = campi("ollama_effort", "Qwen/Qwen3.8-27B-FP8", "off")
    assert valori["reasoning_effort"] == "none"
    assert valori["chat_template_kwargs"]["enable_thinking"] is False


def test_levels_above_off_are_unchanged():
    valori, _ = campi("ollama_effort", "qwen3.6:27b", "high")
    assert valori == {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}}
