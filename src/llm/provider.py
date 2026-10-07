from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    cost_usd: float = 0.0
    pricing_source: str = ""
    provider: str = "fallback"
    api_call_attempted: bool = False
    error: str = ""


class LLMProvider(Protocol):
    def complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        ...

    async def async_complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        ...


class FallbackProvider:
    provider_id = "fallback"
    model = "fallback"

    def complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        text = (
            '{"thought_summary":"Fallback mode: conserve resources and explore locally.",'
            '"public_message":"Continuing local operations under fallback policy.",'
            '"chosen_action":"observe","target":{"type":"self"},"magnitude":1.0,'
            '"resource_offer":{},"resource_request":{},"cooperation_target":""}'
        )
        return LLMResponse(text=text, provider="fallback")

    async def async_complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        return self.complete_json(prompt, schema_hint)
