from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    input_per_million: float
    output_per_million: float
    source: str


def estimate_token_cost(provider_type: str, provider_id: str, model: str, tokens_in: int, tokens_out: int) -> tuple[float, TokenPrice | None]:
    price = price_for_model(provider_type, provider_id, model)
    if price is None:
        return 0.0, None
    cost = (tokens_in / 1_000_000.0) * price.input_per_million + (tokens_out / 1_000_000.0) * price.output_per_million
    return cost, price


def price_for_model(provider_type: str, provider_id: str, model: str) -> TokenPrice | None:
    provider_key = provider_id.lower()
    model_key = model.lower()
    if provider_key.startswith(("gpu_farm", "ollama")):
        return TokenPrice(0.0, 0.0, "local_or_custom_provider")
    # **Qwen parla il protocollo OpenAI ma non ne ha il listino.** Va
    # intercettato PRIMA di `_openai_price`, che sui nomi `qwen*` non trova
    # nulla e restituisce `None`: senza questo ramo la contabilita' di una
    # campagna su DashScope riporterebbe zero dollari, che e' il modo piu'
    # silenzioso di sforare un budget.
    if provider_key == "qwen" or model_key.startswith("qwen"):
        return _qwen_price(model_key)
    if provider_type == "openai":
        return _openai_price(model_key)
    if provider_type == "google":
        return _google_price(model_key)
    if provider_type == "anthropic":
        return _anthropic_price(model_key)
    return None


def _qwen_price(model: str) -> TokenPrice | None:
    """Listino Alibaba Model Studio (DashScope international), dollari per milione.

    Le famiglie sono tre e il prezzo segue quelle, non il singolo nome: `max` e'
    la piu' cara, `plus` la via di mezzo, `flash`/`turbo` le economiche. Un
    modello sconosciuto della stessa famiglia riceve quindi comunque un prezzo
    plausibile invece di zero, che e' il valore pericoloso.
    """
    if "max" in model:
        return TokenPrice(1.60, 6.40, "dashscope_pricing_2026-05")
    if "plus" in model:
        return TokenPrice(0.40, 1.20, "dashscope_pricing_2026-05")
    if "flash" in model or "turbo" in model:
        return TokenPrice(0.05, 0.40, "dashscope_pricing_2026-05")
    return TokenPrice(0.40, 1.20, "dashscope_pricing_2026-05")


def _openai_price(model: str) -> TokenPrice | None:
    # **La famiglia che ragiona mancava del tutto, e o1 e' la piu' cara del
    # listino.** Un modello assente qui non costa zero: costa e basta, e viene
    # riportato zero — lo stesso difetto silenzioso gia' trovato su Qwen. Una
    # sonda su o1 consuma piu' di duemila token di uscita a 60 dollari per
    # milione, e la contabilita' della campagna diceva 0,0000.
    if model.startswith("o1"):
        return TokenPrice(15.00, 60.00, "openai_pricing_2026-05")
    if model.startswith("o3-mini") or model.startswith("o4-mini"):
        return TokenPrice(1.10, 4.40, "openai_pricing_2026-05")
    if model.startswith("o3"):
        return TokenPrice(2.00, 8.00, "openai_pricing_2026-05")
    if model.startswith("gpt-5.4-mini"):
        return TokenPrice(0.75, 4.50, "openai_pricing_2026-05")
    if model.startswith("gpt-5-mini"):
        return TokenPrice(0.25, 2.00, "openai_pricing_2026-05")
    if model.startswith("gpt-5"):
        return TokenPrice(1.25, 10.00, "openai_pricing_2026-05")
    if model.startswith("gpt-4.1-mini"):
        return TokenPrice(0.40, 1.60, "openai_pricing_2026-05")
    if model.startswith("gpt-4.1"):
        return TokenPrice(2.00, 8.00, "openai_pricing_2026-05")
    if model.startswith("gpt-4o-mini"):
        return TokenPrice(0.15, 0.60, "openai_pricing_2026-05")
    if model.startswith("gpt-4o"):
        return TokenPrice(2.50, 10.00, "openai_pricing_2026-05")
    return None


def _google_price(model: str) -> TokenPrice | None:
    if model.startswith("gemini-2.5-flash-lite"):
        return TokenPrice(0.10, 0.40, "gemini_pricing_2026-05")
    if model.startswith("gemini-2.5-flash"):
        return TokenPrice(0.30, 2.50, "gemini_pricing_2026-05")
    return None


def _anthropic_price(model: str) -> TokenPrice | None:
    if "haiku-4-5" in model or "haiku-4.5" in model:
        return TokenPrice(1.00, 5.00, "anthropic_pricing_2026-05")
    if "haiku-3-5" in model or "haiku-3.5" in model:
        return TokenPrice(0.80, 4.00, "anthropic_pricing_2026-05")
    if "sonnet" in model:
        return TokenPrice(3.00, 15.00, "anthropic_pricing_2026-05")
    if "opus-4-6" in model or "opus-4-7" in model or "opus-4.6" in model or "opus-4.7" in model:
        return TokenPrice(5.00, 25.00, "anthropic_pricing_2026-05")
    if "opus" in model:
        return TokenPrice(15.00, 75.00, "anthropic_pricing_2026-05")
    return None
