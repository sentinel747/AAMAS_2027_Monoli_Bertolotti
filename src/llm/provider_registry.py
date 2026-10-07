from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .configured_provider import ConfiguredLLMProvider, load_local_env
from .provider import FallbackProvider, LLMProvider


PROVIDER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "llm_providers.json"

#: Consenso esplicito a costruire un provider che parla davvero con la rete.
#:
#: Le chiavi API dei provider registrati sono presenti nell'ambiente di sviluppo,
#: quindi senza questo interruttore qualunque test o run che risolva un provider
#: consumerebbe credito. Il default e' assente: la fabbrica restituisce il
#: fallback deterministico, e chi vuole chiamare davvero lo dichiara.
LLM_CONSENT_ENV = "MARSABM_ALLOW_LLM_CALLS"

#: I soli valori che valgono come consenso: e' un elenco di SI', non di NO.
#: Un elenco di negazioni lascia passare tutto cio' che non prevede — un refuso
#: come `flase`, il `None` che una shell scrive interpolando una variabile vuota,
#: un `disabled` scritto in buona fede — e proprio il caso ambiguo aprirebbe le
#: chiamate vere. Qui il caso ambiguo costa un fallback, non del credito.
_CONSENT_GRANTED = frozenset({"1", "true", "yes", "on"})


def real_calls_allowed() -> bool:
    """`True` solo se il consenso esplicito e' presente nell'ambiente."""
    return os.getenv(LLM_CONSENT_ENV, "").strip().lower() in _CONSENT_GRANTED


def load_provider_registry(path: str | Path = PROVIDER_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    load_local_env()
    config_path = Path(path)
    if not config_path.exists():
        return {}
    registry = json.loads(config_path.read_text(encoding="utf-8"))
    for _, cfg in registry.items():
        if not isinstance(cfg, dict):
            continue
        api_key_env = str(cfg.get("api_key_env", "") or "")
        provider_type = str(cfg.get("provider_type", "") or "")
        if provider_type == "ollama":
            cfg["is_configured"] = True
            cfg["missing_env"] = ""
            continue
        if api_key_env:
            is_set = bool(os.getenv(api_key_env, ""))
            cfg["is_configured"] = is_set
            cfg["missing_env"] = "" if is_set else api_key_env
        else:
            cfg["is_configured"] = False
            cfg["missing_env"] = ""
    return registry


def create_llm_provider(provider_id: str = "fallback", model: str | None = None, registry: dict[str, dict[str, Any]] | None = None) -> LLMProvider:
    normalized_provider = _normalize_provider_id(provider_id)
    if normalized_provider == "fallback":
        return FallbackProvider()
    if not real_calls_allowed():
        # Consenso assente: nessun provider reale, indipendentemente dalle chiavi.
        # Il controllo sta prima di `load_provider_registry`, che caricherebbe il
        # `.env` locale: il consenso si legge dall'ambiente del processo, non da
        # un file che la fabbrica stessa ha appena importato.
        return FallbackProvider()
    available = registry if registry is not None else load_provider_registry()
    provider_config = available.get(normalized_provider)
    if not provider_config:
        return FallbackProvider()
    selected_model = model or _first_model(provider_config)
    if not selected_model:
        return FallbackProvider()
    return ConfiguredLLMProvider(normalized_provider, provider_config, selected_model)


def resolve_offered_model(provider_id: str, provider_config: dict[str, Any], model: str) -> str:
    """`model` se il provider lo offre; altrimenti un errore, mai un altro modello.

    **Perche' un errore e non una sostituzione.** Il codice qui prendeva il primo
    modello elencato quando quello chiesto non c'era. Una run configurata su
    `qwen3.6:27b` girava allora su `gpt-oss:120b` senza una riga di avviso, e
    l'esperimento attribuiva a un modello cio' che aveva detto un altro: non un
    risultato peggiore, un risultato falso. Un elenco `available_models` vuoto
    significa "non dichiaro nulla" e lascia passare qualunque nome.
    """
    offered = provider_config.get("available_models") or []
    if not offered or model in offered:
        return model
    raise ValueError(
        f"il provider {provider_id!r} non offre il modello {model!r}. "
        f"Modelli dichiarati: {', '.join(str(m) for m in offered)}. "
        "Correggere la configurazione della run oppure aggiungere il modello a "
        "configs/llm_providers.json: sostituirlo qui in silenzio farebbe girare "
        "l'esperimento su un modello diverso da quello riportato."
    )


def create_agent_providers(config: dict[str, Any], llm_count: int) -> list[LLMProvider]:
    """I provider degli agenti LLM, con la temperatura della run applicata.

    **Perche' la temperatura si scrive qui e non nel costruttore.** Le opzioni di
    `ConfiguredProvider` si risolvono dal registro dei provider
    (`configs/llm_providers.json`), che e' una proprieta' della macchina e non
    della run: senza questo passaggio una campagna non potrebbe variare la
    temperatura senza modificare un file condiviso, e due run "a temperatura
    diversa" nello stesso registro sarebbero indistinguibili. E' la stessa
    scrittura sull'attributo che `src/governors/config.py` fa per il consiglio.

    `None` resta distinto da "assente": una temperatura configurata a `null`
    significa "non mandare il campo", che e' cio' che vogliono i modelli della
    famiglia o*/gpt-5, mentre l'assenza significa "usa il valore del registro".
    """
    registry = load_provider_registry()
    providers: list[LLMProvider] = []
    for spec in normalize_llm_assignments(config, llm_count, registry):
        provider = create_llm_provider(spec["provider"], spec.get("model"), registry)
        if "temperature" in spec and hasattr(provider, "temperature"):
            valore = spec["temperature"]
            valore = None if valore is None else float(valore)
            provider.temperature = valore
            if hasattr(provider, "options"):
                provider.options["temperature"] = valore
        providers.append(provider)
    return providers


def normalize_llm_assignments(
    config: dict[str, Any],
    llm_count: int,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # Nessun agente LLM: nessun provider da risolvere, e soprattutto nessun
    # errore da sollevare. Senza questa uscita anticipata una run interamente
    # rule-based con il registro assente passerebbe comunque dalla risoluzione
    # del provider di default, e la validazione stretta qui sotto la fermerebbe
    # per una configurazione che non usa.
    if llm_count <= 0:
        return []
    available = registry if registry is not None else load_provider_registry()
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents", {}), dict) else {}
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    raw_assignments = agents_cfg.get("llm_assignments") or llm_cfg.get("assignments") or []
    if not isinstance(raw_assignments, list):
        raw_assignments = []
    default_provider = _normalize_provider_id(str(llm_cfg.get("provider", "gpt")))
    if default_provider not in available and default_provider != "fallback":
        default_provider = next(iter(available), "fallback")
    default_model = str(llm_cfg.get("model") or _first_model(available.get(default_provider, {})))

    assignments: list[dict[str, Any]] = []
    for index in range(max(0, llm_count)):
        raw = raw_assignments[index] if index < len(raw_assignments) and isinstance(raw_assignments[index], dict) else {}
        provider = _normalize_provider_id(str(raw.get("provider") or raw.get("provider_id") or default_provider))
        if provider not in available and provider != "fallback":
            provider = default_provider
        provider_config = available.get(provider, {})
        model = str(raw.get("model") or (default_model if provider == default_provider else _first_model(provider_config)))
        if provider != "fallback" and model:
            model = resolve_offered_model(provider, provider_config, model)
        voce: dict[str, Any] = {"provider": provider, "model": model or "fallback"}
        # La temperatura viaggia con l'assegnazione quando c'e', per agente e non
        # per registro; il livello di run (`llm.temperature`) vale per tutti e
        # cede a quello per agente. Assente resta assente: e' cio' che distingue
        # "usa il registro" da "manda null".
        if "temperature" in raw:
            voce["temperature"] = raw["temperature"]
        elif "temperature" in llm_cfg:
            voce["temperature"] = llm_cfg["temperature"]
        assignments.append(voce)
    return assignments


def _normalize_provider_id(provider_id: str) -> str:
    provider = (provider_id or "fallback").strip()
    if provider == "openai":
        return "gpt"
    if provider in {"none", "rule", "disabled"}:
        return "fallback"
    return provider


def _first_model(provider_config: dict[str, Any] | None) -> str:
    models = (provider_config or {}).get("available_models", [])
    return str(models[0]) if models else ""
