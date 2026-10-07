from __future__ import annotations

import asyncio
from email.utils import parsedate_to_datetime
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from .provider import FallbackProvider, LLMResponse
from .pricing import estimate_token_cost
from .thinking import campi as campi_thinking, normalizza as normalizza_thinking
from .tls import contesto_ssl


_RATE_STATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT: dict[str, float] = {}

#: Modelli che rifiutano `temperature` e vogliono `max_completion_tokens`.
_REASONING_FAMILY = ("o1", "o3", "o4", "gpt-5")

#: Le opzioni che una voce di provider puo' fissare e un modello sovrascrivere.
#: `extra_body` non e' qui perche' i due livelli si FONDONO invece di sostituirsi
#: (vedi `ConfiguredLLMProvider._extra_body`), e leggerlo anche da qui darebbe due
#: sorgenti per lo stesso campo.
_OVERRIDABLE_OPTIONS = ("temperature", "top_p", "max_tokens", "timeout_seconds")


def resolve_model_options(config: dict[str, Any], model: str) -> dict[str, Any]:
    """Le opzioni effettive per `model`: prima la voce del provider, poi le sue.

    Un modello non elencato in `model_options` non e' un errore: eredita la voce
    del provider, ed e' il caso di ogni provider che non ne dichiara.
    """
    resolved: dict[str, Any] = {
        key: config[key] for key in _OVERRIDABLE_OPTIONS if key in config
    }
    per_model = config.get("model_options")
    if isinstance(per_model, dict):
        specific = per_model.get(model)
        if isinstance(specific, dict):
            resolved.update(specific)
    return resolved


def load_local_env() -> None:
    for path in _candidate_env_paths():
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class ConfiguredLLMProvider:
    """Un client verso un provider configurato, con le opzioni del suo modello.

    **Perche' le opzioni non sono piu' costanti nel codice.** Temperatura, tetto
    dei token e timeout erano cablati a 0,2 / 700 / 45 s. Vanno bene per un
    modello che risponde subito con un JSON corto, e sono tre modi diversi di
    rovinare in silenzio una run con un modello che ragiona: 700 token li consuma
    il ragionamento prima di arrivare al JSON, 45 s scadono prima della risposta,
    e 0,2 non e' la temperatura per cui quel modello e' stato calibrato. In tutti
    e tre i casi non esce un errore: esce il fallback, cioe' una run che sembra
    governata e non lo e'.

    Le opzioni si risolvono in tre livelli, dal piu' specifico: `model_options`
    del modello, poi la voce del provider, poi il default del costruttore.
    """

    def __init__(self, provider_id: str, config: dict[str, Any], model: str, max_tokens: int = 700, timeout: int = 45):
        load_local_env()
        self.provider_id = provider_id
        self.config = config
        self.model = model
        options = resolve_model_options(config, model)
        self.options = options
        self.max_tokens = int(options.get("max_tokens", max_tokens) or max_tokens)
        self.timeout = float(options.get("timeout_seconds", timeout) or timeout)
        # `None` e' distinto da "assente": una temperatura configurata a `null`
        # significa "non mandare il campo", che e' cio' che vogliono i modelli
        # della famiglia o*/gpt-5, mentre l'assenza significa "usa 0,2 come
        # prima". Un solo valore non saprebbe dire le due cose.
        self.temperature = options["temperature"] if "temperature" in options else 0.2
        self.top_p = options.get("top_p")
        self.provider_type = str(config.get("provider_type", "openai"))
        self.base_url = str(config.get("base_url") or _default_base_url(self.provider_type)).rstrip("/")
        self.api_key_env = str(config.get("api_key_env", ""))
        # **`strip()` non e' cosmesi.** Una variabile d'ambiente impostata da
        # Windows con un a-capo finale produce un valore di intestazione HTTP
        # non valido, e urllib solleva `ValueError` PRIMA di aprire la
        # connessione: misurato su `QWEN_CLOUD_API_KEY`, 117 caratteri di cui
        # l'ultimo era `\\n`. L'errore non nomina la chiave e sembra un difetto
        # del client.
        self.api_key = (os.getenv(self.api_key_env, "") or "").strip() if self.api_key_env else ""
        self.fallback = FallbackProvider()
        self.rate_limit_key = f"{self.provider_id}:{self.model}"
        # **Il ragionamento e' un solo interruttore, tradotto per API.** Ogni
        # fornitore lo spegne in un modo diverso e sbagliarlo non da' errore:
        # da' una risposta troncata o una chiamata appesa. Vedi `src/llm/thinking.py`.
        self.thinking_style = str(config.get("thinking_style", "none") or "none")
        self.thinking = normalizza_thinking(options.get("thinking"))
        self._campi_thinking, avviso = campi_thinking(
            self.thinking_style, self.model, self.thinking
        )
        if avviso:
            print(f"[{self.provider_id}:{self.model}] {avviso}", flush=True)

    def complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        if self.provider_type != "ollama" and not self.api_key:
            return self.fallback.complete_json(prompt, schema_hint)
        try:
            if self.provider_type == "openai":
                response = self._openai_compatible(prompt)
            elif self.provider_type == "google":
                response = self._google(prompt)
            elif self.provider_type == "anthropic":
                response = self._anthropic(prompt)
            elif self.provider_type == "ollama":
                response = self._ollama(prompt)
            else:
                return self.fallback.complete_json(prompt, schema_hint)
        except (TimeoutError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return self._fallback_after_error(prompt, schema_hint, exc)
        response.text = _clean_json_response(response.text, self.config)
        return response

    async def async_complete_json(self, prompt: str, schema_hint: dict | None = None) -> LLMResponse:
        if self.provider_type != "ollama" and not self.api_key:
            return await self.fallback.async_complete_json(prompt, schema_hint)
        try:
            if self.provider_type == "openai":
                response = await self._async_openai_compatible(prompt)
            elif self.provider_type == "google":
                response = await self._async_google(prompt)
            elif self.provider_type == "anthropic":
                response = await self._async_anthropic(prompt)
            elif self.provider_type == "ollama":
                response = await self._async_ollama(prompt)
            else:
                return await self.fallback.async_complete_json(prompt, schema_hint)
        except (httpx.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return await self._async_fallback_after_error(prompt, schema_hint, exc)
        response.text = _clean_json_response(response.text, self.config)
        return response

    def _openai_payload(self, prompt: str) -> dict[str, Any]:
        """Il corpo della richiesta, costruito una volta per i due percorsi.

        Sincrono e asincrono lo componevano ciascuno per conto suo, con le stesse
        dodici righe ripetute. I governatori usano l'asincrono e il controllo di
        preflight il sincrono: due copie significano che il preflight puo'
        dichiarare funzionante un corpo diverso da quello che la run manda.
        """
        reasoning_family = self.model.startswith(_REASONING_FAMILY)
        token_field = "max_completion_tokens" if reasoning_family else "max_tokens"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            token_field: self.max_tokens,
        }
        if not reasoning_family:
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.top_p is not None:
                payload["top_p"] = self.top_p
        # `response_format` e' l'unico campo che questo codice manda senza che
        # nessuno lo abbia chiesto, ed e' anche quello che un endpoint
        # OpenAI-compatibile non ufficiale ha piu' probabilita' di rifiutare.
        # L'interruttore esiste perche' la diagnosi si fa con `check_provider`,
        # non riscrivendo il codice a meta' di un esperimento.
        if not self.model.startswith("o1") and not self.config.get("disable_response_format"):
            payload["response_format"] = {"type": "json_object"}
        payload.update(self._extra_body())
        # Dopo `_extra_body` e indipendente da esso: su `api.openai.com` quello
        # ritorna vuoto per progetto, ma `reasoning_effort` va mandato lo stesso
        # ai modelli che ragionano.
        for chiave, valore in self._campi_thinking.items():
            if chiave == "chat_template_kwargs" and isinstance(payload.get(chiave), dict):
                fuso = dict(payload[chiave])
                fuso.update(valore)
                payload[chiave] = fuso
            else:
                payload[chiave] = valore
        return payload

    # -- il secondo protocollo OpenAI: /responses invece di /chat/completions --
    #
    # **Sono due API diverse, non due indirizzi.** Cambiano il nome del campo
    # che porta il prompt (`input` invece di `messages`), quello del tetto di
    # token (`max_output_tokens` invece di `max_tokens`), la forma della
    # risposta (una LISTA `output` di elementi tipizzati invece di `choices`) e
    # perfino i nomi nell'uso (`input_tokens` invece di `prompt_tokens`).
    # Puntare `base_url` a `/responses` senza toccare il resto darebbe un 400 a
    # ogni chiamata --- e per come e' fatto questo esperimento, un 400 a ogni
    # chiamata non e' un errore visibile: e' una run identica alla baseline.

    def _usa_responses(self) -> bool:
        return str(self.config.get("api_style", "chat")).strip().lower() == "responses"

    def _openai_request(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """URL e corpo, scelti dal protocollo dichiarato nel registro."""
        if not self._usa_responses():
            return f"{self.base_url}/chat/completions", self._openai_payload(prompt)
        return f"{self.base_url}/responses", self._responses_payload(prompt)

    def _responses_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if not self.config.get("disable_response_format"):
            # L'equivalente di `response_format` in questa API sta annidato
            # sotto `text`, ed e' il campo che un gateway non ufficiale ha piu'
            # probabilita' di rifiutare: si spegne con lo stesso interruttore.
            payload["text"] = {"format": {"type": "json_object"}}
        payload.update(self._extra_body())
        for chiave, valore in self._campi_thinking.items():
            if chiave == "reasoning_effort":
                # Qui il ragionamento e' un oggetto, non un campo piatto.
                fuso = dict(payload.get("reasoning") or {})
                fuso["effort"] = valore
                payload["reasoning"] = fuso
            elif chiave == "chat_template_kwargs":
                # Estensione della sola API di chat: mandarla qui e' un campo
                # sconosciuto, cioe' un 400 su un endpoint severo.
                continue
            else:
                payload[chiave] = valore
        return payload

    def _openai_response(self, data: dict[str, Any]) -> LLMResponse:
        """Legge la risposta di ENTRAMBI i protocolli.

        Tollerante di proposito: questo endpoint e' un gateway di terze parti e
        non l'API ufficiale, e non possiamo provarlo finche' la chiave nuova non
        arriva. Se la forma non e' quella attesa, il fallback e' l'altra forma
        --- non una risposta vuota, che si travestirebbe da governo silenzioso.
        """
        testo = ""
        grezzo = data.get("output_text")
        if isinstance(grezzo, str) and grezzo.strip():
            testo = grezzo
        elif isinstance(data.get("output"), list):
            pezzi: list[str] = []
            for elemento in data["output"]:
                if not isinstance(elemento, dict):
                    continue
                # Un modello che ragiona mette PRIMA un elemento `reasoning`:
                # prendere `output[0]` restituirebbe il ragionamento al posto
                # della risposta, o una stringa vuota.
                for parte in elemento.get("content") or []:
                    if isinstance(parte, dict) and isinstance(parte.get("text"), str):
                        pezzi.append(parte["text"])
            testo = "".join(pezzi)
        if not testo:
            message = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
            testo = str(message.get("content") or "")

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        tokens_out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        tokens_total = int(usage.get("total_tokens", 0) or tokens_in + tokens_out)
        cost, price = estimate_token_cost(
            self.provider_type, self.provider_id, self.model, tokens_in, tokens_out
        )
        return LLMResponse(
            text=testo or "{}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_usd=cost,
            pricing_source=price.source if price else "",
            api_call_attempted=True,
            provider=f"{self.provider_id}:{self.model}",
        )

    async def _async_openai_compatible(self, prompt: str) -> LLMResponse:
        url, payload = self._openai_request(prompt)
        data = await self._async_request_json(
            url,
            payload,
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        return self._openai_response(data)

    def _google_request(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """URL e corpo per Gemini, costruiti una volta per i due percorsi.

        `temperature` e `topP` non venivano mandate affatto: le opzioni per
        modello esistevano ma su questo provider erano lettera morta, e ogni
        chiamata usava i default del servizio. Solo `topK` passava, e per la sola
        via di `extra_body`.
        """
        model = urllib.parse.quote(self.model, safe="")
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": self.max_tokens,
        }
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["topP"] = self.top_p
        extra = self._extra_body()
        if "top_k" in extra:
            generation_config["topK"] = extra["top_k"]
        # **Il ragionamento consuma il budget di uscita (2026-09-01).** Sui
        # modelli 2.5 e successivi i token di *thinking* si contano dentro
        # `maxOutputTokens`: misurato su `gemini-2.5-flash`, una risposta da
        # 692 token di cui circa 590 di pensiero e il JSON troncato a meta' di
        # una chiave. Il guasto e' silenzioso due volte, perche' la chiamata
        # riesce e il governatore legge "risposta non interpretabile", cioe'
        # "nessuna direttiva". `thinking_budget: 0` lo spegne; un intero
        # positivo lo limita; -1 lo lascia dinamico.
        budget = self._campi_thinking.get("thinking_budget")
        if budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": int(budget)}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{prompt}\n\nReturn one valid JSON object only."}]}],
            "generationConfig": generation_config,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(self.api_key)}"
        return url, payload

    async def _async_google(self, prompt: str) -> LLMResponse:
        url, payload = self._google_request(prompt)
        data = await self._async_request_json(url, payload, {})
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata", {})
        tokens_in = int(usage.get("promptTokenCount", 0) or 0)
        tokens_out = int(usage.get("candidatesTokenCount", 0) or 0)
        tokens_total = int(usage.get("totalTokenCount", 0) or tokens_in + tokens_out)
        cost, price = estimate_token_cost(self.provider_type, self.provider_id, self.model, tokens_in, tokens_out)
        return LLMResponse(
            text="".join(str(part.get("text", "")) for part in parts) or "{}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_usd=cost,
            pricing_source=price.source if price else "",
            api_call_attempted=True,
            provider=f"{self.provider_id}:{self.model}",
        )

    async def _async_anthropic(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": f"{prompt}\n\nReturn only a valid JSON object."}],
        }
        data = await self._async_request_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        usage = data.get("usage", {})
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        tokens_total = tokens_in + tokens_out
        cost, price = estimate_token_cost(self.provider_type, self.provider_id, self.model, tokens_in, tokens_out)
        return LLMResponse(
            text="".join(str(item.get("text", "")) for item in data.get("content", []) if item.get("type") == "text") or "{}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_usd=cost,
            pricing_source=price.source if price else "",
            api_call_attempted=True,
            provider=f"{self.provider_id}:{self.model}",
        )

    async def _async_ollama(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }
        data = await self._async_request_json(f"{self.base_url}/api/chat", payload, {})
        message = data.get("message") or {}
        return LLMResponse(text=str(message.get("content") or "{}"), api_call_attempted=True, provider=f"{self.provider_id}:{self.model}")

    async def _async_request_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **headers}
        last_exc: Exception | None = None
        # `trust_env=False` ignora di proposito SSL_CERT_FILE, quindi il
        # contesto va passato per mano: vedi `src/llm/tls.py`.
        async with httpx.AsyncClient(
            timeout=self.timeout, trust_env=False, verify=contesto_ssl()
        ) as client:
            for attempt in range(self._max_retries() + 1):
                await self._async_wait_for_rate_slot()
                try:
                    response = await client.post(url, json=payload, headers=request_headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt >= self._max_retries() or not self._is_retriable_httpx(exc):
                        raise
                    await asyncio.sleep(self._retry_delay_seconds(exc, attempt))
        raise last_exc or RuntimeError("request failed without an exception")

    def _openai_compatible(self, prompt: str) -> LLMResponse:
        url, payload = self._openai_request(prompt)
        data = self._request_json(
            url,
            payload,
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        return self._openai_response(data)

    def _google(self, prompt: str) -> LLMResponse:
        url, payload = self._google_request(prompt)
        data = self._request_json(url, payload, {})
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata", {})
        tokens_in = int(usage.get("promptTokenCount", 0) or 0)
        tokens_out = int(usage.get("candidatesTokenCount", 0) or 0)
        tokens_total = int(usage.get("totalTokenCount", 0) or tokens_in + tokens_out)
        cost, price = estimate_token_cost(self.provider_type, self.provider_id, self.model, tokens_in, tokens_out)
        return LLMResponse(
            text="".join(str(part.get("text", "")) for part in parts) or "{}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_usd=cost,
            pricing_source=price.source if price else "",
            api_call_attempted=True,
            provider=f"{self.provider_id}:{self.model}",
        )

    def _anthropic(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": f"{prompt}\n\nReturn only a valid JSON object."}],
        }
        data = self._request_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        usage = data.get("usage", {})
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        tokens_total = tokens_in + tokens_out
        cost, price = estimate_token_cost(self.provider_type, self.provider_id, self.model, tokens_in, tokens_out)
        return LLMResponse(
            text="".join(str(item.get("text", "")) for item in data.get("content", []) if item.get("type") == "text") or "{}",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            cost_usd=cost,
            pricing_source=price.source if price else "",
            api_call_attempted=True,
            provider=f"{self.provider_id}:{self.model}",
        )

    def _ollama(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }
        data = self._request_json(f"{self.base_url}/api/chat", payload, {})
        message = data.get("message") or {}
        return LLMResponse(text=str(message.get("content") or "{}"), api_call_attempted=True, provider=f"{self.provider_id}:{self.model}")

    def _fallback_after_error(self, prompt: str, schema_hint: dict | None, exc: Exception) -> LLMResponse:
        response = self.fallback.complete_json(prompt, schema_hint)
        response.api_call_attempted = True
        response.error = _format_provider_error(exc)
        return response

    async def _async_fallback_after_error(self, prompt: str, schema_hint: dict | None, exc: Exception) -> LLMResponse:
        response = await self.fallback.async_complete_json(prompt, schema_hint)
        response.api_call_attempted = True
        response.error = _format_provider_error(exc)
        return response

    def _request_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **headers}
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=request_headers, method="POST")
        # Il contesto e' esplicito e non quello predefinito: vedi `src/llm/tls.py`.
        # Senza, su una macchina con la scansione HTTPS dell'antivirus attiva
        # ogni chiamata muore in `CERTIFICATE_VERIFY_FAILED`, e il governatore
        # — che per progetto non solleva mai — la traduce in "direttiva
        # assente": una run `llm` di ore, zero chiamate riuscite, esito
        # identico alla baseline e nessun errore da nessuna parte.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=contesto_ssl()),
        )
        last_exc: Exception | None = None
        for attempt in range(self._max_retries() + 1):
            self._wait_for_rate_slot()
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt >= self._max_retries() or not self._is_retriable_urllib(exc):
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        raise last_exc or RuntimeError("request failed without an exception")

    async def _async_wait_for_rate_slot(self) -> None:
        interval = self._min_interval_seconds()
        if interval <= 0:
            return
        while True:
            wait_seconds = self._reserve_rate_slot(interval)
            if wait_seconds <= 0:
                return
            await asyncio.sleep(wait_seconds)

    def _wait_for_rate_slot(self) -> None:
        interval = self._min_interval_seconds()
        if interval <= 0:
            return
        while True:
            wait_seconds = self._reserve_rate_slot(interval)
            if wait_seconds <= 0:
                return
            time.sleep(wait_seconds)

    def _reserve_rate_slot(self, interval: float) -> float:
        with _RATE_STATE_LOCK:
            now = time.monotonic()
            next_at = _NEXT_REQUEST_AT.get(self.rate_limit_key, 0.0)
            if next_at <= now:
                _NEXT_REQUEST_AT[self.rate_limit_key] = now + interval
                return 0.0
            return next_at - now

    def _min_interval_seconds(self) -> float:
        if "min_interval_seconds" in self.config:
            return max(0.0, float(self.config.get("min_interval_seconds") or 0.0))
        if self.provider_type == "anthropic":
            return 1.1
        if self.provider_type == "google":
            return 1.5
        if self.provider_type == "openai":
            return 1.0
        return 0.0

    def _max_retries(self) -> int:
        return max(0, int(self.config.get("max_retries", 3)))

    def _retry_statuses(self) -> set[int]:
        configured = self.config.get("retry_statuses")
        if isinstance(configured, list):
            return {int(status) for status in configured}
        if self.provider_type == "anthropic":
            return {429, 500, 529}
        return {429, 500, 502, 503, 504}

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        header_delay = _retry_after_seconds(exc)
        body_delay = _retry_delay_from_error_body(exc)
        if header_delay is not None:
            return min(header_delay, self._max_retry_delay_seconds())
        if body_delay is not None:
            return min(body_delay, self._max_retry_delay_seconds())
        base = float(self.config.get("retry_backoff_base_seconds", 1.0))
        jitter = random.uniform(0.0, float(self.config.get("retry_jitter_seconds", 0.5)))
        return min(base * (2**attempt) + jitter, self._max_retry_delay_seconds())

    def _max_retry_delay_seconds(self) -> float:
        return max(1.0, float(self.config.get("max_retry_delay_seconds", 30.0)))

    def _is_retriable_httpx(self, exc: httpx.HTTPError) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in self._retry_statuses()
        return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError))

    def _is_retriable_urllib(self, exc: urllib.error.URLError) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in self._retry_statuses()
        return True

    def _extra_body(self) -> dict[str, Any]:
        if self.config.get("disable_extra_body"):
            return {}
        if "api.openai.com" in self.base_url:
            return {}
        extra = self.config.get("extra_body")
        merged = dict(extra) if isinstance(extra, dict) else {}
        # Fusione e non sostituzione: la voce del provider porta cio' che vale per
        # tutti i suoi modelli, `model_options` cio' che vale solo per uno.
        model_extra = self.options.get("extra_body")
        if isinstance(model_extra, dict):
            merged.update(model_extra)
        return merged


def _candidate_env_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    return [Path.cwd() / ".env", project_root / ".env"]


def _default_base_url(provider_type: str) -> str:
    if provider_type == "openai":
        return "https://api.openai.com/v1"
    if provider_type == "ollama":
        return "http://localhost:11434"
    return ""


def _clean_json_response(text: str, config: dict[str, Any]) -> str:
    cleaned = str(text or "").strip()
    if config.get("strip_thinking_tags"):
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    extracted = _extract_json_object(cleaned)
    if config.get("enforce_tag_only_output") or extracted:
        return extracted or "{}"
    return cleaned or "{}"


def _extract_json_object(text: str) -> str:
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
    return ""


def _format_provider_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:500] if exc.response is not None else ""
        return f"{type(exc).__name__}: {exc.response.status_code} {body}".strip()
    return f"{type(exc).__name__}: {exc}"


def _retry_after_seconds(exc: Exception) -> float | None:
    headers: Any = None
    if isinstance(exc, httpx.HTTPStatusError):
        headers = exc.response.headers
    elif isinstance(exc, urllib.error.HTTPError):
        headers = exc.headers
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, IndexError, OverflowError):
            return None


def _retry_delay_from_error_body(exc: Exception) -> float | None:
    body = ""
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text
    elif isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", body, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None
