# -*- coding: utf-8 -*-
"""Un solo interruttore per il ragionamento, tradotto in cio' che ogni API vuole.

**Perche' esiste.** Ogni fornitore spegne il ragionamento in un modo diverso, e
sbagliarlo non da' un errore: da' una risposta troncata o una chiamata appesa,
che il governatore legge come "nessuna direttiva". Misurato su questo progetto,
due volte lo stesso giorno:

- `gemini-2.5-flash` conta i token di pensiero dentro `maxOutputTokens`: la
  risposta usciva troncata a meta' di una chiave JSON, e la chiamata riusciva;
- `qwen3.7-plus` su DashScope pretende `enable_thinking` esplicito e, per una
  chiamata non in streaming, **falso**: senza, la richiesta resta appesa fino
  al timeout.

Nessuna delle due condizioni e' visibile dal nome del modello. Perche' non si
ripeta, il livello si sceglie una volta in forma neutra e questo modulo lo
traduce nei campi che quella specifica API vuole, secondo lo `thinking_style`
dichiarato nel registro dei provider.

**Il default e' `off` ovunque**, per una ragione sperimentale e non di costo:
il ragionamento nascosto introduce una variabilita' che non appartiene al
trattamento, e in un confronto appaiato fra bracci si sommerebbe alla varianza
fra semi senza che nessuno l'abbia dichiarata.
"""

from __future__ import annotations

#: I livelli neutri, dal piu' spento al piu' esteso. `dynamic` lascia decidere
#: al modello, dove l'API lo permette.
LIVELLI = ("off", "low", "medium", "high", "dynamic")

#: Come ciascuna famiglia di API esprime il ragionamento. Dichiarato nel
#: registro (`thinking_style`) e non dedotto dall'URL: una deduzione sbaglia in
#: silenzio la prima volta che un endpoint cambia indirizzo.
STILI = (
    "none",              # il modello non ragiona: il livello non ha effetto
    "openai_effort",     # reasoning_effort: minimal|low|medium|high
    "google_budget",     # generationConfig.thinkingConfig.thinkingBudget
    "anthropic_budget",  # thinking: {type, budget_tokens}
    "dashscope_enable",  # enable_thinking + thinking_budget
    "ollama_effort",     # reasoning_effort + chat_template_kwargs.enable_thinking
)

#: Budget in token per i livelli che ne vogliono uno. Non sono tarati: sono le
#: soglie che le rispettive documentazioni indicano come minimo utile, medio e
#: ampio, e vanno cambiate qui se un modello le smentisce.
_BUDGET = {"low": 1024, "medium": 8192, "high": 24576}

#: Le famiglie OpenAI che accettano `reasoning_effort`. Sugli altri modelli il
#: campo e' rifiutato, quindi non va mandato affatto.
_FAMIGLIE_RAGIONANTI = ("o1", "o3", "o4", "gpt-5")

#: Le famiglie che accettano `reasoning_effort` ma NON il valore `minimal`.
#: Sono le serie `o*`: `minimal` e' stato introdotto con gpt-5 e su di loro e'
#: un 400. Verificato a mano il 2026-09-03 su o1, o3-mini e o4-mini.
_SENZA_MINIMAL = ("o1", "o3", "o4")

#: Le famiglie servite via Ollama che NON sanno spegnere il ragionamento:
#: per loro `off` vale `low`, il minimo che accettano.
_SENZA_SPEGNIMENTO = ("gpt-oss",)


def normalizza(livello: str | None) -> str:
    """Un livello sconosciuto vale `off`, che e' il default dichiarato."""
    valore = (livello or "off").strip().lower()
    return valore if valore in LIVELLI else "off"


def _modello_ragiona(model: str) -> bool:
    nome = (model or "").strip().lower()
    return any(nome.startswith(famiglia) for famiglia in _FAMIGLIE_RAGIONANTI)


def campi(stile: str, model: str, livello: str | None) -> tuple[dict, str]:
    """I campi da fondere nella richiesta, e un avviso da stampare una volta.

    L'avviso non e' decorativo: e' il posto in cui una combinazione impossibile
    — ragionamento acceso su una chiamata che DashScope accetta solo spenta —
    si dichiara prima della run invece di manifestarsi come timeout.
    """
    livello = normalizza(livello)
    stile = (stile or "none").strip().lower()
    if stile not in STILI:
        return {}, f"thinking_style={stile!r} sconosciuto: il livello e' ignorato."
    if stile == "none":
        return {}, ""

    if stile == "google_budget":
        if livello == "dynamic":
            return {"thinking_budget": -1}, ""
        if livello == "off":
            return {"thinking_budget": 0}, ""
        return {"thinking_budget": _BUDGET[livello]}, ""

    if stile == "anthropic_budget":
        if livello in ("off", "dynamic"):
            # Anthropic non ha una modalita' dinamica: `dynamic` vale come
            # spento, e lo si dice invece di far finta.
            avviso = "" if livello == "off" else "anthropic non ha 'dynamic': ragionamento spento."
            return {}, avviso
        return {"thinking": {"type": "enabled", "budget_tokens": _BUDGET[livello]}}, ""

    if stile == "dashscope_enable":
        if livello == "off":
            return {"enable_thinking": False}, ""
        return (
            {"enable_thinking": True, "thinking_budget": _BUDGET.get(livello, 8192)},
            "DashScope rifiuta il ragionamento sulle chiamate NON in streaming: "
            f"thinking={livello} rischia un 400 o un timeout. Con questo motore "
            "conviene 'off'.",
        )

    if stile == "openai_effort":
        if not _modello_ragiona(model):
            avviso = (
                ""
                if livello == "off"
                else f"il modello {model!r} non ragiona: thinking={livello} e' ignorato."
            )
            return {}, avviso
        if livello == "dynamic":
            return {}, ""  # nessun campo: decide il servizio
        if livello != "off":
            return {"reasoning_effort": livello}, ""
        # **`minimal` non esiste fuori dalla famiglia gpt-5.** Misurato il
        # 2026-09-03 su o1, o3-mini e o4-mini: HTTP 400, "does not support
        # 'minimal' with this model. Supported values are: 'low', 'medium',
        # 'high', and 'xhigh'". Tre modelli su sei del fornitore erano quindi
        # inutilizzabili come governatori, e il sintomo non era un errore ma una
        # policy vuota --- cioe' una run indistinguibile dalla baseline.
        #
        # Per un modello che non sa spegnere il ragionamento, "off" significa
        # il minimo che accetta, e lo si dichiara invece di fingere.
        if _SENZA_MINIMAL and model.strip().lower().startswith(_SENZA_MINIMAL):
            return (
                {"reasoning_effort": "low"},
                f"{model} non accetta reasoning_effort='minimal': "
                "thinking=off vale 'low', il minimo che offre.",
            )
        return {"reasoning_effort": "minimal"}, ""

    # ollama_effort: la farm accetta entrambe le forme, e le vuole entrambe.
    #
    # **`off` e' "none", non "low" (misurato il 2026-09-07).** Su Ollama 0.32
    # `reasoning_effort: low` e' ragionamento ACCESO a sforzo basso: qwen3.6:27b
    # consumava tutti i 600 token di risposta in pensiero e non arrivava mai al
    # JSON (216 s a chiamata sul prompt del governatore). Con "none" risponde
    # in 11 s senza una riga di ragionamento, e `chat_template_kwargs` copre
    # l'endpoint vLLM. Eccezione: gpt-oss non sa spegnere il ragionamento e
    # "low" e' il suo minimo; resta "low" anche perche' e' la forma con cui
    # tutte le campagne gpt-oss sono state eseguite.
    if livello == "off":
        minimo = "low" if model.strip().lower().startswith(_SENZA_SPEGNIMENTO) else "none"
        return (
            {"reasoning_effort": minimo, "chat_template_kwargs": {"enable_thinking": False}},
            "",
        )
    if livello == "dynamic":
        return {}, ""
    return (
        {"reasoning_effort": livello, "chat_template_kwargs": {"enable_thinking": True}},
        "",
    )


def descrivi(stile: str, model: str, livello: str | None) -> str:
    """Una riga leggibile per il preflight e per i registri di run."""
    valori, avviso = campi(stile, model, livello)
    reso = ", ".join(f"{k}={v}" for k, v in sorted(valori.items())) or "nessun campo"
    coda = f"  [{avviso}]" if avviso else ""
    return f"thinking={normalizza(livello)} -> {reso}{coda}"
