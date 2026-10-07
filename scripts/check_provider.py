"""Preflight di un provider: una chiamata vera, prima di spenderne millecinquecento.

**Questo script fa chiamate API e va eseguito solo dall'utente**, con
`MARSABM_ALLOW_LLM_CALLS=1` nella propria shell. Claude Code non lo esegue: la
policy di questo repository vieta di consumare credito esterno.

**Perche' esiste.** Ogni modo in cui la configurazione di un governatore puo'
essere sbagliata produce, a run, la stessa cosa: nessun errore. `LLMProposer` non
solleva mai per progetto, `ConfiguredLLMProvider` risponde con il fallback quando
la chiave manca, e una risposta fuori schema diventa "direttiva assente". Il
risultato e' una run che si chiama `llm`, dura ore e finisce identica alla
baseline. Le validazioni in `src.governors.config` fermano gli errori
DICHIARATIVI (provider inesistente, modello non offerto, chiave assente); questo
script prova cio' che nessuna validazione puo' sapere senza chiedere: che
l'endpoint risponda, che il modello esista davvero sulla farm, che accetti
`response_format`, e soprattutto **quanto ci mette**.

**La latenza e' il numero che serve.** Il consiglio non aspetta: al confine di
tick successivo, se la chiamata precedente non e' tornata, la ANNULLA e ne avvia
un'altra (`Council.advance`). Se la latenza supera `cadence_steps x tempo di
passo`, ogni tick manca il suo aggiornamento, nessuna direttiva entra mai in
vigore e la run e' la baseline con in piu' il costo delle chiamate. Con un passo
da ~95 ms e cadenza 20 il budget e' ~1,9 s: un modello con `reasoning_effort:
high` non ci sta dentro nemmeno da lontano. Questo script misura la latenza e
dice quale cadenza la rende sostenibile.

Esempi::

    python scripts/check_provider.py --provider gpu_farm --model qwen3.6:27b
    python scripts/check_provider.py --provider gpu_farm --model gpt-oss:20b \\
        --governor --repeat 3 --step-ms 95
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.governors.llm_arm import build_governor_prompt  # noqa: E402
from src.governors.observation import ColonyPicture  # noqa: E402
from src.governors.policy import Bounds, parse_policy  # noqa: E402
from src.llm.provider import FallbackProvider  # noqa: E402
from src.llm.provider_registry import (  # noqa: E402
    LLM_CONSENT_ENV,
    create_llm_provider,
    load_provider_registry,
    real_calls_allowed,
    resolve_offered_model,
)

#: Un quadro plausibile e piccolo: serve a far rispondere il modello, non a
#: rappresentare una colonia vera. Gli aggregati sono quelli che il governatore
#: riceverebbe da una colonia su tre celle.
_SAMPLE_PICTURE = ColonyPicture(
    step=41,
    population=112,
    metrics={
        "food_margin": 1.32,
        "material_margin": 0.87,
        "colony_prosperity_index": 0.64,
        "power_margin": 1.51,
    },
    indicators={
        "food_per_occupant": {"mean": 1.4, "std": 0.9, "min": 0.2, "max": 2.4},
        "ice_per_occupant": {"mean": 0.8, "std": 0.3, "min": 0.4, "max": 1.1},
        "material_per_occupant": {"mean": 2.1, "std": 1.2, "min": 0.5, "max": 3.6},
        "minerals_per_occupant": {"mean": 1.7, "std": 0.6, "min": 0.9, "max": 2.3},
        "occupants": {"mean": 37.3, "std": 5.7, "min": 30.0, "max": 44.0},
    },
    population_stats={
        "health_mean": 0.91, "hydration_mean": 0.78, "satiety_mean": 0.74,
        "morale_mean": 0.83, "fatigue_mean": 0.31, "stress_mean": 0.22,
    },
    structures={"habitat": 5, "greenhouse": 6, "solar_array": 2, "shelter": 1},
    n_cells=3,
)


def _describe(provider) -> str:
    return (
        f"provider={getattr(provider, 'provider_id', '?')} "
        f"model={getattr(provider, 'model', '?')} "
        f"base_url={getattr(provider, 'base_url', '-')} "
        f"timeout={getattr(provider, 'timeout', '-')}s "
        f"max_tokens={getattr(provider, 'max_tokens', '-')} "
        f"temperature={getattr(provider, 'temperature', '-')}"
    )


async def _one_call(provider, prompt: str) -> tuple[float, object]:
    started = time.perf_counter()
    response = await provider.async_complete_json(prompt)
    return time.perf_counter() - started, response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, help="id nel registro, es. gpu_farm")
    parser.add_argument("--model", required=True, help="modello, es. qwen3.6:27b")
    parser.add_argument(
        "--governor", action="store_true",
        help="manda il prompt VERO del governatore e prova a interpretarne la risposta",
    )
    parser.add_argument("--repeat", type=int, default=1, help="chiamate da cui prendere la latenza mediana")
    parser.add_argument(
        "--effort", choices=("none", "low", "medium", "high"),
        help="sovrascrive reasoning_effort per questa sola prova; 'none' lo toglie",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="sovrascrive il tetto dei token per questa sola prova",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, metavar="T",
        help="sovrascrive la temperatura per questa sola prova. A temperatura "
             "alta due run identiche divergono quanto due semi diversi, quindi "
             "va misurata qui la stessa che la run usera'",
    )
    parser.add_argument(
        "--step-ms", type=float, default=95.0,
        help="tempo di passo della simulazione, per calcolare la cadenza sostenibile",
    )
    parser.add_argument(
        "--steps", type=int, default=1500,
        help="lunghezza della run prevista: serve a dire quante direttive la "
             "cadenza consigliata applica davvero",
    )
    args = parser.parse_args()

    if not real_calls_allowed():
        print(
            f"{LLM_CONSENT_ENV} non e' impostata: nessuna chiamata verrebbe fatta e "
            "questo controllo non proverebbe niente.\n"
            f"Eseguire con {LLM_CONSENT_ENV}=1 nella propria shell.",
            file=sys.stderr,
        )
        return 2

    registry = load_provider_registry()
    if args.provider not in registry:
        print(
            f"provider {args.provider!r} assente dal registro. Disponibili: "
            f"{', '.join(sorted(registry)) or 'nessuno'}",
            file=sys.stderr,
        )
        return 2
    provider_config = registry[args.provider]
    missing_env = str(provider_config.get("missing_env") or "")
    if missing_env:
        print(f"variabile d'ambiente {missing_env} non definita in questa shell", file=sys.stderr)
        return 2
    try:
        model = resolve_offered_model(args.provider, provider_config, args.model)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    provider = create_llm_provider(args.provider, model, registry)
    if isinstance(provider, FallbackProvider):
        print("la fabbrica ha restituito il fallback: nessuna chiamata sara' fatta", file=sys.stderr)
        return 2

    # Sovrascritture per la sola diagnosi. `reasoning_effort` e' il parametro che
    # muove di piu' la latenza, quindi va provato piu' volte: farlo modificando
    # `configs/llm_providers.json` a ogni tentativo significa misurare una
    # configurazione e lasciarne in vigore un'altra.
    if args.effort:
        extra = dict(provider.options.get("extra_body") or {})
        if args.effort == "none":
            extra.pop("reasoning_effort", None)
        else:
            extra["reasoning_effort"] = args.effort
        provider.options["extra_body"] = extra
    if args.max_tokens:
        provider.max_tokens = args.max_tokens
    # L'attributo, non il dizionario: `self.temperature` e' letto una volta nel
    # costruttore ed e' quello che finisce nella richiesta. `is not None` perche'
    # zero e' il valore interessante e sarebbe falsy.
    if args.temperature is not None:
        provider.temperature = float(args.temperature)
        provider.options["temperature"] = float(args.temperature)

    bounds = Bounds(0.25, 4.0)
    if args.governor:
        prompt = build_governor_prompt(_SAMPLE_PICTURE, bounds)
    else:
        prompt = 'Rispondi solo con questo JSON: {"ok": true, "modello": "<il tuo nome>"}'

    print(_describe(provider))
    print(f"prompt: {'governatore' if args.governor else 'minimo'} ({len(prompt)} caratteri)")
    print(f"chiamate: {args.repeat}\n", flush=True)

    chiamate_riuscite = 0
    latencies: list[float] = []
    parsed_ok = 0
    directives_ok = 0
    for index in range(args.repeat):
        latency, response = asyncio.run(_one_call(provider, prompt))
        latencies.append(latency)
        text = getattr(response, "text", "") or ""
        error = getattr(response, "error", "") or ""
        attempted = bool(getattr(response, "api_call_attempted", False))
        tokens_in = getattr(response, "tokens_in", 0)
        tokens_out = getattr(response, "tokens_out", 0)
        print(
            f"  chiamata {index + 1}/{args.repeat}: {latency:6.2f} s  "
            f"token in/out={tokens_in}/{tokens_out}  "
            f"{'ERRORE: ' + error if error else 'ok'}"
        )
        if not error and attempted:
            chiamate_riuscite += 1
        if not attempted:
            print("    nessuna chiamata tentata: provider non configurato")
            continue
        try:
            raw = json.loads(text)
            parsed_ok += 1
        except (TypeError, ValueError):
            print(f"    risposta NON e' JSON: {text[:300]!r}")
            continue
        # Un modello che esaurisce il tetto dei token non ha finito di parlare:
        # e' stato TRONCATO, e cio' che arriva e' un frammento. Va detto prima di
        # interpretare il contenuto, perche' spiega sia la latenza (genera il
        # massimo ogni volta) sia il fatto che il JSON finale possa mancare.
        if tokens_out and tokens_out >= provider.max_tokens:
            print(
                f"    TRONCATA: {tokens_out} token di uscita sono il tetto "
                f"({provider.max_tokens}). La risposta e' un frammento, non una "
                "risposta breve; alzare il tetto o cambiare modello."
            )
        if args.governor:
            policy, drops = parse_policy(raw, bounds)
            if policy is None:
                print(f"    JSON valido ma fuori schema; scarti={drops}. Risposta: {text[:300]!r}")
            elif not policy.rules:
                # **Una policy senza regole non e' una policy valida, QUI.**
                # `parse_policy` la restituisce perche' e' ben formata, e per il
                # governatore "zero regole" e' un non-intervento legittimo. Ma
                # qui si sta decidendo se un modello sia utilizzabile come
                # governatore, e un modello che tace sempre non lo e': contarla
                # fra le valide faceva scrivere al preflight "valide: 3/3" per
                # un 2B che consegnava zero contenuto e motivazione vuota.
                print(
                    f"    policy VUOTA: zero regole, motivazione "
                    f"{policy.rationale[:60]!r}. Ben formata ma senza contenuto: "
                    "come governatore questo modello tace."
                )
            else:
                directives_ok += 1
                print(
                    f"    policy: {len(policy.rules)} regole, "
                    f"scarti={drops}, motivazione={policy.rationale[:80]!r}"
                )
        else:
            print(f"    JSON: {json.dumps(raw)[:200]}")

    median = statistics.median(latencies)
    step_s = max(1e-6, args.step_ms / 1000.0)
    needed_cadence = math.ceil(median / step_s)
    print(f"\nlatenza mediana : {median:.2f} s  (min {min(latencies):.2f}, max {max(latencies):.2f})")
    print(f"chiamate riuscite: {chiamate_riuscite}/{args.repeat}")
    print(f"risposte JSON   : {parsed_ok}/{args.repeat}"
          "   (il ripiego e' JSON valido: non conta come riuscita)")
    if args.governor:
        print(f"direttive con celle: {directives_ok}/{args.repeat}")
    print(
        f"\nCon un passo da {args.step_ms:.0f} ms, la cadenza minima perche' la "
        f"chiamata torni prima del tick successivo e' {needed_cadence} passi "
        f"({needed_cadence * 2} con margine x2)."
    )

    # **La cadenza da sola non basta: va confrontata con la lunghezza della run.**
    # Una cadenza piu' lunga della run non applica nessuna direttiva, senza dare
    # un errore e con `governor_misses` a zero. Suggerire una cadenza senza dire
    # quante direttive produce e' come dichiarare un guadagno senza dire su quale
    # config e' preso: il numero c'e', ma non decide niente.
    applied = (args.steps - 1) // max(1, needed_cadence * 2)
    durata_run_s = args.steps * step_s
    print(
        f"Su una run da {args.steps} passi (~{durata_run_s / 60:.1f} min di orologio) "
        f"quella cadenza applica {applied} direttive."
    )
    if applied == 0:
        print(
            f"\nINUTILIZZABILE COSI': una chiamata da {median:.0f} s contro una run "
            f"da ~{durata_run_s:.0f} s. Il consiglio avvierebbe le chiamate e non "
            "raccoglierebbe mai, producendo una run identica alla baseline con un "
            "registro vuoto. Le leve, in ordine di efficacia: un modello piu' "
            "rapido, un `reasoning_effort` piu' basso, una run piu' lunga.",
            file=sys.stderr,
        )
    # **Il comando che riproduce QUESTA misura.** Il preflight valida una
    # configurazione, la run ne usa un'altra, e nessuno se ne accorge finche' non
    # e' finita: e' accaduto con `--effort low` misurato qui e `medium` in run,
    # settantacinque tick mancati su settantacinque. Stampare il comando toglie
    # il passaggio in cui i due si separano.
    if args.governor and directives_ok:
        # **Il limite di attesa va scelto largo, e la ragione e' asimmetrica.**
        # `wait_seconds` e' un TETTO, non un costo fisso: se la chiamata torna in
        # 5 s il tick dura 5 s, qualunque sia il limite. Sbagliarlo per eccesso
        # costa solo su una chiamata davvero appesa; sbagliarlo per difetto
        # trasforma in mancato aggiornamento ogni chiamata sopra la media, e la
        # media qui viene da pochi campioni: fra tre misure il massimo non e' il
        # peggio possibile, e' solo il peggio visto.
        attesa = max(60, math.ceil(max(latencies) * 4))
        print(
            "\nPer eseguire l'esperimento CON GLI STESSI parametri:\n"
            f"  python scripts/run_governor_experiment.py --arms none random scripted llm \\\n"
            f"    --seeds 0 1 2 --steps {args.steps} --agents 300 --cadence 20 \\\n"
            f"    --wait {attesa} --provider {args.provider} --model {args.model}"
            + (f" --effort {args.effort}" if args.effort else "")
            + (f" --temperature {args.temperature}" if args.temperature is not None else "")
            + f"\n(--wait {attesa} e' un TETTO, non un costo: con {median:.1f}s di "
            f"latenza mediana un tick dura {median:.1f}s. Largo apposta, perche' "
            f"{max(latencies):.1f}s e' il peggio VISTO su {args.repeat} campioni, "
            "non il peggio possibile.)"
            f"\n(--cadence 20 e' libera perche' si attende: sono "
            f"{(args.steps - 1) // 20 + 1} direttive, e circa "
            f"{((args.steps - 1) // 20 + 1) * median / 60:.0f} min di attesa "
            "per ciascun seme del braccio llm.)"
        )

    if applied < 5 and applied:
        print(
            f"\nATTENZIONE: {applied} direttive su tutta la run sono poche perche' un "
            "braccio si distingua dalla baseline. Puntare ad almeno 20-30.",
            file=sys.stderr,
        )
    if chiamate_riuscite == 0:
        print(
            f"\nFORNITORE NON UTILIZZABILE: nessuna delle {args.repeat} chiamate "
            "e' riuscita.\n  Ogni risposta qui sopra e' il ripiego deterministico, "
            "non il modello.\n  Una campagna lanciata ora produrrebbe run "
            "indistinguibili dalla baseline.",
            file=sys.stderr,
        )
        return 1
    if args.governor and directives_ok == 0:
        print(
            "\nNESSUNA direttiva valida: con questa configurazione i governatori "
            "sarebbero muti e la run coinciderebbe con la baseline.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
