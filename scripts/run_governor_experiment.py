"""Protocollo storico a quattro bracci, con due bracci SemIf sperimentali.

I quattro bracci storici restano invariati, per la stessa ragione per cui il confronto delle
prestazioni ne aveva tre: `none` e' la baseline, `random` isola l'effetto di
ricevere direttive qualsiasi, `scripted` isola un coordinamento competente ma non
LLM, `llm` isola cio' che aggiunge il modello. Senza `scripted`, un `llm` che batte
la baseline non distingue "l'LLM governa bene" da "coordinare aiuta".

`semif` e `hybrid` sono trattamenti sperimentali aggiuntivi. Per default usano
il provider deterministico `fake`; REST e replay richiedono opzioni esplicite e
i risultati SemIf finiscono in una radice separata.

Il braccio `llm` richiede il consenso esplicito (`MARSABM_ALLOW_LLM_CALLS`): senza,
i suoi provider sono il fallback deterministico e il braccio misura l'impianto,
non un modello.

**Lo scenario non e' `realistic_config` liscia, e la differenza e' misurata.**
Il menu di cella impone alle costruzioni un tetto fisico —
`ceil(occupancy / COLONISTS_PER_STRUCTURE[tipo])`, cioe' una serra ogni 7 coloni —
mentre il bersaglio che la colonia si da' da sola e'
`ceil(occupancy * greenhouse_per_capita)` con `greenhouse_per_capita = 0,25`, cioe'
una serra ogni 4. Il bersaglio di serie e' quindi **irraggiungibile per
costruzione**: la carenza resta positiva per sempre, la cella costruisce serre al
suo ritmo massimo e la baseline tocca il tetto fisico da sola. Misurato su 60
coloni per 30 passi: la baseline sale `1, 2, 2, 3, ..., 9` e si ferma a 9 al passo
16, e da li' in poi tre bracci diversi (nessuno, scritto, spinta massima
moltiplicatore 4,0 + quota 32) chiudono tutti a **9 serre e 69 strutture**. Su uno
scenario cosi' i quattro bracci sono obbligati a dare lo stesso numero, e un
esperimento che lo leggesse riporterebbe "governare non cambia nulla" per una
ragione puramente meccanica.

`run_arm` corregge quindi due numeri, ed e' la stessa correzione detta due volte:

1. `greenhouse_per_capita` da 0,25 a **0,10** — il piano diventa raggiungibile e
   si ferma sotto il tetto fisico;
2. le serre iniziali da `7 ogni 50 coloni` (0,14 pro capite, cioe' **gia' il tetto
   fisico** di 0,1428: su 300 coloni sono 42 serre contro un tetto di 43) a
   `ceil(0,05 x coloni)`, cioe' meta' del piano.

Cosi' la colonia parte sotto il proprio piano, il piano sta sotto il tetto, e
restano due spazi distinti su cui un governatore puo' agire: il RITMO con cui la
colonia raggiunge il proprio piano, e il tratto fra il piano e il tetto fisico che
la colonia da sola non percorrerebbe mai. Misurato su 60 coloni per 30 passi:
`none` chiude a 6 serre — il suo piano, esattamente — e `scripted` con mandato
`life_support` a 9, cioe' il tetto fisico. Sotto quegli spazi non esiste
esperimento; dentro, la differenza fra i bracci e' attribuibile a chi decide.

Le correzioni valgono per tutti i bracci, baseline compresa: sono lo scenario, non
un trattamento.

**Ma quelle correzioni erano state misurate con l'ISRU SPENTA, e con l'ISRU
accesa fanno il danno che dovevano impedire (2026-09-02).** Questo script teneva
`--isru 0.0` mentre il prodotto sta a 1,0: le due correzioni sono quindi state
derivate in un regime e applicate in un altro. Misurato a 300 coloni per 300
passi, seme 3, quattro combinazioni:

    ISRU   piano serre ridotto   vivi   morti   celle   strutture
    0,0    si  (15 serre, 0,10)   489    102      31        874
    0,0    no  (42 serre, 0,25)   482     74      25        855
    1,0    si  (15 serre, 0,10)   300      0       1        466
    1,0    no  (42 serre, 0,25)   560    101      27        985

La terza riga e' una colonia **inerte**: esattamente i fondatori, nessun nato,
nessun morto, una cella sola per mille passi. Non e' una colonia che va male, e'
una colonia in cui non succede niente — il caso in cui tutti i bracci sono
obbligati a dare lo stesso numero, cioe' proprio quello che le correzioni
dovevano evitare. Il difetto non e' in nessuna delle due opzioni presa da sola,
ma nella loro INTERAZIONE, e nessuna delle due lo mostrava da sola.

Il piano ridotto e' percio' passato da comportamento fisso a opzione
(`--piano-serre-ridotto`), **spenta di default**: lo scenario della campagna e'
`realistic_config` liscia con l'ISRU del prodotto, la quarta riga, l'unica in cui
la colonia cresce, perde qualcuno e si espande. Chi rimette il piano ridotto deve
prima rimisurare questa tabella nel proprio regime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parity_harness import realistic_config  # noqa: E402
from src.governors.context import LIVELLI  # noqa: E402
from src.governors.varianti_prompt import SIGLE, variante_da_sigla  # noqa: E402
from src.world.world_generator import MAP_PROFILES  # noqa: E402
from src.core import constants as C  # noqa: E402
from src.simulation.agent_coupled_runner import AgentCoupledRunner  # noqa: E402
from src.semantic_governance.artifacts import SEMANTIC_ARMS  # noqa: E402
from src.world.structures import StructureType  # noqa: E402


class _Tee:
    """Scrive su schermo e su file insieme.

    Il terminale e' l'unico posto dove compaiono l'allarme dei mancati
    aggiornamenti, il costo previsto del regime bloccante e la traiettoria delle
    azioni: quando una run dura un'ora e mezzo, perderlo significa dover
    ricostruire a memoria cosa e' successo. `results.json` dice come e' finita,
    non come e' andata.
    """

    def __init__(self, stream, path: Path) -> None:
        self._stream = stream
        self._handle = path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._handle.write(text)
        self._handle.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._handle.flush()
        self._stream.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __getattr__(self, name):  # isatty, encoding, fileno...
        return getattr(self._stream, name)


def _governor_summary(output_dir) -> dict:
    """Cosa ha fatto il governatore, riassunto dal suo registro.

    Sta qui e non nel confronto perche' `results.json` deve poter essere letto da
    solo: chi lo apre fra sei mesi non deve dedurre da dodici numeri se le
    direttive fossero arrivate.
    """
    if not output_dir:
        return {}
    path = Path(output_dir) / "governor_decisions.jsonl"
    if not path.exists():
        return {"registro": "assente"}
    righe = [json.loads(r) for r in path.read_text(encoding="utf-8").split("\n") if r.strip()]
    if not righe:
        return {"registro": "vuoto", "tick": 0}
    parlanti = [r for r in righe if (r.get("policy") or {}).get("rules")]
    proposte = 0
    latenze: list[float] = []
    token_in = token_out = 0
    for riga in righe:
        g = riga.get("governor") or {}
        if not g:
            continue
        if (g.get("proposal") or {}).get("rules"):
            proposte += 1
        latenze.append(float(g.get("latency_s", 0.0)))
        token_in += int(g.get("tokens_in", 0))
        token_out += int(g.get("tokens_out", 0))
    return {
        "registro": "presente",
        "tick": len(righe),
        "tick_con_policy": len(parlanti),
        "proposte_con_regole": proposte,
        "token_in": token_in,
        "token_out": token_out,
        "latenza_mediana_s": round(statistics.median(latenze), 3) if latenze else 0.0,
        "latenza_max_s": round(max(latenze), 3) if latenze else 0.0,
    }


def _top_rejections(output_dir, quante: int = 5) -> dict[str, int]:
    """I motivi di rifiuto piu' frequenti, azione per azione.

    Dicono cosa la direttiva ha chiesto e il mondo non ha potuto dare. Sul primo
    esperimento vero furono 25.168 `collect_materials` in celle esaurite: senza
    questa voce restavano visibili solo come un tasso di accettazione al 94%,
    cioe' come un numero senza spiegazione. La causa non era pero' cio' che il
    governatore non poteva sapere, ma una quota `-1` che disattivava la
    prenotazione della risorsa: vedi
    `docs/benchmarks/2026-08-20-quota-meno-uno.md`.
    """
    if not output_dir:
        return {}
    path = Path(output_dir) / "action_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    motivi: dict[str, int] = {}
    for azione, dati in (payload.get("by_action") or {}).items():
        for motivo, quanti in (dati.get("rejection_reasons") or {}).items():
            motivi[f"{azione}: {motivo}"] = motivi.get(f"{azione}: {motivo}", 0) + int(quanti)
    return dict(sorted(motivi.items(), key=lambda kv: -kv[1])[:quante])


#: Quanti abitanti fanno di una cella un insediamento invece di un avamposto.
#: Non e' una soglia scelta per far tornare i conti: su trenta run di controllo
#: e' l'unica fra 5, 10, 20 e 50 che separa i due stati della colonia senza
#: sovrapporli — una sola cella insediata da' popolazione 306-337, due o piu' ne
#: danno 373-579, e in mezzo non c'e' niente.
SOGLIA_INSEDIAMENTO = 10


def _expansion(output_dir) -> dict:
    """Se la colonia ha fondato una seconda cella vera, e quanto grande.

    **E' l'esito con un meccanismo dietro, invece della popolazione.** Il piano
    di costruzione fissa `housing target = popolazione della cella`: si costruisce
    per chi c'e', mai per chi potrebbe arrivare. Un satellite da tre persone si
    da' alloggi per tre, le nascite cercano una cella con capienza libera e non
    la trovano mai, e il satellite resta a tre per sempre. Una colonia che rompe
    quello stallo arriva a 373-579 abitanti; una che non lo rompe si ferma fra
    306 e 337, al passo trecento, e non si muove per i millecinquecento
    successivi.

    Un registro assente restituisce un dizionario vuoto e non `espansa: False`:
    "non misurata" e "non si e' espansa" finirebbero nella stessa colonna, e la
    seconda e' un risultato.
    """
    if not output_dir:
        return {}
    path = Path(output_dir) / "cell_infrastructure.json"
    if not path.exists():
        return {}
    celle = json.loads(path.read_text(encoding="utf-8"))
    popolazioni = sorted(
        (int(c.get("population", 0)) for c in celle), reverse=True
    )
    insediate = [p for p in popolazioni if p >= SOGLIA_INSEDIAMENTO]
    return {
        "celle_totali": len(celle),
        "celle_insediate": len(insediate),
        "espansa": len(insediate) >= 2,
        "popolazione_seconda_cella": popolazioni[1] if len(popolazioni) > 1 else 0,
    }


def _margini(output_dir) -> dict:
    """Il MINIMO toccato dai due margini durante la run, non il valore finale.

    Il valore finale non dice niente: `food_margin` e `material_margin` valgono
    `min(2, (scorte + capacita') / (2 x popolazione))`, quindi sono tosati in
    alto a 2 e una colonia che ha sfiorato la carestia a meta' run torna al
    tetto se poi si riprende. E' il minimo a dire se la scorta ha mai smesso di
    essere abbondante, cioe' se una politica di allocazione aveva qualcosa su
    cui essere giusta o sbagliata.

    Misurato su `balanced` (15 semi, tutti i bracci): il minimo di
    `material_margin` sta a 2,00 per `none` e `scripted`, a 1,94-2,00 per
    `random`, e scende a 1,35 di mediana per `llm` — che e' l'unico braccio a
    staccarlo dal tetto. Il margine pero' non arriva mai a mordere, ed e'
    esattamente questa la ragione per cui l'esito non cambia.
    """
    if not output_dir:
        return {}
    path = Path(output_dir) / "state_timeseries.csv"
    if not path.exists():
        return {}
    minimi: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for riga in csv.DictReader(handle):
            for nome in ("food_margin", "material_margin"):
                grezzo = riga.get(nome, "")
                if grezzo in ("", "nan"):
                    continue
                try:
                    valore = float(grezzo)
                except ValueError:
                    continue
                if nome not in minimi or valore < minimi[nome]:
                    minimi[nome] = valore
    return {f"{k}_min": round(v, 6) for k, v in sorted(minimi.items())}


def _isru_di_prodotto() -> float:
    """L'ISRU del prodotto, letta e non ricopiata.

    **Perche' non e' piu' zero (2026-09-02).** Questo script teneva
    `--isru 0.0` come "default storico" mentre `ManualConfigOptions` lo tiene a
    1,0 e `realistic_config` lo legge da li' apposta --- e' scritto nel suo
    docstring: "Due default per una quantita' sola: il secondo si legge dal
    primo". Il secondo era qui, e non lo leggeva.

    Non e' un dettaglio di configurazione. Senza ISRU la colonia non lavora il
    regolito, il materiale da costruzione si esaurisce e **la colonia muore**:
    misurato lanciando la campagna, baseline a 300 coloni per 1000 passi, seme
    3 --- `material_margin` scende 2,0 -> 1,42 -> 0,91 -> 0,57 fra il passo 1 e
    il 301, la popolazione crolla da 504 a 44 fra il 401 e il 501, e al passo
    679 non e' vivo piu' nessuno. Una campagna lanciata cosi' avrebbe
    confrontato quattro governi su una colonia che si estingue in tutti e
    quattro i bracci, e la sola conclusione leggibile sarebbe stata "governare
    non cambia nulla".

    Il default a 400 passi non lo mostrava: l'estinzione arriva dopo, ed e' il
    motivo per cui una prova a vuoto corta puo' dire che l'impianto funziona
    senza dire che lo scenario e' sbagliato.
    """
    from src.experiments.manual_config import ManualConfigOptions

    return float(ManualConfigOptions.isru_material_rate)


_ISRU_DI_PRODOTTO = _isru_di_prodotto()


def _assegnazione(provider: str, model: str, effort: str,
                  temperature: float | None, thinking: str) -> dict:
    """Provider, modello e regolazioni di un braccio linguistico.

    Scritta una volta perche' serve in due posti --- il governo e, quando non
    segue il governo, l'amministrazione --- e due copie divergerebbero in
    silenzio: il caso peggiore e' un decentramento che parla a un modello
    diverso da quello del governo, che misurerebbe il modello e non lo strato.
    """
    spec = {"provider": provider, "model": model}
    if effort:
        # Lo STESSO sforzo con cui il preflight ha misurato la latenza. Senza
        # questo la run parte con quello del registro, e il limite di attesa e'
        # stato scelto su un altro numero.
        spec["effort"] = effort
    if temperature is not None:
        # Serve a rendere informative le RIPETIZIONI dello stesso seme. A
        # temperatura 1,0 due esecuzioni identiche del seme 0 hanno dato
        # manutenzione 1,93x e 6,10x: il confronto appaiato misurava il
        # campionamento del modello, non la politica.
        spec["temperature"] = float(temperature)
    if thinking:
        # L'interruttore unico del ragionamento, tradotto per API in
        # `src/llm/thinking.py`. Va registrato anche quando vale `off`, perche'
        # "spento" e' una scelta e non un'assenza.
        spec["thinking"] = thinking
    return spec


def _uso_api(output_dir) -> dict:
    """Chiamate riuscite e fallite di una run, dal registro che scrive gia'.

    Serve a un guardiano e non a una statistica. Una chiamata fallita non
    solleva: il provider restituisce una risposta vuota, `parse_policy` non
    trova regole, e la run prosegue con una politica che non dice niente ---
    cioe' chiude **identica alla baseline**. E' la modalita' di guasto piu'
    pericolosa di questo esperimento, perche' produce un numero plausibile
    invece di un errore.

    Misurato il 2026-09-03: due bracci interi (3494 e 3450 chiamate) sono
    girati con **zero** chiamate riuscite, e il primo di essi era gia' stato
    letto come risultato --- "l'amministratore senza governo accetta sempre"
    --- quando in realta' nessun amministratore aveva mai parlato.
    """
    if not output_dir:
        return {}
    path = Path(output_dir) / "api_usage.json"
    if not path.exists():
        return {}
    dati = json.loads(path.read_text(encoding="utf-8"))
    return {
        "calls": int(dati.get("calls", 0) or 0),
        "successful_calls": int(dati.get("successful_calls", 0) or 0),
        "failed_calls": int(dati.get("failed_calls", 0) or 0),
        "estimated_cost_usd": float(dati.get("estimated_cost_usd", 0.0) or 0.0),
    }


def _provider_e_modello(voce: str) -> tuple[str, str]:
    """`gpu_farm:gpt-oss:20b` -> (`gpu_farm`, `gpt-oss:20b`): il primo `:` separa.

    Il nome del modello puo' contenere altri due punti (`gpt-oss:20b`), quindi
    si taglia al primo e non all'ultimo.
    """
    provider, sep, modello = voce.partition(":")
    if not sep or not provider or not modello:
        raise SystemExit(f"--admin-models: '{voce}' non e' PROVIDER:MODELLO")
    return provider.strip(), modello.strip()


def _etichetta_senza_agenti(arm: str, context_level: str, administrators: bool, admin_arm: str = "", admin_misto: bool = False, variante_prompt: str = "0", livello_amministratori: str = "", semantic_provider: str = "") -> str:
    """Il nome con cui un braccio si legge nella tabella e nel nome di cartella.

    Un braccio non e' identificato dal solo `arm`: `llm` con contesto completo e
    `llm` cieco sono due trattamenti diversi, e il secondo esiste apposta per
    essere confrontato col primo. Lo stesso vale per lo strato amministrativo.
    """
    nome = f"{arm}:{context_level}" if arm in {"llm", "hybrid"} else arm
    if arm in SEMANTIC_ARMS and semantic_provider:
        nome = f"{nome}:{semantic_provider}"
    # La variante del prompt entra nel nome solo quando non e' quella storica:
    # cosi' le etichette gia' scritte nell'archivio restano quelle, e due
    # varianti non possono finire indistinguibili nello stesso `results.jsonl`.
    if arm in {"llm", "hybrid"} and str(variante_prompt).upper() not in ("", "0"):
        nome = f"{nome}-p{str(variante_prompt).upper()}"
    # Amministratori a un gradino proprio: e' un trattamento diverso e il nome
    # lo dice, altrimenti due run con lo stesso braccio ma prompt locali diversi
    # finirebbero indistinguibili nello stesso registro.
    if administrators and livello_amministratori and livello_amministratori != "completo":
        nome = f"{nome}+amm:{livello_amministratori}"
        return nome
    if not administrators:
        return nome
    # Il braccio dell'amministrazione entra nel nome solo quando NON e' quello
    # del governo: altrimenti ripeterebbe un'informazione gia' contenuta nel
    # nome del braccio, e la casella del 2x2 che conta --- `none+amm:llm`, il
    # decentramento puro --- e' proprio quella in cui i due differiscono.
    if admin_misto:
        # Piu' modelli fra i distretti: e' un trattamento a se', e il nome lo dice.
        return f"{nome}+amm:misto"
    return f"{nome}+amm:{admin_arm}" if admin_arm else f"{nome}+amm"


def _etichetta(arm: str, context_level: str, administrators: bool, admin_arm: str = "", admin_misto: bool = False, variante_prompt: str = "0", livello_amministratori: str = "", semantic_provider: str = "", semantic_agents: str = "") -> str:
    """L'etichetta del braccio, piu' lo strato SemIf degli agenti se acceso.

    `semantic_agents` e' `"<modo>:<provider>"` oppure vuoto: vuoto lascia
    l'etichetta identica a quella storica, cosi' le campagne gia' scritte si
    riprendono con lo stesso nome.
    """
    nome = _etichetta_senza_agenti(
        arm, context_level, administrators, admin_arm,
        admin_misto=admin_misto, variante_prompt=variante_prompt,
        livello_amministratori=livello_amministratori,
        semantic_provider=semantic_provider,
    )
    return f"{nome}+ag:{semantic_agents}" if semantic_agents else nome


def _nome_cartella(etichetta: str) -> str:
    """L'etichetta resa innocua come nome di cartella.

    L'etichetta e' fatta per leggersi in tabella (`llm:completo-pA+amm`) e porta
    caratteri che in un percorso significano altro: i due punti sono illegali su
    Windows e la barra creerebbe un albero annidato invece di una cartella per
    run, rompendo ogni script che cerca le run con un glob a un livello.
    """
    ripulita = etichetta
    for carattere in (":", "/", "\\"):
        ripulita = ripulita.replace(carattere, "_")
    return ripulita


def _semantic_section(
    provider: str,
    *,
    run_id: str,
    endpoint: str = "",
    api_key_env: str = "",
    replay_from: str = "",
    timeout_seconds: float = 10.0,
    min_confidence: float = 0.65,
    governor_decision: str = "keep_previous",
    admin_decision: str = "accept_governor",
    candidate_profile: str = "v1",
    agent_decision: str | None = None,
    budget_usd: float = 0.0,
    typesafe_model: str = "jev-latest",
    question_type: str = "choice",
) -> dict:
    """Costruisce la sola configurazione SemIf, senza leggere segreti."""
    mode = str(provider or "").strip().lower()
    section = {
        "provider": mode,
        "run_id": str(run_id),
        "timeout_seconds": float(timeout_seconds),
        "min_confidence": float(min_confidence),
    }
    profilo = str(candidate_profile or "v1").strip().lower()
    if profilo in ("v2h", "v3", "v4"):
        # v2h decide con la regola del braccio sulla distribuzione intera.
        section["min_confidence"] = 0.0
    if profilo != "v1":
        # Scritto solo quando diverso dal default: le sezioni delle run gia'
        # prodotte (e quindi il loro hash di trattamento) restano identiche.
        section["candidate_profile"] = profilo
    if mode == "fake":
        section["decisions"] = {
            "jev-semif-governor-v1": str(governor_decision),
            "jev-semif-admin-v1": str(admin_decision),
        }
        if profilo == "v2":
            section["decisions"]["jev-semif-governor-v2"] = str(governor_decision)
            section["decisions"]["jev-semif-admin-v2"] = str(admin_decision)
        if agent_decision is not None:
            # Solo per la sezione dello strato agenti: la sezione di governo
            # resta identica a quella storica, e cosi' il suo hash.
            section["decisions"]["jev-semif-agent-v1:L1"] = str(agent_decision)
    elif mode == "replay":
        if not replay_from:
            raise ValueError("semantic replay requires --semantic-replay-from")
        section["replay_from"] = str(replay_from)
    elif mode == "rest":
        if not endpoint:
            raise ValueError("semantic REST requires --jev-endpoint")
        section["endpoint"] = str(endpoint)
        if api_key_env:
            section["api_key_env"] = str(api_key_env)
    elif mode == "laya":
        if not endpoint:
            raise ValueError("semantic laya requires --jev-endpoint (the laya.serve URL)")
        section["endpoint"] = str(endpoint)
        if api_key_env:
            section["api_key_env"] = str(api_key_env)
    elif mode == "typesafe":
        if not float(budget_usd) > 0.0:
            raise ValueError("semantic typesafe requires --typesafe-budget-usd > 0")
        section["budget_usd"] = float(budget_usd)
        section["api_key_env"] = str(api_key_env or "TYPESAFE_AI")
        section["model"] = str(typesafe_model)
        section["question_type"] = str(question_type)
        if endpoint:
            section["endpoint"] = str(endpoint)
    else:
        raise ValueError("semantic provider must be fake, replay, rest, typesafe or laya")
    return section


def _agents_section(
    mode: str,
    semantic: dict,
    *,
    cadence_steps: int = 4,
    levels: int = 1,
    workers: int = 64,
    hard_margin: float = 0.15,
) -> dict | None:
    """La sezione `semantic_agents` di una run, o `None` se lo strato e' spento."""
    modo = str(mode or "off").strip().lower()
    if modo == "off":
        return None
    return {
        "mode": modo,
        "cadence_steps": int(cadence_steps),
        "levels": int(levels),
        "workers": int(workers),
        "hard_margin": float(hard_margin),
        "semantic": dict(semantic),
    }


def _semantic_treatment_hash(
    arm: str, admin_arm: str, section: dict | None, agents_section: dict | None = None
) -> str:
    if arm not in SEMANTIC_ARMS and admin_arm not in SEMANTIC_ARMS and agents_section is None:
        return ""
    semantic = dict(section or {})
    semantic.pop("run_id", None)
    payload = {"arm": arm, "admin_arm": admin_arm, "semantic": semantic}
    if agents_section is not None:
        # `run_id` identifica l'esecuzione e `workers` solo la concorrenza delle
        # richieste (l'ordine delle risposte e' preservato): nessuno dei due e'
        # parte del trattamento. Senza strato il payload resta quello storico.
        agenti = {k: v for k, v in agents_section.items() if k != "workers"}
        interno = dict(agenti.get("semantic") or {})
        interno.pop("run_id", None)
        agenti["semantic"] = interno
        payload["agents"] = agenti
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resume_key(braccio: str, seed: int, semantic_treatment_hash: str = "") -> tuple:
    """Identità di resume; per gli arm storici resta braccio più seme."""
    return str(braccio), int(seed), str(semantic_treatment_hash or "")


def _default_output_root(arms, admin_arm: str = "", semantic_agents: str = "off") -> Path:
    return Path(
        "runs/jev_semif_experiments"
        if any(str(arm).lower() in SEMANTIC_ARMS for arm in arms)
        or str(admin_arm).lower() in SEMANTIC_ARMS
        or str(semantic_agents or "off").lower() != "off"
        else "runs/governor_experiment"
    )


def run_arm(
    arm: str,
    seed: int,
    steps: int,
    agents: int,
    output_dir,
    cadence_steps: int = 20,
    provider: str = "fallback",
    model: str = "",
    log_interval: int = 10,
    wait_seconds: float = 0.0,
    effort: str = "",
    temperature: float | None = None,
    log_agents: bool = True,
    map_profile: str = "balanced",
    dotazione: float = 1.0,
    isru: float | None = None,
    sviluppo: float = 0.30,
    context_level: str = "completo",
    variante_prompt: str = "0",
    thinking: str = "",
    administrators: bool = False,
    cells_per_district: int = 3,
    admin_arm: str = "",
    livello_amministratori: str = "",
    piano_serre_ridotto: bool = False,
    snapshot_interval: int = 0,
    admin_models: list[str] | None = None,
    correzioni_motore: bool = False,
    pavimento_cantieri: float | None = None,
    strato_ambientale: bool = False,
    redistribuzione: bool = False,
    copertura_elettrica_vera: bool = False,
    semantic_provider: str = "fake",
    jev_endpoint: str = "",
    jev_api_key_env: str = "",
    semantic_replay_from: str = "",
    semantic_run_id: str = "",
    semantic_timeout: float = 10.0,
    semantic_min_confidence: float = 0.65,
    semantic_governor_decision: str = "keep_previous",
    semantic_admin_decision: str = "accept_governor",
    semantic_candidate_profile: str = "v1",
    semantic_agents: str = "off",
    semantic_agents_cadence: int = 4,
    semantic_agents_levels: int = 1,
    semantic_agents_workers: int = 64,
    semantic_agents_hard_margin: float = 0.15,
    semantic_agent_decision: str = "unknown",
    typesafe_budget_usd: float = 0.0,
    typesafe_model: str = "jev-latest",
    typesafe_question_type: str = "choice",
) -> dict:
    """Esegue un braccio e restituisce le metriche di confronto."""
    # `None` significa «il default del prodotto», mai «spenta»: vedi
    # `_isru_di_prodotto`. Uno zero scritto qui ha gia' fatto morire una
    # campagna intera.
    isru = _ISRU_DI_PRODOTTO if isru is None else float(isru)
    config = realistic_config(
        agents, steps, seed, map_profile=map_profile, dotazione=dotazione, isru=isru,
        sviluppo=sviluppo,
    )
    # Le due correzioni di scenario, identiche in TUTTI i bracci — baseline
    # compresa — cosi' che restino proprieta' dello scenario e non un trattamento
    # riservato a chi e' governato. Entrambe dicono la stessa cosa: i numeri delle
    # serre devono stare sotto il tetto che la maschera impone (una ogni 7
    # coloni), altrimenti nessun braccio puo' distinguersi. Vedi il docstring del
    # modulo per le traiettorie misurate.
    # Ogni quanti passi la shell stampa la riga di avanzamento. NON e' il passo
    # della simulazione ne' la frequenza dei dati: `state_timeseries.csv`
    # contiene una riga per passo (1, 2, 3... e giorno 7, 14, 21) qualunque sia
    # questo valore. Governa solo cio' che si vede scorrere a schermo, dove un
    # `step=10 day=70` si legge facilmente come "la simulazione avanza di dieci".
    config.setdefault("headless", {})["log_interval_steps"] = max(1, int(log_interval))
    if snapshot_interval > 0:
        # **Snapshot a ogni tornata, ma compatti.** Servono per rivedere la run
        # nell'analysis frontend con i distretti colorati; uno snapshot
        # completo su 360x180 pesa 73 MB, e quaranta per run sono tre
        # gigabyte. Il compatto tiene le sole celle occupate, con strutture o
        # esplorate (vedi `shell_common.compatta_snapshot`).
        config["headless"]["snapshot_interval"] = int(snapshot_interval)
        config["headless"]["snapshot_compact"] = True
    if log_agents:
        # **Senza questo la run non lascia nessuna traccia per azione.**
        # `build_manual_config` fissa `store_memory_logs: False` — vale per GUI,
        # wizard e CLI — e con quello spento `validated_actions.jsonl`,
        # `rejected_actions.jsonl`, `agent_decisions.jsonl`,
        # `agent_conversations.jsonl` e `agent_thoughts.jsonl` restano file da
        # zero byte: il frontend di analisi apre la run e trova solo le
        # aggregate. Per una campagna che spende ore di modello, il dettaglio di
        # cosa gli agenti hanno fatto sotto quelle direttive e' il dato che non
        # si puo' ricostruire dopo.
        config["headless"]["store_memory_logs"] = True
        # Il tetto va alzato con la run, non lasciato al default: a 150.000
        # righe una run da 300 agenti per 800 passi (240.000 azioni) perderebbe
        # in silenzio l'ultimo terzo — ed e' proprio la coda, dove la colonia si
        # e' assestata, quella che serve.
        config["headless"]["max_action_log_rows"] = int(agents * steps * 1.1) + 1000
    # **I due interruttori che le campagne tengono spenti.** Vanno qui e non
    # in un file di scenario perche' sono variabili sperimentali: una campagna
    # che li accende va confrontata con una identica che li tiene spenti, e il
    # confronto ha senso solo se e' il comando a dirlo.
    #
    # `strato_ambientale` costruisce il `PlanetaryCoupler`, e con lui il
    # provider del clima: e' l'UNICO percorso che legge il Mars Climate
    # Database. Spento, il mondo gira su costanti marziane ferme
    # (`sync_static_mars_environment`), che e' cio' che tutte le campagne di
    # questa tesi hanno usato.
    #
    # `redistribuzione` accende le tre fasi di `src/core/redistribution.py`:
    # deposito del surplus nel magazzino di cella, prelievo di chi e' sotto
    # soglia, e flusso fra celle vicine. Spenta, la scarsita' resta locale.
    if strato_ambientale:
        config["environmental_layer"] = {"enabled": True, "role": "background"}
        # Vedi `_esci`: da qui in poi il processo non sapra' piu' terminare da solo.
        global _MCD_TOCCATO
        _MCD_TOCCATO = True
    config.setdefault("redistribution", {})["enabled"] = bool(redistribuzione)
    # **I due interruttori delle correzioni al motore (2026-09-14).** Spenti,
    # il simulatore e' quello con cui sono state misurate le 191 esecuzioni in
    # archivio. Accesi, correggono due difetti trovati dalla revisione:
    # `correzioni_motore` toglie dalle capienze l'ossigeno fantasma degli
    # habitat e le pesa per l'integrita' --- metriche che il governatore LEGGE,
    # quindi non e' cosmesi --- e `pavimento_cantieri` rende misurabile il
    # valore che mette un cantiere aperto sopra ogni altra azione di lavoro.
    # Vanno confrontati appaiati contro gli stessi semi a interruttori spenti.
    config.setdefault("engine_fixes", {})["capacity"] = bool(correzioni_motore)
    if pavimento_cantieri is not None:
        config.setdefault("agents", {})["active_site_priority_floor"] = float(
            pavimento_cantieri
        )
    if piano_serre_ridotto:
        config["redistribution"]["greenhouse_per_capita"] = 0.10
        # La correzione sulle serre scala con la dotazione. Lasciarla fissa al 5%
        # degli agenti significherebbe reintrodurre dalla finestra proprio la
        # capacita' alimentare che la dotazione ridotta toglie dalla porta, e il
        # margine alimentare resterebbe la costante che e' sempre stato.
        config["colony"]["initial_structures"]["greenhouse"] = max(
            0 if dotazione < 1.0 else 1, round(math.ceil(agents * 0.05) * dotazione)
        )
    # **Anche `none` costruisce il blocco, quando ci sono amministratori.** La
    # domanda del disegno e' un 2x2 --- con e senza governo, con e senza
    # amministrazione --- e la casella "amministratori senza governo" e' il
    # decentramento puro: senza di lei non si puo' dire se quel che si misura
    # sia lo strato o la coppia. Il ciclo di simulazione la prevede gia'
    # (`_avanza_amministratori` ha un ramo per quando il governo manca); era
    # questo script a non saperla esprimere.
    uses_semantic = arm in SEMANTIC_ARMS or (
        administrators and admin_arm in SEMANTIC_ARMS
    )
    semantic_section = None
    default_run_id = (
        Path(output_dir).name if output_dir else f"{arm}-seed-{int(seed)}"
    )
    typesafe_kwargs = {
        "budget_usd": typesafe_budget_usd,
        "typesafe_model": typesafe_model,
        "question_type": typesafe_question_type,
    }
    if uses_semantic:
        semantic_section = _semantic_section(
            semantic_provider,
            run_id=semantic_run_id or default_run_id,
            endpoint=jev_endpoint,
            api_key_env=jev_api_key_env,
            replay_from=semantic_replay_from,
            timeout_seconds=semantic_timeout,
            min_confidence=semantic_min_confidence,
            governor_decision=semantic_governor_decision,
            admin_decision=semantic_admin_decision,
            candidate_profile=semantic_candidate_profile,
            **typesafe_kwargs,
        )
    agents_section = _agents_section(
        semantic_agents,
        _semantic_section(
            semantic_provider,
            run_id=semantic_run_id or default_run_id,
            endpoint=jev_endpoint,
            api_key_env=jev_api_key_env,
            replay_from=semantic_replay_from,
            timeout_seconds=semantic_timeout,
            # Lo strato agenti non taglia per confidenza: vedi agent_layer.
            min_confidence=0.0,
            agent_decision=semantic_agent_decision,
            **typesafe_kwargs,
        ) if str(semantic_agents or "off").lower() != "off" else {},
        cadence_steps=semantic_agents_cadence,
        levels=semantic_agents_levels,
        workers=semantic_agents_workers,
        hard_margin=semantic_agents_hard_margin,
    )
    if agents_section is not None:
        config["semantic_agents"] = agents_section
    if arm != "none" or administrators:
        config["governors"] = {
            "arm": arm,
            "cadence_steps": cadence_steps,
            # Il regime va applicato a TUTTI i bracci, non solo a `llm`. I bracci
            # di controllo rispondono in microsecondi, quindi attendere non costa
            # nulla, ma cambia il passo in cui la policy entra in vigore: darlo
            # solo al braccio LLM confronterebbe pianificazioni diverse e
            # attribuirebbe al modello una differenza di calendario.
            "wait_seconds": wait_seconds,
            # Quanto del dominio il governatore vede. E' un braccio a se' e non
            # una variante: `senza_aiuti` toglie la costituzione di riferimento
            # e le avvertenze, `nomi_veri` toglie il significato dei nomi,
            # `cieco` toglie entrambi. Sui bracci non linguistici non cambia
            # nulla, perche' nessuno di loro legge un prompt.
            "context_level": context_level,
            # Quale casella del disegno incrociato sui prompt: la parola usata
            # per cio' che la politica pesa, e se il governatore sa di avere
            # amministratori sotto di se'. Vedi `src/governors/varianti_prompt.py`.
            # `0` e' il prompt storico, parola per parola.
            "prompt_variant": variante_prompt,
        }
        if arm in SEMANTIC_ARMS:
            config["governors"]["semantic"] = dict(semantic_section or {})
        if administrators:
            # Di regola lo strato segue il braccio del governo: un governatore
            # LLM ha amministratori LLM, uno scritto li ha scritti. Altrimenti
            # il confronto misurerebbe la coppia e non il decentramento.
            # `admin_arm` rompe la regola solo dove serve: senza governo non
            # c'e' un braccio da seguire, e seguirlo darebbe amministratori
            # assenti, cioe' la casella vuota invece del decentramento puro.
            sezione = {
                "enabled": True,
                # Quanto del dominio vede il livello locale. Vuoto = come
                # sempre, il prompt completo: e' il comportamento di ogni
                # campagna gia' in archivio.
                "context_level": livello_amministratori or "completo",
                "cells_per_district": int(cells_per_district),
                "follow_governor_arm": not admin_arm,
            }
            if admin_arm:
                sezione["arm"] = admin_arm
                if admin_arm in {"llm", "hybrid"} and provider != "fallback":
                    sezione["assignment"] = _assegnazione(
                        provider, model, effort, temperature, thinking
                    )
                if admin_arm in SEMANTIC_ARMS:
                    sezione["semantic"] = dict(semantic_section or {})
            if admin_models:
                # **Un modello per distretto, pescato da questa lista.** E' la
                # leva per confrontare stili di governo fra modelli: ogni
                # distretto nuovo ne riceve uno, in modo riproducibile a parita'
                # di seme. Il registro per tornata porta provider e modello di
                # ogni decisione, quindi l'analisi puo' separarli.
                sezione["arm"] = "llm"
                sezione["follow_governor_arm"] = False
                sezione["assignments"] = [
                    _assegnazione(p, m, effort, temperature, thinking)
                    for p, m in (_provider_e_modello(voce) for voce in admin_models)
                ]
            config["governors"]["administrators"] = sezione
        if arm in {"llm", "hybrid"} and provider != "fallback":
            # Senza assegnazione esplicita `build_governor` costruisce un
            # provider `fallback`: il braccio `llm` girerebbe allora sul
            # fallback deterministico ANCHE con il consenso attivo, e
            # riporterebbe come "risultato del modello" una run in cui nessun
            # modello ha parlato.
            config["governors"]["assignments"] = [
                _assegnazione(provider, model, effort, temperature, thinking)
            ]
    if copertura_elettrica_vera:
        # **Esperimento di controllo (2026-09-26).** `power_coverage` legge la
        # giacenza di energia avanzata dopo il consumo (~0 a corrente coperta)
        # e il 23% delle celle-passo delle leggi LLM della tesi era preso da
        # regole su di esso. Acceso, l'indicatore legge la copertura vera del
        # passo precedente. Solo i bracci che leggono indicatori ne risentono.
        config.setdefault("governors", {})["copertura_elettrica"] = "vera"
    semantic_treatment_hash = _semantic_treatment_hash(
        arm, admin_arm, semantic_section, agents_section
    )
    runner = AgentCoupledRunner(config)
    # Il governatore va preso PRIMA della run: `run_async` lo chiude in un
    # `finally` e rimette `runner._governor` a `None`, quindi leggerlo dopo
    # darebbe sempre zero mancati aggiornamenti. `close()` non azzera `misses`,
    # quindi il contatore resta leggibile dall'oggetto conservato qui.
    governor = runner._governor
    iniziata = time.perf_counter()
    runner.run(days=steps, output_dir=Path(output_dir) if output_dir else None)
    durata = time.perf_counter() - iniziata

    counts = runner.core.cells.struct_count.sum(axis=(0, 1))
    metriche = runner.world.metrics(include_planetary=False)
    tentate = max(1, runner.actions_attempted_count)
    amministrazione = getattr(runner, "_administration", None)
    uso = _uso_api(output_dir)
    riassunto = _governor_summary(output_dir)
    return {
        "arm": arm,
        # L'etichetta con cui la riga si legge da sola. I gradini di contesto
        # sono bracci separati, non varianti di uno stesso braccio: senza
        # questo campo due righe `llm` con prompt diversi sarebbero
        # indistinguibili in `results.json`.
        "braccio": _etichetta(
            arm, context_level, administrators, admin_arm,
            variante_prompt=variante_prompt,
            livello_amministratori=livello_amministratori,
            semantic_provider=semantic_provider if uses_semantic else "",
            semantic_agents=(
                f"{agents_section['mode']}:{semantic_provider}" if agents_section else ""
            ),
        ),
        "context_level": context_level if arm in {"llm", "hybrid"} else "",
        "administrators": bool(administrators),
        "seed": seed,
        "steps": steps,
        "agents": agents,
        # Come e' stata prodotta: `results.json` deve poter essere letto da solo.
        "config": {
            "cadence_steps": cadence_steps,
            "wait_seconds": wait_seconds,
            "provider": provider if arm in {"llm", "hybrid"} else "",
            "model": model if arm in {"llm", "hybrid"} else "",
            "effort": effort if arm in {"llm", "hybrid"} else "",
            "temperature": temperature if arm in {"llm", "hybrid"} else None,
            "log_agents": bool(log_agents),
            # Senza questo due campagne su scenari diversi producono
            # `results.json` indistinguibili, e il confronto fra scenari — che e'
            # l'unica ragione per cui esistono i profili — diventa impossibile a
            # posteriori.
            "map_profile": map_profile,
            "dotazione": dotazione,
            "isru_material_rate": isru,
            "development_build_priority": sviluppo,
            "thinking": thinking if arm in {"llm", "hybrid"} else "",
            "context_level": context_level if arm in {"llm", "hybrid"} else "",
            "prompt_variant": variante_prompt if arm in {"llm", "hybrid"} else "",
            "administrators": bool(administrators),
            "cells_per_district": int(cells_per_district) if administrators else 0,
            "admin_context_level": livello_amministratori if administrators else "",
            # Senza questo, due campagne su scenari diversi darebbero
            # `results.json` indistinguibili proprio sulla variabile che decide
            # se la colonia e' viva.
            "admin_arm": admin_arm,
            "admin_models": list(admin_models or []),
            "piano_serre_ridotto": bool(piano_serre_ridotto),
        "strato_ambientale": bool(strato_ambientale),
        "redistribuzione": bool(redistribuzione),
        "copertura_elettrica_vera": bool(copertura_elettrica_vera),
        "correzioni_motore": bool(correzioni_motore),
        "pavimento_cantieri": (None if pavimento_cantieri is None
                               else float(pavimento_cantieri)),
            "snapshot_interval": int(snapshot_interval),
            "semantic_provider": semantic_provider if uses_semantic else "",
            "semantic_treatment_hash": semantic_treatment_hash,
            "semantic_agents": (
                {k: v for k, v in agents_section.items() if k != "semantic"}
                | {"provider": semantic_provider}
                if agents_section else None
            ),
        },
        "semantic_treatment_hash": semantic_treatment_hash,
        "population": int(runner.core.agents.alive.sum()),
        "deaths": len(runner.dead_agents),
        "greenhouses": int(counts[C.S[StructureType.GREENHOUSE]]),
        "solar_arrays": int(counts[C.S[StructureType.SOLAR_ARRAY]]),
        "habitats": int(counts[C.S[StructureType.HABITAT]]),
        "structures_total": int(counts.sum()),
        "structure_integrity_mean": round(
            float(metriche.get("structure_integrity_mean", 0.0)), 6
        ),
        "actions_attempted": int(runner.actions_attempted_count),
        "actions_accepted": int(runner.actions_accepted_count),
        "acceptance_rate": round(runner.actions_accepted_count / tentate, 6),
        "wall_clock_s": round(durata, 1),
        "governor_misses": governor.misses if governor is not None else 0,
        # Quante (cella, passo) ogni regola ha catturato, else compreso. E' il
        # numero che dice quale politica ha governato DAVVERO — una regola in
        # ombra dietro una piu' larga qui vale zero, e si vede.
        "policy_hits": dict(sorted(
            (runner.core.governor_policy_hits or {}).items(),
            key=lambda item: item[1], reverse=True,
        )) if runner.core.governor_policy_hits is not None else {},
        # Astensioni e interventi sono la misura del decentramento: uno strato
        # che non tocca mai nulla e' indistinguibile dal governo centralizzato,
        # e uno che tocca sempre non discrimina. Senza questi due numeri la
        # domanda non e' rispondibile a posteriori.
        "amministrazione": amministrazione.riassunto() if amministrazione else {},
        # **Il segnale che distingue "non ha governato" da "non ha parlato".**
        # Una chiamata fallita produce una policy vuota, e una policy vuota e'
        # indistinguibile da un governo che sceglie di non intervenire: la run
        # chiude come la baseline e nessuno se ne accorge. Vedi `_uso_api`.
        "api": uso,
        "llm_muto": bool(uso.get("calls") and not uso.get("successful_calls")),
        # L'altro modo di non governare: le chiamate partono ma non tornano in
        # tempo, vengono annullate, e non compaiono nemmeno fra le fallite.
        # Zero tick con una policy significa che nessuna direttiva e' mai
        # entrata in vigore, qualunque ne sia la ragione.
        "governo_mai_in_vigore": bool(
            arm == "llm" and riassunto.get("tick") and not riassunto.get("tick_con_policy")
        ),
        "governors": riassunto,
        "top_rejections": _top_rejections(output_dir),
        "espansione": _expansion(output_dir),
        "margini": _margini(output_dir),
    }


def _add_semantic_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("JEV/SemIf sperimentale")
    group.add_argument(
        "--semantic-provider", choices=("fake", "replay", "rest", "typesafe", "laya"), default="fake",
        help="provider SemIf. Il default fake è offline; rest richiede --jev-endpoint",
    )
    group.add_argument("--jev-endpoint", default="", help="endpoint REST JEV esplicito")
    group.add_argument(
        "--jev-api-key-env", default="",
        help="nome della variabile d'ambiente che contiene la chiave; mai la chiave",
    )
    group.add_argument("--semantic-replay-from", default="", help="semantic_decisions.jsonl da riprodurre")
    group.add_argument("--semantic-run-id", default="", help="identità stabile richiesta dal replay esatto")
    group.add_argument("--semantic-timeout", type=float, default=10.0)
    group.add_argument("--semantic-min-confidence", type=float, default=0.65)
    group.add_argument("--semantic-governor-decision", default="keep_previous")
    group.add_argument("--semantic-admin-decision", default="accept_governor")
    group.add_argument(
        "--semantic-agents", choices=("off", "all", "hard"), default="off",
        help="strato SemIf dei singoli agenti: off (default), all, hard "
             "(solo agenti con margine fra i due pilastri migliori sotto soglia)",
    )
    group.add_argument("--semantic-agents-cadence", type=int, default=4)
    group.add_argument("--semantic-agents-levels", type=int, choices=(1, 2), default=1)
    group.add_argument("--semantic-agents-workers", type=int, default=64)
    group.add_argument("--semantic-agents-hard-margin", type=float, default=0.15)
    group.add_argument(
        "--semantic-agent-decision", default="unknown",
        help="solo provider fake: pilastro scelto per tutti gli agenti "
             "(unknown = fattori neutri)",
    )
    group.add_argument(
        "--typesafe-budget-usd", type=float, default=0.0,
        help="tetto di spesa cumulativo del registro TypeSafe; 0 = nessuna chiamata",
    )
    group.add_argument("--typesafe-model", default="jev-latest")
    group.add_argument("--typesafe-question-type", default="choice")
    group.add_argument(
        "--semantic-candidate-profile", choices=("v1", "v2", "v2h", "v3", "v4"), default="v1",
        help="profilo candidate di governatore e amministratori; v2 evita le "
             "candidate inerti quando tutte le celle hanno lo stesso valore",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["none", "random", "scripted"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--cadence", type=int, default=20)
    parser.add_argument(
        "--wait", type=float, default=0.0, metavar="SECONDI",
        help="regime BLOCCANTE: al confine di tick la simulazione attende il "
             "consiglio, fino a questo limite. Con 0 (default) non attende mai e "
             "la cadenza deve superare la latenza del provider. Vale per tutti i "
             "bracci, perche' cambia il passo di applicazione",
    )
    parser.add_argument(
        "--effort", default="", choices=("", "none", "low", "medium", "high"),
        help="sforzo di ragionamento dei governatori LLM. Deve essere LO STESSO "
             "passato a check_provider.py: il limite di attesa si sceglie sulla "
             "latenza misurata li', e uno sforzo diverso la cambia",
    )
    parser.add_argument(
        "--provider", default="fallback",
        help="provider dei governatori nel braccio llm, es. gpu_farm. Prima di "
             "usarlo, misurare la latenza con scripts/check_provider.py: se supera "
             "cadence x tempo di passo, nessuna direttiva entra mai in vigore",
    )
    parser.add_argument("--model", default="", help="modello dei governatori, es. qwen3.6:27b")
    parser.add_argument(
        "--temperature", type=float, default=None, metavar="T",
        help="temperatura dei governatori LLM; senza, resta quella del registro "
             "(1,0 per gpt-oss:20b). Serve per le RIPETIZIONI dello stesso seme: "
             "a temperatura alta due run identiche danno rapporti di azione che "
             "differiscono quanto due semi diversi, e il confronto appaiato "
             "misura il campionamento invece della politica",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="ogni quanti passi stampare la riga di avanzamento; 1 per vederli "
             "tutti. I dati in state_timeseries.csv restano comunque per passo",
    )
    parser.add_argument(
        "--no-log-agents", dest="log_agents", action="store_false",
        help="non salvare i log per azione e per agente (validated_actions, "
             "rejected_actions, agent_decisions, thoughts, conversations). Di "
             "default vengono salvati: senza, il frontend di analisi trova solo "
             "le aggregate e il dettaglio non e' ricostruibile a posteriori",
    )
    parser.set_defaults(log_agents=True)
    parser.add_argument(
        "--sviluppo",
        type=float,
        default=0.30,
        help="priorita' del livello di sviluppo a fabbisogno pieno; 0,07 = comportamento storico (mai in menu)",
    )
    parser.add_argument(
        "--isru",
        type=float,
        default=_ISRU_DI_PRODOTTO,
        help=f"conversione regolito -> materiale da costruzione; il default e' "
             f"quello del prodotto ({_ISRU_DI_PRODOTTO:g}), 0 la spegne",
    )
    parser.add_argument(
        "--dotazione",
        type=float,
        default=1.0,
        help="moltiplicatore della dotazione iniziale di strutture; 1,0 = invariata",
    )
    parser.add_argument(
        "--map-profile",
        default="balanced",
        choices=list(MAP_PROFILES),
        help="profilo di mappa; cambia lo scenario per TUTTI i bracci insieme",
    )
    parser.add_argument(
        "--context-levels", nargs="+", default=[], choices=list(LIVELLI),
        help="i gradini di contesto del braccio llm, ciascuno un BRACCIO A SE': "
             "completo (ruolo, glossario, costituzione di riferimento, "
             "avvertenze), senza_aiuti (via costituzione e avvertenze), "
             "nomi_veri (via anche ruolo e glossario: restano i soli nomi "
             "reali), cieco (indicatori e pilastri rinominati in etichette "
             "neutre). Senza, vale solo `completo`",
    )
    parser.add_argument(
        "--livello-amministratori", default="", choices=("",) + tuple(LIVELLI),
        help="quanto del dominio vedono gli AMMINISTRATORI. Vuoto (predefinito) "
             "= prompt completo, come in tutte le campagne dell'archivio. "
             "Metterlo a `cieco` insieme a `--context-levels cieco` toglie il "
             "dominio a tutti e due i livelli, che e' la casella che manca",
    )
    parser.add_argument(
        "--variante-prompt", default="0", choices=sorted(SIGLE.values()),
        help="quale casella del disegno incrociato sui prompt: 0 = testo storico "
             "(italiano, «pilastro», il governatore non sa degli amministratori); "
             "A = italiano con «categoria»; B = A piu' la gerarchia dichiarata; "
             "F = traduzione inglese del testo storico; C = inglese con "
             "«category»; D = C piu' la gerarchia. Vale solo per il braccio llm",
    )
    parser.add_argument(
        "--thinking", default="", choices=("", "off", "low", "medium", "high", "dynamic"),
        help="l'interruttore unico del ragionamento, tradotto per API dal "
             "fornitore. Va dichiarato anche a `off`, perche' spento e' una "
             "scelta e cambia latenza e costo",
    )
    parser.add_argument(
        "--administrators", action="store_true",
        help="attiva lo strato amministrativo: un amministratore ogni "
             "--cells-per-district celle, la cella madre sotto il governo "
             "diretto. Moltiplica le chiamate per il numero di distretti",
    )
    parser.add_argument("--cells-per-district", type=int, default=3)
    parser.add_argument(
        "--admin-arm", default="", choices=("", "none", "random", "scripted", "llm", "semif", "hybrid"),
        help="braccio PROPRIO degli amministratori. Senza, seguono quello "
             "del governo, che e' il default del disegno. Serve per la casella "
             "'amministratori senza governo': li' non c'e' un braccio da "
             "seguire, e seguirlo darebbe amministratori assenti",
    )
    parser.add_argument(
        "--piano-serre-ridotto", action="store_true",
        help="riporta il piano alimentare a 0,10 pro capite e le serre iniziali "
             "al 5%% dei coloni. Era il comportamento fisso di questo script, ed "
             "e' stato misurato con l'ISRU spenta: con l'ISRU del prodotto "
             "congela la colonia ai soli fondatori (vedi il docstring). Non "
             "usarlo senza aver rimisurato quella tabella",
    )
    parser.add_argument(
        "--correzioni-motore",
        action="store_true",
        help="capienze senza l'ossigeno fantasma degli habitat e pesate per "
             "l'integrita' (braccio sperimentale, spento per difetto)",
    )
    parser.add_argument(
        "--pavimento-cantieri",
        type=float,
        default=None,
        help="priorita' minima di un cantiere aperto (1.1 = valore d'archivio; "
             "sotto 1.0 smette di espellere le altre azioni dal menu)",
    )
    parser.add_argument(
        "--strato-ambientale",
        action="store_true",
        help="accende lo strato ambientale: clima dal Mars Climate Database e "
        "modello planetario dinamico. Acceso nelle campagne nuove (dal 15 settembre 2026), spento nell'archivio precedente, "
        "dove il mondo gira su costanti marziane ferme.",
    )
    parser.add_argument(
        "--copertura-elettrica-vera",
        action="store_true",
        help="l'indicatore power_coverage dei governatori legge la copertura "
        "elettrica vera del passo (energia usata / carico) invece della giacenza "
        "avanzata dopo il consumo. Spento di default: le run gia' prodotte "
        "restano identiche.",
    )
    parser.add_argument(
        "--redistribuzione",
        action="store_true",
        help="accende la redistribuzione automatica di cella (deposito, "
        "prelievo, flusso fra celle vicine). Spenta di default: con lei la "
        "scarsita' smette di essere locale.",
    )
    parser.add_argument(
        "--admin-models",
        nargs="+",
        default=[],
        metavar="PROVIDER:MODELLO",
        help="modelli degli amministratori, uno pescato a caso per ogni distretto "
        "nuovo (riproducibile a parita' di seme), es. gpu_farm:gpt-oss:20b "
        "gpu_farm4:Qwen/Qwen3.8-27B-FP8. Con piu' voci il braccio si chiama "
        "+amm:misto. Implica amministratori con braccio llm proprio.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="ogni quanti passi salvare uno snapshot COMPATTO del mondo (celle "
        "occupate, con strutture o esplorate) per il replay nell'analysis "
        "frontend; 0 = nessuno. Alla cadenza del governatore si rivede ogni tornata.",
    )
    _add_semantic_arguments(parser)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = _default_output_root(args.arms, args.admin_arm, args.semantic_agents)

    args.out.mkdir(parents=True, exist_ok=True)
    # Il registro del terminale accanto ai risultati: una run da un'ora e
    # mezzo non deve lasciare solo dodici numeri.
    tee = _Tee(sys.stdout, args.out / "console.log")
    sys.stdout = tee
    if len(args.seeds) < 2 and len(args.arms) > 1:
        # Misurato (2026-08-24): su un seme solo il rumore del campionamento
        # softmax domina l'effetto di una regola blanda — la stessa
        # costituzione ha spostato le raccolte in DIREZIONI OPPOSTE in due
        # scenari. Un confronto fra bracci su un seme non e' un risultato:
        # e' un lancio di moneta messo in tabella.
        print(
            "[governatori] ATTENZIONE: un solo seme. Le differenze fra bracci "
            "su un seme singolo sono in gran parte rumore del campionamento; "
            "usare almeno 3 semi appaiati e leggere la concordanza di segno "
            "(scripts/compare_arms.py).",
            flush=True,
        )
    # **Il piano si costruisce prima di eseguirlo.** I gradini di contesto sono
    # bracci separati: `llm` con quattro gradini sono quattro trattamenti, non
    # uno con un'opzione. Espanderli qui, e non dentro il ciclo, rende il numero
    # di run dichiarabile prima di spendere le ore.
    # **La variante si verifica prima di spendere le ore.** `parole()` solleva
    # per una lingua non ancora tradotta; senza questo controllo la run
    # partirebbe, simulerebbe fino alla prima tornata del governatore e solo li'
    # scoprirebbe che il testo non esiste --- con una cartella a meta' e
    # un'ora buttata.
    if any(arm in {"llm", "hybrid"} for arm in args.arms) or args.admin_arm == "hybrid":
        from src.governors.varianti_prompt import parole as _parole_variante
        _parole_variante(variante_da_sigla(args.variante_prompt))

    semantic_template = None
    agents_template = None
    typesafe_kwargs = {
        "budget_usd": args.typesafe_budget_usd,
        "typesafe_model": args.typesafe_model,
        "question_type": args.typesafe_question_type,
    }
    if args.semantic_agents != "off":
        try:
            agents_template = _agents_section(
                args.semantic_agents,
                _semantic_section(
                    args.semantic_provider,
                    run_id="planned-run",
                    endpoint=args.jev_endpoint,
                    api_key_env=args.jev_api_key_env,
                    replay_from=args.semantic_replay_from,
                    timeout_seconds=args.semantic_timeout,
                    min_confidence=0.0,
                    agent_decision=args.semantic_agent_decision,
                    **typesafe_kwargs,
                ),
                cadence_steps=args.semantic_agents_cadence,
                levels=args.semantic_agents_levels,
                workers=args.semantic_agents_workers,
                hard_margin=args.semantic_agents_hard_margin,
            )
        except ValueError as error:
            parser.error(str(error))
    if any(arm in SEMANTIC_ARMS for arm in args.arms) or args.admin_arm in SEMANTIC_ARMS:
        try:
            semantic_template = _semantic_section(
                args.semantic_provider,
                run_id="planned-run",
                endpoint=args.jev_endpoint,
                api_key_env=args.jev_api_key_env,
                replay_from=args.semantic_replay_from,
                timeout_seconds=args.semantic_timeout,
                min_confidence=args.semantic_min_confidence,
                governor_decision=args.semantic_governor_decision,
                admin_decision=args.semantic_admin_decision,
                candidate_profile=args.semantic_candidate_profile,
                **typesafe_kwargs,
            )
        except ValueError as error:
            parser.error(str(error))

    livelli = args.context_levels or ["completo"]
    piano: list[tuple[str, str, bool, int, str]] = []
    for arm in args.arms:
        gradini = livelli if arm in {"llm", "hybrid"} else [""]
        for livello in gradini:
            for seed in args.seeds:
                trattamento = _semantic_treatment_hash(
                    arm,
                    args.admin_arm,
                    semantic_template if (
                        arm in SEMANTIC_ARMS or args.admin_arm in SEMANTIC_ARMS
                    ) else None,
                    agents_template,
                )
                piano.append((arm, livello, bool(args.administrators), seed, trattamento))

    # La ripresa. Una campagna da ore non deve ricominciare da zero perche' e'
    # caduta all'ultima run: ogni riga finita viene scritta subito su
    # `results.jsonl`, e una ripartenza salta cio' che c'e' gia'.
    percorso_jsonl = args.out / "results.jsonl"
    results: list[dict] = []
    if percorso_jsonl.exists():
        for riga in percorso_jsonl.read_text(encoding="utf-8").splitlines():
            if riga.strip():
                results.append(json.loads(riga))
    fatte = {
        _resume_key(
            r.get("braccio", r.get("arm")),
            int(r["seed"]),
            r.get("semantic_treatment_hash", ""),
        )
        for r in results
    }
    if fatte:
        print(f"[governatori] ripresa: {len(fatte)} run gia' in {percorso_jsonl}", flush=True)

    def planned_key(item):
        arm, livello, amm, seed, trattamento = item
        etichetta = _etichetta(
            arm, livello, amm, args.admin_arm,
            admin_misto=len(args.admin_models) > 1,
            variante_prompt=args.variante_prompt,
            livello_amministratori=args.livello_amministratori,
            semantic_provider=(
                args.semantic_provider
                if arm in SEMANTIC_ARMS or args.admin_arm in SEMANTIC_ARMS
                else ""
            ),
            semantic_agents=(
                f"{args.semantic_agents}:{args.semantic_provider}"
                if args.semantic_agents != "off" else ""
            ),
        )
        return _resume_key(etichetta, seed, trattamento)

    pending = sum(1 for item in piano if planned_key(item) not in fatte)
    print(f"[governatori] piano: {len(piano)} run, {pending} da eseguire", flush=True)
    for indice, (arm, livello, amm, seed, trattamento) in enumerate(piano, start=1):
        etichetta, _, _ = planned_key((arm, livello, amm, seed, trattamento))
        if (etichetta, seed, trattamento) in fatte:
            continue
        print(
            f"[governatori] {indice}/{len(piano)} braccio={etichetta} seme={seed}",
            flush=True,
        )
        esito = run_arm(
            arm,
            seed,
            args.steps,
            args.agents,
            args.out / f"{_nome_cartella(etichetta)}_seed{seed}",
            cadence_steps=args.cadence,
            provider=args.provider,
            model=args.model,
            log_interval=args.log_interval,
            wait_seconds=args.wait,
            effort=args.effort,
            temperature=args.temperature,
            log_agents=args.log_agents,
            map_profile=args.map_profile,
            dotazione=args.dotazione,
            isru=args.isru,
            sviluppo=args.sviluppo,
            context_level=livello or "completo",
            variante_prompt=args.variante_prompt,
            thinking=args.thinking,
            administrators=amm,
            cells_per_district=args.cells_per_district,
            admin_arm=args.admin_arm,
            livello_amministratori=args.livello_amministratori,
            piano_serre_ridotto=args.piano_serre_ridotto,
            correzioni_motore=args.correzioni_motore,
            pavimento_cantieri=args.pavimento_cantieri,
            strato_ambientale=args.strato_ambientale,
            redistribuzione=args.redistribuzione,
            copertura_elettrica_vera=args.copertura_elettrica_vera,
            snapshot_interval=args.snapshot_interval,
            admin_models=list(args.admin_models),
            semantic_provider=args.semantic_provider,
            jev_endpoint=args.jev_endpoint,
            jev_api_key_env=args.jev_api_key_env,
            semantic_replay_from=args.semantic_replay_from,
            semantic_run_id=args.semantic_run_id,
            semantic_timeout=args.semantic_timeout,
            semantic_min_confidence=args.semantic_min_confidence,
            semantic_governor_decision=args.semantic_governor_decision,
            semantic_admin_decision=args.semantic_admin_decision,
            semantic_candidate_profile=args.semantic_candidate_profile,
            semantic_agents=args.semantic_agents,
            semantic_agents_cadence=args.semantic_agents_cadence,
            semantic_agents_levels=args.semantic_agents_levels,
            semantic_agents_workers=args.semantic_agents_workers,
            semantic_agents_hard_margin=args.semantic_agents_hard_margin,
            semantic_agent_decision=args.semantic_agent_decision,
            typesafe_budget_usd=args.typesafe_budget_usd,
            typesafe_model=args.typesafe_model,
            typesafe_question_type=args.typesafe_question_type,
        )
        if esito.get("llm_muto") or esito.get("governo_mai_in_vigore"):
            # **Si ferma tutto, e subito.** Un governo che non parla mai chiude
            # identico alla baseline senza dare errore: proseguire
            # significherebbe spendere ore per riempire un `results.json` di
            # righe che sembrano un risultato. Il 2026-09-03 e' costato due
            # bracci interi, uno dei quali era gia' stato letto come esito.
            #
            # I due modi di non parlare sono distinti e vanno distinti anche
            # nel messaggio, perche' la cura e' diversa: la chiamata FALLISCE
            # (chiave, rete, modello inesistente) oppure NON TORNA IN TEMPO, e
            # in questo secondo caso non risulta nemmeno fra le fallite perche'
            # viene annullata mentre attende. La prima versione di questo
            # guardiano vedeva solo il primo modo, e due modelli piu' lenti
            # sotto carico gli sono passati sotto il naso.
            uso = esito.get("api") or {}
            gov = esito.get("governors") or {}
            if esito.get("llm_muto"):
                diagnosi = (
                    f"ha fatto {uso.get('calls', 0)} chiamate e NESSUNA e' riuscita.\n"
                    "  Cause tipiche: chiave assente o scaduta nell'ambiente DI\n"
                    "  QUESTO processo, provider irraggiungibile, modello non offerto."
                )
            else:
                diagnosi = (
                    f"ha mancato tutti i {gov.get('tick', 0)} confini di tick: "
                    "nessuna\n  direttiva e' mai entrata in vigore.\n"
                    f"  Causa tipica: l'attesa ({args.wait:g}s) e' piu' corta della\n"
                    "  latenza del modello sotto il carico attuale. Misurarla con\n"
                    "  check_provider MENTRE le altre campagne girano, non a vuoto."
                )
            print(
                f"\n[governatori] FERMO: {etichetta} seme {seed} {diagnosi}\n"
                "  La run e' indistinguibile da una senza governo.\n"
                "  Verificare con: python scripts/check_provider.py --provider "
                f"{args.provider} --model {args.model}\n"
                "  Le run gia' completate restano in results.jsonl e la campagna\n"
                "  riparte da li' una volta risolto.",
                flush=True,
            )
            sys.stdout = tee._stream
            tee.close()
            return 3
        results.append(esito)
        with percorso_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(esito, ensure_ascii=False) + "\n")
        print(
            f"[governatori]   -> vivi={esito['population']} "
            f"celle={esito['espansione'].get('celle_totali', '?')} "
            f"{esito['wall_clock_s']:.0f}s",
            flush=True,
        )
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"scritti {len(results)} risultati in {args.out / 'results.json'}")
    print(f"terminale registrato in {args.out / 'console.log'}")
    sys.stdout = tee._stream
    tee.close()
    return 0


#: Vero quando la run ha costruito il `PlanetaryCoupler`, e con lui il Mars
#: Climate Database. Vedi `_esci`.
_MCD_TOCCATO = False


def _esci(codice: int) -> None:
    """Termina il processo, aggirando la chiusura dell'interprete dopo il MCD.

    **Una run con `--strato-ambientale` non esce da sola.** Misurato il 13
    settembre 2026, due volte su due nella coda delle prove e poi riprodotto in
    dieci passi: la simulazione finisce, tutti gli artifact vengono scritti,
    l'ultima riga viene stampata, e il processo resta poi in attesa a tempo
    indefinito, con un solo thread e zero CPU. Non e' un ciclo, e' un blocco, e
    blocca la coda che aspetta quel processo.

    La causa e' stata isolata: importare `fmcd` non basta a provocarlo, ma dopo
    una sola `call_mcd` l'interprete non termina piu'. `fmcd` e' l'estensione
    Fortran del Mars Climate Database, compilata con MinGW e caricata dentro un
    Python costruito con MSVC; la pulizia di fine processo delle due librerie a
    tempo di esecuzione non si accorda.

    A quel punto non c'e' piu' niente da salvare, e si esce senza passare dalla
    finalizzazione. **`os._exit` non basta**, ed e' stato misurato: chiama
    `ExitProcess`, e `ExitProcess` esegue il distacco di ogni DLL caricata,
    cioe' proprio il passaggio in cui la libreria si blocca. Serve
    `TerminateProcess`, che il distacco lo salta: e' l'unica via d'uscita che
    Windows lascia a un processo le cui DLL non sanno chiudersi.

    Lo si fa **soltanto** dopo aver toccato il MCD. Altrove la chiusura
    ordinata e' quella giusta, e saltarla nasconderebbe i problemi invece di
    risolverli.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    if not _MCD_TOCCATO:
        raise SystemExit(codice)
    # Lo si dice: un'uscita che salta la finalizzazione non deve essere
    # invisibile a chi legge il registro della coda.
    print("[governatori] uscita forzata: dopo il MCD l'interprete non termina da solo",
          file=sys.stderr, flush=True)
    if sys.platform == "win32":
        import ctypes

        # I tipi vanno dichiarati. `GetCurrentProcess` restituisce lo
        # pseudo-handle -1, che su 64 bit e' 0xFFFFFFFFFFFFFFFF: col `c_int`
        # predefinito di ctypes viene troncato a 32 bit, e `TerminateProcess`
        # fallisce in silenzio su un handle che non esiste. E' successo.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        if not kernel32.TerminateProcess(kernel32.GetCurrentProcess(), codice):
            print(f"[governatori] TerminateProcess fallita, errore "
                  f"{ctypes.get_last_error()}", file=sys.stderr, flush=True)
    os._exit(codice)


if __name__ == "__main__":
    _esci(main())
