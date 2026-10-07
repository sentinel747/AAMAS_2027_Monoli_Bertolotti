"""Confronto fra bracci: avanzamento, struttura, azioni, e cosa hanno consigliato.

`results.json` restituisce undici numeri per run. La domanda di ricerca ne chiede
altri: quale colonia avanza meglio e in cosa, che cosa fanno gli agenti, che cosa
consigliano i governatori e con quale motivazione. Questo strumento legge gli
artefatti che una run scrive gia' e li mette a confronto.

**Il confronto e' APPAIATO PER SEME, e non e' un dettaglio di presentazione.**
Nei dati reali il seme sposta la popolazione finale da 313 a 451 mentre un
braccio la sposta di pochi punti: una media fra semi diversi misurerebbe il seme
e chiamerebbe il risultato "effetto del governatore". E' lo stesso motivo per cui
i benchmark di questo progetto sono interlacciati A B / B A invece che a blocchi.
Con pochi semi l'unica statistica leggibile e' la **concordanza di segno**: se
tutti i semi si muovono nella stessa direzione la cosa e' suggestiva, se i segni
si mescolano non c'e' niente da dire.

**Il primo controllo e' che ci sia un esperimento.** Una run governata puo'
chiudere identica alla baseline per ragioni che non hanno niente a che vedere con
il governare -- cadenza piu' lunga della run, consenso assente, provider muto -- e
in quel caso il confronto non e' "il governatore non serve" ma "non e' stato
misurato niente". Lo strumento lo dice prima di ogni altra cosa e non lascia che
sia il lettore ad accorgersene.

Esempi::

    python scripts/compare_arms.py runs/governatori_farm
    python scripts/compare_arms.py runs/governatori_farm --out runs/confronto.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Eseguito come `python scripts/compare_arms.py`, `sys.path[0]` e' `scripts/` e
# non la radice: senza questa riga l'import di `src` fallisce e i nomi delle
# azioni restano indici. Gli altri script del progetto fanno lo stesso.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Le colonne su cui si confrontano i bracci. Non tutte quelle disponibili: sono
#: quelle che rispondono a "quale colonia avanza meglio e in cosa".
KEY_METRICS = (
    "population",
    "structures_built",
    "greenhouses",
    "colony_prosperity_index",
    "food_margin",
    "material_margin",
    "average_habitability",
    # La resa di una struttura scala con l'efficienza al quadrato: questa e' la
    # sola metrica che dice se le azioni di manutenzione di un braccio siano
    # servite o siano state spreco. Assente dai registri fino al 2026-08-18, e i
    # confronti precedenti non possono averla.
    "structure_integrity_mean",
)

#: Le metriche che crescono e poi si fermano, per cui "quando" dice piu' di
#: "quanto". Misurato sullo scenario sperimentale da 1500 passi: la baseline
#: raggiunge il proprio numero finale di serre al passo 276-303 su 1500 per due
#: semi su tre, cioe' l'80% della run avviene a costruzione FERMA, e il tetto
#: fisico e' saturo (margine 0). Su un esito cosi' nessun governatore puo'
#: aggiungere una serra, e un confronto sui soli valori finali riporterebbe
#: "governare non cambia nulla" per una ragione puramente meccanica. Il ritmo,
#: invece, resta osservabile.
GROWTH_METRICS = ("structures_built", "greenhouses")

#: Nome di cartella prodotto da `run_governor_experiment.py`: `<braccio>_seed<N>`.
_RUN_DIR = re.compile(r"^(?P<arm>.+)_seed(?P<seed>\d+)$")


def discover_runs(root: Path) -> dict[tuple[str, int], Path]:
    """Le cartelle di run trovate sotto `root`, indicizzate per (braccio, seme).

    **La ricerca e' ricorsiva, e serve a tenere insieme il confronto.** Una
    campagna lunga si lancia in piu' processi paralleli --- un braccio ciascuno,
    perche' l'attesa di rete si sovrapponga invece di sommarsi --- e ognuno
    scrive nella propria cartella. Se qui si guardasse un solo livello, il
    confronto appaiato andrebbe fatto dopo aver spostato a mano le run in una
    cartella comune: un passaggio manuale fra i dati e la lettura dei dati, cioe'
    il punto in cui si perdono i bracci senza accorgersene.

    **Il nome della cartella di flusso fa parte dell'identita' del braccio.**
    Due cartelle `llm_completo_seed3` sotto flussi diversi NON sono due run dello
    stesso trattamento: il nome del braccio porta arm, gradino di contesto e
    strato amministrativo, ma **non il modello**, e due governatori che parlano a
    modelli diversi sono due trattamenti. Scegliere fra loro a caso falserebbe il
    confronto; scartarne uno perderebbe un braccio. Il flusso li distingue, ed e'
    l'informazione che chi ha lanciato la campagna ha messo li' apposta
    (`--out runs/.../modello_gptoss120b`).

    Il prefisso si aggiunge SOLO dove serve, cosi' i nomi restano leggibili nel
    caso normale in cui i bracci sono gia' distinti.
    """
    trovate: list[tuple[str, int, Path]] = []
    for serie in sorted(root.rglob("state_timeseries.csv")):
        cartella = serie.parent
        match = _RUN_DIR.match(cartella.name)
        if match:
            trovate.append((match["arm"], int(match["seed"]), cartella))

    conteggio: Counter = Counter((arm, seme) for arm, seme, _ in trovate)
    found: dict[tuple[str, int], Path] = {}
    for arm, seme, cartella in trovate:
        nome = arm
        if conteggio[(arm, seme)] > 1:
            flusso = cartella.parent.name
            nome = f"{flusso}/{arm}" if flusso and cartella.parent != root else arm
        chiave = (nome, seme)
        if chiave in found:
            raise SystemExit(
                f"due run per {nome} seme {seme}: {found[chiave]} e {cartella}. "
                "Nemmeno il nome del flusso le distingue: rimuoverne una."
            )
        found[chiave] = cartella
    return found


def load_series(run_dir: Path) -> dict[str, list[float]]:
    """Le colonne numeriche della serie storica, come liste parallele a `step`.

    Le colonne non numeriche (sorgente del clima, scenario) vengono saltate:
    servono a documentare la run, non a confrontarla.
    """
    series: dict[str, list[float]] = defaultdict(list)
    with (run_dir / "state_timeseries.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    series[key].append(float(value))
                except ValueError:
                    continue
    return dict(series)


def endpoint(series: dict[str, list[float]]) -> dict[str, float]:
    return {name: series[name][-1] for name in KEY_METRICS if series.get(name)}


def first_divergence(a: dict[str, list[float]], b: dict[str, list[float]]) -> int | None:
    """Il primo passo in cui due run smettono di coincidere, o `None`.

    E' la diagnosi piu' economica che esista su una leva: dice non solo SE il
    braccio ha cambiato qualcosa ma QUANDO. Una divergenza al primo tick utile
    significa che la direttiva e' entrata in vigore; nessuna divergenza significa
    che non e' mai entrata, e allora il confronto degli esiti non ha oggetto.
    """
    steps = a.get("step") or []
    for index in range(min(len(steps), len(b.get("step") or []))):
        for name in KEY_METRICS:
            left, right = a.get(name), b.get(name)
            if not left or not right or index >= min(len(left), len(right)):
                continue
            if left[index] != right[index]:
                return int(steps[index])
    return None


def step_reaching(series: dict[str, list[float]], metric: str, target: float) -> int | None:
    """Il primo passo in cui `metric` raggiunge `target`, o `None` se mai."""
    steps = series.get("step") or []
    values = series.get(metric) or []
    for index, value in enumerate(values):
        if value >= target and index < len(steps):
            return int(steps[index])
    return None


def plateau_step(series: dict[str, list[float]], metric: str) -> int | None:
    """Il passo in cui la metrica raggiunge il valore che avra' alla fine.

    Quando questo numero e' molto minore della lunghezza della run, il resto
    della run e' a metrica ferma: qualunque direttiva applicata dopo agisce su
    una colonia che non ha piu' niente da costruire.
    """
    values = series.get(metric) or []
    return step_reaching(series, metric, values[-1]) if values else None


def action_mix(run_dir: Path) -> dict[str, int]:
    """Azioni ACCETTATE per tipo: cosa hanno effettivamente fatto gli agenti."""
    path = run_dir / "action_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: int(data.get("accepted", 0))
        for name, data in (payload.get("by_action") or {}).items()
    }


#: Vedi `SOGLIA_INSEDIAMENTO` in `run_governor_experiment.py` per come e' scelta.
SOGLIA_INSEDIAMENTO = 10


def settlements(run_dir: Path) -> dict:
    """Le celle che sono diventate insediamenti, non avamposti.

    **E' l'esito con un meccanismo dietro.** Su trenta run di controllo, quelle
    con una sola cella insediata finiscono fra 306 e 337 abitanti e quelle con
    due o piu' fra 373 e 579: due stati separati, senza niente in mezzo. La
    popolazione ne e' una lettura indiretta, e la sua mediana (313) e' esattamente
    il valore di uno dei due stati invece di un valore centrale. Vedi
    `docs/benchmarks/2026-08-20-biforcazione-espansione.md`.
    """
    path = run_dir / "cell_infrastructure.json"
    if not path.exists():
        return {}
    celle = json.loads(path.read_text(encoding="utf-8"))
    popolazioni = sorted((int(c.get("population", 0)) for c in celle), reverse=True)
    insediate = [p for p in popolazioni if p >= SOGLIA_INSEDIAMENTO]
    return {
        "celle_totali": len(celle),
        "celle_insediate": len(insediate),
        "espansa": len(insediate) >= 2,
        "seconda_cella": popolazioni[1] if len(popolazioni) > 1 else 0,
    }


def governor_activity(run_dir: Path) -> dict:
    """Cosa ha fatto il consiglio: tick, mancati, direttive, e cosa ha detto."""
    path = run_dir / "governor_decisions.jsonl"
    if not path.exists():
        return {"registro": "assente"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not rows:
        # Il caso che ha reso nulla una run vera: nessun confine di tick dopo il
        # primo, quindi nessuna direttiva promossa e nessun record scritto.
        return {"registro": "VUOTO", "tick": 0}

    # Formato v2 (governatore unico, 2026-08-24): una riga porta `governor` e
    # `policy`. Le righe del vecchio consiglio (`governors`/`directive`) non
    # vengono interpretate: meglio dichiararle che sommarle a zero.
    if rows and ("governors" in rows[0] or "directive" in rows[0]):
        return {
            "registro": "FORMATO CONSIGLIO (pre-2026-08-24)",
            "tick": len(rows),
            "nota": "registro del consiglio a mandati: leggerlo col commit che lo ha scritto",
        }

    speaking = [row for row in rows if (row.get("policy") or {}).get("rules")]
    weighted_pillars: Counter[str] = Counter()
    indicators_used: Counter[str] = Counter()
    weights: list[float] = []
    for row in speaking:
        for rule in (row["policy"] or {}).get("rules", []):
            condition = rule.get("if")
            if condition:
                indicators_used[str(condition.get("indicator", "?"))] += 1
            for pillar, value in (rule.get("weights") or {}).items():
                weighted_pillars[str(pillar)] += 1
                weights.append(float(value))

    governors = [row.get("governor") for row in rows if row.get("governor")]
    latencies = [float(g.get("latency_s", 0.0)) for g in governors]
    proposte = sum(1 for g in governors if (g.get("proposal") or {}).get("rules"))
    return {
        "registro": "presente",
        "tick": len(rows),
        "mancati": sum(1 for row in rows if row.get("missed")),
        "tick_con_policy": len(speaking),
        "proposte_con_regole": proposte,
        "pilastri_pesati": dict(weighted_pillars.most_common(8)),
        "indicatori_usati": dict(indicators_used.most_common(8)),
        "peso_min_max": (min(weights), max(weights)) if weights else None,
        "regole_per_policy_mediana": (
            statistics.median(len((row["policy"] or {}).get("rules", [])) for row in speaking)
            if speaking else 0
        ),
        "token_in": sum(int(g.get("tokens_in", 0)) for g in governors),
        "token_out": sum(int(g.get("tokens_out", 0)) for g in governors),
        "latenza_mediana_s": statistics.median(latencies) if latencies else 0.0,
        "motivazioni": [
            {"tick": row["tick"], "testo": (row.get("governor") or {}).get("rationale", "")}
            for row in rows
            if (row.get("governor") or {}).get("rationale")
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.4g}"


def build_report(root: Path, baseline: str, rationales: int) -> tuple[str, dict]:
    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"nessuna cartella <braccio>_seed<N> con serie storica sotto {root}")

    arms = sorted({arm for arm, _ in runs})
    seeds = sorted({seed for _, seed in runs})
    if baseline not in arms:
        raise SystemExit(f"braccio di riferimento {baseline!r} assente; presenti: {', '.join(arms)}")

    series = {key: load_series(path) for key, path in runs.items()}
    endpoints = {key: endpoint(value) for key, value in series.items()}
    lines: list[str] = [f"# Confronto fra bracci -- {root.name}", ""]
    lines.append(f"Bracci: {', '.join(arms)}. Semi: {', '.join(str(s) for s in seeds)}.")
    lines.append(f"Riferimento: `{baseline}`. Confronto **appaiato per seme**.")
    lines.append("")

    # -- 1. C'e' un esperimento? -------------------------------------------
    identical: list[tuple[str, int]] = []
    divergences: dict[tuple[str, int], int | None] = {}
    for arm in arms:
        if arm == baseline:
            continue
        for seed in seeds:
            here, there = series.get((arm, seed)), series.get((baseline, seed))
            if here is None or there is None:
                continue
            step = first_divergence(there, here)
            divergences[(arm, seed)] = step
            if step is None:
                identical.append((arm, seed))

    lines.append("## 1. C'e' un esperimento?")
    lines.append("")
    if identical:
        lines.append(
            f"**{len(identical)} run su {len(divergences)} sono IDENTICHE al braccio "
            f"`{baseline}`, passo per passo.** Per quelle il confronto degli esiti "
            "non ha oggetto: non dicono che governare non serva, dicono che non e' "
            "stato misurato niente."
        )
        lines.append("")
        for arm, seed in identical:
            attivita = governor_activity(runs[(arm, seed)])
            tick = int(attivita.get("tick", 0) or 0)
            parlanti = int(attivita.get("tick_con_policy", 0) or 0)
            mancati = int(attivita.get("mancati", 0) or 0)
            if attivita.get("registro") == "VUOTO":
                causa = (
                    "registro del governatore VUOTO: nessun confine di tick dopo il "
                    "primo, cioe' cadenza piu' lunga della run"
                )
            elif parlanti == 0 and mancati >= tick and tick:
                # Il caso misurato il 2026-08-18: 75 tick, 75 mancati, zero
                # policy. Dire "il governatore ha parlato" qui sarebbe falso, e
                # manderebbe a cercare il difetto nel posto sbagliato.
                causa = (
                    f"{mancati} mancati aggiornamenti su {tick} tick: nessuna "
                    "chiamata e' tornata entro il limite. Il governatore non ha "
                    "parlato affatto — vedi `miss_reason` nel registro"
                )
            elif parlanti == 0:
                causa = (
                    f"{tick} tick, nessuno con una policy: il governatore ha "
                    "risposto ma non ha chiesto niente (policy vuote, o risposte "
                    "fuori schema)"
                )
            else:
                causa = (
                    f"registro presente ({tick} tick, {parlanti} con policy): "
                    "il governatore ha parlato ma nessuna condizione ha morso"
                )
            lines.append(f"- `{arm}` seme {seed} -- {causa}")
        lines.append("")
    else:
        lines.append("Nessun braccio coincide con il riferimento: il confronto ha oggetto.")
        lines.append("")
    lines.append("| braccio | seme | primo passo di divergenza |")
    lines.append("|---|---:|---:|")
    for (arm, seed), step in sorted(divergences.items()):
        lines.append(f"| {arm} | {seed} | {step if step is not None else 'mai'} |")
    lines.append("")

    # -- 1-bis. Espansione ---------------------------------------------------
    # Prima delle medie, perche' una media su una grandezza a due gobbe non
    # descrive nessuna delle due.
    espansioni = {
        chiave: settlements(percorso) for chiave, percorso in runs.items()
    }
    if any(espansioni.values()):
        lines.append("## 1-bis. La colonia si e' espansa?")
        lines.append("")
        lines.append(
            "Una colonia che fonda una seconda cella da almeno "
            f"{SOGLIA_INSEDIAMENTO} abitanti finisce fra 373 e 579 coloni; una "
            "che resta su una sola cella finisce fra 306 e 337. Sono due stati "
            "separati, quindi si contano i ribaltamenti invece di fare una media."
        )
        lines.append("")
        lines.append("| braccio | seme | celle insediate | seconda cella | espansa | riferimento |")
        lines.append("|---|---:|---:|---:|---|---|")
        for (arm, seed), dati in sorted(espansioni.items()):
            if not dati:
                continue
            rif = espansioni.get((baseline, seed)) or {}
            stato_rif = (
                "-" if arm == baseline or not rif
                else ("espansa" if rif.get("espansa") else "arrestata")
            )
            lines.append(
                f"| {arm} | {seed} | {dati['celle_insediate']} | "
                f"{dati['seconda_cella']} | "
                f"{'si' if dati['espansa'] else 'no'} | {stato_rif} |"
            )
        ribaltamenti = [
            (arm, seed, dati["espansa"])
            for (arm, seed), dati in sorted(espansioni.items())
            if arm != baseline and dati and (espansioni.get((baseline, seed)) or {})
            and dati["espansa"] != (espansioni[(baseline, seed)] or {}).get("espansa")
        ]
        lines.append("")
        if ribaltamenti:
            su = sum(1 for _, _, e in ribaltamenti if e)
            lines.append(
                f"**{len(ribaltamenti)} ribaltamenti**: {su} verso l'espansione, "
                f"{len(ribaltamenti) - su} verso l'arresto."
            )
        else:
            lines.append(
                "**Nessun ribaltamento**: ogni braccio lascia la colonia nello "
                "stato in cui la baseline la porta."
            )
        lines.append("")

    # -- 2. Esiti appaiati --------------------------------------------------
    lines.append("## 2. Esiti, appaiati per seme")
    lines.append("")
    lines.append(
        "Ogni riga confronta un braccio col riferimento **sullo stesso seme**. "
        "Con pochi semi l'unica lettura difendibile e' la concordanza di segno: "
        "una colonna `segni` come `+++` dice che tutti i semi si muovono nella "
        "stessa direzione, `++-` che non c'e' niente da concludere."
    )
    lines.append("")
    # Oltre sei semi la tabella per-seme diventa illeggibile e, cosa peggiore,
    # smette di essere il modo giusto di guardare i dati: con molte ripetizioni
    # cio' che conta e' la DISTRIBUZIONE degli scarti appaiati, non il valore che
    # ciascun seme ha prodotto. Sotto quella soglia i singoli valori si leggono
    # ancora, e nasconderli toglierebbe informazione senza dare niente in cambio.
    riassunto = len(seeds) > 6
    for metric in KEY_METRICS:
        if not any(metric in endpoints.get((baseline, seed), {}) for seed in seeds):
            continue
        lines.append(f"### {metric}")
        lines.append("")
        if riassunto:
            lines.append(
                "| braccio | scarto mediano | intervallo | concordi | riferimento (mediana) |"
            )
            lines.append("|---|---:|---|---:|---:|")
        else:
            lines.append("| braccio | " + " | ".join(f"seme {s}" for s in seeds) + " | segni |")
            lines.append("|---" * (len(seeds) + 2) + "|")
            base_row = " | ".join(
                _fmt(endpoints.get((baseline, s), {}).get(metric, float("nan"))) for s in seeds
            )
            lines.append(f"| `{baseline}` (riferimento) | {base_row} | |")
        base_values = [
            v for s in seeds
            if (v := endpoints.get((baseline, s), {}).get(metric)) is not None
        ]
        for arm in arms:
            if arm == baseline:
                continue
            deltas: list[float] = []
            cells: list[str] = []
            for seed in seeds:
                here = endpoints.get((arm, seed), {}).get(metric)
                there = endpoints.get((baseline, seed), {}).get(metric)
                if here is None or there is None:
                    cells.append("-")
                    continue
                delta = here - there
                deltas.append(delta)
                cells.append(f"{_fmt(here)} ({delta:+.4g})")
            if not deltas:
                continue
            if riassunto:
                positivi = sum(1 for d in deltas if d > 0)
                # La concordanza si riporta come "il piu' numeroso su quanti", non
                # come "quanti positivi": con 3 su 10 il fatto notevole e' che
                # sette vanno nell'altra direzione, e leggere `3/10` invita a
                # concludere che non succeda niente.
                concordi = max(positivi, len(deltas) - positivi)
                verso = "+" if positivi >= len(deltas) - positivi else "-"
                lines.append(
                    f"| {arm} | {statistics.median(deltas):+.4g} | "
                    f"{min(deltas):+.4g} .. {max(deltas):+.4g} | "
                    f"{concordi}/{len(deltas)} {verso} | "
                    f"{_fmt(statistics.median(base_values)) if base_values else '-'} |"
                )
            else:
                segni = "".join("+" if d > 0 else "-" if d < 0 else "0" for d in deltas)
                lines.append(f"| {arm} | " + " | ".join(cells) + f" | {segni} |")
        lines.append("")
    if riassunto:
        lines.append(
            f"Con {len(seeds)} semi la lettura e' la **distribuzione** degli scarti "
            "appaiati. `concordi` dice quanti semi vanno nella direzione piu' "
            "numerosa: 5/10 e' l'assenza di segnale, 10/10 e' un effetto che il "
            "seme non spiega. L'intervallo dice quanto il seme da solo sposta il "
            "risultato, ed e' il metro con cui va giudicato qualunque scarto."
        )
        lines.append("")

    # -- 2-bis. Quando, non solo quanto ------------------------------------
    lines.append("## 2-bis. Quando, non solo quanto")
    lines.append("")
    lines.append(
        "Le metriche di costruzione **saturano**: il tetto per cella e' "
        "`ceil(occupanti / 7)` per tipo, e una volta raggiunto nessuna direttiva "
        "puo' aggiungere niente. Un confronto sui soli valori finali riporterebbe "
        "allora \"governare non cambia nulla\" per una ragione meccanica. Il ritmo "
        "resta osservabile: la colonna che conta e' **a quale passo un braccio "
        "raggiunge il valore finale del riferimento**, a parita' di seme."
    )
    lines.append("")
    tempi: dict[str, dict[str, int | None]] = {}
    for metric in GROWTH_METRICS:
        if not any(metric in endpoints.get((baseline, seed), {}) for seed in seeds):
            continue
        lines.append(f"### {metric}")
        lines.append("")
        if riassunto:
            lines.append("| braccio | passo mediano | intervallo | mai raggiunto |")
            lines.append("|---|---:|---|---:|")
        else:
            lines.append(
                "| braccio | " + " | ".join(f"seme {s}" for s in seeds) + " | (plateau del braccio) |"
            )
            lines.append("|---" * (len(seeds) + 2) + "|")
        for arm in arms:
            celle: list[str] = []
            plateau: list[str] = []
            raggiunti: list[int] = []
            mai = 0
            for seed in seeds:
                here = series.get((arm, seed))
                target = endpoints.get((baseline, seed), {}).get(metric)
                if here is None or target is None:
                    celle.append("-")
                    plateau.append("-")
                    continue
                quando = step_reaching(here, metric, target)
                tempi[f"{arm}_seed{seed}_{metric}"] = quando
                celle.append(str(quando) if quando is not None else "mai")
                if quando is None:
                    mai += 1
                else:
                    raggiunti.append(quando)
                fermo = plateau_step(here, metric)
                plateau.append(str(fermo) if fermo is not None else "-")
            etichetta = f"`{baseline}` (riferimento)" if arm == baseline else arm
            if riassunto:
                # I "mai" restano CONTATI a parte e non entrano nella mediana: un
                # braccio che non raggiunge il bersaglio non ci ha messo tanto,
                # non ci e' arrivato, e mescolare le due cose renderebbe un
                # fallimento indistinguibile da una lentezza.
                mediana = f"{statistics.median(raggiunti):.0f}" if raggiunti else "-"
                intervallo = (
                    f"{min(raggiunti)} .. {max(raggiunti)}" if raggiunti else "-"
                )
                lines.append(f"| {etichetta} | {mediana} | {intervallo} | {mai}/{len(seeds)} |")
            else:
                lines.append(
                    f"| {etichetta} | " + " | ".join(celle) + " | " + ", ".join(plateau) + " |"
                )
        lines.append("")

    lunghezza = max(
        (len(value.get("step") or []) for value in series.values()), default=0
    )
    fermi_presto = [
        (arm, seed, plateau_step(value, "greenhouses"))
        for (arm, seed), value in sorted(series.items())
        if arm == baseline and plateau_step(value, "greenhouses") is not None
    ]
    if lunghezza and fermi_presto:
        peggiore = min(step for _, _, step in fermi_presto if step)
        if peggiore < lunghezza * 0.5:
            lines.append(
                f"**Finestra utile breve.** Il riferimento raggiunge il proprio "
                f"numero finale di serre al passo {peggiore} su {lunghezza}: oltre "
                f"quel punto la costruzione e' ferma e una direttiva non ha piu' "
                f"nulla su cui agire. La cadenza va scelta perche' piu' direttive "
                f"cadano DENTRO quella finestra, non solo dentro la run."
            )
            lines.append("")

    # -- 3. Cosa hanno fatto gli agenti ------------------------------------
    lines.append("## 3. Cosa hanno fatto gli agenti")
    lines.append("")
    mixes = {key: action_mix(path) for key, path in runs.items()}
    all_actions = sorted({name for mix in mixes.values() for name in mix})
    if all_actions:
        lines.append("Azioni accettate, sommate sui semi, e scarto rispetto al riferimento.")
        lines.append("")
        lines.append("| azione | " + " | ".join(f"`{arm}`" for arm in arms) + " |")
        lines.append("|---" * (len(arms) + 1) + "|")
        totals = {
            arm: {name: sum(mixes.get((arm, s), {}).get(name, 0) for s in seeds) for name in all_actions}
            for arm in arms
        }
        for name in sorted(all_actions, key=lambda n: -totals[baseline].get(n, 0)):
            cells = []
            for arm in arms:
                value = totals[arm].get(name, 0)
                if arm == baseline:
                    cells.append(f"{value:,}")
                else:
                    delta = value - totals[baseline].get(name, 0)
                    cells.append(f"{value:,} ({delta:+,})" if delta else f"{value:,}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    # -- 4. Cosa ha consigliato il consiglio -------------------------------
    lines.append("## 4. Cosa hanno consigliato i governatori")
    lines.append("")
    activity: dict[tuple[str, int], dict] = {}
    for (arm, seed), path in sorted(runs.items()):
        if arm == baseline:
            continue
        activity[(arm, seed)] = governor_activity(path)
    if activity:
        lines.append("| braccio | seme | tick | mancati | con policy | token in/out | latenza med. |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for (arm, seed), data in activity.items():
            if data.get("registro") != "presente":
                lines.append(f"| {arm} | {seed} | registro {data.get('registro')} | | | | |")
                continue
            lines.append(
                f"| {arm} | {seed} | {data['tick']} | {data['mancati']} | "
                f"{data['tick_con_policy']} | {data['token_in']}/{data['token_out']} | "
                f"{data['latenza_mediana_s']:.2f} s |"
            )
        lines.append("")
        for (arm, seed), data in activity.items():
            if data.get("pilastri_pesati"):
                dettaglio = ", ".join(
                    f"{pilastro} x{conteggio}"
                    for pilastro, conteggio in data["pilastri_pesati"].items()
                )
                indicatori = ", ".join(
                    f"{nome} x{conteggio}"
                    for nome, conteggio in (data.get("indicatori_usati") or {}).items()
                ) or "nessuna condizione"
                lines.append(
                    f"`{arm}` seme {seed}: pilastri pesati [{dettaglio}]; "
                    f"condizioni su [{indicatori}]"
                )
        lines.append("")
        for (arm, seed), data in activity.items():
            if data.get("registro") != "presente":
                continue
            if not data.get("peso_min_max"):
                continue
            lines.append(
                f"`{arm}` seme {seed} ha pesato con valori fra "
                f"{data['peso_min_max'][0]:.2f} e {data['peso_min_max'][1]:.2f}; "
                f"regole per policy (mediana): {data['regole_per_policy_mediana']}"
            )
        lines.append("")

    # -- 5. Cosa hanno pensato ---------------------------------------------
    if rationales > 0:
        lines.append("## 5. Cosa hanno pensato")
        lines.append("")
        stampate = 0
        for (arm, seed), data in activity.items():
            for voce in data.get("motivazioni", []):
                if stampate >= rationales:
                    break
                testo = " ".join(str(voce["testo"]).split())
                lines.append(f"- `{arm}` seme {seed}, tick {voce['tick']}: {testo}")
                stampate += 1
        if stampate == 0:
            lines.append("Nessuna motivazione registrata.")
        lines.append("")

    payload = {
        "root": str(root),
        "baseline": baseline,
        "arms": arms,
        "seeds": seeds,
        "endpoints": {f"{arm}_seed{seed}": value for (arm, seed), value in endpoints.items()},
        "first_divergence": {f"{arm}_seed{seed}": step for (arm, seed), step in divergences.items()},
        "steps_to_baseline_final": tempi,
        "identical_to_baseline": [f"{arm}_seed{seed}" for arm, seed in identical],
        "governors": {
            f"{arm}_seed{seed}": {k: v for k, v in data.items() if k != "motivazioni"}
            for (arm, seed), data in activity.items()
        },
    }
    return "\n".join(lines), payload


def _console_must_not_crash_on_model_text() -> None:
    """La console non deve poter far fallire il rapporto.

    Le motivazioni sono testo di un modello, cioe' unicode arbitrario: una di
    esse conteneva U+2011 (trattino insecabile), che la console Windows in cp1252
    non sa codificare, e `print` sollevava `UnicodeEncodeError` **dopo** aver
    calcolato tutto il confronto. Il rapporto piu' interessante -- quello di una
    run LLM vera -- e' anche l'unico che puo' contenere quei caratteri.

    Si cambia solo la gestione degli errori, non la codifica: forzare UTF-8 su
    una console che non lo e' produrrebbe caratteri illeggibili al posto di un
    testo che si legge quasi tutto. Il file scritto con `--out` resta UTF-8 e
    completo: qui si degrada la sola copia a schermo.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - console esotica
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="cartella dell'esperimento")
    parser.add_argument("--baseline", default="none", help="braccio di riferimento")
    parser.add_argument("--rationales", type=int, default=10, help="quante motivazioni riportare")
    parser.add_argument("--out", type=Path, help="scrive il rapporto in un file Markdown")
    parser.add_argument("--json", type=Path, help="scrive i numeri anche come JSON")
    args = parser.parse_args()
    _console_must_not_crash_on_model_text()

    report, payload = build_report(args.root, args.baseline, args.rationales)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nrapporto scritto in {args.out}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 1 if payload["identical_to_baseline"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
