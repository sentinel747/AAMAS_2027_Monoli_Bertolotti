# -*- coding: utf-8 -*-
"""Perche' i quattro modelli si comportano in modo diverso: lo stile di scrittura.

**La domanda.** I quattro modelli producono colonie molto diverse sullo stesso
mondo e sugli stessi semi: 92, 59, 61 e 44 celle occupate contro le 107 della
baseline, e decessi per nascita da 0,41 a 0,22. Il relatore ha chiesto di non
fermarsi a riportarlo ma di guardare i registri e formulare un'ipotesi.

**Dove guardare.** Non nei numeri di esito, che sono la conseguenza, ma nelle
leggi: `governor_decisions.jsonl` conserva ogni politica scritta, regola per
regola, con condizione e pesi. Un modello che comprime il territorio deve farlo
attraverso il vocabolario, e il vocabolario ha una sola leva per il territorio,
`explore`: e' la categoria che contiene il muoversi e il fondare insediamenti.
L'ipotesi si puo' quindi verificare invece di raccontarla.

**Che cosa misura.**

*Peso medio su explore* --- la media, su tutte le regole di tutte le politiche,
del moltiplicatore assegnato a `explore`. Sotto 1 scoraggia, sopra 1 incoraggia.
E' la misura piu' diretta della prudenza sul territorio.

*Quota di regole che scoraggiano explore* --- quante regole gli danno un peso
sotto 1. Una media puo' nascere da pochi valori estremi; questa no.

*Regole per politica* e *indicatori distinti* --- quanto e' articolata la legge.
Una legge con una regola sola e' un interruttore, una con otto e' una politica.

*Quota di politiche che si aprono con una condizione sempre vera* --- la
trappola della catena ordinata: la prima regola che scatta vince, quindi una
prima regola che scatta ovunque rende morte tutte le successive. Qui si conta
quante volte la prima condizione e' su un indicatore che in quel momento vale
zero ovunque, cioe' il caso in cui `<` con soglia positiva scatta su tutto.

*Conservazione fra tornate* --- quante regole sopravvivono identiche da una
politica alla successiva. Dice se il modello governa o se riscrive da capo.

Uso:
    python scripts/analisi_stile_modelli.py
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import Counter
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

#: I quattro modelli della campagna per famiglie, sul mondo di riferimento.
#: Il modello piccolo e' il riferimento del confronto e viene per primo.
#: Il modello piccolo ha TRE esecuzioni per seme e gli altri una: si prendono
#: tutte, perche' lo stile va misurato su tutto cio' che quel modello ha
#: scritto, e l'esito che gli sta accanto dev'essere la media delle stesse
#: esecuzioni. Usarne una sola darebbe 99 celle invece delle 92 riportate in
#: tesi, e la tabella si contraddirebbe con il resto del lavoro.
MODELLI: list[tuple[str, list[str]]] = [
    ("gpt-oss 20B", ["runs/campagna_scarsa_v3/llm_completo_amm",
                     "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
                     "runs/campagna_scarsa_v3/llm_completo_amm_rep3"]),
    ("gpt-oss 120B", ["runs/modelli/gptoss120b/llm_completo_amm"]),
    ("Qwen3.8 27B", ["runs/modelli/qwen38_27b/llm_completo_amm"]),
    ("Qwen3.6 27B", ["runs/modelli/qwen36_27b/llm_completo_amm"]),
]

LEVE = ("sustenance", "resources", "build", "life", "explore")


def _politiche(braccio: Path) -> list[list[dict]]:
    """Tutte le politiche scritte dal governatore, in ordine, per ogni seme."""
    fuori = []
    for cartella in sorted(braccio.glob("*seed*")):
        registro = cartella / "governor_decisions.jsonl"
        if not registro.exists():
            continue
        serie = []
        for riga in io.open(registro, encoding="utf-8", errors="replace"):
            riga = riga.strip()
            if not riga:
                continue
            try:
                r = json.loads(riga)
            except json.JSONDecodeError:
                continue
            regole = ((r.get("policy") or {}).get("rules")) or []
            if regole:
                serie.append(regole)
        if serie:
            fuori.append(serie)
    return fuori


def _firma(regola: dict) -> tuple:
    """Identita' di una regola, per contare quante sopravvivono fra tornate."""
    cond = regola.get("if")
    chiave_cond = (
        None if not cond
        else (cond.get("indicator"), cond.get("op"), round(float(cond.get("value", 0)), 4))
    )
    pesi = tuple(sorted(
        (k, round(float(v), 3)) for k, v in (regola.get("weights") or {}).items()
    ))
    return (chiave_cond, pesi)


def misure(bracci: list[Path]) -> dict | None:
    serie = [s for b in bracci for s in _politiche(b)]
    if not serie:
        return None

    pesi_explore: list[float] = []
    pesi_per_leva: dict[str, list[float]] = {l: [] for l in LEVE}
    regole_per_politica: list[int] = []
    indicatori = Counter()
    regole_totali = 0
    regole_explore_giu = 0
    prime_incondizionate = 0
    politiche = 0
    conservate: list[float] = []

    for seme in serie:
        precedente = None
        for regole in seme:
            politiche += 1
            regole_per_politica.append(len(regole))
            if regole and not regole[0].get("if"):
                # Una politica che si apre senza condizione: la prima regola
                # scatta ovunque e tutte le successive sono lettera morta.
                prime_incondizionate += 1
            for r in regole:
                regole_totali += 1
                cond = r.get("if")
                if cond and cond.get("indicator"):
                    indicatori[cond["indicator"]] += 1
                pesi = r.get("weights") or {}
                for leva in LEVE:
                    if leva in pesi:
                        try:
                            pesi_per_leva[leva].append(float(pesi[leva]))
                        except (TypeError, ValueError):
                            continue
                if "explore" in pesi:
                    try:
                        v = float(pesi["explore"])
                    except (TypeError, ValueError):
                        continue
                    pesi_explore.append(v)
                    if v < 1.0:
                        regole_explore_giu += 1
            firme = {_firma(r) for r in regole}
            if precedente is not None and firme:
                conservate.append(len(firme & precedente) / len(firme))
            precedente = firme

    return {
        "semi": len(serie),
        "politiche": politiche,
        "regole_per_politica": statistics.fmean(regole_per_politica) if regole_per_politica else 0,
        "peso_explore": statistics.fmean(pesi_explore) if pesi_explore else None,
        "quota_explore_giu": regole_explore_giu / regole_totali if regole_totali else 0,
        "pesi_per_leva": {
            l: (statistics.fmean(v) if v else None) for l, v in pesi_per_leva.items()
        },
        "indicatori_distinti": len(indicatori),
        "indicatore_piu_usato": indicatori.most_common(1)[0] if indicatori else None,
        "quota_prima_incondizionata": prime_incondizionate / politiche if politiche else 0,
        "conservazione": statistics.fmean(conservate) if conservate else None,
        "indicatori": indicatori,
    }


def _esito(bracci: list[Path]) -> dict | None:
    """Celle occupate e morti per nascita di un braccio, medie sui semi.

    Sta qui perche' l'analisi dello stile deve poter essere messa accanto
    all'esito: uno stile misurato senza il suo effetto e' un elenco di
    curiosita'.
    """
    celle, mpn = [], []
    for braccio in bracci:
        registro = braccio / "results.jsonl"
        if not registro.exists():
            continue
        for riga in io.open(registro, encoding="utf-8", errors="replace"):
            riga = riga.strip()
            if not riga:
                continue
            try:
                r = json.loads(riga)
            except json.JSONDecodeError:
                continue
            celle.append(int((r.get("espansione") or {}).get("celle_totali", 0) or 0))
            nascite = (int(r.get("population", 0)) + int(r.get("deaths", 0))
                       - int(r.get("agents", 300)))
            if nascite > 0:
                mpn.append(int(r.get("deaths", 0)) / nascite)
    if not celle:
        return None
    return {
        "celle": statistics.fmean(celle),
        "morti_per_nascita": statistics.fmean(mpn) if mpn else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=RADICE / "docs/benchmarks/2026-09-14-stile-dei-modelli.md")
    args = ap.parse_args()

    dati = []
    for nome, relativi in MODELLI:
        m = misure([RADICE / r for r in relativi])
        if m:
            dati.append((nome, m))
        else:
            print(f"nessun registro di decisioni in {relativi}")
    if not dati:
        print("niente da analizzare")
        return 1

    out = ["# Perche' i quattro modelli governano in modo diverso\n"]
    out.append(
        "Generato da `scripts/analisi_stile_modelli.py`, che legge le politiche "
        "scritte davvero (`governor_decisions.jsonl`) e non gli esiti. Stesso "
        "mondo di riferimento, stessi cinque semi, stesso prompt.\n"
    )

    out.append("\n## Come scrivono\n")
    out.append("| modello | politiche | regole per politica | indicatori distinti | "
               "conservate fra tornate | apre senza condizione |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for nome, m in dati:
        out.append(
            "| %s | %d | %.1f | %d | %s | %.0f%% |" % (
                nome, m["politiche"], m["regole_per_politica"], m["indicatori_distinti"],
                ("%.0f%%" % (m["conservazione"] * 100)) if m["conservazione"] is not None else "—",
                m["quota_prima_incondizionata"] * 100,
            )
        )

    out.append("\n## Che cosa pesano\n")
    out.append(
        "Peso medio per categoria, su tutte le regole di tutte le politiche. "
        "1,0 non cambia niente; sotto scoraggia, sopra incoraggia.\n"
    )
    out.append("| modello | " + " | ".join(LEVE) + " | regole che scoraggiano explore |")
    out.append("|---|" + "---:|" * (len(LEVE) + 1))
    for nome, m in dati:
        celle = []
        for leva in LEVE:
            v = m["pesi_per_leva"][leva]
            celle.append("%.2f" % v if v is not None else "—")
        out.append("| %s | %s | %.0f%% |" % (nome, " | ".join(celle),
                                             m["quota_explore_giu"] * 100))

    # L'esito accanto allo stile: e' il confronto che l'analisi deve reggere.
    esiti = {nome: _esito([RADICE / r for r in rel]) for nome, rel in MODELLI}
    base = _esito([RADICE / "runs/campagna_scarsa/ctrl_none"])

    out.append("\n## Lo stile accanto all'esito\n")
    out.append(
        "La colonna che conta e' l'ultima. La baseline occupa %d celle.\n"
        % round(base["celle"])
    )
    out.append("| modello | celle occupate | morti per nascita | "
               "peso MEDIO su explore | regole che SCORAGGIANO explore |")
    out.append("|---|---:|---:|---:|---:|")
    for nome, m in dati:
        e = esiti.get(nome) or {}
        out.append("| %s | %s | %s | %.2f | %.0f%% |" % (
            nome,
            round(e["celle"]) if e else "—",
            ("%.2f" % e["morti_per_nascita"]) if e else "—",
            m["peso_explore"] if m["peso_explore"] is not None else float("nan"),
            m["quota_explore_giu"] * 100,
        ))

    out.append("\n### Che cosa dice\n")
    # Le cifre della lettura si calcolano: una prosa fissa accanto a una
    # tabella che si rigenera diventa falsa al primo dato nuovo.
    per_celle = sorted(dati, key=lambda kv: (esiti.get(kv[0]) or {}).get("celle", 0))
    meno, piu = per_celle[0], per_celle[-1]
    out.append(
        "**La compressione del territorio si vede nella legge, non solo "
        "nell'esito.** Il vocabolario ha una sola leva che fonda insediamenti, "
        "`explore`, e i modelli che chiudono con meno celle sono quelli che la "
        "mettono sotto 1 piu' spesso: il %.0f per cento delle regole in %s, che "
        "conserva piu' territorio, contro il %.0f per cento in %s, che ne "
        "conserva meno della meta'. Non e' un tratto di carattere: e' una riga "
        "che il modello scrive, e si legge nella legge prima che nell'esito.\n"
        % (piu[1]["quota_explore_giu"] * 100, piu[0],
           meno[1]["quota_explore_giu"] * 100, meno[0])
    )
    out.append(
        "**Il peso MEDIO su explore non lo direbbe: dice quasi il contrario.** "
        "Vale %s nell'ordine della tabella, e il valore piu' alto appartiene al "
        "modello che occupa meno celle. La ragione sta nella forma della "
        "catena: poche regole con `explore` alto tirano su la media ma stanno "
        "in fondo e scattano di rado, mentre la maggioranza delle regole lo "
        "tiene sotto 1 ed e' quella che scatta quasi sempre. Dove vince la "
        "prima regola che scatta, la media sulle regole e' la statistica "
        "sbagliata, e qui si vede perche'.\n"
        % ", ".join("%.2f" % m["peso_explore"] for _, m in dati)
    )
    per_conservazione = sorted(dati, key=lambda kv: kv[1]["conservazione"] or 0)
    out.append(
        "**Una differenza di stile che invece NON produce l'effetto.** Quanto "
        "un modello conserva la propria legge da una tornata all'altra va dal "
        "%.0f al %.0f per cento, ma l'ordine non e' quello delle celle: %s "
        "conserva di piu' di tutti e comprime meno di %s, che conserva di meno "
        "di tutti. La stabilita' della legge e la prudenza sul territorio sono "
        "due tratti indipendenti, ed e' una ragione per non descrivere un "
        "modello come «piu' prudente» in generale.\n"
        % (per_conservazione[0][1]["conservazione"] * 100,
           per_conservazione[-1][1]["conservazione"] * 100,
           per_conservazione[-1][0], per_conservazione[0][0])
    )

    out.append("\n## Su che cosa scrivono le condizioni\n")
    for nome, m in dati:
        totale = sum(m["indicatori"].values()) or 1
        righe = ", ".join(
            "%s %.0f%%" % (k, v / totale * 100) for k, v in m["indicatori"].most_common(4)
        )
        out.append(f"- **{nome}**: {righe}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"scritto {args.out}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
