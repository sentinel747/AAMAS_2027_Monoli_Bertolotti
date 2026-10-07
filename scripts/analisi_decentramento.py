# -*- coding: utf-8 -*-
"""Che cosa ha fatto lo strato amministrativo, misurato dentro la run.

**Perche' non basta `compare_arms.py`.** Il confronto fra bracci dice se la
coppia governo+amministratori ha fatto meglio della baseline o del governo
solo; non dice se a fare la differenza sia stato il livello locale. Il registro
per tornata (`administrator_decisions.jsonl`) permette una misura che nessun
braccio esterno permette: nei distretti in cui l'amministratore ha RISCRITTO,
i morti per cella-passo sono scesi di piu' che nei distretti che nello stesso
tick hanno ACCETTATO? E' una differenza nelle differenze --- prima/dopo la
decisione, riscritto/accettato --- e toglie la parte di confondimento piu'
grossa: l'amministratore riscrive dove il distretto sta peggio, quindi il solo
confronto fra distretti riscritti e accettati direbbe che riscrivere uccide.

**Tre esiti, non due.** Dal 2026-09-05 il registro marca le riscritture *a
vuoto* (`rewrite_without_effect`): nessuna regola scattava sulle celle del
distretto, che sono rimaste senza politica. Si contano a parte, perche' sono
l'esito che la coppia accetta/riscrive nascondeva. Sui registri piu' vecchi il
campo manca e ogni riscrittura conta come tale: il rapporto lo dice.

**Il denominatore.** Morti per cella-passo, non per colono-passo: il registro
non porta la popolazione del distretto e il replay per eventi comprime
l'osservazione. Dentro lo stesso distretto le celle sono costanti fra due
tornate, quindi il prima/dopo non ne risente; il confronto fra distretti di
taglia diversa si', e va detto quando si riporta.

Uso::

    python scripts/analisi_decentramento.py runs/campagna_scarsa_v2 \
        --baseline runs/campagna_scarsa/ctrl_none \
        --confronto runs/campagna_scarsa \
        --out docs/benchmarks/2026-09-05-decentramento.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


# ------------------------------------------------------------------ predicati


def stato_del_distretto(decisione: dict) -> str:
    """`accetta`, `riscrive` o `a_vuoto`: il terzo esiste solo dal 2026-09-05."""
    if decisione.get("accepted_government_policy", True):
        return "accetta"
    return "a_vuoto" if decisione.get("rewrite_without_effect") else "riscrive"


def morti_per_cella_e_finestra(morti: list[dict], celle, inizio: int, fine: int) -> int:
    """Quanti decessi in queste celle con `inizio <= death_step < fine`."""
    insieme = {(int(c[0]), int(c[1])) for c in celle}
    conto = 0
    for m in morti:
        passo = m.get("death_step", m.get("step"))
        if passo is None or not (inizio <= int(passo) < fine):
            continue
        y, x = m.get("y"), m.get("x")
        if y is None or x is None:
            continue
        if (int(y), int(x)) in insieme:
            conto += 1
    return conto


def differenza_nelle_differenze(
    tornate: list[dict],
    morti: list[dict],
    salta_finestra_vista: bool = False,
    per_colono: bool = False,
) -> dict:
    """Per ogni decisione con una finestra prima e una dopo: delta dei morti per cella-passo.

    La prima tornata non ha un prima, l'ultima non ha un dopo: restano fuori.
    Le celle sono quelle del distretto alla tornata della decisione, per
    entrambe le finestre, cosi' il prima/dopo confronta lo stesso territorio.

    **`salta_finestra_vista` e' il controllo contro il ritorno alla media.**
    L'amministratore legge nel prompt i morti dell'ultima finestra e riscrive
    dove sono alti: quella finestra e' SELEZIONATA, e il dopo puo' scendere
    anche senza alcun effetto della riscrittura, perche' un picco passa da
    solo. Con il controllo il «prima» e' la finestra precedente a quella vista
    --- che l'amministratore non ha usato per decidere --- e il «dopo» resta
    quella successiva alla decisione. Se l'effetto sopravvive, non e' un
    artefatto della selezione. Costa una tornata in piu' all'inizio.

    **`per_colono`** usa come denominatore i coloni del distretto al momento
    della decisione (`population`, nei registri dal 2026-09-05 sera) invece
    delle celle. Dove il campo manca o vale zero si ricade sulle celle, cosi'
    i registri vecchi restano leggibili con lo stesso codice.
    """
    ordinate = sorted((t for t in tornate if t.get("step") is not None), key=lambda t: int(t["step"]))
    delte: dict[str, list[float]] = {"accetta": [], "riscrive": [], "a_vuoto": []}
    indietro = 2 if salta_finestra_vista else 1
    for i in range(indietro, len(ordinate) - 1):
        inizio_prima = int(ordinate[i - indietro]["step"])
        fine_prima = int(ordinate[i - indietro + 1]["step"])
        passo = int(ordinate[i]["step"])
        dopo_passo = int(ordinate[i + 1]["step"])
        if fine_prima <= inizio_prima or dopo_passo <= passo:
            continue
        for d in ordinate[i].get("last_round", []):
            celle = [tuple(c) for c in d.get("cells", [])]
            if not celle:
                continue
            n = len(celle)
            if per_colono and int(d.get("population") or 0) > 0:
                n = int(d["population"])
            prima = morti_per_cella_e_finestra(morti, celle, inizio_prima, fine_prima) / (n * (fine_prima - inizio_prima))
            dopo = morti_per_cella_e_finestra(morti, celle, passo, dopo_passo) / (n * (dopo_passo - passo))
            delte[stato_del_distretto(d)].append(dopo - prima)

    fuori: dict = {}
    for stato, valori in delte.items():
        fuori[stato] = {
            "decisioni": len(valori),
            "delta_medio": round(statistics.fmean(valori), 6) if valori else None,
        }
    r, a = fuori["riscrive"]["delta_medio"], fuori["accetta"]["delta_medio"]
    fuori["effetto"] = round(r - a, 6) if r is not None and a is not None else None
    return fuori


def demografia(riga: dict) -> dict:
    """Nascite, morti e morti per nascita di una run, dai soli totali di `results.jsonl`.

    **Perche' accanto ai vivi.** «Vivi a fine run» premia chi cresce: un governo
    prudente (meno morti, meno nascite) e uno incapace hanno lo stesso numero.
    Misurato il 2026-09-05: il regime LLM+amministratori ha i morti piu' bassi
    di ogni braccio E le nascite piu' basse, e perde sulla baseline. Senza
    questa riga il verdetto si legge come incapacita'.
    """
    vivi = int(riga.get("population", 0))
    morti = int(riga.get("deaths", 0))
    fondatori = int(riga.get("agents", 0))
    nascite = vivi + morti - fondatori
    return {
        "nascite": nascite,
        "morti": morti,
        "morti_per_nascita": round(morti / nascite, 3) if nascite > 0 else None,
    }


# ------------------------------------------------------------------- lettura


def _jsonl(percorso: Path) -> list[dict]:
    if not percorso.exists():
        return []
    righe = []
    with percorso.open(encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if riga:
                righe.append(json.loads(riga))
    return righe


def _risultati(cartella: Path) -> dict[int, dict]:
    return {int(r["seed"]): r for r in _jsonl(cartella / "results.jsonl")}


def _cartelle_braccio(radice: Path) -> list[Path]:
    return sorted(p for p in radice.iterdir() if p.is_dir() and (p / "results.jsonl").exists())


def _cartelle_run(braccio: Path) -> dict[int, Path]:
    """seme -> cartella della run, letto dal nome `..._seedN`."""
    fuori = {}
    for p in braccio.iterdir():
        if p.is_dir() and "_seed" in p.name:
            try:
                fuori[int(p.name.rsplit("_seed", 1)[1])] = p
            except ValueError:
                continue
    return fuori


def _segni(valori: list[float]) -> str:
    return "".join("+" if v > 0 else "-" if v < 0 else "0" for v in valori)


# ------------------------------------------------------------------ sezioni


def sezione_bracci(radice: Path, baseline: Path) -> list[str]:
    base = _risultati(baseline)
    semi = sorted(base)
    righe = [
        "## 1. Vivi a fine run, appaiati per seme con la baseline",
        "",
        f"Baseline: `{baseline}`. Semi: {', '.join(map(str, semi))}.",
        "",
        "| braccio | " + " | ".join(str(s) for s in semi) + " | media | vs baseline | segni |",
        "|---|" + "---:|" * len(semi) + "---:|---|---|",
    ]
    for cartella in _cartelle_braccio(radice):
        res = _risultati(cartella)
        vivi = [res[s]["population"] if s in res else None for s in semi]
        diff = [res[s]["population"] - base[s]["population"] for s in semi if s in res]
        ok = [v for v in vivi if v is not None]
        etichetta = next((res[s]["braccio"] for s in semi if s in res), cartella.name)
        righe.append(
            f"| {cartella.name} (`{etichetta}`) | "
            + " | ".join(str(v) if v is not None else "-" for v in vivi)
            + f" | {statistics.fmean(ok):.0f} | "
            + " ".join(f"{d:+d}" for d in diff)
            + f" | {_segni(diff)} ({sum(1 for d in diff if d > 0)}/{len(diff)} +) |"
        )
    righe += [
        "",
        "### 1b. Nascite e morti, appaiati per seme",
        "",
        "«Vivi» premia chi cresce: un governo prudente (meno morti, meno nascite) e uno "
        "incapace hanno lo stesso numero. Segni = braccio meno baseline, per seme.",
        "",
        "| braccio | nascite medie | vs baseline | morti medie | vs baseline | morti per nascita |",
        "|---|---:|---|---:|---|---:|",
    ]
    base_demo = {s: demografia(base[s]) for s in semi}
    righe.append(
        f"| baseline | {statistics.fmean(base_demo[s]['nascite'] for s in semi):.0f} | — | "
        f"{statistics.fmean(base_demo[s]['morti'] for s in semi):.0f} | — | "
        f"{statistics.fmean(base_demo[s]['morti_per_nascita'] or 0 for s in semi):.2f} |"
    )
    for cartella in _cartelle_braccio(radice):
        res = _risultati(cartella)
        comuni = [s for s in semi if s in res]
        if not comuni:
            continue
        demo = {s: demografia(res[s]) for s in comuni}
        d_n = [demo[s]["nascite"] - base_demo[s]["nascite"] for s in comuni]
        d_m = [demo[s]["morti"] - base_demo[s]["morti"] for s in comuni]
        righe.append(
            f"| {cartella.name} | {statistics.fmean(demo[s]['nascite'] for s in comuni):.0f} | "
            f"{_segni(d_n)} ({sum(1 for d in d_n if d > 0)}/{len(d_n)} +) | "
            f"{statistics.fmean(demo[s]['morti'] for s in comuni):.0f} | "
            f"{_segni(d_m)} ({sum(1 for d in d_m if d < 0)}/{len(d_m)} meno morti) | "
            f"{statistics.fmean(demo[s]['morti_per_nascita'] or 0 for s in comuni):.2f} |"
        )
    righe.append("")
    return righe


def sezione_confronto(radice: Path, altra: Path) -> list[str]:
    righe = [
        "## 2. Stesso braccio, prima e dopo la correzione",
        "",
        f"Appaiato per seme: `{radice}` meno `{altra}`.",
        "",
        "| braccio | per seme (nuovo - vecchio) | segni | media nuovo | media vecchio |",
        "|---|---|---|---:|---:|",
    ]
    for cartella in _cartelle_braccio(radice):
        vecchia = altra / cartella.name
        if not (vecchia / "results.jsonl").exists():
            continue
        nuovo, vecchio = _risultati(cartella), _risultati(vecchia)
        semi = sorted(set(nuovo) & set(vecchio))
        if not semi:
            continue
        diff = [nuovo[s]["population"] - vecchio[s]["population"] for s in semi]
        righe.append(
            f"| {cartella.name} | " + " ".join(f"{d:+d}" for d in diff)
            + f" | {_segni(diff)} | {statistics.fmean(nuovo[s]['population'] for s in semi):.0f}"
            + f" | {statistics.fmean(vecchio[s]['population'] for s in semi):.0f} |"
        )
    righe.append("")
    return righe


def sezione_strato(radice: Path) -> list[str]:
    righe = [
        "## 3. Che cosa ha fatto lo strato, tornata per tornata",
        "",
        "Una decisione = un distretto in una tornata. «A vuoto» = riscrittura in cui "
        "nessuna regola scattava sulle celle del distretto (campo presente solo nei "
        "registri dal 2026-09-05; dove manca, ogni riscrittura conta come tale).",
        "",
        "| run | tornate | decisioni | accetta | riscrive | a vuoto | campo a vuoto |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cartella in _cartelle_braccio(radice):
        for seme, run in sorted(_cartelle_run(cartella).items()):
            tornate = _jsonl(run / "administrator_decisions.jsonl")
            if not tornate:
                continue
            conti = {"accetta": 0, "riscrive": 0, "a_vuoto": 0}
            campo = False
            for t in tornate:
                for d in t.get("last_round", []):
                    conti[stato_del_distretto(d)] += 1
                    campo = campo or ("rewrite_without_effect" in d)
            tot = sum(conti.values()) or 1
            righe.append(
                f"| {cartella.name}/seme {seme} | {len(tornate)} | {tot} | "
                f"{conti['accetta'] / tot:.0%} | {conti['riscrive'] / tot:.0%} | "
                f"{conti['a_vuoto'] / tot:.0%} | {'si' if campo else 'no'} |"
            )
    righe.append("")
    return righe


def sezione_did(radice: Path) -> list[str]:
    righe = [
        "## 4. Dentro la run: riscrivere ha cambiato i morti?",
        "",
        "Differenza nelle differenze sui morti per cella-passo: per ogni decisione, "
        "(morti dopo - morti prima) nel distretto; poi la media dei distretti che hanno "
        "riscritto meno la media di quelli che hanno accettato nello stesso registro. "
        "Negativo = dove si e' riscritto i morti sono scesi di piu' (o saliti di meno). "
        "Il denominatore sono le celle-passo del distretto, non i coloni-passo: vedi il "
        "modulo per il perche'.",
        "",
        "L'ultima colonna e' il controllo contro il ritorno alla media: il «prima» e' la "
        "finestra PRECEDENTE a quella che l'amministratore ha visto nel prompt. Se l'effetto "
        "sopravvive li', non e' la selezione di un picco che passa da solo.",
        "",
        "| run | decisioni accetta | Δ accetta | decisioni riscrive | Δ riscrive | decisioni a vuoto | Δ a vuoto | effetto (riscrive - accetta) | effetto saltando la finestra vista | effetto per colono-passo | idem, finestra saltata |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_braccio: dict[str, list[float]] = {}
    per_braccio_controllo: dict[str, list[float]] = {}
    for cartella in _cartelle_braccio(radice):
        for seme, run in sorted(_cartelle_run(cartella).items()):
            tornate = _jsonl(run / "administrator_decisions.jsonl")
            if not tornate:
                continue
            morti = _jsonl(run / "dead_agents.jsonl")
            e = differenza_nelle_differenze(tornate, morti)
            c = differenza_nelle_differenze(tornate, morti, salta_finestra_vista=True)
            # Per colono-passo: solo dove il registro porta `population`
            # (dal 2026-09-05 sera); altrove coincide con la colonna per cella
            # e si stampa un trattino per non farlo sembrare una misura.
            ha_popolazione = any(
                int(d.get("population") or 0) > 0 for t in tornate for d in t.get("last_round", [])
            )
            pc = differenza_nelle_differenze(tornate, morti, per_colono=True) if ha_popolazione else None
            pcc = (
                differenza_nelle_differenze(tornate, morti, salta_finestra_vista=True, per_colono=True)
                if ha_popolazione else None
            )

            def _f(v):
                return "-" if v is None else f"{v * 1000:+.2f}"

            righe.append(
                f"| {cartella.name}/seme {seme} | {e['accetta']['decisioni']} | {_f(e['accetta']['delta_medio'])} | "
                f"{e['riscrive']['decisioni']} | {_f(e['riscrive']['delta_medio'])} | "
                f"{e['a_vuoto']['decisioni']} | {_f(e['a_vuoto']['delta_medio'])} | {_f(e['effetto'])} | {_f(c['effetto'])} | "
                f"{_f(pc['effetto']) if pc else '-'} | {_f(pcc['effetto']) if pcc else '-'} |"
            )
            if e["effetto"] is not None:
                per_braccio.setdefault(cartella.name, []).append(e["effetto"])
            if c["effetto"] is not None:
                per_braccio_controllo.setdefault(cartella.name, []).append(c["effetto"])
    righe.append("")
    righe.append("I Δ sono per mille celle-passo.")
    righe.append("")
    for braccio, effetti in per_braccio.items():
        negativi = sum(1 for v in effetti if v < 0)
        controllo = per_braccio_controllo.get(braccio, [])
        neg_c = sum(1 for v in controllo if v < 0)
        righe.append(
            f"- **{braccio}**: effetto negativo (riscrivere ha ridotto i morti piu' che accettare) "
            f"in {negativi}/{len(effetti)} semi; segni {_segni(effetti)}; "
            f"media {statistics.fmean(effetti) * 1000:+.2f} per mille celle-passo. "
            f"Controllo (finestra vista saltata): {neg_c}/{len(controllo)} negativi, segni {_segni(controllo)}, "
            f"media {statistics.fmean(controllo) * 1000:+.2f}." if controllo else
            f"- **{braccio}**: effetto in {negativi}/{len(effetti)} semi; segni {_segni(effetti)}."
        )
    righe.append("")
    return righe


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("radice", type=Path, help="cartella con i bracci (ognuno con results.jsonl e le run)")
    p.add_argument("--baseline", type=Path, required=True, help="cartella del braccio di riferimento")
    p.add_argument("--confronto", type=Path, help="radice di una campagna precedente con gli stessi nomi di braccio")
    p.add_argument("--out", type=Path, help="scrive il rapporto in Markdown")
    args = p.parse_args()

    righe = [f"# Decentramento -- {args.radice}", ""]
    righe += sezione_bracci(args.radice, args.baseline)
    if args.confronto:
        righe += sezione_confronto(args.radice, args.confronto)
    righe += sezione_strato(args.radice)
    righe += sezione_did(args.radice)
    testo = "\n".join(righe)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(testo, encoding="utf-8")
        print(f"scritto {args.out}")
    else:
        print(testo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
