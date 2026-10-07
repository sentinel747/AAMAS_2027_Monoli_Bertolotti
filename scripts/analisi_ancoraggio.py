# -*- coding: utf-8 -*-
"""L'ancoraggio cognitivo, misurato sui registri delle campagne marziane.

**Da dove viene la domanda.** Nello studio sul dilemma del prigioniero
(Monoli et al., WOA 2026) un agente linguistico che ha scritto perche' ritorce
continua a ritorcere anche quando le prove sono state tolte dal prompt: la
propria traccia di ragionamento e' un'ancora. Governatore e amministratori di
questa tesi ricevono nel prompt la propria decisione precedente con la sua
motivazione. La misura gemella e' questa: quanto la decisione di oggi dipende
dalla decisione di ieri, A PARITA' DI EVIDENZA.

**Le tre misure.**

1. *Ancoraggio dell'amministratore.* Per ogni distretto, coppie di tornate
   consecutive (t-1, t). Si stima P(riscrive_t | riscriveva_{t-1}) e
   P(riscrive_t | accettava_{t-1}); la differenza e' l'ancoraggio grezzo. Poi
   si stratifica per l'evidenza che l'amministratore ha visto nel prompt a t,
   i morti nelle sue celle nella finestra [t-1, t): nessuno, oppure almeno
   uno. La differenza DENTRO ogni strato, pesata per il numero di coppie
   (Mantel-Haenszel), e' l'ancoraggio depurato dall'evidenza. Se resta
   positivo, l'amministratore riscrive perche' riscriveva.
2. *Persistenza delle regole dell'amministratore.* Fra due riscritture
   consecutive dello stesso distretto, quota di regole (testo identico) gia'
   presenti nella precedente.
3. *Persistenza della legge del governatore.* Fra tick consecutivi, quota di
   regole (condizione e pesi identici) gia' presenti al tick precedente, e
   quota di tick in cui la catena e' identica alla precedente.

Nessuna nuova esecuzione: solo `administrator_decisions.jsonl`,
`governor_decisions.jsonl` e `dead_agents.jsonl`, che ogni run con
amministratori scrive gia'.

Uso:
    python scripts/analisi_ancoraggio.py <cartella-braccio> [<altra-cartella> ...] [--out file.md]
dove ogni cartella-braccio contiene le run `<braccio>_seed<n>/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analisi_decentramento import (  # noqa: E402
    _cartelle_run,
    _jsonl,
    morti_per_cella_e_finestra,
    stato_del_distretto,
)


# --- 1. ancoraggio dell'amministratore -----------------------------------------------

def coppie_consecutive(tornate: list[dict], morti: list[dict]) -> list[dict]:
    """Per ogni distretto e per ogni coppia di tornate consecutive in cui compare:
    stato precedente, stato attuale, morti visti nella finestra [t-1, t).

    Le astensioni per guasto (policy None e accettata) contano come accettazioni:
    e' cio' che le celle hanno ricevuto ed e' cio' che l'amministratore rilegge.
    """
    ordinate = sorted((t for t in tornate if t.get("step") is not None), key=lambda t: int(t["step"]))
    coppie: list[dict] = []
    for i in range(1, len(ordinate)):
        prima, ora = ordinate[i - 1], ordinate[i]
        s0, s1 = int(prima["step"]), int(ora["step"])
        if s1 <= s0:
            continue
        precedenti = {int(d["district"]): d for d in prima.get("last_round", []) if "district" in d}
        for d in ora.get("last_round", []):
            if "district" not in d or int(d["district"]) not in precedenti:
                continue
            celle = [tuple(c) for c in d.get("cells", [])]
            visti = morti_per_cella_e_finestra(morti, celle, s0, s1) if celle else 0
            coppie.append({
                "distretto": int(d["district"]),
                "prima": "riscrive" if stato_del_distretto(precedenti[int(d["district"])]) != "accetta" else "accetta",
                "ora": "riscrive" if stato_del_distretto(d) != "accetta" else "accetta",
                "morti_visti": int(visti),
                "regole_prima": list(precedenti[int(d["district"])].get("rules") or []),
                "regole_ora": list(d.get("rules") or []),
            })
    return coppie


def _stato_binario(decisione: dict) -> str:
    return "riscrive" if stato_del_distretto(decisione) != "accetta" else "accetta"


def terne_consecutive(tornate: list[dict]) -> list[dict]:
    """Per ogni distretto, terne (t-2, t-1, t): serve al controllo del lag 2.

    **Perche'.** Nel prompt l'amministratore rilegge SOLO la decisione di t-1.
    Se e' il prompt ad ancorarlo, dato che a t-1 ha accettato, cio' che fece a
    t-2 non deve contare; se invece conta, e' lo STATO del distretto (un
    problema cronico) a spingerlo, non la propria traccia. E' la stessa
    logica con cui l'amnesia dell'articolo distingue evidenza e memoria.
    """
    ordinate = sorted((t for t in tornate if t.get("step") is not None), key=lambda t: int(t["step"]))
    terne: list[dict] = []
    for i in range(2, len(ordinate)):
        d2 = {int(d["district"]): d for d in ordinate[i - 2].get("last_round", []) if "district" in d}
        d1 = {int(d["district"]): d for d in ordinate[i - 1].get("last_round", []) if "district" in d}
        for d in ordinate[i].get("last_round", []):
            k = int(d.get("district", -1))
            if k not in d1 or k not in d2:
                continue
            terne.append({"lag2": _stato_binario(d2[k]), "lag1": _stato_binario(d1[k]), "ora": _stato_binario(d)})
    return terne


def controllo_lag2(terne: list[dict]) -> dict:
    """Fra chi ha ACCETTATO a t-1: P(riscrive_t | riscriveva a t-2) meno
    P(riscrive_t | accettava a t-2). Vicino a zero = conta solo cio' che il
    prompt ricorda; positivo = conta lo stato del distretto."""
    base = [t for t in terne if t["lag1"] == "accetta"]
    r = [t for t in base if t["lag2"] == "riscrive"]
    a = [t for t in base if t["lag2"] == "accetta"]
    pr = sum(t["ora"] == "riscrive" for t in r) / len(r) if r else None
    pa = sum(t["ora"] == "riscrive" for t in a) / len(a) if a else None
    return {
        "terne": len(base),
        "p_riscrive_dato_lag2_riscrive": pr,
        "p_riscrive_dato_lag2_accetta": pa,
        "dipendenza_lag2": round(pr - pa, 4) if pr is not None and pa is not None else None,
    }


def _quota(coppie: list[dict], prima: str) -> float | None:
    sel = [c for c in coppie if c["prima"] == prima]
    if not sel:
        return None
    return sum(1 for c in sel if c["ora"] == "riscrive") / len(sel)


def ancoraggio(coppie: list[dict]) -> dict:
    """Grezzo e stratificato per evidenza (morti visti: 0 / >0), con la
    differenza pesata alla Mantel-Haenszel."""
    fuori: dict = {
        "coppie": len(coppie),
        "p_riscrive_dato_riscriveva": _quota(coppie, "riscrive"),
        "p_riscrive_dato_accettava": _quota(coppie, "accetta"),
    }
    r, a = fuori["p_riscrive_dato_riscriveva"], fuori["p_riscrive_dato_accettava"]
    fuori["ancoraggio_grezzo"] = round(r - a, 4) if r is not None and a is not None else None

    strati = {"morti=0": [c for c in coppie if c["morti_visti"] == 0],
              "morti>0": [c for c in coppie if c["morti_visti"] > 0]}
    fuori["strati"] = {}
    pesato_num = pesato_den = 0.0
    for nome, sel in strati.items():
        rr, aa = _quota(sel, "riscrive"), _quota(sel, "accetta")
        diff = round(rr - aa, 4) if rr is not None and aa is not None else None
        fuori["strati"][nome] = {
            "coppie": len(sel),
            "p_riscrive_dato_riscriveva": rr,
            "p_riscrive_dato_accettava": aa,
            "ancoraggio": diff,
        }
        if diff is not None:
            n_r = sum(1 for c in sel if c["prima"] == "riscrive")
            n_a = len(sel) - n_r
            peso = (n_r * n_a) / len(sel) if len(sel) else 0.0
            pesato_num += diff * peso
            pesato_den += peso
    fuori["ancoraggio_depurato"] = round(pesato_num / pesato_den, 4) if pesato_den else None
    # Quanto l'evidenza conta da sola: P(riscrive | morti>0) - P(riscrive | morti=0).
    con = [c for c in coppie if c["morti_visti"] > 0]
    senza = [c for c in coppie if c["morti_visti"] == 0]
    if con and senza:
        fuori["effetto_evidenza"] = round(
            sum(c["ora"] == "riscrive" for c in con) / len(con)
            - sum(c["ora"] == "riscrive" for c in senza) / len(senza), 4)
    else:
        fuori["effetto_evidenza"] = None
    return fuori


# --- 2. persistenza delle regole dell'amministratore -------------------------------------

def persistenza_regole_amministratore(coppie: list[dict]) -> dict:
    """Fra due riscritture consecutive dello stesso distretto: quota delle regole
    di oggi (testo identico) che c'erano gia' ieri."""
    quote: list[float] = []
    identiche = 0
    for c in coppie:
        if c["prima"] != "riscrive" or c["ora"] != "riscrive" or not c["regole_ora"]:
            continue
        prima = set(map(str, c["regole_prima"]))
        ora = [str(r) for r in c["regole_ora"]]
        quote.append(sum(1 for r in ora if r in prima) / len(ora))
        if prima == set(ora):
            identiche += 1
    return {
        "coppie_riscrive_riscrive": len(quote),
        "quota_regole_riprese": round(statistics.fmean(quote), 4) if quote else None,
        "quota_catene_identiche": round(identiche / len(quote), 4) if quote else None,
    }


# --- 3. persistenza della legge del governatore -------------------------------------------

def _chiave_regola(regola: dict) -> str:
    return json.dumps(regola, sort_keys=True)


def persistenza_governatore(decisioni: list[dict]) -> dict:
    """Fra tick consecutivi con una politica: quota di regole gia' presenti al
    tick prima, quota di catene identiche, e la vita media di una regola in tick."""
    catene: list[list[str]] = []
    for r in sorted(decisioni, key=lambda r: int(r.get("tick", 0))):
        regole = ((r.get("policy") or {}).get("rules")) or []
        catene.append([_chiave_regola(x) for x in regole])
    quote: list[float] = []
    identiche = 0
    confronti = 0
    for prima, ora in zip(catene, catene[1:]):
        if not ora:
            continue
        confronti += 1
        quote.append(sum(1 for x in ora if x in set(prima)) / len(ora))
        if set(prima) == set(ora):
            identiche += 1
    # vita di una regola: tick consecutivi in cui resta nella catena
    vite: list[int] = []
    aperte: dict[str, int] = {}
    for catena in catene:
        presenti = set(catena)
        for k in list(aperte):
            if k not in presenti:
                vite.append(aperte.pop(k))
        for k in presenti:
            aperte[k] = aperte.get(k, 0) + 1
    vite.extend(aperte.values())
    return {
        "tick": len(catene),
        "quota_regole_riprese": round(statistics.fmean(quote), 4) if quote else None,
        "quota_catene_identiche": round(identiche / confronti, 4) if confronti else None,
        "vita_media_regola_tick": round(statistics.fmean(vite), 2) if vite else None,
        "vita_max_regola_tick": max(vite) if vite else None,
    }


# --- rapporto --------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def _segno(v) -> str:
    return "?" if v is None else ("+" if v > 0 else "-" if v < 0 else "0")


def analizza_run(cartella: Path) -> dict:
    tornate = _jsonl(cartella / "administrator_decisions.jsonl")
    morti = _jsonl(cartella / "dead_agents.jsonl")
    governatore = _jsonl(cartella / "governor_decisions.jsonl")
    coppie = coppie_consecutive(tornate, morti)
    return {
        "ancoraggio": ancoraggio(coppie),
        "lag2": controllo_lag2(terne_consecutive(tornate)),
        "regole_amministratore": persistenza_regole_amministratore(coppie),
        "governatore": persistenza_governatore(governatore),
    }


def rapporto(radici: list[Path]) -> list[str]:
    righe = ["# Ancoraggio cognitivo nei registri", ""]
    righe.append(
        "Amministratore: P(riscrive oggi | riscriveva ieri) meno P(riscrive oggi | accettava ieri), "
        "grezzo e dentro gli strati di evidenza (morti visti nella finestra: 0 / >0), pesato alla "
        "Mantel-Haenszel. «Evidenza» = P(riscrive | morti>0) - P(riscrive | morti=0). Governatore: "
        "quota di regole riprese dal tick precedente e vita media di una regola."
    )
    righe.append("")
    righe.append("| run | coppie | P(r|r) | P(r|a) | ancoraggio grezzo | morti=0: anc. (n) | morti>0: anc. (n) | ancoraggio depurato | evidenza | lag-2 dato accetta a t-1 (n) | regole amm. riprese | gov.: regole riprese | catene identiche | vita media regola |")
    righe.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    depurati: list[float] = []
    grezzi: list[float] = []
    evidenze: list[float] = []
    gov_riprese: list[float] = []
    lag2: list[float] = []
    for radice in radici:
        for seme, cartella in sorted(_cartelle_run(radice).items()):
            if not (cartella / "administrator_decisions.jsonl").exists():
                continue
            r = analizza_run(cartella)
            a, g, ra, l2 = r["ancoraggio"], r["governatore"], r["regole_amministratore"], r["lag2"]
            s0, s1 = a["strati"]["morti=0"], a["strati"]["morti>0"]
            etichetta = f"{radice.parent.name}/{radice.name}/seme {seme}"
            righe.append(
                f"| {etichetta} | {a['coppie']} | {_fmt(a['p_riscrive_dato_riscriveva'])} | "
                f"{_fmt(a['p_riscrive_dato_accettava'])} | {_fmt(a['ancoraggio_grezzo'])} | "
                f"{_fmt(s0['ancoraggio'])} ({s0['coppie']}) | {_fmt(s1['ancoraggio'])} ({s1['coppie']}) | "
                f"**{_fmt(a['ancoraggio_depurato'])}** | {_fmt(a['effetto_evidenza'])} | "
                f"{_fmt(l2['dipendenza_lag2'])} ({l2['terne']}) | "
                f"{_fmt(ra['quota_regole_riprese'])} | {_fmt(g['quota_regole_riprese'])} | "
                f"{_fmt(g['quota_catene_identiche'])} | {_fmt(g['vita_media_regola_tick'])} |"
            )
            if a["ancoraggio_depurato"] is not None:
                depurati.append(a["ancoraggio_depurato"])
            if a["ancoraggio_grezzo"] is not None:
                grezzi.append(a["ancoraggio_grezzo"])
            if a["effetto_evidenza"] is not None:
                evidenze.append(a["effetto_evidenza"])
            if g["quota_regole_riprese"] is not None:
                gov_riprese.append(g["quota_regole_riprese"])
            if l2["dipendenza_lag2"] is not None:
                lag2.append(l2["dipendenza_lag2"])
    righe.append("")
    if depurati:
        righe.append(
            f"- Ancoraggio dell'amministratore: grezzo medio {statistics.fmean(grezzi):+.3f}, "
            f"depurato dall'evidenza {statistics.fmean(depurati):+.3f}, segni "
            f"{''.join(_segno(v) for v in depurati)} ({sum(1 for v in depurati if v > 0)}/{len(depurati)} positivi). "
            f"Effetto dell'evidenza da sola: {statistics.fmean(evidenze):+.3f}."
        )
    if lag2:
        righe.append(
            f"- Controllo del lag 2 (dato che a t-1 ha accettato): dipendenza da t-2 media {statistics.fmean(lag2):+.3f}, "
            f"segni {''.join(_segno(v) for v in lag2)} ({sum(1 for v in lag2 if v > 0)}/{len(lag2)} positivi). "
            "Vicino a zero = conta solo la decisione che il prompt ricorda; positivo = conta lo stato del distretto."
        )
    if gov_riprese:
        righe.append(
            f"- Governatore: in media il {100 * statistics.fmean(gov_riprese):.0f}% delle regole di un tick "
            "era gia' nella catena del tick precedente."
        )
    return righe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radici", nargs="+", type=Path, help="cartelle di braccio con le run <braccio>_seed<n>/")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    testo = "\n".join(rapporto(args.radici)) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(testo, encoding="utf-8")
        print(f"scritto {args.out}")
    else:
        sys.stdout.write(testo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
