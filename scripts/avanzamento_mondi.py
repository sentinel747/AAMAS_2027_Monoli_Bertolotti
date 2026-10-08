# -*- coding: utf-8 -*-
"""Stato delle code di controllo, dal vivo, in una schermata sola.

Segue piu' bracci insieme (vedi `BRACCI`): per ciascuno una barra sul totale,
e sotto una barra per ogni run in corso, che cosa sta facendo (simula, oppure
aspetta il modello: il governatore o N amministratori), le tornate perse e, per
le run finite, il territorio contro la colonia senza governo dello stesso seme.
In cima la barra complessiva su tutti i bracci e la stima di fine.

**Le tornate perse sono la riga da guardare.** La tesi ne ha zero in ogni run;
una tornata persa lascia in vigore la legge precedente fino alla tornata dopo e
rende la run diversa dal protocollo. Se compare un numero rosso, la farm non sta
dietro.

Per seguire una coda nuova basta aggiungere una voce a `BRACCI`.
Solo lettura: non tocca ne' le code ne' le simulazioni.

Uso:
    python scripts/avanzamento_mondi.py
    python scripts/avanzamento_mondi.py --intervallo 5
    python scripts/avanzamento_mondi.py --una-volta
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
except ImportError:  # senza psutil manca solo «simula / aspetta»
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
MONDI = ("ice_rich", "fragmented", "high_hazard", "scarce_resources")
SEMI = (3, 4, 5, 6, 7)
NOMI = {"ice_rich": "ghiaccio", "fragmented": "frammentato", "high_hazard": "pericoloso",
        "scarce_resources": "scarso", "tesi": "imp. tesi", "system1": "imp. System1"}


@dataclass(frozen=True)
class Braccio:
    titolo: str
    runs: tuple            # (gruppo, seme, cartella relativa a ROOT)
    base: str              # risultati della colonia senza governo, con {gruppo} e {seme}
    passi: int = 1000
    cadenza: int = 25

    @property
    def tornate(self) -> int:
        return self.passi // self.cadenza


def _mondi(prefisso: str) -> tuple:
    return tuple((m, s, f"runs/controllo_copertura_mondi/{m}/{prefisso}{s}")
                 for s in SEMI for m in MONDI)


def _semi(gruppo: str, cartella: str, ripetute: dict | None = None) -> tuple:
    """Una run per seme; `ripetute` = {nome_ripetizione: semi} per le esecuzioni in piu'."""
    runs = [(gruppo, s, cartella.format(seme=s, rep="")) for s in SEMI]
    for rep, semi in (ripetute or {}).items():
        runs += [(gruppo, s, cartella.format(seme=s, rep=rep)) for s in semi]
    return tuple(runs)


BASE_MONDI = "runs/controllo_copertura_mondi/{gruppo}/ctrl_none_oggi_s{seme}/results.json"
BASE_TESI = "runs/base_qwen/ctrl_none/results.json"
BASE_SYSTEM1 = "runs/jev_semif_experiments/campaign_v2h_20260923/none_s{seme}/results.json"

BRACCI = [
    Braccio("mondi, indicatore corretto", _mondi("llm_amm_vera_s"), BASE_MONDI),
    Braccio("mondi, indicatore com'era (motore di oggi)", _mondi("llm_amm_com_era_s"), BASE_MONDI),
    Braccio("confronto: Qwen SCEGLIE (System 1), impostazione tesi",
            _semi("tesi", "runs/confronto_system1/tesi/qwen_s1{rep}_s{seme}", {"_rep2": (3,)}),
            BASE_TESI),
    # Jev in impostazione tesi: sospeso il 28/09 (TypeSafe 402), ripreso il 29/09.
    Braccio("confronto: Jev SCEGLIE (System 1), impostazione tesi",
            _semi("tesi", "runs/confronto_system1/tesi/jev_s1{rep}_s{seme}"), BASE_TESI),
    Braccio("confronto: Qwen SCRIVE (LLM), impostazione System 1",
            _semi("system1", "runs/confronto_system1/system1/qwen_llm{rep}_s{seme}",
                  {"_rep2": SEMI}),
            BASE_SYSTEM1, passi=2000, cadenza=20),
]

VERDE, GIALLO, ROSSO, GRIGIO, GRASSETTO, FINE = (
    "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[1m", "\033[0m")

_processi: dict[int, "psutil.Process"] = {}


def barra(quota: float, larghezza: int = 24) -> str:
    quota = max(0.0, min(1.0, quota))
    piene = int(round(quota * larghezza))
    return "█" * piene + "░" * (larghezza - piene)


def durata(secondi: float | None) -> str:
    if secondi is None or secondi < 0:
        return "--"
    ore, resto = divmod(int(secondi), 3600)
    return f"{ore}h{resto // 60:02d}" if ore else f"{resto // 60}m"


def righe_jsonl(f: Path) -> list[dict]:
    try:
        testo = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for r in testo.splitlines():
        if r.strip():
            try:
                out.append(json.loads(r))
            except json.JSONDecodeError:
                pass  # l'ultima riga puo' essere a meta' scrittura
    return out


def territorio(results: Path, seme: int) -> int | None:
    """Celle occupate del seme dato; il file puo' contenere piu' semi."""
    try:
        rs = json.loads(results.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for r in rs if isinstance(rs, list) else [rs]:
        if int(r.get("seed", seme)) == seme:
            return int(r["espansione"]["celle_totali"])
    return None


def processi_vivi() -> dict[str, "psutil.Process"]:
    """cartella --out (relativa a ROOT, con /) -> processo python della run."""
    if psutil is None:
        return {}
    vivi = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            nome = (p.info["name"] or "").lower()
            cmd = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not nome.startswith("python") or "--out" not in cmd:
            continue
        if not any("run_governor_experiment" in c for c in cmd):
            continue
        cartella = cmd[cmd.index("--out") + 1].replace("\\", "/").rstrip("/")
        relativa = cartella[cartella.find("runs/"):] if "runs/" in cartella else cartella
        vivi[relativa] = _processi.setdefault(p.info["pid"], p)
    return vivi


def stato_run(b: Braccio, gruppo: str, seme: int, relativa: str, vivi: dict) -> dict:
    cartella = ROOT / relativa
    info = {"gruppo": gruppo, "seme": seme, "nome": Path(relativa).name,
            "stato": "attesa", "quota": 0.0}
    sotto = next((d for d in cartella.glob("*_seed*") if d.is_dir()), None) \
        if cartella.is_dir() else None
    gov = righe_jsonl(sotto / "governor_decisions.jsonl") if sotto else []
    info["perse"] = sum(1 for r in gov if r.get("missed"))
    # **System 1: un errore del fornitore vale come una tornata persa.** La
    # decisione la prende il ripiego, non il modello. `selected_fallback` invece
    # e' il modello che sceglie «aspetta»: e' una scelta, non un guasto.
    # semantic_decisions.jsonl esiste solo a fine run; durante la run gli errori
    # si leggono nei registri tornata per tornata (grep sul testo: le decisioni
    # stanno annidate in forme diverse nei due file).
    if sotto is not None:
        for nome in ("governor_decisions.jsonl", "administrator_decisions.jsonl"):
            f = sotto / nome
            if f.exists():
                testo = f.read_text(encoding="utf-8", errors="replace")
                info["perse"] += len(re.findall(r'"fallback_reason": "(?:http_status_\d+|circuit_open)', testo))
    lat = [(r.get("governor") or {}).get("latency_s") for r in gov if not r.get("missed")]
    info["lat_gov"] = [x for x in lat if isinstance(x, (int, float))]
    if (cartella / "results.json").exists():
        info.update(stato="finita", quota=1.0, celle=territorio(cartella / "results.json", seme),
                    base=territorio(ROOT / b.base.format(gruppo=gruppo, seme=seme), seme))
        return info
    istantanee = sotto / "world_snapshots" if sotto else None
    n_ist = sum(1 for _ in istantanee.iterdir()) if istantanee and istantanee.is_dir() else 0
    info["quota"] = min(1.0, max(n_ist, len(gov) - 1, 0) / b.tornate)
    info["passo"] = int(info["quota"] * b.passi)
    proc = vivi.get(relativa)
    if proc is None:
        if sotto is not None:
            info["stato"] = "interrotta"
        return info
    info["stato"] = "in corso"
    try:
        info["trascorso"] = time.time() - proc.create_time()
        cpu = proc.cpu_percent(interval=None)  # dalla chiamata precedente
    except psutil.Error:
        info["trascorso"], cpu = None, None
    amm = righe_jsonl(sotto / "administrator_decisions.jsonl") if sotto else []
    distretti = amm[-1].get("districts", 0) if amm else 0
    # Ogni tornata: prima il governatore (una chiamata), poi gli amministratori
    # (una per distretto). Il processo quasi fermo vuol dire che aspetta il
    # modello; quale dei due lo dice chi ha gia' scritto la propria riga.
    if cpu is None or info["trascorso"] is None or info["trascorso"] < 20:
        info["fa"], info["in_volo"] = "avvio", 0
    elif cpu > 20:
        info["fa"], info["in_volo"] = "simula", 0
    elif len(gov) > len(amm):
        info["fa"], info["in_volo"] = f"attende {distretti} amministratori", max(distretti, 1)
    else:
        info["fa"], info["in_volo"] = "attende il governatore", 1
    return info


def eta_run(r: dict) -> float | None:
    if r.get("trascorso") and r["quota"] >= 0.05:
        return r["trascorso"] / r["quota"] * (1 - r["quota"])
    return None


def schermata() -> str:
    vivi = processi_vivi()
    per_braccio = [(b, [stato_run(b, g, s, rel, vivi) for g, s, rel in b.runs]) for b in BRACCI]
    tutte = [r for _, rr in per_braccio for r in rr]
    correnti = [r for r in tutte if r["stato"] == "in corso"]
    finite = [r for r in tutte if r["stato"] == "finita"]
    in_attesa = [r for r in tutte if r["stato"] in ("attesa", "interrotta")]

    quota = sum(r["quota"] for r in tutte if r["stato"] in ("finita", "in corso")) / len(tutte)
    L = [f"{GRASSETTO}Code di controllo e di confronto{FINE}   "
         f"{len(BRACCI)} bracci, {len(tutte)} run"]
    L.append("")
    L.append(f"TOTALE  {barra(quota, 40)}  {quota:5.1%}   finite {len(finite)}/{len(tutte)}   "
             f"in corso {len(correnti)}   in coda {len(in_attesa)}")
    in_volo = sum(r.get("in_volo", 0) for r in correnti)
    lat = [x for r in correnti for x in r["lat_gov"][-5:]]
    riga = f"        chiamate ai modelli in volo: ~{in_volo}"
    if lat:
        riga += (f"   governatore, ultime risposte: {statistics.median(lat):.1f}s mediana, "
                 f"{max(lat):.0f}s max")
    L.append(riga)
    perse_tot = sum(r["perse"] for r in tutte if r["stato"] in ("finita", "in corso"))
    L.append(f"        tornate perse: {ROSSO if perse_tot else VERDE}{perse_tot}{FINE}  (la tesi: 0)")

    for b, runs in per_braccio:
        fatte = sum(1 for r in runs if r["stato"] == "finita")
        vive = [r for r in runs if r["stato"] == "in corso"]
        q = sum(r["quota"] for r in runs if r["stato"] in ("finita", "in corso")) / len(runs)
        stime = [r["trascorso"] / r["quota"] for r in vive if r.get("trascorso") and r["quota"] >= 0.1]
        eta = None
        if stime:
            # Si simulano i posti della coda (xargs -P): ogni run in attesa parte
            # appena si libera il primo posto, non dopo la run piu' lunga.
            attesa = sum(1 for r in runs if r["stato"] in ("attesa", "interrotta"))
            durata_tipica = statistics.median(stime)
            # una run appena partita non ha ancora una stima propria: vale la
            # durata tipica meno quanto ha gia' fatto
            posti = sorted(eta_run(r) if eta_run(r) is not None
                           else max(0.0, durata_tipica - (r.get("trascorso") or 0)) for r in vive)
            for _ in range(attesa):
                posti[0] += durata_tipica
                posti.sort()
            eta = max(posti, default=0)
        L.append("")
        L.append(f"{GRASSETTO}{b.titolo}{FINE}   ({b.passi} passi)")
        L.append(f"   {barra(q, 30)}  {q:5.1%}   finite {fatte}/{len(runs)}   "
                 f"in corso {len(vive)}   fine stimata tra: {durata(eta)}")
        for r in vive:
            perse = (f"{ROSSO}{r['perse']} perse{FINE}" if r["perse"]
                     else f"{VERDE}0 perse{FINE}")
            fa = r.get("fa", "")
            etichetta = f"{NOMI.get(r['gruppo'], r['gruppo'])} s{r['seme']}"
            if "rep" in r["nome"]:
                etichetta += " (rip.)"
            L.append(f"   {etichetta:<22} {barra(r['quota'])}  "
                     f"{r.get('passo', 0):>4}/{b.passi}  {durata(r.get('trascorso')):>5} fatto, "
                     f"{durata(eta_run(r)):>5} da fare   {perse}   "
                     f"{GIALLO if fa.startswith('attende') else ''}{fa}{FINE}")
        gruppi = list(dict.fromkeys(g for g, _, _ in b.runs))
        for g in gruppi:
            pezzi = []
            for r in (x for x in runs if x["stato"] == "finita" and x["gruppo"] == g):
                if r.get("celle") is not None and r.get("base") is not None:
                    pezzi.append(f"s{r['seme']}{'r' if 'rep' in r['nome'] else ''} "
                                 f"{r['celle'] - r['base']:+d}"
                                 + (f" {ROSSO}({r['perse']} perse){FINE}" if r["perse"] else ""))
            if pezzi:
                L.append(f"   {GRIGIO}finite{FINE} {NOMI.get(g, g):<11} " + "   ".join(pezzi))
        interrotte = [r for r in runs if r["stato"] == "interrotta"]
        if interrotte:
            L.append(f"   {GIALLO}interrotte (la coda le rifa' da capo): "
                     + ", ".join(f"{NOMI.get(r['gruppo'], r['gruppo'])} s{r['seme']}"
                                 for r in interrotte) + FINE)
    L.append("")
    L.append(f"{GRIGIO}finite = territorio contro la colonia senza governo dello stesso seme "
             f"(r = ripetizione)   aggiornato {time.strftime('%H:%M:%S')}   ctrl-c per uscire{FINE}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervallo", type=float, default=10.0)
    ap.add_argument("--una-volta", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # le barre non stanno in cp1252
    os.system("")  # stringa costante vuota: attiva i colori ANSI nella console di Windows
    if args.una_volta:
        processi_vivi()
        for p in _processi.values():  # la CPU si misura fra due letture
            try:
                p.cpu_percent(interval=None)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
        print(schermata())
        return 0
    try:
        while True:
            testo = schermata()
            print("\033[H\033[2J\033[3J" + testo, flush=True)
            time.sleep(args.intervallo)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
