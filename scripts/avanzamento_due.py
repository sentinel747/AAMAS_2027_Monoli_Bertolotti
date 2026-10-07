# -*- coding: utf-8 -*-
"""Le due code delle due GPU, una sopra l'altra, nella stessa schermata.

**Perche' non bastano due finestre.** Le due code girano insieme ma non sono
indipendenti nel modo che conta: si contendono la CPU del portatile e finiscono
nello stesso disco, e se una si ferma sul cancello di qualita' l'altra prosegue
--- quindi la domanda «come stanno andando» ha senso solo al plurale. Due
terminali affiancati costringono a fare la somma a mente ogni volta.

Mostra anche, per ciascuna, se la sentinella di stop e' comparsa: e' la cosa che
si vuole vedere per prima al risveglio.

Uso:
    python scripts/avanzamento_due.py
    python scripts/avanzamento_due.py --intervallo 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import io  # noqa: E402
import time  # noqa: E402

from avanzamento import (  # noqa: E402
    _barra,
    _cadenza_e_passi,
    _cartelle,
    _fatte,
    _piano,
    _quadro,
)

#: Un registro scritto entro questo tempo e' di una run viva.
#:
#: **Il valore segue i limiti di attesa, non il ritmo normale.** Una tornata
#: scrive dopo che il governatore ha risposto (fino a 300s) e dopo che tutti i
#: distretti hanno risposto (limite HTTP 900s): una run viva puo' quindi tacere
#: venti minuti buoni. Tarare la soglia sul ritmo normale, che e' di qualche
#: minuto, faceva sparire dalla lista run che stavano solo aspettando.
FRESCO_S = 1320

#: Oltre questo silenzio la run viene mostrata con da quanto tace. Non e' un
#: allarme: e' la differenza fra «sta macinando» e «sta aspettando il
#: fornitore», che dalla sola barra non si vede.
LENTA_S = 360


def in_corso(piano: str) -> list[str]:
    """Tutte le run che stanno scrivendo adesso, non solo la piu' recente."""
    voci = _piano(Path(piano))
    if not voci:
        return []
    fatte = _fatte(voci)
    cadenza, passi = _cadenza_e_passi(voci)
    adesso = time.time()
    fuori = []
    for cartella in _cartelle(voci):
        try:
            sotto_cartelle = sorted(cartella.iterdir())
        except OSError:
            continue
        for sotto in sotto_cartelle:
            if not sotto.is_dir():
                continue
            registro = sotto / "governor_decisions.jsonl"
            istantanee = sotto / "world_snapshots"
            try:
                if registro.is_file():
                    eta = adesso - registro.stat().st_mtime
                    if eta > FRESCO_S:
                        continue
                    tornate = sum(1 for r in io.open(registro, encoding="utf-8",
                                                     errors="replace") if r.strip())
                elif istantanee.is_dir():
                    # **Il braccio senza governo non tiene registro delle
                    # decisioni.** Le istantanee del mondo pero' le scrive, una
                    # ogni `--snapshot-interval` passi: contarle dice a che punto
                    # sta, ed e' un segnale piu' diretto di una conseguenza.
                    eta = adesso - istantanee.stat().st_mtime
                    if eta > FRESCO_S:
                        continue
                    tornate = sum(1 for _ in istantanee.iterdir())
                else:
                    continue
            except OSError:
                continue
            seme = sotto.name.rsplit("seed", 1)[-1] if "seed" in sotto.name else ""
            if (any(n for n, s, c in voci if c == cartella and s == seme
                    and (n, s) in fatte)):
                continue
            nome = next((n for n, s, c in voci if c == cartella and s == seme), "")
            etichetta = f"{nome or cartella.name} seme {seme}"
            passo = min(passi, tornate * cadenza)
            quota = passo / passi if passi else 0.0
            attesa = f"   ferma da {eta / 60:.0f} min" if eta > LENTA_S else ""
            fuori.append(f"   {etichetta[:44]:<44} {_barra(quota, 18)}  "
                         f"{passo}/{passi}  {quota:5.1%}{attesa}")
    return sorted(fuori)

#: (titolo, piano, log, sentinella di stop)
CODE = [
    ("GPU 1 - Ollama - gpt-oss:20b",
     "runs/piano_base_pulita.txt", "runs/base_pulita.log", "runs/saltate_base_pulita.txt"),
    ("GPU 2 - vLLM - Qwen3.8-27B",
     "runs/piano_base_qwen.txt", "runs/base_qwen.log", "runs/saltate_base_qwen.txt"),
]
#: Le code del paper (01/10): lo stesso protocollo con l'indicatore corretto.
CODE_PAPER = [
    ("GPU 1 - Ollama - gpt-oss:20b (+ 120b, qwen3.6)",
     "runs/piano_paper_oss.txt", "runs/paper_oss.log", "runs/saltate_paper_oss.txt"),
    ("GPU 2 - vLLM - Qwen3.8-27B",
     "runs/piano_paper_qwen.txt", "runs/paper_qwen.log", "runs/saltate_paper_qwen.txt"),
]


def blocco(titolo: str, piano: str, log: str, ferma: str) -> list[str]:
    righe = [f"=== {titolo} ==="]
    # La coda non si ferma piu' su una run bocciata: la rifa' una volta e, se
    # fallisce ancora, la salta. Le saltate vanno viste subito, perche' sono
    # buchi nei dati che nessun'altra riga racconta.
    saltate = Path(ferma)
    if saltate.exists():
        elenco = [r for r in saltate.read_text(encoding="utf-8",
                                               errors="replace").splitlines() if r.strip()]
        if elenco:
            righe.append(f"!! {len(elenco)} RUN SALTATE dopo due tentativi:")
            righe += [f"   {r}" for r in elenco[:5]]
    try:
        # L'ultima riga di ogni quadro ripete ora e istruzioni: in una vista
        # doppia comparirebbe tre volte.
        # Del quadro di base si tiene tutto tranne la riga «ora:», che mostra
        # una run sola: al suo posto vanno tutte quelle vive.
        for r in _quadro(Path(piano), Path(log)):
            if r.startswith("aggiornato ") or r.startswith("ora: "):
                continue
            righe.append(r)
        vive = in_corso(piano)
        if vive:
            righe.append(f"in corso ({len(vive)}):")
            righe += vive
        else:
            righe.append("in corso: nessuna simulazione sta scrivendo.")
    except Exception as errore:  # noqa: BLE001 - un monitor non deve morire
        righe.append(f"non riesco a leggere il quadro: {errore}")
    return righe


def sentinella() -> list[str]:
    """Che cosa sta aspettando la sentinella del fornitore, se ce n'e' una."""
    registro = Path("runs/sentinella.log")
    try:
        righe = [r for r in io.open(registro, encoding="utf-8",
                                    errors="replace").read().splitlines() if r.strip()]
    except OSError:
        return []
    if not righe:
        return []
    ultima = righe[-1]
    attese = sum(1 for r in righe if "ancora occupato" in r)
    eta = (time.time() - registro.stat().st_mtime) / 60
    fuori = [f"sentinella: {ultima[len('[sentinella] '):][:110]}"]
    if attese:
        # Da quando aspetta: la prima riga di attesa dice l'inizio dello stallo.
        prima = next((r for r in righe if "ancora occupato" in r), "")
        fuori.append(f"   {attese} sondaggi a vuoto; l'ultimo {eta:.0f} min fa. "
                     f"Primo: {prima[13:32]}")
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervallo", type=float, default=15.0)
    ap.add_argument("--una-volta", action="store_true")
    ap.add_argument("--paper", action="store_true", help="le code del paper (paper_oss, paper_qwen)")
    args = ap.parse_args()
    if args.paper:
        CODE[:] = CODE_PAPER

    while True:
        righe: list[str] = []
        for titolo, piano, log, ferma in CODE:
            righe += blocco(titolo, piano, log, ferma)
            righe.append("")
        righe += [] if args.paper else sentinella()
        righe.append(f"aggiornato {time.strftime('%H:%M:%S')}   "
                     f"ctrl-c per uscire")
        testo = "\n".join(righe)
        if args.una_volta:
            print(testo)
            return 0
        # **Si cancella anche la cronologia (3J), non solo lo schermo.** Il
        # quadro e' piu' alto di un pannello di terminale: senza 3J le righe
        # che escono dall'alto si depositano nella cronologia a ogni passata, e
        # dopo mezzo minuto la stessa intestazione compare sei volte mozzata.
        # I fotogrammi vecchi del monitor non sono cronologia da salvare: il
        # racconto di cosa e' successo sta nel log della coda.
        print("\033[H\033[2J\033[3J" + testo, flush=True)
        time.sleep(args.intervallo)


if __name__ == "__main__":
    raise SystemExit(main())
