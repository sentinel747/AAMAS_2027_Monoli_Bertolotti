# -*- coding: utf-8 -*-
"""Il modello vede TUTTO il prompt, o il server glielo tronca in silenzio?

**Perche' la domanda non e' oziosa.** Un server che tronca non da' errore: da'
una risposta. Se il taglio togliesse la testa del prompt --- dove stanno il ruolo,
il vocabolario e le regole del formato --- il modello risponderebbe lo stesso,
con una politica magari valida ma scritta senza sapere che cosa gli era stato
chiesto. Il registro non se ne accorgerebbe: nessuna chiamata fallita, nessuna
risposta malformata.

**Come si verifica in un colpo solo.** Si mette una parola d'ordine casuale in
cima a un prompt lungo quanto i nostri (o piu'), si riempie il mezzo di
riempitivo, e in fondo si chiede di ripeterla. Se torna indietro, la testa e'
arrivata. Se non torna, il server ha tagliato --- e si vede *quanto* provando
lunghezze crescenti.

Uso:
  MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_contesto.py
  MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_contesto.py --provider gpu_farm4 \\
      --model Qwen/Qwen3.8-27B-FP8 --lunghezze 6000 12000 20000
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.provider_registry import create_llm_provider  # noqa: E402

#: Quattro caratteri per token e' l'approssimazione d'uso per l'inglese: serve
#: solo a costruire un prompt della lunghezza voluta, non a contare.
CARATTERI_PER_TOKEN = 4

RIEMPITIVO = (
    "Filler line {n}: the colony logs routine telemetry with no operational "
    "content whatsoever and nothing here should influence any answer.\n"
)


def prompt_di_prova(parola: str, token_voluti: int) -> str:
    testa = (
        f"SECRET WORD: {parola}\n"
        "Remember the secret word above. You will be asked for it at the end.\n\n"
    )
    coda = (
        "\n\nTask: reply with a JSON object with exactly one key, "
        '"secret", whose value is the SECRET WORD given at the very top of '
        "this message. If you cannot find it, use the value \"NON_TROVATA\".\n"
    )
    quanti = max(0, token_voluti * CARATTERI_PER_TOKEN - len(testa) - len(coda))
    mezzo = []
    lunghezza = 0
    n = 0
    while lunghezza < quanti:
        riga = RIEMPITIVO.format(n=n)
        mezzo.append(riga)
        lunghezza += len(riga)
        n += 1
    return testa + "".join(mezzo) + coda


def prova(provider, parola: str, token: int) -> tuple[bool, int, str]:
    testo = prompt_di_prova(parola, token)
    risposta = provider.complete_json(testo)
    grezza = getattr(risposta, "text", "") or ""
    dentro = getattr(risposta, "tokens_in", 0) or 0
    errore = getattr(risposta, "error", "") or ""
    if errore:
        return False, int(dentro), f"errore: {errore[:80]}"
    try:
        letto = str(json.loads(grezza).get("secret", "")).strip()
    except (TypeError, ValueError):
        letto = grezza.strip()[:60]
    return (parola in letto), int(dentro), letto[:60]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="gpu_farm")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--lunghezze", type=int, nargs="+",
                    default=[1000, 4000, 6000, 8000, 12000])
    args = ap.parse_args()

    provider = create_llm_provider(args.provider, args.model)
    nome = type(provider).__name__
    print(f"fornitore: {args.provider} / {args.model}  ({nome})")
    if nome == "FallbackProvider":
        print("ATTENZIONE: e' il provider di ripiego, non sta chiamando niente.")
        print("Serve MARSABM_ALLOW_LLM_CALLS=1 e la chiave nel .env locale.")
        return 1

    print("%10s %12s  %-8s %s" % ("token voluti", "token visti", "testa?", "risposta"))
    for quanti in args.lunghezze:
        parola = "ZQ" + secrets.token_hex(4).upper()
        ok, dentro, letto = prova(provider, parola, quanti)
        print("%10d %12d  %-8s %s" % (quanti, dentro, "SI" if ok else "NO", letto))
    print()
    print("«testa? SI» = il modello ha letto la prima riga del prompt, quindi")
    print("nessun troncamento a quella lunghezza.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
