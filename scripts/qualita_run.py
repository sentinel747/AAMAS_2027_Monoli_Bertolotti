# -*- coding: utf-8 -*-
"""Le condizioni che rendono una run non comparabile, lette dal suo registro.

**Perche' un controllo automatico.** La tesi dichiara che una run non e'
comparabile se ha perso tornate del governatore o se ha avuto chiamate fallite:
nel primo caso un intervallo di passi resta senza legge, nel secondo il ripiego
del fornitore prende il posto di una decisione. Nessuna delle due cose fa
fallire la run: finisce, scrive i suoi artifact, e ha l'aria di tutte le altre.
Una campagna di venti ore puo' quindi riempirsi di righe inutilizzabili senza
che nessuno se ne accorga fino all'analisi.

Qui le condizioni si leggono subito dopo ogni run, e la coda si ferma alla
prima. Fermarsi costa qualche ora di macchina; non fermarsi costa la campagna.

**Le tre condizioni.**

*Tornate perse* (`governor_misses`): il governatore non ha consegnato in tempo e
le celle hanno tenuto la legge vecchia.

*Chiamate fallite* (`failed_calls` in `api_usage.json`): la rete o il fornitore
hanno risposto con un errore.

*Guasti degli amministratori* (`provider_failures`): lo stesso, al livello
locale. Va contato a parte, altrimenti un endpoint spento si traveste da stile
di governo --- un amministratore che non parla sembra un amministratore che
accetta.

**Le risposte malformate NON sono una condizione di ammissione.** L'amministratore
voleva riscrivere e non e' stato capito: e' un esito del modello, non un guasto
del banco di prova, e scartare la run per questo toglie il fenomeno invece del
rumore --- per giunta in modo selettivo, perche' i modelli piu' sconclusionati ne
producono di piu' e verrebbero giudicati sui loro giorni migliori. Si contano e
si riportano (`malformate()`), perche' una riscrittura non capita lascia la cella
al governo e senza dirlo somiglierebbe a un'accettazione.

**Una run assente conta come guasto solo se la si e' chiesta per nome.** Senza
`--seme` una cartella incompleta non e' un guasto: le run mancanti non sono
ancora state eseguite. Con `--seme`, invece, la domanda arriva subito dopo un
tentativo di esecuzione, e l'assenza significa che quel tentativo non ha
prodotto nulla --- fornitore caduto, processo morto, disco pieno. Rispondere «in
ordine» in quel caso ha gia' aperto otto buchi silenziosi in una campagna.

Uso:
    python scripts/qualita_run.py runs/prompt/variante_A          # tutta la cartella
    python scripts/qualita_run.py runs/prompt/variante_A --seme 3 # una run sola
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]


def _righe(cartella: Path) -> list[dict]:
    registro = cartella / "results.jsonl"
    fuori = []
    if not registro.exists():
        return fuori
    for riga in io.open(registro, encoding="utf-8", errors="replace"):
        riga = riga.strip()
        if not riga:
            continue
        try:
            fuori.append(json.loads(riga))
        except json.JSONDecodeError:
            continue
    return fuori


def _chiamate_fallite(cartella: Path, seme) -> int | None:
    """Dal file su disco, per le campagne che non lo scrivevano nel risultato.

    **`None` non e' zero.** Se il registro delle chiamate non c'e', o non si
    riesce a leggere, la condizione non e' verificata: restituire zero
    certificherebbe qualcosa che non si e' guardato. E' lo stesso errore gia'
    corretto qui sotto per i guasti del fornitore, e `audit_registri.py` tratta
    da sempre questo caso come un guasto.
    """
    for sotto in sorted(cartella.glob(f"*seed{seme}")):
        f = sotto / "api_usage.json"
        if f.exists():
            try:
                return int(json.loads(f.read_text(encoding="utf-8")).get("failed_calls", 0) or 0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return None
    return None


def guasti(cartella: Path, seme=None) -> list[str]:
    """I motivi per cui le run di questa cartella non sono comparabili.

    Lista vuota = tutto in ordine. Una run che non c'e' non e' un guasto: non
    e' ancora stata eseguita, e distinguere le due cose e' il punto.
    """
    problemi: list[str] = []
    for r in _righe(cartella):
        if seme is not None and int(r.get("seed", -1)) != int(seme):
            continue
        s = r.get("seed")
        perse = int(r.get("governor_misses") or 0)
        if perse:
            problemi.append(f"seme {s}: {perse} tornate del governatore perse")
        api = r.get("api") or {}
        fallite = api.get("failed_calls")
        if fallite is None:
            fallite = _chiamate_fallite(cartella, s)
        else:
            fallite = int(fallite or 0)
        if fallite is None:
            problemi.append(
                f"seme {s}: il registro delle chiamate non c'e', quindi "
                "«nessuna chiamata fallita» non e' verificato")
        elif fallite:
            problemi.append(f"seme {s}: {fallite} chiamate al modello fallite")
        amm = r.get("amministrazione") or {}
        for chiave, come in (("provider_failures",
                              "guasti del fornitore agli amministratori"),):
            if chiave not in amm:
                # **Assente non vuol dire zero.** Il contatore e' stato aggiunto
                # l'11 settembre: sulle run precedenti la sua mancanza veniva
                # letta come «nessun guasto», e la coda certificava una
                # condizione che non aveva modo di controllare. Se ci sono stati
                # amministratori e il contatore non c'e', va detto.
                if amm:
                    problemi.append(
                        f"seme {s}: il contatore «{chiave}» non e' nel registro, "
                        "quindi questa condizione non e' verificata")
                continue
            quanti = int(amm.get(chiave) or 0)
            if quanti:
                problemi.append(f"seme {s}: {quanti} {come}")
        if r.get("llm_muto"):
            problemi.append(f"seme {s}: il governatore non ha mai parlato")
        if r.get("governo_mai_in_vigore"):
            problemi.append(f"seme {s}: nessuna politica e' mai entrata in vigore")
    return problemi


def malformate(cartella: Path, seme=None) -> list[str]:
    """Le risposte non interpretabili: si dicono, non bocciano.

    Vanno dette lo stesso. Una riscrittura che il parser non capisce lascia la
    cella al governo, quindi somiglia a un'accettazione: senza questa riga
    l'analisi leggerebbe obbedienza dove c'e' stato un tentativo fallito.
    """
    fuori: list[str] = []
    for r in _righe(cartella):
        if seme is not None and int(r.get("seed", -1)) != int(seme):
            continue
        amm = r.get("amministrazione") or {}
        quanti = int(amm.get("malformed") or 0)
        if quanti:
            richieste = int(amm.get("interventions") or 0) + int(amm.get("abstentions") or 0)
            su = f" su {richieste} richieste" if richieste else ""
            fuori.append(f"seme {r.get('seed')}: {quanti} risposte degli "
                         f"amministratori non interpretabili{su}")
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cartella", type=Path)
    ap.add_argument("--seme", default=None)
    args = ap.parse_args()

    cartella = args.cartella if args.cartella.is_absolute() else RADICE / args.cartella

    # **Chiesta per nome e non c'e': e' un guasto, non un'assenza innocente.**
    if args.seme is not None:
        semi = {str(r.get("seed")) for r in _righe(cartella)}
        if str(args.seme) not in semi:
            print(f"RUN ASSENTE: il seme {args.seme} non ha prodotto alcun "
                  f"risultato in {cartella}.", file=sys.stderr)
            print("  L'esecuzione non e' arrivata a scrivere: fornitore caduto, "
                  "processo interrotto, o errore prima della fine.", file=sys.stderr)
            return 1

    problemi = guasti(cartella, args.seme)
    for nota in malformate(cartella, args.seme):
        print(f"nota: {nota}")
    if not problemi:
        print("in ordine")
        return 0
    print("RUN NON COMPARABILE:", file=sys.stderr)
    for p in problemi:
        print(f"  - {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
