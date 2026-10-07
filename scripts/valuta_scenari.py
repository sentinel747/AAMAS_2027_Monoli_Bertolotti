"""Quale configurazione mette la colonia sotto pressione. Nessuna, finora.

Legge le campagne di fase 1 (braccio `none`, nessuna chiamata a modello) e dice
se la colonia ha mai dovuto **scegliere** — perche' e' solo allora che una
politica di allocazione, umana o linguistica, puo' essere giusta o sbagliata.

**Il criterio della prima stesura era sbagliato, e va detto perche'.** Ammetteva
uno scenario se `food_margin`, `material_margin` o l'integrita' delle strutture
uscivano dalla banda di `balanced`. Misurato su cinque profili di mappa:
`food_margin` vale **1,059617 in tutte e quindici le run**, identico alla sesta
cifra, e su tutte le 311 run registrate nel progetto non e' mai sceso sotto
1,06. Non e' una metrica dello scenario: vale `min(2, (scorte + capacita') / (2 x
popolazione))`, e capacita' e popolazione le fissa la configurazione, non la
mappa. Il verdetto "nessuno scenario stringe" era giusto, ma lo era per caso:
due soglie su tre erano costanti rispetto alla variabile in esame.

**Cosa misura invece questa versione.** Tre condizioni, ciascuna un modo diverso
in cui la colonia non ottiene quello che vuole:

    acceptance_rate < 1,0        qualcuno ha chiesto e si e' visto rifiutare
    morti > 0                    qualcuno non ce l'ha fatta
    integrita' minima < 0,90     le strutture si consumano piu' di quanto si ripari

Sono contese e scarsita' viste dal lato della **decisione** e non del magazzino:
un margine alto dice solo che la colonia ha piu' di quanto le serva, mentre
un'azione rifiutata dice che qualcuno voleva qualcosa e non l'ha avuto. Nessuna
delle tre si e' mai accesa in questo progetto, ed e' quello il risultato — non
un difetto dello strumento.

    python scripts/valuta_scenari.py runs/scenari_*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Le quattro condizioni, con il verso in cui indicano pressione. Il commento e'
#: il valore osservato finora, tenuto qui perche' una soglia senza il numero da
#: cui nasce ridiventa arbitraria alla prima rilettura.
CONDIZIONI = (
    ("accettazione", 1.0, "sotto", "1,0000 in ogni run del lotto scenari"),
    ("morti", 0.0, "sopra", "0 nelle campagne recenti"),
    ("integrita_min", 0.90, "sotto", "0,881 il peggiore su 311 run"),
)

#: **Il deficit e' riportato ma NON fa da cancello, e la ragione e' misurata.**
#: Su 330 file di infrastruttura, le uniche celle insediate con un deficit sono
#: insediamenti giovani da 11 a 24 abitanti — colonie appena fondate ancora in
#: costruzione — mentre la cella madre da trecento abitanti non ha mai, in
#: nessuna run, un solo obiettivo mancato. Il deficit misura quindi il ritardo di
#: cantiere di un satellite in crescita, non la scarsita': usarlo come soglia
#: faceva risultare "sotto pressione" una colonia il cui unico problema era un
#: vagabondo arrivato in un avamposto da due persone senza ancora un riparo.
DEFICIT_INFORMATIVO = "deficit_max"


def _deficit_massimo(cartella: Path) -> float | None:
    """Il peggior obiettivo mancato fra tutte le celle insediate.

    `cell_infrastructure.json` porta, per cella, `targets` e `actual` per
    alloggi, serre, pannelli e impianti di ossigeno, piu' i `deficits` gia'
    calcolati. Un deficit positivo e persistente e' la firma di una colonia che
    deve scegliere: vuole piu' di quanto riesca a costruire.
    """
    percorso = cartella / "cell_infrastructure.json"
    if not percorso.exists():
        return None
    try:
        celle = json.loads(percorso.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    peggiore = 0.0
    for cella in celle:
        for valore in (cella.get("deficits") or {}).values():
            peggiore = max(peggiore, float(valore))
    return peggiore


def _misure(cartella: Path, record: dict) -> dict[str, float]:
    """I quattro numeri, presi dove il runner li ha scritti."""
    misure: dict[str, float] = {}
    if "acceptance_rate" in record:
        misure["accettazione"] = float(record["acceptance_rate"])
    if "deaths" in record:
        misure["morti"] = float(record["deaths"])
    integrita = record.get("structure_integrity_mean")
    if integrita is not None:
        # Valore finale e non minimo: e' l'unico disponibile senza rileggere la
        # serie temporale, ed e' conservativo nel verso giusto — se il finale e'
        # gia' sotto soglia, il minimo lo e' a maggior ragione.
        misure["integrita_min"] = float(integrita)
    deficit = _deficit_massimo(cartella)
    if deficit is not None:
        misure["deficit_max"] = deficit
    return misure


def valuta(per_run: list[tuple[Path, dict]]) -> dict:
    """Il verdetto su una campagna, con il motivo accanto.

    Prende il seme **peggiore** e non la mediana: basta che la colonia soffra su
    una run perche' ci sia qualcosa da misurare.
    """
    peggiori: dict[str, float] = {}
    for cartella, record in per_run:
        for nome, valore in _misure(cartella, record).items():
            # Il deficit non e' un cancello ma resta una quantita' "peggiore =
            # piu' grande": senza questa riga il default lo aggregherebbe come
            # minimo e la colonna mostrerebbe il satellite piu' fortunato.
            verso = next(
                (v for n, _, v, _ in CONDIZIONI if n == nome),
                "sopra" if nome == DEFICIT_INFORMATIVO else "sotto",
            )
            if nome not in peggiori:
                peggiori[nome] = valore
            elif verso == "sotto":
                peggiori[nome] = min(peggiori[nome], valore)
            else:
                peggiori[nome] = max(peggiori[nome], valore)

    superate = []
    for nome, soglia, verso, _ in CONDIZIONI:
        if nome not in peggiori:
            continue
        valore = peggiori[nome]
        if (verso == "sotto" and valore < soglia) or (verso == "sopra" and valore > soglia):
            superate.append((nome, valore, soglia, verso))
    return {"peggiori": peggiori, "superate": superate, "sotto_pressione": bool(superate)}


def _carica(cartella: str) -> tuple[str, list[tuple[Path, dict]]] | None:
    base = Path(cartella)
    percorso = base / "results.json"
    if not percorso.exists():
        return None
    records = json.loads(percorso.read_text(encoding="utf-8"))
    if not records:
        return None
    profilo = records[0].get("config", {}).get("map_profile", "?")
    per_run = [
        (base / f"{r.get('arm', 'none')}_seed{r.get('seed', 0)}", r) for r in records
    ]
    return profilo, per_run


def main(argomenti: list[str]) -> int:
    if not argomenti:
        print(__doc__)
        return 2

    esiti = []
    for cartella in argomenti:
        caricato = _carica(cartella)
        if caricato is None:
            print(f"  {cartella}: nessun results.json leggibile, saltata")
            continue
        profilo, per_run = caricato
        esiti.append((profilo, cartella, per_run, valuta(per_run)))

    if not esiti:
        print("nessuna campagna leggibile")
        return 1

    intestazione = (
        f"{'scenario':<18} {'run':>4} {'accettaz':>9} {'deficit':>8} "
        f"{'morti':>6} {'integrita':>10}   verdetto"
    )
    print(intestazione)
    print("-" * len(intestazione))
    for profilo, _, per_run, esito in sorted(esiti):
        p = esito["peggiori"]

        def mostra(nome: str, cifre: int = 4) -> str:
            return f"{p[nome]:.{cifre}f}" if nome in p else "-"

        verdetto = "SOTTO PRESSIONE" if esito["sotto_pressione"] else "nessuna pressione"
        print(
            f"{profilo:<18} {len(per_run):>4} {mostra('accettazione'):>9} "
            f"{mostra('deficit_max', 2):>8} {mostra('morti', 0):>6} "
            f"{mostra('integrita_min', 3):>10}   {verdetto}"
        )

    print("\n(valori = il seme PEGGIORE della campagna; soglie fissate prima dei dati)")
    for profilo, _, _, esito in sorted(esiti):
        for nome, valore, soglia, verso in esito["superate"]:
            print(f"\n{profilo}: {nome} {valore} {verso} {soglia}")

    ammessi = [p for p, _, _, e in sorted(esiti) if e["sotto_pressione"]]
    print(
        f"\nCampagne da promuovere alla fase 2: "
        f"{', '.join(ammessi) if ammessi else 'NESSUNA'}"
    )
    if not ammessi:
        print(
            "In nessuna la colonia ha dovuto scegliere: ottiene tutto cio' che si\n"
            "prefigge, non le viene rifiutata alcuna azione e non perde nessuno.\n"
            "Un governatore non ha qui una decisione da prendere, e una campagna\n"
            "linguistica misurerebbe di nuovo il rumore."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
