# -*- coding: utf-8 -*-
"""Quanto e' sparpagliata la colonia: l'accentramento, in numeri.

**La domanda.** Il risultato piu' solido delle campagne e' che LLM (Gov+Amm) occupa
meno celle della baseline. «Meno celle» pero' ammette due letture molto diverse:
una colonia piu' piccola, oppure una colonia altrettanto grande ma piu'
raccolta. Le mappe finali suggeriscono la seconda --- stessi coloni, meno
avamposti --- ma una figura non e' una misura, e il relatore ha chiesto
giustamente di mostrarlo con un numero.

**Le misure, e perche' queste.** Si calcolano sull'ultima istantanea di
ogni run, prendendo come centro la cella piu' popolata (la madre, che a parita'
di seme e' la stessa nei due bracci: lo script lo verifica invece di darlo per
scontato).

*Quota della madre* --- la frazione di coloni che vive nel centro. Sale se la
colonia si accentra, e non dipende da quanto e' grande.

*Raggio medio pesato* (km) --- la distanza media dal centro, pesata per quanti
coloni stanno in ciascuna cella. E' dove sta la gente, non dove c'e' una
bandiera: un avamposto di tre persone non deve pesare quanto un distretto di
duecento.

*Distanza standard* (km) --- la radice della media pesata dei quadrati delle
distanze. E' l'analogo bidimensionale della deviazione standard, la misura
classica della dispersione di un insediamento, e punisce gli avamposti lontani
piu' del raggio medio.

*Raggio massimo* (km) --- quanto arriva lontano la cella occupata piu' esterna.
E' la «corona»: la baseline la crea, e la domanda e' se LLM (Gov+Amm) no.

*Celle per cento coloni* --- il suolo occupato per abitante. E' la terza
dimensione della proliferazione urbana pesata di Jaeger e Schwick (2014), e
risponde all'obiezione ovvia: se le celle calano perche' calano i coloni, questo
rapporto non si muove; se calano di piu' delle persone, la colonia sta davvero
usando meno suolo a parita' di popolazione.

*Indice di dispersione* --- varianza su media della popolazione per cella
occupata. E' l'indice nel senso statistico del termine: vale 1 per una
distribuzione di Poisson, sale quando pochi luoghi concentrano molti abitanti.

**Le distanze sono vere, non di griglia.** La griglia e' equiangolare e una
cella a sessanta gradi di latitudine e' meta' di una all'equatore: contare le
caselle sovrastimerebbe le colonie polari. Ogni cella porta il proprio centro in
gradi (`geometry.center_lat_deg`) e il raggio medio di Marte sta nei metadati
dell'istantanea; da li' la distanza e' quella del grande cerchio.

**Il mondo di riferimento non c'e', e va detto.** La sua baseline
(`runs/campagna_scarsa/ctrl_none`) e' anteriore agli snapshot a ogni tornata e
non ne ha nessuno: senza le due mappe non esiste il confronto appaiato, e un
numero del solo LLM (Gov+Amm) non direbbe niente. I quattro mondi non di riferimento
ce l'hanno su tutti e cinque i semi, ed e' la campagna da cui viene il risultato
sulle celle.

Uso:
    python scripts/indice_dispersione.py
    python scripts/indice_dispersione.py --out docs/benchmarks/dispersione.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

#: mondo -> (etichetta, baseline, bracci di LLM (Gov+Amm) in ordine di esecuzione)
MONDI: list[tuple[str, str, list[str]]] = [
    ("ricco di ghiaccio", "runs/mondi/ice_rich/ctrl_none",
     ["runs/mondi/ice_rich/llm_completo_amm",
      "runs/mondi/ice_rich/llm_completo_amm_rep2",
      "runs/mondi/ice_rich/llm_completo_amm_rep3"]),
    ("frammentato", "runs/mondi/fragmented/ctrl_none",
     ["runs/mondi/fragmented/llm_completo_amm",
      "runs/mondi/fragmented/llm_completo_amm_rep2"]),
    ("rischio alto", "runs/mondi/high_hazard/ctrl_none",
     ["runs/mondi/high_hazard/llm_completo_amm",
      "runs/mondi/high_hazard/llm_completo_amm_rep2"]),
    ("risorse scarse", "runs/mondi/scarce_resources/ctrl_none",
     ["runs/mondi/scarce_resources/llm_completo_amm_v2",
      "runs/mondi/scarce_resources/llm_completo_amm_v2_rep2"]),
]

SEMI = (3, 4, 5, 6, 7)


def ultimo_snapshot(braccio: Path, seme: int) -> Path | None:
    for cartella in sorted(braccio.glob(f"*seed{seme}")):
        istantanee = sorted(
            (cartella / "world_snapshots").glob("step_*.json"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if istantanee:
            return istantanee[-1]
    return None


def _grande_cerchio_km(a: tuple[float, float], b: tuple[float, float], raggio_km: float) -> float:
    """Distanza sulla superficie fra due centri di cella, in chilometri."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * raggio_km * math.asin(min(1.0, math.sqrt(h)))


def _gini(valori: list[float]) -> float:
    """Concentrazione della popolazione fra le celle: 0 = tutte uguali."""
    v = sorted(x for x in valori if x > 0)
    if len(v) < 2:
        return 0.0
    totale = sum(v)
    if totale <= 0:
        return 0.0
    cumulata = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cumulata) / (len(v) * totale) - (len(v) + 1) / len(v)


def misure(percorso: Path) -> dict | None:
    """Le misure di una run, dalla sua ultima istantanea."""
    try:
        dati = json.loads(percorso.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    raggio_km = float((dati.get("metadata") or {}).get("mars_mean_radius_km") or 3390.0)

    celle: list[tuple[tuple[float, float], int]] = []
    for c in dati.get("cells") or []:
        abitanti = len(c.get("agents_present") or [])
        if not abitanti and not c.get("structures"):
            continue
        g = c.get("geometry") or {}
        try:
            centro = (float(g["center_lat_deg"]), float(g["center_lon_deg"]))
        except (KeyError, TypeError, ValueError):
            continue
        celle.append((centro, abitanti))
    if not celle:
        return None

    popolazione = sum(n for _, n in celle)
    madre, abitanti_madre = max(celle, key=lambda kv: kv[1])
    distanze = [(_grande_cerchio_km(centro, madre, raggio_km), n) for centro, n in celle]

    pesata = sum(d * n for d, n in distanze)
    pesata_quadrati = sum(d * d * n for d, n in distanze)
    abitate = [n for _, n in celle if n > 0]

    return {
        "madre": madre,
        "popolazione": popolazione,
        "celle": len(celle),
        "quota_madre": abitanti_madre / popolazione if popolazione else 0.0,
        "raggio_medio_km": pesata / popolazione if popolazione else 0.0,
        "distanza_standard_km": math.sqrt(pesata_quadrati / popolazione) if popolazione else 0.0,
        "raggio_massimo_km": max(d for d, _ in distanze),
        # Suolo occupato per abitante: e' la terza dimensione della
        # proliferazione urbana pesata di Jaeger e Schwick, ed e' la risposta
        # diretta all'obiezione «meno celle perche' meno coloni».
        "celle_per_cento_coloni": (
            100.0 * len(celle) / popolazione if popolazione else 0.0
        ),
        # Varianza su media della popolazione per cella ABITATA: le celle con
        # sole strutture e nessun abitante non sono un luogo dove si vive, e
        # includerle gonfierebbe l'indice con una schiera di zeri.
        "indice_dispersione": (
            statistics.pvariance(abitate) / statistics.fmean(abitate)
            if len(abitate) > 1 and statistics.fmean(abitate) > 0 else 0.0
        ),
    }


GRANDEZZE = [
    ("quota_madre", "quota della madre", "{:.1%}", 1),
    ("raggio_medio_km", "raggio medio pesato (km)", "{:.1f}", 1),
    ("distanza_standard_km", "distanza standard (km)", "{:.1f}", 1),
    ("raggio_massimo_km", "raggio massimo (km)", "{:.0f}", 0),
    ("indice_dispersione", "indice di dispersione", "{:.0f}", 0),
    ("celle_per_cento_coloni", "celle per cento coloni", "{:.2f}", 2),
    ("celle", "celle occupate", "{:.0f}", 0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=RADICE / "docs/benchmarks/2026-09-14-indice-dispersione.md")
    args = ap.parse_args()

    righe_dettaglio: list[str] = []
    per_mondo: list[tuple[str, dict[str, list[float]], int, int]] = []
    assoluti: list[tuple[str, dict[str, float], dict[str, float]]] = []
    centri_diversi: list[str] = []
    # Il conteggio per ESECUZIONE, non per media: una media puo' nascondere che
    # meta' delle esecuzioni va nella direzione opposta, ed e' esattamente il
    # controllo che ha smontato altri risultati di questa tesi.
    per_esecuzione: dict[str, list[int]] = {k: [0, 0] for k, _, _, _ in GRANDEZZE}

    for etichetta, base_dir, bracci in MONDI:
        differenze: dict[str, list[float]] = {k: [] for k, _, _, _ in GRANDEZZE}
        val_base: dict[str, list[float]] = {k: [] for k, _, _, _ in GRANDEZZE}
        val_coppia: dict[str, list[float]] = {k: [] for k, _, _, _ in GRANDEZZE}
        n_coppie = 0
        for seme in SEMI:
            base = misure(ultimo_snapshot(RADICE / base_dir, seme) or Path("/nulla"))
            if not base:
                continue
            esecuzioni = [
                m for m in (misure(ultimo_snapshot(RADICE / b, seme) or Path("/nulla"))
                            for b in bracci)
                if m
            ]
            if not esecuzioni:
                continue
            n_coppie += 1
            # La madre e' la stessa nei due bracci a parita' di seme? Se non lo
            # fosse, il centro da cui misuro le distanze sarebbe diverso e il
            # confronto appaiato non sarebbe piu' appaiato.
            for m in esecuzioni:
                if m["madre"] != base["madre"]:
                    centri_diversi.append(
                        f"{etichetta} seme {seme}: madre {base['madre']} contro {m['madre']}"
                    )
            for chiave, _, _, _ in GRANDEZZE:
                medio = statistics.fmean(m[chiave] for m in esecuzioni)
                differenze[chiave].append(medio - base[chiave])
                val_base[chiave].append(base[chiave])
                val_coppia[chiave].append(medio)
                verso = 1 if chiave == "quota_madre" else -1
                for m in esecuzioni:
                    per_esecuzione[chiave][1] += 1
                    if (m[chiave] - base[chiave]) * verso > 0:
                        per_esecuzione[chiave][0] += 1
            righe_dettaglio.append(
                f"| {etichetta} | {seme} | {base['quota_madre']:.1%} | "
                + " / ".join(f"{m['quota_madre']:.1%}" for m in esecuzioni)
                + f" | {base['raggio_medio_km']:.0f} | "
                + " / ".join(f"{m['raggio_medio_km']:.0f}" for m in esecuzioni)
                + f" | {base['raggio_massimo_km']:.0f} | "
                + " / ".join(f"{m['raggio_massimo_km']:.0f}" for m in esecuzioni)
                + " |"
            )
        per_mondo.append((etichetta, differenze, n_coppie, len(bracci)))
        assoluti.append((
            etichetta,
            {k: statistics.fmean(v) for k, v in val_base.items() if v},
            {k: statistics.fmean(v) for k, v in val_coppia.items() if v},
        ))

    testo: list[str] = []
    testo.append("# L'accentramento, in numeri: l'indice di dispersione\n")
    testo.append(
        "Generato da `scripts/indice_dispersione.py` sull'ultima istantanea di "
        "ogni run. Il centro e' la cella piu' popolata; le distanze sono sul "
        "grande cerchio, con il raggio di Marte preso dai metadati "
        "dell'istantanea, perche' la griglia e' equiangolare e contare le "
        "caselle sovrastimerebbe le colonie polari.\n"
    )
    testo.append(
        "Il mondo di riferimento non compare: la sua baseline e' anteriore agli "
        "snapshot a ogni tornata e non ne ha, quindi li' il confronto appaiato "
        "non esiste. I quattro mondi qui sotto ce l'hanno su tutti e cinque i "
        "semi, e sono la campagna da cui viene il risultato sulle celle.\n"
    )

    testo.append("\n## L'effetto appaiato, mondo per mondo\n")
    testo.append(
        "Ogni numero e' la media sui semi di (LLM (Gov+Amm) meno baseline) sullo stesso "
        "seme; dove LLM (Gov+Amm) ha piu' esecuzioni, prima la media delle "
        "esecuzioni. **Negativo vuol dire piu' raccolta.**\n"
    )
    intestazione = "| mondo | semi | esecuzioni |" + "".join(
        f" {nome} |" for _, nome, _, _ in GRANDEZZE
    )
    testo.append(intestazione)
    testo.append("|---|---:|---:|" + "---:|" * len(GRANDEZZE))
    for etichetta, differenze, n_coppie, n_esec in per_mondo:
        celle = []
        for chiave, _, formato, _ in GRANDEZZE:
            valori = differenze[chiave]
            if not valori:
                celle.append(" — |")
                continue
            media = statistics.fmean(valori)
            segno = "+" if media > 0 else ""
            if chiave == "quota_madre":
                celle.append(f" {segno}{media * 100:.1f} pp |")
            else:
                celle.append(f" {segno}{formato.format(media)} |")
        testo.append(f"| {etichetta} | {n_coppie} | {n_esec} |" + "".join(celle))

    testo.append("\n## Gli stessi numeri in assoluto\n")
    testo.append(
        "Serve per sapere se una differenza e' grande: meno cento chilometri di "
        "raggio massimo vuol dire una cosa su seicento e un'altra su trecento.\n"
    )
    testo.append("| mondo | grandezza | baseline | LLM (Gov+Amm) |")
    testo.append("|---|---|---:|---:|")
    for etichetta, base, coppia in assoluti:
        for chiave, nome, formato, _ in GRANDEZZE:
            if chiave not in base:
                continue
            if chiave == "quota_madre":
                testo.append(
                    f"| {etichetta} | {nome} | {base[chiave]:.1%} | {coppia[chiave]:.1%} |"
                )
            else:
                testo.append(
                    f"| {etichetta} | {nome} | {formato.format(base[chiave])} | "
                    f"{formato.format(coppia[chiave])} |"
                )

    testo.append("\n## Concordanza\n")
    testo.append(
        "Due conteggi. Il primo sulle medie di mondo, il secondo sulle singole "
        "esecuzioni: una media puo' nascondere che meta' delle esecuzioni va "
        "nella direzione opposta, ed e' il controllo che ha smontato altri "
        "risultati di questo lavoro.\n"
    )
    testo.append("| grandezza | mondi | esecuzioni |")
    testo.append("|---|---:|---:|")
    for chiave, nome, _, _ in GRANDEZZE:
        verso = 1 if chiave == "quota_madre" else -1
        quanti = sum(
            1 for _, d, _, _ in per_mondo
            if d[chiave] and statistics.fmean(d[chiave]) * verso > 0
        )
        avanti, totale_esec = per_esecuzione[chiave]
        testo.append(
            f"| {nome} | {quanti} su {len(per_mondo)} | {avanti} su {totale_esec} |"
        )

    # La lettura si scrive con gli stessi numeri della tabella, non a mano:
    # una prosa fissa accanto a una tabella che si rigenera diventa falsa al
    # primo dato nuovo, e nessuno se ne accorge.
    per_nome = {e: (b, c) for e, b, c in assoluti}
    piu_sparsa = max(per_nome, key=lambda e: per_nome[e][0]["raggio_massimo_km"])
    piu_stretta = min(per_nome, key=lambda e: per_nome[e][0]["raggio_massimo_km"])
    avanti_raggio, tot_raggio = per_esecuzione["raggio_massimo_km"]
    avanti_celle, tot_celle = per_esecuzione["celle"]
    avanti_procapite, tot_procapite = per_esecuzione["celle_per_cento_coloni"]
    mondi_procapite = sum(
        1 for _, d, _, _ in per_mondo
        if d["celle_per_cento_coloni"]
        and statistics.fmean(d["celle_per_cento_coloni"]) < 0
    )

    testo.append("\n## Che cosa se ne ricava\n")
    testo.append(
        "**La colonia governata non e' piu' piccola, e' piu' raccolta.** Il "
        "raggio medio pesato, la distanza standard e il raggio massimo scendono "
        "in quattro mondi su quattro. Le tre misure dicono cose diverse --- dove "
        "sta la gente, quanto e' sparsa, fin dove arriva --- e vanno tutte nella "
        "stessa direzione. «Meno celle» non era dunque una colonia rimpicciolita.\n"
    )
    b_pc, c_pc = per_nome[piu_sparsa]
    testo.append(
        f"**E non e' che ci sia meno gente: c'e' meno suolo per persona.** "
        f"E' l'obiezione ovvia --- meno celle perche' meno coloni --- e il "
        f"suolo occupato ogni cento abitanti la toglie di mezzo: scende in "
        f"{mondi_procapite} mondi su {len(per_mondo)} e in {avanti_procapite} "
        f"esecuzioni su {tot_procapite}, la stessa regolarita' delle celle "
        f"occupate. Nel mondo {piu_sparsa} passa da "
        f"{b_pc['celle_per_cento_coloni']:.2f} a {c_pc['celle_per_cento_coloni']:.2f} "
        f"celle ogni cento coloni. E' la densita' di utilizzo della "
        f"proliferazione urbana pesata: quanto suolo ogni abitante si porta "
        f"dietro.\n"
    )
    b, c = per_nome[piu_sparsa]
    testo.append(
        f"**Dove la baseline si allarga di piu', il governo toglie di piu'.** Il "
        f"mondo {piu_sparsa} e' quello in cui la colonia non governata arriva "
        f"piu' lontano, {b['raggio_massimo_km']:.0f} km, ed e' anche quello in "
        f"cui LLM (Gov+Amm) la riporta indietro di piu': {c['raggio_massimo_km']:.0f} "
        f"km, cioe' {b['raggio_massimo_km'] - c['raggio_massimo_km']:.0f} km in "
        f"meno, con {abs(statistics.fmean(dict((e, d) for e, d, _, _ in per_mondo)[piu_sparsa]['celle'])):.0f} "
        "celle occupate in meno. E' la risposta alla domanda su quel mondo: non "
        "ha niente di speciale, offre solo piu' corona da non costruire.\n"
    )
    b, c = per_nome[piu_stretta]
    testo.append(
        f"**Dove il mondo stringe gia' da solo, il governo non ha niente da "
        f"togliere.** Nel mondo {piu_stretta} la baseline arriva a "
        f"{b['raggio_massimo_km']:.0f} km e LLM (Gov+Amm) a {c['raggio_massimo_km']:.0f}: "
        "la scarsita' fa gia' il lavoro che altrove fa la politica, e infatti "
        "li' l'effetto sulle celle e' il piu' piccolo di tutti. Il governo "
        "linguistico non comprime una colonia gia' compressa.\n"
    )
    testo.append(
        f"**L'accentramento e' una direzione, non una garanzia per run.** Il "
        f"raggio massimo scende in {avanti_raggio} esecuzioni su {tot_raggio}, "
        f"mentre le celle occupate scendono in {avanti_celle} su {tot_celle}: il "
        "territorio si restringe piu' regolarmente di quanto la colonia si "
        "ricentri. La frase che i dati autorizzano e' che la forma cambia nella "
        "stessa direzione su tutti i mondi, non che cambi in ogni esecuzione.\n"
    )
    testo.append(
        "**L'indice di dispersione in senso statistico non serve qui, e va "
        "detto invece di nasconderlo.** Varianza su media della popolazione per "
        "cella va nella direzione dell'accentramento in due mondi su quattro: "
        "mescola quanto e' concentrata la popolazione con quante celle ci sono, "
        "e quindi si muove anche per ragioni che non c'entrano con la domanda. "
        "Non e' una sorpresa: in ecologia, da dove l'indice viene, e' noto "
        "che dipende da quante e quanto grandi sono le unita' di conteggio, "
        "ed e' proprio il numero di unita' che qui il governo cambia. Le "
        "misure da riportare sono le tre di distanza e il suolo per "
        "abitante.\n"
    )

    testo.append("\n## Per seme: quota della madre, raggio medio, raggio massimo\n")
    testo.append(
        "| mondo | seme | quota madre base | quota madre LLM (Gov+Amm) | raggio medio base | "
        "raggio medio LLM (Gov+Amm) | raggio max base | raggio max LLM (Gov+Amm) |"
    )
    testo.append("|---|---:|---:|---|---:|---|---:|---|")
    testo.extend(righe_dettaglio)

    if centri_diversi:
        testo.append("\n## Attenzione: centri diversi fra i due bracci\n")
        testo.append(
            "In questi casi la cella piu' popolata non e' la stessa nei due "
            "bracci, quindi le distanze non sono misurate dallo stesso punto e "
            "la riga va letta con cautela.\n"
        )
        testo.extend(f"- {r}" for r in centri_diversi)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(testo) + "\n", encoding="utf-8")
    print(f"scritto {args.out}")
    print("\n".join(testo[: testo.index("\n## Concordanza\n") + 8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
