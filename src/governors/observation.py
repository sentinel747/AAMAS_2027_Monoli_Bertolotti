from __future__ import annotations

"""Il quadro che il governatore osserva: aggregati, non celle.

**Il cambio di scala e' il punto del ridisegno (2026-08-24).** La direttiva
per-cella aveva bisogno dell'elenco delle celle indirizzabili; una policy
globale no — il governatore scrive condizioni, non coordinate, quindi cio' che
gli serve e' la **distribuzione** degli indicatori su cui quelle condizioni
sono scritte: media, deviazione standard, minimo e massimo per ciascun
indicatore del vocabolario, sulle celle dove qualcuno decide. Stile governo di
un paese: vede le statistiche, non i cittadini.

Tre famiglie di numeri, ed e' tutto il quadro:

- le **metriche di colonia** gia' calcolate dalla shell (invariate);
- gli **aggregati degli indicatori di cella** — gli stessi nomi su cui la
  policy puo' scrivere condizioni, cosi' il governatore vede esattamente la
  grandezza che sta per soglia-re;
- le **statistiche di popolazione** (salute, idratazione, sazieta', morale,
  fatica, stress medi) e i **totali delle strutture**, perche' una politica
  ragiona anche su come sta la gente e su cosa esiste gia'.

**Il governatore vede il passo precedente.** Le metriche di colonia si
calcolano nella shell dopo il passo, quindi al tick del passo N il quadro e'
quello di N-1. E' inevitabile ed e' anche fedele: un governo agisce su
rapporti.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from src.core import constants as C
from src.governors.apply import indicator_values
from src.governors.policy import INDICATORS


@dataclass(frozen=True)
class ColonyPicture:
    """Cio' che il governatore riceve."""

    step: int
    population: int
    #: Le metriche di colonia, gia' filtrate ai soli numeri.
    metrics: dict[str, float] = field(default_factory=dict)
    #: Per ogni indicatore del vocabolario: mean, std, min, max sulle celle
    #: occupate. Le chiavi esterne sono i nomi di `policy.INDICATORS`.
    indicators: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Medie sugli agenti vivi: health, hydration, satiety, morale, fatigue,
    #: stress.
    population_stats: dict[str, float] = field(default_factory=dict)
    #: Totali delle strutture dell'intera colonia, per tipo, solo quelle > 0.
    structures: dict[str, int] = field(default_factory=dict)
    #: Su quante celle occupate gli aggregati sono calcolati.
    n_cells: int = 0
    #: Quante (cella, passo) ciascuna regola della politica IN VIGORE ha
    #: catturato dall'ultimo tick, in ordine di scrittura, piu' l'*altrimenti*
    #: in coda. E' il consuntivo della propria legge: senza, una regola larga
    #: scritta per prima affama le successive e chi l'ha scritta non puo'
    #: saperlo. Vuoto al primo tick, quando nessuna politica e' ancora stata
    #: applicata.
    rule_hits: tuple[tuple[str, int], ...] = ()
    #: Gli stessi indicatori, con media e deviazione PESATE per colono: ogni
    #: cella pesa quanti coloni ospita. **Perche' accanto e non al posto
    #: (2026-09-05).** A fine run il 77-84 per cento dei coloni vive nella
    #: cella madre; nella media per cella quella cella conta quanto un
    #: avamposto di tre, e il governatore scriveva leggi per la cella mediana,
    #: che e' un avamposto. Le condizioni della politica restano valutate per
    #: cella, quindi la vista per cella serve ancora: dice dove scattano.
    indicators_weighted: dict[str, dict[str, float]] = field(default_factory=dict)
    #: La cella piu' popolata: coordinate, occupanti, quota della colonia e i
    #: suoi indicatori. E' il Nord con un milione di abitanti: una regola che
    #: non scatta li' non tocca la maggioranza dei coloni, e va detto.
    principale: dict = field(default_factory=dict)
    #: Decessi dall'ultimo tick, (y, x) -> {causa: quanti}. L'unica voce di
    #: ESITO: gli amministratori la ricevono dal 02/09, il governatore no, e
    #: senza di essa decide su giacenze e medie mentre alla frontiera si muore.
    deaths: dict = field(default_factory=dict)


def _aggregate_weighted(values: list[float], weights: list[float]) -> dict[str, float]:
    """Media e deviazione pesate; minimo e massimo restano quelli per cella."""
    array = np.asarray(values, dtype=np.float64)
    pesi = np.asarray(weights, dtype=np.float64)
    totale = float(pesi.sum())
    if totale <= 0:
        return _aggregate(values)
    media = float((array * pesi).sum() / totale)
    varianza = float((pesi * (array - media) ** 2).sum() / totale)
    return {
        "mean": round(media, 3),
        "std": round(varianza ** 0.5, 3),
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def _cell_indicator_rows(cells, positions: list[tuple[int, int, int]]) -> dict[str, list[float]]:
    """I valori degli indicatori, cella occupata per cella occupata.

    Il conteggio degli occupanti arriva dalle POSIZIONI degli agenti e non da
    `cells.occupancy`: il quadro si costruisce nella shell, prima del passo,
    e la colonna degli occupanti del kernel puo' essere di un passo indietro
    rispetto alle righe vive che la shell sta guardando. Sono le stesse
    posizioni con cui il vecchio quadro elencava le celle.
    """
    rows: dict[str, list[float]] = {name: [] for name in INDICATORS}
    copertura = indicator_values("power_coverage", cells)
    manutenzione = indicator_values("structure_integrity", cells)
    for y, x, occupants in positions:
        per_head = float(max(1, occupants))
        rows["occupants"].append(float(occupants))
        rows["food_per_occupant"].append(
            float(cells.cell_res[y, x, C.R["food"]]) / per_head
        )
        rows["ice_per_occupant"].append(
            (float(cells.cell_res[y, x, C.R["ice"]]) + float(cells.water_ice[y, x]))
            / per_head
        )
        rows["material_per_occupant"].append(
            float(cells.cell_res[y, x, C.R["construction_material"]]) / per_head
        )
        rows["minerals_per_occupant"].append(
            float(cells.cell_res[y, x, C.R["minerals"]]) / per_head
        )
        rows["water_per_occupant"].append(
            float(cells.cell_res[y, x, C.R["water"]]) / per_head
        )
        rows["oxygen_per_occupant"].append(
            float(cells.cell_res[y, x, C.R["oxygen"]]) / per_head
        )
        # **Copertura elettrica e manutenzione arrivano dalla stessa funzione
        # che valuta le condizioni** (`apply.indicator_values`), non da un
        # conto riscritto qui: il quadro deve mostrare esattamente la grandezza
        # che la regola soglia-ra', altrimenti il governatore taglia su un
        # numero e ne osserva un altro. E' la stessa ragione per cui i nomi
        # sono quelli di `INDICATORS`.
        rows["power_coverage"].append(float(copertura[y, x]))
        rows["structure_integrity"].append(float(manutenzione[y, x]))
    return rows


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "std": round(float(array.std()), 3),
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def build_picture(
    step, metrics, cells, agents, rows, rule_hits=(), morti_per_cella=None
) -> ColonyPicture:
    """Compone il quadro dalle colonne e dalle metriche gia' calcolate.

    `morti_per_cella` mappa (y, x) -> {causa: quanti} per i soli decessi
    dall'ultimo tick; lo conta la shell una volta per tick e lo da' sia al
    governatore sia agli amministratori, cosi' i due livelli leggono lo stesso
    esito.
    """
    rows = np.asarray(rows, dtype=np.int64)
    seen: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(agents.y[int(row)]), int(agents.x[int(row)]))
        seen[key] = seen.get(key, 0) + 1
    positions = [(y, x, count) for (y, x), count in sorted(seen.items())]

    indicators: dict[str, dict[str, float]] = {}
    indicators_weighted: dict[str, dict[str, float]] = {}
    principale: dict = {}
    if positions:
        righe = _cell_indicator_rows(cells, positions)
        pesi = [float(count) for _, _, count in positions]
        for name, values in righe.items():
            indicators[name] = _aggregate(values)
            indicators_weighted[name] = _aggregate_weighted(values, pesi)
        indice = max(range(len(positions)), key=lambda i: positions[i][2])
        y, x, occupanti = positions[indice]
        principale = {
            "cell": [int(y), int(x)],
            "occupants": int(occupanti),
            "share": round(occupanti / max(1, int(rows.size)), 3),
            "indicators": {name: round(float(values[indice]), 3) for name, values in righe.items()},
        }

    population_stats: dict[str, float] = {}
    if rows.size:
        for column in ("health", "hydration", "satiety", "morale", "fatigue", "stress"):
            values = getattr(agents, column, None)
            if values is None:
                continue
            population_stats[f"{column}_mean"] = round(
                float(np.asarray(values)[rows].mean()), 3
            )

    totals = cells.struct_count.sum(axis=(0, 1))
    structures = {
        C.STRUCTURES[index].value: int(totals[index])
        for index in range(C.NS)
        if int(totals[index]) > 0
    }

    return ColonyPicture(
        step=int(step),
        population=int(rows.size),
        metrics={str(k): float(v) for k, v in (metrics or {}).items()},
        indicators=indicators,
        population_stats=population_stats,
        structures=structures,
        n_cells=len(positions),
        rule_hits=tuple(rule_hits or ()),
        indicators_weighted=indicators_weighted,
        principale=principale,
        deaths={
            (int(c[0]), int(c[1])): {str(k): int(v) for k, v in dict(cause).items()}
            for c, cause in (morti_per_cella or {}).items()
        },
    )


def picture_digest(picture: ColonyPicture) -> str:
    """Impronta stabile del quadro, per il registro.

    Le chiavi vengono ordinate: due quadri uguali devono dare lo stesso digest
    anche se costruiti in ordine diverso.
    """
    payload = {
        "step": picture.step,
        "population": picture.population,
        "metrics": dict(sorted(picture.metrics.items())),
        "indicators": {
            name: dict(sorted(stats.items()))
            for name, stats in sorted(picture.indicators.items())
        },
        "population_stats": dict(sorted(picture.population_stats.items())),
        "structures": dict(sorted(picture.structures.items())),
        "n_cells": picture.n_cells,
        # Entra nel digest perche' entra nel prompt: due quadri che portano
        # consuntivi diversi non sono lo stesso quadro, e un replay che li
        # confondesse verificherebbe meno di quanto dichiara.
        "rule_hits": [list(voce) for voce in picture.rule_hits],
        # I morti entrano per la stessa ragione del consuntivo: stanno nel
        # prompt. I pesati e la cella principale derivano dagli stessi dati
        # gia' impronta-ti e non aggiungono informazione.
        "deaths": {f"{c[0]},{c[1]}": dict(sorted(v.items())) for c, v in sorted(picture.deaths.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()
