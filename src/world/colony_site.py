from __future__ import annotations

import math
from collections.abc import Iterable

from .terrain import traversal_risk_for_cell


# Above this many settlers the initial colony footprint spills onto the nearest
# neighbouring cells (one grid cell is ~59x59 km at the equator, so a single
# cell comfortably hosts any realistic early landing party).
INITIAL_AGENTS_PER_CELL = 1000


def colony_site_score(cell) -> float:
    """Score a candidate colony cell from local resources, safety, and Mars geometry."""
    resources = cell.resources
    water_score = _clamp01((cell.water_ice + resources.ice + cell.liquid_water * 2.0) / 12.0)
    material_score = _clamp01((resources.construction_material + resources.minerals * 0.45) / 10.0)
    energy_score = _clamp01((resources.energy + max(0.0, 1.0 - abs(_lat(cell)) / 75.0)) / 2.0)
    safety_score = _clamp01(1.0 - traversal_risk_for_cell(cell))
    habitability_score = _clamp01(cell.habitability_score * 2.5)
    area_score = _clamp01(float(cell.geometry.get("area_km2", 0.0)) / 3200.0)

    return _clamp01(
        water_score * 0.26
        + material_score * 0.22
        + energy_score * 0.16
        + safety_score * 0.18
        + habitability_score * 0.12
        + area_score * 0.06
    )


#: Semiampiezza della fascia equatoriale entro cui si sceglie il sito, in righe
#: di griglia. E' il vincolo di missione, non un gusto: fuori di qui la
#: severita' polare abbassa la temperatura della cella e il flusso solare, e il
#: sito diventa una condizione diversa invece che una scelta. Restando dentro
#: la fascia, il punteggio confronta celle comparabili.
FASCIA_EQUATORIALE_RIGHE = 10


def find_safe_start(world) -> tuple[int, int]:
    """La cella con il punteggio piu' alto dentro la fascia equatoriale.

    **Perche' non e' piu' «la prima piana di regolito» (2026-08-31).** La
    regola precedente scandiva la fascia dall'alto e da sinistra e si fermava
    al primo terreno di tipo `regolith_plain`. Misurato su sei semi e sei
    profili di mappa, trentasei combinazioni: **y valeva 80 in tutte e trenta**
    e x valeva 0 in ventinove. Non era sfortuna, erano tre difetti sommati:

    - la scansione partiva da `centro - 10` e procedeva verso il basso, quindi
      restituiva sempre la PRIMA riga della fascia e mai il centro, malgrado il
      commento dicesse «outward from the equatorial band»;
    - il regolito copre circa meta' di ogni riga, quindi `x = 0` era gia' un
      match valido nella maggioranza dei semi e il ciclo usciva subito;
    - il criterio era booleano. `colony_site_score` esisteva, pesava acqua,
      materiali, energia, sicurezza, abitabilita' e area, ed era sensibile al
      seme, ma **nessuno la chiamava** per scegliere: serviva solo a pubblicare
      una metrica che misurava quanto il sito scelto fosse peggiore del
      migliore.

    Conseguenza pratica: ogni run partiva dalla stessa cella al bordo sinistro
    della mappa, con **zero minerali** su tutti i semi provati, e il sito, che
    la tesi dichiara controllo sperimentale esplicito, era una costante.

    Ora il sito e' il massimo di `colony_site_score` dentro la fascia. Nessuna
    lista di terreni da escludere: il punteggio contiene gia' il termine di
    sicurezza, e misurato evita da se' tubi di lava e canyon. La scelta e'
    deterministica, il primo massimo in ordine di scansione vince, e cambia con
    il seme: misurata su sei semi da' sei celle diverse, con 5,6-7,0 unita' di
    minerale contro le zero di prima.

    Condivisa dai due motori, cosi' il sito coincide fra esperimenti headless e
    run dalla GUI.
    """
    center_x, center_y = world.width // 2, world.height // 2
    prima_riga = max(0, center_y - FASCIA_EQUATORIALE_RIGHE)
    ultima_riga = min(world.height, center_y + FASCIA_EQUATORIALE_RIGHE + 1)
    migliore: tuple[int, int] | None = None
    punteggio_migliore = -1.0
    for y in range(prima_riga, ultima_riga):
        for x in range(world.width):
            punteggio = colony_site_score(world.get_cell(x, y))
            if punteggio > punteggio_migliore:
                punteggio_migliore = punteggio
                migliore = (x, y)
    if migliore is None:
        return center_x, center_y
    return migliore


def colony_spawn_cells(
    world,
    center_x: int,
    center_y: int,
    total_agents: int,
    max_per_cell: int = INITIAL_AGENTS_PER_CELL,
) -> list[tuple[int, int]]:
    """Cells the colony occupies at start: one cell up to max_per_cell settlers,
    then progressively more nearest-neighbour cells for larger landings.

    Shared by initial agent spawn and initial structure seeding so people and
    buildings always start on the same footprint in both engines.
    """
    center_x = min(world.width - 1, max(0, int(center_x)))
    center_y = min(world.height - 1, max(0, int(center_y)))
    needed = max(1, math.ceil(max(1, int(total_agents)) / max(1, int(max_per_cell))))
    needed = min(needed, world.width * world.height)
    cells: list[tuple[int, int]] = [(center_x, center_y)]
    radius = 1
    while len(cells) < needed and radius < max(world.width, world.height):
        ring = [
            (xx, yy)
            for yy in range(center_y - radius, center_y + radius + 1)
            for xx in range(center_x - radius, center_x + radius + 1)
            if max(abs(xx - center_x), abs(yy - center_y)) == radius
            and 0 <= xx < world.width
            and 0 <= yy < world.height
        ]
        ring.sort(key=lambda c: ((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2, c[1], c[0]))
        cells.extend(ring)
        radius += 1
    return cells[:needed]


def record_colony_site_metadata(world, x: int, y: int) -> None:
    """Point planetary climate sampling at the colony site coordinates instead
    of the default Gale-crater fallback."""
    geometry = world.get_cell(x, y).geometry
    latitude = geometry.get("center_lat_deg")
    longitude = geometry.get("center_lon_deg")
    if latitude is not None:
        world.metadata["center_latitude_deg"] = float(latitude)
    if longitude is not None:
        world.metadata["center_longitude_deg"] = float(longitude)


def best_colony_site(cells: Iterable) -> tuple[object | None, float]:
    best_cell = None
    best_score = -1.0
    for cell in cells:
        score = colony_site_score(cell)
        if score > best_score:
            best_cell = cell
            best_score = score
    return best_cell, max(0.0, best_score)


def colony_site_average(cells: Iterable) -> float:
    scores = [colony_site_score(cell) for cell in cells]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _lat(cell) -> float:
    return float(cell.geometry.get("center_lat_deg", 0.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
