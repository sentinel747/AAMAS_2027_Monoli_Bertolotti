from __future__ import annotations

from dataclasses import dataclass
import random

from .cell import Cell
from .grid import GridWorld
from .mars_geometry import cell_geometry, grid_metadata
from .resources import ResourceBundle
from .structures import (
    COSTO_ACQUA_AVVIAMENTO,
    COSTO_MATERIALE_AVVIAMENTO,
    DOMANDA_MINERALI_SU_MATERIALE,
    STRUTTURE_DI_AVVIAMENTO,
)
from .terrain import TerrainType


MAP_PROFILES = ("balanced", "scarce_resources", "high_hazard", "ice_rich", "mineral_rich", "fragmented", "random")


#: **Il fondo regolitico: Marte non e' roccia morta (2026-09-01).**
#:
#: Fino a oggi le uniche celle con qualcosa da estrarre erano quelle vicine ai
#: ventiquattro centri minerari, e su una mappa 360x180 volevano dire il **6,4%
#: del pianeta**: mediana e nonantesimo percentile entrambi a 0,00. Nella fascia
#: equatoriale, dove le colonie vivono davvero, il **92-94% delle celle era
#: completamente vuoto** -- ne' minerali, ne' materiale, ne' ghiaccio. Un
#: avamposto fondato fuori da una vena non poteva raccogliere nulla e non
#: poteva costruire nulla, e la colonia madre esauriva il proprio giacimento
#: entro il passo 30 (voce C13 della checklist).
#:
#: E' anche in contraddizione con cio' che il modello gia' dichiara altrove: il
#: blocco ISRU di `kernel_biology` dice, testualmente, che «il regolito non e'
#: un giacimento: e' la superficie di Marte, presente in ogni cella». Ma quella
#: superficie era lavorabile SOLO attraverso un impianto, mentre la raccolta a
#: mano leggeva la giacenza -- che era zero quasi ovunque. Il pianeta offriva
#: per capitale cio' che negava al lavoro.
#:
#: **Il fondo non e' scelto: e' quanto serve ad avviare un avamposto.** Le
#: quattro strutture che sbloccano la capienza vitale costano dodici unita' di
#: materiale (`COSTO_MATERIALE_AVVIAMENTO`, letto da `BUILD_COSTS`), e la cella
#: MEDIANA di Marte ne contiene esattamente altrettante. Meta' delle celle ne ha
#: di piu', meta' di meno: il fattore casuale e' simmetrico intorno a uno, e
#: dipende dal seme della mappa come ogni altro tratto del pianeta.
#:
#: **La miscela e' quella che la colonia consuma.** Prima il materiale era
#: `minerali x 0,35`, cioe' il pianeta offriva 2,86 minerali per ogni unita' di
#: materiale mentre il catalogo delle costruzioni ne chiede 0,58: cinque volte
#: piu' minerale di quanto serva, e materiale col contagocce. Ora vale la stessa
#: ripartizione che l'ISRU applica alla propria resa,
#: `DOMANDA_MINERALI_SU_MATERIALE`, che E' `BUILD_COSTS` letto una volta sola.
#: Nessuno dei due beni puo' strozzare l'altro per un disallineamento fra due
#: tabelle.
#:
#: **La geografia non sparisce, cambia mestiere.** I centri minerari restano e
#: si sommano al fondo: una vena vale ancora molte volte il fondo, quindi decide
#: la VELOCITA' della crescita. Il fondo decide soltanto che un avamposto,
#: ovunque lo si fondi, possa esistere.
#:
#: Le calotte polari sono ghiaccio, non regolito: il fondo si spegne in
#: proporzione a `polar_severity`, che il generatore calcola gia'.
REGOLITE_LAVORABILE = COSTO_MATERIALE_AVVIAMENTO

#: **L'acqua sta sotto, e sta sotto ogni cella (2026-09-01).**
#:
#: Il ghiaccio era solo dove il modello lo metteva in superficie: le calotte
#: polari e le sacche sepolte di `_buried_ice`, che alle latitudini basse hanno
#: una probabilita' piccola e sotto il 5% di latitudine sono ZERO per
#: costruzione. Misurato sulla fascia equatoriale, dove le colonie vivono: solo
#: il **3,8-4,5% delle celle** aveva una qualunque forma d'acqua.
#:
#: E' la causa che la voce C9 della checklist aveva isolato senza chiuderla: si
#: muore in avamposti **giovani e senz'acqua**, e quattordici decessi su
#: diciannove avvenivano con il magazzino idrico sotto 0,1 mentre cibo e
#: ossigeno erano abbondanti. Senza tetto di popolazione il fenomeno diventa la
#: prima causa di morte della colonia: misurato a 400 passi, il 40% delle azioni
#: finali sono `refill_water` e `drink_water`, e i morti passano da 16 a 115.
#:
#: **Fisicamente e' la lettura giusta.** L'acqua marziana non sta in pozze: sta
#: sotto, come ghiaccio e come acqua legata nei minerali idrati — il rover
#: Curiosity ne ha misurata circa il 2% in peso nel regolito del cratere Gale,
#: che e' equatoriale. Il ghiaccio massiccio delle medie latitudini resta cio'
#: che era, una CONCENTRAZIONE: `_buried_ice` non cambia di una riga.
#:
#: **La quantita' non e' scelta**: e' l'acqua che serve a una squadra di
#: fondazione per avviare l'avamposto, cioe' il costo idrico dell'avviamento una
#: volta per fondatore. Vale 4,0, che e' anche — e non a caso — la base che
#: `_polar_ice` assegna al margine della calotta: il pianeta resta coerente con
#: se stesso.
#:
#: Resta lavoro estrarla: `COLLECT_ICE` e `REFILL_WATER` chiedono entrambi che
#: la cella abbia ghiaccio, e una scorta finita si esaurisce come un giacimento.
#: Chi vuole un flusso invece di una scorta deve costruire cio' che lo produce.
ACQUA_LAVORABILE = COSTO_ACQUA_AVVIAMENTO * len(STRUTTURE_DI_AVVIAMENTO)


@dataclass(frozen=True)
class MapProfileSettings:
    mineral_centers_factor: float = 1.0
    mineral_scale: float = 8.0
    crater_centers_factor: float = 1.0
    dust_chance: float = 0.08
    canyon_chance: float = 0.04
    lava_tube_chance: float = 0.03
    polar_ice_multiplier: float = 1.0
    radiation_mean: float = 1.0
    radiation_sigma: float = 0.15
    dust_multiplier: float = 1.0
    fragmented_patch_count: int = 0
    fragmented_patch_radius_factor: float = 0.0
    solid_cap_factor: float = 0.08
    transition_factor: float = 0.06


PROFILE_SETTINGS: dict[str, MapProfileSettings] = {
    "balanced": MapProfileSettings(),
    "scarce_resources": MapProfileSettings(mineral_centers_factor=0.55, mineral_scale=4.6, crater_centers_factor=1.25, dust_chance=0.12, dust_multiplier=1.15, solid_cap_factor=0.065, transition_factor=0.045),
    "high_hazard": MapProfileSettings(crater_centers_factor=1.7, dust_chance=0.16, canyon_chance=0.08, lava_tube_chance=0.05, radiation_mean=1.2, radiation_sigma=0.22, dust_multiplier=1.35),
    "ice_rich": MapProfileSettings(mineral_centers_factor=0.9, polar_ice_multiplier=1.65, dust_chance=0.06, solid_cap_factor=0.13, transition_factor=0.10),
    "mineral_rich": MapProfileSettings(mineral_centers_factor=1.8, mineral_scale=12.0, crater_centers_factor=0.9, dust_chance=0.06),
    "fragmented": MapProfileSettings(mineral_centers_factor=1.15, crater_centers_factor=1.45, dust_chance=0.11, canyon_chance=0.08, lava_tube_chance=0.05, fragmented_patch_count=9, fragmented_patch_radius_factor=0.09),
}


class WorldGenerator:
    def __init__(self, seed: int = 0, map_profile: str = "balanced"):
        if map_profile not in MAP_PROFILES:
            raise ValueError(f"unknown map_profile: {map_profile}")
        self.seed = seed
        self.map_profile = self._resolve_profile(seed, map_profile)
        self.requested_map_profile = map_profile
        self.settings = PROFILE_SETTINGS[self.map_profile]
        self.random = random.Random(seed)

    def generate(self, width: int = 50, height: int = 50) -> GridWorld:
        mineral_count = max(1, round(max(3, width // 15) * self.settings.mineral_centers_factor))
        crater_count = max(1, round(max(4, width // 12) * self.settings.crater_centers_factor))
        mineral_centers = [(self.random.randrange(width), self.random.randrange(height)) for _ in range(mineral_count)]
        crater_centers = [(self.random.randrange(width), self.random.randrange(height)) for _ in range(crater_count)]
        fragmented_centers = [
            (self.random.randrange(width), self.random.randrange(height))
            for _ in range(self.settings.fragmented_patch_count)
        ]
        cells: list[list[Cell]] = []
        for y in range(height):
            row: list[Cell] = []
            for x in range(width):
                terrain = self._terrain_at(x, y, width, height, mineral_centers, crater_centers, fragmented_centers)
                elevation = self.random.gauss(0, 1)
                radiation = max(0.4, min(1.8, self.random.gauss(self.settings.radiation_mean, self.settings.radiation_sigma)))
                dust = max(0.0, min(1.0, self.random.random() * 0.7 * self.settings.dust_multiplier))
                polar = self._polar_fraction(y, height)
                # **Le dotazioni nuove pescano da un flusso PROPRIO
                # (2026-09-01).** Sarebbe bastato chiamare `self.random` altre
                # due volte per cella, ma quel generatore e' condiviso con
                # terreno, elevazione, radiazione e polvere: due estrazioni in
                # piu' spostano l'intera sequenza, e lo stesso seme produce un
                # pianeta diverso. Misurato sul seme 44 di una prova: il rischio
                # di attraversamento del vicinato passava da 0,25 a 1,00 perche'
                # la cella era finita su un altro terreno, e una prova che
                # parlava di ricognizioni ha smesso di parlarne.
                #
                # Il precedente e' nel modulo stesso: `_buried_ice` ha gia' un
                # flusso locale per cella. Qui se ne usa uno derivato dalle
                # stesse coordinate, cosi' la dotazione e' riproducibile e
                # indipendente dall'ordine in cui la si calcola.
                dado = self._flusso_di_cella(x, y)
                # Fondo idrico ovunque piu' le concentrazioni gia' esistenti:
                # vedi `ACQUA_LAVORABILE`. La calotta e le sacche sepolte non
                # cambiano, si sommano.
                water_ice = self._polar_ice(x, y, width, height, polar)
                if polar <= 0.0:
                    water_ice += self._buried_ice(x, y, width, height, terrain)
                water_ice = round(
                    water_ice + ACQUA_LAVORABILE * dado.uniform(0.5, 1.5), 3
                )
                # Il fondo regolitico piu' la concentrazione della vena: vedi
                # `REGOLITE_LAVORABILE`. Il fattore e' simmetrico intorno a uno,
                # cosi' la cella mediana vale esattamente il fondo.
                fondo = (
                    REGOLITE_LAVORABILE
                    * (1.0 - polar)
                    * dado.uniform(0.5, 1.5)
                    * self._forza_del_profilo()
                )
                materiale = round(
                    fondo
                    + self._resource_from_centers(
                        x, y, mineral_centers, self.settings.mineral_scale
                    ),
                    3,
                )
                minerals = round(materiale * DOMANDA_MINERALI_SU_MATERIALE, 3)
                cell = Cell(
                    x=x,
                    y=y,
                    terrain=terrain,
                    elevation=elevation,
                    local_temperature_modifier=self.random.gauss(-1.0, 2.5) - polar * 28.0,
                    polar_severity=polar,
                    radiation_level=radiation,
                    dust_level=dust,
                    water_ice=water_ice,
                    resources=ResourceBundle(ice=water_ice * 0.5, minerals=minerals, construction_material=materiale),
                )
                geom = cell_geometry(x, y, width, height)
                cell.geometry = {
                    "center_lat_deg": geom.center_lat_deg,
                    "center_lon_deg": geom.center_lon_deg,
                    "width_m": geom.width_m,
                    "height_m": geom.height_m,
                    "area_m2": geom.area_m2,
                    "area_km2": geom.area_km2,
                }
                cell.baseline_radiation_level = radiation
                cell.baseline_dust_level = dust
                cell.baseline_temperature_modifier = cell.local_temperature_modifier
                cell.recompute_habitability()
                row.append(cell)
            cells.append(row)
        world = GridWorld(width=width, height=height, cells=cells)
        world.metadata.update(grid_metadata(width, height))
        world.metadata.update({"map_profile": self.map_profile, "requested_map_profile": self.requested_map_profile, "seed": self.seed})
        return world

    def _terrain_at(self, x: int, y: int, width: int, height: int, mineral_centers, crater_centers, fragmented_centers) -> TerrainType:
        polar = self._polar_fraction(y, height)
        if polar >= 0.74:
            return TerrainType.ICE_DEPOSIT
        if polar >= 0.28:
            return TerrainType.FROZEN_BASIN if self.random.random() < 0.72 else TerrainType.ICE_DEPOSIT
        if fragmented_centers and self._near_any(x, y, fragmented_centers, min(width, height) * self.settings.fragmented_patch_radius_factor):
            return self.random.choice([TerrainType.CRATER, TerrainType.CANYON, TerrainType.MOUNTAIN, TerrainType.DUST_FIELD])
        if self._near_any(x, y, mineral_centers, min(width, height) * 0.08):
            return self.random.choice([TerrainType.MINERAL_RICH, TerrainType.MOUNTAIN])
        if self._near_any(x, y, crater_centers, min(width, height) * 0.07):
            return TerrainType.CRATER
        roll = self.random.random()
        if roll < self.settings.dust_chance:
            return TerrainType.DUST_FIELD
        if roll < self.settings.dust_chance + self.settings.canyon_chance:
            return TerrainType.CANYON
        if roll < self.settings.dust_chance + self.settings.canyon_chance + self.settings.lava_tube_chance:
            return TerrainType.LAVA_TUBE
        return TerrainType.REGOLITH_PLAIN

    def _near_any(self, x: int, y: int, centers, radius: float) -> bool:
        return any((x - cx) ** 2 + (y - cy) ** 2 <= radius**2 for cx, cy in centers)

    def _flusso_di_cella(self, x: int, y: int) -> random.Random:
        """Un generatore per cella, indipendente dalla sequenza principale.

        Serve alle dotazioni aggiunte dopo che il generatore esisteva gia': se
        pescassero da `self.random`, ogni estrazione in piu' sposterebbe tutto
        cio' che viene dopo e lo stesso seme darebbe un altro pianeta. Le
        costanti sono le stesse di `_buried_ice`, che questo flusso affianca.
        """
        return random.Random((self.seed + 2029) * 1_000_003 + x * 9176 + y * 131)

    def _forza_del_profilo(self) -> float:
        """Quanto il profilo di mappa moltiplica la dotazione minerale totale.

        **Il fondo deve seguire il profilo, e con la STESSA forza delle vene
        (2026-09-01).** Applicando al fondo il solo `mineral_centers_factor`
        (0,55 per `scarce_resources`) la leva della scarsita' si sarebbe
        smussata da un ordine di grandezza a meno della meta': lo scenario
        avrebbe cambiato nome senza mordere, che e' esattamente l'errore che i
        profili servono a evitare.

        Il fattore non e' scelto, e' geometria del generatore. Il valore che
        `_resource_from_centers` somma su un centro e' `max(0, scala - dist)`,
        il cui integrale sul piano vale `pi x scala^3 / 3`: il totale di una
        mappa scala quindi come **numero di centri per il cubo del raggio**. La
        stessa quantita' moltiplica il fondo, cosi' fondo e vene si muovono
        insieme e il profilo conserva la forza che dichiara.

        Per `scarce_resources` vale 0,55 x (4,6/8,0)^3 = 0,105, cioe' il decimo
        promesso dalla sua stessa docstring; per `mineral_rich` 1,8 x
        (12,0/8,0)^3 = 6,08.
        """
        riferimento = MapProfileSettings().mineral_scale
        return self.settings.mineral_centers_factor * (
            self.settings.mineral_scale / riferimento
        ) ** 3

    def _resource_from_centers(self, x: int, y: int, centers, scale: float) -> float:
        value = 0.0
        for cx, cy in centers:
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            value += max(0.0, scale - dist2**0.5)
        return round(value * self.random.uniform(0.3, 1.0), 3)

    def _polar_fraction(self, y: int, height: int) -> float:
        if height <= 1:
            return 0.0
        polar_distance = min(y + 0.5, height - y - 0.5)
        solid_cap_rows = max(2.0, height * self.settings.solid_cap_factor)
        transition_rows = max(2.0, height * self.settings.transition_factor)
        if polar_distance <= solid_cap_rows:
            return 1.0
        if polar_distance <= solid_cap_rows + transition_rows:
            return 1.0 - (polar_distance - solid_cap_rows) / transition_rows
        return 0.0

    def _polar_ice(self, x: int, y: int, width: int, height: int, polar: float | None = None) -> float:
        polar = self._polar_fraction(y, height) if polar is None else polar
        if polar <= 0:
            return 0.0
        longitudinal_texture = 0.82 + 0.18 * self.random.random()
        return round((4.0 + 10.0 * polar) * longitudinal_texture * self.settings.polar_ice_multiplier, 3)

    def _buried_ice(self, x: int, y: int, width: int, height: int, terrain: TerrainType) -> float:
        local_rng = random.Random((self.seed + 1009) * 1_000_003 + x * 9176 + y * 131 + sum(ord(ch) for ch in terrain.value))
        latitude = abs(((y + 0.5) / max(1, height)) - 0.5) * 2.0
        if latitude < 0.05:
            return 0.0

        terrain_chance = {
            TerrainType.CRATER: 0.28,
            TerrainType.CANYON: 0.22,
            TerrainType.MOUNTAIN: 0.16,
            TerrainType.MINERAL_RICH: 0.12,
            TerrainType.REGOLITH_PLAIN: 0.08,
            TerrainType.DUST_FIELD: 0.04,
        }.get(terrain, 0.0)
        chance = terrain_chance * min(1.0, 0.45 + latitude)
        if local_rng.random() >= chance:
            return 0.0

        depth_penalty = 0.55 + latitude * 0.65
        geology_bonus = 1.4 if terrain in {TerrainType.CRATER, TerrainType.CANYON} else 1.0
        return round(local_rng.uniform(0.80, 2.80) * depth_penalty * geology_bonus, 3)

    def _resolve_profile(self, seed: int, map_profile: str) -> str:
        if map_profile != "random":
            return map_profile
        return random.Random(seed).choice([profile for profile in MAP_PROFILES if profile != "random"])
