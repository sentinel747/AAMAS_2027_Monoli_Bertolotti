from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .resources import ResourceBundle


class StructureType(str, Enum):
    SHELTER = "shelter"
    OXYGEN_PLANT = "oxygen_plant"
    SOLAR_ARRAY = "solar_array"
    GREENHOUSE = "greenhouse"
    WATER_EXTRACTOR = "water_extractor"
    HEATER = "heater"
    RESEARCH_LAB = "research_lab"
    STORAGE_DEPOT = "storage_depot"
    HABITAT = "habitat"
    INFIRMARY = "infirmary"
    WEATHER_STATION = "weather_station"


BUILD_COSTS = {
    StructureType.SHELTER: ResourceBundle(construction_material=3, minerals=1),
    StructureType.OXYGEN_PLANT: ResourceBundle(construction_material=4, minerals=3, energy=1),
    StructureType.SOLAR_ARRAY: ResourceBundle(construction_material=2, minerals=2),
    # Water, not ice: colonists never carry ice (the colony well refills
    # canteens with water), so an ice cost froze greenhouse construction at the
    # founding stock for whole runs.
    StructureType.GREENHOUSE: ResourceBundle(construction_material=3, minerals=1, water=1),
    # **Il pozzo costa quanto l'impianto d'ossigeno, e non e' un caso.** Sono
    # lo stesso genere di oggetto: un impianto che estrae dall'ambiente
    # locale un bene vitale che l'ambiente non offre gia' pronto. L'acqua
    # marziana sta sotto, come ghiaccio e come acqua legata nei minerali
    # idrati: tirarla su richiede perforare, scaldare e condensare, che e'
    # il lavoro che l'impianto d'ossigeno fa sull'atmosfera.
    StructureType.WATER_EXTRACTOR: ResourceBundle(
        construction_material=4, minerals=3, energy=1
    ),
    StructureType.HEATER: ResourceBundle(construction_material=2, minerals=2, energy=1),
    StructureType.RESEARCH_LAB: ResourceBundle(construction_material=4, minerals=2, energy=1),
    StructureType.STORAGE_DEPOT: ResourceBundle(construction_material=2, minerals=1),
    StructureType.HABITAT: ResourceBundle(construction_material=5, minerals=3, oxygen=1),
    StructureType.INFIRMARY: ResourceBundle(construction_material=4, minerals=2, oxygen=2, energy=1),
    StructureType.WEATHER_STATION: ResourceBundle(construction_material=2, minerals=1, energy=1),
}

#: **Composizione della domanda del parco impianti.** Quanti minerali costa una
#: struttura per ogni unita' di materiale da costruzione, mediata su tutto il
#: catalogo. Non e' un numero scelto: e' `BUILD_COSTS` stesso, letto qui una
#: volta sola. `kernel_biology` lo usa per ripartire la resa dell'ISRU fra i due
#: prodotti, cosi' la conversione rende esattamente la miscela che la colonia
#: consuma e nessuno dei due beni puo' strozzare l'altro per un disallineamento
#: fra due tabelle. Se un costo cambia, la ripartizione lo segue da sola.
DOMANDA_MINERALI_SU_MATERIALE = (
    sum(float(c.minerals) for c in BUILD_COSTS.values())
    / sum(float(c.construction_material) for c in BUILD_COSTS.values())
)

#: **Le quattro strutture che rendono abitabile una cella.** Sono i termini del
#: `min` in `local_life_support_capacity`: finche' ne manca una la capienza
#: vitale vale zero e la cella non puo' ospitare nessuno. Stanno qui, accanto ai
#: costi, perche' tre parti del modello ne hanno bisogno e nessuna delle tre
#: deve poter riscrivere l'elenco per conto suo: il kit del fondatore
#: (`rule_based_agent`), i posti di cantiere (`build_policy`) e la dotazione di
#: regolito di ogni cella (`world_generator`).
STRUTTURE_DI_AVVIAMENTO = (
    # **L'ordine conta e non e' alfabetico.** E' la sequenza in cui un fondatore
    # tira su l'avamposto (`SETTLE_OPERATIONAL_STRUCTURES` e' questo stesso
    # elenco): prima cio' che mette al sicuro cibo e acqua, poi l'energia che
    # alimenta l'impianto d'ossigeno. Il pozzo chiude la fila perche' la squadra
    # arriva con l'acqua da viaggio nel kit e la serra ne produce una parte.
    StructureType.GREENHOUSE,
    StructureType.SHELTER,
    StructureType.SOLAR_ARRAY,
    StructureType.OXYGEN_PLANT,
    # **L'acqua entra fra i requisiti il 2026-09-01.** La capienza vitale
    # dichiarava che una cella sostiene N persone guardando alloggi, cibo,
    # energia e ossigeno — e mai se poteva DISSETARLE. Un avamposto risultava
    # percio' operativo mentre i suoi abitanti morivano di sete: misurato,
    # quattordici decessi su diciannove con il magazzino idrico sotto 0,1
    # mentre cibo e ossigeno erano abbondanti (voce C9).
    StructureType.WATER_EXTRACTOR,
)

#: Il materiale da costruzione che serve per avviare un avamposto: dodici
#: unita', lette da `BUILD_COSTS` e non scritte a mano.
COSTO_MATERIALE_AVVIAMENTO = sum(
    float(BUILD_COSTS[tipo].construction_material) for tipo in STRUTTURE_DI_AVVIAMENTO
)

#: L'acqua che l'avviamento di un avamposto richiede per essere COSTRUITO: e'
#: nei `BUILD_COSTS` delle quattro strutture, e la serra e' l'unica a chiederne.
COSTO_ACQUA_AVVIAMENTO = sum(
    float(BUILD_COSTS[tipo].water) for tipo in STRUTTURE_DI_AVVIAMENTO
)

BUILD_TIME = {
    StructureType.SHELTER: 1,
    StructureType.OXYGEN_PLANT: 3,
    StructureType.SOLAR_ARRAY: 2,
    StructureType.GREENHOUSE: 2,
    StructureType.WATER_EXTRACTOR: 3,
    StructureType.HEATER: 2,
    StructureType.RESEARCH_LAB: 3,
    StructureType.STORAGE_DEPOT: 1,
    StructureType.HABITAT: 3,
    StructureType.INFIRMARY: 3,
    StructureType.WEATHER_STATION: 2,
}


@dataclass
class Structure:
    type: StructureType
    x: int
    y: int
    local_x_m: float = 0.0
    local_y_m: float = 0.0
    owner: str | None = None
    integrity: float = 1.0  # 0.0 to 1.0

    @property
    def efficiency(self) -> float:
        """Production efficiency based on integrity. Drops sharply below 40%."""
        if self.integrity <= 0:
            return 0.0
        return self.integrity if self.integrity > 0.4 else self.integrity * 0.5

    @property
    def local_effect(self) -> dict[str, float]:
        base_effects = {
            StructureType.SHELTER: {"habitability": 0.12},
            StructureType.OXYGEN_PLANT: {"oxygen": 1.0},
            StructureType.SOLAR_ARRAY: {"energy": 1.5},
            StructureType.GREENHOUSE: {"biomass": 1.0, "habitability": 0.05, "food": 1.0, "water": 0.08},
            # **La resa del pozzo e' la razione di chi serve.** Un colono
            # preleva `RAZIONE_ACQUA` (0,1) dalla giacenza della cella a
            # ogni passo; un pozzo serve `COLONISTS_PER_STRUCTURE` persone
            # (7, le stesse di una serra, cosi' acqua e cibo scalano
            # insieme); la produzione di una struttura vale
            # `local_effect x RESA_ACQUA` (5,2910). Quindi
            # `0,1 x 7 / 5,2910 = 0,1323`. Le tre costanti vivono in tre
            # strati diversi e non possono essere importate qui senza
            # invertire le dipendenze: e' un test a tenere ferma la
            # relazione (`tests/test_pozzo_idrico.py`).
            #
            # Per confronto, una serra ne produce 0,08 x 5,2910 = 0,423 per
            # le stesse sette persone, che ne bevono 0,7: la serra da' sola
            # copriva il 60% del fabbisogno idrico, ed e' la ragione per cui
            # si moriva di sete accanto a una serra.
            StructureType.WATER_EXTRACTOR: {"water": 0.1323},
            StructureType.HEATER: {"temperature": 2.0, "habitability": 0.08},
            StructureType.RESEARCH_LAB: {"knowledge": 1.0},
            StructureType.STORAGE_DEPOT: {"storage": 1.0},
            StructureType.HABITAT: {"habitability": 0.2, "population_support": 1.0, "food": 0.35, "water": 0.05},
            StructureType.INFIRMARY: {"oxygen": 0.6, "habitability": 0.1, "healing_bonus": 0.55, "water": 0.12},
            StructureType.WEATHER_STATION: {"knowledge": 0.2, "prediction_range": 5.0},
        }[self.type]
        
        # Apply efficiency multiplier to all effects
        return {k: v * self.efficiency for k, v in base_effects.items()}

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "x": self.x,
            "y": self.y,
            "local_x_m": float(self.local_x_m),
            "local_y_m": float(self.local_y_m),
            "owner": self.owner,
            "integrity": float(self.integrity),
            "efficiency": float(self.efficiency)
        }
