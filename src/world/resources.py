from __future__ import annotations

from dataclasses import dataclass


def scorta_colonia(inventari, celle, nome: str) -> float:
    """Cio' che la colonia possiede DAVVERO: sacche dei coloni piu' magazzini.

    **Una sola implementazione, perche' due sono gia' costate una correzione
    (2026-08-30).** La stessa somma serve alla natalita' (`population._scorta`)
    e alle metriche pubblicate (`colony_dynamics`), e sommare le sole sacche
    non misura la colonia: con la redistribuzione attiva ogni sacca e' riempita
    fino a un bersaglio scritto nella configurazione, quindi il totale delle
    sacche e' una costante travestita. La natalita' era stata corretta il
    2026-08-27; le metriche no, e continuavano a pubblicare un quarto del
    materiale davvero posseduto.

    `celle` e' l'impronta della colonia, cioe' le celle con strutture: e' li'
    che stanno i magazzini.
    """
    totale = sum(float(getattr(inventario, nome, 0.0)) for inventario in inventari)
    totale += sum(float(getattr(cella.resources, nome, 0.0)) for cella in celle)
    return float(totale)


#: Le risorse con un ciclo vero: si producono, si consumano e si accumulano.
#: `ice`, `biomass`, `tools`, `knowledge` e `med_kits` sono accessorie e
#: sbilancerebbero un indice calcolato su di esse.
RISORSE_CON_CICLO = (
    "food", "water", "oxygen", "construction_material", "minerals", "energy",
)


def scorta_in_comune(celle, nome: str) -> float:
    """Quanto di una risorsa sta nei magazzini di cella, cioe' in comune."""
    return float(sum(float(getattr(cella.resources, nome, 0.0)) for cella in celle))


def indice_di_messa_in_comune(inventari, celle) -> float:
    """Quanta parte di cio' che la colonia possiede e' tenuta IN COMUNE.

    **Che cosa misura, e perche' e' questa la cooperazione di questo modello.**
    Con la redistribuzione spenta — il default — non esistono flussi
    automatici, e in modalita' a preferenze `share_resource` non e' un'azione
    ma un servizio di cella. Cio' che resta, e che c'e' sempre, e' il
    **magazzino condiviso**: `_remove_combined` paga ogni costruzione prima
    dalla sacca di chi lavora e poi dal magazzino della cella, quindi ogni
    opera e' in parte pagata dalla collettivita'. Questo indice e' lo stato di
    quel patto: 0 = ognuno tiene tutto addosso, 1 = tutto e' in comune.

    **Media non pesata sulle risorse, non rapporto degli aggregati.** Il
    rapporto degli aggregati sarebbe dominato dal bene piu' abbondante del
    momento — misurato sulla run `verifica1000_seed9`, il materiale da
    costruzione vale da solo piu' di tutti gli altri messi insieme — e
    oscillerebbe al variare di quello invece che della cooperazione. La media
    risponde a «di un bene qualunque, quanta parte e' in comune».
    """
    quote = []
    for nome in RISORSE_CON_CICLO:
        comune = scorta_in_comune(celle, nome)
        totale = scorta_colonia(inventari, celle, nome)
        if totale > 0.0:
            quote.append(comune / totale)
    if not quote:
        return 0.0
    return float(sum(quote) / len(quote))


@dataclass
class ResourceBundle:
    ice: float = 0.0
    minerals: float = 0.0
    energy: float = 0.0
    oxygen: float = 0.0
    water: float = 0.0
    biomass: float = 0.0
    food: float = 0.0
    tools: float = 0.0
    construction_material: float = 0.0
    knowledge: float = 0.0
    med_kits: float = 0.0

    def add(self, other: "ResourceBundle") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def remove(self, other: "ResourceBundle") -> bool:
        if not self.can_afford(other):
            return False
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) - getattr(other, field))
        return True

    def can_afford(self, cost: "ResourceBundle") -> bool:
        return all(getattr(self, field) >= getattr(cost, field) for field in self.__dataclass_fields__)

    def to_dict(self) -> dict[str, float]:
        return {field: float(getattr(self, field)) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceBundle":
        return cls(**{field: float(data.get(field, 0.0)) for field in cls.__dataclass_fields__})
