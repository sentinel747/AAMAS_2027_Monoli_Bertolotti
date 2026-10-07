"""Governatore: lo strato che indirizza agenti che restano rule-based.

Dal 2026-08-24 il consiglio a N mandati e' stato sostituito da un governatore
unico che scrive la **policy di scelta** — regole globali per condizione di
cella, non direttive per coordinate. Nulla in questo pacchetto entra nel ciclo
per-agente: il governatore vive fuori dalla colonia.
"""

from src.governors.governor import Governor
from src.governors.observation import ColonyPicture, build_picture
from src.governors.policy import (
    Bounds,
    Condition,
    NEUTRAL_WEIGHT,
    Policy,
    PolicyDrops,
    Rule,
    parse_policy,
)

__all__ = [
    "Bounds",
    "ColonyPicture",
    "Condition",
    "Governor",
    "NEUTRAL_WEIGHT",
    "Policy",
    "PolicyDrops",
    "Rule",
    "build_picture",
    "parse_policy",
]
