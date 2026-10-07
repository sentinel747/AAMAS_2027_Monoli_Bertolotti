"""Vectorized needs shared by automatic logistics and cell proposals.

The user's 2026-07-24 v3 revision deliberately promotes the cell from a
resource-only pool to a task-menu publisher. ``src.core.kernel`` remains the
sole decision-path importer: it computes this matrix once for redistribution
and once after transfers for ``cell_proposals``. The proposal module receives
the array as data and does not import this module, preserving a narrow seam.
No module may import this one except `src.core.kernel` and
`src.core.redistribution` - see
`tests/core/test_needs.py::test_needs_matrix_not_imported_by_any_agents_module`,
which scans the whole `src/` tree for accidental coupling. The cell publishes
priorities and quotas, not agent assignments: individual preferences still
make the final stochastic choice. Preference runs enable logistics by default;
historical tree configs still require an explicit flag.

Formulas transcribed from
docs/superpowers/plans/2026-07-19-soa-core-cell-redistribution.md, Task 7
(spec sez. 2.3), with one interpretive gap the plan's own prose leaves open
(documented at NEEDS_MATERIALS_TARGET_FALLBACK below): the "materials" need
groups with water/food/oxygen in the plan's prose ("water/food/oxygen/
materials: max(0, occupants * knapsack_target[res] * 2 - cell_res[res])"),
but `RedistributionConfig`'s own default `knapsack_targets` dict (also given
in the plan) only has water/food/oxygen/med_kits keys - there is no
"materials" resource in `src.core.constants.RESOURCES` and no default target
for it. This module resolves that gap by (a) mapping the "materials" need to
the combined stock of `construction_material` + `minerals` (the two resources
Task 8's flow stage explicitly moves toward `materials`/`build_*` needs), and
(b) falling back to `knapsack_targets.get("construction_material", 2.0)` as
the per-occupant target when the config does not supply one. Both choices are
config-overridable (a caller can set
`knapsack_targets={"construction_material": X, ...}`), and neither is
exercised by the plan's own golden tests, which only assert on water/build_*/
non-settled-cell behavior. This deviation from the plan's underspecified
prose was implemented but never recorded in the thesis diary until the Task
10 review addendum (see Tesi_LaTex/tesiprogress.md and this plan's own errata
note at Task 7's interface bullet) - recorded here for durability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.agents.build_policy import COLONISTS_PER_STRUCTURE
from src.core import constants as C
from src.core.arrays import CellArrays
from src.simulation.step_effects import RIPARAZIONE_MANUTENZIONE
from src.world.occupancy import housing_slots_from_structures
from src.world.structures import StructureType

NEEDS_MATERIALS_TARGET_FALLBACK = 2.0

# Single source of truth for the default per-occupant knapsack targets, shared
# by the dataclass field default AND by `from_config`'s merge-onto-defaults
# below (see from_config's own docstring for why a caller-supplied dict is
# merged rather than substituted wholesale).
_DEFAULT_KNAPSACK_TARGETS: dict[str, float] = {
    "water": 4.0,
    "food": 2.0,
    "oxygen": 1.0,
    "med_kits": 1.0,
    "construction_material": 2.0,
    "minerals": 1.0,
    "energy": 1.0,
    "tools": 1.0,
}


@dataclass
class RedistributionConfig:
    """Parser-level default remains off for historical callers.

    ``kernel.step`` promotes an absent flag to on only for preference runs;
    an explicit flag always wins.
    """

    enabled: bool = False
    knapsack_targets: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_KNAPSACK_TARGETS))
    # Adjacent settled macro-cells share supplies automatically. Zero remains
    # available as an explicit isolation experiment, but user-facing
    # preference runs must not strand construction stock in the mother cell.
    flow_radius: int = 1
    flow_rate_per_step: float = 2.0
    greenhouse_per_capita: float = 0.25

    @classmethod
    def from_config(cls, config: dict) -> "RedistributionConfig":
        """`raw["knapsack_targets"]`, if present, is MERGED onto
        `_DEFAULT_KNAPSACK_TARGETS` rather than substituting it wholesale -
        a caller passing `{"water": 3.0}` overrides only the water target and
        keeps the food/oxygen/med_kits defaults, instead of silently zeroing
        them (a config with a partial override used to zero every need it
        did not mention, since `RedistributionConfig(**kwargs)` replaced the
        whole dict). Every supplied key is validated against
        `src.core.constants.RESOURCES` - `redistribute()` (`redistribution.py`)
        indexes `C.R[res]` directly, which raises a bare `KeyError` for any
        non-resource key (notably `"materials"`, which the plan's own prose
        suggests but which is not a real resource column - see this module's
        docstring); this validation turns that crash into a clear `ValueError`
        at config-parse time instead."""
        raw = config.get("redistribution", {})
        raw = raw if isinstance(raw, dict) else {}
        kwargs: dict = {}
        if "enabled" in raw:
            kwargs["enabled"] = bool(raw["enabled"])
        if isinstance(raw.get("knapsack_targets"), dict):
            supplied = {str(k): float(v) for k, v in raw["knapsack_targets"].items()}
            invalid = sorted(k for k in supplied if k not in C.RESOURCES)
            if invalid:
                raise ValueError(
                    f"redistribution.knapsack_targets has non-resource key(s) {invalid!r}; "
                    f"valid keys are {C.RESOURCES!r} (there is no 'materials' resource - "
                    "use 'construction_material'/'minerals', see needs.py's module docstring "
                    "for how the 'materials' need is derived from those two)"
                )
            kwargs["knapsack_targets"] = {**_DEFAULT_KNAPSACK_TARGETS, **supplied}
        if "flow_radius" in raw:
            kwargs["flow_radius"] = int(raw["flow_radius"])
        if "flow_rate_per_step" in raw:
            kwargs["flow_rate_per_step"] = float(raw["flow_rate_per_step"])
        if "greenhouse_per_capita" in raw:
            kwargs["greenhouse_per_capita"] = float(raw["greenhouse_per_capita"])
        return cls(**kwargs)


def settled_mask(cells: CellArrays) -> np.ndarray:
    """True where a cell hosts at least one completed structure (any type,
    `struct_count.sum(-1) > 0`) OR an active construction site
    (`site_progress >= 0` for any type) - the operational definition of
    "insediata" from spec sez. 2.3."""
    struct_any = cells.struct_count.sum(axis=-1) > 0
    site_any = (cells.site_progress >= 0.0).any(axis=-1)
    return struct_any | site_any


def compute_needs(cells: CellArrays, cfg: RedistributionConfig) -> np.ndarray:
    """Return a `[H, W, NN]` float64 needs matrix, zero everywhere except
    settled cells (`needs *= settled_mask[..., None]` at the end, per the
    plan's own instruction - no interpretive freedom there)."""
    occ = cells.occupancy.astype(np.float64)
    needs = np.zeros((cells.H, cells.W, C.NN), dtype=np.float64)

    for res in ("water", "food", "oxygen"):
        target = float(cfg.knapsack_targets.get(res, 2.0))
        stock = cells.cell_res[..., C.R[res]]
        needs[..., C.ND[res]] = np.maximum(0.0, occ * target * 2.0 - stock)

    materials_target = float(cfg.knapsack_targets.get("construction_material", NEEDS_MATERIALS_TARGET_FALLBACK))
    materials_stock = cells.cell_res[..., C.R["construction_material"]] + cells.cell_res[..., C.R["minerals"]]
    needs[..., C.ND["materials"]] = np.maximum(0.0, occ * materials_target * 2.0 - materials_stock)

    solar_i = C.S[StructureType.SOLAR_ARRAY]
    o2_i = C.S[StructureType.OXYGEN_PLANT]
    gh_i = C.S[StructureType.GREENHOUSE]

    # Keep solar/O2 urgent until the SAME population coverage used by the
    # proposal mask and executor is reached.  The former existence-only flag
    # dropped to zero after the first building in a cell, although saturation
    # still allowed 1 solar/7 colonists and 1 O2 plant/10 colonists.  In the
    # preference top-k that demoted every subsequent unit to background work:
    # the 1500-step audit ended at 37/62 solar and 25/43 O2 units.
    # **Il margine di crescita NON si applica qui, ed e' una misura, non una
    # svista (2026-08-30).** Applicarlo anche al fabbisogno di pannelli,
    # impianti e serre e' stato provato: la colonia costruisce di piu' (28
    # pannelli invece di 23) ma le costruzioni occupano tre dei cinque posti del
    # menu, `explore` e `move` restano fuori, la colonia non esce piu' dalla
    # cella madre, ne esaurisce il giacimento e si ferma a 170 con il 66% dei
    # coloni a `observe`. Senza margine qui: 199 coloni su dieci celle. Il menu
    # `top_k` e' a somma zero, e alzare una priorita' ne abbassa un'altra.
    def coverage_missing(structure_type: StructureType, structure_i: int) -> np.ndarray:
        ratio = float(COLONISTS_PER_STRUCTURE[structure_type])
        target = np.maximum(1.0, np.ceil(occ / ratio))
        missing = np.maximum(
            0.0,
            target - cells.struct_count[..., structure_i].astype(np.float64),
        )
        no_active_site = cells.site_progress[..., structure_i] < 0.0
        return missing * no_active_site

    needs[..., C.ND["build_solar"]] = coverage_missing(
        StructureType.SOLAR_ARRAY, solar_i
    )
    needs[..., C.ND["build_oxygen_plant"]] = coverage_missing(
        StructureType.OXYGEN_PLANT, o2_i
    )
    # **L'acqua e' un requisito vitale come gli altri (2026-09-01).** Il pozzo
    # nasceva nel livello di sviluppo, insieme a laboratori e stazioni meteo, e
    # con `top_k = 5` non entrava quasi mai nel menu: misurato a 400 passi,
    # SETTE pozzi contro quarantacinque serre e quarantatre impianti d'ossigeno,
    # UNA sola cella operativa e quarantatre cantieri fermi — trentaquattro dei
    # quali a un solo requisito mancante, che era sempre lo stesso.
    needs[..., C.ND["build_water"]] = coverage_missing(
        StructureType.WATER_EXTRACTOR,
        C.S[StructureType.WATER_EXTRACTOR],
    )

    gh_target = np.ceil(occ * cfg.greenhouse_per_capita)
    gh_missing = np.maximum(
        0.0,
        gh_target - cells.struct_count[..., gh_i].astype(np.float64),
    )
    gh_missing *= cells.site_progress[..., gh_i] < 0.0
    needs[..., C.ND["build_greenhouse"]] = gh_missing

    count_f = cells.struct_count.astype(np.float64)
    mean_int = np.divide(cells.struct_integrity, count_f, out=np.zeros_like(cells.struct_integrity), where=count_f > 0)
    # **Il bisogno si misura come lo misura l'esecutore (2026-08-27).** Era
    # `max(0, 0,6 - integrita_media) * conteggio`: un bisogno che restava
    # ESATTAMENTE ZERO finche' l'integrita' media non scendeva sotto 0,6. Ma la
    # produzione di una struttura scala linearmente con l'integrita'
    # (`Structure.efficiency`), quindi una cella smette di essere
    # autosufficiente molto prima che la soglia scatti. Misurato sulle sette
    # run del 26-27 agosto (200 coloni, cella senza ghiaccio, acqua prodotta
    # dalle sole strutture): il pareggio fra produzione e consumo d'acqua cade
    # a integrita' 0,82 (passo 46), il bisogno si accendeva a 0,60 (passo 105).
    # Cinquantanove passi — 413 giorni — in cui il magazzino si svuotava e il
    # modello dichiarava che non c'era niente da manutenere. Sei run su sette
    # si sono estinte per disidratazione senza che una sola manutenzione
    # venisse mai proposta, quando UNA SOLA azione avrebbe retto la cella per
    # cento passi.
    #
    # Ora il bisogno E' la grandezza che `integrita_da_ripristinare` calcola
    # per l'esecutore: quanta integrita' una manutenzione recupererebbe
    # davvero, cioe' `min(0,4, 1 - integrita)` sommato sulle strutture. Una
    # sola definizione per chi propone e per chi esegue, e nessuna soglia da
    # tenere allineata a mano con una curva di produzione che non la conosce.
    maintenance_deficit = np.minimum(
        RIPARAZIONE_MANUTENZIONE, np.maximum(0.0, 1.0 - mean_int)
    ) * count_f
    needs[..., C.ND["maintenance"]] = maintenance_deficit.sum(axis=-1)

    # **`habitat_pressure` sono le persone scoperte ORA, e deve restare tale
    # (2026-08-30).** Il margine di crescita che sblocca la costruzione NON si
    # applica qui, e la ragione e' misurata: questo bisogno guida anche la
    # domanda di materiale fra celle (`redistribution._infrastructure_flow_demand`),
    # e una cella in deficit non dona. Gonfiandolo col margine ogni cella
    # abitata risultava per sempre bisognosa, **nessuna donava piu' nulla** e
    # gli avamposti restavano senza materiale. Il margine vive dove serve, cioe'
    # nella priorita' e nella quota di `cell_proposals`.
    shelter_slots = housing_slots_from_structures(
        cells.struct_count[..., C.S[StructureType.SHELTER]],
        cells.struct_count[..., C.S[StructureType.HABITAT]],
        cells.struct_count[..., C.S[StructureType.INFIRMARY]],
    )
    needs[..., C.ND["habitat_pressure"]] = np.maximum(0.0, occ - shelter_slots)

    needs *= settled_mask(cells)[..., None]
    return needs
