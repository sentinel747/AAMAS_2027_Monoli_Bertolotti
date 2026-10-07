from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.experiments.scenarios import get_standard_scenario_config
from src.agents.operational_range import (
    DEFAULT_OPERATIONAL_RANGE_M,
    operational_range_from_config,
)
from src.agents.role_profiles import DEFAULT_ROLE_DISTRIBUTION, ROLE_ORDER


#: Margine con cui la dotazione iniziale deve coprire la colonia fondatrice.
#:
#: **Perche' la dotazione non e' piu' un numero fisso (2026-08-27).** I valori
#: storici — dieci habitat, sette serre, cinque pannelli, cinque impianti —
#: erano tarati su cinquanta agenti e restavano identici a duecento: una colonia
#: di duecento persone partiva quindi con supporto vitale per venti, e
#: `local_life_support_capacity` e' il cancello che decide se puo' nascere
#: qualcuno. Misurato: zero nascite in duecento passi con quarantadue attese, e
#: nelle run del 26-27 agosto l'estinzione. Nessun altro parametro poteva
#: rimediare, perche' la capienza non e' una probabilita': e' un minimo.
#:
#: Il quindici per cento e' il margine perche' la colonia parta **coperta e con
#: spazio per crescere**: a copertura esatta lo spazio libero e' zero e la prima
#: nascita e' gia' vietata.
#: **Non e' la riserva di crescita** (`build_policy.posti_di_riserva`): quella
#: risponde a «quanti posti liberi servono perche' una nascita sia possibile»,
#: questa a «con quanta scorta parte una spedizione fondatrice». Due domande,
#: due numeri, entrambi dichiarati.
MARGINE_DOTAZIONE_INIZIALE = 1.15

#: Il valore dei campi `initial_*` che significa "scala con la popolazione".
DOTAZIONE_DERIVATA = -1


def dotazione_iniziale(agent_count: int) -> dict[str, int]:
    """Le strutture con cui una colonia di `agent_count` persone deve partire.

    Derivata dagli stessi rapporti che `local_life_support_capacity` usa per
    decidere quante persone una cella sostiene (`COLONISTS_PER_STRUCTURE`), non
    da una tabella parallela: due formule per la stessa grandezza sono il
    difetto ricorrente di questo modello, e qui la seconda decideva la vita
    della colonia.

    Gli alloggi restano al rapporto storico di un habitat ogni cinque coloni: il
    resto lo coprono i rifugi, che sono economici e che la colonia costruisce nei
    primi passi. Serre, pannelli e impianti d'ossigeno coprono invece la
    popolazione **piu' il margine**, perche' quelli la colonia non riesce a
    recuperarli in corsa quando il magazzino e' vuoto.
    """
    from math import ceil

    from src.agents.build_policy import COLONISTS_PER_STRUCTURE
    from src.world.structures import StructureType

    n = max(1, int(agent_count))
    coperti = ceil(n * MARGINE_DOTAZIONE_INIZIALE)
    per = COLONISTS_PER_STRUCTURE
    return {
        "habitat": ceil(n / per[StructureType.HABITAT]),
        "greenhouse": ceil(coperti / per[StructureType.GREENHOUSE]),
        "solar_array": ceil(coperti / per[StructureType.SOLAR_ARRAY]),
        "oxygen_plant": ceil(coperti / per[StructureType.OXYGEN_PLANT]),
        # **E i pozzi (2026-09-01).** L'acqua e' entrata fra i termini di
        # `local_life_support_capacity`: senza pozzi la colonia madre nasce
        # con capienza vitale ZERO, quindi non puo' avere figli ne' fondare
        # avamposti. Stesso rapporto e stesso margine degli altri tre.
        "water_extractor": ceil(coperti / per[StructureType.WATER_EXTRACTOR]),
        "storage_depot": max(1, ceil(n / per[StructureType.STORAGE_DEPOT])),
        "weather_station": 1,
    }


MAP_PROFILES = ("balanced", "scarce_resources", "high_hazard", "ice_rich", "mineral_rich", "fragmented", "random")
CLIMATE_SCENARIOS = ("climatology", "clim_minEUV", "clim_maxEUV", "cold", "warm")
DECISION_MODES = ("preferences", "tree")
DECISION_SAMPLINGS = ("softmax", "greedy")
DEFAULT_MCD_RUNTIME_PATH = "data/mcd_runtime"

# Redistribution defaults (task-11-brief.md), matching src/core/needs.py's
# RedistributionConfig field-for-field. NOT imported from src.core.needs on
# purpose: task-10-brief.md's binding modification #2 restricts importers of
# that module to src/core/kernel.py and src/core/redistribution.py ONLY
# (enforced by tests/core/test_needs.py::
# test_needs_matrix_not_imported_by_any_agents_module, which scans the whole
# src/ tree) - src/experiments is a decision-adjacent config layer, not one
# of the two allowed callers, so it must NOT import RedistributionConfig even
# just to read its defaults. Kept in sync by construction instead:
# tests/test_manual_config.py::test_manual_config_redistribution_block_satisfies_redistribution_config_from_config
# builds a real RedistributionConfig.from_config(build_manual_config(...))
# and asserts every value below matches - a drift here fails that test
# immediately rather than silently.
_REDISTRIBUTION_DEFAULT_KNAPSACK_WATER = 4.0
_REDISTRIBUTION_DEFAULT_KNAPSACK_FOOD = 2.0
_REDISTRIBUTION_DEFAULT_KNAPSACK_OXYGEN = 1.0
_REDISTRIBUTION_DEFAULT_FLOW_RADIUS = 1
_REDISTRIBUTION_DEFAULT_FLOW_RATE = 2.0
_REDISTRIBUTION_DEFAULT_GREENHOUSE_PER_CAPITA = 0.25


@dataclass
class ManualConfigOptions:
    # Defaults tarati per una colonia di 50 agenti; la regola di scaling per
    # inventario (per agente) e strutture (totali colonia) e' documentata in
    # docs/MARS_ABM_NOTE_CONFIGURAZIONE.md. GUI React, headless runner e wizard CLI
    # devono restare allineati a questi valori.
    run_name: str = ""
    selected_scenario_id: str = ""
    seed: int = 0
    mode: str = "rule"
    agent_count: int = 50
    initial_food: float = 5
    initial_water: float = 5
    initial_materials: float = 5
    initial_tools: float = 1
    #: Le sei strutture iniziali. `DOTAZIONE_DERIVATA` (-1) = scala con
    #: `agent_count` secondo `dotazione_iniziale`; un valore >= 0 la
    #: sovrascrive ed e' quello che l'utente digita nelle interfacce.
    initial_habitats: int = DOTAZIONE_DERIVATA
    initial_greenhouses: int = DOTAZIONE_DERIVATA
    initial_solar_arrays: int = DOTAZIONE_DERIVATA
    initial_oxygen_plants: int = DOTAZIONE_DERIVATA
    initial_water_extractors: int = DOTAZIONE_DERIVATA
    initial_storage_depots: int = DOTAZIONE_DERIVATA
    initial_weather_stations: int = DOTAZIONE_DERIVATA
    #: Conversione ISRU minerali -> materiale da costruzione, per unita' di
    #: effetto `storage` e per passo. **Accesa dal 2026-08-27**: da spenta il
    #: materiale da costruzione e' una dotazione iniziale non rinnovabile e la
    #: colonia finisce in uno stato ASSORBENTE — esaurita la scorta non puo'
    #: piu' costruire nulla, nemmeno i depositi che ne produrrebbero. Il valore
    #: e' 1,0 e non 2,0 perche' misurati danno lo stesso esito (la conversione
    #: e' limitata dai minerali della cella, non dal tasso): fra due valori
    #: equivalenti si prende il piu' piccolo. Vedi `kernel_biology.update_cells`.
    isru_material_rate: float = 1.0
    #: Priorita' delle costruzioni di sviluppo nel menu di cella. 0,07 e' il
    #: valore storico, e a `cell_proposal_top_k = 5` le tiene fuori dal menu:
    #: laboratori e infermerie hanno domanda non soddisfatta e non vengono mai
    #: costruiti. Alzarlo e' una leva sperimentale.
    development_build_priority: float = 0.30
    #: Temperatura dei modelli degli AGENTI (non del consiglio). `None` = usa il
    #: valore del registro dei provider, che e' una proprieta' della macchina e
    #: non della run: senza questo campo una campagna non potrebbe variarla senza
    #: modificare un file condiviso.
    llm_temperature: float | None = None
    #: Il governatore unico (dal 2026-08-24: niente piu' consiglio, count e
    #: mandati). `"none"` non scrive alcun blocco: `build_governor` legge il
    #: blocco assente come "nessun governatore" e restituisce `None`, e col
    #: `None` il kernel non invoca nemmeno l'applicazione della policy — la
    #: baseline resta bit-exact per costruzione invece che per lettura di
    #: stringa. Averli qui, e non solo negli argomenti dello script, e' cio'
    #: che permette al wizard di configurarli come fa il frontend.
    governors_arm: str = "none"
    governors_cadence_steps: int = 20
    governors_wait_seconds: float = 0.0
    governors_provider: str = "fallback"
    governors_model: str = ""
    governors_effort: str = ""
    governors_temperature: float | None = None
    #: Quanto contesto riceve il braccio `llm`: `completo`, `senza_aiuti`,
    #: `nomi_veri`, `cieco`. E' una variabile sperimentale, non un dettaglio di
    #: prompt: separa "il modello ragiona sui numeri" da "il modello riconosce
    #: il dominio". Vedi `src/governors/context.py`.
    governors_context_level: str = "completo"
    #: Il ragionamento del modello del governatore: `off`, `low`, `medium`,
    #: `high`, `dynamic`. Ogni API lo esprime a modo suo e la traduzione sta in
    #: `src/llm/thinking.py`; il default e' spento perche' un ragionamento
    #: nascosto aggiunge varianza che non appartiene al trattamento.
    governors_thinking: str = "off"
    #: Lo strato amministrativo: l'interruttore della run con o senza distretti.
    administrators_enabled: bool = False
    #: Quante celle tiene un amministratore. La cella madre non conta mai:
    #: resta sotto il governo diretto.
    administrators_cells_per_district: int = 3
    #: Vero: gli amministratori usano braccio, provider, modello e contesto del
    #: governo. Falso: usano i propri, dichiarati nei campi seguenti.
    administrators_follow_governor_arm: bool = True
    administrators_arm: str = "llm"
    administrators_provider: str = "fallback"
    administrators_model: str = ""
    #: Lista `provider:modello` separati da virgola: ogni distretto nuovo ne
    #: pesca uno, riproducibile a parita' di seme. Vuota = il solo modello sopra.
    administrators_models: str = ""
    administrators_thinking: str = "off"
    colony_start_x: int | None = None
    colony_start_y: int | None = None
    population_growth_enabled: bool = True
    growth_probability: float = 0.05
    llm_count: int = 3
    max_llm_calls_per_step: int = 0
    fast_observation_enabled: bool = False
    psychosocial_enabled: bool = False
    # ON (default): le celle si degradano con l'occupazione (pollution_risk)
    # e gli agenti ricevono i SYSTEM ALERT di sovraffollamento/inquinamento.
    # OFF: niente degradazione ne' alert.
    cell_degradation_enabled: bool = True
    # User-facing runs opt into the new policy. The kernel itself retains a
    # missing-key fallback of "tree" for historical raw test configurations.
    decision_mode: str = "preferences"
    decision_sampling: str = "softmax"
    # ON preserves individual preferences, physiological urgency and hard
    # survival overrides. OFF is an experimental cell-only policy.
    individual_survival_priority_enabled: bool = True
    # One operational radius drives both perception and maximum movement. With
    # the weekly default step, 59 km is the configurable upper bound; runs can
    # reduce it down to 0.1 km.
    operational_range_m: float = DEFAULT_OPERATIONAL_RANGE_M
    cell_proposal_top_k: int = 5
    role_distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ROLE_DISTRIBUTION)
    )
    role_preference_randomness: float = 0.25
    max_steps: int | None = 500
    #: Ogni quanti passi si salva uno snapshot del mondo in
    #: `world_snapshots/`. `0` = nessuno snapshot, ed e' il default perche' i
    #: file pesano; **senza snapshot la run non e' rivedibile sulla mappa
    #: dell'analysis frontend**. Prima del 2026-08-31 il valore era scritto a
    #: mano in due posti — `0` cablato nel pannello React senza alcun controllo,
    #: e `0` qui — quindi una run lanciata dalla GUI non poteva produrne
    #: nessuno, mai, e da terminale servivano gli `--overrides` a mano.
    snapshot_interval: int = 0
    days_per_step: int = 7
    max_sim_days: int | None = None
    map_profile: str = "balanced"
    earth_mars_delay_minutes: float = 12
    climate_scenario: str = "climatology"
    environmental_layer_enabled: bool = True
    extreme_events_enabled: bool = True
    extreme_event_chance: float = 0.05
    # Task 11: redistribution (src/core/needs.py's RedistributionConfig,
    # consumed by src/core/kernel.py's step() only when enabled=True).
    # Enabled for user-facing preference runs: sharing is an implicit cell
    # service and never consumes an agent decision step.
    # Defaults below match RedistributionConfig's own defaults field-for
    # -field - see the _REDISTRIBUTION_DEFAULT_* constants above for why they
    # are not imported directly.
    #: **Default OFF dal 2026-08-30, ed e' una correzione doppia.**
    #:
    #: *Il disallineamento.* Il nucleo (`RedistributionConfig.enabled = False`),
    #: i commenti del kernel («disabled -- the default») e
    #: `docs/MARS_ABM_ARCHITETTURA_SIMULAZIONE.md` dicevano tutti **OFF**,
    #: mentre il default che ogni run utente usava diceva **ON**. Documentazione
    #: e codice non coincidevano proprio sul valore piu' visibile.
    #:
    #: *La misura.* Cento coloni, duecentocinquanta passi, tre semi:
    #:
    #:  redistribuzione   seme 9        seme 21           seme 33
    #:  ON                133 vivi, 0   43 vivi, 94 morti  121, 0
    #:  OFF               143, 0        137, 0             142, 0
    #:
    #: OFF vince su **tutti e tre**: piu' coloni, piu' strutture (195/191/192
    #: contro 170/170/169) e **zero morti ovunque**. Il perche' era gia' noto e
    #: registrato come voce C8: la logistica riempie le sacche fino al bersaglio
    #: e **non lascia scorta in magazzino**, quindi ogni opera pubblica che costi
    #: piu' di una sacca diventa impagabile — misurato gia' il 2026-08-27, otto
    #: impianti d'ossigeno contro dodici. E con le sacche tutte allo stesso
    #: bersaglio la carestia arriva per tutti nello stesso passo invece che per
    #: qualcuno alla volta (voce C12).
    #:
    #: Resta accendibile: e' un esperimento sulla logistica, non un difetto.
    redistribution_enabled: bool = False
    redistribution_knapsack_water: float = _REDISTRIBUTION_DEFAULT_KNAPSACK_WATER
    redistribution_knapsack_food: float = _REDISTRIBUTION_DEFAULT_KNAPSACK_FOOD
    redistribution_knapsack_oxygen: float = _REDISTRIBUTION_DEFAULT_KNAPSACK_OXYGEN
    redistribution_flow_radius: int = _REDISTRIBUTION_DEFAULT_FLOW_RADIUS
    redistribution_flow_rate: float = _REDISTRIBUTION_DEFAULT_FLOW_RATE
    redistribution_greenhouse_per_capita: float = _REDISTRIBUTION_DEFAULT_GREENHOUSE_PER_CAPITA
    assignments: list[dict[str, str]] = field(default_factory=list)


def _dotazione_risolta(options: "ManualConfigOptions", agent_count: int) -> dict[str, int]:
    """La dotazione iniziale: derivata dalla popolazione, o sovrascritta a mano.

    Ogni campo e' indipendente dagli altri, cosi' chi vuole provare una colonia
    con poche serre non perde lo scaling di tutto il resto.
    """
    derivata = dotazione_iniziale(agent_count)
    scelte = {
        "habitat": options.initial_habitats,
        "greenhouse": options.initial_greenhouses,
        "solar_array": options.initial_solar_arrays,
        "oxygen_plant": options.initial_oxygen_plants,
        "water_extractor": options.initial_water_extractors,
        "storage_depot": options.initial_storage_depots,
        "weather_station": options.initial_weather_stations,
    }
    return {
        nome: derivata[nome] if valore < 0 else _clamp_int(valore, 0, 1000)
        for nome, valore in scelte.items()
    }


def build_manual_config(options: ManualConfigOptions) -> dict[str, Any]:
    """Build the same scenario payload produced by frontend ScenarioPanel."""
    base = _scenario_base(options.selected_scenario_id)
    clean_run_name = options.run_name.strip()
    agent_count = _clamp_agent_count(options.agent_count)
    llm_count = _effective_llm_count(options.mode, agent_count, options.llm_count)
    max_steps = _clamp_optional(options.max_steps, 1, 1_000_000)
    max_days = _clamp_optional(options.max_sim_days, 1, 10_000_000_000)
    map_profile = _normalize_choice(options.map_profile, MAP_PROFILES, "balanced")
    climate_scenario = _normalize_choice(options.climate_scenario, CLIMATE_SCENARIOS, "climatology")

    base_simulation = _as_dict(base.get("simulation"))
    base_llm = _as_dict(base.get("llm"))
    base_world = _as_dict(base.get("world"))
    base_social = _as_dict(base.get("social"))
    base_climate = _as_dict(base.get("climate"))
    base_environmental = _as_dict(base.get("environmental_layer"))
    base_extreme = _as_dict(base.get("extreme_events"))
    base_colony = _as_dict(base.get("colony"))
    base_initial_structures = _as_dict(base_colony.get("initial_structures"))
    base_population = _as_dict(base.get("population"))

    config = deepcopy(base)
    config["name"] = clean_run_name
    config["seed"] = _clamp_int(options.seed, 0, 2_147_483_647)
    if options.selected_scenario_id:
        config["scenario_id"] = options.selected_scenario_id
    if max_steps is not None:
        config["days"] = max_steps
    else:
        config.pop("days", None)

    config["simulation"] = {
        **base_simulation,
        "days_per_step": _clamp_int(options.days_per_step, 1, 1_000_000),
    }
    if max_days is not None:
        config["simulation"]["max_days"] = max_days
    else:
        config["simulation"].pop("max_days", None)

    config["world"] = {"width": 360, "height": 180, **base_world, "map_profile": map_profile, "cell_degradation": bool(options.cell_degradation_enabled)}
    operational_range_m = operational_range_from_config(
        {"operational_range_m": options.operational_range_m}
    )
    config["agents"] = {
        "count": agent_count,
        "llm_count": llm_count,
        "llm_assignments": _normalized_assignments(options.assignments, llm_count),
        "decision_mode": _normalize_choice(
            options.decision_mode, DECISION_MODES, "preferences"
        ),
        "decision_sampling": _normalize_choice(
            options.decision_sampling, DECISION_SAMPLINGS, "softmax"
        ),
        "individual_survival_priority_enabled": bool(
            options.individual_survival_priority_enabled
        ),
        "operational_range_m": operational_range_m,
        # Compatibility aliases for existing artifacts and analysis code. They
        # are generated from the canonical value and can no longer diverge.
        "vision_radius_m": operational_range_m,
        "movement_distance_m_per_step": operational_range_m,
        "development_build_priority": _clamp_float(options.development_build_priority, 0.0, 1.0),
        "cell_proposal_top_k": _clamp_int(options.cell_proposal_top_k, 1, 15),
        "role_distribution": _normalized_role_percentages(options.role_distribution),
        "role_preference_randomness": _clamp_float(
            options.role_preference_randomness, 0.0, 1.0
        ),
        "initial_inventory": {
            "food": _clamp_float(options.initial_food, 0, 1_000_000),
            "water": _clamp_float(options.initial_water, 0, 1_000_000),
            "construction_material": _clamp_float(options.initial_materials, 0, 1_000_000),
            "tools": _clamp_float(options.initial_tools, 0, 1_000_000),
            "oxygen": 2,
            "energy": 2,
            "minerals": 3,
            "med_kits": 2,
        },
    }
    config["colony"] = {
        **base_colony,
        "initial_structures": {
            **base_initial_structures,
            **_dotazione_risolta(options, agent_count),
        },
        "isru_material_rate": _clamp_float(options.isru_material_rate, 0.0, 1000.0),
    }
    if options.governors_arm != "none":
        blocco: dict[str, Any] = {
            "arm": options.governors_arm,
            "cadence_steps": max(1, int(options.governors_cadence_steps)),
            "wait_seconds": max(0.0, float(options.governors_wait_seconds)),
        }
        if options.governors_arm == "llm" and options.governors_provider != "fallback":
            # Senza assegnazione esplicita `build_governor` costruisce un
            # provider `fallback`: il braccio girerebbe sul deterministico
            # riportando come "risultato del modello" una run in cui nessun
            # modello ha parlato.
            assegnazione: dict[str, Any] = {
                "provider": options.governors_provider,
                "model": options.governors_model,
            }
            if options.governors_effort:
                assegnazione["effort"] = options.governors_effort
            if options.governors_temperature is not None:
                assegnazione["temperature"] = float(options.governors_temperature)
            if options.governors_thinking:
                assegnazione["thinking"] = str(options.governors_thinking)
            blocco["assignments"] = [assegnazione]
        blocco["context_level"] = str(options.governors_context_level or "completo")
        if options.administrators_enabled:
            blocco["administrators"] = _blocco_amministratori(options)
        config["governors"] = blocco
    elif options.administrators_enabled:
        # Decentramento puro: amministratori senza governo. E' una condizione
        # sperimentale legittima — chi decide, se non decide nessuno sopra? — e
        # senza questo ramo sarebbe irraggiungibile, perche' il blocco
        # `governors` viene scritto solo quando esiste un braccio di governo.
        config["governors"] = {
            "arm": "none",
            "cadence_steps": max(1, int(options.governors_cadence_steps)),
            "administrators": _blocco_amministratori(options),
        }

    # Explicit colony site: honored by SimulationController._resolve_colony_start;
    # when unset, the engine picks the shared find_safe_start site.
    if options.colony_start_x is not None and options.colony_start_y is not None:
        world_width = _clamp_int(config["world"].get("width", 360), 1, 1_000_000)
        world_height = _clamp_int(config["world"].get("height", 180), 1, 1_000_000)
        config["colony"]["start_x"] = _clamp_int(options.colony_start_x, 0, world_width - 1)
        config["colony"]["start_y"] = _clamp_int(options.colony_start_y, 0, world_height - 1)
    config["llm"] = {
        **base_llm,
        "max_calls_per_step": _clamp_int(options.max_llm_calls_per_step, 0, 100),
        # Assente quando non impostata: l'assenza significa "usa il registro",
        # mentre un valore — zero compreso — e' una scelta della run.
        **({} if options.llm_temperature is None else {"temperature": float(options.llm_temperature)}),
        "max_calls_per_run": base_llm.get("max_calls_per_run", 0),
        "max_calls_per_day": base_llm.get("max_calls_per_day", 0),
        "max_estimated_cost_usd": base_llm.get("max_estimated_cost_usd", 0),
    }
    config["model"] = {**_as_dict(base.get("model")), "psychosocial_enabled": bool(options.psychosocial_enabled)}
    config["headless"] = {
        **_as_dict(base.get("headless")),
        "fast_observation": bool(options.fast_observation_enabled),
        "rule_based_observation": True,
        # **Acceso di default: una run che non lascia traccia delle azioni non e'
        # analizzabile dopo.** Con questo spento `validated_actions.jsonl`,
        # `rejected_actions.jsonl`, `agent_decisions.jsonl`,
        # `agent_thoughts.jsonl` e `agent_conversations.jsonl` restano file da
        # zero byte, e il frontend di analisi apre la run trovando le sole
        # aggregate: il dettaglio di che cosa ogni agente ha fatto, sotto quali
        # direttive, non e' ricostruibile a posteriori da nessun'altra parte.
        #
        # Costa disco, non tempo: misurato su 300 agenti per 200 passi, +65 MB
        # e nessuna differenza di durata sul percorso `AgentCoupledRunner`.
        # Chi ha bisogno di run leggere lo spegne (`--no-agent-logs`, o questa
        # chiave negli overrides).
        "store_memory_logs": True,
        # Il conteggio dei vicini resta nel log; la lista completa degli ID e'
        # disattivata per evitare file JSONL sproporzionati nelle run lunghe.
        "store_nearby_agent_ids": False,
        "log_interval_steps": 10,
        # **Cento e non dieci, da quando i log per azione sono accesi.**
        # `save_run_artifacts` riscrive ogni file da capo a ogni scarico: con
        # 240.000 righe in memoria, scaricare ogni dieci passi significa
        # riscrivere un file che cresce, ottanta volte, e il costo va col
        # quadrato della lunghezza. Misurato sul percorso della GUI a 200 passi:
        # +14% di durata con scarico ogni dieci. A cento passi le riscritture
        # sono un decimo. Il prezzo e' che una run interrotta perde fino a cento
        # passi di log — gli altri artefatti compresi — e va detto perche' e' una
        # scelta, non un dettaglio.
        "output_flush_interval_steps": 100,
        "biology_update_scope": "active",
        "planetary_grid_refresh_interval_steps": 100,
        "force_final_planetary_grid_refresh": False,
        # Il tetto segue la run invece di essere un numero fisso: a 150.000
        # righe una run da 300 agenti per 800 passi (240.000 azioni) perderebbe
        # in silenzio l'ultimo terzo — ed e' la coda, dove la colonia si e'
        # assestata, quella che serve. Il vecchio commento vale ancora: al
        # tetto la fine della run sparisce dai log e le analisi diventano cieche
        # (visto al passo ~413 con 154 agenti sul default di 50k).
        "max_action_log_rows": max(
            150_000, int(agent_count * max(1, max_steps or 1000) * 1.1) + 1000
        ),
        "max_replay_rows": 250_000,
        "snapshot_interval": max(0, int(options.snapshot_interval)),
        "aggregate_threshold_agents": 50_000,
    }
    config["climate"] = {
        "enabled": True,
        "source": "auto",
        "provider": "call_mcd",
        **base_climate,
        "scenario": climate_scenario,
        "data_path": DEFAULT_MCD_RUNTIME_PATH,
        "mcd_path": DEFAULT_MCD_RUNTIME_PATH,
    }
    config["environmental_layer"] = {**base_environmental, "enabled": bool(options.environmental_layer_enabled), "role": "background"}
    config["social"] = {**base_social, "earth_mars_delay_minutes": _clamp_float(options.earth_mars_delay_minutes, 0, 44)}
    event_chance = _clamp_float(options.extreme_event_chance, 0, 1) if options.extreme_events_enabled else 0
    config["extreme_events"] = {**base_extreme, "enabled": bool(options.extreme_events_enabled), "chance_per_step": event_chance}
    config["extreme_events_enabled"] = bool(options.extreme_events_enabled)
    config["extreme_event_chance_per_step"] = event_chance
    config["population"] = {
        **base_population,
        "enabled": bool(options.population_growth_enabled),
        "daily_spawn_probability": _clamp_float(options.growth_probability, 0, 1),
        "prosperity_growth_scale": _clamp_float(options.growth_probability, 0, 1),
        "min_habitability_for_growth": float(base_population.get("min_habitability_for_growth", 0.02) or 0.02),
    }
    base_redistribution = _as_dict(base.get("redistribution"))
    base_knapsack_targets = _as_dict(base_redistribution.get("knapsack_targets"))
    config["redistribution"] = {
        "enabled": bool(options.redistribution_enabled),
        "knapsack_targets": {
            **base_knapsack_targets,
            "water": _clamp_float(options.redistribution_knapsack_water, 0, 1_000_000),
            "food": _clamp_float(options.redistribution_knapsack_food, 0, 1_000_000),
            "oxygen": _clamp_float(options.redistribution_knapsack_oxygen, 0, 1_000_000),
            "med_kits": float(base_knapsack_targets.get("med_kits", 1.0)),
            "construction_material": float(base_knapsack_targets.get("construction_material", 2.0)),
            "minerals": float(base_knapsack_targets.get("minerals", 1.0)),
            "energy": float(base_knapsack_targets.get("energy", 1.0)),
            "tools": float(base_knapsack_targets.get("tools", 1.0)),
        },
        "flow_radius": _clamp_int(options.redistribution_flow_radius, 0, 1_000_000),
        "flow_rate_per_step": _clamp_float(options.redistribution_flow_rate, 0, 1_000_000),
        "greenhouse_per_capita": _clamp_float(options.redistribution_greenhouse_per_capita, 0, 1_000_000),
    }
    return config


def options_from_scenario_config(config: dict[str, Any], scenario_id: str = "") -> ManualConfigOptions:
    agents = _as_dict(config.get("agents"))
    inventory = _as_dict(agents.get("initial_inventory"))
    colony = _as_dict(config.get("colony"))
    structures = _as_dict(colony.get("initial_structures"))
    population = _as_dict(config.get("population"))
    llm = _as_dict(config.get("llm"))
    headless = _as_dict(config.get("headless"))
    model = _as_dict(config.get("model"))
    simulation = _as_dict(config.get("simulation"))
    world = _as_dict(config.get("world"))
    social = _as_dict(config.get("social"))
    climate = _as_dict(config.get("climate"))
    environmental = _as_dict(config.get("environmental_layer"))
    extreme = _as_dict(config.get("extreme_events"))
    redistribution = _as_dict(config.get("redistribution"))
    redistribution_knapsack = _as_dict(redistribution.get("knapsack_targets"))
    agent_count = _clamp_agent_count(agents.get("count", 15))
    llm_count = _clamp_int(agents.get("llm_count", 0), 0, agent_count)
    mode = "rule" if llm_count <= 0 else "llm" if llm_count >= agent_count else "mix"
    return ManualConfigOptions(
        run_name=str(config.get("name", scenario_id)),
        selected_scenario_id=scenario_id,
        seed=_clamp_int(config.get("seed", 0), 0, 2_147_483_647),
        mode=mode,
        agent_count=agent_count,
        initial_food=_clamp_float(inventory.get("food", 5), 0, 1_000_000),
        initial_water=_clamp_float(inventory.get("water", 5), 0, 1_000_000),
        initial_materials=_clamp_float(inventory.get("construction_material", 5), 0, 1_000_000),
        initial_tools=_clamp_float(inventory.get("tools", 1), 0, 1_000_000),
        initial_habitats=_clamp_int(structures.get("habitat", 1), 0, 1000),
        initial_greenhouses=_clamp_int(structures.get("greenhouse", 1), 0, 1000),
        initial_solar_arrays=_clamp_int(structures.get("solar_array", 0), 0, 1000),
        initial_oxygen_plants=_clamp_int(structures.get("oxygen_plant", 0), 0, 1000),
        initial_water_extractors=_clamp_int(structures.get("water_extractor", 0), 0, 1000),
        initial_storage_depots=_clamp_int(structures.get("storage_depot", 0), 0, 1000),
        initial_weather_stations=_clamp_int(structures.get("weather_station", 0), 0, 1000),
        colony_start_x=_optional_int(colony.get("start_x")),
        colony_start_y=_optional_int(colony.get("start_y")),
        population_growth_enabled=_as_bool(population.get("enabled", True)),
        growth_probability=_clamp_float(population.get("prosperity_growth_scale", population.get("daily_spawn_probability", 0.03)), 0, 1),
        llm_count=llm_count,
        decision_mode=_normalize_choice(
            str(agents.get("decision_mode", "preferences")),
            DECISION_MODES,
            "preferences",
        ),
        decision_sampling=_normalize_choice(
            str(agents.get("decision_sampling", "softmax")),
            DECISION_SAMPLINGS,
            "softmax",
        ),
        individual_survival_priority_enabled=_as_bool(
            agents.get("individual_survival_priority_enabled", True)
        ),
        operational_range_m=operational_range_from_config(agents),
        cell_proposal_top_k=_clamp_int(
            agents.get("cell_proposal_top_k", 5), 1, 15
        ),
        role_distribution=_normalized_role_percentages(
            agents.get("role_distribution", DEFAULT_ROLE_DISTRIBUTION)
        ),
        role_preference_randomness=_clamp_float(
            agents.get("role_preference_randomness", 0.25), 0.0, 1.0
        ),
        max_llm_calls_per_step=_clamp_int(llm.get("max_calls_per_step", 5), 0, 100),
        llm_temperature=(None if llm.get("temperature") is None else float(llm["temperature"])),
        **_governors_from_config(config),
        fast_observation_enabled=_as_bool(headless.get("fast_observation", True)),
        psychosocial_enabled=_as_bool(model.get("psychosocial_enabled", False)),
        cell_degradation_enabled=_as_bool(world.get("cell_degradation", True)),
        # Il ripiego e' il default della dataclass, non uno zero scritto qui:
        # con lo zero, ricaricare un config salvato PRIMA che questa chiave
        # esistesse spegneva la conversione ISRU in silenzio — cioe' riportava
        # la run nella configurazione che si estingue verso il passo 350.
        isru_material_rate=float(
            _as_dict(config.get("colony")).get("isru_material_rate", ManualConfigOptions.isru_material_rate)
            or 0.0
        ),
        snapshot_interval=max(0, _clamp_int(headless.get("snapshot_interval", 0), 0, 1_000_000)),
        max_steps=_optional_int(config.get("days")),
        days_per_step=_clamp_int(simulation.get("days_per_step", 7), 1, 1_000_000),
        max_sim_days=_optional_int(simulation.get("max_days")),
        map_profile=_normalize_choice(str(world.get("map_profile", "balanced")), MAP_PROFILES, "balanced"),
        earth_mars_delay_minutes=_clamp_float(social.get("earth_mars_delay_minutes", 12), 0, 44),
        climate_scenario=_normalize_choice(str(climate.get("scenario", "climatology")), CLIMATE_SCENARIOS, "climatology"),
        environmental_layer_enabled=_as_bool(environmental.get("enabled", True)),
        extreme_events_enabled=_as_bool(config.get("extreme_events_enabled", extreme.get("enabled", True))),
        extreme_event_chance=_clamp_float(config.get("extreme_event_chance_per_step", extreme.get("chance_per_step", 0.1)), 0, 1),
        redistribution_enabled=_as_bool(redistribution.get("enabled", False)),
        redistribution_knapsack_water=_clamp_float(
            redistribution_knapsack.get("water", _REDISTRIBUTION_DEFAULT_KNAPSACK_WATER), 0, 1_000_000
        ),
        redistribution_knapsack_food=_clamp_float(
            redistribution_knapsack.get("food", _REDISTRIBUTION_DEFAULT_KNAPSACK_FOOD), 0, 1_000_000
        ),
        redistribution_knapsack_oxygen=_clamp_float(
            redistribution_knapsack.get("oxygen", _REDISTRIBUTION_DEFAULT_KNAPSACK_OXYGEN), 0, 1_000_000
        ),
        redistribution_flow_radius=_clamp_int(
            redistribution.get("flow_radius", _REDISTRIBUTION_DEFAULT_FLOW_RADIUS), 0, 1_000_000
        ),
        redistribution_flow_rate=_clamp_float(
            redistribution.get("flow_rate_per_step", _REDISTRIBUTION_DEFAULT_FLOW_RATE), 0, 1_000_000
        ),
        redistribution_greenhouse_per_capita=_clamp_float(
            redistribution.get("greenhouse_per_capita", _REDISTRIBUTION_DEFAULT_GREENHOUSE_PER_CAPITA), 0, 1_000_000
        ),
        assignments=list(agents.get("llm_assignments", [])) if isinstance(agents.get("llm_assignments"), list) else [],
    )


def _governors_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """I campi del governatore, o i default quando il blocco manca.

    Il blocco assente e' il caso normale e significa "nessun governatore":
    vanno restituiti i default, altrimenti rileggere uno scenario non governato
    lascerebbe in piedi il governatore di quello precedente. Un blocco del
    vecchio consiglio (con `count`/`mandates`) viene riletto per quello che il
    governatore unico ne usera': braccio, cadenza, attesa e la PRIMA
    assegnazione — `read_settings` avvisa a run del resto.
    """
    blocco = _as_dict(config.get("governors"))
    if not blocco:
        return {}
    prima = _as_dict((blocco.get("assignments") or [{}])[0]) if blocco.get("assignments") else {}
    return {
        "governors_arm": str(blocco.get("arm", "none")),
        "governors_cadence_steps": int(blocco.get("cadence_steps", 20) or 20),
        "governors_wait_seconds": float(blocco.get("wait_seconds", 0.0) or 0.0),
        "governors_provider": str(prima.get("provider", "fallback")),
        "governors_model": str(prima.get("model", "")),
        "governors_effort": str(prima.get("effort", "")),
        "governors_temperature": (None if prima.get("temperature") is None
                                  else float(prima["temperature"])),
        "governors_context_level": str(blocco.get("context_level", "completo")),
        "governors_thinking": str(prima.get("thinking", "off")),
        **_amministratori_from_config(blocco),
    }


def _blocco_amministratori(options) -> dict[str, Any]:
    """La sezione `administrators`, scritta da un solo posto."""
    blocco: dict[str, Any] = {
        "enabled": True,
        "cells_per_district": max(1, int(options.administrators_cells_per_district)),
        "follow_governor_arm": bool(options.administrators_follow_governor_arm),
    }
    if not options.administrators_follow_governor_arm:
        blocco["arm"] = str(options.administrators_arm or "llm")
        if blocco["arm"] == "llm" and options.administrators_provider != "fallback":
            blocco["assignment"] = {
                "provider": str(options.administrators_provider),
                "model": str(options.administrators_model or ""),
                "thinking": str(options.administrators_thinking or "off"),
            }
    voci = [v.strip() for v in str(options.administrators_models or "").split(",") if v.strip()]
    if voci:
        blocco["arm"] = "llm"
        blocco["follow_governor_arm"] = False
        blocco["assignments"] = []
        for voce in voci:
            provider, sep, modello = voce.partition(":")
            if sep and provider and modello:
                blocco["assignments"].append({
                    "provider": provider,
                    "model": modello,
                    "thinking": str(options.administrators_thinking or "off"),
                })
    return blocco


def _amministratori_from_config(blocco: dict[str, Any]) -> dict[str, Any]:
    sezione = _as_dict(blocco.get("administrators"))
    if not sezione:
        return {}
    assegnazione = _as_dict(sezione.get("assignment"))
    return {
        "administrators_enabled": bool(sezione.get("enabled", False)),
        "administrators_cells_per_district": int(sezione.get("cells_per_district", 3) or 3),
        "administrators_follow_governor_arm": bool(sezione.get("follow_governor_arm", True)),
        "administrators_arm": str(sezione.get("arm", "llm")),
        "administrators_provider": str(assegnazione.get("provider", "fallback")),
        "administrators_model": str(assegnazione.get("model", "")),
        "administrators_thinking": str(assegnazione.get("thinking", "off")),
        # La lista torna nella forma `provider:modello,provider:modello`.
        "administrators_models": ",".join(
            f"{a.get('provider', 'fallback')}:{a.get('model', '')}"
            for a in (sezione.get("assignments") or [])
            if isinstance(a, dict)
        ),
    }


def _scenario_base(scenario_id: str) -> dict[str, Any]:
    if not scenario_id:
        return {}
    config = get_standard_scenario_config(scenario_id)
    config["scenario_version"] = "standard-scenarios-v1"
    return config


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return minimum
    return min(maximum, max(minimum, number))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    return min(maximum, max(minimum, number))


def _clamp_agent_count(value: Any) -> int:
    return _clamp_int(value, 1, 10_000_000)


def _clamp_optional(value: Any, minimum: int, maximum: int) -> int | None:
    if value in {None, ""}:
        return None
    return _clamp_int(value, minimum, maximum)


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _effective_llm_count(mode: str, agent_count: int, llm_count: int) -> int:
    if mode == "rule":
        return 0
    if mode == "llm":
        return agent_count
    return _clamp_int(llm_count, 0, agent_count)


def _normalized_assignments(assignments: list[dict[str, str]], llm_count: int) -> list[dict[str, str]]:
    normalized = []
    for index in range(llm_count):
        assignment = assignments[index] if index < len(assignments) and isinstance(assignments[index], dict) else {}
        normalized.append({"provider": str(assignment.get("provider", "gpt")), "model": str(assignment.get("model", "fallback"))})
    return normalized


def _normalized_role_percentages(raw: Any) -> dict[str, float]:
    source = raw if isinstance(raw, dict) else DEFAULT_ROLE_DISTRIBUTION
    values = {role: max(0.0, float(source.get(role, 0.0))) for role in ROLE_ORDER}
    total = sum(values.values())
    if total <= 0.0:
        values = dict(DEFAULT_ROLE_DISTRIBUTION)
        total = sum(values.values())
    return {role: value / total * 100.0 for role, value in values.items()}


def _normalize_choice(value: str, choices: tuple[str, ...], fallback: str) -> str:
    return value if value in choices else fallback
