from __future__ import annotations

import math
import random

from .rule_based_agent import RuleBasedAgent
from .operational_range import (
    DEFAULT_OPERATIONAL_RANGE_M,
    clamp_operational_range_m,
    operational_range_from_config,
)
from .role_profiles import (
    apply_role_traits,
    assign_initial_roles,
    sample_role,
    sample_role_preferences,
    sample_role_skills,
)
from .build_policy import (
    POPOLAZIONE_PER_DADO_DI_NATALITA,
    local_life_support_capacity,
)
from src.world.colony_site import INITIAL_AGENTS_PER_CELL, colony_spawn_cells
from src.world.resources import ResourceBundle, scorta_colonia
from src.world.mars_geometry import MARS_MEAN_RADIUS_KM


NAMES = ["John", "Sarah", "Alex", "Elena", "Marcus", "Kira", "David", "Li", "Omar", "Sofia", "Victor", "Nia", "Hans", "Anya", "Zayn"]


def vision_radius_cells(radius_m: float, world_height: int) -> int:
    """Convert the supported 0.1-59 km vision range to grid rows."""
    radius_m = clamp_operational_range_m(radius_m)
    cell_side_m = (
        math.pi * MARS_MEAN_RADIUS_KM * 1000.0 / max(1, int(world_height))
    )
    return max(1, math.ceil(radius_m / cell_side_m))

def spawn_initial_agents(
    count: int,
    world,
    seed: int = 0,
    llm_count: int = 0,
    llm_provider=None,
    llm_providers: list | None = None,
    cost_tracker=None,
    initial_inventory: dict | None = None,
    start_x: int | None = None,
    start_y: int | None = None,
    total_population: int | None = None,
    max_agents_per_cell: int = INITIAL_AGENTS_PER_CELL,
    operational_range_m: float = DEFAULT_OPERATIONAL_RANGE_M,
    role_distribution: dict | None = None,
    role_preference_randomness: float = 0.25,
) -> dict[str, RuleBasedAgent]:
    rng = random.Random(seed)
    agents: dict[str, RuleBasedAgent] = {}
    name_usage: dict[str, int] = {}

    base_x = start_x if start_x is not None else world.width // 2
    base_y = start_y if start_y is not None else world.height // 2
    # total_population lets aggregate mode size the footprint on the real
    # colony population even when only a sample of agents is instantiated.
    spawn_cells = colony_spawn_cells(world, base_x, base_y, total_population or count, max_agents_per_cell)

    operational_range_m = clamp_operational_range_m(operational_range_m)
    assigned_roles = assign_initial_roles(count, role_distribution, seed)

    for idx in range(count):
        role = assigned_roles[idx]
        x, y = spawn_cells[idx % len(spawn_cells)]
        geom = world.get_cell(x, y).geometry
        local_x_m = rng.random() * float(geom.get("width_m", 1.0))
        local_y_m = rng.random() * float(geom.get("height_m", 1.0))
        
        agent_cls = RuleBasedAgent
        mode = "rule_based"
        provider_id = "Rule"
        
        if idx < llm_count:
            from .llm_agent import LLMAgent
            agent_cls = LLMAgent
            mode = "llm"
            provider = llm_providers[idx] if llm_providers and idx < len(llm_providers) else llm_provider
            provider_id = provider.provider_id.capitalize() if hasattr(provider, "provider_id") else "LLM"

        # Unique name generation: Provider_Name [Suffix]
        base_name = NAMES[idx % len(NAMES)]
        full_base = f"{provider_id}_{base_name}"
        
        count_used = name_usage.get(full_base, 0)
        if count_used == 0:
            agent_name = full_base
        else:
            # Suffix: A, B, C...
            suffix = chr(64 + count_used) # 65 is 'A'
            agent_name = f"{full_base} {suffix}"
        
        name_usage[full_base] = count_used + 1

        kwargs = {"provider": provider, "cost_tracker": cost_tracker} if idx < llm_count else {}
        agent = agent_cls(
            agent_id=f"agent_{idx:03d}",
            name=agent_name,
            role=role,
            x=x,
            y=y,
            perception_radius=vision_radius_cells(operational_range_m, world.height),
            perception_radius_m=operational_range_m,
            movement_distance_m_per_step=operational_range_m,
            local_x_m=local_x_m,
            local_y_m=local_y_m,
            mode=mode,
            **kwargs,
        )
        if initial_inventory:
            agent.inventory = ResourceBundle.from_dict({**agent.inventory.to_dict(), **initial_inventory})
        _assign_psychological_profile(agent, rng)
        apply_role_traits(agent)
        agent.pillar_preferences = tuple(
            sample_role_preferences(
                seed, idx, role, randomness=role_preference_randomness
            )
        )
        agent.pillar_skills = tuple(
            sample_role_skills(
                seed, idx, role, randomness=role_preference_randomness
            )
        )
        agents[agent.agent_id] = agent
        world.place_agent(agent.agent_id, x, y, local_x_m, local_y_m)
    world.metadata["_agent_id_counter"] = count
    return agents


#: Ogni quanti coloni la colonia tira un proprio dado di natalita'.
#:
#: **Prima la crescita era assoluta (2026-08-27).** `maybe_spawn_agent`
#: restituisce al piu' UN agente ed e' chiamato una volta per passo, quindi la
#: colonia guadagnava al massimo una persona alla settimana **qualunque fosse la
#: sua dimensione**: 52 nascite all'anno terrestre con cento abitanti come con
#: centomila. Misurato sulla configurazione stabile: 0,21 nascite per passo,
#: costanti, cioe' una crescita LINEARE di ~11 persone all'anno — il 9,2% annuo
#: a centoventi abitanti e lo 0,11% a diecimila. Una popolazione reale cresce in
#: proporzione a se' stessa.
#:
#: La costante vive in `build_policy` perche' la legge anche la riserva di
#: posti che rende una nascita possibile (`posti_di_riserva`): dichiararla in
#: due punti sarebbe la solita coppia di formule destinata a divergere.


#: Le risorse che un neonato porta con se': solo cio' che una persona consuma.
#:
#: **Perche' solo queste (2026-08-30).** Il corredo veniva dal valore di default
#: della dataclass: cinque unita' di materiale da costruzione, tre di minerali,
#: due di energia e due di kit medici, cioe' l'equipaggiamento di un colono
#: partito dalla Terra. Addosso a un bambino nato nella colonia non ha senso.
CORREDO_DEL_NEONATO = ("food", "water", "oxygen")


def _residenti(cell, agents: dict) -> list:
    """Gli oggetti agente che abitano la cella, in ordine deterministico.

    `cell.agents_present` contiene **identificativi**, non oggetti: leggerne
    `.inventory` direttamente restituisce silenziosamente nulla, ed e' un errore
    che questo modulo ha gia' commesso una volta. L'ordinamento per
    identificativo tiene ferma la riproducibilita'.
    """
    return [
        agents[identificativo]
        for identificativo in sorted(cell.agents_present, key=str)
        if identificativo in agents
    ]


def _disponibile_nella_cella(cell, agents: dict, nome: str) -> float:
    """Quanto la comunita' di quella cella possiede davvero di `nome`.

    Magazzino **piu'** le sacche di chi ci abita: e' la stessa definizione di
    «cio' che la colonia possiede» gia' usata dalla natalita' e dalle metriche
    (`world.resources.scorta_colonia`). Guardare il solo magazzino direbbe che
    una colonia non puo' nutrire un figlio mentre i genitori hanno le borse
    piene — e con la redistribuzione spenta e' esattamente cio' che accade.
    """
    totale = float(getattr(cell.resources, nome, 0.0))
    for residente in _residenti(cell, agents):
        totale += float(getattr(residente.inventory, nome, 0.0))
    return totale


def _preleva_dalla_cella(cell, agents: dict, nome: str, quantita: float) -> float:
    """Preleva dal magazzino e, se non basta, dalle sacche dei residenti.

    Ordine deterministico per identificativo: la riproducibilita' e' un
    invariante di questo progetto e non va spesa per una comodita'.
    """
    residuo = max(0.0, float(quantita))
    preso = 0.0
    disponibile = float(getattr(cell.resources, nome, 0.0))
    quota = min(disponibile, residuo)
    if quota > 0.0:
        setattr(cell.resources, nome, disponibile - quota)
        preso += quota
        residuo -= quota
    if residuo <= 1.0e-12:
        return preso
    for residente in _residenti(cell, agents):
        inventario = residente.inventory
        disponibile = float(getattr(inventario, nome, 0.0))
        quota = min(disponibile, residuo)
        if quota <= 0.0:
            continue
        setattr(inventario, nome, disponibile - quota)
        preso += quota
        residuo -= quota
        if residuo <= 1.0e-12:
            break
    return preso


def corredo_richiesto() -> dict[str, float]:
    """Quanto costa alla colonia mettere al mondo un colono.

    Letto dalla dataclass invece che riscritto altrove: due tabelle per la
    stessa grandezza sono il difetto ricorrente di questo modello.
    """
    modello = RuleBasedAgent(agent_id="_", name="_", role="colonist", x=0, y=0)
    tutto = modello.inventory.to_dict()
    return {nome: float(tutto.get(nome, 0.0)) for nome in CORREDO_DEL_NEONATO}


def _next_agent_index(agents: dict, world) -> int:
    """Monotonic agent index: never reuses the id of a living or dead agent."""
    counter = world.metadata.get("_agent_id_counter")
    if counter is None:
        counter = 0
        for agent_id in agents:
            try:
                counter = max(counter, int(str(agent_id).rsplit("_", 1)[-1]) + 1)
            except ValueError:
                continue
    counter = max(int(counter), len(agents))
    world.metadata["_agent_id_counter"] = counter + 1
    return counter


def _best_supported_birth_cell(world, agents: dict, start_x: int, start_y: int):
    """Choose an inhabited colony cell with a genuinely free support place.

    The maintained structure-position index keeps this proportional to the
    colony footprint, not to the full planetary grid.  Spare capacity is the
    minimum of housing and complete local life support, so births can use a
    healthy outpost without silently overfilling either system.
    """
    positions = set(getattr(world, "_structure_positions", set()) or set())
    positions.add((int(start_x), int(start_y)))
    corredo = corredo_richiesto()
    candidates = []
    for x, y in positions:
        cell = world.get_cell(int(x), int(y))
        if not cell.agents_present:
            continue
        population = len(cell.agents_present)
        capacity = min(
            float(cell.occupancy_capacity()),
            float(local_life_support_capacity(cell)),
        )
        spare = capacity - population
        if spare + 1.0e-9 < 1.0:
            continue
        # **Non si nasce dove la cella non puo' mantenere il nuovo venuto
        # (2026-08-30).** Da quando il corredo viene PRELEVATO invece che creato
        # dal nulla, una cella a scorte esaurite metterebbe al mondo coloni
        # senz'acqua, e con due soli passi di finestra letale morirebbero
        # subito. Il cancello e' la condizione fisica ovvia, e usa la stessa
        # tabella che poi viene prelevata.
        if any(
            _disponibile_nella_cella(cell, agents, nome) + 1.0e-9 < quantita
            for nome, quantita in corredo.items()
        ):
            continue
        wrapped_dx = min(abs(int(x) - int(start_x)), world.width - abs(int(x) - int(start_x)))
        distance = wrapped_dx + abs(int(y) - int(start_y))
        candidates.append(
            (spare, float(cell.habitability_score), -distance, -int(y), -int(x), cell)
        )
    return max(candidates, key=lambda row: row[:-1])[-1] if candidates else None


def maybe_spawn_agent(
    agents: dict,
    world,
    config: dict,
    day: int,
    seed: int = 0,
    operational_range_m: float = DEFAULT_OPERATIONAL_RANGE_M,
    tentativo: int = 0,
    _memo: dict | None = None,
):
    """Un dado di natalita'.

    `_memo` (2026-09-24) e' la memoria del passo passata da
    `maybe_spawn_agents`: cella di nascita, abitabilita' dell'impronta e
    probabilita' di crescita dipendono solo da coloni, inventari e strutture,
    che fra due dadi cambiano soltanto se il primo ha fatto nascere qualcuno
    (e allora la memoria viene svuotata). Senza memoria il comportamento e'
    quello di sempre. Misura: a 2000 coloni i dadi erano ~40 per passo e ognuno
    risommava l'inventario di tutta la colonia quattro volte.
    """
    memo = _memo if _memo is not None else {}
    population = config.get("population", {})
    if not population.get("enabled", False):
        return None
    # **Il tetto non e' piu' del modello (2026-09-01).** `max_agents` valeva
    # 100 per default e fermava OGNI run di riferimento a esattamente cento
    # coloni: non un esito della simulazione ma una censura sull'esito, e su di
    # essa poggiavano confronti di politiche che a quel punto misuravano il
    # tetto. Misurato: quattro run su sei del protocollo dei governatori
    # finivano a 100 esatti.
    #
    # A fermare la crescita bastano i vincoli che il modello ha gia', e sono
    # tutti locali: `_best_supported_birth_cell` richiede una cella con un posto
    # LIBERO fra quelli che il supporto vitale sostiene, e
    # `min_habitability_for_growth` richiede che l'impronta della colonia sia
    # abitabile. Una colonia cresce finche' costruisce, e smette quando smette:
    # e' la risposta giusta, e viene dal mondo invece che da una costante.
    #
    # La chiave resta LEGGIBILE per gli scenari di prova che vogliono una
    # popolazione bloccata, ma la sua ASSENZA ora significa «nessun limite»,
    # mentre prima significava «cinquanta».
    tetto = population.get("max_agents")
    if tetto is not None and len(agents) >= int(tetto):
        return None
    colony_cfg = config.get("colony", {}) if isinstance(config.get("colony"), dict) else {}
    x = min(world.width - 1, max(0, int(colony_cfg.get("start_x", world.width // 2))))
    y = min(world.height - 1, max(0, int(colony_cfg.get("start_y", world.height // 2))))
    if "cella" not in memo:
        memo["cella"] = _best_supported_birth_cell(world, agents, x, y)
    birth_cell = memo["cella"]
    if birth_cell is None:
        return None
    x, y = int(birth_cell.x), int(birth_cell.y)
    # `tentativo` entra nel seme perche' i dadi di uno stesso passo devono
    # essere indipendenti: senza, un primo esito negativo lascia `len(agents)`
    # invariato e tutti i dadi successivi ripeterebbero identico quel "no".
    # A `tentativo = 0` il seme e' quello di prima, bit per bit.
    rng = random.Random(seed + day + len(agents) + tentativo * 7919)
    if "crescita" not in memo:
        memo["crescita"] = _colony_growth_probability(agents, world, population)
    growth_probability = memo["crescita"]
    max_probability = float(population.get("daily_spawn_probability", population.get("growth_probability", 0.01)))
    # Both engines run this check once per simulation step, so the per-day
    # probability is compounded over the days covered by the step.
    daily_probability = min(max_probability, growth_probability)
    simulation_cfg = config.get("simulation", {}) if isinstance(config.get("simulation"), dict) else {}
    days_per_step = max(1.0, float(simulation_cfg.get("days_per_step", 1) or 1))
    step_probability = 1.0 - (1.0 - min(1.0, daily_probability)) ** days_per_step
    if rng.random() > step_probability:
        return None
    # Births depend on conditions where people actually live: the average
    # habitability of the colony footprint (cells with structures), not the
    # planet-wide mean, which stays ~0.004 for decades on early Mars.
    if "abitabilita" not in memo:
        memo["abitabilita"] = colony_local_habitability(world)
    if memo["abitabilita"] < float(population.get("min_habitability_for_growth", 0.02)):
        return None
    idx = _next_agent_index(agents, world)
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents"), dict) else {}
    role_distribution = agents_cfg.get("role_distribution")
    role_randomness = float(agents_cfg.get("role_preference_randomness", 0.25))
    role = sample_role(seed, idx, role_distribution)
    geom = world.get_cell(x, y).geometry
    local_x_m = rng.random() * float(geom.get("width_m", 1.0))
    local_y_m = rng.random() * float(geom.get("height_m", 1.0))
    range_m = operational_range_from_config(agents_cfg, operational_range_m)
    agent = RuleBasedAgent(
        agent_id=f"agent_{idx:03d}",
        name=f"Settler {idx}",
        role=role,
        x=x,
        y=y,
        perception_radius=vision_radius_cells(range_m, world.height),
        perception_radius_m=range_m,
        movement_distance_m_per_step=range_m,
        local_x_m=local_x_m,
        local_y_m=local_y_m,
    )
    _assign_psychological_profile(agent, rng)
    apply_role_traits(agent)
    agent.pillar_preferences = tuple(
        sample_role_preferences(seed, idx, role, randomness=role_randomness)
    )
    agent.pillar_skills = tuple(
        sample_role_skills(seed, idx, role, randomness=role_randomness)
    )
    # **Il corredo lo paga la cella dove il colono nasce (2026-08-30).**
    #
    # Era l'unica sorgente di massa non pagata del modello, e il bilancio la
    # dichiarava onestamente come «fuori dal passo (nascite, runner)». Misurata:
    # venticinque nascite in trecento passi creavano **575 unita' dal nulla**.
    # Ora la riga sparisce e il bilancio si chiude senza alcuna eccezione.
    #
    # **La conseguenza e' misurata e va conosciuta.** Il corredo creava anche
    # tre minerali e cinque unita' di materiale a testa, ed erano quelli a
    # finanziare le costruzioni: misurato, con la cella madre esaurita
    # (minerali 0,00 e materiale 0,00 dal passo 40) **nessun colono puo' piu'
    # permettersi una serra**, che costa un minerale e tre di materiale. Tolto
    # il sussidio, una colonia che ha esaurito il proprio giacimento non
    # costruisce piu'. Non e' un difetto di questa correzione: e' il modello che
    # non ha un canale di rifornimento, e ora si vede.
    cella_di_nascita = world.get_cell(x, y)
    for nome, richiesto in agent.inventory.to_dict().items():
        if nome not in CORREDO_DEL_NEONATO:
            setattr(agent.inventory, nome, 0.0)
            continue
        setattr(
            agent.inventory,
            nome,
            _preleva_dalla_cella(cella_di_nascita, agents, nome, float(richiesto)),
        )

    agents[agent.agent_id] = agent
    world.place_agent(agent.agent_id, x, y, local_x_m, local_y_m)
    world.log_event(
        "population_grew",
        f"{agent.name} added by internal colony prosperity",
        agent_id=agent.agent_id,
        x=x,
        y=y,
    )
    return agent


def maybe_spawn_agents(
    agents: dict,
    world,
    config: dict,
    day: int,
    seed: int = 0,
    operational_range_m: float = DEFAULT_OPERATIONAL_RANGE_M,
) -> list:
    """Le nascite di UN passo, in numero proporzionale alla popolazione.

    Tira un dado ogni `POPOLAZIONE_PER_DADO_DI_NATALITA` coloni invece di uno
    solo per l'intera colonia: vedi la nota sulla costante per il perche'. Ogni
    dado passa comunque per tutti i cancelli di `maybe_spawn_agent` — capienza
    di alloggio e di supporto vitale in testa — quindi una colonia satura non
    guadagna nulla dall'essere numerosa, che e' esattamente il comportamento
    voluto.

    Restituisce la lista dei nati, vuota quando non ne nasce nessuno.
    """
    population = config.get("population", {})
    if not population.get("enabled", False):
        return []
    dadi = max(1, round(len(agents) / POPOLAZIONE_PER_DADO_DI_NATALITA))
    nati = []
    memo: dict = {}
    for tentativo in range(dadi):
        nato = maybe_spawn_agent(
            agents, world, config, day, seed, operational_range_m, tentativo=tentativo,
            _memo=memo,
        )
        if nato is not None:
            nati.append(nato)
            # La nascita ha cambiato coloni, inventari e magazzino: tutto cio'
            # che la memoria teneva va ricalcolato.
            memo.clear()
    return nati


def colony_cells(world) -> list:
    """Le celle su cui la colonia insiste, cioe' quelle con strutture.

    Estratta da `colony_local_habitability` perche' serve anche alla
    prosperita': l'impronta della colonia e' una sola, e calcolarla in due modi
    sarebbe il solito difetto di questo modello — due formule per la stessa
    grandezza che smettono di coincidere senza che nessuno se ne accorga.
    Lista vuota quando non e' stato costruito ancora nulla.
    """
    cells = getattr(world, "_cells", None)
    if cells is not None and hasattr(cells, "structure_positions"):
        return [world.get_cell(x, y) for x, y in cells.structure_positions()]
    positions = getattr(world, "_structure_positions", None)
    if positions:
        candidate = [world.get_cell(x, y) for x, y in positions]
    else:
        candidate = [cell for row in world.cells for cell in row]
    return [cell for cell in candidate if cell.structures]


def colony_local_habitability(world) -> float:
    """Average habitability of the cells the colony occupies (cells with
    structures); falls back to the planetary mean when nothing is built."""
    structure_cells = colony_cells(world)
    if not structure_cells:
        return float(world.metrics().get("average_habitability", 0.0))
    return sum(
        cell.habitability_score for cell in structure_cells
    ) / len(structure_cells)


def _scorta(agents: dict, world, nome: str) -> float:
    """Cio' che la colonia possiede DAVVERO: sacche dei coloni piu' magazzini.

    **Prima si sommavano le sole sacche (2026-08-27).** Con la redistribuzione
    attiva ogni sacca viene riempita fino a un bersaglio fisso scritto nella
    configurazione, quindi `food_stock / (popolazione x 3)` valeva
    `2,0 / 3,0 = 0,667` SEMPRE — misurato 0,666 su una colonia stabile di
    centoventi coloni. La prosperita' non misurava la prosperita': misurava
    `redistribution.knapsack_targets.food`, e una colonia affamata riceveva la
    stessa identica natalita' di una ricca. Il magazzino della cella, dove la
    colonia tiene le riserve che contano, non entrava nel conto.
    """
    return scorta_colonia(
        (agent.inventory for agent in agents.values()), colony_cells(world), nome
    )


def _colony_growth_probability(agents: dict, world, population_cfg: dict) -> float:
    population = max(1, len(agents))
    food_stock = _scorta(agents, world, "food")
    water_stock = _scorta(agents, world, "water")
    materials_stock = _scorta(agents, world, "construction_material")
    tools_stock = _scorta(agents, world, "tools")
    structure_counts = world.structure_type_counts()
    greenhouse_count = structure_counts.get("greenhouse", 0)
    habitat_count = structure_counts.get("habitat", 0)
    prosperity = min(1.0, food_stock / (population * 3.0))
    prosperity = min(prosperity, water_stock / (population * 2.0))
    prosperity = min(prosperity, (materials_stock + tools_stock) / max(1.0, population * 1.5))
    prosperity = min(prosperity, (greenhouse_count + habitat_count * 0.6) / max(1.0, population * 0.25))
    base = float(population_cfg.get("prosperity_growth_scale", 0.02))
    return max(0.0, base * prosperity)


def _assign_psychological_profile(agent: RuleBasedAgent, rng: random.Random) -> None:
    """Stable individual priors inspired by isolated/confined mission analogues."""
    agent.cooperation = max(0.25, min(0.98, rng.gauss(0.70, 0.11)))
    agent.risk_tolerance = max(0.12, min(0.88, rng.gauss(agent.risk_tolerance, 0.12)))
    agent.curiosity = max(0.18, min(0.95, rng.gauss(agent.curiosity, 0.14)))
    agent.survival_priority = max(0.45, min(0.98, rng.gauss(agent.survival_priority, 0.10)))
    agent.protocol_compliance = max(0.45, min(0.99, rng.gauss(0.84, 0.10)))
    agent.autonomy_preference = max(0.15, min(0.92, rng.gauss(0.52, 0.14)))
    agent.morale = max(0.55, min(0.98, rng.gauss(0.82, 0.07)))
    agent.stress_index = max(0.04, min(0.35, rng.gauss(0.15, 0.06)))
