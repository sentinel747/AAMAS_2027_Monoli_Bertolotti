"""La contesa sulla raccolta da pool condiviso di cella non deve produrre
proposte condannate in partenza.

Tutti gli agenti decidono sullo stesso snapshot del mondo: senza coordinamento
uno stock sottile (il gocciolio di materiali dei depositi) attira l'intera
folla e solo il primo esecutore riesce — nella run 120726_testlogic2 il 94%
dei rifiuti (19.9k su 21.2k) era `collect_materials: no materials found`.
Il registro di prenotazioni intra-step ferma le proposte quando lo stock
visibile e' gia' impegnato; un singolo rifiuto residuo mette la raccolta in
cooldown per l'intera finestra.
"""
from __future__ import annotations

from src.agents.action_space import ActionType
from src.agents.population import spawn_initial_agents
from src.agents.rule_based_agent import (
    BUILD_REJECTION_THRESHOLD,
    COLLECT_REJECTION_THRESHOLD,
)
from src.world.world_generator import WorldGenerator


def _setup(pool_materials: float, pool_minerals: float, n_agents: int = 2):
    world = WorldGenerator(seed=7).generate(10, 10)
    world.step = 3
    agents = list(spawn_initial_agents(n_agents, world, seed=7, start_x=5, start_y=5).values())
    cell = world.get_cell(5, 5)
    cell.resources.construction_material = pool_materials
    cell.resources.minerals = pool_minerals
    for agent in agents:
        # Vitali pieni e inventario scarico: l'unico ramo attivo e' la raccolta.
        agent.satiety = 1.0
        agent.hydration = 1.0
        agent.inventory.ice = 5.0
        agent.inventory.construction_material = 0.0
        agent.inventory.minerals = 5.0
    return world, cell, agents


def test_second_collector_same_step_does_not_propose_doomed_grab():
    world, cell, (first, second) = _setup(pool_materials=1.0, pool_minerals=0.0)

    a1 = first._resource_action(cell, existing=set(), world=world)
    a2 = second._resource_action(cell, existing=set(), world=world)

    assert a1 is not None and a1.action == ActionType.COLLECT_MATERIALS
    assert a2 is None  # pool da 1.0 gia' interamente prenotato dal primo


def test_claim_ledger_resets_when_step_advances():
    world, cell, (first, second) = _setup(pool_materials=1.0, pool_minerals=0.0)

    assert first._resource_action(cell, existing=set(), world=world).action == ActionType.COLLECT_MATERIALS
    world.step += 1
    a2 = second._resource_action(cell, existing=set(), world=world)
    assert a2 is not None and a2.action == ActionType.COLLECT_MATERIALS


def test_larger_pool_admits_exactly_as_many_collectors_as_it_can_serve():
    world, cell, agents = _setup(pool_materials=2.0, pool_minerals=0.0, n_agents=3)

    proposals = [a._resource_action(cell, existing=set(), world=world) for a in agents]
    kinds = [p.action if p else None for p in proposals]
    # 2.0 unita' = due prese piene (1.0); il terzo trova tutto gia' prenotato.
    assert kinds[:2] == [ActionType.COLLECT_MATERIALS, ActionType.COLLECT_MATERIALS]
    assert kinds[2] is None


def test_mineral_branch_respects_claims_too():
    world, cell, (first, second) = _setup(pool_materials=0.0, pool_minerals=1.0)

    a1 = first._resource_action(cell, existing=set(), world=world)
    a2 = second._resource_action(cell, existing=set(), world=world)

    assert a1 is not None and a1.action == ActionType.COLLECT_MINERALS
    assert a2 is None


def test_without_world_falls_back_to_plain_threshold():
    world, cell, (first, second) = _setup(pool_materials=1.0, pool_minerals=0.0)

    a1 = first._resource_action(cell, existing=set(), world=None)
    a2 = second._resource_action(cell, existing=set(), world=None)

    assert a1 is not None and a1.action == ActionType.COLLECT_MATERIALS
    assert a2 is not None and a2.action == ActionType.COLLECT_MATERIALS


def test_single_rejection_puts_collect_on_cooldown_but_not_builds():
    world, cell, (agent, _) = _setup(pool_materials=1.0, pool_minerals=0.0)
    agent.memory.add_failure(ActionType.COLLECT_MATERIALS.value, None, step=int(world.step))

    assert agent._recently_rejected(ActionType.COLLECT_MATERIALS, world, COLLECT_REJECTION_THRESHOLD)
    # La soglia dei build resta a 2: un singolo rifiuto non blocca il piano.
    assert not agent._recently_rejected(ActionType.COLLECT_MATERIALS, world, BUILD_REJECTION_THRESHOLD)
    assert agent._resource_action(cell, existing=set(), world=world) is None

    world.step += 20  # fuori dalla finestra di 8 step
    action = agent._resource_action(cell, existing=set(), world=world)
    assert action is not None and action.action == ActionType.COLLECT_MATERIALS


def _preference_collectors(quota_value: int, n_agents: int = 5, stock: float = 1.0) -> int:
    """Quanti agenti scelgono `collect_materials` nello stesso passo, in coda al
    percorso a preferenze, con la quota di cella fissata a `quota_value`."""
    import numpy as np

    from src.agents import pillars
    from src.agents.preference_agent import decide_preferences

    world, _cell, agents = _setup(pool_materials=stock, pool_minerals=0.0, n_agents=n_agents)
    priority = np.ones(pillars.N_ACTIONS, dtype=np.float64)
    # La spinta di una direttiva: senza, gli agenti si distribuiscono e la
    # contesa non si vede.
    priority[pillars.ACTION_INDEX[ActionType.COLLECT_MATERIALS]] = 50.0
    quota = np.full(pillars.N_ACTIONS, int(quota_value), dtype=np.int32)
    claims: dict = {}
    scelte = [
        decide_preferences(
            agent,
            world,
            np.ones(pillars.N_ACTIONS, dtype=np.bool_),
            0.0,
            "greedy",
            claims,
            int(world.step),
            action_priority_row=priority,
            quota_row=quota,
        )
        for agent in agents
    ]
    return sum(
        1
        for scelta in scelte
        if scelta is not None and scelta.action == ActionType.COLLECT_MATERIALS
    )


def test_a_quota_of_minus_one_lifts_the_headcount_cap_not_the_stock():
    """`-1` significa "nessun tetto di teste", non "nessuna prenotazione".

    E' l'unico valore di quota che un braccio di controllo non emette mai
    (`RandomProposer` estrae da 1 in su, `ScriptedProposer` parte dal pavimento
    della cella) e che il prompt del braccio LLM invece offre esplicitamente. Con
    `-1`, `_claim_cap` restituisce `None` e `_record_claim` usciva prima di
    iscrivere la risorsa: il registro delle prenotazioni restava vuoto, la
    seconda guardia di `_apply_claim_limits` non si chiudeva mai e l'intera
    folla proponeva la raccolta su uno stock che ne serve uno. Misurato in run
    vere: 25.168 `collect_materials` rifiutate con "no materials found in this
    cell" su un solo seme, e nessun rifiuto negli altri bracci.

    Il tetto NUMERICO, anche altissimo, non ha mai avuto il problema: e' la sola
    scorciatoia del `None` a saltare la contabilita'.
    """
    assert _preference_collectors(32) == 1, "il riferimento che gia' funziona"
    assert _preference_collectors(-1) == 1
