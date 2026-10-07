import numpy as np
from src.agents.base_agent import BaseAgent
from src.agents.vitals import (
    LETHAL_STEPS_WITHOUT_FOOD,
    LETHAL_STEPS_WITHOUT_WATER,
    PASSI_SINTOMO_SENZA_ACQUA,
    PASSI_SINTOMO_SENZA_CIBO,
)
from src.agents.vitals import tick_agent_vitals, tick_agent_psychosocial
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel_vitals import tick_vitals

VITAL_COLS = ("health", "oxygen", "hydration", "satiety", "fatigue")
OBJ_ATTR = {"oxygen": "oxygen_level"}


def _grid(width: int, height: int) -> GridWorld:
    # GridWorld has no width/height/seed convenience constructor (plain dataclass
    # requiring a pre-built cells grid) — build it the same way
    # tests/core/test_cell_arrays.py and tests/core/test_views.py do.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


#: **Il passo di questi test e' quello reale (2026-08-25).**
#:
#: Valevano 3650 giorni — un passo da dieci anni, eredita' della vecchia
#: normalizzazione `dt = dt_days/3650` che comprimeva qualunque durata in un
#: intervallo unitario. Da quando i vitali di privazione sono per passo, un
#: passo decennale non descrive piu' nulla: il colono muore di sete e di fame
#: al primo tick, e da li' in poi i due motori divergono per una ragione che
#: non c'entra con cio' che il test verifica (il kernel salta le righe morte,
#: il motore a oggetti continua a ticchettarle — differenza documentata e
#: voluta). Il passo settimanale e' quello che le run usano davvero.
PASSO_SETTIMANALE = 7.0


def _scenario():
    w = _grid(4, 3)
    home = w.get_cell(1, 1)
    home.structures.append(Structure(type=StructureType.OXYGEN_PLANT, x=1, y=1))
    home.structures.append(Structure(type=StructureType.GREENHOUSE, x=1, y=1))
    w.get_cell(3, 2).radiation_level = 1.4
    agents = {}
    rng = np.random.default_rng(0)
    for i, (x, y) in enumerate([(1, 1), (1, 1), (3, 2), (0, 0)]):
        a = BaseAgent(agent_id=f"a{i}", name=f"a{i}", role="colonist", x=x, y=y)
        # **Vitali alti e borracce piene (2026-08-25).** Questo scenario
        # confronta l'ARITMETICA dei due motori passo per passo: se un colono
        # muore a meta' confronto, cio' che si misura da li' in poi e' la
        # differenza documentata sul trattamento dei morti (il kernel salta le
        # righe non vive, il motore a oggetti continua a ticchettarle) e non
        # l'aritmetica. Con il consumo per passo un'idratazione di 0,2 non
        # sopravvive a un passo, ed e' giusto che sia cosi': la morte ha il suo
        # test dedicato piu' sotto.
        a.hydration = float(rng.uniform(0.7, 1.0))
        a.satiety = float(rng.uniform(0.7, 1.0))
        a.oxygen_level = float(rng.uniform(0.7, 1.0))
        a.fatigue = float(rng.uniform(0.0, 0.7))
        a.inventory.water = float(rng.uniform(2.0, 3.0))
        a.inventory.food = float(rng.uniform(2.0, 3.0))
        a.inventory.oxygen = float(rng.uniform(1.0, 2.0))
        agents[a.agent_id] = a
    return w, agents


def test_tick_vitals_matches_object_engine_step_by_step():
    w, agents = _scenario()
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    for _round in range(3):
        for a in agents.values():
            cell = w.get_cell(a.x, a.y)
            tick_agent_vitals(a, cell, dt)
        n = aa.n
        tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                    drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))
        for aid, a in agents.items():
            row = aa.index[aid]
            for col in VITAL_COLS:
                obj_val = getattr(a, OBJ_ATTR.get(col, col))
                assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                    f"{aid}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")
            assert aa.steps_without_water[row] == a.steps_without_water
            assert aa.steps_without_food[row] == a.steps_without_food


def test_tick_vitals_psychosocial_matches_object_engine():
    w, agents = _scenario()
    for a in agents.values():
        w.place_agent(a.agent_id, a.x, a.y)
    # Precondition: the neighborhood mix must exercise both signs of
    # isolation_pressure (vitals.py:228) — a crowded cell (nearby > 0) and
    # an isolated one (nearby == 0) — or a sign error there could pass unnoticed.
    nearby_counts = {
        aid: len(w.get_agents_in_range(a.x, a.y, 1)) - 1 for aid, a in agents.items()
    }
    assert any(n > 0 for n in nearby_counts.values()), "scenario needs an agent with neighbors"
    assert any(n == 0 for n in nearby_counts.values()), "scenario needs an isolated agent"

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    delay_minutes = 12.0
    config = {"social": {"earth_mars_delay_minutes": delay_minutes}}
    for _round in range(3):
        for a in agents.values():
            cell = w.get_cell(a.x, a.y)
            tick_agent_vitals(a, cell, dt)
            tick_agent_psychosocial(a, cell, w, dt, config=config)
        n = aa.n
        tick_vitals(aa, ca, dt, psychosocial=True, delay_minutes=delay_minutes,
                    drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))
        for aid, a in agents.items():
            row = aa.index[aid]
            for col in VITAL_COLS:
                obj_val = getattr(a, OBJ_ATTR.get(col, col))
                assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                    f"{aid}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")
            for col, obj_attr in (
                ("stress", "stress_index"),
                ("morale", "morale"),
                ("cooperation", "cooperation"),
                ("compliance", "protocol_compliance"),
            ):
                obj_val = getattr(a, obj_attr)
                assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                    f"{aid}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")


def test_lethal_clocks_kill_at_exact_steps():
    w = _grid(2, 2)
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    a.inventory.water = 0.0
    a.inventory.ice = 0.0
    a.inventory.food = 0.0
    aa = AgentArrays.from_agents({"a0": a})
    ca = CellArrays.from_world(w)
    deaths = []
    for step in range(1, 9):
        out = tick_vitals(aa, ca, PASSO_SETTIMANALE, psychosocial=False, delay_minutes=12.0,
                          drank_mask=np.zeros(1, bool), ate_mask=np.zeros(1, bool))
        deaths.extend(out["deaths"])
        if deaths:
            break
    assert deaths and deaths[0][1] == "dehydration"
    assert aa.steps_without_water[0] == LETHAL_STEPS_WITHOUT_WATER, (
        "letale ESATTAMENTE alla finestra dichiarata"
    )


def test_polar_branch_matches_object_engine_and_emits_memory_event():
    # vitals.py:134-144. polar_severity=0.6 with no structures on the cell
    # gives life_support=0, so exposure == polar_exposure == 0.6 >= 0.55:
    # this must also fire the "severe polar" memory event on both engines.
    w = _grid(2, 2)
    cell = w.get_cell(0, 0)
    cell.polar_severity = 0.6
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    agents = {"a0": a}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE

    tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)

    n = aa.n
    out = tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                       drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row = aa.index["a0"]
    for col in VITAL_COLS:
        obj_val = getattr(a, OBJ_ATTR.get(col, col))
        assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
            f"{col}: obj={obj_val} vec={getattr(aa, col)[row]}")

    expected_text = "Severe polar cold exposure: retreat or build life support before sustained work."
    assert expected_text in a.memory.recent_events
    assert (row, expected_text) in out["memory_events"]


def test_healing_bonus_from_infirmary_matches_object_engine():
    # vitals.py:198-206. healing_bonus is only ever non-zero via an INFIRMARY
    # structure's local_effect (structures.py), so this is the only way to
    # exercise the hb_mask branch.
    w = _grid(2, 2)
    cell = w.get_cell(0, 0)
    cell.structures.append(Structure(type=StructureType.INFIRMARY, x=0, y=0))
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    a.health = 0.5
    a.oxygen_level = 0.5
    a.hydration = 0.5
    a.fatigue = 0.5
    agents = {"a0": a}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE

    tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)

    n = aa.n
    tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row = aa.index["a0"]
    for col in VITAL_COLS:
        obj_val = getattr(a, OBJ_ATTR.get(col, col))
        assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
            f"{col}: obj={obj_val} vec={getattr(aa, col)[row]}")
    # sanity: the healing bonus actually fired this step (net of the small
    # radiation/oxygen/hydration drains also active this same step)
    assert a.health > 0.5


def test_ice_ration_fallback_matches_object_engine():
    # vitals.py:35-42 consume_water_ration: water==0 forces the water->ice
    # fallback branch (kernel_vitals.py:202 ice_ok).
    w = _grid(2, 2)
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    a.inventory.water = 0.0
    a.inventory.ice = 1.0
    agents = {"a0": a}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE

    tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)

    n = aa.n
    tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row = aa.index["a0"]
    for col in VITAL_COLS:
        obj_val = getattr(a, OBJ_ATTR.get(col, col))
        assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
            f"{col}: obj={obj_val} vec={getattr(aa, col)[row]}")
    assert np.isclose(a.inventory.ice, 0.9, atol=1e-5)
    assert np.isclose(float(aa.inv[row, C.R["ice"]]), 0.9, atol=1e-5)
    assert aa.steps_without_water[row] == 0 == a.steps_without_water


def test_starvation_clock_symptom_and_death_at_exact_steps():
    # Orologio del cibo, da capo a fondo: sintomo a
    # `PASSI_SINTOMO_SENZA_CIBO`, morte esattamente a
    # `LETHAL_STEPS_WITHOUT_FOOD`, con causa "starvation". Il colono porta
    # acqua in abbondanza perche' l'orologio della sete, piu' corto, non vinca
    # prima la gara della causa di morte.
    w = _grid(2, 2)
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    a.inventory.water = 3.0
    a.inventory.food = 0.0
    a.inventory.ice = 0.0
    agents = {"a0": a}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    obj_cell = w.get_cell(a.x, a.y)

    deaths: list = []
    for step in range(1, LETHAL_STEPS_WITHOUT_FOOD + 2):
        tick_agent_vitals(a, obj_cell, dt)
        n = aa.n
        out = tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                           drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))
        deaths.extend(out["deaths"])
        row = aa.index["a0"]
        for col in VITAL_COLS:
            obj_val = getattr(a, OBJ_ATTR.get(col, col))
            assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                f"step {step} {col}: obj={obj_val} vec={getattr(aa, col)[row]}")
        assert aa.steps_without_food[row] == a.steps_without_food
        assert aa.steps_without_water[row] == a.steps_without_water == 0
        if step == PASSI_SINTOMO_SENZA_CIBO:
            expected = (
                f"Starvation symptoms after {PASSI_SINTOMO_SENZA_CIBO} "
                "consecutive steps without food."
            )
            assert expected in a.memory.recent_events
            assert (row, expected) in out["memory_events"]
        if deaths:
            break

    assert a.steps_without_food == LETHAL_STEPS_WITHOUT_FOOD
    assert deaths and deaths[0] == (aa.index["a0"], "starvation")
    assert a.health <= 0.0


def test_multiple_deaths_in_same_tick_are_all_reported():
    # Two agents both hit hydration<=0 in the vitals phase of the SAME
    # tick_vitals call: both must be marked dead and both must appear
    # exactly once in "deaths" (no row double-reported, none dropped).
    w = _grid(2, 2)
    a1 = BaseAgent(agent_id="a1", name="a1", role="colonist", x=0, y=0)
    a1.hydration = 0.0
    a1.inventory.water = 0.0
    a1.inventory.ice = 0.0
    a2 = BaseAgent(agent_id="a2", name="a2", role="colonist", x=1, y=1)
    a2.hydration = 0.0
    a2.inventory.water = 0.0
    a2.inventory.ice = 0.0
    agents = {"a1": a1, "a2": a2}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    n = aa.n

    out = tick_vitals(aa, ca, dt, psychosocial=False, delay_minutes=12.0,
                       drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    rows_seen = [r for r, _ in out["deaths"]]
    assert len(rows_seen) == 2
    assert len(rows_seen) == len(set(rows_seen))  # no row reported twice
    death_rows = dict(out["deaths"])
    assert death_rows[aa.index["a1"]] == "dehydration"
    assert death_rows[aa.index["a2"]] == "dehydration"
    assert not aa.alive[aa.index["a1"]]
    assert not aa.alive[aa.index["a2"]]


def test_mixed_vitals_and_psychosocial_deaths_in_same_tick():
    # Minor 6 (review round 2): test_multiple_deaths_in_same_tick_are_all_reported
    # only exercises two vitals-phase deaths, never against the object engine.
    # The duplicate-row risk that Important 1's died_mask exclusion guards
    # against (kernel_vitals.py's post-psychosocial death sweep) is only real
    # when a vitals-phase death and a psychosocial-phase death coexist in the
    # SAME tick_vitals call. Reproduce exactly that: one agent dies from
    # dehydration in the vitals phase, a second dies from the high-stress
    # branch in the psychosocial phase, and a third survives — then compare
    # every column of all three against the object engine.
    # **Griglia 4x4 e non 2x2 (2026-08-31).** Su una 2x2 ogni colono e'
    # vicino di tutti gli altri, quindi `psy_death` riceveva compagnia e
    # la pressione da isolamento si spegneva: lo stress aggiornato si
    # fermava a 0,806, sotto la soglia di danno di 0,82. Qui i tre stanno
    # a distanza due l'uno dall'altro, cosi' la prova misura il ramo che
    # dichiara di misurare invece di dipendere dalla taglia della griglia.
    w = _grid(4, 4)
    w.get_cell(2, 0).radiation_level = 0.0  # keep psy_death's vitals-phase health drain at ~0
    a_vit = BaseAgent(agent_id="vit_death", name="vit_death", role="colonist", x=0, y=0)
    a_vit.hydration = 0.0
    # Borraccia vuota: la razione personale ora reidrata davvero.
    a_vit.inventory.water = 0.0
    a_vit.inventory.ice = 0.0
    w.get_cell(2, 0).dust_level = 1.0
    a_psy = BaseAgent(agent_id="psy_death", name="psy_death", role="colonist", x=2, y=0)
    a_psy.health = 0.01
    # **Perche' lo stress qui e' 1,0 e la polvere e' al massimo
    # (2026-08-31).** Da quando lo strato ha un punto fisso, uno stress
    # imposto non resta dov'e' stato messo: il richiamo verso la linea di
    # base (0,15) toglie 0,2395 in un passo. Perche' il ramo di danno sopra
    # 0,82 scatti davvero servono le spinte, non l'assegnazione: colono
    # solo (0,045) e polvere piena (0,035) portano il valore aggiornato a
    # 0,855. La polvere e' scelta apposta al posto della radiazione perche'
    # non tocca la salute nella fase dei vitali, che questa prova vuole
    # tenere ferma.
    a_psy.stress_index = 1.0
    a_surv = BaseAgent(agent_id="survivor", name="survivor", role="colonist", x=0, y=2)
    agents = {"vit_death": a_vit, "psy_death": a_psy, "survivor": a_surv}
    for a in agents.values():
        w.place_agent(a.agent_id, a.x, a.y)

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    delay_minutes = 12.0
    config = {"social": {"earth_mars_delay_minutes": delay_minutes}}

    # object reference, mirroring tick_all_agents ordering: vitals for every
    # agent, THEN psychosocial for every agent, with no death sweep between.
    for a in agents.values():
        tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)
    assert a_vit.health <= 0.0  # sanity: really dies in the vitals phase
    assert a_psy.health > 0.0  # sanity: still alive after the vitals phase alone
    for a in agents.values():
        tick_agent_psychosocial(a, w.get_cell(a.x, a.y), w, dt, config=config)
    assert a_psy.health <= 0.0  # sanity: dies via the psychosocial high-stress branch
    assert a_surv.health > 0.0  # sanity: survivor stays alive throughout

    n = aa.n
    out = tick_vitals(aa, ca, dt, psychosocial=True, delay_minutes=delay_minutes,
                       drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row_vit = aa.index["vit_death"]
    row_psy = aa.index["psy_death"]
    row_surv = aa.index["survivor"]

    rows_seen = [r for r, _ in out["deaths"]]
    assert len(rows_seen) == 2
    assert len(rows_seen) == len(set(rows_seen))  # no row reported twice, none dropped
    death_rows = dict(out["deaths"])
    assert death_rows[row_vit] == "dehydration"
    assert death_rows[row_psy] == "health_collapse"
    assert not aa.alive[row_vit]
    assert not aa.alive[row_psy]
    assert aa.alive[row_surv]

    for row, a in ((row_vit, a_vit), (row_psy, a_psy), (row_surv, a_surv)):
        for col in VITAL_COLS:
            obj_val = getattr(a, OBJ_ATTR.get(col, col))
            assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                f"{a.agent_id}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")
        for col, obj_attr in (
            ("stress", "stress_index"),
            ("morale", "morale"),
            ("cooperation", "cooperation"),
            ("compliance", "protocol_compliance"),
        ):
            obj_val = getattr(a, obj_attr)
            assert np.isclose(getattr(aa, col)[row], obj_val, atol=1e-5), (
                f"{a.agent_id}.{col}: obj={obj_val} vec={getattr(aa, col)[row]}")


def test_nearby_counts_agent_that_dies_this_tick():
    # Finding 1: the object pipeline only removes a dead agent from the world
    # AFTER tick_all_agents finishes both phases for every agent, so a
    # neighbor's psychosocial "nearby" this same tick must still count an
    # agent that dies in this tick's vitals phase. Reproduce that ordering
    # on the object side (no world.remove_agent call between the two
    # phases) and assert the survivor's psychosocial columns match.
    w = _grid(2, 2)
    a_dying = BaseAgent(agent_id="dying", name="dying", role="colonist", x=0, y=0)
    a_dying.hydration = 0.0
    # Borraccia vuota: la razione personale ora reidrata davvero.
    a_dying.inventory.water = 0.0
    a_dying.inventory.ice = 0.0
    a_survivor = BaseAgent(agent_id="survivor", name="survivor", role="colonist", x=0, y=0)
    agents = {"dying": a_dying, "survivor": a_survivor}
    for a in agents.values():
        w.place_agent(a.agent_id, a.x, a.y)
    # sanity: the survivor's pre-tick neighborhood includes the about-to-die agent
    assert len(w.get_agents_in_range(a_survivor.x, a_survivor.y, 1)) - 1 == 1

    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    delay_minutes = 12.0
    config = {"social": {"earth_mars_delay_minutes": delay_minutes}}

    # object reference, mirroring agent_coupled_runner.py: tick_all_agents
    # runs vitals+psychosocial for every agent, and only THEN (outside this
    # scope) does the runner sweep health<=0 and call world.remove_agent.
    for a in agents.values():
        tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)
    assert a_dying.health <= 0.0  # sanity: really dies in the vitals phase
    for a in agents.values():
        tick_agent_psychosocial(a, w.get_cell(a.x, a.y), w, dt, config=config)

    n = aa.n
    out = tick_vitals(aa, ca, dt, psychosocial=True, delay_minutes=delay_minutes,
                       drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row_dying = aa.index["dying"]
    row_survivor = aa.index["survivor"]
    death_rows = dict(out["deaths"])
    assert death_rows.get(row_dying) == "dehydration"
    assert not aa.alive[row_dying]
    assert aa.alive[row_survivor]

    for col, obj_attr in (
        ("stress", "stress_index"),
        ("morale", "morale"),
        ("cooperation", "cooperation"),
        ("compliance", "protocol_compliance"),
    ):
        obj_val = getattr(a_survivor, obj_attr)
        assert np.isclose(getattr(aa, col)[row_survivor], obj_val, atol=1e-5), (
            f"survivor.{col}: obj={obj_val} vec={getattr(aa, col)[row_survivor]}")

    # Important 1 (review round 2): tick_all_agents (vitals.py:241-252) applies
    # tick_agent_psychosocial to EVERY agent it is given, including one that
    # already died in this same tick's vitals phase (world.remove_agent only
    # runs after the whole tick) — and that post-psychosocial state is what
    # agent.to_dict() later captures into dead_agents.jsonl. So a_dying, who
    # died above in the vitals phase, must ALSO have received the
    # psychosocial update on the object side (it did, via the loop above) and
    # the vectorized kernel must match it here. Before the Important-1 fix
    # this failed: the vectorized kernel excluded already-dead rows from the
    # psychosocial update (rows_p = agents.alive_rows() re-read after the
    # vitals-phase death sweep), leaving aa.stress[row_dying] frozen at its
    # pre-tick default instead of the post-psychosocial value.
    for col, obj_attr in (
        ("stress", "stress_index"),
        ("morale", "morale"),
        ("cooperation", "cooperation"),
        ("compliance", "protocol_compliance"),
    ):
        obj_val = getattr(a_dying, obj_attr)
        assert np.isclose(getattr(aa, col)[row_dying], obj_val, atol=1e-5), (
            f"dying.{col}: obj={obj_val} vec={getattr(aa, col)[row_dying]}")


def test_high_stress_death_in_psychosocial_phase_reported_same_tick():
    # Finding 2: vitals.py:236-238 can drain health to 0 in the psychosocial
    # phase itself. The object pipeline sweeps health<=0 once, AFTER both
    # tick_agent_vitals and tick_agent_psychosocial have run for every agent
    # this tick — so this death must be reported in THIS SAME tick_vitals
    # call's "deaths", not one tick later. (Also exercises a death_cause
    # other than "dehydration": hydration/satiety/oxygen are all healthy
    # here, so the cause must resolve to "health_collapse".)
    w = _grid(2, 2)
    cell = w.get_cell(0, 0)
    cell.radiation_level = 0.0  # keep the vitals-phase health drain at ~0
    cell.dust_level = 1.0
    a = BaseAgent(agent_id="a0", name="a0", role="colonist", x=0, y=0)
    a.health = 0.01
    # **Perche' lo stress qui e' 1,0 e la polvere e' al massimo
    # (2026-08-31).** Da quando lo strato ha un punto fisso, uno stress
    # imposto non resta dov'e' stato messo: il richiamo verso la linea di
    # base (0,15) toglie 0,2395 in un passo. Perche' il ramo di danno sopra
    # 0,82 scatti davvero servono le spinte, non l'assegnazione: colono
    # solo (0,045) e polvere piena (0,035) portano il valore aggiornato a
    # 0,855. La polvere e' scelta apposta al posto della radiazione perche'
    # non tocca la salute nella fase dei vitali, che questa prova vuole
    # tenere ferma.
    a.stress_index = 1.0
    w.place_agent(a.agent_id, a.x, a.y)
    agents = {"a0": a}
    aa = AgentArrays.from_agents(agents)
    ca = CellArrays.from_world(w)
    dt = PASSO_SETTIMANALE
    delay_minutes = 12.0
    config = {"social": {"earth_mars_delay_minutes": delay_minutes}}

    tick_agent_vitals(a, w.get_cell(a.x, a.y), dt)
    assert a.health > 0.0  # sanity: still alive after the vitals phase alone
    tick_agent_psychosocial(a, w.get_cell(a.x, a.y), w, dt, config=config)
    assert a.health <= 0.0  # sanity: this scenario is designed to kill via psychosocial

    n = aa.n
    out = tick_vitals(aa, ca, dt, psychosocial=True, delay_minutes=delay_minutes,
                       drank_mask=np.zeros(n, bool), ate_mask=np.zeros(n, bool))

    row = aa.index["a0"]
    assert not aa.alive[row]
    assert out["deaths"] == [(row, "health_collapse")]
    # Minor 7 (review round 2): numerically compare the post-psychosocial
    # health/stress against the object engine, not just alive/deaths — the
    # object values are already available above (a.health, a.stress_index).
    assert np.isclose(aa.health[row], a.health, atol=1e-5), (
        f"health: obj={a.health} vec={aa.health[row]}")
    assert np.isclose(aa.stress[row], a.stress_index, atol=1e-5), (
        f"stress: obj={a.stress_index} vec={aa.stress[row]}")
