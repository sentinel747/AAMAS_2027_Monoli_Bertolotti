"""Lo scenario come variabile, invece di un solo mondo abbondante.

Tutta la campagna sui governatori fin qui e' girata su `balanced`, e la' non c'e'
niente di scarso: su quindici semi e quattro bracci il minimo di `food_margin`
non scende mai sotto 1,06 e l'integrita' media delle strutture non sotto 0,88.
L'unico braccio che stacca `material_margin` dal proprio tetto e' `llm` (minimo
mediano 1,35 contro 2,00), ma nemmeno quello arriva a mordere. Vedi
`docs/benchmarks/2026-08-21-margini-e-scenari.md`.

Questi test tengono ferme le due cose che rendono confrontabile una campagna per
scenario: che il profilo arrivi davvero alla config, e che non passarlo lasci la
baseline dov'era.
"""

import csv

import pytest

from scripts.parity_harness import realistic_config
from scripts.run_governor_experiment import _margini
from src.world.world_generator import MAP_PROFILES, WorldGenerator


def test_the_default_config_is_still_balanced():
    """Il default non deve muoversi: su di esso poggia il digest bit-exact.

    `realistic_config` e' la configurazione di riferimento della parita'. Se
    aggiungere il parametro ne avesse cambiato il valore predefinito, ogni
    confronto storico sarebbe scivolato in silenzio.
    """
    assert realistic_config(50, 10, 0)["world"]["map_profile"] == "balanced"


def test_passing_the_default_explicitly_changes_nothing():
    assert realistic_config(50, 10, 0) == realistic_config(50, 10, 0, map_profile="balanced")


@pytest.mark.parametrize("profilo", [p for p in MAP_PROFILES if p != "random"])
def test_the_profile_reaches_the_config(profilo):
    assert realistic_config(50, 10, 0, map_profile=profilo)["world"]["map_profile"] == profilo


def test_scarcity_is_a_real_lever_and_not_a_label():
    """Il profilo scarso deve davvero togliere minerali, non solo cambiare nome.

    Il fattore dichiarato nelle impostazioni e' 0,55 sul numero di centri, ma si
    compone con `mineral_scale` (8,0 -> 4,6): l'effetto sul totale e' di un
    ordine di grandezza, non del 45%. Un test che si accontentasse di
    `!= balanced` lascerebbe passare una campagna in cui lo scenario non morde,
    che e' esattamente l'errore che questi profili servono a evitare.
    """
    def minerali(profilo):
        mondo = WorldGenerator(0, map_profile=profilo).generate(120, 120)
        return sum(c.resources.minerals for riga in mondo.cells for c in riga)

    assert minerali("scarce_resources") < minerali("balanced") / 5
    assert minerali("mineral_rich") > minerali("balanced") * 3


def test_hazard_raises_radiation_which_is_what_wears_structures():
    """`high_hazard` deve alzare la radiazione: e' il canale che consuma.

    La radiazione entra in `rad_wear` (kernel_biology) e nell'usura degli agenti
    (kernel_vitals). Senza questo controllo, "scenario ostile" resterebbe
    un'etichetta e la manutenzione continuerebbe a non avere nulla da riparare.
    """
    def radiazione(profilo):
        mondo = WorldGenerator(0, map_profile=profilo).generate(120, 120)
        celle = [c for riga in mondo.cells for c in riga]
        return sum(c.radiation_level for c in celle) / len(celle)

    assert radiazione("high_hazard") > radiazione("balanced") * 1.1


def _timeseries(tmp_path, righe):
    percorso = tmp_path / "state_timeseries.csv"
    with percorso.open("w", encoding="utf-8", newline="") as handle:
        scrittore = csv.DictWriter(handle, fieldnames=["step", "food_margin", "material_margin"])
        scrittore.writeheader()
        for riga in righe:
            scrittore.writerow(riga)
    return tmp_path


def test_the_margin_recorded_is_the_minimum_not_the_last_value(tmp_path):
    """Una colonia che ha sfiorato la carestia e si e' ripresa deve risultarlo.

    I due margini sono tosati in alto a 2, quindi il valore finale di una run
    che e' scesa a 0,9 a meta' strada e poi e' risalita e' 2,00 — cioe'
    indistinguibile da una run che non ha mai avuto un problema.
    """
    cartella = _timeseries(tmp_path, [
        {"step": 1, "food_margin": 2.0, "material_margin": 2.0},
        {"step": 2, "food_margin": 1.1, "material_margin": 0.9},
        {"step": 3, "food_margin": 2.0, "material_margin": 2.0},
    ])
    assert _margini(cartella) == {"food_margin_min": 1.1, "material_margin_min": 0.9}


def test_a_run_without_timeseries_reports_nothing_instead_of_zero(tmp_path):
    """Assente e' diverso da zero: uno zero finirebbe nella colonna 'carestia'."""
    assert _margini(tmp_path) == {}
    assert _margini(None) == {}


def test_the_worst_seed_decides_the_verdict_not_the_median(tmp_path):
    """Basta che la colonia soffra su UNA run perche' ci sia da misurare.

    Con la mediana, una configurazione che uccide su un seme su tre risulterebbe
    innocua — cioe' verrebbe scartato proprio il caso in cui una politica di
    allocazione ha qualcosa da decidere.
    """
    from scripts.valuta_scenari import valuta

    esito = valuta([
        (tmp_path, {"acceptance_rate": 1.0, "deaths": 0, "structure_integrity_mean": 0.99}),
        (tmp_path, {"acceptance_rate": 1.0, "deaths": 4, "structure_integrity_mean": 0.99}),
        (tmp_path, {"acceptance_rate": 1.0, "deaths": 0, "structure_integrity_mean": 0.99}),
    ])
    assert esito["sotto_pressione"]
    assert esito["peggiori"]["morti"] == 4


def test_a_colony_that_gets_everything_it_wants_is_not_under_pressure(tmp_path):
    """Il criterio deve saper dire di no, altrimenti non e' un criterio.

    Questi sono i valori misurati su tutti e cinque i profili di mappa: se li
    ammettesse, ammetterebbe qualunque cosa.
    """
    from scripts.valuta_scenari import valuta

    esito = valuta([
        (tmp_path, {"acceptance_rate": 1.0, "deaths": 0, "structure_integrity_mean": 0.995})
    ])
    assert not esito["sotto_pressione"]
    assert esito["superate"] == []


def test_the_deficit_column_reports_the_worst_cell_not_the_luckiest(tmp_path):
    """Il deficit non fa da cancello ma resta un "peggiore = piu' grande".

    Toltolo dalle condizioni, il verso di aggregazione predefinito lo trattava
    come le altre — cioe' come minimo — e la colonna mostrava il satellite piu'
    fortunato della campagna invece del piu' sofferente.
    """
    import json

    from scripts.valuta_scenari import valuta

    def cartella(nome, deficit):
        d = tmp_path / nome
        d.mkdir()
        (d / "cell_infrastructure.json").write_text(
            json.dumps([{"x": 0, "y": 0, "population": 300, "deficits": {"housing": deficit}}]),
            encoding="utf-8",
        )
        return d

    esito = valuta([
        (cartella("a", 0.0), {"acceptance_rate": 1.0, "deaths": 0}),
        (cartella("b", 7.0), {"acceptance_rate": 1.0, "deaths": 0}),
    ])
    assert esito["peggiori"]["deficit_max"] == 7.0
    # ...e resta comunque fuori dal verdetto
    assert not esito["sotto_pressione"]


def test_l_arnese_di_verifica_usa_il_default_del_prodotto():
    """Le verifiche devono girare sulla configurazione che le run usano davvero.

    `realistic_config` alimenta i benchmark, l'audit di conservazione della
    massa e l'arnese di parita'. Teneva `isru = 0.0` scritto nel proprio corpo
    mentre `ManualConfigOptions` lo tiene a 1,0: ogni verifica girava percio'
    sul ramo SPENTO di cio' che decide se la colonia sopravvive, e il difetto
    che quel ramo copre — il materiale come dotazione non rinnovabile, e la
    morte per manutenzione non pagata al passo ~350 — non poteva apparire in
    nessuna misura di controllo. Due default per una quantita' sola.
    """
    from src.experiments.manual_config import ManualConfigOptions
    from scripts.parity_harness import realistic_config

    atteso = float(ManualConfigOptions.isru_material_rate)
    assert atteso > 0.0, "sanita': il prodotto tiene la conversione accesa"
    assert realistic_config(50, 10, 0)["colony"]["isru_material_rate"] == atteso
    assert realistic_config(50, 10, 0) == realistic_config(50, 10, 0, isru=atteso)


def test_the_isru_conversion_reaches_the_config():
    from scripts.parity_harness import realistic_config

    assert realistic_config(50, 10, 0, isru=5.0)["colony"]["isru_material_rate"] == 5.0


def test_la_conversione_rende_uguale_in_una_cella_con_e_senza_giacimento():
    """La regola di costruzione deve essere la stessa in ogni cella colonizzata.

    **Cosa sorvegliava il test precedente, e perche' non serve piu'.** Fino al
    2026-08-31 l'ISRU consumava i `minerals` della cella e il test verificava
    che non ne consumasse piu' di quanti ce ne fossero. Quell'invariante non
    esiste piu': la conversione lavora il REGOLITO, che e' la superficie e non
    un giacimento. (Il vecchio test riscriveva `np.minimum` nel proprio corpo e
    verificava se stesso, non il motore: qui si chiama `update_cells`.)

    **Cosa sorveglia questo.** Due celle identiche per parco impianti, una su un
    giacimento e una senza, producono lo STESSO materiale, e nessuna delle due
    perde minerali. E' il requisito esplicito del modello: un colono che
    colonizza una cella qualunque deve poter fare cio' che farebbe nella cella
    madre. Legata ai minerali, la conversione era inerte proprio nel sito della
    colonia (misurato: `minerals = 0,00`, e solo il 6,4% della mappa ha un
    giacimento).
    """
    import numpy as np
    from src.agents.base_agent import BaseAgent
    from src.core import constants as C
    from src.core.arrays import AgentArrays, CellArrays
    from src.core.kernel_biology import update_cells
    from src.world.cell import Cell
    from src.world.grid import GridWorld
    from src.world.structures import Structure, StructureType
    from src.world.terrain import TerrainType

    celle = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(4)] for y in range(3)]
    mondo = GridWorld(width=4, height=3, cells=celle)
    for x in (1, 3):
        mondo.add_structure(Structure(type=StructureType.STORAGE_DEPOT, x=x, y=1))
    ricca = mondo.get_cell(1, 1)
    povera = mondo.get_cell(3, 1)
    ricca.resources.minerals = 500.0
    povera.resources.minerals = 0.0

    agenti = {"a0": BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1)}
    ca = CellArrays.from_world(mondo)
    aa = AgentArrays.from_agents(agenti)
    stato = {"mean_temperature_c": -63.0, "pressure_pa": 600.0, "liquid_water_stability": 0.0}
    update_cells(ca, aa, stato, 7.0, cell_degradation=True, full_grid=True,
                 isru_material_rate=1.0)

    mat = ca.cell_res[..., C.R["construction_material"]]
    minerali = ca.cell_res[..., C.R["minerals"]]
    assert mat[1, 1] > 0.0, "sanita': la conversione deve aver prodotto qualcosa"
    assert np.isclose(mat[1, 1], mat[1, 3]), (
        f"il giacimento cambia la resa in materiale: ricca={mat[1, 1]} povera={mat[1, 3]}")
    # L'impianto rende anche minerali, e ne rende ALTRETTANTI nelle due celle.
    assert minerali[1, 3] > 0.0, "una cella senza giacimento deve comunque produrre minerali"
    assert np.isclose(minerali[1, 1] - 500.0, minerali[1, 3]), (
        "il giacimento non deve cambiare la resa in minerali: "
        f"ricca=+{minerali[1, 1] - 500.0} povera=+{minerali[1, 3]}")
    assert minerali[1, 1] >= 500.0, "la conversione non deve consumare il giacimento"
    # La resa TOTALE si ripartisce, non si aggiunge: e' la stessa quantita' di
    # prima divisa fra i due beni secondo `BUILD_COSTS`.
    from src.world.structures import DOMANDA_MINERALI_SU_MATERIALE as r
    assert np.isclose(minerali[1, 3] / mat[1, 3], r), (
        f"la miscela non segue BUILD_COSTS: {minerali[1, 3] / mat[1, 3]} invece di {r}")


def test_the_endowment_scales_inventory_and_not_only_structures():
    """E' l'inventario a pesare, non le strutture.

    A 300 agenti la dotazione iniziale vale 1500 unita' di materiale, mentre i
    dodici depositi ne fabbricano ~68 in ottocento passi. Scalare le sole
    strutture lasciava intatto il 96% della scorta, ed e' la ragione per cui a
    dotazione 0,1 `material_margin` restava a 1,96.
    """
    from scripts.parity_harness import realistic_config

    pieno = realistic_config(300, 10, 0)
    ridotto = realistic_config(300, 10, 0, dotazione=0.3)
    assert ridotto["agents"]["initial_inventory"]["construction_material"] < (
        pieno["agents"]["initial_inventory"]["construction_material"]
    )
    assert ridotto["colony"]["initial_structures"]["storage_depot"] < (
        pieno["colony"]["initial_structures"]["storage_depot"]
    )


def test_the_development_tier_can_now_reach_the_menu():
    """Il valore storico 0,07 rendeva i laboratori irraggiungibili, ed era un difetto.

    `observe` non ha alcun effetto — `ActionResult(True, "observed area")` — e
    con priorita' 0,20 stava sopra il livello di sviluppo. A
    `cell_proposal_top_k = 5` questo rendeva laboratori e infermerie
    impossibili da costruire **anche quando il modello stesso ne dichiarava il
    fabbisogno**: due sue parti si contraddicevano. Il default ora sta sopra il
    non-fare, cosi' una domanda insoddisfatta puo' entrare in gara.
    """
    from scripts.parity_harness import realistic_config

    config = realistic_config(50, 10, 0)
    assert config["agents"]["development_build_priority"] > 0.20
    assert config["agents"]["cell_proposal_top_k"] == 5
    # Il valore storico resta raggiungibile per riprodurre le run precedenti.
    assert realistic_config(50, 10, 0, sviluppo=0.07)["agents"]["development_build_priority"] == 0.07


def test_the_development_priority_scales_with_how_many_are_missing():
    """Piena solo quando ne mancano parecchie, altrimenti proporzionale.

    Con la sola frazione di fabbisogno un avamposto da dodici coloni, cui ne
    serve UNA di ciascun tipo e non ne ha nessuna, metteva ogni tipo di
    sviluppo al massimo: cinque azioni a pari punteggio riempivano il menu e ne
    cacciavano la raccolta dalle serre mature.
    """
    from src.core.cell_proposals import DEVELOPMENT_FULL_DEFICIT

    assert DEVELOPMENT_FULL_DEFICIT > 1.0


def test_the_agent_llm_temperature_is_absent_by_default():
    """Assente e' diverso da zero, e da 0,2.

    `ConfiguredProvider` usa 0,2 quando la chiave manca e non manda affatto il
    campo quando vale `None` — sono tre stati distinti, e scrivere sempre un
    numero ne cancellerebbe due. L'assenza significa "usa il registro dei
    provider", che e' una proprieta' della macchina e non della run.
    """
    from src.experiments.manual_config import ManualConfigOptions, build_manual_config

    assert "temperature" not in build_manual_config(ManualConfigOptions(run_name="t"))["llm"]
    acceso = build_manual_config(ManualConfigOptions(run_name="t", llm_temperature=0.7))
    assert acceso["llm"]["temperature"] == 0.7


def test_the_run_temperature_reaches_the_agent_assignments():
    """Senza questo passaggio la temperatura resterebbe nel registro condiviso.

    Tre livelli, dal piu' specifico: quello per agente vince su quello di run, e
    l'assenza di entrambi lascia decidere il registro.
    """
    from src.llm.provider_registry import normalize_llm_assignments

    nessuna = normalize_llm_assignments({"agents": {}, "llm": {}}, 1)
    assert "temperature" not in nessuna[0]

    di_run = normalize_llm_assignments({"agents": {}, "llm": {"temperature": 0.7}}, 1)
    assert di_run[0]["temperature"] == 0.7

    per_agente = normalize_llm_assignments(
        {"agents": {"llm_assignments": [{"temperature": 0.2}]}, "llm": {"temperature": 0.7}}, 1
    )
    assert per_agente[0]["temperature"] == 0.2
