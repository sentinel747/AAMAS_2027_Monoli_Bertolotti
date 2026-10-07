from src.core.needs import RedistributionConfig
from src.experiments.manual_config import ManualConfigOptions, build_manual_config, options_from_scenario_config
from src.experiments.scenarios import get_standard_scenario_config


def test_manual_config_defaults_match_frontend_manual_panel():
    config = build_manual_config(ManualConfigOptions(run_name="manual_unit"))

    assert config["name"] == "manual_unit"
    assert config["seed"] == 0
    assert config["days"] == 500
    assert config["simulation"] == {"days_per_step": 7}
    assert config["world"] == {"width": 360, "height": 180, "map_profile": "balanced", "cell_degradation": True}
    assert config["agents"]["count"] == 50
    assert config["agents"]["llm_count"] == 0
    assert config["agents"]["decision_mode"] == "preferences"
    assert config["agents"]["decision_sampling"] == "softmax"
    assert config["agents"]["individual_survival_priority_enabled"] is True
    assert config["agents"]["operational_range_m"] == 59_000.0
    assert config["agents"]["vision_radius_m"] == 59_000.0
    assert config["agents"]["movement_distance_m_per_step"] == 59_000.0
    assert sum(config["agents"]["role_distribution"].values()) == 100.0
    assert config["agents"]["role_distribution"]["engineer"] == 20.0
    assert config["agents"]["initial_inventory"]["food"] == 5
    assert config["agents"]["initial_inventory"]["oxygen"] == 2
    # Dal 2026-08-27 la dotazione SCALA con la popolazione: a cinquanta coloni
    # (il default) copre 58 persone, cioe' la colonia parte coperta e con spazio
    # per una prima nascita. `ScenarioPanel.tsx` porta la stessa funzione.
    assert config["colony"]["initial_structures"]["habitat"] == 10
    assert config["colony"]["initial_structures"]["greenhouse"] == 9
    assert config["colony"]["initial_structures"]["solar_array"] == 9
    assert config["colony"]["initial_structures"]["oxygen_plant"] == 6
    assert config["colony"]["initial_structures"]["storage_depot"] == 2
    assert config["colony"]["initial_structures"]["weather_station"] == 1
    # Accesa dal 2026-08-27: da spenta il materiale da costruzione e' una
    # dotazione non rinnovabile e la colonia finisce in uno stato assorbente.
    # Il pannello della GUI porta lo stesso valore, ed e' questo test a tenerli
    # allineati: un default diverso fra le due interfacce farebbe divergere run
    # lanciate con gli stessi parametri.
    assert config["colony"]["isru_material_rate"] == 1.0
    # **Assente, non "none".** `build_council` restituisce `None` quando il
    # blocco manca, e col `None` il kernel non invoca nemmeno l'applicazione
    # della direttiva: la baseline resta bit-exact per costruzione. Sia
    # `ScenarioPanel.tsx` sia `_apply_governors` in headless_runner.py omettono
    # il blocco quando il braccio e' `none`, ed e' questo test a tenerli fermi.
    assert "governors" not in config
    assert config["population"]["enabled"] is True
    # **Nessun tetto, e la sua ASSENZA e' la proprieta' (2026-09-01).** La
    # chiave valeva 100 e fermava ogni run di riferimento a cento coloni
    # esatti: una censura sull'esito, non un esito. A fermare la crescita
    # bastano i posti liberi che il supporto vitale sostiene.
    assert "max_agents" not in config["population"]
    assert config["population"]["daily_spawn_probability"] == 0.05
    assert config["llm"]["max_calls_per_step"] == 0
    assert config["headless"]["fast_observation"] is False
    # Acceso dal 2026-08-20: una run che non lascia traccia delle azioni non e'
    # analizzabile dopo, e il costo misurato e' disco e non tempo. Il pannello
    # della GUI (`ScenarioPanel.tsx`) porta lo stesso valore, ed e' questo test a
    # tenerli allineati: se uno dei due cambia da solo, le run lanciate dalle due
    # interfacce smettono di salvare le stesse cose.
    assert config["headless"]["store_memory_logs"] is True
    # Il tetto segue la run invece di essere fisso a 150.000: con 50 agenti e
    # 500 passi resta il pavimento, ma una run lunga non perde piu' la coda.
    assert config["headless"]["max_action_log_rows"] == 150_000
    assert config["headless"]["output_flush_interval_steps"] == 100
    assert config["headless"]["store_nearby_agent_ids"] is False
    assert config["headless"]["snapshot_interval"] == 0
    assert config["headless"]["max_replay_rows"] == 250_000
    assert config["model"]["psychosocial_enabled"] is False
    assert config["climate"]["provider"] == "call_mcd"
    assert config["climate"]["data_path"] == "data/mcd_runtime"
    assert config["environmental_layer"]["enabled"] is True
    assert config["extreme_events"]["chance_per_step"] == 0.05
    assert config["social"]["earth_mars_delay_minutes"] == 12
    # Default OFF dal 2026-08-30: coincide con `RedistributionConfig.enabled`,
    # con i commenti del kernel e con la nota di architettura, che dicevano
    # tutti OFF mentre questo valore diceva ON. Misurato: OFF vince su tutti
    # e tre i semi di prova, con zero morti contro 94.
    assert config["redistribution"]["enabled"] is False


def test_manual_config_llm_modes_and_assignments():
    config = build_manual_config(
        ManualConfigOptions(
            run_name="llm_unit",
            mode="mix",
            agent_count=6,
            llm_count=2,
            assignments=[{"provider": "openai", "model": "gpt-4o-mini"}],
        )
    )

    assert config["agents"]["count"] == 6
    assert config["agents"]["llm_count"] == 2
    assert config["agents"]["llm_assignments"] == [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "gpt", "model": "fallback"},
    ]

    all_llm = build_manual_config(ManualConfigOptions(run_name="all_llm", mode="llm", agent_count=6, llm_count=2))
    assert all_llm["agents"]["llm_count"] == 6


def test_manual_config_colony_site_explicit_round_trip():
    config = build_manual_config(ManualConfigOptions(run_name="site_unit", colony_start_x=210, colony_start_y=95))

    assert config["colony"]["start_x"] == 210
    assert config["colony"]["start_y"] == 95

    options = options_from_scenario_config(config)
    assert options.colony_start_x == 210
    assert options.colony_start_y == 95


def test_manual_config_colony_site_auto_omits_keys_and_clamps_manual():
    auto = build_manual_config(ManualConfigOptions(run_name="site_auto"))
    assert "start_x" not in auto["colony"]
    assert "start_y" not in auto["colony"]
    assert options_from_scenario_config(auto).colony_start_x is None

    clamped = build_manual_config(ManualConfigOptions(run_name="site_clamp", colony_start_x=4000, colony_start_y=-5))
    assert clamped["colony"]["start_x"] == 359
    assert clamped["colony"]["start_y"] == 0


def test_options_from_standard_scenario_round_trip_keeps_frontend_overrides():
    scenario = get_standard_scenario_config("standard_v1_balanced_colony")
    options = options_from_scenario_config(scenario, "standard_v1_balanced_colony")
    config = build_manual_config(options)

    assert config["scenario_id"] == "standard_v1_balanced_colony"
    assert config["name"] == scenario["name"]
    assert config["world"]["width"] == scenario["world"]["width"]
    assert config["world"]["height"] == scenario["world"]["height"]
    assert config["world"]["map_profile"] == scenario["world"].get("map_profile", "balanced")
    assert config["agents"]["count"] == scenario["agents"]["count"]


def test_manual_config_exports_redistribution_section_default_disabled():
    # Task 11 (task-11-brief.md): redistribution.enabled defaults to False in
    # the single config source, matching RedistributionConfig's own default
    # (src/core/needs.py) - the baseline stays byte-equivalent to the
    # pre-refactor engine unless a caller explicitly opts in.
    config = build_manual_config(ManualConfigOptions(run_name="redistribution_default"))
    red = config["redistribution"]
    assert red["enabled"] is False
    assert red["knapsack_targets"]["water"] == 4.0
    assert red["knapsack_targets"]["food"] == 2.0
    assert red["knapsack_targets"]["oxygen"] == 1.0
    assert red["knapsack_targets"]["construction_material"] == 2.0
    assert red["knapsack_targets"]["minerals"] == 1.0
    assert red["knapsack_targets"]["energy"] == 1.0
    assert red["knapsack_targets"]["tools"] == 1.0
    assert red["flow_radius"] == 1
    assert red["flow_rate_per_step"] == 2.0
    assert red["greenhouse_per_capita"] == 0.25


def test_manual_config_redistribution_block_satisfies_redistribution_config_from_config():
    # The exported "redistribution" block's shape must be exactly what
    # RedistributionConfig.from_config (src/core/needs.py) consumes -
    # construct it directly from build_manual_config's own output rather than
    # asserting on the dict shape by hand, so a future field-name drift
    # between the two modules fails loudly here instead of only at runtime
    # inside src.core.kernel.step.
    options = ManualConfigOptions(
        run_name="redistribution_enabled",
        redistribution_enabled=True,
        redistribution_knapsack_water=3.5,
        redistribution_knapsack_food=2.5,
        redistribution_knapsack_oxygen=1.5,
        redistribution_flow_radius=9,
        redistribution_flow_rate=4.0,
        redistribution_greenhouse_per_capita=0.4,
    )
    config = build_manual_config(options)
    rc = RedistributionConfig.from_config(config)

    assert rc.enabled is True
    assert rc.knapsack_targets["water"] == 3.5
    assert rc.knapsack_targets["food"] == 2.5
    assert rc.knapsack_targets["oxygen"] == 1.5
    # med_kits has no ManualConfigOptions field (kept minimal, per the brief) -
    # RedistributionConfig.from_config merges the supplied dict onto its own
    # defaults, so it must still be present at its default.
    assert rc.knapsack_targets["med_kits"] == 1.0
    assert rc.flow_radius == 9
    assert rc.flow_rate_per_step == 4.0
    assert rc.greenhouse_per_capita == 0.4

    # Default automatic-cell-logistics options must also round-trip cleanly through
    # RedistributionConfig.from_config without raising.
    default_rc = RedistributionConfig.from_config(build_manual_config(ManualConfigOptions()))
    assert default_rc.enabled is False

    # Drift guard: build_manual_config's six duplicated numeric literals
    # (forced by task-10-brief.md's anti-leak guard, which forbids
    # src/experiments/ from importing src.core.needs - see
    # src/experiments/manual_config.py's own comment) must stay
    # byte-equivalent to RedistributionConfig's REAL dataclass defaults, not
    # to a second hand-typed copy of the same numbers. Comparing against a
    # bare RedistributionConfig() here means a future default change in
    # src/core/needs.py (e.g. flow_radius 6 -> 8) with no matching edit in
    # manual_config.py fails this assertion instead of passing silently.
    bare_defaults = RedistributionConfig()
    assert default_rc.flow_radius == bare_defaults.flow_radius
    assert default_rc.flow_rate_per_step == bare_defaults.flow_rate_per_step
    assert default_rc.greenhouse_per_capita == bare_defaults.greenhouse_per_capita
    assert default_rc.knapsack_targets["water"] == bare_defaults.knapsack_targets["water"]
    assert default_rc.knapsack_targets["food"] == bare_defaults.knapsack_targets["food"]
    assert default_rc.knapsack_targets["oxygen"] == bare_defaults.knapsack_targets["oxygen"]


def test_manual_config_redistribution_options_round_trip():
    # options -> config -> options must preserve every redistribution field,
    # following the same _as_dict/_as_bool/_clamp_float pattern the other
    # sections use (options_from_scenario_config).
    options = ManualConfigOptions(
        run_name="redistribution_round_trip",
        redistribution_enabled=True,
        redistribution_knapsack_water=3.25,
        redistribution_knapsack_food=2.75,
        redistribution_knapsack_oxygen=1.25,
        redistribution_flow_radius=11,
        redistribution_flow_rate=5.5,
        redistribution_greenhouse_per_capita=0.33,
    )
    config = build_manual_config(options)
    round_tripped = options_from_scenario_config(config)

    assert round_tripped.redistribution_enabled is True
    assert round_tripped.redistribution_knapsack_water == 3.25
    assert round_tripped.redistribution_knapsack_food == 2.75
    assert round_tripped.redistribution_knapsack_oxygen == 1.25
    assert round_tripped.redistribution_flow_radius == 11
    assert round_tripped.redistribution_flow_rate == 5.5
    assert round_tripped.redistribution_greenhouse_per_capita == 0.33

    # Defaults round-trip too (disabled stays disabled).
    default_round_tripped = options_from_scenario_config(build_manual_config(ManualConfigOptions()))
    assert default_round_tripped.redistribution_enabled is False
    assert default_round_tripped.redistribution_knapsack_water == 4.0
    assert default_round_tripped.redistribution_flow_radius == 1


def test_decision_and_operational_range_round_trip():
    options = ManualConfigOptions(
        run_name="rt",
        decision_mode="tree",
        decision_sampling="greedy",
        individual_survival_priority_enabled=False,
        operational_range_m=120_000.0,
        cell_proposal_top_k=7,
        role_distribution={"biologist": 1, "technician": 1, "engineer": 2, "medic": 0, "coordinator": 0, "explorer": 0},
        role_preference_randomness=0.40,
    )
    config = build_manual_config(options)
    assert config["agents"]["decision_mode"] == "tree"
    assert config["agents"]["decision_sampling"] == "greedy"
    assert config["agents"]["individual_survival_priority_enabled"] is False
    assert config["agents"]["operational_range_m"] == 59_000.0
    assert config["agents"]["vision_radius_m"] == 59_000.0
    assert config["agents"]["movement_distance_m_per_step"] == 59_000.0
    assert config["agents"]["cell_proposal_top_k"] == 7
    assert config["agents"]["role_distribution"]["engineer"] == 50.0
    assert config["agents"]["role_preference_randomness"] == 0.40
    back = options_from_scenario_config(config)
    assert back.decision_mode == "tree"
    assert back.decision_sampling == "greedy"
    assert back.individual_survival_priority_enabled is False
    assert back.operational_range_m == 59_000.0
    assert back.cell_proposal_top_k == 7
    assert back.role_distribution["engineer"] == 50.0
    assert back.role_preference_randomness == 0.40


def test_decision_mode_invalid_falls_back():
    config = build_manual_config(
        ManualConfigOptions(
            run_name="x", decision_mode="banana", decision_sampling="invalid"
        )
    )
    assert config["agents"]["decision_mode"] == "preferences"
    assert config["agents"]["decision_sampling"] == "softmax"


def test_headless_decision_flags_map_to_options():
    from scripts.headless_runner import _options_from_args, build_parser

    args = build_parser().parse_args(
        [
            "--run-name",
            "flags",
            "--decision-mode",
            "tree",
            "--decision-sampling",
            "greedy",
            "--operational-range-km",
            "25",
            "--no-individual-survival-priority",
        ]
    )
    options = _options_from_args(args)
    assert options.decision_mode == "tree"
    assert options.decision_sampling == "greedy"
    assert options.operational_range_m == 25_000.0
    assert options.individual_survival_priority_enabled is False


def test_operational_range_is_clamped_to_supported_range():
    too_short = build_manual_config(
        ManualConfigOptions(run_name="short_move", operational_range_m=1.0)
    )
    too_long = build_manual_config(
        ManualConfigOptions(run_name="long_move", operational_range_m=120_000.0)
    )

    assert too_short["agents"]["operational_range_m"] == 100.0
    assert too_short["agents"]["vision_radius_m"] == 100.0
    assert too_short["agents"]["movement_distance_m_per_step"] == 100.0
    assert too_long["agents"]["operational_range_m"] == 59_000.0
    assert too_long["agents"]["vision_radius_m"] == 59_000.0
    assert too_long["agents"]["movement_distance_m_per_step"] == 59_000.0
    assert options_from_scenario_config(too_long).operational_range_m == 59_000.0


def test_legacy_split_ranges_migrate_to_strictest_bound():
    legacy = {"agents": {"vision_radius_m": 5_000.0, "movement_distance_m_per_step": 42_000.0}}
    options = options_from_scenario_config(legacy)
    assert options.operational_range_m == 5_000.0
    normalized = build_manual_config(options)
    assert normalized["agents"]["vision_radius_m"] == 5_000.0
    assert normalized["agents"]["movement_distance_m_per_step"] == 5_000.0


def test_the_wizard_and_the_gui_panel_cover_the_same_options():
    """Le due interfacce devono restare la stessa cosa, con o senza finestra.

    Il controllo e' testuale e non funzionale, ed e' deliberato: verifica che
    ogni campo di `ManualConfigOptions` sia *nominato* sia in
    `src/cli/config_wizard.py` sia in `frontend/.../ScenarioPanel.tsx`. Non
    prova che il controllo funzioni — a quello servono i test del wizard — ma
    intercetta il caso che si e' gia' presentato due volte: un campo aggiunto
    alla configurazione e cablato in una sola delle due interfacce, che rende
    due run "con gli stessi parametri" silenziosamente diverse.

    Il frontend usa camelCase, quindi si accetta l'una o l'altra forma.
    """
    import dataclasses
    from pathlib import Path

    radice = Path(__file__).resolve().parents[1]
    wizard = (radice / "src" / "cli" / "config_wizard.py").read_text(encoding="utf-8")
    pannello = (radice / "frontend" / "src" / "components" / "ScenarioPanel.tsx").read_text(encoding="utf-8")

    def nominato(campo: str, testo: str) -> bool:
        parti = campo.split("_")
        camel = parti[0] + "".join(p.capitalize() for p in parti[1:])
        return campo in testo or camel in testo

    #: Campi il cui nome nel pannello differisce troppo perche' la regola
    #: camelCase lo trovi. Vanno elencati, non ignorati in blocco: un elenco
    #: vuoto di eccezioni e' l'unico che non nasconde nulla per sbaglio.
    ALIAS_PANNELLO = {
        "governors_cadence_steps": "governorsCadence",
        "governors_wait_seconds": "governorsWait",
        # Il pannello costruisce direttamente gli oggetti provider/modello
        # per distretto; il wizard conserva invece il formato testuale che il
        # builder accetta ("provider:modello,..."). Sono la stessa leva.
        "administrators_models": "administratorAssignments",
    }

    mancanti = []
    for campo in (f.name for f in dataclasses.fields(ManualConfigOptions)):
        nel_wizard = nominato(campo, wizard)
        nel_pannello = nominato(campo, pannello) or ALIAS_PANNELLO.get(campo, "\0") in pannello
        if not (nel_wizard and nel_pannello):
            mancanti.append(
                f"{campo}: wizard={'si' if nel_wizard else 'NO'} pannello={'si' if nel_pannello else 'NO'}"
            )

    assert not mancanti, "opzioni cablate in una sola interfaccia:\n  " + "\n  ".join(mancanti)


def test_default_endowment_covers_the_founding_colony_at_any_size():
    """Il difetto che ha estinto le run del 26-27 agosto, in forma di test.

    La dotazione era un numero fisso tarato su cinquanta agenti: a duecento
    coloni la colonia partiva con supporto vitale per venti. `spare capacity` e'
    un cancello, non una probabilita': sotto zero non nasce nessuno, e con la
    ritaratura dell'usura la colonia si estingueva.
    """
    from src.agents.build_policy import COLONISTS_PER_STRUCTURE
    from src.experiments.manual_config import dotazione_iniziale
    from src.world.structures import StructureType

    for n in (10, 50, 120, 200, 1000):
        d = dotazione_iniziale(n)
        capienza = min(
            d["greenhouse"] * COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE],
            d["solar_array"] * COLONISTS_PER_STRUCTURE[StructureType.SOLAR_ARRAY],
            d["oxygen_plant"] * COLONISTS_PER_STRUCTURE[StructureType.OXYGEN_PLANT],
        )
        assert capienza > n, f"a {n} coloni la capienza e' {capienza}"


def test_explicit_endowment_still_overrides_the_derived_one():
    """I campi restano una leva: chi vuole una colonia povera deve poterla avere."""
    config = build_manual_config(
        ManualConfigOptions(run_name="override", agent_count=200, initial_greenhouses=3)
    )
    strutture = config["colony"]["initial_structures"]
    assert strutture["greenhouse"] == 3
    # gli altri campi restano derivati, uno per uno
    assert strutture["oxygen_plant"] == 23


def test_no_population_ceiling_reaches_the_config():
    """Nessuna taglia di colonia riceve un tetto, ed e' il punto.

    Fino al 2026-09-01 la config portava `max_agents`, e il valore storico (100)
    diventava <= alla popolazione appena si superavano i cento agenti: a duecento
    coloni valeva duecento, cioe' crescita zero consentita, in silenzio. La
    correzione di allora derivava il tetto dalla taglia; quella di oggi lo toglie
    del tutto, perche' il modello ha gia' i vincoli giusti e sono locali — un
    posto libero fra quelli che il supporto vitale sostiene, e un'impronta
    abitabile.

    L'assenza della chiave e' cio' che va sorvegliato: `maybe_spawn_agent` la
    legge ancora per gli scenari di prova che vogliono una popolazione bloccata,
    e se qualcuno la rimettesse nella config di riferimento le run tornerebbero a
    misurare il tetto invece della politica.
    """
    for n in (50, 120, 200, 500):
        config = build_manual_config(ManualConfigOptions(run_name="tetto", agent_count=n))
        assert "max_agents" not in config["population"], n


