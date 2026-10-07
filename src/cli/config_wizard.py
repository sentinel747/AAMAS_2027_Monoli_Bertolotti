"""Interactive terminal configuration wizard for headless simulation runs.

Presents the same knobs as the GUI ScenarioPanel (same ManualConfigOptions,
same build_manual_config) as navigable terminal menus: arrow keys / numbers
to move between sections, submenus per section, defaults kept unless edited,
and a final review screen that launches the run.
"""

from __future__ import annotations

from src.cli import tui
from src.cli.tui import Menu, MenuItem
from src.experiments.manual_config import (
    CLIMATE_SCENARIOS,
    DECISION_MODES,
    DECISION_SAMPLINGS,
    MAP_PROFILES,
    ManualConfigOptions,
    dotazione_iniziale,
    options_from_scenario_config,
)
from src.experiments.scenarios import get_standard_scenario_config, list_standard_scenarios
from src.llm.provider_registry import load_provider_registry

WORLD_WIDTH = 360
WORLD_HEIGHT = 180

MAP_PROFILE_HINTS = {
    "balanced": "risorse bilanciate (default)",
    "scarce_resources": "risorse scarse ovunque",
    "high_hazard": "terreno piu' pericoloso",
    "ice_rich": "molto ghiaccio d'acqua",
    "mineral_rich": "molti minerali/materiali",
    "fragmented": "risorse a macchie isolate",
    "random": "distribuzione casuale dal seed",
}

CLIMATE_HINTS = {
    "climatology": "anno marziano medio MCD (default)",
    "clim_minEUV": "minimo solare",
    "clim_maxEUV": "massimo solare",
    "cold": "scenario freddo",
    "warm": "scenario caldo",
}


def run_wizard(options: ManualConfigOptions | None = None) -> ManualConfigOptions | None:
    """Menu loop; returns the confirmed options, or None to exit without running."""
    options = options or ManualConfigOptions()
    while True:
        choice = tui.select_menu(_main_menu(options))
        if choice in (None, "exit"):
            if tui.confirm("Uscire senza avviare la run?", default=True):
                return None
            continue
        if choice == "run":
            options = _section_run(options)
        elif choice == "import":
            importata = _section_import(options)
            if importata is not None:
                options = importata
        elif choice == "time":
            _section_time(options)
        elif choice == "world":
            _section_world(options)
        elif choice == "colony":
            _section_colony(options)
        elif choice == "agents":
            _section_agents(options)
        elif choice == "events":
            _section_events(options)
        elif choice == "advanced":
            _section_advanced(options)
        elif choice == "governors":
            _section_governors(options)
        elif choice == "launch":
            if _review_and_confirm(options):
                return options


# --- main menu ----------------------------------------------------------------


def _main_menu(options: ManualConfigOptions) -> Menu:
    steps = "illimitati" if options.max_steps is None else str(options.max_steps)
    days = "illimitati" if options.max_sim_days is None else str(options.max_sim_days)
    items = [
        MenuItem("Riparti da una run gia' eseguita", "import", hint="importa tutte le impostazioni, dalla piu' recente"),
        MenuItem("Run e scenario", "run", hint=f"{options.run_name or '(nome da impostare)'} | seed {options.seed} | base: {options.selected_scenario_id or 'manuale'}"),
        MenuItem("Tempo di simulazione", "time", hint=f"{steps} step x {options.days_per_step} gg/step, max {days} gg"),
        MenuItem("Mondo e clima", "world", hint=f"mappa {options.map_profile}, clima {options.climate_scenario}"),
        MenuItem("Colonia", "colony", hint=f"sito {_site_label(options)}, {_structures_total(options)} strutture iniziali"),
        MenuItem("Agenti e popolazione", "agents", hint=_agents_hint(options)),
        MenuItem("Eventi estremi", "events", hint=_events_hint(options)),
        MenuItem("Governatore", "governors", hint=_governors_hint(options)),
        MenuItem("Avanzate", "advanced", hint=_advanced_hint(options)),
        MenuItem("", "", kind="separator"),
        MenuItem("Riepilogo e AVVIA la run", "launch"),
        MenuItem("Esci senza avviare", "exit"),
    ]
    return Menu(
        "Mars ABM - Configurazione run da terminale",
        items,
        subtitle="Ogni sezione mostra i valori correnti; entra per modificarli, poi 'Riepilogo e AVVIA'.",
    )


def _site_label(options: ManualConfigOptions) -> str:
    if options.colony_start_x is None or options.colony_start_y is None:
        return "auto"
    return f"({options.colony_start_x}, {options.colony_start_y})"


def _suggerimento(options: ManualConfigOptions, chiave: str) -> str:
    """Il numero, e se e' derivato lo dice: cosi' si vede che segue i coloni."""
    valore = _dotazione(options)[chiave]
    derivato = {
        "habitat": options.initial_habitats, "greenhouse": options.initial_greenhouses,
        "solar_array": options.initial_solar_arrays, "oxygen_plant": options.initial_oxygen_plants,
        "water_extractor": options.initial_water_extractors,
        "storage_depot": options.initial_storage_depots,
        "weather_station": options.initial_weather_stations,
    }[chiave] < 0
    return f"{valore} (scalati sui coloni)" if derivato else str(valore)


def _dotazione(options: ManualConfigOptions) -> dict[str, int]:
    """La dotazione COME SARA', risolvendo i campi lasciati alla derivazione.

    Senza questa risoluzione il menu mostrerebbe il sentinella `-1` al posto
    del numero di strutture con cui la colonia parte davvero.
    """
    derivata = dotazione_iniziale(options.agent_count)
    return {
        "habitat": options.initial_habitats if options.initial_habitats >= 0 else derivata["habitat"],
        "greenhouse": options.initial_greenhouses if options.initial_greenhouses >= 0 else derivata["greenhouse"],
        "solar_array": options.initial_solar_arrays if options.initial_solar_arrays >= 0 else derivata["solar_array"],
        "oxygen_plant": options.initial_oxygen_plants if options.initial_oxygen_plants >= 0 else derivata["oxygen_plant"],
        "water_extractor": options.initial_water_extractors if options.initial_water_extractors >= 0 else derivata["water_extractor"],
        "storage_depot": options.initial_storage_depots if options.initial_storage_depots >= 0 else derivata["storage_depot"],
        "weather_station": options.initial_weather_stations if options.initial_weather_stations >= 0 else derivata["weather_station"],
    }


def _structures_total(options: ManualConfigOptions) -> int:
    return sum(_dotazione(options).values())


def _agents_hint(options: ManualConfigOptions) -> str:
    growth = "crescita on" if options.population_growth_enabled else "crescita off"
    mode = options.mode if options.mode != "mix" else f"mix ({options.llm_count} LLM)"
    return f"{options.agent_count} agenti, {mode}, {growth}"


def _events_hint(options: ManualConfigOptions) -> str:
    if not options.extreme_events_enabled:
        return "off"
    return f"on, p={options.extreme_event_chance:g}/step"


def _advanced_hint(options: ManualConfigOptions) -> str:
    fast = "on" if options.fast_observation_enabled else "off"
    psycho = "on" if options.psychosocial_enabled else "off"
    degradation = "on" if options.cell_degradation_enabled else "off"
    survival = "individuale" if options.individual_survival_priority_enabled else "solo cella"
    return (
        f"decisione {options.decision_mode}/{options.decision_sampling}, "
        f"priorita {survival}, "
        f"raggio operativo {options.operational_range_m / 1000:g} km, fast obs {fast}, "
        f"psicosociale {psycho}, degrado celle {degradation}, "
        f"ritardo Terra-Marte {options.earth_mars_delay_minutes:g} min"
    )


# --- sections -------------------------------------------------------------------


def _section_import(options: ManualConfigOptions) -> ManualConfigOptions | None:
    """Riparti da una run gia' eseguita, cambiandone poi una cosa sola.

    Restituisce le opzioni importate, oppure `None` se non si e' scelto nulla —
    e in quel caso il chiamante tiene quelle correnti invece di azzerarle.

    Il nome della run arriva vuoto di proposito da `options_from_run`: ripeterla
    con lo stesso nome affiancherebbe artefatti indistinguibili, che e'
    esattamente la confusione che questa scorciatoia dovrebbe risparmiare. Il
    riepilogo lo chiede prima di avviare.
    """
    from src.experiments.run_import import list_importable_runs, options_from_run

    disponibili = list_importable_runs(limit=40)
    if not disponibili:
        tui.print_line("Nessuna run con configurazione leggibile da importare.")
        tui.print_line("Ne serve almeno una completata in outputs/runs o runs/.")
        return None

    items = [
        MenuItem(riassunto.label(), riassunto.run_id, hint=riassunto.run_id)
        for riassunto in disponibili
    ]
    items += [MenuItem("", "", kind="separator"), MenuItem("Indietro", "back", kind="back")]
    scelta = tui.select_menu(Menu(
        "Riparti da una run",
        items,
        subtitle="Dalla piu' recente. Importa tutte le impostazioni; il nome resta da scegliere.",
    ))
    if scelta is None or scelta == "back":
        return None

    importate = options_from_run(scelta)
    if importate is None:
        tui.print_line("Configurazione della run non leggibile: nulla e' stato cambiato.")
        return None
    tui.print_line(f"Impostazioni importate da '{scelta}'. Scegli un nome nuovo prima di avviare.")
    return importate


def _section_run(options: ManualConfigOptions) -> ManualConfigOptions:
    while True:
        items = [
            MenuItem("Nome run", "name", hint=options.run_name or "(obbligatorio: cartella in outputs/runs)"),
            MenuItem("Scenario standard di partenza", "scenario", hint=options.selected_scenario_id or "manuale (nessuno)"),
            MenuItem("Seed", "seed", hint=str(options.seed)),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Run e scenario", items))
        if choice is None:
            return options
        if choice == "name":
            options.run_name = tui.prompt_text("Nome run", default=options.run_name, note="Diventa la cartella in outputs/runs; usa lettere/numeri/underscore.")
        elif choice == "seed":
            options.seed = int(tui.prompt_number("Seed", default=options.seed, minimum=0, maximum=2_147_483_647, note="Stesso seed + stessa config = run riproducibile."))
        elif choice == "scenario":
            options = _pick_scenario(options)
    return options


def _pick_scenario(options: ManualConfigOptions) -> ManualConfigOptions:
    try:
        scenarios = list_standard_scenarios()
    except Exception as exc:  # file YAML mancante o invalido: il wizard resta usabile
        tui.print_line(f"Scenari standard non disponibili: {exc}")
        return options
    items = [MenuItem("Configurazione manuale (nessuno scenario)", "__manual__", hint="mantiene i valori correnti")]
    for scenario in scenarios:
        items.append(MenuItem(str(scenario.get("label", scenario["id"])), str(scenario["id"]), hint=str(scenario.get("description", ""))[:70]))
    items.append(MenuItem("Indietro", "back", kind="back"))
    choice = tui.select_menu(
        Menu("Scenario standard", items, subtitle="ATTENZIONE: scegliere uno scenario sovrascrive tutte le sezioni con i suoi valori."),
        start_value=options.selected_scenario_id or "__manual__",
    )
    if choice is None:
        return options
    if choice == "__manual__":
        options.selected_scenario_id = ""
        return options
    typed_name = options.run_name
    loaded = options_from_scenario_config(get_standard_scenario_config(choice), choice)
    if typed_name:
        loaded.run_name = typed_name
    return loaded


def _section_time(options: ManualConfigOptions) -> None:
    while True:
        steps = "nessun limite" if options.max_steps is None else str(options.max_steps)
        days = "nessun limite" if options.max_sim_days is None else str(options.max_sim_days)
        items = [
            MenuItem("Numero massimo di step", "steps", hint=steps),
            MenuItem("Giorni per step", "dps", hint=str(options.days_per_step)),
            MenuItem("Giorni simulati massimi", "days", hint=days),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Tempo di simulazione", items, subtitle="Senza alcun limite la run termina solo per estinzione o target di abitabilita'."))
        if choice is None:
            return
        if choice == "steps":
            options.max_steps = _optional_from(tui.prompt_number("Step massimi", default=options.max_steps, minimum=1, maximum=1_000_000, optional=True, note="'-' = nessun limite di step."))
        elif choice == "dps":
            options.days_per_step = int(tui.prompt_number("Giorni per step", default=options.days_per_step, minimum=1, maximum=1_000_000, note="7 = passo settimanale, 3650 = passo decennale (deep time)."))
        elif choice == "days":
            options.max_sim_days = _optional_from(tui.prompt_number("Giorni simulati massimi", default=options.max_sim_days, minimum=1, maximum=10_000_000_000, optional=True, note="'-' = nessun limite di giorni."))


def _section_world(options: ManualConfigOptions) -> None:
    while True:
        items = [
            MenuItem("Profilo mappa", "map", hint=options.map_profile),
            MenuItem("Scenario climatico MCD", "climate", hint=options.climate_scenario),
            MenuItem("Layer ambientale marziano", "env", hint="on" if options.environmental_layer_enabled else "off"),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Mondo e clima", items, subtitle=f"Griglia fissa {WORLD_WIDTH}x{WORLD_HEIGHT} celle (equal-angle su Marte)."))
        if choice is None:
            return
        if choice == "map":
            options.map_profile = _pick_choice("Profilo mappa", MAP_PROFILES, options.map_profile, MAP_PROFILE_HINTS)
        elif choice == "climate":
            options.climate_scenario = _pick_choice("Scenario climatico MCD", CLIMATE_SCENARIOS, options.climate_scenario, CLIMATE_HINTS)
        elif choice == "env":
            options.environmental_layer_enabled = not options.environmental_layer_enabled


def _section_colony(options: ManualConfigOptions) -> None:
    while True:
        items = [
            MenuItem("Sito colonia", "site", hint=_site_label(options) + (" = sicuro equatoriale" if options.colony_start_x is None else "")),
            MenuItem("", "", kind="separator"),
            MenuItem("Habitat iniziali", "habitat", hint=_suggerimento(options, "habitat")),
            MenuItem("Serre iniziali", "greenhouse", hint=_suggerimento(options, "greenhouse")),
            MenuItem("Pannelli solari iniziali", "solar", hint=_suggerimento(options, "solar_array")),
            MenuItem("Impianti ossigeno iniziali", "oxygen", hint=_suggerimento(options, "oxygen_plant")),
            MenuItem("Pozzi idrici iniziali", "pozzi", hint=_suggerimento(options, "water_extractor")),
            MenuItem("Depositi iniziali", "depot", hint=str(options.initial_storage_depots)),
            MenuItem("Stazioni meteo iniziali", "weather", hint=str(options.initial_weather_stations)),
            MenuItem("", "", kind="separator"),
            MenuItem("Cibo iniziale (per agente)", "food", hint=f"{options.initial_food:g}"),
            MenuItem("Acqua iniziale (per agente)", "water", hint=f"{options.initial_water:g}"),
            MenuItem("Materiali iniziali (per agente)", "materials", hint=f"{options.initial_materials:g}"),
            MenuItem("Attrezzi iniziali (per agente)", "tools", hint=f"{options.initial_tools:g}"),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Colonia", items))
        if choice is None:
            return
        if choice == "site":
            _edit_colony_site(options)
        elif choice == "habitat":
            options.initial_habitats = int(tui.prompt_number("Habitat iniziali", default=_dotazione(options)["habitat"], minimum=0, maximum=1000))
        elif choice == "greenhouse":
            options.initial_greenhouses = int(tui.prompt_number("Serre iniziali", default=_dotazione(options)["greenhouse"], minimum=0, maximum=1000))
        elif choice == "solar":
            options.initial_solar_arrays = int(tui.prompt_number("Pannelli solari iniziali", default=_dotazione(options)["solar_array"], minimum=0, maximum=1000))
        elif choice == "oxygen":
            options.initial_oxygen_plants = int(tui.prompt_number("Impianti ossigeno iniziali", default=_dotazione(options)["oxygen_plant"], minimum=0, maximum=1000))
        elif choice == "pozzi":
            options.initial_water_extractors = int(tui.prompt_number("Pozzi idrici iniziali", default=_dotazione(options)["water_extractor"], minimum=0, maximum=1000))
        elif choice == "depot":
            options.initial_storage_depots = int(tui.prompt_number("Depositi iniziali", default=options.initial_storage_depots, minimum=0, maximum=1000))
        elif choice == "weather":
            options.initial_weather_stations = int(tui.prompt_number("Stazioni meteo iniziali", default=options.initial_weather_stations, minimum=0, maximum=1000))
        elif choice == "food":
            options.initial_food = float(tui.prompt_number("Cibo iniziale per agente", default=options.initial_food, minimum=0, maximum=1_000_000, integer=False))
        elif choice == "water":
            options.initial_water = float(tui.prompt_number("Acqua iniziale per agente", default=options.initial_water, minimum=0, maximum=1_000_000, integer=False))
        elif choice == "materials":
            options.initial_materials = float(tui.prompt_number("Materiali iniziali per agente", default=options.initial_materials, minimum=0, maximum=1_000_000, integer=False))
        elif choice == "tools":
            options.initial_tools = float(tui.prompt_number("Attrezzi iniziali per agente", default=options.initial_tools, minimum=0, maximum=1_000_000, integer=False))


def _edit_colony_site(options: ManualConfigOptions) -> None:
    items = [
        MenuItem("Automatico (sito sicuro)", "auto", hint="prima piana di regolite nella fascia equatoriale"),
        MenuItem("Manuale (scegli x, y)", "manual", hint=_site_label(options) if options.colony_start_x is not None else ""),
        MenuItem("Indietro", "back", kind="back"),
    ]
    choice = tui.select_menu(Menu("Sito colonia", items, subtitle="Il sito scelto e' dove nascono agenti, strutture iniziali e nuovi coloni."))
    if choice is None:
        return
    if choice == "auto":
        options.colony_start_x = None
        options.colony_start_y = None
        return
    default_x = options.colony_start_x if options.colony_start_x is not None else WORLD_WIDTH // 2
    default_y = options.colony_start_y if options.colony_start_y is not None else WORLD_HEIGHT // 2
    options.colony_start_x = int(tui.prompt_number("Colonna X", default=default_x, minimum=0, maximum=WORLD_WIDTH - 1, note=f"0-{WORLD_WIDTH - 1}: 0 = 180 ovest, {WORLD_WIDTH // 2} = meridiano 0, {WORLD_WIDTH - 1} = 180 est."))
    options.colony_start_y = int(tui.prompt_number("Riga Y", default=default_y, minimum=0, maximum=WORLD_HEIGHT - 1, note=f"0-{WORLD_HEIGHT - 1}: 0 = polo nord, {WORLD_HEIGHT // 2} = equatore, {WORLD_HEIGHT - 1} = polo sud."))


def _section_agents(options: ManualConfigOptions) -> None:
    while True:
        items = [
            MenuItem("Modalita' agenti", "mode", hint={"rule": "rule (baseline, nessuna API)", "mix": "mix rule + LLM", "llm": "tutti LLM"}[options.mode]),
            MenuItem("Numero agenti iniziali", "count", hint=str(options.agent_count)),
        ]
        if options.mode == "mix":
            items.append(MenuItem("Agenti LLM (nel mix)", "llm_count", hint=str(min(options.llm_count, options.agent_count))))
        if options.mode != "rule":
            items.append(MenuItem("Chiamate LLM max per step", "llm_calls", hint=str(options.max_llm_calls_per_step)))
            items.append(MenuItem("Provider e modelli LLM", "assign", hint=_assignments_hint(options)))
        items.extend(
            [
                MenuItem("", "", kind="separator"),
                MenuItem("Crescita interna popolazione", "growth", hint="on" if options.population_growth_enabled else "off"),
                MenuItem("Probabilita' di crescita", "growth_p", hint=f"{options.growth_probability:g}"),
                MenuItem("", "", kind="separator"),
                MenuItem("Indietro", "back", kind="back"),
            ]
        )
        choice = tui.select_menu(Menu("Agenti e popolazione", items))
        if choice is None:
            return
        if choice == "mode":
            options.mode = _pick_choice(
                "Modalita' agenti",
                ("rule", "mix", "llm"),
                options.mode,
                {"rule": "solo agenti a regole: baseline senza chiamate API", "mix": "alcuni agenti guidati da LLM", "llm": "tutti gli agenti guidati da LLM (richiede API key)"},
            )
        elif choice == "count":
            options.agent_count = int(tui.prompt_number("Numero agenti iniziali", default=options.agent_count, minimum=1, maximum=10_000))
            options.llm_count = min(options.llm_count, options.agent_count)
        elif choice == "llm_count":
            options.llm_count = int(tui.prompt_number("Agenti LLM nel mix", default=min(options.llm_count, options.agent_count), minimum=0, maximum=options.agent_count))
        elif choice == "llm_calls":
            options.max_llm_calls_per_step = int(tui.prompt_number("Chiamate LLM max per step", default=options.max_llm_calls_per_step, minimum=0, maximum=100))
        elif choice == "assign":
            _section_llm_assignments(options)
        elif choice == "growth":
            options.population_growth_enabled = not options.population_growth_enabled
        elif choice == "growth_p":
            options.growth_probability = float(tui.prompt_number("Probabilita' di crescita", default=options.growth_probability, minimum=0, maximum=1, integer=False, note="0-1: scala la probabilita' di nuove nascite legata alla prosperita'."))


def _effective_llm_total(options: ManualConfigOptions) -> int:
    if options.mode == "rule":
        return 0
    if options.mode == "llm":
        return options.agent_count
    return min(options.llm_count, options.agent_count)


def _assignments_hint(options: ManualConfigOptions) -> str:
    total = _effective_llm_total(options)
    if total <= 0:
        return "nessun agente LLM"
    counts: dict[str, int] = {}
    for index in range(total):
        raw = options.assignments[index] if index < len(options.assignments) and isinstance(options.assignments[index], dict) else {}
        key = f"{raw.get('provider') or 'gpt'}/{raw.get('model') or 'fallback'}"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{count}x {key}" for key, count in counts.items())


def _provider_status(registry: dict, provider: str) -> str:
    cfg = registry.get(provider)
    if not isinstance(cfg, dict):
        return " (provider sconosciuto: usera' il fallback)"
    if cfg.get("is_configured"):
        return ""
    missing = str(cfg.get("missing_env") or cfg.get("api_key_env") or "")
    return f" (manca {missing}: usera' il fallback)" if missing else " (non configurato)"


def _pad_assignments(options: ManualConfigOptions, total: int, registry: dict) -> None:
    default_provider = "gpt" if "gpt" in registry else next(iter(registry), "gpt")
    default_model = str((registry.get(default_provider) or {}).get("available_models", ["fallback"])[0])
    normalized: list[dict[str, str]] = []
    for index in range(total):
        raw = options.assignments[index] if index < len(options.assignments) and isinstance(options.assignments[index], dict) else {}
        normalized.append({"provider": str(raw.get("provider") or default_provider), "model": str(raw.get("model") or default_model)})
    options.assignments = normalized


def _pick_assignment(registry: dict, current: dict[str, str] | None) -> dict[str, str] | None:
    """Provider first, then one of its models; None if the user backs out."""
    provider_items = []
    for provider_id, cfg in registry.items():
        if not isinstance(cfg, dict):
            continue
        models = cfg.get("available_models", [])
        hint = f"{len(models)} modelli{_provider_status(registry, provider_id)}"
        provider_items.append(MenuItem(provider_id, provider_id, hint=hint))
    provider_items.append(MenuItem("Indietro", "back", kind="back"))
    provider = tui.select_menu(
        Menu("Provider LLM", provider_items, subtitle="Provider senza API key configurata ripiegano sul FallbackProvider."),
        start_value=(current or {}).get("provider"),
    )
    if provider is None:
        return None
    models = [str(model) for model in (registry.get(provider) or {}).get("available_models", [])] or ["fallback"]
    model_items = [MenuItem(model, model) for model in models]
    model_items.append(MenuItem("Indietro", "back", kind="back"))
    model = tui.select_menu(Menu(f"Modello per {provider}", model_items), start_value=(current or {}).get("model"))
    if model is None:
        return None
    return {"provider": provider, "model": model}


def _section_llm_assignments(options: ManualConfigOptions) -> None:
    """Same per-LLM-agent provider/model picker offered by the GUI ScenarioPanel."""
    try:
        registry = load_provider_registry()
    except Exception as exc:
        tui.print_line(f"Registry provider LLM non disponibile: {exc}")
        return
    if not registry:
        tui.print_line("Nessun provider in configs/llm_providers.json: gli agenti LLM useranno il fallback.")
        return
    while True:
        total = _effective_llm_total(options)
        if total <= 0:
            tui.print_line("Nessun agente LLM: imposta prima modalita' mix o llm.")
            return
        _pad_assignments(options, total, registry)
        items = [MenuItem("Imposta tutti gli agenti LLM", "all", hint=_assignments_hint(options)), MenuItem("", "", kind="separator")]
        for index in range(total):
            spec = options.assignments[index]
            items.append(MenuItem(f"LLM {index + 1}", f"llm_{index}", hint=f"{spec['provider']} / {spec['model']}{_provider_status(registry, spec['provider'])}"))
        items.extend([MenuItem("", "", kind="separator"), MenuItem("Indietro", "back", kind="back")])
        choice = tui.select_menu(Menu("Provider e modelli agenti LLM", items, subtitle="Come nella GUI: un provider e un modello per ogni agente LLM."))
        if choice is None:
            return
        if choice == "all":
            spec = _pick_assignment(registry, options.assignments[0] if options.assignments else None)
            if spec:
                options.assignments = [dict(spec) for _ in range(total)]
        elif choice.startswith("llm_"):
            index = int(choice.removeprefix("llm_"))
            spec = _pick_assignment(registry, options.assignments[index])
            if spec:
                options.assignments[index] = spec


def _section_events(options: ManualConfigOptions) -> None:
    while True:
        items = [
            MenuItem("Eventi estremi", "enabled", hint="on" if options.extreme_events_enabled else "off"),
            MenuItem("Probabilita' per step", "chance", hint=f"{options.extreme_event_chance:g}" + ("" if options.extreme_events_enabled else " (ignorata: off)")),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Eventi estremi", items, subtitle="Tempeste di polvere, sciami meteorici, ondate di freddo: fronti che attraversano la mappa."))
        if choice is None:
            return
        if choice == "enabled":
            options.extreme_events_enabled = not options.extreme_events_enabled
        elif choice == "chance":
            options.extreme_event_chance = float(tui.prompt_number("Probabilita' evento per step", default=options.extreme_event_chance, minimum=0, maximum=1, integer=False, note="0 = mai, 1 = tentativo a ogni step."))


def _governors_hint(options: ManualConfigOptions) -> str:
    if options.governors_arm == "none":
        return "nessun governatore"
    dettaglio = f"una policy ogni {options.governors_cadence_steps} passi"
    if options.governors_arm == "llm":
        modello = options.governors_model or "(modello non scelto)"
        temperatura = "registro" if options.governors_temperature is None else f"T={options.governors_temperature:g}"
        return f"{options.governors_arm}: {dettaglio}, {options.governors_provider}/{modello}, {temperatura}"
    return f"{options.governors_arm}: {dettaglio}"


def _section_governors(options: ManualConfigOptions) -> None:
    """Il governatore unico, con gli stessi campi del pannello della GUI.

    `none` non scrive alcun blocco in configurazione, e non e' un dettaglio di
    stile: `build_governor` legge il blocco assente come "nessun governatore" e
    restituisce `None`, col quale il kernel non invoca nemmeno la funzione che
    applica la policy. La baseline resta quindi bit-exact per costruzione
    invece che per lettura di una stringa.
    """
    while True:
        governato = options.governors_arm != "none"
        llm = options.governors_arm == "llm"
        items = [
            MenuItem("Braccio", "arm", hint=_governors_hint(options)),
        ]
        if governato:
            items += [
                MenuItem("Cadenza (passi fra una policy e l'altra)", "cadence", hint=str(options.governors_cadence_steps)),
                MenuItem(
                    "Attesa massima (s)", "wait",
                    hint=f"{options.governors_wait_seconds:g}"
                         + (" - non bloccante: resta in vigore la policy precedente" if options.governors_wait_seconds <= 0
                            else " - bloccante: il passo attende il governatore"),
                ),
            ]
        if llm:
            items += [
                MenuItem("Provider", "provider", hint=options.governors_provider),
                MenuItem("Modello", "model", hint=options.governors_model or "(nessuno: girerebbe sul fallback)"),
                MenuItem("Sforzo di ragionamento", "effort", hint=options.governors_effort or "(default del provider)"),
                MenuItem(
                    "Temperatura", "temperature",
                    hint="dal registro" if options.governors_temperature is None else f"{options.governors_temperature:g}"
                         + (" - riproducibile: stessa fotografia, stessa policy" if options.governors_temperature == 0 else ""),
                ),
                MenuItem(
                    "Ragionamento", "thinking",
                    hint=f"{options.governors_thinking}"
                         + (" - spento: nessuna varianza estranea al trattamento"
                            if options.governors_thinking == "off" else ""),
                ),
                MenuItem(
                    "Contesto del prompt", "context",
                    hint=_contesto_hint(options.governors_context_level),
                ),
            ]
        items += [
            MenuItem("", "", kind="separator"),
            MenuItem(
                "Amministratori di distretto", "administrators",
                hint=_amministratori_hint(options),
            ),
        ]
        items += [MenuItem("", "", kind="separator"), MenuItem("Indietro", "back", kind="back")]
        choice = tui.select_menu(Menu("Governatore", items))
        if choice is None or choice == "back":
            return
        if choice == "arm":
            options.governors_arm = _pick_choice(
                "Braccio del governatore", ("none", "random", "scripted", "llm"), options.governors_arm,
                {
                    "none": "nessun governatore (baseline)",
                    "random": "policy arbitrarie entro gli stessi limiti",
                    "scripted": "la costituzione a soglie, non linguistica",
                    "llm": "la policy la scrive un modello",
                },
            )
        elif choice == "cadence":
            options.governors_cadence_steps = int(tui.prompt_number(
                "Passi fra una policy e l'altra", default=options.governors_cadence_steps,
                minimum=1, maximum=100_000, integer=True,
            ))
        elif choice == "wait":
            options.governors_wait_seconds = float(tui.prompt_number(
                "Attesa massima per la risposta (s)", default=options.governors_wait_seconds,
                minimum=0.0, maximum=3600.0, integer=False,
                note="0 = non bloccante: il passo prosegue e resta in vigore la policy precedente.",
            ))
        elif choice == "provider":
            options.governors_provider = tui.prompt_text(
                "Provider del governatore", default=options.governors_provider,
                note="'fallback' fa girare il braccio llm sul deterministico: nessun modello parlerebbe.",
            ).strip() or "fallback"
        elif choice == "model":
            options.governors_model = tui.prompt_text(
                "Modello del governatore", default=options.governors_model,
            ).strip()
        elif choice == "effort":
            options.governors_effort = _pick_choice(
                "Sforzo di ragionamento", ("", "none", "low", "medium", "high"), options.governors_effort,
            )
        elif choice == "thinking":
            options.governors_thinking = _pick_choice(
                "Ragionamento del modello",
                ("off", "low", "medium", "high", "dynamic"),
                options.governors_thinking,
                {
                    "off": "spento (default): niente varianza nascosta",
                    "low": "budget minimo",
                    "medium": "budget medio",
                    "high": "budget ampio, latenza alta",
                    "dynamic": "decide il servizio, dove l'API lo permette",
                },
            )
        elif choice == "context":
            options.governors_context_level = _pick_choice(
                "Contesto del prompt",
                ("completo", "senza_aiuti", "nomi_veri", "cieco"),
                options.governors_context_level,
                {
                    "completo": "ruolo, glossario, costituzione, avvertenze",
                    "senza_aiuti": "senza costituzione ne' avvertenze",
                    "nomi_veri": "solo nomi reali e numeri, nessun ruolo",
                    "cieco": "etichette neutre: nulla nomina il dominio",
                },
            )
        elif choice == "administrators":
            _section_administrators(options)
        elif choice == "temperature":
            options.governors_temperature = tui.prompt_number(
                "Temperatura del governatore", default=options.governors_temperature,
                minimum=0.0, maximum=2.0, integer=False, optional=True,
                note="0 rende la run riproducibile: la stessa fotografia da' la stessa policy. '-' lascia decidere il provider.",
            )


def _contesto_hint(livello: str) -> str:
    return {
        "completo": "tutto: ruolo, glossario, costituzione, avvertenze",
        "senza_aiuti": "senza costituzione ne' avvertenze",
        "nomi_veri": "solo nomi reali e numeri",
        "cieco": "etichette neutre: nulla nomina il dominio",
    }.get(livello, livello)


def _amministratori_hint(options: ManualConfigOptions) -> str:
    if not options.administrators_enabled:
        return "spenti: la politica va dal governo direttamente alle celle"
    dettaglio = f"uno ogni {options.administrators_cells_per_district} celle"
    if options.administrators_follow_governor_arm:
        return f"{dettaglio}, stesso braccio del governo"
    modello = options.administrators_model or "(modello non scelto)"
    return f"{dettaglio}, braccio proprio {options.administrators_arm}/{modello}"


def _section_administrators(options: ManualConfigOptions) -> None:
    """Lo strato fra il governo e le celle.

    Un amministratore ogni N celle; la cella madre resta sempre sotto il
    governo diretto. La politica non va dal governo alle celle: va dal governo
    agli amministratori, che la leggono gia' risolta sulle proprie celle e
    decidono se accettarla o riscriverla per il proprio distretto.
    """
    while True:
        attivi = options.administrators_enabled
        proprio = attivi and not options.administrators_follow_governor_arm
        items = [
            MenuItem("Attivi", "enabled", hint=_amministratori_hint(options)),
        ]
        if attivi:
            items += [
                MenuItem(
                    "Celle per amministratore", "cells",
                    hint=f"{options.administrators_cells_per_district} (la madre non conta)",
                ),
                MenuItem(
                    "Segue il braccio del governo", "follow",
                    hint="si - il confronto resta attribuibile al livello e non al modello"
                         if options.administrators_follow_governor_arm
                         else "no - braccio e modello propri",
                ),
                MenuItem(
                    "Modelli per distretto", "models",
                    hint=options.administrators_models or "(usa il modello singolo o quello del governo)",
                ),
            ]
        if proprio:
            items += [
                MenuItem("Braccio proprio", "arm", hint=options.administrators_arm),
                MenuItem("Provider", "provider", hint=options.administrators_provider),
                MenuItem("Modello", "model", hint=options.administrators_model or "(nessuno)"),
                MenuItem("Ragionamento", "thinking", hint=options.administrators_thinking),
            ]
        items += [MenuItem("", "", kind="separator"), MenuItem("Indietro", "back", kind="back")]
        choice = tui.select_menu(Menu("Amministratori di distretto", items))
        if choice is None or choice == "back":
            return
        if choice == "enabled":
            options.administrators_enabled = not options.administrators_enabled
        elif choice == "cells":
            options.administrators_cells_per_district = int(tui.prompt_number(
                "Celle per amministratore", default=options.administrators_cells_per_district,
                minimum=1, maximum=100, integer=True,
                note="La cella madre non entra in nessun distretto: resta di competenza diretta del governo.",
            ))
        elif choice == "follow":
            options.administrators_follow_governor_arm = not options.administrators_follow_governor_arm
        elif choice == "models":
            options.administrators_models = tui.prompt_text(
                "Modelli per distretto (provider:modello, separati da virgole)",
                default=options.administrators_models,
            ).strip()
        elif choice == "arm":
            options.administrators_arm = _pick_choice(
                "Braccio degli amministratori", ("none", "random", "scripted", "llm"),
                options.administrators_arm,
                {
                    "none": "si astengono sempre: decentramento a potere nullo",
                    "random": "riscrivono sempre, con regole arbitrarie",
                    "scripted": "intervengono dove il distretto sta sotto la media di colonia",
                    "llm": "chiedono a un modello",
                },
            )
        elif choice == "provider":
            options.administrators_provider = tui.prompt_text(
                "Provider degli amministratori", default=options.administrators_provider,
            ).strip() or "fallback"
        elif choice == "model":
            options.administrators_model = tui.prompt_text(
                "Modello degli amministratori", default=options.administrators_model,
            ).strip()
        elif choice == "thinking":
            options.administrators_thinking = _pick_choice(
                "Ragionamento degli amministratori",
                ("off", "low", "medium", "high", "dynamic"),
                options.administrators_thinking,
            )


def _section_redistribution(options: ManualConfigOptions) -> None:
    """Obiettivi di scorta personale e flusso fra celle.

    `greenhouse_per_capita` non e' un parametro fra gli altri: decide quante
    serre la colonia si prefigge, ed e' la correzione di scenario che ogni
    campagna sui governatori applica a TUTTI i bracci per tenerli confrontabili.
    Averlo qui evita di doverlo passare da un override a mano.
    """
    while True:
        items = [
            MenuItem("Attiva", "enabled", hint="on" if options.redistribution_enabled else "off"),
            MenuItem("Obiettivo acqua per colono", "water", hint=f"{options.redistribution_knapsack_water:g}"),
            MenuItem("Obiettivo cibo per colono", "food", hint=f"{options.redistribution_knapsack_food:g}"),
            MenuItem("Obiettivo ossigeno per colono", "oxygen", hint=f"{options.redistribution_knapsack_oxygen:g}"),
            MenuItem("Raggio del flusso (celle)", "radius", hint=str(options.redistribution_flow_radius)),
            MenuItem("Tasso del flusso", "rate", hint=f"{options.redistribution_flow_rate:g}"),
            MenuItem("Serre per colono", "greenhouse", hint=f"{options.redistribution_greenhouse_per_capita:g}"),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Redistribuzione", items))
        if choice is None or choice == "back":
            return
        if choice == "enabled":
            options.redistribution_enabled = not options.redistribution_enabled
        elif choice == "water":
            options.redistribution_knapsack_water = float(tui.prompt_number(
                "Obiettivo di acqua per colono", default=options.redistribution_knapsack_water,
                minimum=0.0, maximum=100.0, integer=False))
        elif choice == "food":
            options.redistribution_knapsack_food = float(tui.prompt_number(
                "Obiettivo di cibo per colono", default=options.redistribution_knapsack_food,
                minimum=0.0, maximum=100.0, integer=False))
        elif choice == "oxygen":
            options.redistribution_knapsack_oxygen = float(tui.prompt_number(
                "Obiettivo di ossigeno per colono", default=options.redistribution_knapsack_oxygen,
                minimum=0.0, maximum=100.0, integer=False))
        elif choice == "radius":
            options.redistribution_flow_radius = int(tui.prompt_number(
                "Raggio del flusso fra celle", default=options.redistribution_flow_radius,
                minimum=0, maximum=50, integer=True))
        elif choice == "rate":
            options.redistribution_flow_rate = float(tui.prompt_number(
                "Tasso del flusso", default=options.redistribution_flow_rate,
                minimum=0.0, maximum=10.0, integer=False))
        elif choice == "greenhouse":
            options.redistribution_greenhouse_per_capita = float(tui.prompt_number(
                "Serre per colono", default=options.redistribution_greenhouse_per_capita,
                minimum=0.0, maximum=10.0, integer=False,
                note="Le campagne sui governatori lo portano a 0,10 su tutti i bracci.",
            ))


def _section_advanced(options: ManualConfigOptions) -> None:
    while True:
        items = [
            MenuItem("Schema decisionale", "decision", hint=options.decision_mode),
            MenuItem("Campionamento preferenze", "sampling", hint=options.decision_sampling),
            MenuItem(
                "Priorita di sopravvivenza individuale",
                "survival_priority",
                hint="on: autonomia individuale" if options.individual_survival_priority_enabled else "off: solo proposte cella",
            ),
            MenuItem("Raggio operativo: visione + movimento (km/step)", "range", hint=f"{options.operational_range_m / 1000:g}"),
            MenuItem("Gruppi professionali (%)", "roles", hint=", ".join(f"{key}:{value:g}" for key, value in options.role_distribution.items())),
            MenuItem("Osservazione veloce", "fast", hint=("on: percezione ridotta, run piu' rapida" if options.fast_observation_enabled else "off: percezione completa (default)")),
            MenuItem("Modello psicosociale legacy", "psycho", hint=("on" if options.psychosocial_enabled else "off") + " - stress/morale per-agente aggiornati ogni tick"),
            MenuItem("Degrado celle e SYSTEM ALERT", "degradation", hint=("on" if options.cell_degradation_enabled else "off") + " - inquinamento da occupazione e alert sovraffollamento in memoria agenti"),
            MenuItem("Ritardo comunicazioni Terra-Marte (min)", "delay", hint=f"{options.earth_mars_delay_minutes:g} (3-22 reale; alza stress e bisogno di autonomia)"),
            MenuItem("Ampiezza del menu di cella (top-k)", "topk", hint=f"{options.cell_proposal_top_k} azioni di lavoro proposte per cella"),
            MenuItem(
                "Priorita costruzioni di sviluppo", "development",
                hint=f"{options.development_build_priority:g}"
                     + (" - storico: sotto observe (0,20), quindi laboratori e infermerie non entrano mai nel menu"
                        if options.development_build_priority < 0.20 else " - alzata: entrano in gara"),
            ),
            MenuItem(
                "Snapshot del mondo (ogni N passi)", "snapshot",
                hint=(f"{options.snapshot_interval}" if options.snapshot_interval > 0 else "0")
                     + (" - nessuno: la run NON sara' rivedibile sulla mappa dell'analysis frontend"
                        if options.snapshot_interval <= 0 else " - replay disponibile"),
            ),
            MenuItem(
                "Conversione ISRU regolito -> materiale + minerali", "isru",
                hint=f"{options.isru_material_rate:g}"
                     + (" - SPENTA: materiale e minerali sono dotazione non rinnovabile e la colonia muore verso il passo 350"
                        if options.isru_material_rate <= 0 else " - accesa: i depositi lavorano il regolito della cella"),
            ),
            MenuItem(
                "Redistribuzione delle scorte", "redistribution",
                hint=("on" if options.redistribution_enabled else "off")
                     + f" - serre {options.redistribution_greenhouse_per_capita:g}/colono, "
                       f"raggio {options.redistribution_flow_radius}, tasso {options.redistribution_flow_rate:g}",
            ),
            MenuItem(
                "Temperatura modelli AGENTI", "llm_temp",
                hint="dal registro provider" if options.llm_temperature is None else f"{options.llm_temperature:g}",
            ),
            MenuItem("", "", kind="separator"),
            MenuItem("Indietro", "back", kind="back"),
        ]
        choice = tui.select_menu(Menu("Avanzate", items))
        if choice is None:
            return
        if choice == "decision":
            options.decision_mode = _pick_choice(
                "Schema decisionale", DECISION_MODES, options.decision_mode,
                {"preferences": "preferenze stocastiche (default)", "tree": "albero storico"},
            )
        elif choice == "sampling":
            options.decision_sampling = _pick_choice(
                "Campionamento preferenze", DECISION_SAMPLINGS, options.decision_sampling,
                {"softmax": "campionamento proporzionale", "greedy": "massimo score deterministico"},
            )
        elif choice == "survival_priority":
            options.individual_survival_priority_enabled = not options.individual_survival_priority_enabled
        elif choice == "topk":
            options.cell_proposal_top_k = int(tui.prompt_number(
                "Azioni di lavoro proposte per cella",
                default=options.cell_proposal_top_k, minimum=1, maximum=15, integer=True,
                note="Il menu tiene le prime k per priorita'. A 5, il livello di sviluppo (0,07) resta fuori.",
            ))
        elif choice == "development":
            options.development_build_priority = float(tui.prompt_number(
                "Priorita delle costruzioni di sviluppo",
                default=options.development_build_priority, minimum=0.0, maximum=1.0, integer=False,
                note="0,07 e' lo storico e sta sotto observe (0,20): laboratori e infermerie non entrano mai nel menu.",
            ))
        elif choice == "snapshot":
            options.snapshot_interval = int(tui.prompt_number(
                "Snapshot del mondo ogni N passi (0 = nessuno)",
                default=options.snapshot_interval, minimum=0, maximum=100000, integer=True,
            ))
        elif choice == "isru":
            options.isru_material_rate = float(tui.prompt_number(
                "Conversione ISRU regolito -> materiale + minerali",
                default=options.isru_material_rate, minimum=0.0, maximum=1000.0, integer=False,
                note="0 = spenta (default storico): il materiale resta una dotazione iniziale non rinnovabile. ~5 pareggia il consumo.",
            ))
        elif choice == "redistribution":
            _section_redistribution(options)
        elif choice == "llm_temp":
            # `optional` restituisce `None` su '-', ed e' esattamente lo stato che
            # serve: "usa il registro dei provider" e' distinto sia da zero sia da
            # qualunque numero, e un solo valore non saprebbe dire le due cose.
            options.llm_temperature = tui.prompt_number(
                "Temperatura dei modelli AGENTI",
                default=options.llm_temperature, minimum=0.0, maximum=2.0,
                integer=False, optional=True,
                note="'-' lascia decidere il registro dei provider, che e' una proprieta' della macchina e non della run.",
            )
        elif choice == "range":
            options.operational_range_m = 1000.0 * float(
                tui.prompt_number(
                    "Raggio operativo in km",
                    default=options.operational_range_m / 1000.0,
                    minimum=0.1,
                    maximum=59,
                    integer=False,
                    note="Unico limite per cio che l'agente puo vedere e raggiungere nello step; 59 km coprono circa una macro-cella.",
                )
            )
        elif choice == "roles":
            for role in ("biologist", "technician", "engineer", "medic", "coordinator", "explorer"):
                options.role_distribution[role] = float(
                    tui.prompt_number(
                        f"Percentuale {role}",
                        default=options.role_distribution.get(role, 0.0),
                        minimum=0,
                        maximum=100,
                        integer=False,
                        note="Le percentuali vengono normalizzate automaticamente a 100.",
                    )
                )
            options.role_preference_randomness = float(
                tui.prompt_number(
                    "Variazione individuale entro il gruppo (0-1)",
                    default=options.role_preference_randomness,
                    minimum=0,
                    maximum=1,
                    integer=False,
                )
            )
        elif choice == "fast":
            options.fast_observation_enabled = not options.fast_observation_enabled
        elif choice == "psycho":
            options.psychosocial_enabled = not options.psychosocial_enabled
        elif choice == "degradation":
            options.cell_degradation_enabled = not options.cell_degradation_enabled
        elif choice == "delay":
            options.earth_mars_delay_minutes = float(tui.prompt_number("Ritardo Terra-Marte in minuti", default=options.earth_mars_delay_minutes, minimum=0, maximum=44, integer=False))


# --- review ---------------------------------------------------------------------


def _review_and_confirm(options: ManualConfigOptions) -> bool:
    tui.clear_screen()
    tui.print_heading("Riepilogo configurazione")
    for line in summary_lines(options):
        tui.print_line(line)
    tui.print_line("")
    if not options.run_name.strip():
        tui.print_line("Il nome run e' obbligatorio (diventa la cartella in outputs/runs).")
        options.run_name = tui.prompt_text("Nome run", default="")
        if not options.run_name.strip():
            tui.print_line("Nome run ancora vuoto: torno al menu.")
            return False
    return tui.confirm("Avviare la run adesso?", default=True)


def summary_lines(options: ManualConfigOptions) -> list[str]:
    steps = "nessun limite" if options.max_steps is None else str(options.max_steps)
    days = "nessun limite" if options.max_sim_days is None else str(options.max_sim_days)
    site = "auto: prima piana di regolite nella fascia equatoriale" if options.colony_start_x is None else f"manuale ({options.colony_start_x}, {options.colony_start_y})"
    structures = ", ".join(
        f"{count} {label}"
        for label, count in (
            ("habitat", options.initial_habitats),
            ("serre", options.initial_greenhouses),
            ("solari", options.initial_solar_arrays),
            ("ossigeno", options.initial_oxygen_plants),
            ("depositi", options.initial_storage_depots),
            ("meteo", options.initial_weather_stations),
        )
        if count
    ) or "nessuna"
    lines = [
        f"Run:      {options.run_name or '(nome da impostare)'} | seed {options.seed} | scenario base: {options.selected_scenario_id or 'manuale'}",
        f"Tempo:    {steps} step x {options.days_per_step} gg/step, massimo {days} giorni simulati",
        f"Mondo:    mappa {options.map_profile}, clima {options.climate_scenario}, layer ambientale {'on' if options.environmental_layer_enabled else 'off'}",
        f"Colonia:  sito {site}",
        f"          strutture iniziali: {structures}",
        f"          inventario per agente: cibo {options.initial_food:g}, acqua {options.initial_water:g}, materiali {options.initial_materials:g}, attrezzi {options.initial_tools:g}",
        f"Agenti:   {_agents_hint(options)}",
        f"Eventi:   {_events_hint(options)}",
        f"Avanzate: {_advanced_hint(options)}",
    ]
    if options.mode != "rule":
        lines.append(f"LLM:      {_assignments_hint(options)}, max {options.max_llm_calls_per_step} chiamate/step (servono API key configurate)")
    return lines


# --- helpers --------------------------------------------------------------------


def _pick_choice(title: str, choices: tuple[str, ...], current: str, hints: dict[str, str] | None = None) -> str:
    items = [MenuItem(choice, choice, hint=(hints or {}).get(choice, "") + (" [attuale]" if choice == current else "")) for choice in choices]
    items.append(MenuItem("Indietro", "back", kind="back"))
    picked = tui.select_menu(Menu(title, items), start_value=current)
    return picked if picked is not None else current


def _optional_from(value: float | int | None) -> int | None:
    return None if value is None else int(value)
