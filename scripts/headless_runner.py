"""Headless simulation runner.

Senza argomenti apre un menu di configurazione interattivo nel terminale
(frecce/numeri + Invio, stessi parametri del pannello GUI); con gli argomenti
espliciti resta completamente scriptabile come prima. In entrambi i casi la
run usa lo stesso SimulationController della GUI e salva gli stessi artifact.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import e path relativi (data/mcd_runtime, outputs/) funzionano solo dalla
# radice del repo: ancoriamola dal percorso dello script, non dal cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.api.state_store import SimulationController
from src.experiments.manual_config import ManualConfigOptions, build_manual_config, options_from_scenario_config
from src.experiments.scenarios import get_standard_scenario_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless Simulation Runner (stesso motore e stessi artifact della GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            "  python scripts/headless_runner.py                     menu interattivo di configurazione\n"
            "  python scripts/headless_runner.py -i --seed 7         menu interattivo precompilato\n"
            "  python scripts/headless_runner.py --run-name test01 --steps 120 --days-per-step 7 --operational-range-km 59\n"
        ),
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Apri il menu di configurazione interattivo (default se lanciato senza argomenti)")
    parser.add_argument("--scenario-id", type=str, default="", help="Standard scenario id to preload before manual overrides")
    parser.add_argument("--run-name", type=str, default="", help="Unique name for the run folder (obbligatorio in modalita' non interattiva)")
    parser.add_argument("--agents", type=int, help="Override agent count")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run (default non interattivo: 100)")
    parser.add_argument("--mode", choices=["rule", "mix", "llm"], default=None, help="Agent mode matching the frontend manual config")
    parser.add_argument("--llm-count", type=int, default=None, help="LLM agents used only in mix mode")
    parser.add_argument("--map-profile", type=str, default=None, help="Map profile matching the frontend manual config")
    parser.add_argument(
        "--llm-temperature", type=float, default=None, metavar="T",
        help="temperatura dei modelli degli AGENTI; assente = valore del registro provider",
    )
    parser.add_argument(
        "--dotazione", type=float, default=None, metavar="FATTORE",
        help="moltiplicatore della dotazione iniziale (strutture e inventario); 1,0 = invariata",
    )
    parser.add_argument(
        "--sviluppo", type=float, default=None, metavar="P",
        help="priorita' del livello di sviluppo a fabbisogno pieno; 0,07 = comportamento storico",
    )
    parser.add_argument(
        "--isru", type=float, default=None, metavar="TASSO",
        help="conversione ISRU del regolito in materiale da costruzione E minerali, ripartiti secondo BUILD_COSTS; 0 = spenta, e con 0 la colonia muore verso il passo 350 (misurato)",
    )
    parser.add_argument("--decision-mode", choices=["tree", "preferences"], default=None, help="Schema decisionale agenti (default: preferences)")
    parser.add_argument("--decision-sampling", choices=["greedy", "softmax"], default=None, help="Campionamento della scelta a preferenze")
    parser.add_argument(
        "--individual-survival-priority",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ON: priorita/guardie individuali; OFF: decisioni basate solo sulle proposte della cella",
    )
    parser.add_argument("--operational-range-km", type=float, default=None, help="Raggio unico di visione e movimento/step in km, limitato a 0.1-59 (default: 59)")
    parser.add_argument("--vision-radius-km", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vision-radius", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--movement-distance-km", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Simulation seed")
    parser.add_argument("--days-per-step", type=int, default=None, help="Days advanced by each decision step")
    parser.add_argument("--max-sim-days", type=int, default=None, help="Maximum simulated days")
    parser.add_argument(
        "--snapshot-interval", type=int, default=None, metavar="PASSI",
        help="salva uno snapshot del mondo ogni PASSI passi in world_snapshots/; 0 = nessuno. "
             "SENZA snapshot la run non e' rivedibile sulla mappa dell'analysis frontend",
    )
    parser.add_argument("--colony-x", type=int, default=None, help="Colonna X del sito colonia (0-359); senza, sito automatico sicuro")
    parser.add_argument("--colony-y", type=int, default=None, help="Riga Y del sito colonia (0-179); senza, sito automatico sicuro")
    parser.add_argument(
        "--no-agent-logs", dest="agent_logs", action="store_false",
        help="non salvare i log per azione e per agente (validated_actions, "
             "rejected_actions, agent_decisions, thoughts, conversations). Di "
             "default vengono salvati: costano disco (~65 MB ogni 200 passi con "
             "300 agenti) e quasi niente tempo, e senza di loro il dettaglio "
             "della run non e' ricostruibile dopo",
    )
    parser.set_defaults(agent_logs=True)

    # -- Consiglio dei governatori -------------------------------------------
    # La GUI puo' gia' costruirlo dalla configurazione; da terminale si passava
    # solo da `--overrides` con un blocco JSON scritto a mano, che e' un modo
    # per sbagliare in silenzio l'unico parametro (`cadence_steps`) capace di
    # rendere nullo l'intero esperimento.
    parser.add_argument(
        "--governors-arm", choices=["none", "random", "scripted", "llm"], default="none",
        help="braccio del governatore: `none` lo spegne (default), `random` "
             "emette policy qualsiasi, `scripted` la costituzione a soglie, "
             "`llm` fa scrivere la policy a un modello",
    )
    parser.add_argument(
        "--governors-cadence", type=int, default=20,
        help="ogni quanti passi il governatore riscrive la policy. **Se supera "
             "la lunghezza della run nessuna policy entra mai in vigore** e la "
             "run coincide con la baseline: il governatore lo verifica e si "
             "rifiuta di partire",
    )
    parser.add_argument(
        "--governors-wait", type=float, default=0.0, metavar="SECONDI",
        help="regime BLOCCANTE: al confine di tick la simulazione attende il "
             "governatore fino a questo limite. Con 0 non attende mai",
    )
    parser.add_argument("--governors-provider", default="fallback", help="provider del braccio llm, es. gpu_farm")
    parser.add_argument("--governors-model", default="", help="modello del braccio llm, es. gpt-oss:20b")
    parser.add_argument(
        "--governors-effort", default="", choices=["", "none", "low", "medium", "high"],
        help="sforzo di ragionamento; lo STESSO misurato con check_provider.py",
    )
    parser.add_argument(
        "--governors-context", default="completo",
        choices=["completo", "senza_aiuti", "nomi_veri", "cieco"],
        help=(
            "quanto contesto riceve il governatore. `completo` = ruolo, glossario, "
            "costituzione di riferimento e avvertenze; `senza_aiuti` toglie "
            "costituzione e avvertenze; `nomi_veri` toglie anche il ruolo e il "
            "glossario; `cieco` rinomina indicatori e pilastri in etichette neutre. "
            "E' una variabile sperimentale: separa il ragionare sui numeri dal "
            "riconoscere il dominio"
        ),
    )
    parser.add_argument(
        "--governors-thinking", default="off",
        choices=["off", "low", "medium", "high", "dynamic"],
        help=(
            "ragionamento del modello del governatore. Ogni API lo esprime a modo "
            "suo (reasoning_effort, thinkingBudget, enable_thinking) e la "
            "traduzione e' automatica. Il default e' spento: un ragionamento "
            "nascosto aggiunge varianza che non appartiene al trattamento"
        ),
    )
    parser.add_argument(
        "--administrators", action="store_true",
        help=(
            "attiva lo strato amministrativo: un amministratore ogni N celle, la "
            "cella madre sotto governo diretto. La politica non va dal governo "
            "alle celle ma dal governo agli amministratori e da questi alle celle"
        ),
    )
    parser.add_argument(
        "--administrators-cells", type=int, default=3, metavar="N",
        help="quante celle tiene un amministratore (default 3)",
    )
    parser.add_argument(
        "--administrators-own-arm", action="store_true",
        help=(
            "gli amministratori NON seguono il braccio del governo ma usano il "
            "proprio (--administrators-arm/provider/model). Senza questo, "
            "ereditano braccio, provider, modello e contesto del governo, che e' "
            "cio' che rende il confronto centralizzato/decentrato attribuibile "
            "al livello e non al modello"
        ),
    )
    parser.add_argument(
        "--administrators-arm", default="llm",
        choices=["none", "random", "scripted", "llm"],
        help="braccio proprio degli amministratori (serve --administrators-own-arm)",
    )
    parser.add_argument("--administrators-provider", default="fallback")
    parser.add_argument("--administrators-model", default="")
    parser.add_argument(
        "--administrators-models", nargs="+", default=[], metavar="PROVIDER:MODELLO",
        help="lista di modelli per gli amministratori: ogni distretto nuovo ne pesca "
        "uno, riproducibile a parita' di seme. Implica braccio llm proprio.",
    )
    parser.add_argument(
        "--administrators-thinking", default="off",
        choices=["off", "low", "medium", "high", "dynamic"],
    )
    parser.add_argument(
        "--governors-temperature", type=float, default=None, metavar="T",
        help="temperatura del governatore; senza, quella del registro",
    )
    parser.add_argument("--overrides", type=str, help="JSON object with final deep config overrides")
    return parser


def _deep_update(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _options_from_args(args: argparse.Namespace) -> ManualConfigOptions:
    if args.scenario_id:
        options = options_from_scenario_config(get_standard_scenario_config(args.scenario_id), args.scenario_id)
    else:
        options = ManualConfigOptions()
    if args.run_name:
        options.run_name = args.run_name
    if args.seed is not None:
        options.seed = args.seed
    if args.mode is not None:
        options.mode = args.mode
    if args.agents:
        options.agent_count = args.agents
    if args.llm_count is not None:
        options.llm_count = args.llm_count
    if args.map_profile is not None:
        options.map_profile = args.map_profile
    if args.isru is not None:
        options.isru_material_rate = args.isru
    if args.snapshot_interval is not None:
        options.snapshot_interval = max(0, args.snapshot_interval)
    if args.llm_temperature is not None:
        options.llm_temperature = args.llm_temperature
    if args.sviluppo is not None:
        options.development_build_priority = args.sviluppo
    if args.dotazione is not None:
        # Scala insieme strutture iniziali e inventario per agente. E'
        # l'inventario a pesare: a 300 agenti `initial_materials = 5` fa 1500
        # unita' di materiale, mentre i dodici depositi ne fabbricano ~68 in
        # ottocento passi. La stessa regola di `realistic_options`.
        fattore = float(args.dotazione)
        options.initial_food *= fattore
        options.initial_water *= fattore
        options.initial_materials *= fattore
        options.initial_tools *= fattore
        for campo in (
            "initial_habitats", "initial_greenhouses", "initial_solar_arrays",
            "initial_oxygen_plants", "initial_storage_depots", "initial_weather_stations",
        ):
            setattr(options, campo, max(0, round(getattr(options, campo) * fattore)))
    if args.decision_mode is not None:
        options.decision_mode = args.decision_mode
    if args.decision_sampling is not None:
        options.decision_sampling = args.decision_sampling
    if args.individual_survival_priority is not None:
        options.individual_survival_priority_enabled = args.individual_survival_priority
    if args.operational_range_km is not None:
        options.operational_range_m = args.operational_range_km * 1000.0
    else:
        legacy_ranges = []
        if args.vision_radius_km is not None:
            legacy_ranges.append(args.vision_radius_km * 1000.0)
        elif args.vision_radius is not None:
            legacy_ranges.append(args.vision_radius)
        if args.movement_distance_km is not None:
            legacy_ranges.append(args.movement_distance_km * 1000.0)
        if legacy_ranges:
            options.operational_range_m = min(legacy_ranges)
    if args.steps is not None:
        options.max_steps = args.steps
    if args.days_per_step is not None:
        options.days_per_step = args.days_per_step
    if args.max_sim_days is not None:
        options.max_sim_days = args.max_sim_days
    if args.colony_x is not None and args.colony_y is not None:
        options.colony_start_x = args.colony_x
        options.colony_start_y = args.colony_y
    return options


def _apply_agent_logs(config: dict, enabled: bool) -> None:
    """Accende o spegne i log per azione, senza toccare il resto."""
    config.setdefault("headless", {})["store_memory_logs"] = bool(enabled)


def _apply_governors(config: dict, args) -> None:
    """Scrive il blocco `governors`, o lo lascia assente.

    Assente e non `arm: "none"`: `build_governor` legge il blocco assente come
    "nessun governatore" e restituisce `None`, e con `None` il kernel non
    invoca nemmeno la funzione di applicazione — la baseline resta bit-exact
    per costruzione invece che per verifica. Scrivere comunque il blocco
    funzionerebbe, ma toglierebbe quella garanzia dal codice e la lascerebbe a
    una lettura di stringa.
    """
    arm = str(getattr(args, "governors_arm", "none") or "none")
    amministratori = bool(getattr(args, "administrators", False))
    if arm == "none" and not amministratori:
        return
    blocco = {
        "arm": arm,
        "cadence_steps": int(getattr(args, "governors_cadence", 20)),
        "wait_seconds": float(getattr(args, "governors_wait", 0.0) or 0.0),
    }
    provider = str(getattr(args, "governors_provider", "fallback") or "fallback")
    if arm == "llm" and provider != "fallback":
        assegnazione = {"provider": provider, "model": str(getattr(args, "governors_model", "") or "")}
        effort = str(getattr(args, "governors_effort", "") or "")
        if effort:
            assegnazione["effort"] = effort
        pensiero = str(getattr(args, "governors_thinking", "off") or "off")
        if pensiero:
            assegnazione["thinking"] = pensiero
        temperatura = getattr(args, "governors_temperature", None)
        if temperatura is not None:
            assegnazione["temperature"] = float(temperatura)
        blocco["assignments"] = [assegnazione]
    blocco["context_level"] = str(getattr(args, "governors_context", "completo") or "completo")
    if amministratori:
        sezione = {
            "enabled": True,
            "cells_per_district": max(1, int(getattr(args, "administrators_cells", 3) or 3)),
            "follow_governor_arm": not bool(getattr(args, "administrators_own_arm", False)),
        }
        if not sezione["follow_governor_arm"]:
            sezione["arm"] = str(getattr(args, "administrators_arm", "llm") or "llm")
            provider_amm = str(getattr(args, "administrators_provider", "fallback") or "fallback")
            if sezione["arm"] == "llm" and provider_amm != "fallback":
                sezione["assignment"] = {
                    "provider": provider_amm,
                    "model": str(getattr(args, "administrators_model", "") or ""),
                    "thinking": str(getattr(args, "administrators_thinking", "off") or "off"),
                }
        lista = list(getattr(args, "administrators_models", []) or [])
        if lista:
            pensiero = str(getattr(args, "administrators_thinking", "off") or "off")
            sezione["arm"] = "llm"
            sezione["follow_governor_arm"] = False
            sezione["assignments"] = []
            for voce in lista:
                provider_voce, sep, modello = str(voce).partition(":")
                if not sep or not provider_voce or not modello:
                    raise SystemExit(f"--administrators-models: '{voce}' non e' PROVIDER:MODELLO")
                sezione["assignments"].append({"provider": provider_voce, "model": modello, "thinking": pensiero})
        blocco["administrators"] = sezione
    config["governors"] = blocco


def _apply_overrides(config: dict, raw_overrides: str | None) -> None:
    if not raw_overrides:
        return
    overrides = json.loads(raw_overrides)
    if not isinstance(overrides, dict):
        raise ValueError("--overrides must be a JSON object")
    _deep_update(config, overrides)


def _fmt(value, width: int, decimals: int | None = None) -> str:
    if value is None:
        return "-".rjust(width)
    if decimals is not None:
        return f"{float(value):.{decimals}f}".rjust(width)
    return str(value).rjust(width)


def _execute_run(config: dict) -> None:
    print("Inizializzazione simulazione (mondo, clima MCD, agenti)...")
    controller = SimulationController(config)
    controller.created_by = "cli"
    colony_cfg = config.get("colony", {}) if isinstance(config.get("colony"), dict) else {}
    site = f"({colony_cfg.get('start_x')}, {colony_cfg.get('start_y')})"
    max_steps = config.get("days")
    days_per_step = (config.get("simulation") or {}).get("days_per_step", 7)
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents"), dict) else {}

    print()
    print("--- Avvio run headless ---")
    print(f"  Run:            {config.get('name', '')}  (run_id: {controller.run_id})")
    print(f"  Seed:           {config.get('seed', 0)}")
    print(f"  Agenti:         {agents_cfg.get('count', '?')} (LLM: {agents_cfg.get('llm_count', 0)})")
    print(f"  Sito colonia:   {site}")
    print(f"  Passo:          {days_per_step} giorni/step, step massimi: {max_steps if max_steps is not None else 'nessun limite'}")
    print(f"  Artifact:       outputs/runs/{controller.run_id}")
    print()

    previous_dead = 0
    last_printed_step = 0
    try:
        while True:
            controller.step()
            # step() may set stop_reason right after executing the final step:
            # print that step's progress line before deciding to exit.
            if controller.step_index > last_printed_step:
                last_printed_step = controller.step_index
                state = controller.current_state()
                metrics = state.get("metrics", {})
                dead_total = len(getattr(controller, "dead_agents", []))
                deaths = dead_total - previous_dead
                previous_dead = dead_total
                line = (
                    f"Step {_fmt(controller.step_index, 5)}"
                    f" | Anno {_fmt(state.get('day', 0) / 365.25, 7, 1)}"
                    f" | Pop {_fmt(metrics.get('population', 0), 4)}"
                    f" | Salute {_fmt(metrics.get('average_agent_health', 0.0), 5, 2)}"
                    f" | Cibo {_fmt(metrics.get('food_stock'), 7, 1)}"
                    f" | Acqua {_fmt(metrics.get('water_stock'), 7, 1)}"
                )
                if deaths > 0:
                    line += f" | MORTI +{deaths}"
                active_event = getattr(controller, "active_event", None)
                if isinstance(active_event, dict) and active_event.get("type"):
                    line += f" | Evento: {active_event.get('type')}"
                print(line, flush=True)
            if controller.stop_reason:
                break
    except KeyboardInterrupt:
        print("\nInterruzione manuale: salvo gli artifact parziali...")
        controller.stop()
        print(f"Artifact salvati in outputs/runs/{controller.last_saved_run_id} (stop_reason=stopped_by_user)")
        print("La cartella e' importabile in analysis-frontend come una run completa.")
        return
    except Exception as exc:
        print(f"Simulation failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    final_population = len(controller.agents)
    dead_total = len(getattr(controller, "dead_agents", []))
    print()
    print("--- Run completata ---")
    print(f"  Motivo stop:        {controller.stop_reason}")
    print(f"  Step eseguiti:      {controller.step_index}")
    print(f"  Giorni simulati:    {controller.world.day} (~{controller.world.day / 365.25:.1f} anni)")
    print(f"  Popolazione finale: {final_population} (morti: {dead_total})")
    print(f"  Artifact:           {controller.output_dir if controller.output_dir else 'outputs/runs/' + controller.run_id}")
    print("  Importa la cartella in analysis-frontend (npm run dev) per l'analisi completa e il replay.")


def run_headless():
    parser = build_parser()
    args = parser.parse_args()
    if (args.colony_x is None) != (args.colony_y is None):
        parser.error("--colony-x e --colony-y vanno usati insieme")

    interactive = args.interactive or len(sys.argv) == 1
    if interactive:
        from src.cli.config_wizard import run_wizard

        options = run_wizard(_options_from_args(args))
        if options is None:
            print("Nessuna run avviata.")
            return
    else:
        if not args.run_name:
            parser.error("--run-name e' obbligatorio in modalita' non interattiva (oppure lancia senza argomenti per il menu interattivo)")
        options = _options_from_args(args)
        if args.steps is None:
            options.max_steps = 100  # default storico della modalita' non interattiva

    config = build_manual_config(options)
    _apply_agent_logs(config, getattr(args, "agent_logs", True))
    _apply_governors(config, args)
    # Gli overrides restano l'ultima parola: chi ne passa uno sta correggendo a
    # mano, e deve poter correggere anche cio' che le opzioni hanno appena
    # scritto.
    _apply_overrides(config, args.overrides)
    _execute_run(config)


if __name__ == "__main__":
    run_headless()
