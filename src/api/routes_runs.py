from pathlib import Path
import csv
import datetime as _dt
import json
import os
import re
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.visualization.replay import load_replay_manifest

router = APIRouter()

RUNS_ROOT = Path("outputs/runs")

#: Le radici in cui cercare le run, in ordine di precedenza.
#:
#: **Non solo `outputs/runs`.** La GUI scrive li', ma gli esperimenti da
#: terminale scrivono dove dice `--out`, e per l'intera campagna dei governatori
#: quello e' stato `runs/`: quaranta run vere, nessuna analizzabile dal frontend,
#: perche' l'elenco guardava una cartella sola. `MARSABM_RUNS_ROOT` le
#: sovrascrive entrambe (piu' percorsi separati come nel PATH del sistema).
DEFAULT_RUNS_ROOTS = ("outputs/runs", "runs")

#: Separatore fra cartella d'esperimento e run dentro l'identificatore.
#:
#: Non `/` di proposito: un identificatore con la barra renderebbe ambigua la
#: rotta `/api/runs/{run_id}/files/{file_path:path}` e costringerebbe a fidarsi
#: di come il server decodifica `%2F`. Con un carattere che nelle rotte non
#: compare, l'ambiguita' non esiste e l'identificatore resta leggibile.
RUN_SEPARATOR = "~"

#: Quante righe di uno stream finiscono nel payload dell'analisi.
#:
#: Con i log per azione accesi una run da 300 agenti per 800 passi scrive circa
#: 260 MB di `validated_actions.jsonl` (misurato: 59,5 MB a 200 passi).
#: Materializzarle tutte renderebbe inutilizzabile proprio il frontend che i log
#: dovevano servire. Il **conteggio** resta quello vero — un totale tagliato al
#: tetto direbbe che la colonia ha agito meno di quanto ha agito — e le righe
#: oltre il tetto restano leggibili dalla rotta `/files`, che serve un file alla
#: volta.
MAX_STREAM_ROWS = 5000


def _run_roots() -> list[Path]:
    raw = os.environ.get("MARSABM_RUNS_ROOT", "")
    if raw.strip():
        return [Path(p) for p in raw.split(os.pathsep) if p.strip()]
    return [Path(p) for p in DEFAULT_RUNS_ROOTS]


#: Fino a quanti livelli sotto una radice si cerca una run. Le campagne da
#: terminale scrivono `runs/<campagna>/<braccio>/<run>`: tre livelli. Fino al
#: 2026-09-06 se ne leggevano due, e con le radici di default la tendina
#: mostrava solo le run della GUI: per vedere una campagna bisognava puntare
#: `MARSABM_RUNS_ROOT` alla sua cartella, una campagna alla volta.
MAX_RUN_DEPTH = 3


def _is_run_dir(path: Path) -> bool:
    """Una cartella e' una run se porta gli artefatti che ogni run scrive.

    Serve a distinguere `runs/campagna/` (contenitore) da
    `runs/campagna/llm_seed0/` (run), senza dover conoscere i nomi delle
    campagne.
    """
    return (path / "run_metadata.json").exists() or (path / "final_metrics.json").exists()


def discover_runs() -> dict[str, Path]:
    """Identificatore -> cartella, cercando anche un livello di annidamento.

    A parita' di identificatore vince la prima radice: `outputs/runs` ha la
    precedenza su `runs/`, cosi' una run aperta dalla GUI resta quella che la
    GUI ha scritto.
    """
    trovate: dict[str, Path] = {}
    roots = _run_roots()
    # **Con piu' radici l'identificatore porta il nome della radice.** Tre
    # campagne (`campagna_scarsa`, `_v2`, `_v3`) hanno tutte
    # `llm_completo_amm/llm_completo+amm_seed3`: senza il prefisso vince la
    # prima e le altre due non si possono aprire, e chi guarda crede di
    # leggere una run mentre ne legge un'altra (misurato il 2026-09-06). Con
    # una radice sola gli identificatori restano quelli di sempre, cosi' le
    # run aperte dalla GUI non cambiano nome. La coppia di default
    # (`outputs/runs` e `runs`) non conta come "piu' radici": li' la
    # precedenza della prima e' voluta, e il prefisso cambierebbe il nome di
    # ogni run della GUI.
    prefisso_radice = bool(os.environ.get("MARSABM_RUNS_ROOT", "").strip()) and len(roots) > 1
    for root in roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
            testa = f"{root.name}{RUN_SEPARATOR}" if prefisso_radice else ""
            _raccogli_run(root, testa, 1, trovate)
        except OSError:
            # Una radice illeggibile non deve far fallire l'elenco delle altre.
            continue
    return trovate


def _raccogli_run(cartella: Path, prefisso: str, profondita: int, trovate: dict[str, Path]) -> None:
    """Scende nelle cartelle-contenitore finche' trova run, fino a `MAX_RUN_DEPTH`.

    Una run non viene attraversata (dentro ci sono `world_snapshots/` e simili,
    che non sono run); una cartella senza run dentro non compare.
    """
    for child in sorted(cartella.iterdir()):
        if not child.is_dir():
            continue
        nome = f"{prefisso}{child.name}"
        if _is_run_dir(child):
            trovate.setdefault(nome, child)
            continue
        if profondita < MAX_RUN_DEPTH:
            try:
                _raccogli_run(child, f"{nome}{RUN_SEPARATOR}", profondita + 1, trovate)
            except OSError:
                continue


def _run_updated_at(path: Path) -> str:
    """Quando la run e' stata scritta l'ultima volta, ISO 8601 senza fuso.

    Prima `updated_at` di `run_metadata.json` (lo scrive ogni run, alla fine e
    a ogni salvataggio); se manca, la data di modifica del file piu' recente
    fra metriche finali e metadati. Vuoto solo se non c'e' niente di leggibile.
    """
    meta = path / "run_metadata.json"
    try:
        valore = json.loads(meta.read_text(encoding="utf-8")).get("updated_at")
        if valore:
            return str(valore)
    except (OSError, ValueError, AttributeError):
        pass
    tempi = []
    for nome in ("final_metrics.json", "run_metadata.json"):
        try:
            tempi.append((path / nome).stat().st_mtime)
        except OSError:
            continue
    if not tempi:
        return ""
    return _dt.datetime.fromtimestamp(max(tempi)).isoformat(timespec="seconds")


def _resolve_run(run_id: str) -> Path:
    percorso = discover_runs().get(run_id)
    if percorso is None:
        raise HTTPException(status_code=404, detail="run not found")
    return percorso.resolve()


@router.get("/api/runs")
def runs():
    # Filesystem-only: listing saved runs must never boot a simulation
    # (the lazy controller construction loads MCD data and a full world).
    # **Dalla piu' recente**, con l'ora: la tendina del frontend raggruppa per
    # giorno e ordina per ora di esecuzione, cosi' chi ha lanciato quaranta
    # campagne le trova nell'ordine in cui le ha fatte e non in quello dei nomi.
    righe = [
        {"run_id": run_id, "path": str(path), "updated_at": _run_updated_at(path)}
        for run_id, path in discover_runs().items()
    ]
    righe.sort(key=lambda r: (r["updated_at"], r["run_id"]), reverse=True)
    return righe


@router.get("/api/runs/importable")
def importable_runs(limit: int = 40):
    """Le run da cui ripartire, dalla piu' recente.

    Deve stare **prima** di `/api/runs/{run_id}`: FastAPI risolve le rotte in
    ordine di dichiarazione, e con l'ordine inverso "importable" verrebbe letto
    come identificatore di una run e restituirebbe sempre 404.
    """
    from src.experiments.run_import import list_importable_runs

    return [
        {
            "run_id": riassunto.run_id,
            "label": riassunto.label(),
            "name": riassunto.name,
            "seed": riassunto.seed,
            "steps": riassunto.steps,
            "agents": riassunto.agents,
            "map_profile": riassunto.map_profile,
            "governors": riassunto.governors,
        }
        for riassunto in list_importable_runs(limit=limit)
    ]


@router.get("/api/runs/{run_id}/config")
def run_config(run_id: str):
    """La configurazione con cui una run e' stata eseguita, per ripeterla."""
    from src.experiments.run_import import load_run_config

    percorso = _resolve_run(run_id)
    config = load_run_config(percorso)
    if config is None:
        raise HTTPException(status_code=404, detail="run has no readable config")
    return config


@router.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    percorso = discover_runs().get(run_id)
    metrics: dict[str, Any] = {}
    if percorso is not None:
        summary = percorso / "final_metrics.json"
        if summary.exists():
            try:
                metrics = json.loads(summary.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError):
                metrics = {}
    return {"run_id": run_id, "exists": percorso is not None, "metrics": metrics}


@router.get("/api/runs/{run_id}/analysis")
def run_analysis(run_id: str):
    return _load_run_analysis(run_id)


@router.get("/api/runs/{run_id}/replay")
def run_replay(run_id: str):
    return load_replay_manifest(str(_resolve_run(run_id)))


#: Le colonne della serie temporale che il confronto porta con se'. Un
#: sottoinsieme: `state_timeseries.csv` ha 140 colonne e mille righe per run,
#: e il confronto ne apre molte insieme.
COMPARE_COLUMNS = (
    "step", "day", "population", "survival_rate", "average_agent_health",
    "local_food_shortfall_index", "local_water_shortfall_index", "underfed_population_now",
    "colony_prosperity_index", "life_support_reliability", "mission_operational_readiness",
    "colony_site_score", "food_stock", "water_stock", "material_stock", "tool_stock",
    "colony_area_m2", "colony_claimed_area_m2", "occupied_cells", "structure_cells",
    "exploration_coverage", "structures_built", "earth_input_sufficiency_index",
    "colony_density_per_km2", "structure_integrity_mean", "average_habitability",
    "vegetation", "resource_pooling_index", "cohesion_index", "cooperation_index",
    "initial_agent_count",
)
#: Quante righe al massimo per serie: a schermo un grafico largo mille pixel
#: non distingue piu' di trecento punti, e il payload resta sotto i 100 KB.
COMPARE_MAX_ROWS = 300
_SNAPSHOT_NAME = re.compile(r"step_(\d+)_day_(\d+)\.json$")


def _numero(valore):
    try:
        numero = float(valore)
    except (TypeError, ValueError):
        return None
    return numero if numero == numero else None  # NaN fuori


def _serie_compatta(path: Path, colonne=COMPARE_COLUMNS, max_righe: int = COMPARE_MAX_ROWS) -> list[dict]:
    """La serie temporale ridotta alle colonne del confronto e campionata.

    Il campionamento tiene una riga ogni `k` e SEMPRE l'ultima: e' l'esito
    finale, e un confronto che lo perdesse per arrotondamento mentirebbe.
    """
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            lettore = csv.DictReader(handle)
            presenti = [c for c in colonne if c in (lettore.fieldnames or [])]
            righe = [
                {c: _numero(riga.get(c)) for c in presenti}
                for riga in lettore
            ]
    except OSError:
        return []
    if len(righe) <= max_righe:
        return righe
    passo = -(-len(righe) // max_righe)  # ceil
    campione = righe[::passo]
    if campione[-1] is not righe[-1]:
        campione.append(righe[-1])
    return campione


def _conta_righe(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for riga in handle if riga.strip())
    except OSError:
        return 0


def _ultima_riga_json(path: Path) -> dict:
    ultima: dict = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for riga in handle:
                if riga.strip():
                    try:
                        candidata = json.loads(riga)
                    except ValueError:
                        continue
                    if isinstance(candidata, dict):
                        ultima = candidata
    except OSError:
        pass
    return ultima


def _elenco_snapshot(run_path: Path) -> list[dict]:
    """Gli snapshot con il passo letto dal nome, in ordine di passo."""
    cartella = run_path / "world_snapshots"
    if not cartella.is_dir():
        return []
    voci = []
    try:
        for file in cartella.iterdir():
            m = _SNAPSHOT_NAME.search(file.name)
            if m and file.is_file():
                voci.append({
                    "step": int(m.group(1)),
                    "day": int(m.group(2)),
                    "path": f"world_snapshots/{file.name}",
                })
    except OSError:
        return []
    voci.sort(key=lambda v: v["step"])
    return voci


def _firma_dei_file(run_path: Path) -> list[str]:
    """I nomi dei file di primo livello: due run si confrontano se hanno la
    stessa struttura di output, e questa e' la struttura."""
    try:
        nomi = sorted(f.name for f in run_path.iterdir() if f.is_file())
    except OSError:
        return []
    if (run_path / "world_snapshots").is_dir():
        nomi.append("world_snapshots/")
    return nomi


def _leggi_json(path: Path) -> dict:
    try:
        dati = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return {}
    return dati if isinstance(dati, dict) else {}


@router.get("/api/runs/{run_id}/compare")
def run_compare(run_id: str):
    """Cio' che serve a mettere una run accanto alle altre, e niente di piu'.

    L'analisi intera pesa 5 MB per run (base statica compresa) e il confronto
    ne apre molte insieme: qui esiti finali, serie compatta, elenco degli
    snapshot e firma dei file. La base statica arriva a parte, e solo se si
    accendono i mini replay.
    """
    run_path = _resolve_run(run_id)
    finali = _leggi_json(run_path / "final_metrics.json")
    meta = _leggi_json(run_path / "run_metadata.json")
    token_usage = _leggi_json(run_path / "token_usage.json") or _leggi_json(run_path / "api_usage.json")
    api_usage = _leggi_json(run_path / "api_usage.json")
    serie = _serie_compatta(run_path / "state_timeseries.csv")
    prima = serie[0] if serie else {}
    ultima = serie[-1] if serie else {}
    morti = _conta_righe(run_path / "dead_agents.jsonl")
    vivi = _numero(finali.get("population")) or _numero(ultima.get("population")) or 0.0
    iniziali = (
        _numero(prima.get("population")) or _numero(finali.get("initial_agent_count"))
        or _numero(prima.get("initial_agent_count")) or 0.0
    )
    amministratori = _ultima_riga_json(run_path / "administrator_decisions.jsonl")
    parti = run_id.split(RUN_SEPARATOR)
    etichetta = {
        "campaign": parti[0] if len(parti) >= 3 else "",
        "arm": parti[-2] if len(parti) >= 2 else "",
        "run": parti[-1],
    }
    sopravvivenza = _numero(finali.get("survival_rate"))
    if sopravvivenza is None:
        sopravvivenza = _numero(ultima.get("survival_rate"))
    return {
        "run_id": run_id,
        "path": str(run_path),
        "updated_at": _run_updated_at(run_path),
        "label": etichetta,
        "seed": meta.get("seed"),
        "map_profile": meta.get("map_profile"),
        # L'ultimo passo della serie, non il numero di righe: la riga 0 e' il
        # passo 0, e mille passi sono 1001 righe.
        "steps": int(_numero(ultima.get("step")) or 0),
        "summary": {
            "population_final": int(vivi),
            "population_initial": int(iniziali),
            "deaths": morti,
            "births": max(0, int(vivi) + morti - int(iniziali)),
            "survival_rate": sopravvivenza,
            "governor_ticks": _conta_righe(run_path / "governor_decisions.jsonl"),
            "administrator_districts": int(amministratori.get("districts") or 0),
            "administrator_interventions": int(amministratori.get("interventions") or 0),
            "administrator_abstentions": int(amministratori.get("abstentions") or 0),
            "administrator_rewrites_without_effect": int(amministratori.get("rewrites_without_effect") or 0),
            "tokens_total": int(token_usage.get("tokens_total", 0) or 0),
            "api_calls": int(api_usage.get("calls", 0) or 0),
            "snapshots": len(_elenco_snapshot(run_path)),
        },
        "final_metrics": {
            k: v for k, v in finali.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
        },
        "timeseries": serie,
        "timeseries_columns": list(prima.keys()),
        "snapshots": _elenco_snapshot(run_path),
        "signature": _firma_dei_file(run_path),
    }


@router.get("/api/runs/{run_id}/static_base")
def run_static_base(run_id: str):
    """La base statica del mondo ridotta ai campi che il replay disegna."""
    run_path = _resolve_run(run_id)
    file = run_path / "world_static_base.json"
    if not file.exists():
        raise HTTPException(status_code=404, detail="run has no static base")
    try:
        base = json.loads(file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        raise HTTPException(status_code=500, detail="static base unreadable")
    return _base_statica_snella(base)


@router.get("/api/runs/{run_id}/files/{file_path:path}")
def run_file_content(run_id: str, file_path: str):
    run_path = _resolve_run(run_id)
    target_path = (run_path / file_path).resolve()

    if run_path != target_path and run_path not in target_path.parents:
        raise HTTPException(status_code=404, detail="file not found")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    if target_path.suffix.lower() == ".json":
        # **I byte, non l'oggetto.** Fino al 2026-09-06 il file veniva letto,
        # interpretato e riserializzato dal framework: uno snapshot compatto da
        # 615 KB impiegava 6-8 secondi, e il replay a cinque run in parallelo
        # ne impiegava undici per fotogramma. Il browser lo interpreta da solo;
        # un file corrotto lo si scopre li', come prima si scopriva qui.
        try:
            return Response(content=target_path.read_bytes(), media_type="application/json")
        except OSError as exc:
            raise HTTPException(status_code=404, detail=f"file unreadable: {file_path}") from exc
    elif target_path.suffix.lower() in {".jsonl", ".csv"}:
        # For large files, we might want to stream or limit, but for now simple parse
        return _parse_run_file(target_path, run_path)

    return {"content": target_path.read_text(encoding="utf-8", errors="replace")}


def _load_run_analysis(run_id: str) -> dict[str, Any]:
    run_path = _resolve_run(run_id)
    if not run_path.is_dir():
        raise HTTPException(status_code=404, detail="run not found")

    # Optimization: Separate snapshots from other artifacts to avoid parsing large world states in bulk
    all_paths = sorted(run_path.rglob("*"))
    files = []
    snapshots = []
    
    for p in all_paths:
        try:
            if not p.is_file():
                continue
            rel = p.relative_to(run_path).as_posix()
            if rel.startswith("world_snapshots/"):
                snapshots.append({
                    "path": rel,
                    "name": p.name,
                    "size_bytes": p.stat().st_size
                })
            else:
                files.append(_parse_run_file(p, run_path))
        except (OSError, FileNotFoundError):
            # A live run can rotate/delete files between listing and reading:
            # skip instead of failing the whole analysis payload.
            continue

    by_path = {file["path"]: file for file in files}
    final_metrics = _data_for(by_path, "final_metrics.json", {})
    api_usage = _data_for(by_path, "api_usage.json", {})
    token_usage = _data_for(by_path, "token_usage.json", api_usage)
    run_metadata = _data_for(by_path, "run_metadata.json", {})
    config = _data_for(by_path, "config.yaml", {})
    replay_static_base = _base_statica_snella(_data_for(by_path, "world_static_base.json", {}))
    events = _rows_for(by_path, "events.jsonl")
    decisions = _rows_for(by_path, "agent_decisions.jsonl")
    validated_actions = _rows_for(by_path, "validated_actions.jsonl")
    rejected_actions = _rows_for(by_path, "rejected_actions.jsonl")
    replay_events = _rows_for(by_path, "replay_events.jsonl") or _replay_events_from_actions(validated_actions)
    conversations = _rows_for(by_path, "agent_conversations.jsonl")
    thoughts = _rows_for(by_path, "agent_thoughts.jsonl")
    states = _rows_for(by_path, "agent_states.jsonl")
    dead_agents = _rows_for(by_path, "dead_agents.jsonl")
    llm_usage = _rows_for(by_path, "llm_usage.jsonl")
    # **Le decisioni del consiglio erano scritte e non lette.** Trecentosessanta
    # kilobyte per run — razionale di ogni governatore, proposta per mandato,
    # token, latenza, direttiva adottata — e nessun consumatore che le aprisse:
    # per una campagna il cui costo e' quasi tutto in chiamate al modello, era la
    # voce piu' cara e la meno visibile.
    governor_decisions = _rows_for(by_path, "governor_decisions.jsonl")
    # Le tornate dei distretti. Una riga per tornata, con la decisione di
    # ciascun amministratore: se ha accettato la politica del governo o se
    # l'ha riscritta, con quali regole e perche'. Senza questo, di una run
    # decentrata il frontend di analisi mostrerebbe soltanto l'esito, e
    # un'amministrazione che si astiene sempre sarebbe indistinguibile da
    # una run centralizzata.
    administrator_decisions = _rows_for(by_path, "administrator_decisions.jsonl")
    timeseries = _rows_for(by_path, "state_timeseries.csv")
    social_timeseries = _rows_for(by_path, "social_network_metrics.csv")

    return {
        "run_id": run_id,
        "path": str(run_path),
        "overview": {
            "files": len(files) + len(snapshots),
            "total_size_bytes": sum(int(f.get("size_bytes", 0)) for f in files) + sum(s["size_bytes"] for s in snapshots),
            "steps": _count_for(by_path, "state_timeseries.csv"),
            "events": _count_for(by_path, "events.jsonl"),
            "llm_decisions": _count_for(by_path, "agent_decisions.jsonl"),
            "validated_actions": _count_for(by_path, "validated_actions.jsonl"),
            "rejected_actions": _count_for(by_path, "rejected_actions.jsonl"),
            "replay_events": len(replay_events),
            "conversations": _count_for(by_path, "agent_conversations.jsonl"),
            "thoughts": _count_for(by_path, "agent_thoughts.jsonl"),
            "agent_state_rows": _count_for(by_path, "agent_states.jsonl"),
            "dead_agents": _count_for(by_path, "dead_agents.jsonl"),
            "llm_usage_rows": _count_for(by_path, "llm_usage.jsonl"),
            "governor_ticks": _count_for(by_path, "governor_decisions.jsonl"),
            "governor_directives": sum(
                1 for row in governor_decisions
                if (row.get("policy") or {}).get("rules")
                or (row.get("directive") or {}).get("cells")
            ),
            "governor_misses": sum(
                1 for row in governor_decisions if row.get("missed")
            ),
            "administrator_rounds": _count_for(by_path, "administrator_decisions.jsonl"),
            # Interventi e astensioni sono cumulativi nel registro: l'ultima
            # riga li porta gia' sommati, e sommarli di nuovo qui li
            # conterebbe una volta per tornata.
            "administrator_interventions": int(
                (administrator_decisions[-1].get("interventions") or 0)
                if administrator_decisions else 0
            ),
            "administrator_abstentions": int(
                (administrator_decisions[-1].get("abstentions") or 0)
                if administrator_decisions else 0
            ),
            "administrator_districts": int(
                (administrator_decisions[-1].get("districts") or 0)
                if administrator_decisions else 0
            ),
            "tokens_in": int(token_usage.get("tokens_in", 0) or 0) if isinstance(token_usage, dict) else 0,
            "tokens_out": int(token_usage.get("tokens_out", 0) or 0) if isinstance(token_usage, dict) else 0,
            "tokens_total": int(token_usage.get("tokens_total", 0) or 0) if isinstance(token_usage, dict) else 0,
            "snapshots": len(snapshots),
        },
        "metadata": run_metadata,
        "config": config,
        "final_metrics": final_metrics,
        "api_usage": api_usage,
        "token_usage": token_usage,
        "replay_static_base": replay_static_base,
        "insights": {
            "event_types": _count_by(events, "type"),
            "action_counts": _count_by(validated_actions + rejected_actions, "action"),
            "llm_providers": _count_by(decisions, "llm_provider"),
            "event_providers": _count_event_provider(events),
            "llm_errors": _llm_errors(events),
            "governor_mandates": _count_governor_mandates(governor_decisions),
            "latest_metrics": timeseries[-1] if timeseries else final_metrics,
            "first_metrics": timeseries[0] if timeseries else {},
            "timeseries_columns": list(timeseries[0].keys()) if timeseries else [],
            "social_columns": list(social_timeseries[0].keys()) if social_timeseries else [],
        },
        "streams": {
            "events": events,
            "agent_decisions": decisions,
            "validated_actions": validated_actions,
            "rejected_actions": rejected_actions,
            "replay_events": replay_events,
            "agent_conversations": conversations,
            "agent_thoughts": thoughts,
            "agent_states": states,
            "dead_agents": dead_agents,
            "llm_usage": llm_usage,
            "governor_decisions": governor_decisions,
            "administrator_decisions": administrator_decisions,
            "state_timeseries": timeseries,
            "social_network_metrics": social_timeseries,
        },
        "snapshots": snapshots,
        # **L'elenco dei file e' un elenco, non una seconda copia della run.**
        # Prima portava dentro anche `data`, `rows` e `text` di ogni file: su una
        # run vera il payload pesava 160 MB, di cui 80 erano la griglia del mondo
        # serializzata una seconda volta (la prima e' `replay_static_base`). Il
        # contenuto di un singolo file resta a un colpo di distanza, sulla rotta
        # `/api/runs/{id}/files/{percorso}`.
        "files": [
            {k: v for k, v in file.items() if k not in ("rows", "data", "text")}
            for file in files
        ],
    }



def _parse_run_file(path: Path, run_path: Path) -> dict[str, Any]:
    rel = path.relative_to(run_path).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    entry: dict[str, Any] = {
        "path": rel,
        "name": path.name,
        "folder": path.parent.relative_to(run_path).as_posix() if path.parent != run_path else "",
        "size_bytes": path.stat().st_size,
        "kind": suffix.lstrip(".") or "text",
        "row_count": 0,
        "columns": [],
    }
    try:
        if suffix == ".jsonl":
            righe_grezze = [line for line in text.splitlines() if line.strip()]
            totale = len(righe_grezze)
            rows = [_parse_json_line(line) for line in righe_grezze[:MAX_STREAM_ROWS]]
            rows = [row for row in rows if row is not None]
            entry.update({
                "kind": "jsonl", "rows": rows, "row_count": totale,
                "truncated": totale > len(rows), "columns": _columns(rows),
            })
        elif suffix == ".csv":
            tutte = list(csv.DictReader(text.splitlines()))
            totale = len(tutte)
            rows = tutte[:MAX_STREAM_ROWS]
            entry.update({
                "kind": "csv", "rows": rows, "row_count": totale,
                "truncated": totale > len(rows),
                "columns": list(tutte[0].keys()) if tutte else [],
            })
        elif suffix == ".json":
            data = json.loads(text) if text.strip() else {}
            rows = data if isinstance(data, list) else []
            entry.update({"kind": "json", "data": data, "row_count": len(rows), "columns": _columns(rows)})
        elif suffix in {".yaml", ".yml"}:
            entry.update({"kind": "yaml", "data": yaml.safe_load(text) or {}})
        else:
            entry.update({"kind": "markdown" if suffix == ".md" else "text", "text": text})
    except (json.JSONDecodeError, yaml.YAMLError, csv.Error, ValueError) as exc:
        # One half-written or legacy file must not 500 the whole analysis payload.
        entry.update({"kind": "error", "parse_error": str(exc)})
    return entry


def _parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return {"parse_error": line}
    return data if isinstance(data, dict) else {"value": data}


#: I soli campi di cella che il replay disegna: terreno e abitabilita' per il
#: colore, strutture e presenze per i marcatori, esplorata per il velo.
_CAMPI_CELLA_REPLAY = ("x", "y", "terrain", "habitability", "structures", "agents_present", "explored")


def _base_statica_snella(base: Any) -> Any:
    """La base geografica ridotta ai campi che il replay disegna.

    **Il numero.** Su 360x180 `world_static_base.json` pesa 73 MB, perche' ogni
    cella porta trenta campi (risorse, geometria, radiazione...). Allegata
    intera all'analisi, la risposta faceva 92 MB per run e il proxy di Vite si
    arrendeva prima che arrivasse (misurato il 2026-09-06). Ridotta a sette
    campi pesa 5 MB. Il file intero resta leggibile dalla rotta `/files`.
    """
    if not isinstance(base, dict) or not isinstance(base.get("cells"), list):
        return base
    snella = dict(base)
    snella["cells"] = [
        {k: c[k] for k in _CAMPI_CELLA_REPLAY if k in c}
        for c in base["cells"]
        if isinstance(c, dict)
    ]
    return snella


def _data_for(files: dict[str, dict[str, Any]], path: str, default: Any) -> Any:
    file = files.get(path)
    return file.get("data", default) if file else default


def _rows_for(files: dict[str, dict[str, Any]], path: str) -> list[dict[str, Any]]:
    file = files.get(path)
    return list(file.get("rows", [])) if file else []


def _count_for(files: dict[str, dict[str, Any]], path: str) -> int:
    """Quante righe ha il file, non quante ne sono state materializzate.

    `_rows_for` restituisce al massimo `MAX_STREAM_ROWS` righe: usarne la
    lunghezza per l'overview trasformerebbe il tetto di trasporto in un dato
    sulla simulazione.
    """
    file = files.get(path)
    return int(file.get("row_count", 0)) if file else 0


def _columns(rows: list[Any]) -> list[str]:
    columns: list[str] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    return columns


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _count_event_provider(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        data = event.get("data", {})
        provider = data.get("provider") if isinstance(data, dict) else None
        if provider:
            counts[str(provider)] = counts.get(str(provider), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _count_governor_mandates(rows: list[dict[str, Any]]) -> dict[str, int]:
    """In quanti tick il governatore ha davvero proposto qualcosa.

    Dal ridisegno 2026-08-24 non esistono piu' mandati: per un registro v2 il
    conteggio e' un'unica voce, `governor`, con i tick la cui proposta conteneva
    almeno una regola. Le righe del vecchio consiglio (lista `governors` con
    proposte per cella) restano leggibili con il conteggio per mandato di
    allora: un registro storico non deve diventare illeggibile.
    """
    counts: dict[str, int] = {}
    for row in rows:
        governor = row.get("governor")
        if isinstance(governor, dict):
            proposal = governor.get("proposal")
            if isinstance(proposal, dict) and proposal.get("rules"):
                counts["governor"] = counts.get("governor", 0) + 1
            continue
        for member in row.get("governors") or []:
            if not isinstance(member, dict):
                continue
            proposal = member.get("proposal")
            if isinstance(proposal, dict) and not proposal.get("cells"):
                continue
            mandate = str(member.get("mandate") or "sconosciuto")
            counts[mandate] = counts.get(mandate, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _llm_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        data = event.get("data", {})
        error = data.get("error") if isinstance(data, dict) else ""
        if error:
            rows.append({"day": event.get("day"), "step": event.get("step"), "message": event.get("message"), "error": error})
    return rows


def _replay_events_from_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in actions:
        rows.append(
            {
                "step": row.get("step"),
                "day": row.get("day"),
                "agent_id": row.get("agent_id"),
                "agent_name": row.get("agent_name"),
                "mode": row.get("mode"),
                "action": row.get("action"),
                "x": row.get("x"),
                "y": row.get("y"),
                "local_x_m": row.get("local_x_m"),
                "local_y_m": row.get("local_y_m"),
                "target": row.get("target") if isinstance(row.get("target"), dict) else {},
                "result_data": row.get("result_data") if isinstance(row.get("result_data"), dict) else {},
                "message": row.get("request_message") or row.get("message"),
            }
        )
    return rows
