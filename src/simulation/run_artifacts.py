from __future__ import annotations

"""Unified run-artifact writer shared by BOTH simulation engines.

A run launched from the GUI (SimulationController) and one launched from the
terminal (AgentCoupledRunner / scripts/headless_runner.py) must leave behind
the exact same file set, so the analysis-frontend can load either
interchangeably.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from src.agents.build_policy import COLONISTS_PER_STRUCTURE
from src.social_network.metrics import compute_social_metrics
from src.world.occupancy import housing_slots_from_structures
from src.world.structures import StructureType

PASSIVE_ACTIONS = frozenset({"observe", "do_nothing"})


def action_productivity_metrics(
    action_counts,
    rejection_counts,
    *,
    accepted: int,
) -> dict[str, float]:
    """Accepted-action share excluding passive observation/no-op choices."""
    rejected_by_action: dict[str, int] = {}
    for key, count in (rejection_counts or {}).items():
        action, _reason = key
        action = str(action)
        rejected_by_action[action] = rejected_by_action.get(action, 0) + int(count)
    productive = sum(
        max(0, int(count) - rejected_by_action.get(str(action), 0))
        for action, count in (action_counts or {}).items()
        if str(action) not in PASSIVE_ACTIONS
    )
    accepted = max(0, int(accepted))
    return {
        "productive_actions_accepted": float(productive),
        "productive_action_rate": productive / max(1, accepted),
        "passive_action_rate": max(0, accepted - productive) / max(1, accepted),
    }


def build_action_summary(
    action_counts,
    rejection_counts,
    *,
    attempted: int,
    rejected: int,
) -> dict:
    """Build a compact lossless rejection summary independent of row logs."""
    by_action: dict[str, dict] = {}
    for action, count in sorted((action_counts or {}).items()):
        by_action[str(action)] = {
            "attempted": int(count),
            "accepted": int(count),
            "rejected": 0,
            "rejection_reasons": {},
        }
    for key, count in sorted((rejection_counts or {}).items()):
        action, reason = key
        row = by_action.setdefault(
            str(action),
            {"attempted": int(count), "accepted": 0, "rejected": 0, "rejection_reasons": {}},
        )
        row["rejected"] += int(count)
        row["accepted"] = max(0, int(row["attempted"]) - int(row["rejected"]))
        row["rejection_reasons"][str(reason)] = int(count)
    accepted = max(0, int(attempted) - int(rejected))
    return {
        "attempted": int(attempted),
        "accepted": accepted,
        "rejected": int(rejected),
        **action_productivity_metrics(
            action_counts,
            rejection_counts,
            accepted=accepted,
        ),
        "by_action": by_action,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # **L'unione delle chiavi, non quelle della prima riga (2026-08-30).**
    # `DictWriter` solleva `ValueError` appena una riga porta un campo che la
    # prima non aveva: bastava aggiungere una metrica calcolata solo in certe
    # condizioni per far fallire il salvataggio dell'intera run a fine
    # esecuzione, cioe' nel momento peggiore. L'ordine di prima apparizione
    # tiene stabile l'intestazione per chi confronta due file.
    intestazione: dict[str, None] = {}
    for riga in rows:
        for chiave in riga:
            intestazione.setdefault(chiave, None)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(intestazione), restval="")
        writer.writeheader()
        writer.writerows(rows)


def world_static_base_payload(world) -> dict:
    cells = []
    for row in world.cells:
        for cell in row:
            data = cell.to_public_dict()
            data["agents_present"] = []
            data["agent_positions_m"] = {}
            data["structures"] = []
            data["construction_sites"] = {}
            cells.append(data)
    return {
        "width": world.width,
        "height": world.height,
        "day": world.day,
        "metadata": dict(world.metadata),
        "cells": cells,
    }


def build_replay_event(step: int, day: float, agent, request, result) -> dict:
    target = getattr(request, "target", None)
    return {
        "step": step,
        "day": day,
        "agent_id": getattr(agent, "agent_id", ""),
        "agent_name": getattr(agent, "name", ""),
        "mode": getattr(agent, "mode", "rule_based"),
        "action": request.action.value,
        "x": getattr(agent, "x", None),
        "y": getattr(agent, "y", None),
        "local_x_m": getattr(agent, "local_x_m", 0.0),
        "local_y_m": getattr(agent, "local_y_m", 0.0),
        "target": target if isinstance(target, dict) else {},
        "result_data": getattr(result, "data", {}),
        "message": getattr(request, "message", None) or result.message,
    }


def build_cell_infrastructure_summary(world) -> list[dict]:
    """Final auditable life-support coverage and warehouse stock per cell."""
    rows: list[dict] = []
    for cell_row in world.cells:
        for cell in cell_row:
            population = len(cell.agents_present)
            counts = {structure_type: 0 for structure_type in StructureType}
            for structure in cell.structures:
                counts[structure.type] += 1
            if population <= 0 and not any(counts.values()) and not cell.construction_sites:
                continue
            housing = float(
                housing_slots_from_structures(
                    counts[StructureType.SHELTER],
                    counts[StructureType.HABITAT],
                    counts[StructureType.INFIRMARY],
                )
            )
            targets = {
                "housing": float(population),
                "greenhouse": float(-(-population // COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE])) if population else 0.0,
                "solar_array": float(-(-population // COLONISTS_PER_STRUCTURE[StructureType.SOLAR_ARRAY])) if population else 0.0,
                "oxygen_plant": float(-(-population // COLONISTS_PER_STRUCTURE[StructureType.OXYGEN_PLANT])) if population else 0.0,
            }
            actual = {
                "housing": float(housing),
                "greenhouse": float(counts[StructureType.GREENHOUSE]),
                "solar_array": float(counts[StructureType.SOLAR_ARRAY]),
                "oxygen_plant": float(counts[StructureType.OXYGEN_PLANT]),
            }
            deficits = {
                name: max(0.0, targets[name] - actual[name]) for name in targets
            }
            construction_sites = []
            extraction_sites = []
            for site_key, progress in sorted(cell.construction_sites.items()):
                raw_key = str(site_key)
                if raw_key == "ice_extraction":
                    # Backward-compatible visibility for worlds created by
                    # the former multi-step COLLECT_ICE implementation.  This
                    # is resource work, never an unfinished building.
                    extraction_sites.append(
                        {
                            "site_key": raw_key,
                            "resource": "ice",
                            "progress_percent": float(progress),
                        }
                    )
                    continue
                structure_type, separator, raw_position = raw_key.partition("@")
                local_x_m = None
                local_y_m = None
                if separator:
                    try:
                        raw_x, raw_y = raw_position.split(":", 1)
                        local_x_m = float(raw_x)
                        local_y_m = float(raw_y)
                    except ValueError:
                        # Preserve malformed/legacy keys in the artifact even
                        # when their local position cannot be decoded.
                        pass
                construction_sites.append(
                    {
                        "site_key": raw_key,
                        "structure_type": structure_type,
                        "progress_percent": float(progress),
                        "local_x_m": local_x_m,
                        "local_y_m": local_y_m,
                    }
                )
            rows.append(
                {
                    "x": int(cell.x),
                    "y": int(cell.y),
                    "population": int(population),
                    "targets": targets,
                    "actual": actual,
                    "deficits": deficits,
                    "life_support_covered": not any(deficits.values()),
                    "open_construction_site_count": len(construction_sites),
                    "construction_sites": construction_sites,
                    "open_extraction_site_count": len(extraction_sites),
                    "extraction_sites": extraction_sites,
                    "warehouse": {
                        resource: float(getattr(cell.resources, resource, 0.0))
                        for resource in (
                            "water",
                            "food",
                            "oxygen",
                            "construction_material",
                            "minerals",
                            "energy",
                        )
                    },
                }
            )
    return sorted(rows, key=lambda row: (-row["population"], row["y"], row["x"]))


def _cache_metrics(world) -> dict[str, float]:
    cells = getattr(world, "_cells", None)
    if cells is not None and hasattr(cells, "cache_metrics"):
        return cells.cache_metrics()
    return {
        "structure_view_hits": 0.0,
        "structure_view_misses": 0.0,
        "structure_position_hits": 0.0,
        "structure_position_misses": 0.0,
        "structure_invalidated_entries": 0.0,
        "structure_cache_hit_rate": 0.0,
        "structure_cached_cell_count": 0.0,
    }


def _map_profiles(world, config: dict) -> tuple[str, str]:
    world_cfg = config.get("world", {}) if isinstance(config.get("world"), dict) else {}
    configured = str(world_cfg.get("map_profile") or config.get("map_profile") or "balanced")
    return (
        str(world.metadata.get("map_profile", configured)),
        str(world.metadata.get("requested_map_profile", configured)),
    )


def research_summary_markdown(summary: dict) -> str:
    metrics = summary.get("final_colony_environment_state", summary.get("final_planetary_state", {}))
    social = summary.get("social_dynamics", {})
    ops = summary.get("operational_readiness", {})
    perf = summary.get("agent_performance", {})
    params = summary.get("simulation_parameters", {})

    return (
        "# Simulation Research Report\n\n"
        f"**Run ID:** {summary.get('run_id')}\n"
        f"**Date:** {summary.get('timestamp')}\n\n"
        "## 1. Simulation Parameters\n"
        f"- Total Steps: {params.get('steps')}\n"
        f"- Days per Step: {params.get('days_per_step')}\n"
        f"- Total Simulated Days: {float(params.get('total_simulated_days') or 0.0):.1f}\n\n"
        "## 2. Colony Environment Background\n"
        f"- Avg Habitability: {metrics.get('average_habitability', 0):.4f}\n"
        f"- Colony Environment Progress: {metrics.get('colony_environment_progress_index', metrics.get('planetary_terraforming_progress_index', 0)):.4f}\n"
        f"- Environmental Layer Enabled: {metrics.get('environmental_layer_enabled', 0):.0f}\n"
        f"- Total Vegetation: {metrics.get('vegetation', 0):.2f}\n\n"
        "## 3. Social Dynamics & Cooperation\n"
        f"- **Cooperation Index:** {social.get('cooperation_index', 0):.4f}\n"
        f"- Crew Stress Index: {social.get('crew_stress_index', 0):.4f}\n"
        f"- Cohesion Index: {social.get('cohesion_index', 0):.4f}\n"
        f"- Protocol Compliance: {social.get('protocol_compliance_index', 0):.4f}\n"
        f"- Communication Density: {social.get('communication_density', 0):.4f}\n"
        f"- Total Resource Sharing: {social.get('resource_sharing_events', 0)}\n\n"
        "## 4. Operational Readiness\n"
        f"- Life Support Reliability: {ops.get('life_support_reliability', 0):.4f}\n"
        f"- ECLSS Margin: {ops.get('eclss_margin', 0):.4f}\n"
        f"- ISRU Capacity Index: {ops.get('isru_capacity_index', 0):.4f}\n"
        f"- Mission Operational Readiness: {ops.get('mission_operational_readiness', 0):.4f}\n\n"
        "## 5. Agent Survival & Performance\n"
        f"- Survival Rate: {perf.get('survival_rate', 0) * 100:.1f}%\n"
        f"- Avg Health: {perf.get('avg_health', 0):.2f}\n"
        f"- Action Rejection Rate: {perf.get('rejection_rate', 0) * 100:.1f}%\n"
    )


def save_run_artifacts(
    output_dir: Path,
    *,
    run_id: str,
    config: dict,
    world,
    world_static_base: dict,
    agents: dict,
    dead_agents: list[dict],
    global_metrics: list[dict],
    validated_actions: list[dict],
    rejected_actions: list[dict],
    agent_decisions: list[dict],
    conversations: list[dict],
    thoughts: list[dict],
    replay_events: list[dict],
    social,
    cost_tracker,
    time_scale,
    current_metrics: dict,
    climate_metadata: dict,
    stop_reason: str = "",
    created_by: str = "",
    final: bool = True,
    active_event: dict | None = None,
    upcoming_event: dict | None = None,
    actions_attempted: int | None = None,
    actions_rejected: int | None = None,
    action_counts=None,
    rejection_counts=None,
    aggregate_state: dict | None = None,
    action_log_truncated: bool = False,
    replay_log_truncated: bool = False,
    replay_compactions: int = 0,
    store_memory_logs: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_profile, requested_map_profile = _map_profiles(world, config)
    attempted = int(actions_attempted if actions_attempted is not None else len(validated_actions) + len(rejected_actions))
    rejected_count = int(actions_rejected if actions_rejected is not None else len(rejected_actions))
    action_summary = build_action_summary(
        action_counts,
        rejection_counts,
        attempted=attempted,
        rejected=rejected_count,
    )

    (output_dir / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "created_by": created_by,
                "final": final,
                "stop_reason": stop_reason,
                "seed": int(config.get("seed", 0)),
                "map_profile": map_profile,
                "requested_map_profile": requested_map_profile,
                "climate": climate_metadata,
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Core data streams (JSONL for easy parsing of big data)
    write_jsonl(output_dir / "events.jsonl", world.events)
    write_jsonl(output_dir / "agent_decisions.jsonl", agent_decisions)
    write_jsonl(output_dir / "validated_actions.jsonl", validated_actions)
    write_jsonl(output_dir / "rejected_actions.jsonl", rejected_actions)
    (output_dir / "action_summary.json").write_text(
        json.dumps(action_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output_dir / "agent_conversations.jsonl", conversations)
    write_jsonl(output_dir / "agent_thoughts.jsonl", thoughts)
    ordered_replay_events = sorted(
        replay_events,
        key=lambda row: (int(row.get("step", 0) or 0), float(row.get("day", 0.0) or 0.0), str(row.get("agent_id", ""))),
    )
    write_jsonl(output_dir / "replay_events.jsonl", ordered_replay_events)
    (output_dir / "world_static_base.json").write_text(json.dumps(world_static_base, ensure_ascii=False), encoding="utf-8")
    if aggregate_state:
        write_jsonl(output_dir / "agent_states.jsonl", [{"day": world.day, "aggregate_population": aggregate_state["population"], "mode": "aggregate"}])
    else:
        write_jsonl(output_dir / "agent_states.jsonl", [{"day": world.day, **agent.to_dict()} for agent in agents.values()])
    write_jsonl(output_dir / "dead_agents.jsonl", dead_agents)

    # Timeseries (CSV for Excel/Pandas)
    write_csv(output_dir / "state_timeseries.csv", global_metrics)

    # Social Network Analysis export (the real graph, identical on both engines)
    (output_dir / "social_network.json").write_text(json.dumps(social.export_graph(), indent=2), encoding="utf-8")

    cell_infrastructure = build_cell_infrastructure_summary(world)
    populated_cells = [row for row in cell_infrastructure if row["population"] > 0]
    underserved_population = sum(
        row["population"] for row in populated_cells if not row["life_support_covered"]
    )
    final_metrics = {
        "run_id": run_id,
        **(global_metrics[-1] if global_metrics else {}),
        **current_metrics,
        **_cache_metrics(world),
        "infrastructure_cell_count": float(len(cell_infrastructure)),
        "populated_cell_count": float(len(populated_cells)),
        "life_support_covered_cell_count": float(
            sum(row["life_support_covered"] for row in populated_cells)
        ),
        "underserved_population": float(underserved_population),
        "open_construction_site_count": float(
            sum(row["open_construction_site_count"] for row in cell_infrastructure)
        ),
        "open_extraction_site_count": float(
            sum(row["open_extraction_site_count"] for row in cell_infrastructure)
        ),
        "local_life_support_population_coverage": 1.0
        - underserved_population / max(1.0, float(sum(row["population"] for row in populated_cells))),
    }
    (output_dir / "cell_infrastructure.json").write_text(
        json.dumps(cell_infrastructure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    social_metrics = compute_social_metrics(social)

    research_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "simulation_parameters": {
            "steps": len(global_metrics),
            "days_per_step": time_scale.days_per_step,
            "total_simulated_days": world.day,
            "seed": int(config.get("seed", 0)),
            "map_profile": map_profile,
            "requested_map_profile": requested_map_profile,
        },
        "final_colony_environment_state": final_metrics,
        "final_planetary_state": final_metrics,
        "extreme_events": {
            "active": active_event,
            "upcoming": upcoming_event,
        },
        "social_dynamics": {
            "cooperation_index": final_metrics.get("cooperation_index", social_metrics.get("cooperation_index", 0.0)),
            "logistics_cooperation_index": final_metrics.get("logistics_cooperation_index", 0.0),
            "communication_density": social_metrics.get("communication_frequency", 0.0),
            "crew_stress_index": final_metrics.get("crew_stress_index", 0.0),
            "crew_morale_index": final_metrics.get("crew_morale_index", 0.0),
            "cohesion_index": final_metrics.get("cohesion_index", 0.0),
            "conflict_risk_index": final_metrics.get("conflict_risk_index", 0.0),
            "protocol_compliance_index": final_metrics.get("protocol_compliance_index", 0.0),
            "governance_autonomy_score": final_metrics.get("governance_autonomy_score", 0.0),
            "total_interactions": len(conversations) + int(final_metrics.get("automatic_redistribution_active_steps", 0)),
            "resource_sharing_events": (
                sum(1 for a in validated_actions if a.get("action") == "share_resource")
                + int(final_metrics.get("automatic_redistribution_active_steps", 0))
            ),
            "automatic_resource_deposited_total": final_metrics.get("automatic_resource_deposited_total", 0.0),
            "automatic_resource_withdrawn_total": final_metrics.get("automatic_resource_withdrawn_total", 0.0),
            "automatic_intercell_flow_total": final_metrics.get("automatic_intercell_flow_total", 0.0),
            "automatic_intercell_flow_events": final_metrics.get("automatic_intercell_flow_events", 0.0),
        },
        "operational_readiness": {
            "life_support_reliability": final_metrics.get("life_support_reliability", 0.0),
            "eclss_margin": final_metrics.get("eclss_margin", 0.0),
            "power_margin": final_metrics.get("power_margin", 0.0),
            "isru_capacity_index": final_metrics.get("isru_capacity_index", 0.0),
            "mission_operational_readiness": final_metrics.get("mission_operational_readiness", 0.0),
        },
        "agent_performance": {
            "survival_rate": final_metrics.get("survival_rate", len(agents) / max(1, int((config.get("agents") or {}).get("count", 1)))),
            "avg_health": final_metrics.get("average_agent_health", sum(a.health for a in agents.values()) / max(1, len(agents))),
            "decisions_made": len(agent_decisions),
            "rejection_rate": rejected_count / max(1, attempted),
            "action_summary": action_summary,
        },
    }
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "research_summary.json").write_text(json.dumps(research_summary, indent=2), encoding="utf-8")

    (output_dir / "api_usage.json").write_text(json.dumps(cost_tracker.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "token_usage.json").write_text(json.dumps(cost_tracker.to_dict(), indent=2), encoding="utf-8")
    write_jsonl(output_dir / "llm_usage.jsonl", cost_tracker.call_records)
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "snapshots": "world_snapshots",
                "events": "events.jsonl",
                "replay_events": "replay_events.jsonl",
                "action_summary": "action_summary.json",
                "cell_infrastructure": "cell_infrastructure.json",
                "static_base": "world_static_base.json",
                "memory_logs_enabled": bool(store_memory_logs),
                "action_log_rows": len(validated_actions) + len(rejected_actions),
                "action_log_truncated": bool(action_log_truncated),
                "replay_rows": len(ordered_replay_events),
                "replay_last_step": max((int(row.get("step", 0) or 0) for row in ordered_replay_events), default=0),
                "replay_truncated": bool(replay_log_truncated),
                "replay_compactions": int(replay_compactions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(research_summary_markdown(research_summary), encoding="utf-8")
