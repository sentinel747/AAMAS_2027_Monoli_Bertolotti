from __future__ import annotations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def planetary_metrics(world_metrics: dict) -> dict[str, float]:
    habitability = world_metrics.get("average_habitability", 0.0)
    vegetation = world_metrics.get("vegetation", 0.0)
    oxygen_plants = world_metrics.get("oxygen_plants", 0.0)
    background_progress = min(1.0, habitability * 0.6 + vegetation * 0.001 + oxygen_plants * 0.02)
    return {
        "habitability_index": habitability,
        "colony_environment_progress_index": background_progress,
        "terraforming_progress_index": background_progress,
        "ecological_risk": max(0.0, 1.0 - habitability),
        "resource_depletion": 1.0 / (1.0 + world_metrics.get("total_ice", 0.0) + world_metrics.get("total_minerals", 0.0)),
    }


def agent_metrics(
    agents: dict,
    actions: list[dict],
    initial_agent_count: int | None = None,
    initial_agent_ids: set | None = None,
) -> dict[str, float]:
    accepted = [row for row in actions if row.get("accepted")]
    structures = [row for row in actions if str(row.get("action", "")).startswith("build_")]
    initial_count = max(1, int(initial_agent_count or len(agents) or 1))
    rejection_rate = (len(actions) - len(accepted)) / max(1, len(actions))
    # Survival tracks the founding cohort: births can push the population above
    # the initial count. Without cohort ids (older configs) the ratio is capped.
    if initial_agent_ids:
        survival_rate = sum(
            1 for aid, a in agents.items() if aid in initial_agent_ids and a.health > 0
        ) / max(1, len(initial_agent_ids))
    else:
        survival_rate = min(1.0, sum(1 for a in agents.values() if a.health > 0) / initial_count)
    return {
        "initial_agent_count": float(initial_count),
        "survival_rate": survival_rate,
        "average_resources": sum(sum(a.inventory.to_dict().values()) for a in agents.values()) / max(1, len(agents)),
        "action_diversity": len({row.get("action") for row in accepted}),
        "actions_attempted": float(len(actions)),
        "actions_accepted": float(len(accepted)),
        "action_acceptance_rate": len(accepted) / max(1, len(actions)),
        "action_rejection_rate": rejection_rate,
        "structures_built_by_agents": float(len(structures)),
        "communication_frequency": float(sum(1 for row in accepted if row.get("action") == "communicate")),
    }


def composite_agent_score(metrics: dict) -> dict[str, float]:
    """Common comparative score for rule-based, mixed and LLM runs.

    The score intentionally uses fields produced by both the CLI experiment runner
    and the live GUI controller. Missing fields degrade to neutral/zero values so
    old configs and partial runs remain readable.
    """
    population = float(metrics.get("population", 0.0))
    initial = float(metrics.get("initial_agent_count") or metrics.get("configured_agent_count") or max(1.0, population))
    survival_rate = metrics.get("survival_rate")
    if survival_rate is None:
        survival_rate = min(1.0, population / max(1.0, initial))
    avg_health = metrics.get("average_agent_health")
    if avg_health is None:
        avg_health = metrics.get("avg_health", survival_rate)
    survival_component = _clamp01(0.78 * float(survival_rate) + 0.22 * float(avg_health))

    background_progress = float(
        metrics.get(
            "colony_environment_progress_index",
            metrics.get(
                "planetary_terraforming_progress_index",
                metrics.get("terraforming_progress_index", 0.0),
            ),
        )
    )
    colony_prosperity = float(
        metrics.get(
            "colony_prosperity_index",
            metrics.get("terraforming_progress_index", 0.0),
        )
    )
    habitability = float(metrics.get("average_habitability", metrics.get("habitability_index", 0.0)))
    structures = float(metrics.get("structures_built", metrics.get("structures_built_by_agents", 0.0)))
    food_margin = float(metrics.get("food_margin", 0.0))
    material_margin = float(metrics.get("material_margin", 0.0))
    colony_component = _clamp01(
        0.34 * colony_prosperity
        + 0.22 * _clamp01(food_margin)
        + 0.18 * _clamp01(material_margin)
        + 0.14 * _clamp01(structures / 20.0)
        + 0.12 * _clamp01(background_progress + habitability)
    )

    social_stability = float(metrics.get("social_stability_score", 0.0))
    cooperation = float(metrics.get("cooperation_index", 0.0))
    communication = float(metrics.get("communication_frequency", 0.0))
    cohesion = float(metrics.get("cohesion_index", social_stability))
    conflict_risk = float(metrics.get("conflict_risk_index", 0.0))
    # cooperation_index is already 0..1 (per-relationship average);
    # communication_frequency stays a lifetime counter, so it is compared as a
    # per-colonist per-step rate (~0.35 conversations/colonist/step = full).
    steps_for_rates = max(1.0, float(metrics.get("step", metrics.get("steps", 1.0)) or 1.0))
    communication_rate = communication / steps_for_rates / max(1.0, population)
    cooperation_component = _clamp01(
        0.35 * social_stability
        + 0.25 * cohesion
        + 0.20 * _clamp01(cooperation)
        + 0.15 * _clamp01(communication_rate / 0.35)
        - 0.15 * conflict_risk
    )

    acceptance = metrics.get("action_acceptance_rate")
    if acceptance is None:
        attempted = float(metrics.get("actions_attempted", 0.0))
        rejected = float(metrics.get("actions_rejected", 0.0))
        acceptance = 1.0 - rejected / max(1.0, attempted)
    step = max(1.0, float(metrics.get("step", metrics.get("steps", 1.0)) or 1.0))
    progress_per_step = _clamp01(max(background_progress, colony_prosperity) / step * 50.0)
    llm_cost = float(metrics.get("estimated_cost_usd", 0.0))
    cost_penalty = _clamp01(llm_cost / 10.0)
    efficiency_component = _clamp01(
        0.52 * _clamp01(float(acceptance))
        + 0.20 * progress_per_step
        + 0.10 * _clamp01(float(metrics.get("action_diversity", 0.0)) / 10.0)
        + 0.10 * _clamp01(float(metrics.get("protocol_compliance_index", 0.0)))
        + 0.08 * _clamp01(float(metrics.get("mission_operational_readiness", 0.0)))
        - 0.20 * cost_penalty
    )

    score = _clamp01(
        0.35 * survival_component
        + 0.30 * colony_component
        + 0.20 * cooperation_component
        + 0.15 * efficiency_component
    )
    return {
        "survival_component": survival_component,
        "colony_component": colony_component,
        "terraforming_component": colony_component,
        "cooperation_component": cooperation_component,
        "efficiency_component": efficiency_component,
        "composite_agent_score": score,
    }


def emergence_indicators(metrics: dict) -> dict[str, bool]:
    return {
        "settlement_clustering": metrics.get("structures_built", 0) >= 3,
        "trade_network_emergence": metrics.get("cooperation_index", 0) > 0.15,
        "leader_or_hub_emergence": metrics.get("network_density", 0) > 0.4,
        "environmental_protection_behavior": metrics.get("greenhouses", 0) > 0,
    }
