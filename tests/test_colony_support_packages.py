from src.simulation.agent_coupled_runner import AgentCoupledRunner


def test_rich_initial_support_package_increases_prosperity_and_internal_growth(tmp_path):
    poor = _run_support_case(
        tmp_path,
        "poor",
        initial_inventory={"food": 1, "water": 1, "construction_material": 0, "tools": 0, "oxygen": 1, "energy": 1, "minerals": 0, "med_kits": 1},
        initial_structures={},
    )
    rich = _run_support_case(
        tmp_path,
        "rich",
        initial_inventory={"food": 12, "water": 12, "construction_material": 10, "tools": 4, "oxygen": 2, "energy": 2, "minerals": 4, "med_kits": 2},
        # Three habitats provide six local housing/support slots: four
        # founders plus at least one physically admissible birth. Two
        # habitats are exactly saturated and must no longer grow.
        # **E un pozzo (2026-09-01).** L'acqua e' entrata fra i termini della
        # capienza vitale: senza, la dotazione "ricca" non sostiene nessuno e
        # non nasce nessuno, e la prova misurerebbe la mancanza d'acqua invece
        # della differenza fra i due pacchetti.
        initial_structures={"habitat": 3, "greenhouse": 2, "solar_array": 2, "oxygen_plant": 1, "water_extractor": 1, "storage_depot": 1},
    )

    assert rich["population"] > rich["initial_agent_count"]
    assert poor["population"] == poor["initial_agent_count"]
    assert rich["colony_prosperity_index"] > poor["colony_prosperity_index"]
    assert rich["food_margin"] > poor["food_margin"]
    assert rich["material_margin"] > poor["material_margin"]
    assert rich["structures_built"] > poor["structures_built"]


def _run_support_case(tmp_path, name: str, initial_inventory: dict, initial_structures: dict) -> dict:
    runner = AgentCoupledRunner(
        {
            "name": name,
            "seed": 101,
            "days": 3,
            "simulation": {"days_per_step": 1, "max_days": 30},
            "environmental_layer": {"enabled": False},
            "world": {"width": 16, "height": 16, "map_profile": "balanced"},
            "agents": {"count": 4, "llm_count": 0, "initial_inventory": initial_inventory},
            "colony": {"initial_structures": initial_structures},
            "population": {
                "enabled": True,
                "max_agents": 8,
                "daily_spawn_probability": 1.0,
                "prosperity_growth_scale": 1.0,
                "min_habitability_for_growth": 0.0,
            },
        }
    )
    output_dir = tmp_path / name
    runner.run(days=3, output_dir=output_dir)
    return runner.global_metrics[-1]
