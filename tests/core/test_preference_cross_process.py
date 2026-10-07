import json
import subprocess
import sys
import textwrap


_WORKER = textwrap.dedent(
    """
    import json
    import numpy as np
    from src.agents.population import spawn_initial_agents
    from src.core import kernel
    from src.core.shell_common import build_core_state
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=11).generate(32, 20)
    agents = spawn_initial_agents(10, world, seed=11)
    core, _world_view = build_core_state(world, agents)
    config = {
        "seed": 11,
        "agents": {
            "decision_mode": "preferences",
            "decision_sampling": "softmax",
        },
    }
    for step_index in range(1, 21):
        kernel.step(core, step_index, 7.0, config, None)
    count = core.agents.n
    print(json.dumps({
        "x": core.agents.x[:count].tolist(),
        "y": core.agents.y[:count].tolist(),
        "health": np.round(core.agents.health[:count], 12).tolist(),
        "alive": core.agents.alive[:count].tolist(),
    }))
    """
)


def test_preference_mode_is_cross_process_deterministic():
    runs = []
    for _ in range(2):
        process = subprocess.run(
            [sys.executable, "-c", _WORKER],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert runs[0] == runs[1]
