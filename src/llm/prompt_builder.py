from __future__ import annotations

from pathlib import Path
import json


PROMPT_DIR = Path(__file__).parent / "prompts"


def build_agent_decision_prompt(agent, observation, available_actions: list[str], global_state: dict | None = None, current_step: int = 0) -> str:
    template = (PROMPT_DIR / "agent_decision_prompt.md").read_text(encoding="utf-8")
    # sanitize global_state: remove any converted time fields so agents only see integer steps
    time_keys = {"simulated_day", "simulated_year", "days_per_step", "years_per_step", "day"}
    safe_global = {k: v for k, v in (global_state or {}).items() if k not in time_keys}
    payload = {
        "agent": agent.to_dict(),
        "memory_summary": agent.memory.summarize(),
        "local_observation": observation.__dict__,
        "available_actions": available_actions,
        "global_state": safe_global,
    }
    prompt = template.replace("{{CURRENT_STEP}}", str(current_step))
    return prompt.replace("{{CONTEXT_JSON}}", json.dumps(payload, indent=2))
