from __future__ import annotations

"""Deterministic professional groups for v3 preference agents."""

from dataclasses import dataclass
import math

import numpy as np

from src.agents import pillars


ROLE_ORDER = (
    "biologist",
    "technician",
    "engineer",
    "medic",
    "coordinator",
    "explorer",
)
DEFAULT_ROLE_DISTRIBUTION: dict[str, float] = {
    "biologist": 20.0,
    "technician": 20.0,
    "engineer": 20.0,
    "medic": 15.0,
    "coordinator": 10.0,
    "explorer": 15.0,
}


@dataclass(frozen=True)
class RoleProfile:
    preferences: tuple[float, ...]
    skills: tuple[float, ...]
    cooperation_delta: float = 0.0
    curiosity_delta: float = 0.0
    risk_delta: float = 0.0
    compliance_delta: float = 0.0


ROLE_PROFILES: dict[str, RoleProfile] = {
    "biologist": RoleProfile(
        (0.32, 0.10, 0.18, 0.12, 0.08, 0.20),
        (1.30, 0.85, 1.05, 0.95, 0.95, 1.00),
        curiosity_delta=0.05,
    ),
    "technician": RoleProfile(
        (0.10, 0.30, 0.28, 0.08, 0.08, 0.16),
        (0.90, 1.30, 1.25, 0.90, 0.90, 1.00),
        compliance_delta=0.05,
    ),
    "engineer": RoleProfile(
        (0.08, 0.18, 0.42, 0.10, 0.08, 0.14),
        (0.85, 1.10, 1.40, 0.95, 0.90, 0.95),
        compliance_delta=0.06,
    ),
    "medic": RoleProfile(
        (0.14, 0.08, 0.10, 0.45, 0.15, 0.08),
        (1.00, 0.85, 0.90, 1.45, 1.15, 0.80),
        cooperation_delta=0.10,
        risk_delta=-0.05,
    ),
    "coordinator": RoleProfile(
        (0.16, 0.12, 0.14, 0.16, 0.32, 0.10),
        (1.00, 1.00, 1.00, 1.05, 1.45, 0.90),
        cooperation_delta=0.15,
        compliance_delta=0.04,
    ),
    "explorer": RoleProfile(
        (0.12, 0.15, 0.08, 0.10, 0.07, 0.48),
        (0.90, 1.05, 0.80, 0.90, 0.85, 1.45),
        curiosity_delta=0.15,
        risk_delta=0.15,
    ),
    "colonist": RoleProfile(
        tuple([1.0 / pillars.N_PILLARS] * pillars.N_PILLARS),
        tuple([1.0] * pillars.N_PILLARS),
    ),
}

_STREAM_ROLE = 303
_STREAM_ROLE_PREF = 404
_STREAM_ROLE_SKILL = 505


def normalize_role_distribution(raw: dict | None) -> dict[str, float]:
    """Return known-role fractions summing to one.

    Missing config deliberately retains the historical generic colonist so
    raw tree/test configurations do not silently change semantics.
    """
    if not isinstance(raw, dict):
        return {"colonist": 1.0}
    values = {
        role: max(0.0, float(raw.get(role, 0.0)))
        for role in ROLE_ORDER
    }
    total = sum(values.values())
    if total <= 0.0:
        values = dict(DEFAULT_ROLE_DISTRIBUTION)
        total = sum(values.values())
    return {role: value / total for role, value in values.items()}


def assign_initial_roles(
    count: int, raw_distribution: dict | None, seed: int
) -> list[str]:
    """Allocate exact largest-remainder counts, then shuffle deterministically."""
    distribution = normalize_role_distribution(raw_distribution)
    roles = list(distribution)
    exact = [distribution[role] * max(0, int(count)) for role in roles]
    counts = [math.floor(value) for value in exact]
    remainder = max(0, int(count)) - sum(counts)
    order = sorted(range(len(roles)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    assigned = [role for role, amount in zip(roles, counts) for _ in range(amount)]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _STREAM_ROLE]))
    if assigned:
        permutation = rng.permutation(len(assigned))
        assigned = [assigned[int(index)] for index in permutation]
    return assigned


def sample_role(seed: int, agent_index: int, raw_distribution: dict | None) -> str:
    distribution = normalize_role_distribution(raw_distribution)
    roles = list(distribution)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agent_index), _STREAM_ROLE])
    )
    return roles[int(rng.choice(len(roles), p=list(distribution.values())))]


def sample_role_preferences(
    seed: int,
    agent_index: int,
    role: str,
    randomness: float = 0.25,
) -> np.ndarray:
    if role == "colonist":
        return pillars.sample_pillar_preferences(seed, agent_index)
    profile = ROLE_PROFILES.get(role, ROLE_PROFILES["colonist"])
    randomness = min(1.0, max(0.0, float(randomness)))
    concentration = 80.0 * (1.0 - randomness) + 6.0
    alpha = np.asarray(profile.preferences, dtype=np.float64) * concentration + 0.2
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agent_index), _STREAM_ROLE_PREF])
    )
    return rng.dirichlet(alpha).astype(np.float64, copy=False)


def sample_role_skills(
    seed: int,
    agent_index: int,
    role: str,
    randomness: float = 0.25,
) -> np.ndarray:
    if role == "colonist":
        return np.ones(pillars.N_PILLARS, dtype=np.float64)
    profile = ROLE_PROFILES.get(role, ROLE_PROFILES["colonist"])
    randomness = min(1.0, max(0.0, float(randomness)))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agent_index), _STREAM_ROLE_SKILL])
    )
    sigma = 0.03 + 0.14 * randomness
    return np.clip(
        rng.normal(np.asarray(profile.skills, dtype=np.float64), sigma),
        0.50,
        1.60,
    ).astype(np.float64, copy=False)


def apply_role_traits(agent) -> None:
    profile = ROLE_PROFILES.get(str(agent.role), ROLE_PROFILES["colonist"])
    agent.cooperation = min(0.99, max(0.0, agent.cooperation + profile.cooperation_delta))
    agent.curiosity = min(0.99, max(0.0, agent.curiosity + profile.curiosity_delta))
    agent.risk_tolerance = min(0.95, max(0.0, agent.risk_tolerance + profile.risk_delta))
    agent.protocol_compliance = min(
        0.99, max(0.0, agent.protocol_compliance + profile.compliance_delta)
    )
