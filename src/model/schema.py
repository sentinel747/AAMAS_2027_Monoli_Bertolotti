from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Variable:
    original_name: str
    safe_name: str
    variable_type: str
    formula: str = ""
    initial_value: str = ""
    state_transition: str = ""
    unit: str = ""
    bounds: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_warnings: list[str] = field(default_factory=list)


@dataclass
class Equation:
    variable: str
    expression: str
    kind: str
    original_expression: str = ""
    supported: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class Dependency:
    source: str
    target: str
    relation: str = "unknown"
    label: str = ""


@dataclass
class SimulationConfig:
    start_time: float = 0.0
    end_time: float = 100.0
    timestep: float = 1.0
    start_step: int = 0
    random_seed: int = 0
    selected_outputs: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    time: float
    step: int = 0
    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    config: SimulationConfig
    states: list[SimulationState]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    config: SimulationConfig
    notes: str = ""
    initial_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_formulas: list[str] = field(default_factory=list)


@dataclass
class DynamicModel:
    name: str
    variables: dict[str, Variable]
    equations: dict[str, Equation]
    dependencies: list[Dependency]
    config: SimulationConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def state_variables(self) -> list[Variable]:
        return [v for v in self.variables.values() if v.variable_type in {"stock", "state_with_output"}]

    def algebraic_variables(self) -> list[Variable]:
        return [v for v in self.variables.values() if v.variable_type not in {"stock", "state_with_output"}]
