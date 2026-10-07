from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.model.schema import DynamicModel, SimulationConfig, SimulationResult, SimulationState
from src.model.scientific_overlay import apply_scientific_corrections, has_scientific_overlay
from src.model.formula_translator import FormulaTranslationError, SafeFormulaEvaluator

from .constraints import sanitize_value


class SimulationEngine:
    def __init__(self, model: DynamicModel, base_dir: str | Path = "Terraformazione"):
        self.model = model
        self.evaluator = SafeFormulaEvaluator(base_dir=base_dir, seed=model.config.random_seed)
        self.warning_counts: dict[str, int] = {}

    def initialize(self, config: SimulationConfig | None = None) -> SimulationState:
        config = config or self.model.config
        values: dict[str, Any] = {"time": config.start_time, "timeD": config.timestep, "step": int(getattr(config, 'start_step', 0))}
        warnings: list[str] = []
        for variable in self.model.variables.values():
            values[variable.original_name] = 0.0
            values[variable.safe_name] = 0.0
        for _ in range(4):
            for variable in self._algebraic_variables():
                if variable.formula:
                    value = self._try_eval(
                        variable.original_name,
                        self._compiled_formula(variable.original_name, "formula", variable.formula),
                        values,
                        warnings,
                        default=values.get(variable.original_name, 0.0),
                        already_translated=self._has_compiled(variable.original_name, "formula"),
                    )
                    values[variable.original_name] = value
                    values[variable.safe_name] = value
        for variable in self.model.state_variables():
            if variable.initial_value:
                value = self._try_eval(
                    variable.original_name,
                    self._compiled_formula(variable.original_name, "initial_value", variable.initial_value),
                    values,
                    warnings,
                    default=values.get(variable.original_name, 0.0),
                    already_translated=self._has_compiled(variable.original_name, "initial_value"),
                )
                values[variable.original_name] = value
                values[variable.safe_name] = value
        for _ in range(3):
            for variable in self._algebraic_variables():
                if variable.formula:
                    value = self._try_eval(
                        variable.original_name,
                        self._compiled_formula(variable.original_name, "formula", variable.formula),
                        values,
                        warnings,
                        default=values.get(variable.original_name, 0.0),
                        already_translated=self._has_compiled(variable.original_name, "formula"),
                    )
                    values[variable.original_name] = value
                    values[variable.safe_name] = value
        if has_scientific_overlay(self.model):
            warnings.extend(apply_scientific_corrections(self.model, values, time=config.start_time, timestep=config.timestep))
        return SimulationState(time=config.start_time, step=int(getattr(config, 'start_step', 0)), values=values, warnings=warnings)

    def run(self, config: SimulationConfig | None = None, max_steps: int | None = None) -> SimulationResult:
        config = config or self.model.config
        state = self.initialize(config)
        states = [state]
        warnings: list[str] = list(state.warnings)
        total_steps = int((config.end_time - config.start_time) / config.timestep)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        for step in range(total_steps):
            state = self.step(state, config)
            states.append(state)
            warnings.extend(state.warnings)
        return SimulationResult(config=config, states=states, warnings=warnings)

    def run_until(
        self,
        initial_state: SimulationState | None = None,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        stop_when: Any | None = None,
    ) -> SimulationResult:
        config = config or self.model.config
        state = initial_state or self.initialize(config)
        states = [state]
        warnings: list[str] = list(state.warnings)
        default_steps = int((config.end_time - state.time) / config.timestep)
        total_steps = min(default_steps, max_steps) if max_steps is not None else default_steps
        for _ in range(max(0, total_steps)):
            if stop_when and stop_when(state):
                break
            state = self.step(state, config)
            states.append(state)
            warnings.extend(state.warnings)
            if state.time >= config.end_time:
                break
        return SimulationResult(config=config, states=states, warnings=warnings)

    def step(self, state: SimulationState, config: SimulationConfig | None = None) -> SimulationState:
        config = config or self.model.config
        current = dict(state.values)
        next_values = dict(current)
        warnings: list[str] = []
        t = state.time + config.timestep
        current["time"] = state.time
        current["timeD"] = config.timestep
        current["step"] = int(getattr(state, 'step', int(current.get('step', 0))))
        next_step = int(getattr(state, 'step', int(current.get('step', 0)))) + 1
        for _ in range(2):
            for variable in self._algebraic_variables():
                if not variable.formula:
                    continue
                value = self._try_eval(
                    variable.original_name,
                    self._compiled_formula(variable.original_name, "formula", variable.formula),
                    current,
                    warnings,
                    default=current.get(variable.original_name, 0.0),
                    already_translated=self._has_compiled(variable.original_name, "formula"),
                )
                value, value_warnings = sanitize_value(variable.original_name, value)
                warnings.extend(value_warnings)
                current[variable.original_name] = value
                current[variable.safe_name] = value
                next_values[variable.original_name] = value
                next_values[variable.safe_name] = value
        for variable in self.model.state_variables():
            formula = variable.state_transition or variable.formula
            if not formula:
                continue
            local = dict(current)
            local["this"] = current.get(variable.original_name, 0.0)
            local["me"] = local["this"]
            field = "state_transition" if variable.state_transition else "formula"
            value = self._try_eval(
                variable.original_name,
                self._compiled_formula(variable.original_name, field, formula),
                local,
                warnings,
                default=current.get(variable.original_name, 0.0),
                already_translated=self._has_compiled(variable.original_name, field),
            )
            value, value_warnings = sanitize_value(variable.original_name, value)
            warnings.extend(value_warnings)
            next_values[variable.original_name] = value
            next_values[variable.safe_name] = value
        next_values["time"] = t
        next_values["timeD"] = config.timestep
        next_values["step"] = next_step
        if has_scientific_overlay(self.model):
            warnings.extend(apply_scientific_corrections(self.model, next_values, previous_values=current, time=t, timestep=config.timestep))
        return SimulationState(time=t, step=next_step, values=next_values, warnings=warnings)

    def _try_eval(
        self,
        name: str,
        formula: str,
        context: dict[str, Any],
        warnings: list[str],
        default: Any = 0.0,
        already_translated: bool = False,
    ) -> Any:
        try:
            if already_translated:
                return self.evaluator.evaluate_translated(formula, context)
            return self.evaluator.evaluate(formula, context)
        except FormulaTranslationError as exc:
            key = f"{name}:{formula}"
            self.warning_counts[key] = self.warning_counts.get(key, 0) + 1
            if self.warning_counts[key] <= 1:
                warnings.append(f"{name}: {exc}; using previous/default value")
            return default

    def _compiled_formula(self, name: str, field: str, fallback: str) -> str:
        variable = self.model.variables[name]
        return variable.metadata.get("compiled_formulas", {}).get(field, {}).get("python") or fallback

    def _has_compiled(self, name: str, field: str) -> bool:
        variable = self.model.variables[name]
        return bool(variable.metadata.get("compiled_formulas", {}).get(field, {}).get("python"))

    def _algebraic_variables(self):
        return sorted(
            self.model.algebraic_variables(),
            key=lambda variable: (
                0 if variable.variable_type == "input" else 1,
                len(variable.dependencies),
                variable.original_name,
            ),
        )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
