from __future__ import annotations

from pathlib import Path
import json

from src.model.formula_translator import translate_formula

from .scientific_overlay import enable_scientific_overlay
from .schema import Dependency, DynamicModel, Equation, SimulationConfig, Variable


def load_static_terraformazione_model() -> DynamicModel:
    from src.generated.terraformazione_static_model import MODEL_DATA

    return enable_scientific_overlay(build_dynamic_model_from_json(MODEL_DATA))


def load_dynamic_model(path: str | Path) -> DynamicModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_dynamic_model_from_json(data)


def build_dynamic_model_from_json(data: dict) -> DynamicModel:
    settings = data.get("simulation_settings", {})
    config = SimulationConfig(
        start_time=float(settings.get("start_time") or 0.0),
        end_time=min(float(settings.get("final_time") or 100.0), 600.0),
        timestep=float(settings.get("timestep") or 1.0),
        selected_outputs=[
            "PressioneAtmosferica",
            "O2",
            "CO2",
            "N2",
            "TempMediaMarte",
            "AcquaSuperficiale",
            "Veg",
            "Pop",
        ],
    )
    variables: dict[str, Variable] = {}
    equations: dict[str, Equation] = {}
    for item in data.get("variables", []):
        variable = Variable(
            original_name=item["original_name"],
            safe_name=item["safe_name"],
            variable_type=item["variable_type"],
            formula=item.get("formula", ""),
            initial_value=item.get("initial_value", ""),
            state_transition=item.get("state_transition", ""),
            unit=item.get("unit", ""),
            bounds=item.get("bounds", {}),
            dependencies=item.get("incoming_dependencies", []),
            metadata=item.get("metadata", {}),
            parser_warnings=item.get("parser_warnings", []),
        )
        variables[variable.original_name] = variable
        expression = variable.state_transition if variable.variable_type in {"stock", "state_with_output"} else variable.formula
        compiled = variable.metadata.get("compiled_formulas", {})
        compiled_key = "state_transition" if variable.variable_type in {"stock", "state_with_output"} else "formula"
        compiled_item = compiled.get(compiled_key, {})
        if compiled_item.get("python"):
            translated_expression = compiled_item["python"]
            supported = bool(compiled_item.get("supported", True))
            warnings = list(compiled_item.get("warnings", []))
        else:
            tr = translate_formula(expression or variable.initial_value or "0")
            translated_expression = tr.translated
            supported = tr.supported
            warnings = tr.warnings
        equations[variable.original_name] = Equation(
            variable=variable.original_name,
            expression=translated_expression,
            kind="state_transition" if variable.variable_type in {"stock", "state_with_output"} else "algebraic",
            original_expression=expression,
            supported=supported,
            warnings=warnings,
        )
    deps = [
        Dependency(
            source=d.get("source", ""),
            target=d.get("target", ""),
            relation=d.get("relation", "unknown"),
            label=d.get("label", ""),
        )
        for d in data.get("dependencies", [])
    ]
    return DynamicModel(
        name=data.get("model_name", "Terraformazione"),
        variables=variables,
        equations=equations,
        dependencies=deps,
        config=config,
        metadata={
            "source_file": data.get("source_file"),
            "stgraph_version": data.get("stgraph_version"),
            "parser_warnings": data.get("parser_warnings", []),
        },
    )
