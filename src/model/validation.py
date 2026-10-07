from __future__ import annotations

from .schema import DynamicModel, ModelValidationReport


def validate_model(model: DynamicModel) -> ModelValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    unsupported = [name for name, eq in model.equations.items() if not eq.supported]
    if not model.variables:
        errors.append("model contains no variables")
    for dep in model.dependencies:
        if dep.source not in model.variables:
            warnings.append(f"dependency source not found as variable: {dep.source}")
        if dep.target not in model.variables:
            warnings.append(f"dependency target not found as variable: {dep.target}")
    return ModelValidationReport(valid=not errors, errors=errors, warnings=warnings, unsupported_formulas=unsupported)
