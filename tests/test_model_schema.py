from src.model.dynamic_model import load_static_terraformazione_model


def test_static_dynamic_model_loads_without_json_or_stg_parse():
    model = load_static_terraformazione_model()
    assert model.name
    assert "O2" in model.variables
    assert model.variables["O2"].variable_type == "stock"
    assert model.variables["O2"].metadata.get("compiled_formulas")
