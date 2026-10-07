from src.model.formula_translator import SafeFormulaEvaluator, translate_formula


def test_translate_conditional_and_power():
    result = translate_formula("if(x>0,2^3,0)")
    assert "stg_if" in result.translated
    assert "**" in result.translated


def test_safe_evaluator_executes_scalar_formula():
    evaluator = SafeFormulaEvaluator(base_dir="Terraformazione")
    assert evaluator.evaluate("if(x>0,max(1,2),0)", {"x": 1}) == 2
