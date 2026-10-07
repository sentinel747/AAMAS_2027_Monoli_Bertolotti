from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import ast
import math
import operator
import random
import re
from typing import Any

import numpy as np


@dataclass
class TranslationResult:
    original: str
    translated: str
    supported: bool
    warnings: list[str]


class FormulaTranslationError(ValueError):
    pass


def translate_formula(formula: str) -> TranslationResult:
    warnings: list[str] = []
    translated = (formula or "").strip()
    if not translated:
        return TranslationResult(formula, "", False, ["empty formula"])
    translated = re.sub(r"\+/\+/([A-Za-z_]\w*)", r"sum_all(\1)", translated)
    translated = translated.replace("&&", " and ").replace("||", " or ")
    translated = translated.replace("$i0", "i0").replace("$i1", "i1")
    translated = _replace_power(translated)
    translated = re.sub(r"\bif\s*\(", "stg_if(", translated)
    translated = re.sub(r"\bmod\s*\(", "stg_mod(", translated)
    translated = re.sub(r"\brandInt\s*\(", "rand_int(", translated)
    translated = re.sub(r"\brand\s*\(", "stg_rand(", translated)
    translated = re.sub(r"\bstdDev\s*\(", "std_dev(", translated)
    translated = re.sub(r"\breadFromXLS\s*\(", "read_from_xls(", translated)
    translated = translated.replace("true", "True").replace("false", "False")
    if re.search(r"\[[^\]]*:[^\]]*\]", translated):
        warnings.append("slice syntax is STGraph-specific and may not be executable")
    if "." in translated and re.search(r"\b[A-Za-z_]\w+\.[A-Za-z_]\w+", translated):
        warnings.append("submodel dotted references are not directly supported")
    supported = not warnings
    return TranslationResult(formula, translated, supported, warnings)


def _replace_power(text: str) -> str:
    return text.replace("^", "**")


class SafeFormulaEvaluator:
    def __init__(self, base_dir: str | Path | None = None, seed: int = 0):
        self.base_dir = Path(base_dir or ".")
        self.random = random.Random(seed)

    def evaluate(self, formula: str, context: dict[str, Any]) -> Any:
        result = translate_formula(formula)
        if not result.translated:
            raise FormulaTranslationError("empty formula")
        return self.evaluate_translated(result.translated, context)

    def evaluate_translated(self, translated_formula: str, context: dict[str, Any]) -> Any:
        if not translated_formula:
            raise FormulaTranslationError("empty formula")
        env = self._env(context)
        try:
            tree = ast.parse(translated_formula, mode="eval")
        except SyntaxError as exc:
            raise FormulaTranslationError(f"syntax error after translation: {exc}") from exc
        self._validate_ast(tree)
        try:
            return eval(compile(tree, "<stg-formula>", "eval"), {"__builtins__": {}}, env)
        except Exception as exc:
            raise FormulaTranslationError(f"evaluation failed: {exc}") from exc

    def _env(self, context: dict[str, Any]) -> dict[str, Any]:
        env = dict(context)
        env.update(
            {
                "abs": abs,
                "array": stg_array,
                "cos": math.cos,
                "e": math.e,
                "exp": math.exp,
                "int": int,
                "max": safe_max,
                "mean": mean,
                "min": safe_min,
                "np": np,
                "pi": math.pi,
                "pow": pow,
                "round": round,
                "set": stg_set,
                "sigmoid": sigmoid,
                "sin": math.sin,
                "sqrt": math.sqrt,
                "std_dev": std_dev,
                "stg_if": stg_if,
                "stg_mod": lambda a, b: a % b,
                "sum": sum,
                "sum_all": sum_all,
                "time": context.get("time", 0.0),
                "timeD": context.get("timeD", 1.0),
                "i0": context.get("i0", 0),
                "i1": context.get("i1", 0),
                "rand_int": self._rand_int,
                "stg_rand": self._rand,
                "read_from_xls": self._read_from_xls,
            }
        )
        return env

    def _validate_ast(self, tree: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Subscript,
            ast.Slice,
            ast.IfExp,
            ast.And,
            ast.Or,
            ast.Not,
            ast.USub,
            ast.UAdd,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise FormulaTranslationError(f"unsafe AST node: {type(node).__name__}")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise FormulaTranslationError("only direct function calls are allowed")

    def _rand(self, *args: float) -> float:
        if not args:
            return self.random.random()
        if len(args) == 2:
            return self.random.uniform(float(args[0]), float(args[1]))
        raise FormulaTranslationError("rand supports zero or two arguments")

    def _rand_int(self, *args: int) -> int:
        if len(args) == 1:
            return self.random.randrange(max(1, int(args[0])))
        if len(args) == 2:
            return self.random.randint(int(args[0]), int(args[1]))
        raise FormulaTranslationError("randInt supports one or two arguments")

    def _read_from_xls(self, filename: str, sheet: int, row0: int, col0: int, row1: int, col1: int) -> list[float]:
        path = self.base_dir / filename
        return read_xls_range(path, int(sheet), int(row0), int(col0), int(row1), int(col1))


def stg_if(condition: Any, when_true: Any, when_false: Any = 0) -> Any:
    return when_true if bool(np.all(condition)) else when_false


def stg_array(shape: Any, fill: Any = 0) -> np.ndarray:
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if isinstance(shape, (int, float)):
        shape = [int(shape)]
    dims = tuple(int(v) for v in shape)
    return np.full(dims, fill)


def stg_set(arr: Any, *args: Any) -> Any:
    result = np.array(arr, copy=True)
    if len(args) < 2:
        return result
    indexes = args[:-1]
    value = args[-1]
    try:
        if len(indexes) == 1:
            result[_to_index(indexes[0])] = value
        else:
            result[tuple(_to_index(i) for i in indexes)] = value
    except Exception:
        return result
    return result


def _to_index(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_index(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_to_index(v) for v in value)
    return int(value)


def sigmoid(xs: Any, ys: Any, x: float) -> float:
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    if len(xs_arr) == 0:
        return 0.0
    if len(xs_arr) == 1:
        return float(ys_arr[0])
    return float(np.interp(float(x), xs_arr, ys_arr))


def mean(value: Any) -> float:
    return float(np.nanmean(np.array(value, dtype=float)))


def std_dev(value: Any) -> float:
    return float(np.nanstd(np.array(value, dtype=float)))


def sum_all(value: Any) -> float:
    return float(np.nansum(np.array(value, dtype=float)))


def safe_min(*args: Any) -> Any:
    if len(args) == 1:
        return float(np.nanmin(np.array(args[0], dtype=float)))
    return min(args)


def safe_max(*args: Any) -> Any:
    if len(args) == 1:
        return float(np.nanmax(np.array(args[0], dtype=float)))
    return max(args)


@lru_cache(maxsize=8)
def read_xls_range(path: Path, sheet: int, row0: int, col0: int, row1: int, col1: int) -> list[float]:
    try:
        import xlrd
    except ImportError as exc:
        raise FormulaTranslationError("xlrd is required to read legacy .xls files") from exc
    if not path.exists():
        raise FormulaTranslationError(f"XLS file not found: {path}")
    book = xlrd.open_workbook(str(path))
    sheet_index = max(0, sheet - 1)
    ws = book.sheet_by_index(sheet_index)
    values: list[float] = []
    for row in range(max(0, row0 - 1), min(ws.nrows, row1)):
        for col in range(max(0, col0 - 1), min(ws.ncols, col1)):
            cell = ws.cell_value(row, col)
            if cell == "":
                values.append(0.0)
            else:
                try:
                    values.append(float(cell))
                except (TypeError, ValueError):
                    values.append(0.0)
    return values
