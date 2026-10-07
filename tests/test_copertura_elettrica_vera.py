"""Interruttore `copertura elettrica vera` (2026-09-26).

`power_coverage` del vocabolario dei governatori divide la giacenza di energia
della cella per il carico, ma la giacenza e' letta dopo che il kernel ha
consumato l'energia e dopo i prelievi dei coloni: vale ~0 in ogni cella con
impianti anche quando la corrente basta. Nelle run LLM della tesi il 23% delle
celle-passo catturate da una regola lo era da regole su questo indicatore.
L'interruttore, spento per default, fa leggere la copertura che il kernel
calcola a ogni passo (energia usata / carico).
"""

import numpy as np

from scripts.parity_harness import build_config
from src.governors.apply import indicator_values
from src.simulation.agent_coupled_runner import AgentCoupledRunner

STEPS = 4


def _runner(accesa: bool, tmp_path):
    config = build_config("realistic", 20, STEPS, 101, "softmax")
    if accesa:
        config.setdefault("governors", {})["copertura_elettrica"] = "vera"
    runner = AgentCoupledRunner(config)
    runner.run(days=STEPS, output_dir=tmp_path / ("on" if accesa else "off"))
    return runner


def test_off_by_default_adds_nothing_to_the_cells(tmp_path):
    runner = _runner(False, tmp_path)
    cells = runner.core.cells
    assert not getattr(cells, "copertura_vera_attiva", False)
    assert not hasattr(cells, "copertura_elettrica")


def test_on_the_indicator_reads_the_kernel_coverage(tmp_path):
    runner = _runner(True, tmp_path)
    cells = runner.core.cells
    vera = indicator_values("power_coverage", cells)
    assert vera is cells.copertura_elettrica
    assert vera.shape == cells.occupancy.shape
    assert np.all(vera >= 0.0) and np.all(vera <= 1.0 + 1e-12)
    # Dove ci sono impianti la corrente copre il carico: la copertura vera non e'
    # zero, mentre la giacenza avanzata lo e' quasi sempre.
    con_carico = cells.struct_count.sum(axis=-1) > 0
    assert con_carico.any()
    assert float(vera[con_carico].mean()) > 0.5
    cells.copertura_vera_attiva = False
    giacenza = indicator_values("power_coverage", cells)
    assert float(giacenza[con_carico].mean()) < float(vera[con_carico].mean())


def test_before_the_first_step_coverage_is_full(tmp_path):
    config = build_config("realistic", 20, STEPS, 101, "softmax")
    config.setdefault("governors", {})["copertura_elettrica"] = "vera"
    runner = AgentCoupledRunner(config)
    vera = indicator_values("power_coverage", runner.core.cells)
    assert np.all(vera == 1.0)
