

def test_structure_integrity_is_reported_and_is_a_mean_per_structure():
    """La resa di una struttura scala con l'efficienza al QUADRATO, e sotto 0,4
    si dimezza ancora: senza questa metrica non si puo' dire se le azioni di
    manutenzione di un braccio siano state prudenza o spreco.

    La prima run governata vera ha reso la lacuna evidente -- il braccio LLM ha
    speso da due a quattro volte e mezzo piu' manutenzioni della baseline, e
    nessun artefatto permetteva di giudicarle.
    """
    import numpy as np

    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    runner = AgentCoupledRunner(realistic_config(30, 3, 0))
    runner.run(days=3, output_dir=None)
    metriche = runner.world.metrics()

    assert "structure_integrity_mean" in metriche
    media = metriche["structure_integrity_mean"]
    assert 0.0 < media <= 1.0, f"un'integrita' media fuori da (0, 1]: {media}"

    cells = runner.core.cells
    attesa = float(cells.struct_integrity.sum()) / float(cells.struct_count.sum())
    assert media == attesa, "deve essere la media PER STRUTTURA, non per cella"


def test_no_structures_means_no_division_by_zero():
    """Una colonia senza strutture non deve far esplodere le metriche."""
    from src.core.arrays import AgentArrays, CellArrays
    from src.core.views import WorldView

    vuote = CellArrays(height=4, width=4)
    assert float(vuote.struct_count.sum()) == 0.0
    mondo = WorldView(vuote, AgentArrays(capacity=1), {}, {})
    assert mondo.metrics(include_planetary=False)["structure_integrity_mean"] == 0.0
