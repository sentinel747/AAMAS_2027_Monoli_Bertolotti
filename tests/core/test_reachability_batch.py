from __future__ import annotations

"""Equivalenza fra i tre bracci di raggiungibilita' (Task 4, migrazione Rust).

Un benchmark che confronta tre implementazioni ha senso solo se le tre calcolano
davvero la stessa cosa. Questi test sono quindi il prerequisito del gate
prestazionale, non un complemento: verificano che ``scalar``, ``numpy`` e ``rust``
producano risultati identici, inclusi i casi limite che distinguono le tre
formulazioni (antimeridiano, assenza totale o parziale di supporto, soglie di
razione esatte).

Il braccio ``rust`` si salta quando il modulo nativo non e' compilato: un checkout
senza toolchain Rust deve restare verde.
"""

import numpy as np
import pytest

from src.core import native
from src.core.reachability_batch import (
    DEFAULT_RATION_PER_STEP,
    DEFAULT_SAFETY_STEPS,
    NO_SUPPORT,
    reachability_batch_numpy,
    reachability_batch_rust,
    reachability_scalar,
    support_distance_field_numpy,
    support_distance_field_rust,
)

requires_native = pytest.mark.skipif(
    not native.is_available(),
    reason=f"kernel nativo non disponibile: {native.unavailable_reason()}",
)

GRID_W, GRID_H = 36, 24


def _random_case(seed: int, agent_count: int = 40, pair_count: int = 300):
    rng = np.random.default_rng(seed)
    food_columns = rng.integers(0, GRID_W, size=int(rng.integers(1, 5)))
    food_rows = rng.integers(0, GRID_H, size=food_columns.size)
    water_columns = rng.integers(0, GRID_W, size=int(rng.integers(1, 7)))
    water_rows = rng.integers(0, GRID_H, size=water_columns.size)
    return {
        "food_targets": (food_columns, food_rows),
        "water_targets": (water_columns, water_rows),
        "pair_agent": rng.integers(0, agent_count, size=pair_count),
        "pair_cell": rng.integers(0, GRID_W * GRID_H, size=pair_count),
        # Scorte deliberatamente strette: con inventari abbondanti ogni
        # implementazione direbbe sempre "si'" e il test non distinguerebbe nulla.
        "agent_food": rng.uniform(0.0, 3.0, size=agent_count),
        "agent_water": rng.uniform(0.0, 3.0, size=agent_count),
        "agent_steps_per_cell": rng.integers(1, 4, size=agent_count),
    }


def _numpy_verdicts(case) -> np.ndarray:
    food_field = support_distance_field_numpy(GRID_W, GRID_H, *case["food_targets"])
    water_field = support_distance_field_numpy(GRID_W, GRID_H, *case["water_targets"])
    return reachability_batch_numpy(
        case["pair_agent"], case["pair_cell"], food_field, water_field,
        case["agent_food"], case["agent_water"], case["agent_steps_per_cell"],
    )


def _rust_verdicts(case) -> np.ndarray:
    food_field = support_distance_field_rust(GRID_W, GRID_H, *case["food_targets"])
    water_field = support_distance_field_rust(GRID_W, GRID_H, *case["water_targets"])
    return reachability_batch_rust(
        case["pair_agent"], case["pair_cell"], food_field, water_field,
        case["agent_food"], case["agent_water"], case["agent_steps_per_cell"],
    )


def _scalar_verdicts(case) -> np.ndarray:
    return reachability_scalar(
        case["pair_agent"], case["pair_cell"], GRID_W, GRID_H,
        case["food_targets"], case["water_targets"],
        case["agent_food"], case["agent_water"], case["agent_steps_per_cell"],
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_numpy_matches_the_scalar_oracle(seed: int) -> None:
    case = _random_case(seed)
    np.testing.assert_array_equal(_numpy_verdicts(case), _scalar_verdicts(case))


@requires_native
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_rust_matches_the_scalar_oracle(seed: int) -> None:
    case = _random_case(seed)
    np.testing.assert_array_equal(_rust_verdicts(case), _scalar_verdicts(case))


@requires_native
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_rust_field_matches_numpy_field(seed: int) -> None:
    """I campi di distanza devono coincidere cella per cella, non solo i verdetti.

    Confrontare solo i verdetti finali lascerebbe passare due errori che si
    compensano: una distanza sbagliata su una cella che nessuna coppia visita, e
    una soglia sbagliata che la maschera.
    """
    case = _random_case(seed)
    for targets in (case["food_targets"], case["water_targets"]):
        np.testing.assert_array_equal(
            np.asarray(support_distance_field_rust(GRID_W, GRID_H, *targets)),
            support_distance_field_numpy(GRID_W, GRID_H, *targets),
        )


def test_field_wraps_across_the_antimeridian_numpy() -> None:
    field = support_distance_field_numpy(GRID_W, GRID_H, [0], [0])
    # La colonna 35 e' adiacente alla 0 su una griglia cilindrica larga 36.
    assert field[GRID_W - 1] == 1
    assert field[0] == 0


@requires_native
def test_field_wraps_across_the_antimeridian_rust() -> None:
    field = np.asarray(support_distance_field_rust(GRID_W, GRID_H, [0], [0]))
    assert field[GRID_W - 1] == 1
    assert field[0] == 0


def test_world_without_support_allows_free_exploration() -> None:
    empty = np.full(GRID_W * GRID_H, NO_SUPPORT, dtype=np.int32)
    verdicts = reachability_batch_numpy(
        [0], [0], empty, empty, [0.0], [0.0], [1]
    )
    assert bool(verdicts[0]), "senza alcun supporto l'esplorazione resta libera"


def test_half_supported_world_is_refused() -> None:
    """Un solo tipo di supporto non garantisce il rientro: va negato.

    E' il ramo che distingue "mondo di test senza insediamento" da "mondo reale in
    cui manca l'acqua": confonderli manderebbe agenti fuori supporto.
    """
    absent = np.full(GRID_W * GRID_H, NO_SUPPORT, dtype=np.int32)
    present = np.ones(GRID_W * GRID_H, dtype=np.int32)
    assert not bool(
        reachability_batch_numpy([0], [0], absent, present, [99.0], [99.0], [1])[0]
    )
    assert not bool(
        reachability_batch_numpy([0], [0], present, absent, [99.0], [99.0], [1])[0]
    )


@pytest.mark.parametrize("carried,expected", [(0.6, True), (0.59, False)])
def test_ration_threshold_is_exact(carried: float, expected: bool) -> None:
    """Distanza 3, 1 passo/cella, margine 2 -> servono esattamente 0,6.

    La soglia va verificata sul confine: un errore di segno o un `>` al posto di
    `>=` passerebbe inosservato su valori lontani dalla soglia.
    """
    field = np.full(1, 3, dtype=np.int32)
    verdicts = reachability_batch_numpy(
        [0], [0], field, field, [carried], [carried], [1],
        DEFAULT_RATION_PER_STEP, DEFAULT_SAFETY_STEPS,
    )
    assert bool(verdicts[0]) is expected


@requires_native
@pytest.mark.parametrize("carried,expected", [(0.6, True), (0.59, False)])
def test_ration_threshold_is_exact_in_rust(carried: float, expected: bool) -> None:
    field = np.full(1, 3, dtype=np.int32)
    verdicts = reachability_batch_rust(
        [0], [0], field, field, [carried], [carried], [1],
        DEFAULT_RATION_PER_STEP, DEFAULT_SAFETY_STEPS,
    )
    assert bool(verdicts[0]) is expected


@requires_native
def test_rust_rejects_out_of_range_indices_with_a_named_error() -> None:
    """Un indice fuori intervallo deve dire QUALE array e' sbagliato.

    Senza questa validazione nel ponte l'errore diventerebbe un panic convertito
    in `PanicException`, con un messaggio che non aiuta a trovare la causa.
    """
    field = np.zeros(4, dtype=np.int32)
    with pytest.raises(ValueError, match="pair_cell"):
        reachability_batch_rust([0], [99], field, field, [1.0], [1.0], [1])
    with pytest.raises(ValueError, match="pair_agent"):
        reachability_batch_rust([5], [0], field, field, [1.0], [1.0], [1])


@requires_native
def test_rust_rejects_mismatched_lengths() -> None:
    field = np.zeros(4, dtype=np.int32)
    with pytest.raises(ValueError, match="stessa lunghezza"):
        reachability_batch_rust([0, 1], [0], field, field, [1.0], [1.0], [1])


@requires_native
def test_rust_accepts_non_contiguous_input_after_conversion() -> None:
    """Il wrapper normalizza le viste con passo: il ponte non deve leggerle male.

    Una colonna di una matrice NumPy non e' contigua; interpretarla come tale
    darebbe valori presi dalle posizioni sbagliate, in silenzio.
    """
    matrix = np.arange(24, dtype=np.int32).reshape(4, 6)
    strided_columns = matrix[:, 0]
    assert not strided_columns.flags["C_CONTIGUOUS"] or strided_columns.strides[0] != 4
    field = np.asarray(
        support_distance_field_rust(GRID_W, GRID_H, strided_columns % GRID_W, [0, 1, 2, 3])
    )
    expected = support_distance_field_numpy(
        GRID_W, GRID_H, np.asarray(strided_columns) % GRID_W, [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(field, expected)
