import numpy as np
import pytest

from ic2_reactor.compute_backend import ScalarPackedEvaluator
from ic2_reactor.components import COMPONENTS
from ic2_reactor.kernel_abi import (
    COMPONENT_CODE_BY_ID,
    COMPONENT_ID_BY_CODE,
    COMPONENT_KERNEL_TABLE,
    pack_layouts,
    packed_neighbor_table,
    unpack_layout,
)
from ic2_reactor.optimizer import evaluate_layout


def test_packed_layout_batch_round_trips_and_is_contiguous():
    first = tuple(["uranium_single", "reactor_heat_vent", *(["empty"] * 16)])
    second = tuple(["empty", "uranium_quad", *(["heat_vent"] * 16)])

    batch = pack_layouts([first, second], 3, [0, 120])

    assert batch.component_codes.dtype == np.uint8
    assert batch.component_codes.flags.c_contiguous
    assert batch.initial_hull_heat.dtype == np.int32
    assert batch.batch_size == 2
    assert batch.slots == 18
    assert unpack_layout(batch, 0) == first
    assert unpack_layout(batch, 1) == second


def test_component_kernel_table_matches_scalar_registry():
    for component_id, code in COMPONENT_CODE_BY_ID.items():
        spec = COMPONENTS[component_id]
        assert COMPONENT_ID_BY_CODE[code] == component_id
        assert COMPONENT_KERNEL_TABLE.max_heat[code] == spec.max_heat
        assert COMPONENT_KERNEL_TABLE.max_damage[code] == spec.max_damage
        assert COMPONENT_KERNEL_TABLE.rod_count[code] == spec.rod_count
        assert COMPONENT_KERNEL_TABLE.accepts_heat[code] == spec.accepts_heat
        assert COMPONENT_KERNEL_TABLE.is_coolable[code] == spec.is_coolable


def test_packed_neighbor_table_uses_official_order_and_is_immutable():
    neighbors = packed_neighbor_table(3)
    assert neighbors.shape == (18, 4)
    assert tuple(neighbors[4]) == (3, 5, 1, 7)
    assert tuple(neighbors[0]) == (1, 3, -1, -1)
    with pytest.raises(ValueError):
        neighbors[0, 0] = 2


def test_packed_layout_rejects_unknown_components():
    with pytest.raises(ValueError, match="unknown component id"):
        pack_layouts([["missing", *(["empty"] * 17)]], 3)


def test_scalar_packed_backend_matches_candidate_evaluation():
    stable = tuple(["uranium_single", "reactor_heat_vent", *(["empty"] * 16)])
    unsafe = tuple(["uranium_quad", *(["empty"] * 17)])
    layouts = [stable, unsafe]

    packed = pack_layouts(layouts, 3)
    results = ScalarPackedEvaluator().evaluate(packed, 40_000)

    for index, layout in enumerate(layouts):
        candidate = evaluate_layout(layout, 3, 40_000, use_certificate=False)
        assert results.safe_game_ticks[index] == candidate.safe_game_ticks
        assert results.average_eu_per_tick[index] == candidate.average_eu_per_tick
        assert results.total_eu[index] == candidate.total_eu
        assert results.safety_margin[index] == candidate.safety_margin
