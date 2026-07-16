import random

import numpy as np

from ic2_reactor.components import COMPONENT_IDS
from ic2_reactor.compute_backend import ScalarPackedEvaluator
from ic2_reactor.kernel_abi import decode_mark, encode_mark, pack_layouts
from ic2_reactor.numba_backend import NumbaPackedEvaluator


def assert_backends_equal(layouts, columns, max_reactor_ticks, initial_hull_heat=0):
    batch = pack_layouts(layouts, columns, initial_hull_heat)
    expected = ScalarPackedEvaluator().evaluate(batch, max_reactor_ticks)
    actual = NumbaPackedEvaluator().evaluate(batch, max_reactor_ticks)

    for field in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_mark_numeric_encoding_round_trips_public_strings():
    for mark, stable in (
        (None, False),
        ("Mark I-I", True),
        ("Mark I-I-SUC", True),
        ("Mark II-3", False),
        ("Mark II-E-SUC", False),
        ("Mark III", False),
        ("Mark IV-SUC", False),
        ("Mark V", False),
    ):
        family, level, flags = encode_mark(mark, stable)
        assert decode_mark(family, level, flags) == mark


def test_numba_backend_matches_scalar_on_semantic_boundaries():
    empty = ["empty"] * 18
    layouts = []

    stable = empty.copy()
    stable[:2] = ["uranium_single", "reactor_heat_vent"]
    layouts.append(tuple(stable))

    meltdown = empty.copy()
    meltdown[4] = "uranium_quad"
    layouts.append(tuple(meltdown))

    exchanger_before_plating = empty.copy()
    exchanger_before_plating[:4] = [
        "reactor_heat_exchanger",
        "heat_capacity_plating",
        "uranium_dual",
        "coolant_10k",
    ]
    layouts.append(tuple(exchanger_before_plating))

    finite_reflector = empty.copy()
    finite_reflector[3:6] = [
        "uranium_single",
        "neutron_reflector",
        "component_heat_vent",
    ]
    layouts.append(tuple(finite_reflector))

    condensator = empty.copy()
    condensator[4] = "uranium_single"
    for position in (1, 3, 5, 7):
        condensator[position] = "lzh_condensator"
    condensator[17] = "reactor_heat_vent"
    layouts.append(tuple(condensator))

    assert_backends_equal(layouts, 3, 40_000, [0, 0, 25, 0, 0])


def test_numba_backend_matches_scalar_on_seeded_random_layouts():
    rng = random.Random(221)
    components = list(COMPONENT_IDS)
    layouts = [
        tuple(rng.choice(components) if rng.random() < 0.55 else "empty" for _ in range(18))
        for _ in range(32)
    ]
    initial_heat = [rng.randrange(0, 8_000) for _ in layouts]

    assert_backends_equal(layouts, 3, 4_000, initial_heat)


def test_numba_backend_matches_scalar_at_maximum_chamber_width():
    rng = random.Random(8221)
    components = list(COMPONENT_IDS)
    layouts = [
        tuple(rng.choice(components) if rng.random() < 0.35 else "empty" for _ in range(54))
        for _ in range(12)
    ]

    assert_backends_equal(layouts, 9, 2_500)
