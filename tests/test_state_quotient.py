from __future__ import annotations

from itertools import product

from ic2_reactor.mathematical_model import Graph
from ic2_reactor.state_quotient import (
    ContinuationParetoTable,
    ParetoPoint,
    append_packed_code,
    canonical_label_orbit,
    dominates,
    product_chain_antichain_bound,
    rectangular_symmetry_permutations,
    transform_labels,
    unpack_packed_codes,
)


def test_product_chain_antichain_width_is_exact_or_safely_bounded() -> None:
    cube = product_chain_antichain_bound(((0, 1),) * 3)
    assert cube.exact
    assert cube.width == 3
    assert cube.rank_sum == 3

    rectangle = product_chain_antichain_bound(((0, 2), (10, 14)))
    assert rectangle.exact
    assert rectangle.width == 3

    fallback = product_chain_antichain_bound(
        ((0, 1),) * 3,
        maximum_coefficients=2,
    )
    assert not fallback.exact
    assert fallback.width == 4
    assert fallback.width >= cube.width


def test_packed_trace_round_trip_preserves_leading_zero_codes() -> None:
    key = (0,)
    for code in (0, 2, 0, 1):
        key = append_packed_code(key, code, 3)
    assert len(key) == 1
    assert unpack_packed_codes(key, 4, 3) == (0, 2, 0, 1)


def test_static_rectangle_has_four_symmetries_but_ordered_dynamic_has_one() -> None:
    graph = Graph.rectangular(2, 3)
    static = rectangular_symmetry_permutations(
        graph,
        preserve_event_order=False,
    )
    dynamic = rectangular_symmetry_permutations(
        graph,
        preserve_event_order=True,
    )
    assert len(static) == 4
    assert dynamic == (tuple(graph.vertices),)


def test_square_static_layer_has_full_dihedral_group() -> None:
    assert len(rectangular_symmetry_permutations(
        Graph.rectangular(3, 3),
        preserve_event_order=False,
    )) == 8


def test_orbit_canonicalization_merges_only_valid_images() -> None:
    graph = Graph.rectangular(2, 3)
    permutations = rectangular_symmetry_permutations(
        graph,
        preserve_event_order=False,
    )
    labels = ("b", "a", "a", "a", "a", "a")
    canonical = canonical_label_orbit(labels, permutations)
    assert canonical == min(transform_labels(labels, item) for item in permutations)
    assert canonical_label_orbit(canonical, permutations) == canonical


def test_pareto_table_rejects_dominated_and_keeps_incomparable_points() -> None:
    table = ContinuationParetoTable()
    key = (3, ("fuel", "empty"), 7)
    assert table.insert(key, ParetoPoint(100, 80, (4, 9), ("b",)))
    assert not table.insert(key, ParetoPoint(95, 81, (4, 8), ("worse",)))
    assert table.insert(key, ParetoPoint(105, 90, (5, 9), ("hotter",)))
    assert table.insert(key, ParetoPoint(100, 80, (4, 9), ("a",)))
    frontier = table.frontier(key)
    assert len(frontier) == 2
    assert any(point.tie_key == ("a",) for point in frontier)
    assert table.dominated_rejections == 1
    assert table.equal_replacements == 1


def test_dominance_is_preserved_by_every_common_additive_suffix() -> None:
    better = ParetoPoint(20, 10, (8, 4))
    worse = ParetoPoint(18, 12, (7, 4))
    assert dominates(better, worse)
    for suffix_power, suffix_heat, use_first, use_second in product(
        range(3),
        range(3),
        range(3),
        range(3),
    ):
        better_completed = ParetoPoint(
            better.power + suffix_power,
            better.generated_heat + suffix_heat,
            (
                better.residual_capacities[0] - use_first,
                better.residual_capacities[1] - use_second,
            ),
        )
        worse_completed = ParetoPoint(
            worse.power + suffix_power,
            worse.generated_heat + suffix_heat,
            (
                worse.residual_capacities[0] - use_first,
                worse.residual_capacities[1] - use_second,
            ),
        )
        assert dominates(better_completed, worse_completed)
