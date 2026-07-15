from __future__ import annotations

from itertools import product

from ic2_reactor.degree_signature_dp import RectangularDegreeSignatureCounter
from ic2_reactor.mathematical_model import (
    AggregatePattern,
    Graph,
    PowerComponent,
    ReactorProblem,
)


def problem(*fuel_ids: str) -> ReactorProblem:
    return ReactorProblem(
        graph=Graph.rectangular(2, 2),
        rod_budget=2,
        exact_rods=True,
        power_components=(
            PowerComponent("empty", 0, 0, False),
            *(PowerComponent(item, 1, 1, True) for item in fuel_ids),
            PowerComponent("reflector", 0, 0, True),
        ),
        cooling_components=(),
    )


def pattern(
    counts: tuple[tuple[str, int, int], ...],
    active_cells: int,
) -> AggregatePattern:
    return AggregatePattern(
        active_cells=active_cells,
        generated_heat=0,
        slack=0,
        required_relief=0,
        maximum_available_relief=0,
        margin=0,
        fuel_degree_counts=counts,
    )


def brute_count(instance: ReactorProblem, target: AggregatePattern) -> int:
    fuel_ids = {item for item, _degree, _count in target.fuel_degree_counts}
    choices = ("empty", *sorted(fuel_ids), "unknown")
    expected = {
        (item, degree): count
        for item, degree, count in target.fuel_degree_counts
    }
    result = 0
    for labels in product(choices, repeat=instance.graph.size):
        active = tuple(label != "empty" for label in labels)
        if sum(active) != target.active_cells:
            continue
        actual = {key: 0 for key in expected}
        valid = True
        for vertex, label in enumerate(labels):
            if label not in fuel_ids:
                continue
            degree = sum(active[n] for n in instance.graph.neighbours[vertex])
            if (label, degree) not in actual:
                valid = False
                break
            actual[label, degree] += 1
        if valid and actual == expected:
            result += 1
    return result


def test_counter_matches_brute_force_for_same_type_edge() -> None:
    instance = problem("fuel")
    target = pattern((("fuel", 1, 2),), active_cells=2)
    result = RectangularDegreeSignatureCounter(instance).count(target)
    assert result.proven
    assert result.count == brute_count(instance, target) == 4


def test_counter_counts_oriented_fuel_reflector_edges() -> None:
    instance = problem("fuel")
    target = pattern((("fuel", 1, 1),), active_cells=2)
    result = RectangularDegreeSignatureCounter(instance).count(target)
    assert result.count == brute_count(instance, target) == 8


def test_counter_distinguishes_fuel_types_and_rejects_triangle_signature() -> None:
    instance = problem("a", "b")
    oriented = pattern((("a", 1, 1), ("b", 1, 1)), active_cells=2)
    impossible = pattern((("a", 2, 3),), active_cells=3)
    counter = RectangularDegreeSignatureCounter(instance)
    assert counter.count(oriented).count == brute_count(instance, oriented) == 8
    assert counter.count(impossible).count == 0
