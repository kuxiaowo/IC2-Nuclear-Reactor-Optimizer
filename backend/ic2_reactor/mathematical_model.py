"""Independent, parameterised mathematical model for IC2 reactor design.

The optimiser in :mod:`ic2_reactor.optimizer` is intentionally *not* used in
this module.  This file starts from a graph, a small component catalogue and
the published pulse/heat equations.  The production simulator is only used by
``IC2CycleOracle`` as an independent witness checker after a layout has been
constructed.

There are three deliberately separated levels:

``closed_form_upper_bound``
    A constant-time convexity/heat-conservation proof.  It is weak but exact
    as an upper bound and discards most impossible power levels without search.

``PowerHeatMaster``
    A graph-generic CP-SAT model for exact power and generated heat, augmented
    by an optimistic inventory-aware cooling envelope.  It proves a strict
    upper bound; feasibility here is only a necessary condition for a safe
    reactor.

``IC2CycleOracle``
    A deterministic exact checker for a completed six-row IC2 layout.  It
    returns a reachable repeated thermal state as a positive certificate and
    never interprets a horizon timeout as infeasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import ceil, floor
from time import perf_counter
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Graph:
    """Finite undirected graph with an explicit deterministic update order."""

    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    neighbours: tuple[tuple[int, ...], ...]
    update_order: tuple[int, ...]
    rows: int | None = None
    columns: int | None = None

    @classmethod
    def rectangular(cls, rows: int, columns: int) -> "Graph":
        if rows <= 0 or columns <= 0:
            raise ValueError("rows and columns must be positive")
        size = rows * columns
        adjacency: list[list[int]] = [[] for _ in range(size)]
        edges: list[tuple[int, int]] = []
        # This ordering is part of the mathematical instance: left, right,
        # up, down is also the IC2 fuel heat-distribution order.
        for index in range(size):
            row, column = divmod(index, columns)
            if column > 0:
                adjacency[index].append(index - 1)
            if column + 1 < columns:
                adjacency[index].append(index + 1)
                edges.append((index, index + 1))
            if row > 0:
                adjacency[index].append(index - columns)
            if row + 1 < rows:
                adjacency[index].append(index + columns)
                edges.append((index, index + columns))
        return cls(
            vertices=tuple(range(size)),
            edges=tuple(edges),
            neighbours=tuple(tuple(values) for values in adjacency),
            update_order=tuple(range(size)),
            rows=rows,
            columns=columns,
        )

    @classmethod
    def from_edges(
        cls,
        vertex_count: int,
        edges: Iterable[tuple[int, int]],
        *,
        update_order: Sequence[int] | None = None,
    ) -> "Graph":
        if vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        canonical: set[tuple[int, int]] = set()
        adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
        for raw_first, raw_second in edges:
            first, second = int(raw_first), int(raw_second)
            if first == second:
                raise ValueError("self loops are not supported")
            if not (0 <= first < vertex_count and 0 <= second < vertex_count):
                raise ValueError("edge endpoint is outside the graph")
            edge = (min(first, second), max(first, second))
            canonical.add(edge)
            adjacency[first].add(second)
            adjacency[second].add(first)
        order = tuple(range(vertex_count)) if update_order is None else tuple(update_order)
        if sorted(order) != list(range(vertex_count)):
            raise ValueError("update_order must be a permutation of the vertices")
        return cls(
            vertices=tuple(range(vertex_count)),
            edges=tuple(sorted(canonical)),
            neighbours=tuple(tuple(sorted(values)) for values in adjacency),
            update_order=order,
        )

    @property
    def size(self) -> int:
        return len(self.vertices)

    @property
    def maximum_degree(self) -> int:
        return max(map(len, self.neighbours), default=0)


@dataclass(frozen=True, slots=True)
class PowerComponent:
    """A label participating in uranium pulses.

    ``rods == 0`` represents a permanent reflector.  ``accepts_pulse`` is
    intentionally separate from ``rods`` so that arbitrary catalogues can be
    modelled without IC2-specific conditionals.
    """

    id: str
    rods: int
    internal_pulses: int
    accepts_pulse: bool

    def __post_init__(self) -> None:
        if self.rods < 0 or self.internal_pulses < 0:
            raise ValueError("power component values must be non-negative")


@dataclass(frozen=True, slots=True)
class CoolingComponent:
    """Persistent external heat-removal parameters used by safe relaxations."""

    id: str
    self_vent: int = 0
    side_vent: int = 0
    inventory: int | None = None

    def __post_init__(self) -> None:
        if self.self_vent < 0 or self.side_vent < 0:
            raise ValueError("vent rates must be non-negative")
        if self.inventory is not None and self.inventory < 0:
            raise ValueError("inventory must be non-negative or None")


IC2_POWER_COMPONENTS: tuple[PowerComponent, ...] = (
    PowerComponent("empty", 0, 0, False),
    PowerComponent("uranium_single", 1, 1, True),
    PowerComponent("uranium_dual", 2, 2, True),
    PowerComponent("uranium_quad", 4, 3, True),
    # Finite reflectors are absent: an item whose damage strictly increases
    # cannot belong to an indefinitely safe no-maintenance cycle.
    PowerComponent("iridium_reflector", 0, 0, True),
)


IC2_PERSISTENT_COOLING: tuple[CoolingComponent, ...] = (
    CoolingComponent("overclocked_heat_vent", self_vent=20),
    CoolingComponent("component_heat_vent", side_vent=4),
    CoolingComponent("advanced_heat_vent", self_vent=12),
    CoolingComponent("heat_vent", self_vent=6),
    CoolingComponent("reactor_heat_vent", self_vent=5),
)


IC2_NON_POWER_COMPONENTS: tuple[str, ...] = (
    "empty",
    "heat_vent",
    "advanced_heat_vent",
    "reactor_heat_vent",
    "component_heat_vent",
    "overclocked_heat_vent",
    "coolant_10k",
    "coolant_30k",
    "coolant_60k",
    "heat_exchanger",
    "advanced_heat_exchanger",
    "reactor_heat_exchanger",
    "component_heat_exchanger",
    "reactor_plating",
    "heat_capacity_plating",
    "containment_plating",
    "rsh_condensator",
    "lzh_condensator",
)


@dataclass(frozen=True, slots=True)
class ReactorProblem:
    """Complete input to the static model, with no 6x9/25 constants."""

    graph: Graph
    rod_budget: int
    exact_rods: bool
    power_components: tuple[PowerComponent, ...]
    cooling_components: tuple[CoolingComponent, ...]
    layout_components: tuple[str, ...] = ()
    component_limits: tuple[tuple[str, int | None], ...] = ()
    eu_per_pulse: int = 5
    heat_scale: int = 2
    ruleset: str = "generic"

    def __post_init__(self) -> None:
        if self.rod_budget <= 0:
            raise ValueError("rod_budget must be positive")
        if self.eu_per_pulse <= 0 or self.heat_scale <= 0:
            raise ValueError("energy and heat scales must be positive")
        ids = [item.id for item in self.power_components]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("power component ids must be non-empty and unique")
        if self.power_components[0] != PowerComponent("empty", 0, 0, False):
            raise ValueError("the first power component must be the empty label")
        if not any(item.rods > 0 for item in self.power_components):
            raise ValueError("at least one fuel component is required")
        limit_ids = [item for item, _limit in self.component_limits]
        if len(limit_ids) != len(set(limit_ids)):
            raise ValueError("component limit ids must be unique")
        known = set(ids) | set(self.layout_components)
        if unknown := set(limit_ids) - known:
            raise ValueError(f"limits reference disabled or unknown components: {sorted(unknown)}")
        if any(limit is not None and limit < 0 for _item, limit in self.component_limits):
            raise ValueError("component limits must be non-negative or None")

    @property
    def max_rods_per_cell(self) -> int:
        return max(item.rods for item in self.power_components)

    @property
    def fuel_codes(self) -> tuple[int, ...]:
        return tuple(index for index, item in enumerate(self.power_components) if item.rods > 0)


def ic2_mark_i_problem(
    *,
    rows: int = 6,
    columns: int = 9,
    rod_budget: int = 25,
    exact_rods: bool = True,
    enabled_components: Iterable[str] | None = None,
    component_limits: Mapping[str, int | None] | None = None,
) -> ReactorProblem:
    """Build the IC2 Mark-I instance, including unlimited-inventory semantics.

    An omitted component limit means "only limited by free reactor slots".
    Enabled storage/exchanger/plating components do not appear in the first
    heat-conservation envelope because they cannot destroy heat; they remain
    available to later layout construction and exact verification.
    """

    enabled = None if enabled_components is None else frozenset(enabled_components)
    limits = {} if component_limits is None else dict(component_limits)

    power = tuple(
        item
        for item in IC2_POWER_COMPONENTS
        if item.id == "empty" or enabled is None or item.id in enabled
    )
    cooling = tuple(
        CoolingComponent(
            item.id,
            self_vent=item.self_vent,
            side_vent=item.side_vent,
            inventory=limits.get(item.id),
        )
        for item in IC2_PERSISTENT_COOLING
        if enabled is None or item.id in enabled
    )
    layout_components = tuple(
        item for item in IC2_NON_POWER_COMPONENTS if enabled is None or item in enabled
    )
    known_enabled = {item.id for item in power} | set(layout_components)
    if unknown_limits := set(limits) - known_enabled:
        raise ValueError(f"limits reference disabled or unknown components: {sorted(unknown_limits)}")
    return ReactorProblem(
        graph=Graph.rectangular(rows, columns),
        rod_budget=rod_budget,
        exact_rods=exact_rods,
        power_components=power,
        cooling_components=cooling,
        layout_components=layout_components,
        component_limits=tuple(sorted(limits.items())),
        ruleset="ic2-experimental-2.8.221-mark-i",
    )


@dataclass(frozen=True, slots=True)
class StaticMetrics:
    rods: int
    active_cells: int
    pulse_units: int
    power: int
    generated_heat: int
    degrees: tuple[int, ...]


def evaluate_power_skeleton(problem: ReactorProblem, labels: Sequence[str]) -> StaticMetrics:
    """Evaluate the exact graph-local power and heat equations."""

    if len(labels) != problem.graph.size:
        raise ValueError("skeleton length does not match graph size")
    catalogue = {item.id: item for item in problem.power_components}
    try:
        selected = tuple(catalogue[label] for label in labels)
    except KeyError as error:
        raise ValueError(f"unknown power label: {error.args[0]}") from error
    active = tuple(item.accepts_pulse for item in selected)
    degrees = tuple(
        sum(active[other] for other in problem.graph.neighbours[index])
        for index in problem.graph.vertices
    )
    rods = pulse_units = generated_heat = 0
    for item, degree in zip(selected, degrees, strict=True):
        rods += item.rods
        if item.rods == 0:
            continue
        pulses = item.internal_pulses + degree
        pulse_units += item.rods * pulses
        generated_heat += problem.heat_scale * item.rods * pulses * (pulses + 1)
    return StaticMetrics(
        rods=rods,
        active_cells=sum(active),
        pulse_units=pulse_units,
        power=problem.eu_per_pulse * pulse_units,
        generated_heat=generated_heat,
        degrees=degrees,
    )


def minimum_heat_for_pulse_units(rods: int, pulse_units: int, *, heat_scale: int = 2) -> int:
    """Convex integer lower bound on heat for a fixed total pulse count.

    Relaxing bundled rods into individual rods can only lower the optimum, so
    this remains a sound lower bound when dual/quad fuel stacks are present.
    """

    if rods <= 0 or pulse_units < 0:
        raise ValueError("rods must be positive and pulse_units non-negative")
    low, high_count = divmod(pulse_units, rods)
    low_count = rods - high_count
    return heat_scale * (
        low_count * low * (low + 1)
        + high_count * (low + 1) * (low + 2)
    )


def optimistic_cooling_values(problem: ReactorProblem) -> tuple[int, ...]:
    """Per-slot capacities sorted for a sound aggregate cooling envelope.

    Side vents are credited with the *maximum graph degree* and every credited
    neighbour is allowed to be coolable simultaneously.  Those relaxations can
    only overestimate real cooling, which is the required direction for a
    proof-producing upper bound.
    """

    values: list[int] = []
    slots = problem.graph.size
    maximum_degree = problem.graph.maximum_degree
    for item in problem.cooling_components:
        capacity = item.self_vent + item.side_vent * maximum_degree
        count = slots if item.inventory is None else min(item.inventory, slots)
        values.extend([capacity] * count)
    values.extend([0] * slots)
    values.sort(reverse=True)
    return tuple(values[:slots])


def cooling_envelope(problem: ReactorProblem) -> tuple[int, ...]:
    """Return ``V[c]``: optimistic maximum heat removal using ``c`` slots."""

    values = optimistic_cooling_values(problem)
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ClosedFormBound:
    power_upper_bound: int
    pulse_units_upper_bound: int
    heat_at_bound: int
    cooling_upper_bound: int
    rods_used_for_bound: int
    minimum_fuel_cells: int
    derivation: str


@dataclass(frozen=True, slots=True)
class AnalyticalCutProof:
    """A machine-checkable implication that removes complete power layers."""

    power_upper_bound: int
    excluded_power_levels: tuple[int, ...]
    assumptions: tuple[str, ...]
    checks: tuple[tuple[str, int], ...]
    derivation: str


@dataclass(frozen=True, slots=True)
class AggregatePattern:
    active_cells: int
    generated_heat: int
    slack: int
    required_relief: int
    maximum_available_relief: int
    margin: int
    fuel_degree_counts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class AggregateOverloadAnalysis:
    """Count-space relaxation for one exact power tier."""

    power: int
    pattern_count: int
    minimum_relief_margin: int | None
    excluded: bool
    weakest_pattern: AggregatePattern | None
    surviving_patterns: tuple[AggregatePattern, ...] = ()
    structurally_surviving_patterns: tuple[AggregatePattern, ...] = ()


@dataclass(frozen=True, slots=True)
class AggregateRoutingProfile:
    """One count-space choice of direct versus hull fuel routing."""

    required_relief: int
    exchanger_relief_bound: int
    maximum_available_relief: int
    margin: int
    direct_fuel_counts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class RouteConditionedOverloadAnalysis:
    """Joint routing/relief relaxation for one aggregate fuel pattern."""

    excluded: bool
    minimum_margin: int
    profile_count: int
    surviving_profiles: tuple[AggregateRoutingProfile, ...]


def _component_exchanger_bound_from_direct_loads(
    maximum_degree: int,
    direct_loads: Sequence[int],
    *,
    ideal_receiver_sink: int,
    exchange_side: int,
) -> int:
    ordered_loads = sorted(direct_loads, reverse=True)
    best = 0
    for fuel_neighbours in range(min(maximum_degree, len(ordered_loads)) + 1):
        direct = sum(ordered_loads[:fuel_neighbours])
        remaining_edges = maximum_degree - fuel_neighbours
        for incoming_edges in range(remaining_edges + 1):
            outgoing_edges = remaining_edges - incoming_edges
            transmitted = min(
                direct + exchange_side * incoming_edges,
                exchange_side * outgoing_edges,
            )
            baseline_direct_sink = (
                min(ideal_receiver_sink, direct) if fuel_neighbours else 0
            )
            best = max(best, transmitted - baseline_direct_sink)
    return best


def component_exchanger_relief_bound(
    problem: ReactorProblem,
    pattern: AggregatePattern,
    *,
    ideal_receiver_sink: int = 20,
    exchange_side: int = 36,
) -> int:
    """Bound one side exchanger's useful relief for an aggregate pattern.

    A receiver adjacent to ``k`` fuel cells has only ``Delta-k`` remaining
    component edges.  Of those, some may bring transshipment heat in and the
    rest carry heat out.  The function enumerates this constant-size local
    edge cut and uses the hottest possible ordered direct load contributed by
    the pattern.  With no direct fuel neighbour, all transmitted heat counts
    as relief.  With direct heat, an ideal receiver in the baseline would have
    removed up to ``ideal_receiver_sink`` of that direct load, so only the
    remainder plus transshipment is additional local relief.

    Geometry compatibility between the selected fuel cells is ignored, which
    can only increase the bound.  Thus the result is safe for count-space and
    structural relaxations and is intentionally not a realizability claim.
    """

    if ideal_receiver_sink <= 0 or exchange_side <= 0:
        raise ValueError("exchanger capacities must be positive")
    maximum_degree = problem.graph.maximum_degree
    items = {item.id: item for item in problem.power_components if item.rods > 0}
    direct_loads: list[int] = []
    for item_id, degree, count in pattern.fuel_degree_counts:
        if item_id not in items:
            raise ValueError(f"pattern contains unknown fuel label: {item_id}")
        if degree < 0 or degree > maximum_degree or count < 0:
            raise ValueError("pattern has an invalid degree or count")
        receivers = maximum_degree - degree
        if receivers == 0 or count == 0:
            continue
        item = items[item_id]
        pulses = item.internal_pulses + degree
        per_rod_heat = problem.heat_scale * pulses * (pulses + 1)
        loads = [0] * receivers
        for _rod in range(item.rods):
            remaining = per_rod_heat
            for receiver in range(receivers):
                amount = remaining // (receivers - receiver)
                remaining -= amount
                loads[receiver] += amount
        direct_loads.extend([max(loads)] * count)

    return _component_exchanger_bound_from_direct_loads(
        maximum_degree,
        direct_loads,
        ideal_receiver_sink=ideal_receiver_sink,
        exchange_side=exchange_side,
    )


def route_conditioned_overload_analysis(
    problem: ReactorProblem,
    pattern: AggregatePattern,
    *,
    ideal_receiver_sink: int = 20,
    hull_draw: int = 36,
    side_vent_cost: int = 4,
    side_vent_relief: int = 16,
    exchanger_cost: int = 20,
    exchange_side: int = 36,
) -> RouteConditionedOverloadAnalysis:
    """Jointly choose fuel heat routes and the exchanger capacity bound.

    The older aggregate relaxation independently chose the smaller of direct
    and hull overload for every fuel state, then granted the largest exchanger
    capacity obtainable from *any* direct state.  Those choices need not be
    compatible.  Here ``n+1`` choices for each repeated state specify how many
    copies route directly; the remaining copies use the hull.  The local
    exchanger cut is then recomputed only from directly routed fuel loads.

    Side vents remain an impossible four-effective-edge knapsack, so positive
    margin for every profile is still a sound exclusion without enumerating
    positions or component labels.
    """

    if min(
        ideal_receiver_sink,
        hull_draw,
        side_vent_cost,
        side_vent_relief,
        exchanger_cost,
        exchange_side,
    ) <= 0:
        raise ValueError("route-conditioned capacities must be positive")
    if pattern.slack < 0:
        raise ValueError("aggregate pattern slack must be non-negative")
    maximum_degree = problem.graph.maximum_degree
    items = {item.id: item for item in problem.power_components if item.rods > 0}
    states = []
    for item_id, degree, count in pattern.fuel_degree_counts:
        if item_id not in items:
            raise ValueError(f"pattern contains unknown fuel label: {item_id}")
        if degree < 0 or degree > maximum_degree or count < 0:
            raise ValueError("pattern has an invalid degree or count")
        item = items[item_id]
        pulses = item.internal_pulses + degree
        per_rod_heat = problem.heat_scale * pulses * (pulses + 1)
        total_heat = item.rods * per_rod_heat
        receivers = maximum_degree - degree
        loads: list[int] = []
        if receivers:
            loads = [0] * receivers
            for _rod in range(item.rods):
                remaining = per_rod_heat
                for receiver in range(receivers):
                    amount = remaining // (receivers - receiver)
                    remaining -= amount
                    loads[receiver] += amount
        direct_relief = (
            sum(max(0, load - ideal_receiver_sink) for load in loads)
            if loads
            else None
        )
        full_draws, remainder = divmod(total_heat, hull_draw)
        hull_relief = (
            full_draws * max(0, hull_draw - ideal_receiver_sink)
            + max(0, remainder - ideal_receiver_sink)
        )
        states.append((
            item_id,
            degree,
            count,
            direct_relief,
            hull_relief,
            max(loads, default=0),
        ))

    profiles: list[AggregateRoutingProfile] = []

    def enumerate_routes(
        index: int,
        required: int,
        direct_loads: tuple[int, ...],
        direct_counts: tuple[tuple[str, int, int], ...],
    ) -> None:
        if index == len(states):
            exchanger_relief = _component_exchanger_bound_from_direct_loads(
                maximum_degree,
                direct_loads,
                ideal_receiver_sink=ideal_receiver_sink,
                exchange_side=exchange_side,
            )
            maximum_available = max(
                exchanger_relief * exchangers
                + side_vent_relief
                * ((pattern.slack - exchanger_cost * exchangers) // side_vent_cost)
                for exchangers in range(pattern.slack // exchanger_cost + 1)
            )
            profiles.append(AggregateRoutingProfile(
                required_relief=required,
                exchanger_relief_bound=exchanger_relief,
                maximum_available_relief=maximum_available,
                margin=required - maximum_available,
                direct_fuel_counts=tuple(
                    entry for entry in direct_counts if entry[2]
                ),
            ))
            return
        item_id, degree, count, direct_relief, hull_relief, direct_load = states[index]
        direct_range = range(count + 1) if direct_relief is not None else (0,)
        for direct_count in direct_range:
            enumerate_routes(
                index + 1,
                required
                + direct_count * (0 if direct_relief is None else direct_relief)
                + (count - direct_count) * hull_relief,
                (*direct_loads, *((direct_load,) * direct_count)),
                (*direct_counts, (item_id, degree, direct_count)),
            )

    enumerate_routes(0, 0, (), ())
    minimum_margin = min(profile.margin for profile in profiles)
    survivors = tuple(profile for profile in profiles if profile.margin <= 0)
    return RouteConditionedOverloadAnalysis(
        excluded=minimum_margin > 0,
        minimum_margin=minimum_margin,
        profile_count=len(profiles),
        surviving_profiles=survivors,
    )


def _graph_is_bipartite(graph: Graph) -> bool:
    colours: dict[int, int] = {}
    for root in graph.vertices:
        if root in colours:
            continue
        colours[root] = 0
        queue = [root]
        for vertex in queue:
            for neighbour in graph.neighbours[vertex]:
                if neighbour not in colours:
                    colours[neighbour] = 1 - colours[vertex]
                    queue.append(neighbour)
                elif colours[neighbour] == colours[vertex]:
                    return False
    return True


def _degree_count_vectors(total: int, maximum_degree: int):
    def generate(degree: int, remaining: int, prefix: tuple[int, ...]):
        if degree == maximum_degree:
            yield (*prefix, remaining)
            return
        for count in range(remaining + 1):
            yield from generate(degree + 1, remaining - count, (*prefix, count))

    yield from generate(0, total, ())


@cache
def _bipartite_degree_counts_possible(
    fixed_counts: tuple[int, ...],
    unknown_vertices: int,
    maximum_degree: int,
) -> bool:
    """Cached Gale--Ryser test on a canonical degree histogram."""
    for unknown_counts in _degree_count_vectors(unknown_vertices, maximum_degree):
        counts = [
            fixed_counts[degree] + unknown_counts[degree]
            for degree in range(maximum_degree + 1)
        ]

        def assign_left(degree: int, left_counts: tuple[int, ...]):
            if degree > maximum_degree:
                left = sorted(
                    (
                        value
                        for value, count in enumerate(left_counts)
                        for _ in range(count)
                    ),
                    reverse=True,
                )
                right = sorted(
                    (
                        value
                        for value, total_count in enumerate(counts)
                        for _ in range(total_count - left_counts[value])
                    ),
                    reverse=True,
                )
                if sum(left) != sum(right):
                    return False
                if (left and left[0] > len(right)) or (right and right[0] > len(left)):
                    return False
                return all(
                    sum(left[:size])
                    <= sum(min(size, value) for value in right)
                    for size in range(1, len(left) + 1)
                )
            return any(
                assign_left(degree + 1, (*left_counts, left_count))
                for left_count in range(counts[degree] + 1)
            )

        if assign_left(0, ()):
            return True
    return False


def bipartite_degree_sequence_possible(
    fixed_degrees: Sequence[int],
    *,
    unknown_vertices: int = 0,
    maximum_degree: int = 4,
) -> bool:
    """Return whether some simple bipartite graph can realise the degrees.

    Unknown vertices may take any degree from zero through ``maximum_degree``.
    The test is a relaxation of embedding in the supplied reactor grid, hence
    failure is a sound structural exclusion while success proves no embedding.
    Repeated aggregate patterns share a cached canonical Gale--Ryser result.
    """

    if unknown_vertices < 0 or maximum_degree < 0:
        raise ValueError("degree-sequence parameters must be non-negative")
    if any(degree < 0 or degree > maximum_degree for degree in fixed_degrees):
        return False
    fixed_counts = tuple(
        sum(degree == value for degree in fixed_degrees)
        for value in range(maximum_degree + 1)
    )
    return _bipartite_degree_counts_possible(
        fixed_counts,
        unknown_vertices,
        maximum_degree,
    )


def closed_form_upper_bound(problem: ReactorProblem) -> ClosedFormBound:
    """Prove an immediate power upper bound by convexity and heat conservation."""

    envelope = cooling_envelope(problem)
    degree = problem.graph.maximum_degree
    best: tuple[int, int, int, int, int] | None = None
    rod_counts = (
        (problem.rod_budget,)
        if problem.exact_rods
        else range(1, problem.rod_budget + 1)
    )
    maximum_internal = max(item.internal_pulses for item in problem.power_components if item.rods)
    for rods in rod_counts:
        minimum_fuel_cells = ceil(rods / problem.max_rods_per_cell)
        free_slots = max(0, problem.graph.size - minimum_fuel_cells)
        cooling = envelope[free_slots]
        maximum_pulses = rods * (maximum_internal + degree)
        for pulse_units in range(maximum_pulses, -1, -1):
            heat = minimum_heat_for_pulse_units(
                rods, pulse_units, heat_scale=problem.heat_scale
            )
            if heat <= cooling:
                candidate = (
                    problem.eu_per_pulse * pulse_units,
                    pulse_units,
                    heat,
                    rods,
                    minimum_fuel_cells,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
                break
    if best is None:
        # This is reachable only with a catalogue that provides no cooling and
        # a positive-pulse lower bound imposed elsewhere.  Zero is always a
        # mathematically safe upper bound for this relaxed objective.
        best = (0, 0, 0, 1, 1)
    power, pulses, heat, rods, fuel_cells = best
    cooling = envelope[max(0, problem.graph.size - fuel_cells)]
    return ClosedFormBound(
        power_upper_bound=power,
        pulse_units_upper_bound=pulses,
        heat_at_bound=heat,
        cooling_upper_bound=cooling,
        rods_used_for_bound=rods,
        minimum_fuel_cells=fuel_cells,
        derivation=(
            "integer convexity of q(p)=heat_scale*p*(p+1), minimum fuel-cell "
            "count, and optimistic persistent-vent heat conservation"
        ),
    )


def aggregate_overload_analysis(
    problem: ReactorProblem,
    power: int,
    *,
    ideal_receiver_sink: int = 20,
    hull_draw: int = 36,
    side_vent_cost: int = 4,
    side_vent_relief: int = 16,
    exchanger_cost: int = 20,
    exchanger_relief: int = 88,
    counted_fuel_ids: frozenset[str] | None = None,
) -> AggregateOverloadAnalysis:
    """Enumerate aggregate fuel-degree counts and test local overload relief.

    The caller supplies deliberately optimistic primitive capacities.  A
    ``exchanger_relief=88`` is the IC2 direct-receiver increment: a component
    exchanger that spends one edge receiving fuel heat has at most three
    36/t outgoing edges, but it replaces an ideal receiver that already
    disposed of 20/t, so its *additional* overload relief is at most
    ``3*36-20=88``.  A pure transshipment exchanger is bounded by 72/t.
    A positive margin for every count pattern is therefore a sound tier
    exclusion whenever those primitives dominate the real component catalogue.
    No vertex positions are enumerated.
    """

    if not problem.exact_rods or power < 0 or power % problem.eu_per_pulse:
        return AggregateOverloadAnalysis(power, 0, None, False, None, (), ())
    if min(
        ideal_receiver_sink,
        hull_draw,
        side_vent_cost,
        side_vent_relief,
        exchanger_cost,
        exchanger_relief,
    ) <= 0:
        raise ValueError("aggregate overload capacities must be positive")
    counted_fuel_ids = (
        frozenset(item.id for item in problem.power_components if item.rods > 0)
        if counted_fuel_ids is None
        else counted_fuel_ids
    )
    unknown_counted = counted_fuel_ids - {
        item.id for item in problem.power_components if item.rods > 0
    }
    if unknown_counted:
        raise ValueError(f"counted fuel ids are unknown or non-fuel: {sorted(unknown_counted)}")
    target_pulses = power // problem.eu_per_pulse
    relaxed_minimum_heat = minimum_heat_for_pulse_units(
        problem.rod_budget,
        target_pulses,
        heat_scale=problem.heat_scale,
    )
    maximum_active_cells = max(
        0,
        problem.graph.size - ceil(relaxed_minimum_heat / ideal_receiver_sink),
    )
    state_types = tuple(
        (
            item.id,
            degree,
            item.rods,
            item.rods * (item.internal_pulses + degree),
            problem.heat_scale
            * item.rods
            * (item.internal_pulses + degree)
            * (item.internal_pulses + degree + 1),
        )
        for item in problem.power_components
        if item.rods > 0
        for degree in range(problem.graph.maximum_degree + 1)
    )

    def receiver_relief(item_id: str, degree: int) -> int:
        item = next(value for value in problem.power_components if value.id == item_id)
        pulses = item.internal_pulses + degree
        per_rod_heat = problem.heat_scale * pulses * (pulses + 1)
        total_heat = item.rods * per_rod_heat
        receiver_count = problem.graph.maximum_degree - degree
        direct_relief = 10**12
        for count in range(1, receiver_count + 1):
            loads = [0] * count
            for _rod in range(item.rods):
                remaining = per_rod_heat
                for receiver in range(count):
                    amount = remaining // (count - receiver)
                    remaining -= amount
                    loads[receiver] += amount
            direct_relief = min(
                direct_relief,
                sum(max(0, load - ideal_receiver_sink) for load in loads),
            )
        full_draws, remainder = divmod(total_heat, hull_draw)
        hull_relief = (
            full_draws * max(0, hull_draw - ideal_receiver_sink)
            + max(0, remainder - ideal_receiver_sink)
        )
        return min(direct_relief, hull_relief)

    relief_by_state = tuple(
        receiver_relief(item_id, degree) if item_id in counted_fuel_ids else 0
        for item_id, degree, _rods, _pulses, _heat in state_types
    )
    margins: list[int] = []
    patterns: list[AggregatePattern] = []
    weakest: AggregatePattern | None = None

    def enumerate_counts(
        state_index: int,
        rods: int,
        pulses: int,
        fuel_cells: int,
        heat: int,
        required_relief: int,
        counts: tuple[int, ...],
    ) -> None:
        nonlocal weakest
        if (
            rods > problem.rod_budget
            or pulses > target_pulses
            or fuel_cells > maximum_active_cells
            or heat > ideal_receiver_sink * (problem.graph.size - fuel_cells)
        ):
            return
        if state_index == len(state_types):
            if rods != problem.rod_budget or pulses != target_pulses:
                return
            for reflectors in range(maximum_active_cells - fuel_cells + 1):
                active_cells = fuel_cells + reflectors
                slack = (
                    ideal_receiver_sink * (problem.graph.size - active_cells) - heat
                )
                if slack < 0:
                    continue
                maximum_relief = max(
                    exchanger_relief * exchangers
                    + side_vent_relief
                    * ((slack - exchanger_cost * exchangers) // side_vent_cost)
                    for exchangers in range(slack // exchanger_cost + 1)
                )
                margins.append(required_relief - maximum_relief)
                pattern = AggregatePattern(
                    active_cells=active_cells,
                    generated_heat=heat,
                    slack=slack,
                    required_relief=required_relief,
                    maximum_available_relief=maximum_relief,
                    margin=required_relief - maximum_relief,
                    fuel_degree_counts=tuple(
                        (item_id, degree, count)
                        for count, (item_id, degree, _rods, _pulses, _heat) in zip(
                            counts, state_types, strict=True
                        )
                        if count
                    ),
                )
                patterns.append(pattern)
                if weakest is None or pattern.margin < weakest.margin:
                    weakest = pattern
            return
        _item, _degree, state_rods, state_pulses, state_heat = state_types[state_index]
        maximum_count = min(
            maximum_active_cells - fuel_cells,
            (problem.rod_budget - rods) // state_rods,
            (target_pulses - pulses) // state_pulses,
        )
        for count in range(maximum_count + 1):
            enumerate_counts(
                state_index + 1,
                rods + count * state_rods,
                pulses + count * state_pulses,
                fuel_cells + count,
                heat + count * state_heat,
                required_relief + count * relief_by_state[state_index],
                (*counts, count),
            )

    enumerate_counts(0, 0, 0, 0, 0, 0, ())
    minimum_margin = min(margins) if margins else None
    survivors = tuple(pattern for pattern in patterns if pattern.margin <= 0)
    if _graph_is_bipartite(problem.graph):
        structural_survivors = tuple(
            pattern
            for pattern in survivors
            if bipartite_degree_sequence_possible(
                tuple(
                    degree
                    for _item, degree, count in pattern.fuel_degree_counts
                    for _ in range(count)
                ),
                unknown_vertices=(
                    pattern.active_cells
                    - sum(count for _item, _degree, count in pattern.fuel_degree_counts)
                ),
                maximum_degree=problem.graph.maximum_degree,
            )
        )
    else:
        structural_survivors = survivors
    return AggregateOverloadAnalysis(
        power=power,
        pattern_count=len(margins),
        minimum_relief_margin=minimum_margin,
        excluded=bool(margins) and minimum_margin is not None and minimum_margin > 0,
        weakest_pattern=weakest,
        surviving_patterns=survivors,
        structurally_surviving_patterns=structural_survivors,
    )


def derive_ic2_top_tier_cut(problem: ReactorProblem) -> AnalyticalCutProof | None:
    """Derive the 480/475/470/465/460 cuts when assumptions hold.

    This is an instance corollary attached to the generic model, not a hidden
    constant in the solver.  Every structural and catalogue assumption is
    checked before the stronger bound is returned; otherwise the function
    declines to produce a proof.
    """

    easy = closed_form_upper_bound(problem)
    if easy.power_upper_bound <= 455:
        return AnalyticalCutProof(
            power_upper_bound=easy.power_upper_bound,
            excluded_power_levels=(),
            assumptions=("closed-form heat conservation already dominates",),
            checks=(("closed_form_upper", easy.power_upper_bound),),
            derivation="convex heat lower bound and aggregate cooling envelope",
        )
    expected_power = {
        "empty": (0, 0, False),
        "uranium_single": (1, 1, True),
        "uranium_dual": (2, 2, True),
        "uranium_quad": (4, 3, True),
        "iridium_reflector": (0, 0, True),
    }
    actual_power = {
        item.id: (item.rods, item.internal_pulses, item.accepts_pulse)
        for item in problem.power_components
    }
    if (
        problem.graph.size != 54
        or problem.graph.maximum_degree > 4
        or problem.rod_budget != 25
        or not problem.exact_rods
        or problem.eu_per_pulse != 5
        or problem.heat_scale != 2
        or actual_power != expected_power
    ):
        return None
    cooling_rates = {
        item.id: item.self_vent + item.side_vent * problem.graph.maximum_degree
        for item in problem.cooling_components
    }
    if (
        cooling_rates.get("overclocked_heat_vent") != 20
        or cooling_rates.get("component_heat_vent", 0) > 16
        or any(value > 20 for value in cooling_rates.values())
        or sum(value == 20 for value in cooling_rates.values()) != 1
    ):
        return None
    maximum_shared_neighbours = max(
        (
            len(set(problem.graph.neighbours[first]) & set(problem.graph.neighbours[second]))
            for first in problem.graph.vertices
            for second in problem.graph.vertices
            if first != second
        ),
        default=0,
    )
    if maximum_shared_neighbours > 2:
        return None

    heat_480 = minimum_heat_for_pulse_units(25, 96)
    heat_475 = minimum_heat_for_pulse_units(25, 95)
    minimum_fuel_cells = ceil(25 / 4)
    capacity_7 = 20 * (54 - minimum_fuel_cells)
    minimum_quad_heat = 2 * 4 * 3 * 4
    best_quad_direct_sink = 4 * 20 + maximum_shared_neighbours * 4
    # These assertions are the executable arithmetic leaves of the proof.
    if not (
        heat_480 == 936
        and capacity_7 == 940
        and capacity_7 - 4 == heat_480
        and capacity_7 - 8 < heat_480
        and minimum_quad_heat == 96
        and best_quad_direct_sink == 88 < minimum_quad_heat
        and heat_475 == 920
        and 20 * (54 - 9) < heat_475
        and 20 * (54 - 8) == heat_475
    ):
        return None

    # At 475 with seven active cells, 25 rods force 6Q+S.  Internal pulse
    # units are 73.  4*sum(d_Q)+d_S=22 has only (5,2), whose total degree 7 is
    # odd and contradicts the handshaking lemma.
    degree_solutions = tuple(
        (quad_degree_sum, single_degree)
        for quad_degree_sum in range(6 * problem.graph.maximum_degree + 1)
        for single_degree in range(problem.graph.maximum_degree + 1)
        if 4 * quad_degree_sum + single_degree == 22
    )
    if degree_solutions != ((5, 2),) or sum(degree_solutions[0]) % 2 != 1:
        return None

    # The remaining 470, 465 and 460 layers can be closed without geometric
    # enumeration.  A component exchanger that directly accepts fuel heat has
    # at most three remaining side edges.  Their 3*36 total must also replace
    # the 20/t ideal receiver occupying that slot in the baseline, so the
    # additional overload relief is at most 88/t.  Without direct fuel input,
    # flow conservation lowers its pure transshipment bound to 72/t.
    counted_quads = frozenset({"uranium_quad"})
    analysis_470 = aggregate_overload_analysis(
        problem,
        470,
        counted_fuel_ids=counted_quads,
    )
    analysis_465 = aggregate_overload_analysis(
        problem,
        465,
        counted_fuel_ids=counted_quads,
    )
    analysis_460 = aggregate_overload_analysis(problem, 460)
    surviving_460 = analysis_460.surviving_patterns
    if (
        analysis_470.pattern_count != 13
        or not analysis_470.excluded
        or analysis_470.minimum_relief_margin is None
        or analysis_465.pattern_count != 82
        or not analysis_465.excluded
        or analysis_465.minimum_relief_margin is None
        or analysis_460.pattern_count != 219
        or not analysis_460.excluded
        or analysis_460.minimum_relief_margin is None
        or len(surviving_460) != 0
        or len(analysis_460.structurally_surviving_patterns) != 0
    ):
        return None

    return AnalyticalCutProof(
        power_upper_bound=455,
        excluded_power_levels=(480, 475, 470, 465, 460),
        assumptions=(
            "54-cell undirected graph of maximum degree four",
            "exactly 25 rods with IC2 S/D/Q bundle equations",
            "all persistent vents remove at most 20 heat per occupied slot",
            "two distinct cells share at most two neighbours",
            "row-ordered overclocked hull draw is 36 and its best tight-tier sink is 24",
        ),
        checks=(
            ("minimum_heat_480", heat_480),
            ("seven_active_cooling", capacity_7),
            ("minimum_quad_heat", minimum_quad_heat),
            ("best_quad_direct_sink", best_quad_direct_sink),
            ("minimum_heat_475", heat_475),
            ("seven_active_degree_sum", sum(degree_solutions[0])),
            ("aggregate_patterns_470", analysis_470.pattern_count),
            ("minimum_470_relief_margin", analysis_470.minimum_relief_margin),
            ("aggregate_patterns_465", analysis_465.pattern_count),
            ("minimum_465_relief_margin", analysis_465.minimum_relief_margin),
            ("aggregate_patterns_460", analysis_460.pattern_count),
            ("minimum_460_relief_margin", analysis_460.minimum_relief_margin),
            ("surviving_overload_patterns_460", len(surviving_460)),
            (
                "structurally_surviving_patterns_460",
                len(analysis_460.structurally_surviving_patterns),
            ),
        ),
        derivation=(
            "discrete convexity fixes the tight heat budget; the 480 layer then "
            "has insufficient local quad cooling (including the only possible "
            "component-vent gain), while the final seven-active branch of the "
            "475 layer violates the handshaking parity lemma; the 470 layer "
            "has only thirteen aggregate fuel-degree patterns, each of which "
            "requires more local overload relief than its cooling slack can "
            "buy; the same count-space proof closes 465 after generously "
            "crediting every directly-fed exchanger with 88/t of additional "
            "relief; at 460, counting the same unavoidable overload for every "
            "fuel type eliminates all 219 aggregate patterns directly"
        ),
    )


@dataclass(frozen=True, slots=True)
class MasterSolution:
    status: str
    feasible: bool
    proven_optimal: bool
    power: int | None
    generated_heat: int | None
    active_cells: int | None
    skeleton: tuple[str, ...] | None
    strict_power_upper_bound: int
    elapsed_seconds: float
    conflicts: int
    branches: int


class PowerHeatMaster:
    """Exact static power model plus a sound aggregate thermal relaxation."""

    def __init__(self, problem: ReactorProblem) -> None:
        self.problem = problem

    def build(
        self,
        *,
        use_cooling_envelope: bool = True,
        minimum_power: int | None = None,
        exact_power: int | None = None,
        maximum_heat: int | None = None,
    ):
        try:
            from ortools.sat.python import cp_model
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError("PowerHeatMaster requires OR-Tools") from error

        problem = self.problem
        graph = problem.graph
        types = problem.power_components
        model = cp_model.CpModel()
        one_hot = [
            [model.new_bool_var(f"x_{vertex}_{code}") for code in range(len(types))]
            for vertex in graph.vertices
        ]
        active = [model.new_bool_var(f"active_{vertex}") for vertex in graph.vertices]
        degrees = [
            model.new_int_var(0, len(graph.neighbours[vertex]), f"degree_{vertex}")
            for vertex in graph.vertices
        ]
        power_terms = []
        heat_terms = []

        for vertex in graph.vertices:
            model.add_exactly_one(one_hot[vertex])
            model.add(active[vertex] == sum(
                one_hot[vertex][code]
                for code, item in enumerate(types)
                if item.accepts_pulse
            ))
            model.add(degrees[vertex] == sum(active[other] for other in graph.neighbours[vertex]))
            for code in problem.fuel_codes:
                item = types[code]
                degree_states = []
                for degree in range(len(graph.neighbours[vertex]) + 1):
                    state = model.new_bool_var(f"z_{vertex}_{code}_{degree}")
                    model.add(degrees[vertex] == degree).only_enforce_if(state)
                    degree_states.append(state)
                    pulses = item.internal_pulses + degree
                    power_terms.append(
                        problem.eu_per_pulse * item.rods * pulses * state
                    )
                    heat_terms.append(
                        problem.heat_scale * item.rods * pulses * (pulses + 1) * state
                    )
                model.add(sum(degree_states) == one_hot[vertex][code])

        rods = sum(
            types[code].rods * one_hot[vertex][code]
            for vertex in graph.vertices
            for code in problem.fuel_codes
        )
        if problem.exact_rods:
            model.add(rods == problem.rod_budget)
        else:
            model.add(rods <= problem.rod_budget)
            model.add(rods >= 1)

        limits = dict(problem.component_limits)
        for code, item in enumerate(types):
            if code == 0:
                # In the power master x[v,empty] means "any cooling-layer
                # component", not necessarily a physically empty slot.
                continue
            limit = limits.get(item.id)
            if limit is not None:
                model.add(sum(one_hot[vertex][code] for vertex in graph.vertices) <= limit)

        maximum_pulse_units = (
            problem.rod_budget
            * (max(item.internal_pulses for item in types if item.rods) + graph.maximum_degree)
        )
        maximum_power_value = problem.eu_per_pulse * maximum_pulse_units
        maximum_internal = max(item.internal_pulses for item in types if item.rods)
        maximum_heat_value = (
            problem.heat_scale
            * problem.rod_budget
            * (graph.maximum_degree + maximum_internal)
            * (graph.maximum_degree + maximum_internal + 1)
        )
        power = model.new_int_var(0, maximum_power_value, "power")
        heat = model.new_int_var(0, maximum_heat_value, "generated_heat")
        active_count = model.new_int_var(0, graph.size, "active_count")
        model.add(power == sum(power_terms))
        model.add(heat == sum(heat_terms))
        model.add(active_count == sum(active))

        if use_cooling_envelope:
            envelope = cooling_envelope(problem)
            free_count = model.new_int_var(0, graph.size, "free_count")
            cooling = model.new_int_var(0, envelope[-1], "optimistic_cooling")
            model.add(free_count == graph.size - active_count)
            model.add_element(free_count, envelope, cooling)
            model.add(heat <= cooling)
        if minimum_power is not None:
            model.add(power >= minimum_power)
        if exact_power is not None:
            model.add(power == exact_power)
        if maximum_heat is not None:
            model.add(heat <= maximum_heat)
        model.maximize(power)
        variables = {
            "one_hot": one_hot,
            "active": active,
            "degrees": degrees,
            "power": power,
            "heat": heat,
            "active_count": active_count,
        }
        return model, variables

    def solve(
        self,
        *,
        seconds: float = 60.0,
        workers: int = 1,
        random_seed: int = 221,
        use_cooling_envelope: bool = True,
        minimum_power: int | None = None,
        exact_power: int | None = None,
        maximum_heat: int | None = None,
    ) -> MasterSolution:
        from ortools.sat.python import cp_model

        if seconds <= 0:
            raise ValueError("seconds must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        model, variables = self.build(
            use_cooling_envelope=use_cooling_envelope,
            minimum_power=minimum_power,
            exact_power=exact_power,
            maximum_heat=maximum_heat,
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = random_seed
        started = perf_counter()
        status_code = solver.solve(model)
        elapsed = perf_counter() - started
        status = solver.status_name(status_code)
        feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        proven = status_code == cp_model.OPTIMAL
        # CP-SAT's best bound is a mathematical upper bound for maximisation.
        # ``ceil`` protects against a downward floating representation error.
        raw_bound = solver.best_objective_bound
        strict_bound = max(0, ceil(raw_bound - 1e-9))
        if not feasible:
            return MasterSolution(
                status=status,
                feasible=False,
                proven_optimal=proven,
                power=None,
                generated_heat=None,
                active_cells=None,
                skeleton=None,
                strict_power_upper_bound=strict_bound,
                elapsed_seconds=elapsed,
                conflicts=solver.num_conflicts,
                branches=solver.num_branches,
            )

        one_hot = variables["one_hot"]
        types = self.problem.power_components
        skeleton = tuple(
            types[next(
                code for code, flag in enumerate(one_hot[vertex]) if solver.value(flag)
            )].id
            for vertex in self.problem.graph.vertices
        )
        power = solver.value(variables["power"])
        if proven:
            strict_bound = power
        return MasterSolution(
            status=status,
            feasible=True,
            proven_optimal=proven,
            power=power,
            generated_heat=solver.value(variables["heat"]),
            active_cells=solver.value(variables["active_count"]),
            skeleton=skeleton,
            strict_power_upper_bound=strict_bound,
            elapsed_seconds=elapsed,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
        )


@dataclass(frozen=True, slots=True)
class CycleCertificate:
    outcome: str
    safe: bool
    conclusive: bool
    power: int
    generated_heat: int
    transient_length: int | None
    period_length: int | None
    checked_ticks: int
    peak_hull_heat: int
    peak_component_heat: int
    failure_tick: int | None = None
    failure_component: int | None = None


class IC2CycleOracle:
    """Exact positive/negative checker for a fixed completed IC2 layout."""

    def verify(
        self,
        layout: Sequence[str],
        *,
        columns: int,
        max_ticks: int = 100_000,
        time_limit_seconds: float | None = None,
    ) -> CycleCertificate:
        if columns <= 0 or len(layout) != 6 * columns:
            raise ValueError("the locked IC2 ruleset has six rows")
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")

        # The generic verifier owns cycle logic; this class only translates the
        # IC2 adapter's metrics into the public IC2 certificate shape.
        from .cycle_proof import DeterministicCycleVerifier, IC2TransitionSystem

        proof = DeterministicCycleVerifier().verify(
            IC2TransitionSystem(columns),
            tuple(layout),
            max_steps=max_ticks,
            time_limit_seconds=time_limit_seconds,
        )
        metrics = dict(proof.last_metrics)
        return CycleCertificate(
            outcome=proof.outcome,
            safe=proof.safe,
            conclusive=proof.conclusive,
            power=int(metrics.get("power") or 0),
            generated_heat=int(metrics.get("generated_heat") or 0),
            transient_length=proof.transient_length,
            period_length=proof.period_length,
            checked_ticks=proof.checked_steps,
            peak_hull_heat=int(metrics.get("peak_hull_heat") or 0),
            peak_component_heat=int(metrics.get("peak_component_heat") or 0),
            failure_tick=proof.failure_step,
            failure_component=(
                int(metrics["failure_component"])
                if metrics.get("failure_component") is not None
                else None
            ),
        )


def format_rectangular(labels: Sequence[str], rows: int, columns: int) -> str:
    if len(labels) != rows * columns:
        raise ValueError("label count does not match rectangular dimensions")
    return "\n".join(
        " ".join(labels[row * columns:(row + 1) * columns])
        for row in range(rows)
    )
