"""Layout-aware optimistic heat-flow relaxation and min-cut certificate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .mathematical_model import ReactorProblem, evaluate_power_skeleton


@dataclass(frozen=True, slots=True)
class HeatFlowComponent:
    """Catalogue entry for the optimistic heat-flow relaxation.

    This is deliberately smaller than any simulator component type.  A new
    ruleset can use the relaxation by supplying these transfer capacities;
    the generic flow model has no dependency on the IC2 catalogue.
    """

    accepts_heat: bool = False
    self_vent: int = 0
    side_vent: int = 0
    hull_draw: int = 0
    exchange_side: int = 0
    exchange_hull: int = 0
    # Condensator-like storage may stop accepting once full.  Ordered heat
    # relaxations may choose either state independently, which is optimistic.
    optional_heat_acceptance: bool = False

    def __post_init__(self) -> None:
        values = (
            self.self_vent,
            self.side_vent,
            self.hull_draw,
            self.exchange_side,
            self.exchange_hull,
        )
        if any(value < 0 for value in values):
            raise ValueError("heat-flow capacities must be non-negative")
        if self.optional_heat_acceptance and not self.accepts_heat:
            raise ValueError("optional heat acceptance requires accepts_heat=True")


@dataclass(frozen=True, slots=True)
class ThermalFlowBound:
    generated_heat: int
    maximum_removable_heat: int
    necessary_condition_satisfied: bool
    source_side_slots: tuple[int, ...]
    sink_side_slots: tuple[int, ...]
    deficit: int
    cut_template: "ThermalCutTemplate"


@dataclass(frozen=True, slots=True)
class ThermalCutEvaluation:
    generated_heat: int
    cut_capacity: int
    necessary_condition_satisfied: bool
    deficit: int


@dataclass(frozen=True, slots=True)
class ThermalCutTemplate:
    """A fixed node partition yielding a valid cut for every layout.

    Edge capacities remain functions of component labels.  Re-evaluating the
    same partition on another layout gives a globally valid Benders necessary
    inequality ``generated_heat <= cut_capacity``.
    """

    source_storage_slots: tuple[int, ...]
    source_generator_slots: tuple[int, ...]
    hull_source_side: bool

    def evaluate(
        self,
        problem: ReactorProblem,
        layout: Sequence[str],
        heat_catalogue: Mapping[str, HeatFlowComponent],
    ) -> ThermalCutEvaluation:
        network = _build_flow_network(problem, layout, heat_catalogue)
        if any(vertex not in problem.graph.vertices for vertex in self.source_storage_slots):
            raise ValueError("cut contains a storage slot outside the graph")
        if any(vertex not in problem.graph.vertices for vertex in self.source_generator_slots):
            raise ValueError("cut contains a generator slot outside the graph")
        source_side = {network.source}
        source_side.update(
            network.storage_offset + vertex for vertex in self.source_storage_slots
        )
        source_side.update(
            network.generator_offset + vertex for vertex in self.source_generator_slots
        )
        if self.hull_source_side:
            source_side.add(network.hull)
        capacity = sum(
            edge_capacity
            for start, end, edge_capacity in network.edges
            if start in source_side and end not in source_side
        )
        return ThermalCutEvaluation(
            generated_heat=network.generated_heat,
            cut_capacity=capacity,
            necessary_condition_satisfied=capacity >= network.generated_heat,
            deficit=max(0, network.generated_heat - capacity),
        )


IDEAL_COOLING_LABEL = "__ideal_componentwise_cooling_dominator__"


def componentwise_cooling_dominator(
    problem: ReactorProblem,
    heat_catalogue: Mapping[str, HeatFlowComponent],
) -> HeatFlowComponent:
    """Return one deliberately impossible component dominating every free label.

    The result combines the largest capacity of each enabled non-power label
    in a single slot.  It is therefore suitable only for *necessary* tests:
    failure with this super-component proves failure for every real cooling
    completion, while success proves nothing.
    """

    power_ids = {item.id for item in problem.power_components}
    free_labels = ({"empty"} | set(problem.layout_components)) - (
        power_ids - {"empty"}
    )
    missing = free_labels - heat_catalogue.keys()
    if missing:
        raise ValueError(f"heat catalogue is missing enabled labels: {sorted(missing)}")
    specifications = [heat_catalogue[label] for label in free_labels]
    return HeatFlowComponent(
        accepts_heat=any(item.accepts_heat for item in specifications),
        self_vent=max((item.self_vent for item in specifications), default=0),
        side_vent=max((item.side_vent for item in specifications), default=0),
        hull_draw=max((item.hull_draw for item in specifications), default=0),
        exchange_side=max((item.exchange_side for item in specifications), default=0),
        exchange_hull=max((item.exchange_hull for item in specifications), default=0),
    )


def _idealized_skeleton_layout(
    problem: ReactorProblem,
    skeleton: Sequence[str],
    heat_catalogue: Mapping[str, HeatFlowComponent],
) -> tuple[tuple[str, ...], dict[str, HeatFlowComponent]]:
    if len(skeleton) != problem.graph.size:
        raise ValueError("skeleton length does not match graph")
    power_ids = {item.id for item in problem.power_components}
    if unknown := set(skeleton) - power_ids:
        raise ValueError(f"unknown power labels: {sorted(unknown)}")
    if missing_power := power_ids - heat_catalogue.keys():
        raise ValueError(
            f"heat catalogue is missing power labels: {sorted(missing_power)}"
        )
    ideal = componentwise_cooling_dominator(problem, heat_catalogue)
    ideal_catalogue = {
        item.id: heat_catalogue[item.id]
        for item in problem.power_components
    }
    ideal_catalogue[IDEAL_COOLING_LABEL] = ideal
    layout = tuple(
        IDEAL_COOLING_LABEL if label == "empty" else label
        for label in skeleton
    )
    return layout, ideal_catalogue


def skeleton_heat_flow_bound(
    problem: ReactorProblem,
    skeleton: Sequence[str],
    heat_catalogue: Mapping[str, HeatFlowComponent],
) -> ThermalFlowBound:
    """Test a power skeleton with an ideal component in every cooling slot.

    In addition to combining mutually exclusive real capabilities, the base
    flow network lets fuel choose both hull and neighbouring receiver routes.
    The relaxation is thus safely optimistic and requires only one max-flow;
    a deficit excludes all real cooling assignments for the skeleton.
    """

    layout, ideal_catalogue = _idealized_skeleton_layout(
        problem,
        skeleton,
        heat_catalogue,
    )
    return layout_heat_flow_bound(problem, layout, ideal_catalogue)


def evaluate_skeleton_thermal_cut(
    problem: ReactorProblem,
    skeleton: Sequence[str],
    heat_catalogue: Mapping[str, HeatFlowComponent],
    cut: ThermalCutTemplate,
) -> ThermalCutEvaluation:
    """Re-evaluate an ideal-dominator cut on another power skeleton."""

    layout, ideal_catalogue = _idealized_skeleton_layout(
        problem,
        skeleton,
        heat_catalogue,
    )
    return cut.evaluate(problem, layout, ideal_catalogue)


@dataclass(frozen=True, slots=True)
class _FlowNetwork:
    generated_heat: int
    node_count: int
    edges: tuple[tuple[int, int, int], ...]
    storage_offset: int
    generator_offset: int
    hull: int
    source: int
    sink: int


def _maximum_flow_with_cut(
    node_count: int,
    edges: Sequence[tuple[int, int, int]],
    source: int,
    sink: int,
) -> tuple[int, frozenset[int]]:
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(start: int, end: int, capacity: int) -> None:
        if capacity <= 0:
            return
        forward = [end, capacity, len(graph[end])]
        reverse = [start, 0, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(reverse)

    for start, end, capacity in edges:
        add_edge(start, end, capacity)

    total = 0
    while True:
        level = [-1] * node_count
        level[source] = 0
        queue = [source]
        for node in queue:
            for end, capacity, _reverse in graph[node]:
                if capacity > 0 and level[end] < 0:
                    level[end] = level[node] + 1
                    queue.append(end)
        if level[sink] < 0:
            break
        cursor = [0] * node_count

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                end, capacity, reverse_index = edge
                if capacity > 0 and level[end] == level[node] + 1:
                    pushed = send(end, min(amount, capacity))
                    if pushed:
                        edge[1] -= pushed
                        graph[end][reverse_index][1] += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while pushed := send(source, 10**9):
            total += pushed

    reachable = {source}
    queue = [source]
    for node in queue:
        for end, capacity, _reverse in graph[node]:
            if capacity > 0 and end not in reachable:
                reachable.add(end)
                queue.append(end)
    return total, frozenset(reachable)


def _build_flow_network(
    problem: ReactorProblem,
    layout: Sequence[str],
    heat_catalogue: Mapping[str, HeatFlowComponent],
) -> _FlowNetwork:
    graph = problem.graph
    if len(layout) != graph.size:
        raise ValueError("layout length does not match graph")
    unknown = set(layout) - heat_catalogue.keys()
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")
    power_by_id = {item.id: item for item in problem.power_components}
    skeleton = tuple(item if item in power_by_id else "empty" for item in layout)
    metrics = evaluate_power_skeleton(problem, skeleton)

    slots = graph.size
    storage_offset = 0
    generator_offset = slots
    hull = slots * 2
    source = hull + 1
    sink = source + 1
    edges: list[tuple[int, int, int]] = []

    for vertex, item in enumerate(layout):
        spec = heat_catalogue[item]
        power_spec = power_by_id.get(item)
        if power_spec is not None and power_spec.rods > 0:
            degree = metrics.degrees[vertex]
            pulses = power_spec.internal_pulses + degree
            heat = problem.heat_scale * power_spec.rods * pulses * (pulses + 1)
            generator = generator_offset + vertex
            edges.append((source, generator, heat))
            # Optimistically grant the hull and every adjacent accepting store.
            edges.append((generator, hull, heat))
            for neighbour in graph.neighbours[vertex]:
                if heat_catalogue[layout[neighbour]].accepts_heat:
                    edges.append((generator, storage_offset + neighbour, heat))

        if spec.accepts_heat:
            external = spec.self_vent + sum(
                heat_catalogue[layout[neighbour]].side_vent
                for neighbour in graph.neighbours[vertex]
            )
            edges.append((storage_offset + vertex, sink, external))

        if spec.hull_draw and spec.accepts_heat:
            edges.append((hull, storage_offset + vertex, spec.hull_draw))

        if spec.exchange_side or spec.exchange_hull:
            for neighbour in graph.neighbours[vertex]:
                if not heat_catalogue[layout[neighbour]].accepts_heat:
                    continue
                capacity = spec.exchange_side
                edges.append((storage_offset + vertex, storage_offset + neighbour, capacity))
                edges.append((storage_offset + neighbour, storage_offset + vertex, capacity))
            if spec.exchange_hull:
                edges.append((storage_offset + vertex, hull, spec.exchange_hull))
                edges.append((hull, storage_offset + vertex, spec.exchange_hull))

    return _FlowNetwork(
        generated_heat=metrics.generated_heat,
        node_count=sink + 1,
        edges=tuple(edges),
        storage_offset=storage_offset,
        generator_offset=generator_offset,
        hull=hull,
        source=source,
        sink=sink,
    )


def layout_heat_flow_bound(
    problem: ReactorProblem,
    layout: Sequence[str],
    heat_catalogue: Mapping[str, HeatFlowComponent],
) -> ThermalFlowBound:
    """Return a sound necessary average-flow condition for a fixed layout.

    The network deliberately ignores update order, ratios, rounding, finite
    capacities and competition between transfers.  Fuel may even choose the
    hull when a real acceptor exists.  Therefore the maximum flow can only be
    *larger* than physically realisable heat removal: a deficit proves
    infeasibility, while a saturated flow proves nothing by itself.
    """

    graph = problem.graph
    network = _build_flow_network(problem, layout, heat_catalogue)
    flow, reachable = _maximum_flow_with_cut(
        network.node_count,
        network.edges,
        network.source,
        network.sink,
    )
    cut_template = ThermalCutTemplate(
        source_storage_slots=tuple(
            vertex
            for vertex in graph.vertices
            if network.storage_offset + vertex in reachable
        ),
        source_generator_slots=tuple(
            vertex
            for vertex in graph.vertices
            if network.generator_offset + vertex in reachable
        ),
        hull_source_side=network.hull in reachable,
    )
    source_slots = tuple(
        vertex
        for vertex in graph.vertices
        if (
            network.storage_offset + vertex in reachable
            or network.generator_offset + vertex in reachable
        )
    )
    sink_slots = tuple(vertex for vertex in graph.vertices if vertex not in source_slots)
    return ThermalFlowBound(
        generated_heat=network.generated_heat,
        maximum_removable_heat=flow,
        necessary_condition_satisfied=flow >= network.generated_heat,
        source_side_slots=source_slots,
        sink_side_slots=sink_slots,
        deficit=max(0, network.generated_heat - flow),
        cut_template=cut_template,
    )
