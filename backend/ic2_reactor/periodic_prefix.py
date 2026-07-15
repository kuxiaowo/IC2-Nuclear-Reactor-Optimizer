"""Exact-integer periodic prefix-flow relaxation for ordered thermal updates.

Any safe deterministic cycle induces a feasible circulation after averaging
each reservoir state and transfer over the cycle.  This time-expanded network
keeps event order, finite storage gates, exact fuel heat splitting and transfer
rate upper bounds, while relaxing nonlinear exchanger direction and magnitude.
Hence an infeasible circulation is a rigorous positive-drift/prefix-capacity
certificate; a feasible circulation is only a necessary condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .mathematical_model import ReactorProblem, evaluate_power_skeleton
from .thermal_relaxation import _maximum_flow_with_cut


@dataclass(frozen=True, slots=True)
class PrefixHeatComponent:
    heat_capacity: int = 0
    accepts_fuel_heat: bool = False
    optional_fuel_acceptance: bool = False
    self_vent: int = 0
    side_vent: int = 0
    hull_draw: int = 0
    exchange_side: int = 0
    exchange_hull: int = 0
    hull_capacity_bonus: int = 0

    def __post_init__(self) -> None:
        values = (
            self.heat_capacity,
            self.self_vent,
            self.side_vent,
            self.hull_draw,
            self.exchange_side,
            self.exchange_hull,
            self.hull_capacity_bonus,
        )
        if any(value < 0 for value in values):
            raise ValueError("periodic-prefix capacities must be non-negative")
        if self.accepts_fuel_heat and self.heat_capacity <= 0:
            raise ValueError("fuel heat acceptance requires positive storage capacity")
        if self.optional_fuel_acceptance and not self.accepts_fuel_heat:
            raise ValueError("optional fuel acceptance requires accepts_fuel_heat=True")


def componentwise_prefix_dominator(
    problem: ReactorProblem,
    catalogue: Mapping[str, PrefixHeatComponent],
) -> PrefixHeatComponent:
    """Combine all best free-slot prefix capacities into one optional receiver."""

    power_ids = {item.id for item in problem.power_components}
    free_labels = ({"empty"} | set(problem.layout_components)) - (
        power_ids - {"empty"}
    )
    if missing := free_labels - catalogue.keys():
        raise ValueError(f"prefix catalogue is missing free labels: {sorted(missing)}")
    specifications = [catalogue[label] for label in free_labels]
    accepts = any(item.accepts_fuel_heat for item in specifications)
    return PrefixHeatComponent(
        heat_capacity=max((item.heat_capacity for item in specifications), default=0),
        accepts_fuel_heat=accepts,
        optional_fuel_acceptance=accepts,
        self_vent=max((item.self_vent for item in specifications), default=0),
        side_vent=max((item.side_vent for item in specifications), default=0),
        hull_draw=max((item.hull_draw for item in specifications), default=0),
        exchange_side=max((item.exchange_side for item in specifications), default=0),
        exchange_hull=max((item.exchange_hull for item in specifications), default=0),
        hull_capacity_bonus=max(
            (item.hull_capacity_bonus for item in specifications),
            default=0,
        ),
    )


@dataclass(frozen=True, slots=True)
class PeriodicPrefixFlowResult:
    feasible: bool
    generated_heat: int
    required_circulation: int
    routed_circulation: int
    deficit: int
    reservoir_count: int
    event_count: int
    node_count: int
    edge_count: int
    source_side_nodes: tuple[str, ...]
    cut_template: "PeriodicPrefixCutTemplate"


@dataclass(frozen=True, slots=True)
class PeriodicPrefixCutEvaluation:
    lower_bound_into_source_side: int
    upper_bound_out_of_source_side: int
    necessary_condition_satisfied: bool
    deficit: int


@dataclass(frozen=True, slots=True)
class PeriodicPrefixCutTemplate:
    """A Hoffman circulation inequality on canonical event-slot nodes."""

    source_side_nodes: tuple[str, ...]

    def evaluate(
        self,
        problem: ReactorProblem,
        layout: Sequence[str],
        catalogue: Mapping[str, PrefixHeatComponent],
        *,
        base_hull_capacity: int,
    ) -> PeriodicPrefixCutEvaluation:
        builder, _generated = _build_periodic_network(
            problem,
            layout,
            catalogue,
            base_hull_capacity=base_hull_capacity,
        )
        known = set(builder.names)
        if unknown := set(self.source_side_nodes) - known:
            raise ValueError(f"prefix cut has unknown canonical nodes: {sorted(unknown)}")
        source_side = {
            index
            for index, name in enumerate(builder.names)
            if name in self.source_side_nodes
        }
        lower_in = sum(
            edge.lower
            for edge in builder.edges
            if edge.start not in source_side and edge.end in source_side
        )
        upper_out = sum(
            edge.upper
            for edge in builder.edges
            if edge.start in source_side and edge.end not in source_side
        )
        return PeriodicPrefixCutEvaluation(
            lower_bound_into_source_side=lower_in,
            upper_bound_out_of_source_side=upper_out,
            necessary_condition_satisfied=lower_in <= upper_out,
            deficit=max(0, lower_in - upper_out),
        )


@dataclass(frozen=True, slots=True)
class _LowerEdge:
    start: int
    end: int
    lower: int
    upper: int


class _NetworkBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.edges: list[_LowerEdge] = []

    def node(self, name: str) -> int:
        result = len(self.names)
        self.names.append(name)
        return result

    def edge(self, start: int, end: int, upper: int, lower: int = 0) -> None:
        if not 0 <= lower <= upper:
            raise ValueError("lower-bound edge must satisfy 0 <= lower <= upper")
        if upper:
            self.edges.append(_LowerEdge(start, end, lower, upper))

    def circulation(self) -> tuple[bool, int, int, frozenset[int], int, int]:
        balances = [0] * len(self.names)
        residual_edges: list[tuple[int, int, int]] = []
        for edge in self.edges:
            balances[edge.start] -= edge.lower
            balances[edge.end] += edge.lower
            residual_edges.append((edge.start, edge.end, edge.upper - edge.lower))
        super_source = len(self.names)
        super_sink = super_source + 1
        required = 0
        for node, balance in enumerate(balances):
            if balance > 0:
                residual_edges.append((super_source, node, balance))
                required += balance
            elif balance < 0:
                residual_edges.append((node, super_sink, -balance))
        routed, reachable = _maximum_flow_with_cut(
            len(self.names) + 2,
            residual_edges,
            super_source,
            super_sink,
        )
        return (
            routed == required,
            required,
            routed,
            reachable,
            super_source,
            super_sink,
        )


def _ordered_split(total: int, receivers: Sequence[int]) -> tuple[dict[int, int], int]:
    shares = {receiver: 0 for receiver in receivers}
    remaining = total
    pending = list(receivers)
    while pending and remaining > 0:
        amount = remaining // len(pending)
        remaining -= amount
        shares[pending.pop(0)] += amount
    return shares, remaining


def _build_periodic_network(
    problem: ReactorProblem,
    layout: Sequence[str],
    catalogue: Mapping[str, PrefixHeatComponent],
    *,
    base_hull_capacity: int,
) -> tuple[_NetworkBuilder, int]:
    graph = problem.graph
    if len(layout) != graph.size:
        raise ValueError("layout length does not match graph")
    if base_hull_capacity <= 0:
        raise ValueError("base_hull_capacity must be positive")
    if missing := set(layout) - catalogue.keys():
        raise ValueError(f"prefix catalogue is missing labels: {sorted(missing)}")
    if any(catalogue[label].optional_fuel_acceptance for label in layout):
        raise ValueError(
            "fixed-layout prefix flow requires resolved fuel-acceptance choices"
        )
    power_by_id = {item.id: item for item in problem.power_components}
    skeleton = tuple(label if label in power_by_id else "empty" for label in layout)
    metrics = evaluate_power_skeleton(problem, skeleton)
    if problem.exact_rods and metrics.rods != problem.rod_budget:
        raise ValueError("layout does not satisfy the exact rod budget")

    hull_key = -1
    # Every slot has canonical event nodes, even when its current label has
    # zero storage.  Zero-capacity gates disable it.  This layout-independent
    # topology lets a failed min-cut be re-evaluated on other labelings.
    reservoirs = (hull_key, *graph.vertices)
    capacity = {hull_key: base_hull_capacity + sum(
        catalogue[label].hull_capacity_bonus for label in layout
    )}
    capacity.update({
        vertex: catalogue[layout[vertex]].heat_capacity
        for vertex in graph.vertices
    })
    if capacity[hull_key] <= 0:
        raise ValueError("total hull capacity must be positive")

    order = graph.update_order
    event_count = len(order)
    builder = _NetworkBuilder()
    external = builder.node("environment")
    before: dict[tuple[int, int], int] = {}
    after: dict[tuple[int, int], int] = {}
    for event in range(event_count):
        for reservoir in reservoirs:
            label = "hull" if reservoir == hull_key else f"slot_{reservoir}"
            before[event, reservoir] = builder.node(f"e{event}:{label}:in")
            after[event, reservoir] = builder.node(f"e{event}:{label}:out")
            builder.edge(
                before[event, reservoir],
                after[event, reservoir],
                capacity[reservoir],
            )

    def following(event: int, reservoir: int) -> int:
        return before[(event + 1) % event_count, reservoir]

    generated_heat = 0
    for event, vertex in enumerate(order):
        label = layout[vertex]
        spec = catalogue[label]
        power_spec = power_by_id.get(label)
        next_event = (event + 1) % event_count

        # The current slot always passes through a canonical intermediate pool.
        # Hull draw enters this pool and may be vented during the same action.
        special = {vertex}
        for reservoir in reservoirs:
            if reservoir not in special:
                builder.edge(
                    after[event, reservoir],
                    before[next_event, reservoir],
                    capacity[reservoir],
                )

        middle_in = builder.node(f"e{event}:slot_{vertex}:action_pool_in")
        middle_out = builder.node(f"e{event}:slot_{vertex}:action_pool_out")
        builder.edge(after[event, vertex], middle_in, capacity[vertex])
        builder.edge(after[event, hull_key], middle_in, spec.hull_draw)
        builder.edge(middle_in, middle_out, capacity[vertex])
        builder.edge(middle_out, before[next_event, vertex], capacity[vertex])
        builder.edge(middle_out, external, spec.self_vent)

        if spec.side_vent:
            for neighbour in graph.neighbours[vertex]:
                builder.edge(after[event, neighbour], external, spec.side_vent)

        if spec.exchange_side or spec.exchange_hull:
            for neighbour in graph.neighbours[vertex]:
                builder.edge(
                    after[event, vertex],
                    before[next_event, neighbour],
                    spec.exchange_side,
                )
                builder.edge(
                    after[event, neighbour],
                    before[next_event, vertex],
                    spec.exchange_side,
                )
            if spec.exchange_hull:
                builder.edge(
                    after[event, vertex],
                    before[next_event, hull_key],
                    spec.exchange_hull,
                )
                builder.edge(
                    after[event, hull_key],
                    before[next_event, vertex],
                    spec.exchange_hull,
                )

        if power_spec is not None and power_spec.rods > 0:
            pulse_count = power_spec.internal_pulses + metrics.degrees[vertex]
            per_rod_heat = problem.heat_scale * pulse_count * (pulse_count + 1)
            acceptors = tuple(
                neighbour
                for neighbour in graph.neighbours[vertex]
                if catalogue[layout[neighbour]].accepts_fuel_heat
            )
            total_shares = {receiver: 0 for receiver in acceptors}
            hull_share = 0
            for _rod in range(power_spec.rods):
                shares, remainder = _ordered_split(per_rod_heat, acceptors)
                for receiver, amount in shares.items():
                    total_shares[receiver] += amount
                hull_share += remainder
            for receiver, amount in total_shares.items():
                if amount:
                    builder.edge(
                        external,
                        before[next_event, receiver],
                        amount,
                        amount,
                    )
            if hull_share:
                builder.edge(
                    external,
                    before[next_event, hull_key],
                    hull_share,
                    hull_share,
                )
            generated_heat += power_spec.rods * per_rod_heat

    return builder, generated_heat


def periodic_prefix_flow_bound(
    problem: ReactorProblem,
    layout: Sequence[str],
    catalogue: Mapping[str, PrefixHeatComponent],
    *,
    base_hull_capacity: int,
) -> PeriodicPrefixFlowResult:
    """Check a fixed layout by an integer time-expanded cyclic circulation.

    Monotone condensator-like stores should be represented with
    ``accepts_fuel_heat=False`` and zero useful storage: in a repeated state
    they cannot have positive net input because their heat never decreases.
    Hull capacity bonuses are granted from the beginning of every pass, which
    is optimistic when the real rules activate plating later in update order.
    """

    builder, generated_heat = _build_periodic_network(
        problem,
        layout,
        catalogue,
        base_hull_capacity=base_hull_capacity,
    )

    feasible, required, routed, reachable, super_source, super_sink = (
        builder.circulation()
    )
    source_names = tuple(
        builder.names[node]
        for node in sorted(reachable)
        if node < len(builder.names)
    )
    # Super nodes are not included in the human-readable cut; their presence
    # is implicit in the lower-bound circulation transformation.
    _ = super_source, super_sink
    cut_template = PeriodicPrefixCutTemplate(source_names)
    cut_evaluation = cut_template.evaluate(
        problem,
        layout,
        catalogue,
        base_hull_capacity=base_hull_capacity,
    )
    if cut_evaluation.deficit != required - routed:
        raise AssertionError("Hoffman cut deficit disagrees with transformed max-flow")
    return PeriodicPrefixFlowResult(
        feasible=feasible,
        generated_heat=generated_heat,
        required_circulation=required,
        routed_circulation=routed,
        deficit=required - routed,
        reservoir_count=problem.graph.size + 1,
        event_count=problem.graph.size,
        node_count=len(builder.names) + 2,
        edge_count=len(builder.edges),
        source_side_nodes=source_names,
        cut_template=cut_template,
    )
