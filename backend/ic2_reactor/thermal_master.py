"""Full-label CP-SAT master with reusable thermal min-cut inequalities."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Mapping, Sequence

from .mathematical_model import AggregatePattern, ReactorProblem
from .periodic_prefix import PeriodicPrefixCutTemplate, PrefixHeatComponent
from .thermal_relaxation import HeatFlowComponent, ThermalCutTemplate


def _ordered_split(total: int, mask: int, width: int) -> tuple[tuple[int, ...], int]:
    receivers = [position for position in range(width) if mask & (1 << position)]
    shares = [0] * width
    remaining = total
    for offset, position in enumerate(receivers):
        count_left = len(receivers) - offset
        shares[position] = remaining // count_left
        remaining -= shares[position]
    return tuple(shares), remaining


@dataclass(frozen=True, slots=True)
class ThermalMasterSolution:
    status: str
    feasible: bool
    proven_optimal: bool
    power: int | None
    generated_heat: int | None
    layout: tuple[str, ...] | None
    cut_capacities: tuple[int, ...]
    strict_power_upper_bound: int
    elapsed_seconds: float
    conflicts: int
    branches: int
    fixed_skeleton_core: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PowerSkeletonNoGood:
    """A proven-infeasible conjunction of partial power-label assignments."""

    assignments: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        vertices = [vertex for vertex, _label in self.assignments]
        if len(vertices) != len(set(vertices)):
            raise ValueError("a power skeleton core may mention each vertex only once")


class ThermalCutMaster:
    """Exact static power model over every enabled component label.

    Each :class:`ThermalCutTemplate` contributes the globally necessary
    inequality ``generated_heat <= capacity_of_fixed_partition(layout)``.
    Capacities are reconstructed from unary label variables and exact
    Boolean/integer products, so the model is not tied to IC2 component ids.
    """

    def __init__(
        self,
        problem: ReactorProblem,
        heat_catalogue: Mapping[str, HeatFlowComponent],
        *,
        prefix_catalogue: Mapping[str, PrefixHeatComponent] | None = None,
        base_hull_capacity: int | None = None,
    ) -> None:
        labels = tuple(dict.fromkeys((
            *(item.id for item in problem.power_components),
            *problem.layout_components,
        )))
        if not labels or labels[0] != "empty":
            raise ValueError("the full catalogue must begin with the empty label")
        missing = set(labels) - heat_catalogue.keys()
        if missing:
            raise ValueError(f"heat catalogue is missing enabled labels: {sorted(missing)}")
        self.problem = problem
        self.heat_catalogue = dict(heat_catalogue)
        self.prefix_catalogue = (
            None if prefix_catalogue is None else dict(prefix_catalogue)
        )
        self.base_hull_capacity = base_hull_capacity
        self.labels = labels
        self.code_by_id = {label: code for code, label in enumerate(labels)}

    def _validate_cut(self, cut: ThermalCutTemplate) -> None:
        vertices = set(self.problem.graph.vertices)
        if not set(cut.source_storage_slots) <= vertices:
            raise ValueError("thermal cut has a storage vertex outside the graph")
        if not set(cut.source_generator_slots) <= vertices:
            raise ValueError("thermal cut has a generator vertex outside the graph")
        if len(cut.source_storage_slots) != len(set(cut.source_storage_slots)):
            raise ValueError("thermal cut repeats a storage vertex")
        if len(cut.source_generator_slots) != len(set(cut.source_generator_slots)):
            raise ValueError("thermal cut repeats a generator vertex")

    def build(
        self,
        *,
        cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        excluded_layouts: Sequence[Sequence[str]] = (),
        enforce_cuts: bool = True,
        enforce_full_flow: bool = False,
        enforce_ordered_distribution_flow: bool = False,
        enforce_periodic_prefix_flow: bool = False,
        minimum_power: int | None = None,
        exact_power: int | None = None,
        maximum_power_limit: int | None = None,
        aggregate_fuel_degree_counts: Mapping[tuple[str, int], int] | None = None,
        exact_active_cells: int | None = None,
        conditional_aggregate_patterns: Mapping[
            int, Sequence[AggregatePattern]
        ] | None = None,
        weighted_label_limits: Sequence[tuple[Mapping[str, int], int]] = (),
        fixed_power_skeleton: Sequence[str] | None = None,
        excluded_power_cores: Sequence[PowerSkeletonNoGood] = (),
        extract_fixed_skeleton_core: bool = False,
        infer_empty_skeleton_from_active_count: bool = False,
    ):
        try:
            from ortools.sat.python import cp_model
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError("ThermalCutMaster requires OR-Tools") from error

        problem = self.problem
        graph = problem.graph
        labels = self.labels
        for cut in cuts:
            self._validate_cut(cut)
        if prefix_cuts or enforce_periodic_prefix_flow:
            if self.prefix_catalogue is None or self.base_hull_capacity is None:
                raise ValueError(
                    "periodic prefix flow requires a prefix catalogue and hull capacity"
                )
            if self.base_hull_capacity <= 0:
                raise ValueError("base hull capacity must be positive")
            missing_prefix = set(labels) - self.prefix_catalogue.keys()
            if missing_prefix:
                raise ValueError(
                    f"prefix catalogue is missing enabled labels: {sorted(missing_prefix)}"
                )
        if enforce_full_flow and enforce_ordered_distribution_flow:
            raise ValueError("choose either free-source flow or ordered-distribution flow")
        model = cp_model.CpModel()
        one_hot = [
            [model.new_bool_var(f"x_{vertex}_{code}") for code in range(len(labels))]
            for vertex in graph.vertices
        ]
        for vertex in graph.vertices:
            model.add_exactly_one(one_hot[vertex])
        label_code = [
            model.new_int_var(0, len(labels) - 1, f"label_{vertex}")
            for vertex in graph.vertices
        ]
        for vertex in graph.vertices:
            model.add(label_code[vertex] == sum(
                code * one_hot[vertex][code] for code in range(len(labels))
            ))

        if extract_fixed_skeleton_core and fixed_power_skeleton is None:
            raise ValueError("fixed-skeleton core extraction requires a fixed skeleton")
        power_ids = {item.id for item in problem.power_components}
        nonempty_power_codes = tuple(
            self.code_by_id[item.id]
            for item in problem.power_components
            if item.id != "empty"
        )
        fixed_assumption_assignments: dict[int, tuple[int, str]] = {}
        if fixed_power_skeleton is not None:
            if len(fixed_power_skeleton) != graph.size:
                raise ValueError("fixed power skeleton length does not match graph")
            unknown_power = set(fixed_power_skeleton) - power_ids
            if unknown_power:
                raise ValueError(
                    f"fixed skeleton has unknown power labels: {sorted(unknown_power)}"
                )
            if infer_empty_skeleton_from_active_count:
                if not extract_fixed_skeleton_core:
                    raise ValueError(
                        "empty-skeleton inference is only defined for core extraction"
                    )
                if any(
                    item.id != "empty" and not item.accepts_pulse
                    for item in problem.power_components
                ):
                    raise ValueError(
                        "empty-skeleton inference requires every nonempty power "
                        "label to be active"
                    )
                nonempty_count = sum(label != "empty" for label in fixed_power_skeleton)
                if exact_active_cells != nonempty_count:
                    raise ValueError(
                        "empty-skeleton inference requires the exact active count "
                        "to equal the fixed nonempty count"
                    )
            for vertex, label in enumerate(fixed_power_skeleton):
                if label == "empty" and infer_empty_skeleton_from_active_count:
                    continue
                if label == "empty":
                    constraint = model.add(sum(
                        one_hot[vertex][code] for code in nonempty_power_codes
                    ) == 0)
                else:
                    constraint = model.add(
                        one_hot[vertex][self.code_by_id[label]] == 1
                    )
                if extract_fixed_skeleton_core:
                    assumption = model.new_bool_var(
                        f"fixed_skeleton_assumption_{vertex}"
                    )
                    constraint.only_enforce_if(assumption)
                    model.add_assumption(assumption)
                    fixed_assumption_assignments[assumption.index] = (vertex, label)

        for core_index, core in enumerate(excluded_power_cores):
            if not core.assignments:
                model.add(0 == 1)
                continue
            matches = []
            for vertex, label in core.assignments:
                if vertex not in graph.vertices:
                    raise ValueError("power skeleton core has a vertex outside the graph")
                if label not in power_ids:
                    raise ValueError(f"power skeleton core has unknown label: {label}")
                if label == "empty":
                    matches.append(
                        1 - sum(
                            one_hot[vertex][code]
                            for code in nonempty_power_codes
                        )
                    )
                else:
                    matches.append(one_hot[vertex][self.code_by_id[label]])
            model.add(sum(matches) <= len(matches) - 1)

        for layout in excluded_layouts:
            if len(layout) != graph.size:
                raise ValueError("excluded layout length does not match graph")
            unknown = set(layout) - self.code_by_id.keys()
            if unknown:
                raise ValueError(f"excluded layout has unknown labels: {sorted(unknown)}")
            model.add(sum(
                one_hot[vertex][self.code_by_id[label]]
                for vertex, label in enumerate(layout)
            ) <= graph.size - 1)

        active = [model.new_bool_var(f"active_{vertex}") for vertex in graph.vertices]
        accepts_heat = [
            model.new_bool_var(f"accepts_heat_{vertex}") for vertex in graph.vertices
        ]
        degrees = [
            model.new_int_var(0, len(graph.neighbours[vertex]), f"degree_{vertex}")
            for vertex in graph.vertices
        ]

        def unary(vertex: int, attribute: str, *, accepting_only: bool = False):
            return sum(
                (
                    getattr(self.heat_catalogue[label], attribute)
                    if not accepting_only or self.heat_catalogue[label].accepts_heat
                    else 0
                )
                * one_hot[vertex][code]
                for code, label in enumerate(labels)
            )

        unary_variable_cache = {}

        def unary_variable(vertex: int, attribute: str, upper: int):
            key = (vertex, attribute)
            variable = unary_variable_cache.get(key)
            if variable is None:
                variable = model.new_int_var(0, upper, f"{attribute}_{vertex}")
                model.add(variable == unary(vertex, attribute))
                unary_variable_cache[key] = variable
            return variable

        for vertex in graph.vertices:
            model.add(active[vertex] == sum(
                one_hot[vertex][self.code_by_id[item.id]]
                for item in problem.power_components
                if item.accepts_pulse
            ))
            model.add(accepts_heat[vertex] == sum(
                one_hot[vertex][code]
                for code, label in enumerate(labels)
                if self.heat_catalogue[label].accepts_heat
            ))
            model.add(degrees[vertex] == sum(
                active[other] for other in graph.neighbours[vertex]
            ))

        maximum_internal = max(
            item.internal_pulses for item in problem.power_components if item.rods > 0
        )
        max_vertex_heat = max(
            problem.heat_scale
            * item.rods
            * (item.internal_pulses + graph.maximum_degree)
            * (item.internal_pulses + graph.maximum_degree + 1)
            for item in problem.power_components
            if item.rods > 0
        )
        maximum_pulse_units = problem.rod_budget * (
            maximum_internal + graph.maximum_degree
        )
        maximum_power = problem.eu_per_pulse * maximum_pulse_units
        maximum_heat = problem.heat_scale * problem.rod_budget * (
            maximum_internal + graph.maximum_degree
        ) * (maximum_internal + graph.maximum_degree + 1)

        vertex_heat = [
            model.new_int_var(0, max_vertex_heat, f"vertex_heat_{vertex}")
            for vertex in graph.vertices
        ]
        vertex_power = [
            model.new_int_var(0, maximum_power, f"vertex_power_{vertex}")
            for vertex in graph.vertices
        ]
        fuel_degree_states: dict[tuple[str, int], list] = {
            (item.id, degree): []
            for item in problem.power_components
            if item.rods > 0
            for degree in range(graph.maximum_degree + 1)
        }
        for vertex in graph.vertices:
            heat_terms = []
            power_terms = []
            for item in problem.power_components:
                if item.rods <= 0:
                    continue
                code = self.code_by_id[item.id]
                degree_states = []
                for degree in range(len(graph.neighbours[vertex]) + 1):
                    state = model.new_bool_var(f"z_{vertex}_{code}_{degree}")
                    model.add(degrees[vertex] == degree).only_enforce_if(state)
                    degree_states.append(state)
                    fuel_degree_states[item.id, degree].append(state)
                    pulses = item.internal_pulses + degree
                    power_terms.append(
                        problem.eu_per_pulse * item.rods * pulses * state
                    )
                    heat_terms.append(
                        problem.heat_scale
                        * item.rods
                        * pulses
                        * (pulses + 1)
                        * state
                    )
                model.add(sum(degree_states) == one_hot[vertex][code])
            model.add(vertex_power[vertex] == sum(power_terms))
            model.add(vertex_heat[vertex] == sum(heat_terms))

        if aggregate_fuel_degree_counts is not None:
            unknown_counts = set(aggregate_fuel_degree_counts) - set(fuel_degree_states)
            if unknown_counts:
                raise ValueError(
                    f"unknown fuel-degree count keys: {sorted(unknown_counts)}"
                )
            for key, states in fuel_degree_states.items():
                model.add(sum(states) == int(aggregate_fuel_degree_counts.get(key, 0)))
        if exact_active_cells is not None:
            if not 0 <= exact_active_cells <= graph.size:
                raise ValueError("exact_active_cells is outside the graph")
            model.add(sum(active) == exact_active_cells)

        rods = sum(
            item.rods * one_hot[vertex][self.code_by_id[item.id]]
            for vertex in graph.vertices
            for item in problem.power_components
            if item.rods > 0
        )
        if problem.exact_rods:
            model.add(rods == problem.rod_budget)
        else:
            model.add(rods >= 1)
            model.add(rods <= problem.rod_budget)

        limits = dict(problem.component_limits)
        for label, limit in limits.items():
            if limit is not None:
                code = self.code_by_id[label]
                model.add(sum(one_hot[v][code] for v in graph.vertices) <= limit)
        for weights, upper_bound in weighted_label_limits:
            unknown_labels = set(weights) - self.code_by_id.keys()
            if unknown_labels:
                raise ValueError(
                    f"weighted label limit has unknown labels: {sorted(unknown_labels)}"
                )
            if upper_bound < 0 or any(weight < 0 for weight in weights.values()):
                raise ValueError("weighted label limits must be non-negative")
            model.add(sum(
                int(weight) * one_hot[vertex][self.code_by_id[label]]
                for label, weight in weights.items()
                for vertex in graph.vertices
            ) <= int(upper_bound))

        power = model.new_int_var(0, maximum_power, "power")
        heat = model.new_int_var(0, maximum_heat, "generated_heat")
        model.add(power == sum(vertex_power))
        model.add(heat == sum(vertex_heat))
        if minimum_power is not None:
            model.add(power >= minimum_power)
        if exact_power is not None:
            model.add(power == exact_power)
        if maximum_power_limit is not None:
            model.add(power <= maximum_power_limit)
        if conditional_aggregate_patterns:
            for tier, patterns in conditional_aggregate_patterns.items():
                tier = int(tier)
                if tier < 0 or tier > maximum_power:
                    raise ValueError("conditional aggregate tier is outside power range")
                at_tier = model.new_bool_var(f"conditional_tier_{tier}")
                model.add(power == tier).only_enforce_if(at_tier)
                model.add(power != tier).only_enforce_if(at_tier.negated())
                selectors = []
                for pattern_index, pattern in enumerate(patterns):
                    requested = {
                        (item, degree): count
                        for item, degree, count in pattern.fuel_degree_counts
                    }
                    if len(requested) != len(pattern.fuel_degree_counts):
                        raise ValueError("conditional pattern repeats a fuel-degree state")
                    if unknown := set(requested) - set(fuel_degree_states):
                        raise ValueError(
                            "conditional pattern has unknown fuel-degree states: "
                            f"{sorted(unknown)}"
                        )
                    if not 0 <= pattern.active_cells <= graph.size:
                        raise ValueError("conditional pattern active count is invalid")
                    selector = model.new_bool_var(
                        f"conditional_tier_{tier}_pattern_{pattern_index}"
                    )
                    selectors.append(selector)
                    for key, states in fuel_degree_states.items():
                        model.add(
                            sum(states) == int(requested.get(key, 0))
                        ).only_enforce_if(selector)
                    model.add(
                        sum(active) == pattern.active_cells
                    ).only_enforce_if(selector)
                model.add(sum(selectors) == at_tier)

        max_self = max(
            (spec.self_vent for label, spec in self.heat_catalogue.items() if label in self.code_by_id),
            default=0,
        )
        max_side = max(
            (spec.side_vent for label, spec in self.heat_catalogue.items() if label in self.code_by_id),
            default=0,
        )
        max_hull_draw = max(
            (spec.hull_draw for label, spec in self.heat_catalogue.items() if label in self.code_by_id),
            default=0,
        )
        max_exchange_side = max(
            (spec.exchange_side for label, spec in self.heat_catalogue.items() if label in self.code_by_id),
            default=0,
        )
        max_exchange_hull = max(
            (spec.exchange_hull for label, spec in self.heat_catalogue.items() if label in self.code_by_id),
            default=0,
        )
        degree = graph.maximum_degree
        coarse_cut_capacity = graph.size * (
            (degree + 2) * max_vertex_heat
            + max_self
            + degree * max_side
            + max_hull_draw
            + degree * max_exchange_side
            + max_exchange_hull
        )
        cut_capacities = []

        def product_int_bool(name: str, integer, boolean, upper: int):
            product = model.new_int_var(0, upper, name)
            model.add_multiplication_equality(product, [integer, boolean])
            return product

        flow_variables = []
        if enforce_full_flow:
            source_to_generator = []
            generator_to_hull = []
            generator_to_storage = {}
            storage_to_sink = []
            hull_to_storage = []
            storage_to_hull = []
            storage_exchange = {}

            for vertex in graph.vertices:
                source_flow = model.new_int_var(
                    0,
                    max_vertex_heat,
                    f"flow_source_generator_{vertex}",
                )
                model.add(source_flow <= vertex_heat[vertex])
                source_to_generator.append(source_flow)
                flow_variables.append(source_flow)

                hull_flow = model.new_int_var(
                    0,
                    max_vertex_heat,
                    f"flow_generator_hull_{vertex}",
                )
                model.add(hull_flow <= vertex_heat[vertex])
                generator_to_hull.append(hull_flow)
                flow_variables.append(hull_flow)

                outgoing_generator = [hull_flow]
                for neighbour in graph.neighbours[vertex]:
                    capacity = product_int_bool(
                        f"flow_cap_generator_{vertex}_storage_{neighbour}",
                        vertex_heat[vertex],
                        accepts_heat[neighbour],
                        max_vertex_heat,
                    )
                    edge_flow = model.new_int_var(
                        0,
                        max_vertex_heat,
                        f"flow_generator_{vertex}_storage_{neighbour}",
                    )
                    model.add(edge_flow <= capacity)
                    generator_to_storage[vertex, neighbour] = edge_flow
                    outgoing_generator.append(edge_flow)
                    flow_variables.append(edge_flow)
                model.add(source_flow == sum(outgoing_generator))

            exchange_accept_capacity = {}
            for vertex in graph.vertices:
                for neighbour in graph.neighbours[vertex]:
                    exchange_accept_capacity[vertex, neighbour] = product_int_bool(
                        f"flow_cap_exchange_{vertex}_accept_{neighbour}",
                        unary_variable(vertex, "exchange_side", max_exchange_side),
                        accepts_heat[neighbour],
                        max_exchange_side,
                    )

            for vertex in graph.vertices:
                external_terms = [unary(vertex, "self_vent", accepting_only=True)]
                for neighbour in graph.neighbours[vertex]:
                    external_terms.append(product_int_bool(
                        f"flow_cap_sidevent_{vertex}_{neighbour}",
                        unary_variable(neighbour, "side_vent", max_side),
                        accepts_heat[vertex],
                        max_side,
                    ))
                sink_flow = model.new_int_var(
                    0,
                    max_self + degree * max_side,
                    f"flow_storage_{vertex}_sink",
                )
                model.add(sink_flow <= sum(external_terms))
                storage_to_sink.append(sink_flow)
                flow_variables.append(sink_flow)

                hull_in = model.new_int_var(
                    0,
                    max_hull_draw + max_exchange_hull,
                    f"flow_hull_storage_{vertex}",
                )
                model.add(hull_in <= (
                    unary(vertex, "hull_draw", accepting_only=True)
                    + unary(vertex, "exchange_hull")
                ))
                hull_to_storage.append(hull_in)
                flow_variables.append(hull_in)

                hull_out = model.new_int_var(
                    0,
                    max_exchange_hull,
                    f"flow_storage_{vertex}_hull",
                )
                model.add(hull_out <= unary(vertex, "exchange_hull"))
                storage_to_hull.append(hull_out)
                flow_variables.append(hull_out)

                for neighbour in graph.neighbours[vertex]:
                    edge_flow = model.new_int_var(
                        0,
                        2 * max_exchange_side,
                        f"flow_storage_{vertex}_{neighbour}",
                    )
                    model.add(edge_flow <= (
                        exchange_accept_capacity[vertex, neighbour]
                        + exchange_accept_capacity[neighbour, vertex]
                    ))
                    storage_exchange[vertex, neighbour] = edge_flow
                    flow_variables.append(edge_flow)

            for vertex in graph.vertices:
                incoming = [hull_to_storage[vertex]]
                outgoing = [storage_to_sink[vertex], storage_to_hull[vertex]]
                for neighbour in graph.neighbours[vertex]:
                    incoming.append(generator_to_storage[neighbour, vertex])
                    incoming.append(storage_exchange[neighbour, vertex])
                    outgoing.append(storage_exchange[vertex, neighbour])
                model.add(sum(incoming) == sum(outgoing))

            model.add(
                sum(generator_to_hull) + sum(storage_to_hull)
                == sum(hull_to_storage)
            )
            model.add(sum(source_to_generator) == heat)
            model.add(sum(storage_to_sink) == heat)

        if enforce_ordered_distribution_flow:
            mandatory_accepts = []
            optional_accepts = []
            for vertex in graph.vertices:
                mandatory = model.new_bool_var(f"ordered_mandatory_accept_{vertex}")
                optional = model.new_bool_var(f"ordered_optional_accept_{vertex}")
                model.add(mandatory == sum(
                    one_hot[vertex][code]
                    for code, label in enumerate(labels)
                    if (
                        self.heat_catalogue[label].accepts_heat
                        and not self.heat_catalogue[label].optional_heat_acceptance
                    )
                ))
                model.add(optional == sum(
                    one_hot[vertex][code]
                    for code, label in enumerate(labels)
                    if self.heat_catalogue[label].optional_heat_acceptance
                ))
                mandatory_accepts.append(mandatory)
                optional_accepts.append(optional)

            direct_to_storage = {}
            direct_to_hull = []
            power_spec_by_id = {
                item.id: item for item in problem.power_components
            }
            for source in graph.vertices:
                neighbours = graph.neighbours[source]
                edge_accepts = []
                for position, target in enumerate(neighbours):
                    accepted = model.new_bool_var(
                        f"ordered_accept_{source}_{target}"
                    )
                    model.add(accepted >= mandatory_accepts[target])
                    model.add(
                        accepted <= mandatory_accepts[target] + optional_accepts[target]
                    )
                    edge_accepts.append(accepted)
                mask = model.new_int_var(
                    0,
                    (1 << len(neighbours)) - 1,
                    f"ordered_mask_{source}",
                )
                model.add(mask == sum(
                    (1 << position) * accepted
                    for position, accepted in enumerate(edge_accepts)
                ))
                outputs = []
                for target in neighbours:
                    output = model.new_int_var(
                        0,
                        max_vertex_heat,
                        f"ordered_direct_{source}_{target}",
                    )
                    direct_to_storage[source, target] = output
                    outputs.append(output)
                    flow_variables.append(output)
                hull_output = model.new_int_var(
                    0,
                    max_vertex_heat,
                    f"ordered_direct_{source}_hull",
                )
                direct_to_hull.append(hull_output)
                flow_variables.append(hull_output)

                rows = []
                for code, label in enumerate(labels):
                    power_spec = power_spec_by_id.get(label)
                    for adjacent_degree in range(len(neighbours) + 1):
                        for mask_value in range(1 << len(neighbours)):
                            if power_spec is None or power_spec.rods <= 0:
                                shares = (0,) * len(neighbours)
                                hull_share = 0
                            else:
                                pulses = power_spec.internal_pulses + adjacent_degree
                                per_rod_heat = (
                                    problem.heat_scale * pulses * (pulses + 1)
                                )
                                per_rod_shares, per_rod_hull = _ordered_split(
                                    per_rod_heat,
                                    mask_value,
                                    len(neighbours),
                                )
                                shares = tuple(
                                    power_spec.rods * value for value in per_rod_shares
                                )
                                hull_share = power_spec.rods * per_rod_hull
                            rows.append((
                                code,
                                adjacent_degree,
                                mask_value,
                                *shares,
                                hull_share,
                            ))
                model.add_allowed_assignments(
                    [label_code[source], degrees[source], mask, *outputs, hull_output],
                    rows,
                )

            exchange_accept_capacity = {}
            for vertex in graph.vertices:
                for neighbour in graph.neighbours[vertex]:
                    exchange_accept_capacity[vertex, neighbour] = product_int_bool(
                        f"ordered_cap_exchange_{vertex}_accept_{neighbour}",
                        unary_variable(vertex, "exchange_side", max_exchange_side),
                        accepts_heat[neighbour],
                        max_exchange_side,
                    )

            storage_to_sink = []
            hull_to_storage = []
            storage_to_hull = []
            storage_exchange = {}
            for vertex in graph.vertices:
                external_terms = [unary(vertex, "self_vent", accepting_only=True)]
                for neighbour in graph.neighbours[vertex]:
                    external_terms.append(product_int_bool(
                        f"ordered_cap_sidevent_{vertex}_{neighbour}",
                        unary_variable(neighbour, "side_vent", max_side),
                        accepts_heat[vertex],
                        max_side,
                    ))
                sink_flow = model.new_int_var(
                    0,
                    max_self + degree * max_side,
                    f"ordered_flow_storage_{vertex}_sink",
                )
                model.add(sink_flow <= sum(external_terms))
                storage_to_sink.append(sink_flow)
                flow_variables.append(sink_flow)

                hull_in = model.new_int_var(
                    0,
                    max_hull_draw + max_exchange_hull,
                    f"ordered_flow_hull_storage_{vertex}",
                )
                model.add(hull_in <= (
                    unary(vertex, "hull_draw", accepting_only=True)
                    + unary(vertex, "exchange_hull")
                ))
                hull_to_storage.append(hull_in)
                flow_variables.append(hull_in)

                hull_out = model.new_int_var(
                    0,
                    max_exchange_hull,
                    f"ordered_flow_storage_{vertex}_hull",
                )
                model.add(hull_out <= unary(vertex, "exchange_hull"))
                storage_to_hull.append(hull_out)
                flow_variables.append(hull_out)

                for neighbour in graph.neighbours[vertex]:
                    edge_flow = model.new_int_var(
                        0,
                        2 * max_exchange_side,
                        f"ordered_flow_storage_{vertex}_{neighbour}",
                    )
                    model.add(edge_flow <= (
                        exchange_accept_capacity[vertex, neighbour]
                        + exchange_accept_capacity[neighbour, vertex]
                    ))
                    storage_exchange[vertex, neighbour] = edge_flow
                    flow_variables.append(edge_flow)

            for vertex in graph.vertices:
                incoming = [hull_to_storage[vertex]]
                outgoing = [storage_to_sink[vertex], storage_to_hull[vertex]]
                for neighbour in graph.neighbours[vertex]:
                    incoming.append(direct_to_storage[neighbour, vertex])
                    incoming.append(storage_exchange[neighbour, vertex])
                    outgoing.append(storage_exchange[vertex, neighbour])
                model.add(sum(incoming) == sum(outgoing))

            model.add(
                sum(direct_to_hull) + sum(storage_to_hull)
                == sum(hull_to_storage)
            )
            model.add(sum(storage_to_sink) == heat)

        prefix_cut_violations = []
        if prefix_cuts or enforce_periodic_prefix_flow:
            prefix_catalogue = self.prefix_catalogue
            assert prefix_catalogue is not None
            assert self.base_hull_capacity is not None

            def prefix_unary(vertex: int, attribute: str):
                return sum(
                    getattr(prefix_catalogue[label], attribute)
                    * one_hot[vertex][code]
                    for code, label in enumerate(labels)
                )

            prefix_mandatory_accepts = [
                model.new_bool_var(f"prefix_mandatory_accepts_{vertex}")
                for vertex in graph.vertices
            ]
            prefix_optional_accepts = [
                model.new_bool_var(f"prefix_optional_accepts_{vertex}")
                for vertex in graph.vertices
            ]
            for vertex in graph.vertices:
                model.add(prefix_mandatory_accepts[vertex] == sum(
                    one_hot[vertex][code]
                    for code, label in enumerate(labels)
                    if (
                        prefix_catalogue[label].accepts_fuel_heat
                        and not prefix_catalogue[label].optional_fuel_acceptance
                    )
                ))
                model.add(prefix_optional_accepts[vertex] == sum(
                    one_hot[vertex][code]
                    for code, label in enumerate(labels)
                    if prefix_catalogue[label].optional_fuel_acceptance
                ))

            prefix_direct_to_storage = {}
            prefix_direct_to_hull = []
            power_spec_by_id = {
                item.id: item for item in problem.power_components
            }
            for source in graph.vertices:
                neighbours = graph.neighbours[source]
                edge_accepts = []
                for target in neighbours:
                    accepted = model.new_bool_var(
                        f"prefix_accept_{source}_{target}"
                    )
                    model.add(accepted >= prefix_mandatory_accepts[target])
                    model.add(
                        accepted
                        <= prefix_mandatory_accepts[target]
                        + prefix_optional_accepts[target]
                    )
                    edge_accepts.append(accepted)
                mask = model.new_int_var(
                    0,
                    (1 << len(neighbours)) - 1,
                    f"prefix_mask_{source}",
                )
                model.add(mask == sum(
                    (1 << position) * accepted
                    for position, accepted in enumerate(edge_accepts)
                ))
                outputs = []
                for target in neighbours:
                    output = model.new_int_var(
                        0,
                        max_vertex_heat,
                        f"prefix_direct_{source}_{target}",
                    )
                    prefix_direct_to_storage[source, target] = output
                    outputs.append(output)
                hull_output = model.new_int_var(
                    0,
                    max_vertex_heat,
                    f"prefix_direct_{source}_hull",
                )
                prefix_direct_to_hull.append(hull_output)

                rows = []
                for code, label in enumerate(labels):
                    power_spec = power_spec_by_id.get(label)
                    for adjacent_degree in range(len(neighbours) + 1):
                        for mask_value in range(1 << len(neighbours)):
                            if power_spec is None or power_spec.rods <= 0:
                                shares = (0,) * len(neighbours)
                                hull_share = 0
                            else:
                                pulses = power_spec.internal_pulses + adjacent_degree
                                per_rod_heat = (
                                    problem.heat_scale * pulses * (pulses + 1)
                                )
                                per_rod_shares, per_rod_hull = _ordered_split(
                                    per_rod_heat,
                                    mask_value,
                                    len(neighbours),
                                )
                                shares = tuple(
                                    power_spec.rods * value
                                    for value in per_rod_shares
                                )
                                hull_share = power_spec.rods * per_rod_hull
                            rows.append((
                                code,
                                adjacent_degree,
                                mask_value,
                                *shares,
                                hull_share,
                            ))
                model.add_allowed_assignments(
                    [label_code[source], degrees[source], mask, *outputs, hull_output],
                    rows,
                )

            event_by_vertex = {
                vertex: event for event, vertex in enumerate(graph.update_order)
            }
            event_count = graph.size

            def reservoir_label(reservoir: int) -> str:
                return "hull" if reservoir == -1 else f"slot_{reservoir}"

            def before_name(event: int, reservoir: int) -> str:
                return f"e{event}:{reservoir_label(reservoir)}:in"

            def after_name(event: int, reservoir: int) -> str:
                return f"e{event}:{reservoir_label(reservoir)}:out"

            def pool_name(event: int, vertex: int, suffix: str) -> str:
                return f"e{event}:slot_{vertex}:action_pool_{suffix}"

            canonical_names = {"environment"}
            for event in range(event_count):
                for reservoir in (-1, *graph.vertices):
                    canonical_names.add(before_name(event, reservoir))
                    canonical_names.add(after_name(event, reservoir))
                vertex = graph.update_order[event]
                canonical_names.add(pool_name(event, vertex, "in"))
                canonical_names.add(pool_name(event, vertex, "out"))

            hull_capacity = (
                self.base_hull_capacity
                + sum(
                    prefix_unary(vertex, "hull_capacity_bonus")
                    for vertex in graph.vertices
                )
            )

            if enforce_periodic_prefix_flow:
                # Embed the complete lower-bound circulation directly.  This
                # is equivalent to imposing every Hoffman inequality at once
                # and avoids repeatedly adding enormous cut templates.
                max_hull_capacity = self.base_hull_capacity + graph.size * max(
                    (
                        spec.hull_capacity_bonus
                        for spec in prefix_catalogue.values()
                    ),
                    default=0,
                )
                max_prefix_arc = max(
                    max_hull_capacity,
                    max_vertex_heat,
                    *(
                        max(
                            spec.heat_capacity,
                            spec.self_vent,
                            spec.side_vent,
                            spec.hull_draw,
                            spec.exchange_side,
                            spec.exchange_hull,
                        )
                        for spec in prefix_catalogue.values()
                    ),
                )
                incoming = {name: [] for name in canonical_names}
                outgoing = {name: [] for name in canonical_names}
                prefix_circulation_flows = []

                def circulation_arc(
                    start: str,
                    end: str,
                    capacity,
                    name: str,
                ) -> None:
                    arc = model.new_int_var(0, max_prefix_arc, name)
                    model.add(arc <= capacity)
                    outgoing[start].append(arc)
                    incoming[end].append(arc)
                    prefix_circulation_flows.append(arc)

                def fixed_circulation_arc(start: str, end: str, amount) -> None:
                    outgoing[start].append(amount)
                    incoming[end].append(amount)

                for event, vertex in enumerate(graph.update_order):
                    next_event = (event + 1) % event_count
                    for reservoir in (-1, *graph.vertices):
                        cap = (
                            hull_capacity
                            if reservoir == -1
                            else prefix_unary(reservoir, "heat_capacity")
                        )
                        circulation_arc(
                            before_name(event, reservoir),
                            after_name(event, reservoir),
                            cap,
                            f"prefix_flow_gate_{event}_{reservoir}",
                        )
                        if reservoir != vertex:
                            circulation_arc(
                                after_name(event, reservoir),
                                before_name(next_event, reservoir),
                                cap,
                                f"prefix_flow_carry_{event}_{reservoir}",
                            )

                    vertex_capacity = prefix_unary(vertex, "heat_capacity")
                    middle_in = pool_name(event, vertex, "in")
                    middle_out = pool_name(event, vertex, "out")
                    circulation_arc(
                        after_name(event, vertex),
                        middle_in,
                        vertex_capacity,
                        f"prefix_flow_action_in_{event}_{vertex}",
                    )
                    circulation_arc(
                        after_name(event, -1),
                        middle_in,
                        prefix_unary(vertex, "hull_draw"),
                        f"prefix_flow_hull_draw_{event}_{vertex}",
                    )
                    circulation_arc(
                        middle_in,
                        middle_out,
                        vertex_capacity,
                        f"prefix_flow_action_gate_{event}_{vertex}",
                    )
                    circulation_arc(
                        middle_out,
                        before_name(next_event, vertex),
                        vertex_capacity,
                        f"prefix_flow_action_carry_{event}_{vertex}",
                    )
                    circulation_arc(
                        middle_out,
                        "environment",
                        prefix_unary(vertex, "self_vent"),
                        f"prefix_flow_self_vent_{event}_{vertex}",
                    )

                    for neighbour in graph.neighbours[vertex]:
                        circulation_arc(
                            after_name(event, neighbour),
                            "environment",
                            prefix_unary(vertex, "side_vent"),
                            f"prefix_flow_side_vent_{event}_{vertex}_{neighbour}",
                        )
                        circulation_arc(
                            after_name(event, vertex),
                            before_name(next_event, neighbour),
                            prefix_unary(vertex, "exchange_side"),
                            f"prefix_flow_exchange_out_{event}_{vertex}_{neighbour}",
                        )
                        circulation_arc(
                            after_name(event, neighbour),
                            before_name(next_event, vertex),
                            prefix_unary(vertex, "exchange_side"),
                            f"prefix_flow_exchange_in_{event}_{vertex}_{neighbour}",
                        )
                    circulation_arc(
                        after_name(event, vertex),
                        before_name(next_event, -1),
                        prefix_unary(vertex, "exchange_hull"),
                        f"prefix_flow_to_hull_{event}_{vertex}",
                    )
                    circulation_arc(
                        after_name(event, -1),
                        before_name(next_event, vertex),
                        prefix_unary(vertex, "exchange_hull"),
                        f"prefix_flow_from_hull_{event}_{vertex}",
                    )

                    for target in graph.neighbours[vertex]:
                        fixed_circulation_arc(
                            "environment",
                            before_name(next_event, target),
                            prefix_direct_to_storage[vertex, target],
                        )
                    fixed_circulation_arc(
                        "environment",
                        before_name(next_event, -1),
                        prefix_direct_to_hull[vertex],
                    )

                for name in canonical_names:
                    model.add(sum(incoming[name]) == sum(outgoing[name]))
                flow_variables.extend(prefix_circulation_flows)

            for cut_index, cut in enumerate(prefix_cuts):
                source_side = set(cut.source_side_nodes)
                if unknown_nodes := source_side - canonical_names:
                    raise ValueError(
                        "periodic prefix cut has unknown canonical nodes: "
                        f"{sorted(unknown_nodes)}"
                    )
                terms = []

                def upper_edge(start: str, end: str, edge_capacity) -> None:
                    if start in source_side and end not in source_side:
                        terms.append(-edge_capacity)

                def fixed_edge(start: str, end: str, amount) -> None:
                    if start not in source_side and end in source_side:
                        terms.append(amount)
                    elif start in source_side and end not in source_side:
                        terms.append(-amount)

                for event, vertex in enumerate(graph.update_order):
                    next_event = (event + 1) % event_count
                    for reservoir in (-1, *graph.vertices):
                        cap = (
                            hull_capacity
                            if reservoir == -1
                            else prefix_unary(reservoir, "heat_capacity")
                        )
                        upper_edge(
                            before_name(event, reservoir),
                            after_name(event, reservoir),
                            cap,
                        )
                        if reservoir != vertex:
                            upper_edge(
                                after_name(event, reservoir),
                                before_name(next_event, reservoir),
                                cap,
                            )

                    vertex_capacity = prefix_unary(vertex, "heat_capacity")
                    middle_in = pool_name(event, vertex, "in")
                    middle_out = pool_name(event, vertex, "out")
                    upper_edge(
                        after_name(event, vertex),
                        middle_in,
                        vertex_capacity,
                    )
                    upper_edge(
                        after_name(event, -1),
                        middle_in,
                        prefix_unary(vertex, "hull_draw"),
                    )
                    upper_edge(middle_in, middle_out, vertex_capacity)
                    upper_edge(
                        middle_out,
                        before_name(next_event, vertex),
                        vertex_capacity,
                    )
                    upper_edge(
                        middle_out,
                        "environment",
                        prefix_unary(vertex, "self_vent"),
                    )

                    for neighbour in graph.neighbours[vertex]:
                        upper_edge(
                            after_name(event, neighbour),
                            "environment",
                            prefix_unary(vertex, "side_vent"),
                        )
                        upper_edge(
                            after_name(event, vertex),
                            before_name(next_event, neighbour),
                            prefix_unary(vertex, "exchange_side"),
                        )
                        upper_edge(
                            after_name(event, neighbour),
                            before_name(next_event, vertex),
                            prefix_unary(vertex, "exchange_side"),
                        )
                    upper_edge(
                        after_name(event, vertex),
                        before_name(next_event, -1),
                        prefix_unary(vertex, "exchange_hull"),
                    )
                    upper_edge(
                        after_name(event, -1),
                        before_name(next_event, vertex),
                        prefix_unary(vertex, "exchange_hull"),
                    )

                    for target in graph.neighbours[vertex]:
                        fixed_edge(
                            "environment",
                            before_name(next_event, target),
                            prefix_direct_to_storage[vertex, target],
                        )
                    fixed_edge(
                        "environment",
                        before_name(next_event, -1),
                        prefix_direct_to_hull[vertex],
                    )

                violation = model.new_int_var(
                    -10**12,
                    10**12,
                    f"prefix_cut_violation_{cut_index}",
                )
                model.add(violation == sum(terms))
                model.add(violation <= 0)
                prefix_cut_violations.append(violation)

        for cut_index, cut in enumerate(cuts):
            storage_source = set(cut.source_storage_slots)
            generator_source = set(cut.source_generator_slots)
            terms = []

            # source -> generator and generator -> hull/storage arcs
            for vertex in graph.vertices:
                if vertex not in generator_source:
                    terms.append(vertex_heat[vertex])
                else:
                    if not cut.hull_source_side:
                        terms.append(vertex_heat[vertex])
                    for neighbour in graph.neighbours[vertex]:
                        if neighbour not in storage_source:
                            terms.append(product_int_bool(
                                f"cut_{cut_index}_fuel_{vertex}_store_{neighbour}",
                                vertex_heat[vertex],
                                accepts_heat[neighbour],
                                max_vertex_heat,
                            ))

            # storage -> environment, hull -> storage, exchange arcs
            for vertex in graph.vertices:
                if vertex in storage_source:
                    terms.append(unary(vertex, "self_vent", accepting_only=True))
                    for neighbour in graph.neighbours[vertex]:
                        terms.append(product_int_bool(
                            f"cut_{cut_index}_sidevent_{vertex}_{neighbour}",
                            unary_variable(neighbour, "side_vent", max_side),
                            accepts_heat[vertex],
                            max_side,
                        ))
                elif cut.hull_source_side:
                    terms.append(unary(vertex, "hull_draw", accepting_only=True))

                if cut.hull_source_side != (vertex in storage_source):
                    terms.append(unary(vertex, "exchange_hull"))

                for neighbour in graph.neighbours[vertex]:
                    if (vertex in storage_source) == (neighbour in storage_source):
                        continue
                    terms.append(product_int_bool(
                        f"cut_{cut_index}_exchange_{vertex}_{neighbour}",
                        unary_variable(vertex, "exchange_side", max_exchange_side),
                        accepts_heat[neighbour],
                        max_exchange_side,
                    ))

            capacity = model.new_int_var(
                0,
                max(0, coarse_cut_capacity),
                f"cut_capacity_{cut_index}",
            )
            model.add(capacity == sum(terms))
            if enforce_cuts:
                model.add(heat <= capacity)
            cut_capacities.append(capacity)

        model.maximize(power)
        return model, {
            "one_hot": one_hot,
            "label_code": label_code,
            "power": power,
            "model_power_upper": maximum_power,
            "heat": heat,
            "cut_capacities": cut_capacities,
            "flow_variables": flow_variables,
            "prefix_cut_violations": prefix_cut_violations,
            "fixed_assumption_assignments": fixed_assumption_assignments,
        }

    def solve(
        self,
        *,
        cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        excluded_layouts: Sequence[Sequence[str]] = (),
        enforce_full_flow: bool = False,
        enforce_ordered_distribution_flow: bool = False,
        enforce_periodic_prefix_flow: bool = False,
        seconds: float = 60.0,
        workers: int = 1,
        random_seed: int = 221,
        minimum_power: int | None = None,
        exact_power: int | None = None,
        maximum_power_limit: int | None = None,
        aggregate_fuel_degree_counts: Mapping[tuple[str, int], int] | None = None,
        exact_active_cells: int | None = None,
        conditional_aggregate_patterns: Mapping[
            int, Sequence[AggregatePattern]
        ] | None = None,
        weighted_label_limits: Sequence[tuple[Mapping[str, int], int]] = (),
        fixed_power_skeleton: Sequence[str] | None = None,
        excluded_power_cores: Sequence[PowerSkeletonNoGood] = (),
        extract_fixed_skeleton_core: bool = False,
        infer_empty_skeleton_from_active_count: bool = False,
    ) -> ThermalMasterSolution:
        from ortools.sat.python import cp_model

        if seconds <= 0:
            raise ValueError("seconds must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        model, variables = self.build(
            cuts=cuts,
            prefix_cuts=prefix_cuts,
            excluded_layouts=excluded_layouts,
            enforce_full_flow=enforce_full_flow,
            enforce_ordered_distribution_flow=enforce_ordered_distribution_flow,
            enforce_periodic_prefix_flow=enforce_periodic_prefix_flow,
            minimum_power=minimum_power,
            exact_power=exact_power,
            maximum_power_limit=maximum_power_limit,
            aggregate_fuel_degree_counts=aggregate_fuel_degree_counts,
            exact_active_cells=exact_active_cells,
            conditional_aggregate_patterns=conditional_aggregate_patterns,
            weighted_label_limits=weighted_label_limits,
            fixed_power_skeleton=fixed_power_skeleton,
            excluded_power_cores=excluded_power_cores,
            extract_fixed_skeleton_core=extract_fixed_skeleton_core,
            infer_empty_skeleton_from_active_count=(
                infer_empty_skeleton_from_active_count
            ),
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = random_seed
        started = perf_counter()
        status_code = solver.solve(model)
        elapsed = perf_counter() - started
        status = solver.status_name(status_code)
        if status_code == cp_model.MODEL_INVALID:
            raise RuntimeError(f"invalid thermal master: {model.validate()}")
        feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        # CP-SAT's INFEASIBLE status is also a complete proof result for this
        # finite master domain; ``proven_optimal`` means the answer is final,
        # not that a primal optimum necessarily exists.
        proven = status_code in (cp_model.OPTIMAL, cp_model.INFEASIBLE)
        # CP-SAT may expose the numeric default 0 as ``best_objective_bound``
        # when status is UNKNOWN and no incumbent/bound was established.  It
        # is not a certificate for this non-negative maximisation model.  In
        # that case retain the explicit model upper bound; only a feasible or
        # final solve may tighten it from the solver's objective bound.
        model_power_upper = min(
            variables["model_power_upper"],
            (
                maximum_power_limit
                if maximum_power_limit is not None
                else variables["model_power_upper"]
            ),
            exact_power if exact_power is not None else variables["model_power_upper"],
        )
        strict_bound = (
            max(0, ceil(solver.best_objective_bound - 1e-9))
            if feasible or status_code in (cp_model.OPTIMAL, cp_model.INFEASIBLE)
            else model_power_upper
        )
        if not feasible:
            fixed_core: tuple[tuple[int, str], ...] = ()
            if status_code == cp_model.INFEASIBLE and extract_fixed_skeleton_core:
                assignment_by_index = variables["fixed_assumption_assignments"]
                fixed_core = tuple(
                    assignment_by_index[int(literal)]
                    for literal in solver.sufficient_assumptions_for_infeasibility()
                )
            return ThermalMasterSolution(
                status=status,
                feasible=False,
                proven_optimal=proven,
                power=None,
                generated_heat=None,
                layout=None,
                cut_capacities=(),
                strict_power_upper_bound=strict_bound,
                elapsed_seconds=elapsed,
                conflicts=solver.num_conflicts,
                branches=solver.num_branches,
                fixed_skeleton_core=fixed_core,
            )
        layout = tuple(
            self.labels[next(
                code
                for code, flag in enumerate(variables["one_hot"][vertex])
                if solver.value(flag)
            )]
            for vertex in self.problem.graph.vertices
        )
        power = solver.value(variables["power"])
        if proven:
            strict_bound = power
        return ThermalMasterSolution(
            status=status,
            feasible=True,
            proven_optimal=proven,
            power=power,
            generated_heat=solver.value(variables["heat"]),
            layout=layout,
            cut_capacities=tuple(
                solver.value(item) for item in variables["cut_capacities"]
            ),
            strict_power_upper_bound=strict_bound,
            elapsed_seconds=elapsed,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
        )
