"""Auditable work bounds for the proof-producing optimisation pipeline.

These counts are deliberately independent of solver runtime.  They separate
what is known from the input alone from what must be measured after pruning;
in particular, a six-hour claim is accepted only when a finite remaining-unit
count and a measured conservative unit cost are both supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, floor, log2

from .frontier_dp import RectangularFrontierPowerDP
from .frontier_automata import (
    FactorAutomatonSelection,
    rectangular_frontier_orders,
)
from .factorized_layout_dp import FactorizedLayoutDPResult
from .factorized_cooling_master import CoolingOrderSelection
from .mathematical_model import ReactorProblem, closed_form_upper_bound
from .pareto_frontier_dp import ParetoFrontierDPResult
from .state_quotient import product_chain_antichain_bound
from .terminal_cut_quotient import TerminalCutScheduleProfile


def _label_assignment_count(problem: ReactorProblem, *, full_layout: bool) -> int:
    """Count rod-feasible label strings while deliberately ignoring geometry.

    This is the coefficient of a small generating polynomial raised to
    ``|V|``.  It is an exact count of strings satisfying only the rod equation,
    hence an auditable upper bound after adding inventory, power and heat cuts.
    """

    power_ids = {item.id for item in problem.power_components}
    if full_layout:
        extra_zero_rod_labels = len(set(problem.layout_components) - power_ids)
    else:
        extra_zero_rod_labels = 0
    multiplicity_by_rods: dict[int, int] = {}
    for item in problem.power_components:
        multiplicity_by_rods[item.rods] = multiplicity_by_rods.get(item.rods, 0) + 1
    multiplicity_by_rods[0] = multiplicity_by_rods.get(0, 0) + extra_zero_rod_labels

    budget = problem.rod_budget
    counts = [0] * (budget + 1)
    counts[0] = 1
    for _vertex in problem.graph.vertices:
        following = [0] * (budget + 1)
        for used, prefix_count in enumerate(counts):
            if not prefix_count:
                continue
            for rods, multiplicity in multiplicity_by_rods.items():
                if used + rods <= budget:
                    following[used + rods] += prefix_count * multiplicity
        counts = following
    if problem.exact_rods:
        return counts[budget]
    return sum(counts[1:])


@dataclass(frozen=True, slots=True)
class ProofWorkEnvelope:
    vertices: int
    edges: int
    maximum_degree: int
    incumbent_lower_bound: int
    static_upper_bound: int
    open_power_tiers: tuple[int, ...]
    power_labels: int
    full_labels: int
    aggregate_signature_types: int
    aggregate_count_vector_bound_per_tier: int
    rod_feasible_power_skeletons: int
    rod_feasible_full_layouts: int
    frontier_width: int | None
    frontier_state_bound: int | None
    frontier_transition_bound: int | None

    @property
    def open_tier_count(self) -> int:
        return len(self.open_power_tiers)


def proof_work_envelope(
    problem: ReactorProblem,
    *,
    incumbent_lower_bound: int = 0,
    static_upper_bound: int | None = None,
) -> ProofWorkEnvelope:
    """Return finite pre-search bounds derived solely from the model input."""

    if incumbent_lower_bound < 0:
        raise ValueError("incumbent_lower_bound must be non-negative")
    upper = (
        closed_form_upper_bound(problem).power_upper_bound
        if static_upper_bound is None
        else int(static_upper_bound)
    )
    if upper < 0:
        raise ValueError("static_upper_bound must be non-negative")
    step = problem.eu_per_pulse
    first_open = ((incumbent_lower_bound // step) + 1) * step
    open_tiers = (
        tuple(range(first_open, upper + 1, step))
        if first_open <= upper
        else ()
    )
    fuel_types = sum(item.rods > 0 for item in problem.power_components)
    signature_types = fuel_types * (problem.graph.maximum_degree + 1)
    # Number of non-negative signature count vectors with total at most |V|.
    aggregate_bound = comb(problem.graph.size + signature_types, signature_types)

    frontier_width = frontier_states = frontier_transitions = None
    if problem.graph.rows is not None and problem.graph.columns is not None:
        signature = RectangularFrontierPowerDP(problem).complexity_signature()
        frontier_width = signature["frontier_width"]
        frontier_states = signature["coarse_state_bound"]
        frontier_transitions = frontier_states * len(problem.power_components)

    full_labels = len(
        set(item.id for item in problem.power_components)
        | set(problem.layout_components)
    )
    return ProofWorkEnvelope(
        vertices=problem.graph.size,
        edges=len(problem.graph.edges),
        maximum_degree=problem.graph.maximum_degree,
        incumbent_lower_bound=incumbent_lower_bound,
        static_upper_bound=upper,
        open_power_tiers=open_tiers,
        power_labels=len(problem.power_components),
        full_labels=full_labels,
        aggregate_signature_types=signature_types,
        aggregate_count_vector_bound_per_tier=aggregate_bound,
        rod_feasible_power_skeletons=_label_assignment_count(
            problem,
            full_layout=False,
        ),
        rod_feasible_full_layouts=_label_assignment_count(
            problem,
            full_layout=True,
        ),
        frontier_width=frontier_width,
        frontier_state_bound=frontier_states,
        frontier_transition_bound=frontier_transitions,
    )


@dataclass(frozen=True, slots=True)
class SixHourProjection:
    remaining_work_units: int
    conservative_seconds_per_unit: float
    workers: int
    parallel_efficiency: float
    wall_time_budget_seconds: float
    required_core_seconds: float
    available_effective_core_seconds: float
    projected_wall_seconds: float
    fits_budget: bool
    maximum_units_in_budget: int


@dataclass(frozen=True, slots=True)
class MeasuredFrontierWork:
    proven: bool
    completed_layers: int
    raw_transitions: int
    dominated_rejections: int
    upper_bound_rejections: int
    peak_continuation_keys: int
    peak_pareto_points: int
    peak_antichain_width: int
    peak_automaton_state_tuples: int
    elapsed_seconds: float
    measured_transitions_per_second: float
    equivalent_successor_merges: int = 0
    automaton_transition_cache_hits: int = 0
    automaton_transition_cache_misses: int = 0


@dataclass(frozen=True, slots=True)
class MeasuredFactorizedLayoutWork:
    proven: bool
    completed_layers: int
    raw_label_transitions: int
    equivalent_successor_merges: int
    dominated_rejections: int
    peak_continuation_keys: int
    peak_pareto_points: int
    peak_antichain_width: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CompiledCoolingDomainWork:
    """Analytic work bound for a declared collection of fixed skeleton domains."""

    submitted_domains: int
    bounded_domains: int
    inventory_infeasible_domains: int
    total_raw_transition_bound: int
    peak_continuation_key_bound: int
    peak_pareto_point_bound: int
    all_open_domains_accounted: bool
    work_count_complete: bool
    missing_domain_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JointLayoutAnalyticWork:
    """Finite pre-DP bound for the one-level full-layout quotient master."""

    bound_complete: bool
    full_label_count: int
    static_power_behaviour_classes: int
    degree_tracked_power_behaviour_classes: int
    graph_separator_state_bound: int
    rod_state_bound: int
    cut_state_bound: int
    no_good_state_bound: int
    continuation_key_bound: int
    independent_degree_metric_pairs_total: int
    maximum_metric_pairs_per_rod_state: int
    inventory_chain_bounds: tuple[tuple[int, int], ...]
    cut_score_chain_bounds: tuple[tuple[int, int], ...]
    inventory_cut_antichain_width_bound: int | None
    peak_pareto_point_bound: int | None
    raw_label_transition_bound: int | None


@dataclass(frozen=True, slots=True)
class ExactEliminationChoice:
    """Auditable comparison of two exact variable-elimination strategies."""

    comparison_complete: bool
    selected_strategy: str | None
    joint_transition_bound: int | None
    conditioned_transition_bound: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class TerminalCutQuotientWork:
    """Storage and primitive-operation envelope for one all-cuts signature."""

    saturation: int
    peak_live_terminals: int
    cut_vector_entries: int
    bits_per_value: int
    packed_bytes_per_signature: int
    log2_raw_vector_count_bound: float
    primitive_values_per_full_vector_pass: int
    coarse_full_scan_value_operations_bound: int
    pointwise_dominance_sound: bool = True


@dataclass(frozen=True, slots=True)
class AverageFlowPrimalEncodingSize:
    """Additional compact-flow model size, excluding the shared layout model."""

    vertices: int
    undirected_edges: int
    flow_variables: int
    capacity_product_variables: int
    additional_integer_variables: int
    capacity_constraints: int
    multiplication_equalities: int
    conservation_and_total_constraints: int
    additional_constraints: int


@dataclass(frozen=True, slots=True)
class GlobalDynamicStateWorkBound:
    """Finite exhaustive-cycle fallback summed without listing layouts."""

    vertices: int
    rod_budget: int
    exact_rods: bool
    label_count: int
    rod_feasible_layout_count: int
    hull_bonus_bucket_count: int
    total_safe_state_step_bound: int
    maximum_single_layout_safe_state_bound: int
    decimal_digits: int
    inventory_limits_relaxed: bool


def global_dynamic_state_work_bound(
    problem: ReactorProblem,
    component_state_factors: dict[str, int],
    *,
    hull_capacity_bonuses: dict[str, int] | None = None,
    base_hull_capacity: int = 10_000,
    critical_ratio_numerator: int = 85,
    critical_ratio_denominator: int = 100,
) -> GlobalDynamicStateWorkBound:
    """Sum a complete deterministic-state fallback by generating function.

    For label ``c``, ``s_c`` is the number of its future-relevant safe local
    states and ``b_c`` its hull-capacity bonus.  After ``n`` slots the table
    contains coefficients of

    ``(sum_c s_c * z**rods(c) * y**b_c) ** n``.

    Multiplying each accepted bonus bucket by its safe hull-state count gives
    the sum of the fixed-layout pigeonhole bounds.  Component inventories are
    deliberately relaxed, so the result remains a rigorous upper bound even
    when many counted layouts are unavailable.
    """

    if base_hull_capacity <= 0:
        raise ValueError("base hull capacity must be positive")
    if not 0 < critical_ratio_numerator < critical_ratio_denominator:
        raise ValueError("critical ratio must lie strictly between zero and one")
    labels = tuple(dict.fromkeys((
        *(item.id for item in problem.power_components),
        *problem.layout_components,
    )))
    if missing := set(labels) - set(component_state_factors):
        raise ValueError(f"dynamic state factors are missing labels: {sorted(missing)}")
    factors = {
        label: int(component_state_factors[label]) for label in labels
    }
    if any(value <= 0 for value in factors.values()):
        raise ValueError("dynamic component state factors must be positive")
    bonuses = (
        {label: 0 for label in labels}
        if hull_capacity_bonuses is None
        else {
            label: int(hull_capacity_bonuses.get(label, 0))
            for label in labels
        }
    )
    if any(value < 0 for value in bonuses.values()):
        raise ValueError("hull capacity bonuses must be non-negative")
    rods_by_label = {item.id: item.rods for item in problem.power_components}
    choices = tuple(
        (
            rods_by_label.get(label, 0),
            bonuses[label],
            factors[label],
        )
        for label in labels
    )

    # (rods, hull bonus) -> (sum of local-state products, maximum product)
    layer: dict[tuple[int, int], tuple[int, int]] = {(0, 0): (1, 1)}
    for _vertex in problem.graph.vertices:
        following: dict[tuple[int, int], tuple[int, int]] = {}
        for (used_rods, bonus), (weighted_sum, maximum_product) in layer.items():
            for rods, added_bonus, factor in choices:
                following_rods = used_rods + rods
                if following_rods > problem.rod_budget:
                    continue
                key = (following_rods, bonus + added_bonus)
                old_sum, old_maximum = following.get(key, (0, 0))
                following[key] = (
                    old_sum + weighted_sum * factor,
                    max(old_maximum, maximum_product * factor),
                )
        layer = following

    accepted_rods = (
        (problem.rod_budget,)
        if problem.exact_rods
        else tuple(range(1, problem.rod_budget + 1))
    )
    total = 0
    maximum_single = 0
    bonus_buckets = set()
    for (rods, bonus), (weighted_sum, maximum_product) in layer.items():
        if rods not in accepted_rods:
            continue
        hull_states = (
            (base_hull_capacity + bonus) * critical_ratio_numerator
            // critical_ratio_denominator
        )
        total += weighted_sum * hull_states
        maximum_single = max(maximum_single, maximum_product * hull_states)
        bonus_buckets.add(bonus)
    if total <= 0:
        raise ValueError("dynamic work domain contains no rod-feasible layouts")
    return GlobalDynamicStateWorkBound(
        vertices=problem.graph.size,
        rod_budget=problem.rod_budget,
        exact_rods=problem.exact_rods,
        label_count=len(labels),
        rod_feasible_layout_count=_label_assignment_count(
            problem,
            full_layout=True,
        ),
        hull_bonus_bucket_count=len(bonus_buckets),
        total_safe_state_step_bound=total,
        maximum_single_layout_safe_state_bound=maximum_single,
        decimal_digits=len(str(total)),
        inventory_limits_relaxed=bool(problem.component_limits),
    )


def ic2_global_dynamic_state_work_bound(
    problem: ReactorProblem,
) -> GlobalDynamicStateWorkBound:
    """Catalogue adapter for the locked IC2 auto-refuel thermal quotient."""

    from .components import COMPONENTS

    labels = set(item.id for item in problem.power_components) | set(
        problem.layout_components
    )
    if missing := labels - COMPONENTS.keys():
        raise ValueError(f"IC2 dynamic catalogue is missing labels: {sorted(missing)}")
    state_factors = {}
    bonuses = {}
    for label in labels:
        spec = COMPONENTS[label]
        heat_states = spec.max_heat + 1 if spec.accepts_heat else 1
        damage_states = (
            spec.max_damage
            if spec.kind == "reflector" and spec.max_damage > 0
            else 1
        )
        state_factors[label] = heat_states * damage_states
        bonuses[label] = spec.hull_capacity_bonus
    return global_dynamic_state_work_bound(
        problem,
        state_factors,
        hull_capacity_bonuses=bonuses,
    )


def average_flow_primal_encoding_size(
    problem: ReactorProblem,
) -> AverageFlowPrimalEncodingSize:
    """Count the exact compact average-flow encoding before solving it.

    For ``n`` slots and ``m`` undirected adjacencies the implementation uses
    ``5n + 4m`` flow variables, ``6m`` label-capacity products and
    ``7n + 10m + 3`` constraints.  Max-flow/min-cut duality makes this one
    polynomial-size representation equivalent to the entire average-cut
    family; the count says nothing about CP propagation time.
    """

    vertices = problem.graph.size
    edges = len(problem.graph.edges)
    flow_variables = 5 * vertices + 4 * edges
    product_variables = 6 * edges
    capacity_constraints = flow_variables
    multiplication_equalities = product_variables
    conservation = 2 * vertices + 3
    return AverageFlowPrimalEncodingSize(
        vertices=vertices,
        undirected_edges=edges,
        flow_variables=flow_variables,
        capacity_product_variables=product_variables,
        additional_integer_variables=flow_variables + product_variables,
        capacity_constraints=capacity_constraints,
        multiplication_equalities=multiplication_equalities,
        conservation_and_total_constraints=conservation,
        additional_constraints=(
            capacity_constraints + multiplication_equalities + conservation
        ),
    )


def terminal_cut_quotient_work_bound(
    profile: TerminalCutScheduleProfile,
    *,
    saturation: int,
) -> TerminalCutQuotientWork:
    """Size the all-cuts quotient without materializing any cut vectors.

    Every coordinate lies in ``{0, ..., saturation}``, so the raw number of
    possible vectors is at most ``(saturation + 1) ** entries``.  Its base-two
    logarithm is retained instead of constructing a mostly useless enormous
    integer.  Reachability and pointwise Pareto dominance make the observed
    set much smaller, but are deliberately not assumed by this pre-run bound.
    """

    threshold = int(saturation)
    if threshold <= 0:
        raise ValueError("terminal-cut saturation must be positive")
    entries = profile.peak_cut_vector_entries
    bits = max(1, threshold.bit_length())
    return TerminalCutQuotientWork(
        saturation=threshold,
        peak_live_terminals=profile.peak_live_terminals,
        cut_vector_entries=entries,
        bits_per_value=bits,
        packed_bytes_per_signature=ceil(entries * bits / 8),
        log2_raw_vector_count_bound=entries * log2(threshold + 1),
        primitive_values_per_full_vector_pass=entries,
        coarse_full_scan_value_operations_bound=(
            profile.coarse_full_scan_value_operations_bound
        ),
    )


def _independent_degree_metric_pair_counts(
    problem: ReactorProblem,
) -> tuple[int, int]:
    """Count a local-degree relaxation, never graph layouts.

    Every fuel cell may independently choose any graph degree.  This superset
    forgets adjacency consistency but retains the exact rod, power and heat
    equations.  The resulting small generating-function DP bounds how many
    different ``(power, heat)`` coordinates can occur inside one rod state.
    """

    choices = tuple(dict.fromkeys(
        (
            item.rods,
            problem.eu_per_pulse * item.rods * (item.internal_pulses + degree),
            problem.heat_scale
            * item.rods
            * (item.internal_pulses + degree)
            * (item.internal_pulses + degree + 1),
        )
        for item in problem.power_components
        if item.rods > 0
        for degree in range(problem.graph.maximum_degree + 1)
    ))
    by_rods: list[set[tuple[int, int]]] = [
        set() for _ in range(problem.rod_budget + 1)
    ]
    by_rods[0].add((0, 0))
    for _cell in problem.graph.vertices:
        following = [set(values) for values in by_rods]
        for used_rods, values in enumerate(by_rods):
            for power, heat in values:
                for rods, added_power, added_heat in choices:
                    if used_rods + rods <= problem.rod_budget:
                        following[used_rods + rods].add((
                            power + added_power,
                            heat + added_heat,
                        ))
        if following == by_rods:
            break
        by_rods = following
    return (
        sum(map(len, by_rods)),
        max(map(len, by_rods), default=1),
    )


def _raster_graph_separator_state_bound(
    problem: ReactorProblem,
    placement_order: tuple[int, ...],
) -> tuple[int, int, int]:
    graph = problem.graph
    if graph.rows is None or graph.columns is None:
        raise ValueError("joint layout analytic bound requires a rectangular graph")
    if placement_order not in rectangular_frontier_orders(graph):
        raise ValueError("joint layout bound requires a minimum-width raster order")
    signatures = {
        (
            item.rods,
            item.internal_pulses if item.rods else 0,
            item.accepts_pulse,
        )
        for item in problem.power_components
    }
    # Every non-power full-layout label has the empty static behaviour, already
    # present by the ReactorProblem invariant.
    degree_tracked = sum(rods > 0 for rods, _internal, _active in signatures)
    fixed_degree = len(signatures) - degree_tracked
    width = min(graph.rows, graph.columns)
    rank = {vertex: step for step, vertex in enumerate(placement_order)}
    peak = 1
    for step in range(graph.size):
        major, minor = divmod(step, width)
        layer_bound = 1
        for frontier_minor in range(width):
            if frontier_minor <= minor:
                vertex_step = major * width + frontier_minor
            elif major > 0:
                vertex_step = (major - 1) * width + frontier_minor
            else:
                # The initial placeholder is the unique empty behaviour.
                layer_bound *= 1
                continue
            vertex = placement_order[vertex_step]
            placed_neighbours = sum(
                rank[neighbour] <= step
                for neighbour in graph.neighbours[vertex]
            )
            layer_bound *= (
                fixed_degree + degree_tracked * (placed_neighbours + 1)
            )
        peak = max(peak, layer_bound)
    return peak, len(signatures), degree_tracked


def joint_layout_analytic_work_bound(
    problem: ReactorProblem,
    selection: CoolingOrderSelection,
) -> JointLayoutAnalyticWork:
    """Bound a full-domain quotient master without executing its layout DP.

    The bound intentionally multiplies several independently reachable
    supersets, so it is conservative and often loose.  Its value is that every
    factor is finite, auditable and independent of a lucky solver trajectory.
    """

    order = tuple(selection.placement_order)
    graph_bound, behaviour_count, degree_tracked = (
        _raster_graph_separator_state_bound(problem, order)
    )
    full_labels = len(
        set(item.id for item in problem.power_components)
        | set(problem.layout_components)
    )
    no_good_bound = (
        selection.no_good_trie_nodes + 1
        if selection.relevant_layout_no_goods
        else 1
    )
    continuation_keys = (
        graph_bound
        * (problem.rod_budget + 1)
        * selection.structural_state_product_bound
        * no_good_bound
    )
    metric_total, metric_per_rods = _independent_degree_metric_pair_counts(problem)
    inventory_bounds = tuple(
        (0, int(limit))
        for _label, limit in problem.component_limits
        if limit is not None and int(limit) < problem.graph.size
    )
    score_bounds = tuple(selection.cut_score_chain_bounds)
    score_complete = (
        selection.submitted_cut_count == 0
        or selection.cut_score_antichain_width_bound is not None
    )
    if score_complete:
        resource_width = product_chain_antichain_bound((
            *inventory_bounds,
            *score_bounds,
        )).width
        peak_points = continuation_keys * metric_per_rods * resource_width
        transition_bound = problem.graph.size * full_labels * peak_points
    else:
        resource_width = peak_points = transition_bound = None
    return JointLayoutAnalyticWork(
        bound_complete=score_complete,
        full_label_count=full_labels,
        static_power_behaviour_classes=behaviour_count,
        degree_tracked_power_behaviour_classes=degree_tracked,
        graph_separator_state_bound=graph_bound,
        rod_state_bound=problem.rod_budget + 1,
        cut_state_bound=selection.structural_state_product_bound,
        no_good_state_bound=no_good_bound,
        continuation_key_bound=continuation_keys,
        independent_degree_metric_pairs_total=metric_total,
        maximum_metric_pairs_per_rod_state=metric_per_rods,
        inventory_chain_bounds=inventory_bounds,
        cut_score_chain_bounds=score_bounds,
        inventory_cut_antichain_width_bound=resource_width,
        peak_pareto_point_bound=peak_points,
        raw_label_transition_bound=transition_bound,
    )


def choose_exact_elimination_strategy(
    joint: JointLayoutAnalyticWork,
    conditioned: CompiledCoolingDomainWork,
) -> ExactEliminationChoice:
    """Choose only when both complete-domain transition bounds are present.

    An incomplete conditioned-domain ledger can hide unsubmitted skeletons;
    an incomplete joint ledger can hide an unbounded cut resource.  Treating
    either missing quantity as zero would manufacture a false performance
    result, so this function deliberately returns ``None`` until both sides
    are comparable.
    """

    joint_bound = (
        joint.raw_label_transition_bound if joint.bound_complete else None
    )
    conditioned_bound = (
        conditioned.total_raw_transition_bound
        if conditioned.work_count_complete
        else None
    )
    if joint_bound is None or conditioned_bound is None:
        missing = []
        if joint_bound is None:
            missing.append("joint full-domain bound")
        if conditioned_bound is None:
            missing.append("all conditioned domains")
        return ExactEliminationChoice(
            comparison_complete=False,
            selected_strategy=None,
            joint_transition_bound=joint_bound,
            conditioned_transition_bound=conditioned_bound,
            reason="missing " + " and ".join(missing),
        )
    selected = (
        "joint_full_layout"
        if joint_bound <= conditioned_bound
        else "conditioned_skeleton_domains"
    )
    return ExactEliminationChoice(
        comparison_complete=True,
        selected_strategy=selected,
        joint_transition_bound=joint_bound,
        conditioned_transition_bound=conditioned_bound,
        reason="selected the smaller certified raw-transition upper bound",
    )


def summarize_compiled_cooling_domains(
    selections: tuple[CoolingOrderSelection, ...],
    *,
    all_open_domains_accounted: bool,
) -> CompiledCoolingDomainWork:
    """Sum pre-DP cooling bounds without claiming coverage not supplied.

    A no-good trie contributes at most ``nodes + 1`` states (the extra state
    is the common mismatch sink).  Multiplying by this number is conservative
    because not every trie/cut-state combination is reachable.  Exact cycle
    verification is intentionally outside this ledger.
    """

    total_transitions = 0
    peak_keys = peak_points = 0
    bounded = infeasible = 0
    missing = []
    for index, selection in enumerate(selections):
        if selection.fixed_inventory_infeasible:
            infeasible += 1
            bounded += 1
            continue
        if (
            selection.raw_transitions_without_no_goods_bound is None
            or selection.peak_pareto_points_without_no_goods_bound is None
        ):
            missing.append(index)
            continue
        no_good_multiplier = (
            selection.no_good_trie_nodes + 1
            if selection.relevant_layout_no_goods
            else 1
        )
        domain_keys = (
            selection.structural_state_product_bound * no_good_multiplier
        )
        domain_points = (
            selection.peak_pareto_points_without_no_goods_bound
            * no_good_multiplier
        )
        domain_transitions = (
            selection.raw_transitions_without_no_goods_bound
            * no_good_multiplier
        )
        total_transitions += domain_transitions
        peak_keys = max(peak_keys, domain_keys)
        peak_points = max(peak_points, domain_points)
        bounded += 1
    complete = all_open_domains_accounted and not missing
    return CompiledCoolingDomainWork(
        submitted_domains=len(selections),
        bounded_domains=bounded,
        inventory_infeasible_domains=infeasible,
        total_raw_transition_bound=total_transitions,
        peak_continuation_key_bound=peak_keys,
        peak_pareto_point_bound=peak_points,
        all_open_domains_accounted=all_open_domains_accounted,
        work_count_complete=complete,
        missing_domain_indices=tuple(missing),
    )


def summarize_factorized_layout_work(
    result: FactorizedLayoutDPResult,
) -> MeasuredFactorizedLayoutWork:
    """Expose full-label behaviour quotienting as an auditable work ledger."""

    layers = result.layer_statistics
    return MeasuredFactorizedLayoutWork(
        proven=result.proven,
        completed_layers=max(0, len(layers) - 1),
        raw_label_transitions=result.raw_transitions,
        equivalent_successor_merges=result.equivalent_successor_merges,
        dominated_rejections=result.dominated_rejections,
        peak_continuation_keys=max(
            (layer.continuation_keys for layer in layers),
            default=0,
        ),
        peak_pareto_points=max(
            (layer.pareto_points for layer in layers),
            default=0,
        ),
        peak_antichain_width=max(
            (layer.maximum_antichain_width for layer in layers),
            default=0,
        ),
        elapsed_seconds=result.elapsed_seconds,
    )


def summarize_frontier_work(
    result: ParetoFrontierDPResult,
) -> MeasuredFrontierWork:
    """Convert one DP run into a solver-independent auditable work ledger."""

    layers = result.layer_statistics
    rate = (
        result.raw_transitions / result.elapsed_seconds
        if result.elapsed_seconds > 0
        else 0.0
    )
    return MeasuredFrontierWork(
        proven=result.proven,
        completed_layers=max(0, len(layers) - 1),
        raw_transitions=result.raw_transitions,
        dominated_rejections=result.dominated_rejections,
        upper_bound_rejections=result.upper_bound_rejections,
        peak_continuation_keys=max(
            (layer.continuation_keys for layer in layers),
            default=0,
        ),
        peak_pareto_points=max(
            (layer.pareto_points for layer in layers),
            default=0,
        ),
        peak_antichain_width=max(
            (layer.maximum_antichain_width for layer in layers),
            default=0,
        ),
        peak_automaton_state_tuples=max(
            (layer.automaton_state_tuples for layer in layers),
            default=0,
        ),
        elapsed_seconds=result.elapsed_seconds,
        measured_transitions_per_second=rate,
        equivalent_successor_merges=result.equivalent_successor_merges,
        automaton_transition_cache_hits=(
            result.automaton_transition_cache_hits
        ),
        automaton_transition_cache_misses=(
            result.automaton_transition_cache_misses
        ),
    )


@dataclass(frozen=True, slots=True)
class FrontierMemoryProjection:
    peak_continuation_keys: int
    peak_pareto_points: int
    conservative_bytes_per_key: int
    conservative_bytes_per_point: int
    fixed_overhead_bytes: int
    projected_peak_bytes: int
    memory_budget_bytes: int
    fits_budget: bool


@dataclass(frozen=True, slots=True)
class FactorCompilationLedger:
    cut_count: int
    raw_factor_table_entries: int
    quotient_factor_table_entries: int
    table_reduction_ratio: float | None
    per_variable_representations: int
    residual_function_representations: int
    maximum_assignment_state_bound: int
    maximum_residual_state_bound: int


def summarize_factor_compilation(
    selections: tuple[FactorAutomatonSelection, ...],
) -> FactorCompilationLedger:
    """Aggregate cut-compilation state bounds without executing a layout DP."""

    raw = sum(item.raw_factor_table_entries for item in selections)
    quotient = sum(item.quotient_factor_table_entries for item in selections)
    return FactorCompilationLedger(
        cut_count=len(selections),
        raw_factor_table_entries=raw,
        quotient_factor_table_entries=quotient,
        table_reduction_ratio=(raw / quotient if quotient else None),
        per_variable_representations=sum(
            item.selected_representation == "per_variable_function_quotient"
            for item in selections
        ),
        residual_function_representations=sum(
            item.selected_representation == "conditioned_residual_functions"
            for item in selections
        ),
        maximum_assignment_state_bound=max(
            (item.assignment_peak_product for item in selections),
            default=1,
        ),
        maximum_residual_state_bound=max(
            (item.residual_peak_product for item in selections),
            default=1,
        ),
    )


@dataclass(frozen=True, slots=True)
class ExactProofCapacityCertificate:
    time_projection: SixHourProjection
    memory_projection: FrontierMemoryProjection
    work_count_complete: bool
    conservative_unit_cost_measured: bool
    conservative_memory_cost_measured: bool
    parallel_efficiency_measured: bool
    certified_fit: bool
    failure_reasons: tuple[str, ...]


def certify_exact_proof_capacity(
    time_projection: SixHourProjection,
    memory_projection: FrontierMemoryProjection,
    *,
    work_count_complete: bool,
    conservative_unit_cost_measured: bool,
    conservative_memory_cost_measured: bool,
    parallel_efficiency_measured: bool,
) -> ExactProofCapacityCertificate:
    """Issue a positive capacity certificate only when every premise exists."""

    reasons = []
    if not work_count_complete:
        reasons.append("post-cut remaining work count is incomplete")
    if not conservative_unit_cost_measured:
        reasons.append("conservative unit-time quantile is unmeasured")
    if not conservative_memory_cost_measured:
        reasons.append("conservative per-state memory cost is unmeasured")
    if time_projection.workers > 1 and not parallel_efficiency_measured:
        reasons.append("multi-worker parallel efficiency is unmeasured")
    if not time_projection.fits_budget:
        reasons.append("projected effective core-seconds exceed the time budget")
    if not memory_projection.fits_budget:
        reasons.append("projected peak memory exceeds the memory budget")
    return ExactProofCapacityCertificate(
        time_projection=time_projection,
        memory_projection=memory_projection,
        work_count_complete=work_count_complete,
        conservative_unit_cost_measured=conservative_unit_cost_measured,
        conservative_memory_cost_measured=conservative_memory_cost_measured,
        parallel_efficiency_measured=parallel_efficiency_measured,
        certified_fit=not reasons,
        failure_reasons=tuple(reasons),
    )


def project_frontier_memory(
    peak_continuation_keys: int,
    peak_pareto_points: int,
    *,
    conservative_bytes_per_key: int,
    conservative_bytes_per_point: int,
    memory_budget_bytes: int,
    fixed_overhead_bytes: int = 0,
) -> FrontierMemoryProjection:
    """Check a measured key/point ledger against an explicit memory budget."""

    counts = (peak_continuation_keys, peak_pareto_points, fixed_overhead_bytes)
    if any(value < 0 for value in counts):
        raise ValueError("frontier counts and fixed overhead must be non-negative")
    if conservative_bytes_per_key <= 0 or conservative_bytes_per_point <= 0:
        raise ValueError("conservative byte costs must be positive")
    if memory_budget_bytes <= 0:
        raise ValueError("memory budget must be positive")
    projected = (
        fixed_overhead_bytes
        + peak_continuation_keys * conservative_bytes_per_key
        + peak_pareto_points * conservative_bytes_per_point
    )
    return FrontierMemoryProjection(
        peak_continuation_keys=peak_continuation_keys,
        peak_pareto_points=peak_pareto_points,
        conservative_bytes_per_key=conservative_bytes_per_key,
        conservative_bytes_per_point=conservative_bytes_per_point,
        fixed_overhead_bytes=fixed_overhead_bytes,
        projected_peak_bytes=projected,
        memory_budget_bytes=memory_budget_bytes,
        fits_budget=projected <= memory_budget_bytes,
    )


def project_proof_budget(
    remaining_work_units: int,
    conservative_seconds_per_unit: float,
    *,
    workers: int,
    parallel_efficiency: float = 0.70,
    wall_time_budget_seconds: float = 21_600.0,
) -> SixHourProjection:
    """Project a measured finite work ledger onto a wall-clock budget.

    ``remaining_work_units`` must be counted *after* all claimed family cuts;
    the theoretical layout count is not silently substituted with a guess.
    ``conservative_seconds_per_unit`` is expected to come from a microbenchmark
    upper quantile, not the fastest observed run.
    """

    if remaining_work_units < 0:
        raise ValueError("remaining_work_units must be non-negative")
    if conservative_seconds_per_unit <= 0:
        raise ValueError("conservative_seconds_per_unit must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 0 < parallel_efficiency <= 1:
        raise ValueError("parallel_efficiency must be in (0, 1]")
    if wall_time_budget_seconds <= 0:
        raise ValueError("wall_time_budget_seconds must be positive")

    required = remaining_work_units * conservative_seconds_per_unit
    effective_workers = workers * parallel_efficiency
    available = wall_time_budget_seconds * effective_workers
    projected_wall = required / effective_workers
    maximum_units = floor(available / conservative_seconds_per_unit)
    return SixHourProjection(
        remaining_work_units=remaining_work_units,
        conservative_seconds_per_unit=conservative_seconds_per_unit,
        workers=workers,
        parallel_efficiency=parallel_efficiency,
        wall_time_budget_seconds=wall_time_budget_seconds,
        required_core_seconds=required,
        available_effective_core_seconds=available,
        projected_wall_seconds=projected_wall,
        fits_budget=required <= available,
        maximum_units_in_budget=maximum_units,
    )
