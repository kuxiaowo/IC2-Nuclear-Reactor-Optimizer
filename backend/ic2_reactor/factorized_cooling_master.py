"""Reusable fixed-skeleton cooling master built from factorized proof cuts."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from math import prod
from time import perf_counter
from typing import Mapping, Sequence

from .factorized_layout_dp import (
    FactorizedLayoutDPResult,
    FactorizedLayoutFeasibilityDP,
)
from .frontier_automata import (
    ExcludedLayoutsAutomaton,
    FactorAutomatonSelection,
    FrontierConstraintAutomaton,
    JointLocalFactorThresholdAutomaton,
    LocalScoreFactor,
    NormalizedFactorConstraint,
    condition_factor_constraint,
    factor_hypergraph_frontier_orders,
    mobius_decompose_local_factor,
    normalize_factor_constraint,
    rectangular_frontier_orders,
    remove_factorwise_implied_constraints,
    select_factor_automaton,
)
from .mathematical_model import PowerComponent, ReactorProblem
from .pareto_frontier_dp import (
    ParetoFrontierDPResult,
    RectangularParetoPowerHeatDP,
)
from .periodic_cut_automaton import (
    PeriodicPrefixCutFactorization,
    compile_periodic_prefix_cut,
)
from .periodic_prefix import PeriodicPrefixCutTemplate, PrefixHeatComponent
from .thermal_cut_automaton import ThermalCutFactorization, compile_thermal_cut
from .thermal_relaxation import HeatFlowComponent, ThermalCutTemplate
from .thermal_terminal_cut import AverageFlowTerminalCutAutomaton
from .state_quotient import product_chain_antichain_bound


@dataclass(frozen=True, slots=True)
class CoolingOrderSelection:
    candidate_order_count: int
    placement_order: tuple[int, ...]
    structural_state_product_bound: int
    submitted_cut_count: int
    distinct_factor_constraint_count: int
    raw_factor_table_entries: int
    quotient_factor_table_entries: int
    per_variable_representations: int
    residual_function_representations: int
    submitted_layout_no_goods: int
    relevant_layout_no_goods: int
    no_good_trie_nodes: int
    joint_per_variable_representation: bool
    separate_constraint_state_bounds: tuple[int, ...]
    cut_score_chain_bounds: tuple[tuple[int, int], ...]
    cut_score_antichain_width_bound: int | None
    cut_score_antichain_width_exact: bool
    inventory_chain_bounds: tuple[tuple[int, int], ...] = ()
    resource_antichain_width_bound: int | None = None
    resource_antichain_width_exact: bool = False
    peak_pareto_points_without_no_goods_bound: int | None = None
    raw_transitions_without_no_goods_bound: int | None = None
    fixed_inventory_infeasible: bool = False
    specialized_factor_cache_hits: int = 0
    specialized_factor_cache_misses: int = 0
    specialized_factor_cache_entries: int = 0
    specialized_factor_cache_limit: int = 0
    maximum_local_specializations_for_submitted_cuts: int = 0
    lower_order_factor_decompositions: int = 0
    complete_average_flow: bool = False
    complete_average_flow_peak_terminals: int = 0
    complete_average_flow_vector_entries: int = 0
    complete_average_flow_state_bound: int = 1


@dataclass(frozen=True, slots=True)
class CoolingCompilationResult:
    proven: bool
    selection: CoolingOrderSelection | None
    elapsed_seconds: float
    stop_reason: str


class FactorizedCoolingCutMaster:
    """Intersect reusable thermal cuts without retaining raw cooling history."""

    def __init__(
        self,
        problem: ReactorProblem,
        heat_catalogue: Mapping[str, HeatFlowComponent],
        *,
        prefix_catalogue: Mapping[str, PrefixHeatComponent] | None = None,
        base_hull_capacity: int = 10_000,
        specialized_factor_cache_limit: int = 20_000,
    ) -> None:
        if base_hull_capacity <= 0:
            raise ValueError("base hull capacity must be positive")
        if specialized_factor_cache_limit <= 0:
            raise ValueError("specialized factor cache limit must be positive")
        self.problem = problem
        self.heat_catalogue = dict(heat_catalogue)
        self.prefix_catalogue = (
            None if prefix_catalogue is None else dict(prefix_catalogue)
        )
        self.base_hull_capacity = base_hull_capacity
        self.specialized_factor_cache_limit = specialized_factor_cache_limit
        self._specialized_factor_cache: OrderedDict[
            tuple[int, tuple[tuple[int, int], ...]],
            NormalizedFactorConstraint,
        ] = OrderedDict()
        self.specialized_factor_cache_hits = 0
        self.specialized_factor_cache_misses = 0
        self.lower_order_factor_decompositions = 0
        power_labels = tuple(item.id for item in problem.power_components)
        self.labels = tuple(dict.fromkeys((*power_labels, *problem.layout_components)))
        if missing := set(self.labels) - self.heat_catalogue.keys():
            raise ValueError(f"heat catalogue is missing labels: {sorted(missing)}")
        if self.prefix_catalogue is not None:
            if missing := set(self.labels) - self.prefix_catalogue.keys():
                raise ValueError(f"prefix catalogue is missing labels: {sorted(missing)}")
        power_nonempty = set(power_labels) - {"empty"}
        self.free_labels = tuple(
            label
            for label in self.labels
            if label not in power_nonempty
        )
        code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        self.free_codes = tuple(code_by_label[label] for label in self.free_labels)
        graph = problem.graph
        self.placement_orders = (
            rectangular_frontier_orders(graph)
            if graph.rows is not None and graph.columns is not None
            else (graph.update_order,)
        )
        self.placement_order = self.placement_orders[0]
        self.last_order_selection = CoolingOrderSelection(
            candidate_order_count=len(self.placement_orders),
            placement_order=self.placement_order,
            structural_state_product_bound=1,
            submitted_cut_count=0,
            distinct_factor_constraint_count=0,
            raw_factor_table_entries=0,
            quotient_factor_table_entries=0,
            per_variable_representations=0,
            residual_function_representations=0,
            submitted_layout_no_goods=0,
            relevant_layout_no_goods=0,
            no_good_trie_nodes=0,
            joint_per_variable_representation=False,
            separate_constraint_state_bounds=(),
            cut_score_chain_bounds=(),
            cut_score_antichain_width_bound=1,
            cut_score_antichain_width_exact=True,
        )
        self.component_limits = dict(problem.component_limits)
        power_by_label = {item.id: item for item in problem.power_components}
        # Lift the full component domain into the static power DP.  Cooling
        # labels have exactly the empty label's power behaviour, while their
        # distinct thermal behaviour remains visible to the cut automata.
        # This permits one exact quotient DP over layouts instead of an outer
        # loop that enumerates power skeletons one at a time.
        self.full_layout_problem = ReactorProblem(
            graph=problem.graph,
            rod_budget=problem.rod_budget,
            exact_rods=problem.exact_rods,
            power_components=tuple(
                power_by_label.get(
                    label,
                    PowerComponent(label, 0, 0, False),
                )
                for label in self.labels
            ),
            cooling_components=problem.cooling_components,
            layout_components=(),
            component_limits=problem.component_limits,
            eu_per_pulse=problem.eu_per_pulse,
            heat_scale=problem.heat_scale,
            ruleset=f"{problem.ruleset}:joint-full-layout-quotient",
        )
        self._average_cache: dict[
            tuple[ThermalCutTemplate, tuple[int, ...]],
            ThermalCutFactorization,
        ] = {}
        self._average_factor_cache: dict[
            ThermalCutTemplate,
            tuple[LocalScoreFactor, ...],
        ] = {}
        self._prefix_cache: dict[
            tuple[PeriodicPrefixCutTemplate, tuple[int, ...]],
            PeriodicPrefixCutFactorization,
        ] = {}
        self._prefix_factor_cache: dict[
            PeriodicPrefixCutTemplate,
            tuple[LocalScoreFactor, ...],
        ] = {}

    @property
    def cached_average_cuts(self) -> int:
        return len(self._average_factor_cache)

    @property
    def cached_prefix_cuts(self) -> int:
        return len(self._prefix_factor_cache)

    @property
    def cached_average_order_automata(self) -> int:
        return len(self._average_cache)

    @property
    def cached_prefix_order_automata(self) -> int:
        return len(self._prefix_cache)

    @staticmethod
    def _cut_score_antichain_report(
        automata: Sequence[FrontierConstraintAutomaton],
    ) -> tuple[tuple[tuple[int, int], ...], int | None, bool]:
        bounds = []
        for automaton in automata:
            reporter = getattr(automaton, "pareto_resource_chain_bounds", None)
            if reporter is None:
                return (), None, False
            reported = reporter()
            if reported is None:
                return (), None, False
            bounds.extend(reported)
        result = product_chain_antichain_bound(tuple(bounds))
        return tuple(bounds), result.width, result.exact

    def _condition_constraint_cached(
        self,
        factors: Sequence[LocalScoreFactor],
        fixed_codes: Mapping[int, int],
    ) -> NormalizedFactorConstraint:
        """Reuse exact local partial evaluations across global skeletons."""

        residuals = []
        threshold = 0
        for factor in factors:
            local_assignment = tuple(
                (
                    vertex,
                    factor.canonical_code(position, fixed_codes[vertex]),
                )
                for position, vertex in enumerate(factor.scope)
                if vertex in fixed_codes
            )
            key = (id(factor), local_assignment)
            partial = self._specialized_factor_cache.get(key)
            if partial is None:
                self.specialized_factor_cache_misses += 1
                partial = condition_factor_constraint(
                    (factor,),
                    dict(local_assignment),
                    threshold=0,
                    allowed_codes=self.free_codes,
                )
                if len(partial.factors) == 1:
                    residual = partial.factors[0]
                    constant, pieces = mobius_decompose_local_factor(residual)
                    maximum_piece_scope = max(
                        (len(piece.scope) for piece in pieces),
                        default=0,
                    )
                    if maximum_piece_scope < len(residual.scope):
                        partial = normalize_factor_constraint(
                            pieces,
                            threshold=partial.threshold - constant,
                        )
                        self.lower_order_factor_decompositions += 1
                self._specialized_factor_cache[key] = partial
                if (
                    len(self._specialized_factor_cache)
                    > self.specialized_factor_cache_limit
                ):
                    self._specialized_factor_cache.popitem(last=False)
            else:
                self.specialized_factor_cache_hits += 1
                self._specialized_factor_cache.move_to_end(key)
            residuals.extend(partial.factors)
            threshold += partial.threshold
        return normalize_factor_constraint(
            residuals,
            threshold=threshold,
        )

    def _average_automaton(
        self,
        cut: ThermalCutTemplate,
        placement_order: tuple[int, ...],
    ) -> ThermalCutFactorization:
        key = (cut, placement_order)
        compiled = self._average_cache.get(key)
        if compiled is None:
            factors = self._average_factor_cache.get(cut)
            if factors is None:
                compiled = compile_thermal_cut(
                    self.problem,
                    cut,
                    self.heat_catalogue,
                    self.labels,
                    placement_order=placement_order,
                )
                factors = compiled.factors
                self._average_factor_cache[cut] = factors
            else:
                automaton, selection = select_factor_automaton(
                    placement_order,
                    factors,
                    threshold=0,
                )
                compiled = ThermalCutFactorization(
                    self.labels,
                    factors,
                    automaton,
                    selection,
                )
            self._average_cache[key] = compiled
        return compiled

    def _prefix_automaton(
        self,
        cut: PeriodicPrefixCutTemplate,
        placement_order: tuple[int, ...],
    ) -> PeriodicPrefixCutFactorization:
        if self.prefix_catalogue is None:
            raise ValueError("periodic cuts require a prefix catalogue")
        key = (cut, placement_order)
        compiled = self._prefix_cache.get(key)
        if compiled is None:
            factors = self._prefix_factor_cache.get(cut)
            if factors is None:
                compiled = compile_periodic_prefix_cut(
                    self.problem,
                    cut,
                    self.prefix_catalogue,
                    self.labels,
                    base_hull_capacity=self.base_hull_capacity,
                    placement_order=placement_order,
                )
                factors = compiled.factors
                self._prefix_factor_cache[cut] = factors
            else:
                automaton, selection = select_factor_automaton(
                    placement_order,
                    factors,
                    threshold=0,
                )
                compiled = PeriodicPrefixCutFactorization(
                    self.labels,
                    factors,
                    automaton,
                    selection,
                )
            self._prefix_cache[key] = compiled
        return compiled

    def _select_order_and_automata(
        self,
        average_cuts: Sequence[ThermalCutTemplate],
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate],
        *,
        excluded_layouts: Sequence[Sequence[str]] = (),
        deadline: float | None = None,
    ) -> tuple[
        tuple[int, ...],
        tuple[FrontierConstraintAutomaton, ...],
    ] | None:
        best: tuple[
            int,
            int,
            tuple[int, ...],
            tuple[FrontierConstraintAutomaton, ...],
            tuple[FactorAutomatonSelection, ...],
            tuple[int, ...],
            bool,
        ] | None = None
        for placement_order in self.placement_orders:
            if deadline is not None and perf_counter() >= deadline:
                return None
            compiled = []
            for cut in average_cuts:
                compiled.append(self._average_automaton(cut, placement_order))
                if deadline is not None and perf_counter() >= deadline:
                    return None
            for cut in prefix_cuts:
                compiled.append(self._prefix_automaton(cut, placement_order))
                if deadline is not None and perf_counter() >= deadline:
                    return None
            strongest_by_function = {}
            for item in compiled:
                signature = normalize_factor_constraint(item.factors)
                if signature.threshold <= 0:
                    continue
                previous = strongest_by_function.get(signature.factors)
                if previous is None or signature.threshold > previous[0]:
                    strongest_by_function[signature.factors] = (
                        signature.threshold,
                        item,
                    )
            all_normalized_constraints = tuple(
                NormalizedFactorConstraint(factors, threshold_and_item[0])
                for factors, threshold_and_item in strongest_by_function.items()
            )
            item_by_constraint = {
                NormalizedFactorConstraint(factors, threshold_and_item[0]): (
                    threshold_and_item[1]
                )
                for factors, threshold_and_item in strongest_by_function.items()
            }
            normalized_constraints = remove_factorwise_implied_constraints(
                all_normalized_constraints
            )
            distinct_compiled = tuple(
                item_by_constraint[constraint]
                for constraint in normalized_constraints
            )
            separate_bounds = tuple(
                item.selection.selected_state_bound_including_guaranteed
                for item in distinct_compiled
            )
            separate_bound = prod(separate_bounds)
            structural_bound = separate_bound
            automata = tuple(item.automaton for item in distinct_compiled)
            joint_representation = False
            if len(normalized_constraints) > 1:
                joint = JointLocalFactorThresholdAutomaton(
                    placement_order,
                    normalized_constraints,
                )
                joint_bound = joint.complexity_profile().peak_quotient_label_product
                if joint_bound < separate_bound:
                    structural_bound = joint_bound
                    automata = (joint,)
                    joint_representation = True
            no_good_state_bound = 1
            if excluded_layouts:
                no_goods = ExcludedLayoutsAutomaton(
                    placement_order,
                    self.labels,
                    excluded_layouts,
                )
                no_good_state_bound = no_goods.trie_node_count + 1
            selections = tuple(item.selection for item in distinct_compiled)
            candidate = (
                structural_bound * no_good_state_bound,
                structural_bound,
                placement_order,
                automata,
                selections,
                separate_bounds,
                joint_representation,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:  # pragma: no cover - every graph has one scan order
            raise AssertionError("cooling master has no placement order")
        (
            _combined_bound,
            bound,
            placement_order,
            automata,
            selections,
            separate_bounds,
            joint_representation,
        ) = best
        (
            cut_score_bounds,
            cut_score_width,
            cut_score_width_exact,
        ) = self._cut_score_antichain_report(automata)
        self.last_order_selection = CoolingOrderSelection(
            candidate_order_count=len(self.placement_orders),
            placement_order=placement_order,
            structural_state_product_bound=bound,
            submitted_cut_count=len(average_cuts) + len(prefix_cuts),
            distinct_factor_constraint_count=len(selections),
            raw_factor_table_entries=sum(
                item.raw_factor_table_entries for item in selections
            ),
            quotient_factor_table_entries=sum(
                item.quotient_factor_table_entries for item in selections
            ),
            per_variable_representations=sum(
                item.selected_representation == "per_variable_function_quotient"
                for item in selections
            ) if not joint_representation else 1,
            residual_function_representations=(
                sum(
                    item.selected_representation == "conditioned_residual_functions"
                    for item in selections
                )
                if not joint_representation
                else 0
            ),
            submitted_layout_no_goods=0,
            relevant_layout_no_goods=0,
            no_good_trie_nodes=0,
            joint_per_variable_representation=joint_representation,
            separate_constraint_state_bounds=separate_bounds,
            cut_score_chain_bounds=cut_score_bounds,
            cut_score_antichain_width_bound=cut_score_width,
            cut_score_antichain_width_exact=cut_score_width_exact,
        )
        return placement_order, automata

    def _select_specialized_order_and_automata(
        self,
        average_cuts: Sequence[ThermalCutTemplate],
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate],
        *,
        fixed_codes: Mapping[int, int],
        deadline: float | None = None,
    ) -> tuple[
        tuple[int, ...],
        tuple[FrontierConstraintAutomaton, ...],
    ] | None:
        """Condition every cut on one fixed power skeleton before selection.

        All remaining variables use ``self.free_codes``.  Consequently a
        fuel-injection star at a free centre vanishes, a star at a fixed fuel
        centre loses the centre and every fixed neighbour, and its remaining
        labels are quotiented only by cooling behaviour.  This is an exact
        partial evaluation, not a relaxation.
        """

        cache_hits_before = self.specialized_factor_cache_hits
        cache_misses_before = self.specialized_factor_cache_misses
        decompositions_before = self.lower_order_factor_decompositions
        bootstrap_order = self.placement_orders[0]
        submitted: list[NormalizedFactorConstraint] = []
        submitted_factors_by_id: dict[int, LocalScoreFactor] = {}
        for cut in average_cuts:
            factors = self._average_factor_cache.get(cut)
            if factors is None:
                factors = self._average_automaton(cut, bootstrap_order).factors
            submitted_factors_by_id.update((id(factor), factor) for factor in factors)
            submitted.append(self._condition_constraint_cached(
                factors,
                fixed_codes,
            ))
            if deadline is not None and perf_counter() >= deadline:
                return None
        for cut in prefix_cuts:
            factors = self._prefix_factor_cache.get(cut)
            if factors is None:
                factors = self._prefix_automaton(cut, bootstrap_order).factors
            submitted_factors_by_id.update((id(factor), factor) for factor in factors)
            submitted.append(self._condition_constraint_cached(
                factors,
                fixed_codes,
            ))
            if deadline is not None and perf_counter() >= deadline:
                return None

        strongest_by_function: dict[
            tuple[LocalScoreFactor, ...],
            int,
        ] = {}
        for constraint in submitted:
            if constraint.threshold <= 0:
                continue
            previous = strongest_by_function.get(constraint.factors)
            if previous is None or constraint.threshold > previous:
                strongest_by_function[constraint.factors] = constraint.threshold
        constraints = remove_factorwise_implied_constraints(tuple(
            NormalizedFactorConstraint(factors, threshold)
            for factors, threshold in strongest_by_function.items()
        ))

        best: tuple[
            int,
            tuple[int, ...],
            tuple[FrontierConstraintAutomaton, ...],
            tuple[FactorAutomatonSelection, ...],
            tuple[int, ...],
            bool,
        ] | None = None
        evaluated_orders: set[tuple[int, ...]] = set()

        def compile_candidate(
            placement_order: tuple[int, ...],
        ) -> tuple[
            int,
            tuple[int, ...],
            tuple[FrontierConstraintAutomaton, ...],
            tuple[FactorAutomatonSelection, ...],
            tuple[int, ...],
            bool,
        ] | None:
            if deadline is not None and perf_counter() >= deadline:
                return None
            automata = []
            selections = []
            for constraint in constraints:
                automaton, selection = select_factor_automaton(
                    placement_order,
                    constraint.factors,
                    threshold=constraint.threshold,
                    allowed_codes=(
                        self.free_codes if constraint.factors else None
                    ),
                )
                automata.append(automaton)
                selections.append(selection)
                if deadline is not None and perf_counter() >= deadline:
                    return None
            selection_tuple = tuple(selections)
            separate_bounds = tuple(
                item.selected_state_bound_including_guaranteed
                for item in selection_tuple
            )
            structural_bound = prod(separate_bounds)
            automaton_tuple: tuple[FrontierConstraintAutomaton, ...] = tuple(
                automata
            )
            joint_representation = False
            if len(constraints) > 1 and any(item.factors for item in constraints):
                joint = JointLocalFactorThresholdAutomaton(
                    placement_order,
                    constraints,
                )
                joint_bound = joint.complexity_profile(
                    allowed_codes=self.free_codes,
                ).peak_quotient_label_product
                if joint_bound < structural_bound:
                    structural_bound = joint_bound
                    automaton_tuple = (joint,)
                    joint_representation = True
            return (
                structural_bound,
                placement_order,
                automaton_tuple,
                selection_tuple,
                separate_bounds,
                joint_representation,
            )

        for placement_order in self.placement_orders:
            candidate = compile_candidate(placement_order)
            if candidate is None:
                return None
            evaluated_orders.add(placement_order)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        # A conditioned factor graph can be sparse even though the original
        # reactor grid is not.  Only pay for order search when the four exact
        # raster profiles are still large enough for the possible reduction
        # to matter.
        if best is not None and best[0] > 4096:
            cut_aware_orders = factor_hypergraph_frontier_orders(
                self.problem.graph.vertices,
                tuple(
                    factor
                    for constraint in constraints
                    for factor in constraint.factors
                ),
                allowed_codes=self.free_codes,
                beam_width=24,
                deadline=deadline,
            )
            if deadline is not None and perf_counter() >= deadline:
                return None
            for placement_order in cut_aware_orders:
                if placement_order in evaluated_orders:
                    continue
                candidate = compile_candidate(placement_order)
                if candidate is None:
                    return None
                evaluated_orders.add(placement_order)
                if candidate[:2] < best[:2]:
                    best = candidate

        if best is None:  # pragma: no cover - every graph has one scan order
            raise AssertionError("specialized cooling master has no placement order")
        (
            bound,
            placement_order,
            automata,
            selections,
            separate_bounds,
            joint_representation,
        ) = best
        (
            cut_score_bounds,
            cut_score_width,
            cut_score_width_exact,
        ) = self._cut_score_antichain_report(automata)
        fixed_counts = Counter(fixed_codes.values())
        free_vertex_count = self.problem.graph.size - len(fixed_codes)
        code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        inventory_bounds = []
        fixed_inventory_infeasible = False
        for label, raw_limit in self.component_limits.items():
            if raw_limit is None:
                continue
            code = code_by_label[label]
            remaining = int(raw_limit) - fixed_counts[code]
            if remaining < 0:
                fixed_inventory_infeasible = True
                break
            if code in self.free_codes and remaining < free_vertex_count:
                inventory_bounds.append((0, remaining))
        if fixed_inventory_infeasible:
            resource_width = 0
            resource_width_exact = True
            peak_point_bound = 0
            transition_bound = 0
        elif cut_score_width is None:
            resource_width = None
            resource_width_exact = False
            peak_point_bound = None
            transition_bound = None
        else:
            resource_report = product_chain_antichain_bound((
                *inventory_bounds,
                *cut_score_bounds,
            ))
            resource_width = resource_report.width
            resource_width_exact = resource_report.exact
            peak_point_bound = bound * resource_width
            attempted_labels_by_layer_sum = (
                len(fixed_codes)
                + free_vertex_count * len(self.free_codes)
            )
            transition_bound = (
                attempted_labels_by_layer_sum * peak_point_bound
            )
        nonempty_power_codes = tuple(
            code_by_label[item.id]
            for item in self.problem.power_components
            if item.id != "empty"
        )
        maximum_local_specializations = sum(
            prod(
                1 + len({
                    (
                        factor.projections[position][code]
                        if factor.projections
                        else code
                    )
                    for code in nonempty_power_codes
                })
                for position in range(len(factor.scope))
            )
            for factor in submitted_factors_by_id.values()
        )
        self.last_order_selection = CoolingOrderSelection(
            candidate_order_count=len(evaluated_orders),
            placement_order=placement_order,
            structural_state_product_bound=bound,
            submitted_cut_count=len(average_cuts) + len(prefix_cuts),
            distinct_factor_constraint_count=len(selections),
            raw_factor_table_entries=sum(
                item.raw_factor_table_entries for item in selections
            ),
            quotient_factor_table_entries=sum(
                item.quotient_factor_table_entries for item in selections
            ),
            per_variable_representations=(
                1
                if joint_representation
                else sum(
                    item.selected_representation
                    == "per_variable_function_quotient"
                    for item in selections
                )
            ),
            residual_function_representations=(
                0
                if joint_representation
                else sum(
                    item.selected_representation
                    == "conditioned_residual_functions"
                    for item in selections
                )
            ),
            submitted_layout_no_goods=0,
            relevant_layout_no_goods=0,
            no_good_trie_nodes=0,
            joint_per_variable_representation=joint_representation,
            separate_constraint_state_bounds=separate_bounds,
            cut_score_chain_bounds=cut_score_bounds,
            cut_score_antichain_width_bound=cut_score_width,
            cut_score_antichain_width_exact=cut_score_width_exact,
            inventory_chain_bounds=tuple(inventory_bounds),
            resource_antichain_width_bound=resource_width,
            resource_antichain_width_exact=resource_width_exact,
            peak_pareto_points_without_no_goods_bound=peak_point_bound,
            raw_transitions_without_no_goods_bound=transition_bound,
            fixed_inventory_infeasible=fixed_inventory_infeasible,
            specialized_factor_cache_hits=(
                self.specialized_factor_cache_hits - cache_hits_before
            ),
            specialized_factor_cache_misses=(
                self.specialized_factor_cache_misses - cache_misses_before
            ),
            specialized_factor_cache_entries=len(self._specialized_factor_cache),
            specialized_factor_cache_limit=self.specialized_factor_cache_limit,
            maximum_local_specializations_for_submitted_cuts=(
                maximum_local_specializations
            ),
            lower_order_factor_decompositions=(
                self.lower_order_factor_decompositions - decompositions_before
            ),
        )
        return placement_order, automata

    def compile_cuts(
        self,
        *,
        average_cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        time_limit_seconds: float | None = None,
    ) -> CoolingCompilationResult:
        """Compile and compare cut automata without searching any layout."""

        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        selected = self._select_order_and_automata(
            average_cuts,
            prefix_cuts,
            deadline=deadline,
        )
        if selected is None:
            return CoolingCompilationResult(
                proven=False,
                selection=None,
                elapsed_seconds=perf_counter() - started,
                stop_reason="cut_compilation_time_limit",
            )
        return CoolingCompilationResult(
            proven=True,
            selection=self.last_order_selection,
            elapsed_seconds=perf_counter() - started,
            stop_reason="compiled",
        )

    def compile_cuts_for_skeleton(
        self,
        skeleton: Sequence[str],
        *,
        average_cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        time_limit_seconds: float | None = None,
    ) -> CoolingCompilationResult:
        """Partially evaluate cuts for a skeleton, without running layout DP."""

        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        skeleton_tuple = tuple(skeleton)
        if len(skeleton_tuple) != self.problem.graph.size:
            raise ValueError("power skeleton has the wrong size")
        power_ids = {item.id for item in self.problem.power_components}
        if unknown := set(skeleton_tuple) - power_ids:
            raise ValueError(f"skeleton has unknown power labels: {sorted(unknown)}")
        code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        fixed_codes = {
            vertex: code_by_label[label]
            for vertex, label in enumerate(skeleton_tuple)
            if label != "empty"
        }
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        selected = self._select_specialized_order_and_automata(
            average_cuts,
            prefix_cuts,
            fixed_codes=fixed_codes,
            deadline=deadline,
        )
        if selected is None:
            return CoolingCompilationResult(
                proven=False,
                selection=None,
                elapsed_seconds=perf_counter() - started,
                stop_reason="skeleton_cut_compilation_time_limit",
            )
        return CoolingCompilationResult(
            proven=True,
            selection=self.last_order_selection,
            elapsed_seconds=perf_counter() - started,
            stop_reason="skeleton_conditioned_cuts_compiled",
        )

    def solve_joint_layouts(
        self,
        *,
        average_cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        excluded_layouts: Sequence[Sequence[str]] = (),
        enforce_complete_average_flow: bool = False,
        incumbent_lower_bound: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> ParetoFrontierDPResult:
        """Solve one full-label master without enumerating power skeletons.

        The DP key is the product of the graph separator behaviour and the
        exact future functions of all submitted cuts/no-goods.  Power,
        generated heat, inventories and monotone cut scores are Pareto
        coordinates.  Therefore prefixes merge only when every suffix has the
        same feasible continuations; dominated points can never improve a
        common suffix.

        The result is exact for the current master relaxation.  As with every
        logic-Benders master, dynamic safety still requires the deterministic
        cycle verifier; a failed representative adds a cut or exact no-good
        and rebuilds this quotient.
        """

        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )
        normalized_exclusions = tuple(tuple(layout) for layout in excluded_layouts)
        for layout in normalized_exclusions:
            if len(layout) != self.problem.graph.size:
                raise ValueError("excluded layout has the wrong size")
            if unknown := set(layout) - set(self.labels):
                raise ValueError(
                    f"excluded layout has unknown labels: {sorted(unknown)}"
                )
        selected = self._select_order_and_automata(
            () if enforce_complete_average_flow else average_cuts,
            prefix_cuts,
            excluded_layouts=normalized_exclusions,
            deadline=deadline,
        )
        if selected is None:
            return ParetoFrontierDPResult(
                proven=False,
                frontier=(),
                states_visited=0,
                raw_transitions=0,
                dominated_rejections=0,
                upper_bound_rejections=0,
                removed_points=0,
                peak_layer_points=0,
                peak_antichain_width=0,
                layer_statistics=(),
                frontier_width=min(
                    self.problem.graph.rows or self.problem.graph.size,
                    self.problem.graph.columns or self.problem.graph.size,
                ),
                elapsed_seconds=perf_counter() - started,
                stop_reason="cut_compilation_time_limit",
            )
        placement_order, selected_automata = selected
        automata = list(selected_automata)
        complete_average_automaton = None
        if enforce_complete_average_flow:
            complete_average_automaton = AverageFlowTerminalCutAutomaton(
                self.problem,
                self.heat_catalogue,
                self.labels,
                placement_order=placement_order,
            )
            automata.append(complete_average_automaton)
            score_bounds = (
                *self.last_order_selection.cut_score_chain_bounds,
                (0, complete_average_automaton.maximum_heat),
            )
            score_width = product_chain_antichain_bound(score_bounds)
            self.last_order_selection = replace(
                self.last_order_selection,
                structural_state_product_bound=(
                    self.last_order_selection.structural_state_product_bound
                    * complete_average_automaton.profile.discrete_state_bound
                ),
                distinct_factor_constraint_count=(
                    self.last_order_selection.distinct_factor_constraint_count + 1
                ),
                separate_constraint_state_bounds=(
                    *self.last_order_selection.separate_constraint_state_bounds,
                    complete_average_automaton.profile.discrete_state_bound,
                ),
                cut_score_chain_bounds=score_bounds,
                cut_score_antichain_width_bound=score_width.width,
                cut_score_antichain_width_exact=score_width.exact,
                peak_pareto_points_without_no_goods_bound=None,
                raw_transitions_without_no_goods_bound=None,
                complete_average_flow=True,
                complete_average_flow_peak_terminals=(
                    complete_average_automaton.profile.schedule.peak_live_terminals
                ),
                complete_average_flow_vector_entries=(
                    complete_average_automaton.profile.schedule.peak_cut_vector_entries
                ),
                complete_average_flow_state_bound=(
                    complete_average_automaton.profile.discrete_state_bound
                ),
            )
        no_good_automaton = None
        if normalized_exclusions:
            no_good_automaton = ExcludedLayoutsAutomaton(
                placement_order,
                self.labels,
                normalized_exclusions,
            )
            automata.append(no_good_automaton)
        self.last_order_selection = replace(
            self.last_order_selection,
            submitted_layout_no_goods=len(normalized_exclusions),
            relevant_layout_no_goods=(
                0
                if no_good_automaton is None
                else no_good_automaton.excluded_layout_count
            ),
            no_good_trie_nodes=(
                0
                if no_good_automaton is None
                else no_good_automaton.trie_node_count
            ),
        )
        remaining = None if deadline is None else max(0.0, deadline - perf_counter())
        if remaining == 0.0:
            return ParetoFrontierDPResult(
                proven=False,
                frontier=(),
                states_visited=0,
                raw_transitions=0,
                dominated_rejections=0,
                upper_bound_rejections=0,
                removed_points=0,
                peak_layer_points=0,
                peak_antichain_width=0,
                layer_statistics=(),
                frontier_width=min(
                    self.problem.graph.rows or self.problem.graph.size,
                    self.problem.graph.columns or self.problem.graph.size,
                ),
                elapsed_seconds=perf_counter() - started,
                stop_reason="time_limit",
            )
        result = RectangularParetoPowerHeatDP(
            self.full_layout_problem,
            automata=tuple(automata),
            placement_order=placement_order,
            # Every slot already has its actual component label.  The number
            # of prospective cooling slots used by the skeleton relaxation is
            # no longer a future resource and must not widen the antichain.
            track_active_slot_resource=False,
        ).solve(
            incumbent_lower_bound=incumbent_lower_bound,
            time_limit_seconds=remaining,
        )
        return replace(result, elapsed_seconds=perf_counter() - started)

    def solve(
        self,
        skeleton: Sequence[str],
        *,
        average_cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        excluded_layouts: Sequence[Sequence[str]] = (),
        time_limit_seconds: float | None = None,
    ) -> FactorizedLayoutDPResult:
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = (
            None if time_limit_seconds is None else started + time_limit_seconds
        )

        def unknown_result(stop_reason: str) -> FactorizedLayoutDPResult:
            return FactorizedLayoutDPResult(
                proven=False,
                feasible=False,
                layout=None,
                raw_transitions=0,
                equivalent_successor_merges=0,
                dominated_rejections=0,
                removed_points=0,
                peak_layer_points=0,
                peak_antichain_width=0,
                layer_statistics=(),
                elapsed_seconds=perf_counter() - started,
                stop_reason=stop_reason,
            )

        power_ids = {item.id for item in self.problem.power_components}
        skeleton_tuple = tuple(skeleton)
        if len(skeleton_tuple) != self.problem.graph.size:
            raise ValueError("power skeleton has the wrong size")
        if unknown := set(skeleton_tuple) - power_ids:
            raise ValueError(f"skeleton has unknown power labels: {sorted(unknown)}")
        normalized_exclusions = tuple(tuple(layout) for layout in excluded_layouts)
        for layout in normalized_exclusions:
            if len(layout) != self.problem.graph.size:
                raise ValueError("excluded layout has the wrong size")
            if unknown := set(layout) - set(self.labels):
                raise ValueError(
                    f"excluded layout has unknown labels: {sorted(unknown)}"
                )
        fixed = {
            vertex: label
            for vertex, label in enumerate(skeleton_tuple)
            if label != "empty"
        }
        preflight = FactorizedLayoutFeasibilityDP(
            self.placement_order,
            self.labels,
            free_labels=self.free_labels,
            fixed_labels=fixed,
            component_limits=self.component_limits,
        )
        if preflight.inventory_impossible:
            result = preflight.solve()
            return replace(result, elapsed_seconds=perf_counter() - started)
        code_by_label = {
            label: code for code, label in enumerate(self.labels)
        }
        selected = self._select_specialized_order_and_automata(
            average_cuts,
            prefix_cuts,
            fixed_codes={
                vertex: code_by_label[label]
                for vertex, label in fixed.items()
            },
            deadline=deadline,
        )
        if selected is None:
            return unknown_result("cut_compilation_time_limit")
        placement_order, selected_automata = selected
        automata = list(selected_automata)
        relevant_exclusions = tuple(
            layout
            for layout in normalized_exclusions
            if all(layout[vertex] == label for vertex, label in fixed.items())
        )
        no_good_automaton = None
        if relevant_exclusions:
            no_good_automaton = ExcludedLayoutsAutomaton(
                placement_order,
                self.labels,
                relevant_exclusions,
            )
            automata.append(no_good_automaton)
        order_selection = self.last_order_selection
        self.last_order_selection = CoolingOrderSelection(
            candidate_order_count=order_selection.candidate_order_count,
            placement_order=order_selection.placement_order,
            structural_state_product_bound=(
                order_selection.structural_state_product_bound
            ),
            submitted_cut_count=order_selection.submitted_cut_count,
            distinct_factor_constraint_count=(
                order_selection.distinct_factor_constraint_count
            ),
            raw_factor_table_entries=order_selection.raw_factor_table_entries,
            quotient_factor_table_entries=(
                order_selection.quotient_factor_table_entries
            ),
            per_variable_representations=(
                order_selection.per_variable_representations
            ),
            residual_function_representations=(
                order_selection.residual_function_representations
            ),
            submitted_layout_no_goods=len(normalized_exclusions),
            relevant_layout_no_goods=(
                0
                if no_good_automaton is None
                else no_good_automaton.excluded_layout_count
            ),
            no_good_trie_nodes=(
                0
                if no_good_automaton is None
                else no_good_automaton.trie_node_count
            ),
            joint_per_variable_representation=(
                order_selection.joint_per_variable_representation
            ),
            separate_constraint_state_bounds=(
                order_selection.separate_constraint_state_bounds
            ),
            cut_score_chain_bounds=order_selection.cut_score_chain_bounds,
            cut_score_antichain_width_bound=(
                order_selection.cut_score_antichain_width_bound
            ),
            cut_score_antichain_width_exact=(
                order_selection.cut_score_antichain_width_exact
            ),
            inventory_chain_bounds=order_selection.inventory_chain_bounds,
            resource_antichain_width_bound=(
                order_selection.resource_antichain_width_bound
            ),
            resource_antichain_width_exact=(
                order_selection.resource_antichain_width_exact
            ),
            peak_pareto_points_without_no_goods_bound=(
                order_selection.peak_pareto_points_without_no_goods_bound
            ),
            raw_transitions_without_no_goods_bound=(
                order_selection.raw_transitions_without_no_goods_bound
            ),
            fixed_inventory_infeasible=(
                order_selection.fixed_inventory_infeasible
            ),
            specialized_factor_cache_hits=(
                order_selection.specialized_factor_cache_hits
            ),
            specialized_factor_cache_misses=(
                order_selection.specialized_factor_cache_misses
            ),
            specialized_factor_cache_entries=(
                order_selection.specialized_factor_cache_entries
            ),
            specialized_factor_cache_limit=(
                order_selection.specialized_factor_cache_limit
            ),
            maximum_local_specializations_for_submitted_cuts=(
                order_selection.maximum_local_specializations_for_submitted_cuts
            ),
            lower_order_factor_decompositions=(
                order_selection.lower_order_factor_decompositions
            ),
        )
        remaining = (
            None if deadline is None else deadline - perf_counter()
        )
        if remaining is not None and remaining <= 0:
            return unknown_result("cut_compilation_time_limit")
        result = FactorizedLayoutFeasibilityDP(
            placement_order,
            self.labels,
            free_labels=self.free_labels,
            fixed_labels=fixed,
            component_limits=self.component_limits,
            automata=automata,
        ).solve(time_limit_seconds=remaining)
        return replace(result, elapsed_seconds=perf_counter() - started)
