"""Finite cut automata composable with rectangular frontier dynamic programs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Callable, Hashable, Mapping, Protocol, Sequence

from .mathematical_model import AggregatePattern, Graph, ReactorProblem


def rectangular_frontier_order(graph: Graph) -> tuple[int, ...]:
    """Return the minimum-width scan order used by the rectangular DP."""

    return rectangular_frontier_orders(graph)[0]


def rectangular_frontier_orders(graph: Graph) -> tuple[tuple[int, ...], ...]:
    """Return all four axis flips of a minimum-width rectangular scan.

    Every returned order has the same geometric separator width.  Cut values
    can nevertheless make their live residual-function products different,
    so a factorized master may cheaply compare these orders before its DP.
    """

    if graph.rows is None or graph.columns is None:
        raise ValueError("frontier order requires rectangular metadata")
    rows, columns = graph.rows, graph.columns
    result = []
    for reverse_major in (False, True):
        for reverse_minor in (False, True):
            if rows <= columns:
                majors = range(columns - 1, -1, -1) if reverse_major else range(columns)
                minors = range(rows - 1, -1, -1) if reverse_minor else range(rows)
                order = tuple(
                    row * columns + column
                    for column in majors
                    for row in minors
                )
            else:
                majors = range(rows - 1, -1, -1) if reverse_major else range(rows)
                minors = (
                    range(columns - 1, -1, -1)
                    if reverse_minor
                    else range(columns)
                )
                order = tuple(
                    row * columns + column
                    for row in majors
                    for column in minors
                )
            if order not in result:
                result.append(order)
    return tuple(result)


def factor_hypergraph_frontier_orders(
    vertices: Sequence[int],
    factors: Sequence["LocalScoreFactor"],
    *,
    allowed_codes: Sequence[int] | None = None,
    beam_width: int = 24,
    deadline: float | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Generate cut-aware orders by minimizing the live behaviour product.

    After a fixed skeleton is substituted, the remaining factor hypergraph
    can be much sparser than the original rectangle.  Raster orders are then
    needlessly constrained by geometric width.  This bounded beam search is
    over *variable orders*, not layouts: its score is the product of local
    behaviour classes on variables already placed but still referenced by an
    unfinished factor.  Any returned permutation is exact; a poor order only
    costs time and is rejected later by the exact automaton profile.
    """

    order_domain = tuple(vertices)
    if not order_domain or len(order_domain) != len(set(order_domain)):
        raise ValueError("factor-order vertices must be non-empty and unique")
    if beam_width <= 0:
        raise ValueError("factor-order beam width must be positive")
    vertex_index = {vertex: index for index, vertex in enumerate(order_domain)}
    known = set(order_domain)
    if unknown := {
        vertex
        for factor in factors
        for vertex in factor.scope
        if vertex not in known
    }:
        raise ValueError(f"factor-order scopes use unknown vertices: {sorted(unknown)}")
    code_counts = {factor.code_count for factor in factors}
    if len(code_counts) > 1:
        raise ValueError("factor-order factors use different code domains")
    code_count = next(iter(code_counts), 1)
    allowed = (
        tuple(range(code_count))
        if allowed_codes is None
        else tuple(dict.fromkeys(int(code) for code in allowed_codes))
    )
    if not allowed or (
        factors and any(not 0 <= code < code_count for code in allowed)
    ):
        raise ValueError("factor-order allowed codes are outside the domain")

    effective = tuple(
        factor for factor in factors if len(factor.scope) > 1
    )
    if not effective:
        canonical = order_domain
        reverse = tuple(reversed(canonical))
        return (canonical,) if reverse == canonical else (canonical, reverse)
    scope_masks = tuple(sum(
        1 << vertex_index[vertex] for vertex in factor.scope
    ) for factor in effective)
    projections: tuple[dict[int, tuple[int, ...]], ...] = tuple(
        {
            vertex: tuple(
                (
                    factor.projections[position][code]
                    if factor.projections
                    else code
                )
                for code in allowed
            )
            for position, vertex in enumerate(factor.scope)
        }
        for factor in effective
    )
    all_mask = (1 << len(order_domain)) - 1
    profile_cache: dict[int, tuple[int, int, int]] = {}

    def profile(prefix_mask: int) -> tuple[int, int, int]:
        cached = profile_cache.get(prefix_mask)
        if cached is not None:
            return cached
        open_indices = []
        live_mask = 0
        for factor_index, scope_mask in enumerate(scope_masks):
            placed = scope_mask & prefix_mask
            if placed and scope_mask & ~prefix_mask:
                open_indices.append(factor_index)
                live_mask |= placed
        behaviour_product = 1
        live_count = live_mask.bit_count()
        bits = live_mask
        while bits:
            lowest = bits & -bits
            position = lowest.bit_length() - 1
            vertex = order_domain[position]
            incident = tuple(
                factor_index
                for factor_index in open_indices
                if vertex in projections[factor_index]
            )
            signatures = {
                tuple(projections[factor_index][vertex][code_index]
                      for factor_index in incident)
                for code_index in range(len(allowed))
            }
            behaviour_product *= len(signatures)
            bits ^= lowest
        result = (behaviour_product, live_count, len(open_indices))
        profile_cache[prefix_mask] = result
        return result

    # (peak behaviour product, current product, live count, open factors,
    #  prefix mask, deterministic order)
    beam: list[tuple[int, int, int, int, int, tuple[int, ...]]] = [
        (1, 1, 0, 0, 0, ())
    ]
    for _depth in range(len(order_domain)):
        if deadline is not None and perf_counter() >= deadline:
            return ()
        best_by_mask: dict[
            int,
            tuple[int, int, int, int, int, tuple[int, ...]],
        ] = {}
        for peak, _current, _live, _open, prefix_mask, prefix in beam:
            remaining = all_mask ^ prefix_mask
            while remaining:
                lowest = remaining & -remaining
                position = lowest.bit_length() - 1
                following_mask = prefix_mask | lowest
                current, live, open_count = profile(following_mask)
                candidate = (
                    max(peak, current),
                    current,
                    live,
                    open_count,
                    following_mask,
                    (*prefix, order_domain[position]),
                )
                previous = best_by_mask.get(following_mask)
                if previous is None or (
                    candidate[:4], candidate[5]
                ) < (
                    previous[:4], previous[5]
                ):
                    best_by_mask[following_mask] = candidate
                remaining ^= lowest
        beam = sorted(
            best_by_mask.values(),
            key=lambda item: (item[:4], item[5]),
        )[:beam_width]
    best_order = beam[0][5]
    reverse = tuple(reversed(best_order))
    return (best_order,) if reverse == best_order else (best_order, reverse)


@dataclass(frozen=True, slots=True)
class FrontierTransitionContext:
    step: int
    vertex: int
    placed_code: int
    major: int
    minor: int
    placed_neighbours: tuple[tuple[int, int], ...]
    previous_frontier: tuple[tuple[int, int], ...]
    next_frontier: tuple[tuple[int, int], ...]
    finalized_vertex: int | None
    finalized_entry: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class FrontierAutomatonTransition:
    state: Hashable
    resources: tuple[int, ...] = ()


class FrontierConstraintAutomaton(Protocol):
    """A sufficient finite state for one family constraint/cut."""

    def initial_state(self) -> Hashable: ...

    def initial_resources(self) -> tuple[int, ...]: ...

    def advance(
        self,
        state: Hashable,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None: ...

    def accepts(
        self,
        state: Hashable,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class AggregateDegreeAutomatonState:
    remaining_counts: tuple[int, ...]
    remaining_active_cells: int


class AggregateDegreeAutomaton:
    """Compile one aggregate fuel-degree signature into a frontier automaton."""

    def __init__(self, problem: ReactorProblem, pattern: AggregatePattern) -> None:
        self.problem = problem
        self.pattern = pattern
        self.items = problem.power_components
        self.active_by_code = tuple(item.accepts_pulse for item in self.items)
        self.fuel_by_code = {
            code: item.id
            for code, item in enumerate(self.items)
            if item.rods > 0
        }
        requested = tuple(pattern.fuel_degree_counts)
        self.keys = tuple((item, degree) for item, degree, _count in requested)
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("aggregate pattern repeats a fuel-degree state")
        known = set(self.fuel_by_code.values())
        if unknown := {item for item, _degree in self.keys} - known:
            raise ValueError(f"aggregate pattern has unknown fuels: {sorted(unknown)}")
        if any(
            count < 0
            or degree < 0
            or degree > problem.graph.maximum_degree
            for _item, degree, count in requested
        ):
            raise ValueError("aggregate pattern has an invalid state")
        self.initial_counts = tuple(count for _item, _degree, count in requested)
        fuel_cells = sum(self.initial_counts)
        if not fuel_cells <= pattern.active_cells <= problem.graph.size:
            raise ValueError("aggregate pattern active-cell count is inconsistent")
        self.key_index = {key: index for index, key in enumerate(self.keys)}

    def initial_state(self) -> AggregateDegreeAutomatonState:
        return AggregateDegreeAutomatonState(
            self.initial_counts,
            self.pattern.active_cells,
        )

    def initial_resources(self) -> tuple[int, ...]:
        return ()

    def _consume(
        self,
        state: AggregateDegreeAutomatonState,
        entry: tuple[int, int],
    ) -> AggregateDegreeAutomatonState | None:
        code, degree = entry
        if not self.active_by_code[code]:
            return state
        active_remaining = state.remaining_active_cells - 1
        if active_remaining < 0:
            return None
        counts = state.remaining_counts
        fuel = self.fuel_by_code.get(code)
        if fuel is not None:
            index = self.key_index.get((fuel, degree))
            if index is None or counts[index] <= 0:
                return None
            values = list(counts)
            values[index] -= 1
            counts = tuple(values)
        return AggregateDegreeAutomatonState(counts, active_remaining)

    def advance(
        self,
        state: AggregateDegreeAutomatonState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if resources:
            raise ValueError("aggregate degree automaton has no Pareto resources")
        if context.finalized_entry is None:
            return FrontierAutomatonTransition(state)
        following = self._consume(state, context.finalized_entry)
        return (
            None
            if following is None
            else FrontierAutomatonTransition(following)
        )

    def accepts(
        self,
        state: AggregateDegreeAutomatonState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        if resources:
            raise ValueError("aggregate degree automaton has no Pareto resources")
        current: AggregateDegreeAutomatonState | None = state
        for _vertex, code, degree in final_frontier:
            if current is None:
                return False
            current = self._consume(current, (code, degree))
        return (
            current is not None
            and current.remaining_active_cells == 0
            and not any(current.remaining_counts)
        )


class LocalRewardThresholdAutomaton:
    """Require a non-negative sum of local vertex/edge rewards to reach a target.

    Rewards are accumulated exactly while below ``threshold``.  Every value at
    or above the threshold is future-equivalent and therefore represented by
    one saturated state.  Each grid edge is evaluated once, when its later
    endpoint is placed; each vertex is evaluated once, when it leaves the
    frontier (or during finalization).
    """

    def __init__(
        self,
        threshold: int,
        *,
        vertex_reward: Callable[[int, int, int], int] | None = None,
        edge_reward: Callable[[int, int, int, int], int] | None = None,
    ) -> None:
        if threshold < 0:
            raise ValueError("reward threshold must be non-negative")
        self.threshold = threshold
        self.vertex_reward = vertex_reward
        self.edge_reward = edge_reward

    def initial_state(self) -> int:
        return 0

    def initial_resources(self) -> tuple[int, ...]:
        return (0,)

    def _add(self, value: int, reward: int) -> int:
        if reward < 0:
            raise ValueError("local threshold automata require non-negative rewards")
        return min(self.threshold, value + reward)

    def advance(
        self,
        state: int,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition:
        if len(resources) != 1:
            raise ValueError("local reward automaton requires one Pareto resource")
        reward = 0
        if self.edge_reward is not None:
            reward += sum(
                self.edge_reward(
                    neighbour_vertex,
                    neighbour_code,
                    context.vertex,
                    context.placed_code,
                )
                for neighbour_vertex, neighbour_code in context.placed_neighbours
            )
        if (
            self.vertex_reward is not None
            and context.finalized_vertex is not None
            and context.finalized_entry is not None
        ):
            code, degree = context.finalized_entry
            reward += self.vertex_reward(context.finalized_vertex, code, degree)
        return FrontierAutomatonTransition(
            state,
            (self._add(resources[0], reward),),
        )

    def accepts(
        self,
        state: int,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = state
        if len(resources) != 1:
            raise ValueError("local reward automaton requires one Pareto resource")
        reward = 0
        if self.vertex_reward is not None:
            reward = sum(
                self.vertex_reward(vertex, code, degree)
                for vertex, code, degree in final_frontier
            )
        return self._add(resources[0], reward) == self.threshold


@dataclass(frozen=True, slots=True)
class LocalScoreFactor:
    """A complete finite factor table, optionally over per-variable code classes.

    ``code_count`` remains the size of the public/global label domain.  When
    ``projections`` is present, each variable first maps global codes to exact
    local behaviour classes and ``values`` stores only the Cartesian product
    of those classes.  This avoids materializing ``code_count ** arity`` when
    a cut observes only a few label features.
    """

    scope: tuple[int, ...]
    code_count: int
    values: tuple[int, ...]
    projections: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.scope or len(self.scope) != len(set(self.scope)):
            raise ValueError("factor scope must contain distinct vertices")
        if self.code_count <= 0:
            raise ValueError("factor code count must be positive")
        if self.projections and len(self.projections) != len(self.scope):
            raise ValueError("factor projections must match the factor arity")
        class_counts = []
        for projection in self.projections:
            if len(projection) != self.code_count:
                raise ValueError("factor projection does not cover the code domain")
            classes = set(projection)
            if classes != set(range(len(classes))):
                raise ValueError("factor projection classes must be contiguous")
            class_counts.append(len(classes))
        expected = 1
        for count in (
            class_counts
            if self.projections
            else (self.code_count,) * len(self.scope)
        ):
            expected *= count
        if len(self.values) != expected:
            raise ValueError("factor table does not cover its full finite domain")

    @classmethod
    def tabulate(
        cls,
        scope: Sequence[int],
        code_count: int,
        evaluator: Callable[[tuple[int, ...]], int],
    ) -> "LocalScoreFactor":
        canonical_scope = tuple(scope)
        return cls(
            canonical_scope,
            code_count,
            tuple(
                int(evaluator(codes))
                for codes in product(range(code_count), repeat=len(canonical_scope))
            ),
        )

    @classmethod
    def tabulate_quotiented(
        cls,
        scope: Sequence[int],
        code_count: int,
        signatures_by_position: Sequence[Sequence[Hashable]],
        evaluator: Callable[[tuple[int, ...]], int],
    ) -> "LocalScoreFactor":
        """Tabulate once per proven local behaviour class, not per raw label.

        The supplied signature for a position is a proof obligation: labels
        with the same signature must be interchangeable in ``evaluator`` for
        every assignment to the other positions.  Compilers derive these
        signatures directly from the label attributes read by their formula.
        """

        canonical_scope = tuple(scope)
        signatures = tuple(tuple(row) for row in signatures_by_position)
        if len(signatures) != len(canonical_scope):
            raise ValueError("factor signatures must match the factor arity")
        projections = []
        representatives = []
        for row in signatures:
            if len(row) != code_count:
                raise ValueError("factor signatures do not cover the code domain")
            class_by_signature: dict[Hashable, int] = {}
            projection = []
            class_representatives = []
            for code, signature in enumerate(row):
                factor_class = class_by_signature.get(signature)
                if factor_class is None:
                    factor_class = len(class_by_signature)
                    class_by_signature[signature] = factor_class
                    class_representatives.append(code)
                projection.append(factor_class)
            projections.append(tuple(projection))
            representatives.append(tuple(class_representatives))
        values = tuple(
            int(evaluator(tuple(
                representatives[position][factor_class]
                for position, factor_class in enumerate(classes)
            )))
            for classes in product(*(
                range(len(items)) for items in representatives
            ))
        )
        return cls(
            canonical_scope,
            code_count,
            values,
            tuple(projections),
        )

    def class_count(self, position: int) -> int:
        if not 0 <= position < len(self.scope):
            raise ValueError("factor position is outside the scope")
        return (
            len(set(self.projections[position]))
            if self.projections
            else self.code_count
        )

    def class_representatives(self, position: int) -> tuple[int, ...]:
        """Return one global label code for every local factor class."""

        if not 0 <= position < len(self.scope):
            raise ValueError("factor position is outside the scope")
        if not self.projections:
            return tuple(range(self.code_count))
        projection = self.projections[position]
        representatives = [-1] * self.class_count(position)
        for code, factor_class in enumerate(projection):
            if representatives[factor_class] < 0:
                representatives[factor_class] = code
        return tuple(representatives)

    def canonical_code(self, position: int, code: int) -> int:
        """Map a global code to its class's first deterministic representative."""

        if not 0 <= code < self.code_count:
            raise ValueError("factor assignment code is outside its domain")
        if not self.projections:
            return code
        factor_class = self.projections[position][code]
        return self.class_representatives(position)[factor_class]

    def evaluate(self, codes: Sequence[int]) -> int:
        if len(codes) != len(self.scope):
            raise ValueError("factor assignment has the wrong arity")
        index = 0
        for position, code in enumerate(codes):
            if not 0 <= code < self.code_count:
                raise ValueError("factor assignment code is outside its domain")
            factor_class = (
                self.projections[position][code]
                if self.projections
                else code
            )
            index = index * self.class_count(position) + factor_class
        return self.values[index]

    def condition(
        self,
        vertex: int,
        code: int,
    ) -> int | "LocalScoreFactor":
        """Substitute one variable, returning a constant or residual factor."""

        if vertex not in self.scope:
            raise ValueError("conditioned vertex is outside the factor scope")
        if not 0 <= code < self.code_count:
            raise ValueError("conditioned code is outside the factor domain")
        position = self.scope.index(vertex)
        if len(self.scope) == 1:
            return self.evaluate((code,))
        remaining_scope = self.scope[:position] + self.scope[position + 1:]
        remaining_positions = tuple(
            index for index in range(len(self.scope)) if index != position
        )
        representatives = tuple(
            self.class_representatives(index) for index in remaining_positions
        )
        values = []
        for remaining_classes in product(*(
            range(len(items)) for items in representatives
        )):
            assignment = [
                representatives[index][factor_class]
                for index, factor_class in enumerate(remaining_classes)
            ]
            assignment.insert(position, code)
            values.append(self.evaluate(assignment))
        return LocalScoreFactor(
            remaining_scope,
            self.code_count,
            tuple(values),
            (
                tuple(
                    self.projections[index]
                    for index in remaining_positions
                )
                if self.projections
                else ()
            ),
        )

    @property
    def minimum(self) -> int:
        return min(self.values)

    @property
    def maximum(self) -> int:
        return max(self.values)


@dataclass(frozen=True, slots=True)
class NormalizedFactorConstraint:
    """Canonical factorwise form of ``sum(factors) >= threshold``."""

    factors: tuple[LocalScoreFactor, ...]
    threshold: int


def normalize_factor_constraint(
    factors: Sequence[LocalScoreFactor],
    *,
    threshold: int = 0,
) -> NormalizedFactorConstraint:
    """Move additive factor constants into the threshold and sort the sum.

    This detects equal constraints despite factor order or separately embedded
    constants.  It deliberately does not expand factors to a union scope.
    """

    grouped: dict[
        tuple[tuple[int, ...], int, tuple[tuple[int, ...], ...]],
        list[int],
    ] = {}
    adjusted_threshold = int(threshold)
    for factor in factors:
        offset = factor.minimum
        adjusted_threshold -= offset
        values = tuple(value - offset for value in factor.values)
        if not any(values):
            continue
        key = (factor.scope, factor.code_count, factor.projections)
        accumulated = grouped.get(key)
        if accumulated is None:
            grouped[key] = list(values)
        else:
            for index, value in enumerate(values):
                accumulated[index] += value
    normalized = []
    for (scope, code_count, projections), raw_values in grouped.items():
        # The minima of separate factors need not occur at the same local
        # assignment.  After adding equal-scope tables, extract the stronger
        # shared constant once more.
        offset = min(raw_values)
        adjusted_threshold -= offset
        values = tuple(value - offset for value in raw_values)
        if any(values):
            normalized.append(LocalScoreFactor(
                scope,
                code_count,
                values,
                projections,
            ))
    normalized.sort(key=lambda factor: (
        factor.scope,
        factor.code_count,
        factor.projections,
        factor.values,
    ))
    return NormalizedFactorConstraint(
        tuple(normalized),
        adjusted_threshold,
    )


def factorwise_constraint_implies(
    stronger: NormalizedFactorConstraint,
    weaker: NormalizedFactorConstraint,
) -> bool:
    """Prove ``stronger => weaker`` by a separable slack lower bound.

    For constraints ``A(x)>=a`` and ``B(x)>=b``, implication follows when
    ``B(x)-b - (A(x)-a) >= 0`` for every assignment.  Equal table domains are
    subtracted pointwise; unmatched factors use their exact individual
    minima.  Summing those minima is a rigorous (possibly non-tight) lower
    bound even when scopes overlap.
    """

    def table_map(
        constraint: NormalizedFactorConstraint,
    ) -> dict[
        tuple[tuple[int, ...], int, tuple[tuple[int, ...], ...]],
        LocalScoreFactor,
    ]:
        return {
            (factor.scope, factor.code_count, factor.projections): factor
            for factor in constraint.factors
        }

    first = table_map(stronger)
    second = table_map(weaker)
    lower_bound = stronger.threshold - weaker.threshold
    for key in first.keys() | second.keys():
        stronger_factor = first.get(key)
        weaker_factor = second.get(key)
        if stronger_factor is None:
            lower_bound += weaker_factor.minimum
        elif weaker_factor is None:
            lower_bound -= stronger_factor.maximum
        else:
            lower_bound += min(
                right - left
                for left, right in zip(
                    stronger_factor.values,
                    weaker_factor.values,
                    strict=True,
                )
            )
    return lower_bound >= 0


def remove_factorwise_implied_constraints(
    constraints: Sequence[NormalizedFactorConstraint],
) -> tuple[NormalizedFactorConstraint, ...]:
    """Keep a deterministic antichain under proved factorwise implication."""

    retained: list[NormalizedFactorConstraint] = []
    for candidate in constraints:
        if any(
            factorwise_constraint_implies(existing, candidate)
            for existing in retained
        ):
            continue
        retained = [
            existing
            for existing in retained
            if not factorwise_constraint_implies(candidate, existing)
        ]
        retained.append(candidate)
    return tuple(retained)


def condition_factor_constraint(
    factors: Sequence[LocalScoreFactor],
    fixed_codes: Mapping[int, int],
    *,
    threshold: int = 0,
    allowed_codes: Sequence[int] | None = None,
) -> NormalizedFactorConstraint:
    """Substitute fixed labels and return an equivalent normalized inequality.

    If ``sum(factors) >= threshold`` and conditioning produces constants with
    sum ``c``, the residual constraint is ``sum(residuals) >= threshold-c``.
    This is especially valuable for fixed power skeletons: every fuel-centred
    heat-injection star loses its centre variable, while stars at free centres
    become constant over the cooling-only domain.
    """

    residuals = []
    constant = 0
    for original in factors:
        residual: int | LocalScoreFactor = original
        for vertex in tuple(original.scope):
            if vertex not in fixed_codes or isinstance(residual, int):
                continue
            if vertex not in residual.scope:
                continue
            residual = residual.condition(vertex, fixed_codes[vertex])
        if isinstance(residual, int):
            constant += residual
        else:
            residuals.append(
                residual
                if allowed_codes is None
                else restrict_local_factor_domain(residual, allowed_codes)
            )
    return normalize_factor_constraint(
        residuals,
        threshold=int(threshold) - constant,
    )


def restrict_local_factor_domain(
    factor: LocalScoreFactor,
    allowed_codes: Sequence[int],
) -> LocalScoreFactor:
    """Retabulate a factor only on a proved common remaining code domain.

    Disallowed raw codes are mapped to the first allowed class because they
    are unreachable by contract.  On every allowed assignment the returned
    factor equals the original exactly.  Existing projection classes are
    reused, so this costs the quotient table size rather than
    ``len(allowed_codes) ** arity``.
    """

    allowed = tuple(dict.fromkeys(int(code) for code in allowed_codes))
    if not allowed or any(not 0 <= code < factor.code_count for code in allowed):
        raise ValueError("restricted factor codes must be a non-empty domain subset")
    projections = []
    representatives = []
    for position in range(len(factor.scope)):
        original_projection = (
            factor.projections[position]
            if factor.projections
            else tuple(range(factor.code_count))
        )
        restricted_by_original: dict[int, int] = {}
        class_representatives = []
        projection = [0] * factor.code_count
        for code in allowed:
            original_class = original_projection[code]
            restricted_class = restricted_by_original.get(original_class)
            if restricted_class is None:
                restricted_class = len(restricted_by_original)
                restricted_by_original[original_class] = restricted_class
                class_representatives.append(code)
            projection[code] = restricted_class
        projections.append(tuple(projection))
        representatives.append(tuple(class_representatives))
    values = tuple(
        factor.evaluate(tuple(
            representatives[position][factor_class]
            for position, factor_class in enumerate(classes)
        ))
        for classes in product(*(
            range(len(items)) for items in representatives
        ))
    )
    return LocalScoreFactor(
        factor.scope,
        factor.code_count,
        values,
        tuple(projections),
    )


def mobius_decompose_local_factor(
    factor: LocalScoreFactor,
) -> tuple[int, tuple[LocalScoreFactor, ...]]:
    """Return the exact anchored interaction decomposition of one factor.

    Class zero at every position is the anchor.  The component on subset
    ``S`` is the factor restricted to ``S`` minus every proper-subset
    component.  Summing the returned constant and factors reconstructs the
    original table exactly.  Callers may keep the original factor when the
    full-scope component is nonzero, avoiding a gratuitous increase in factor
    count; when it vanishes, the decomposition proves a lower-order model.
    """

    arity = len(factor.scope)
    representatives = tuple(
        factor.class_representatives(position) for position in range(arity)
    )
    class_counts = tuple(len(items) for items in representatives)
    baseline_codes = tuple(items[0] for items in representatives)
    constant = factor.evaluate(baseline_codes)
    components: dict[int, LocalScoreFactor] = {}

    for subset_size in range(1, arity + 1):
        for mask in range(1, 1 << arity):
            if mask.bit_count() != subset_size:
                continue
            positions = tuple(
                position for position in range(arity) if mask & (1 << position)
            )
            values = []
            for subset_classes in product(*(
                range(class_counts[position]) for position in positions
            )):
                full_classes = [0] * arity
                for position, factor_class in zip(
                    positions,
                    subset_classes,
                    strict=True,
                ):
                    full_classes[position] = factor_class
                codes = tuple(
                    representatives[position][full_classes[position]]
                    for position in range(arity)
                )
                value = factor.evaluate(codes) - constant
                proper = (mask - 1) & mask
                while proper:
                    component = components.get(proper)
                    if component is not None:
                        proper_positions = tuple(
                            position
                            for position in range(arity)
                            if proper & (1 << position)
                        )
                        value -= component.evaluate(tuple(
                            representatives[position][full_classes[position]]
                            for position in proper_positions
                        ))
                    proper = (proper - 1) & mask
                values.append(value)
            if any(values):
                components[mask] = LocalScoreFactor(
                    tuple(factor.scope[position] for position in positions),
                    factor.code_count,
                    tuple(values),
                    (
                        tuple(factor.projections[position] for position in positions)
                        if factor.projections
                        else ()
                    ),
                )
    return constant, tuple(
        components[mask] for mask in sorted(components)
    )


@dataclass(frozen=True, slots=True)
class LocalFactorAutomatonState:
    live_assignments: tuple[tuple[int, int], ...] = ()
    guaranteed: bool = False


@dataclass(frozen=True, slots=True)
class LocalFactorComplexityProfile:
    factor_count: int
    maximum_scope_size: int
    peak_live_variables: int
    peak_raw_label_product: int
    peak_quotient_label_product: int
    quotient_product_by_step: tuple[int, ...]


class LocalFactorThresholdAutomaton:
    """Compile a signed bounded-scope inequality into a frontier automaton.

    The accepted inequality is ``sum(factors) >= threshold``.  Accumulated
    score is a monotone Pareto resource, not part of the discrete key.  Thus a
    larger score dominates a smaller score whenever the live factor boundary
    is identical.  Exact suffix extrema permit sound early rejection and a
    single canonical state once even the worst continuation must satisfy the
    inequality.
    """

    placement_only = True

    def __init__(
        self,
        placement_order: Sequence[int],
        factors: Sequence[LocalScoreFactor],
        *,
        threshold: int = 0,
    ) -> None:
        self.placement_order = tuple(placement_order)
        if not self.placement_order or len(self.placement_order) != len(
            set(self.placement_order)
        ):
            raise ValueError("placement order must contain distinct vertices")
        self.rank = {
            vertex: step for step, vertex in enumerate(self.placement_order)
        }
        self.factors = tuple(factors)
        self.threshold = int(threshold)
        code_counts = {factor.code_count for factor in self.factors}
        if len(code_counts) > 1:
            raise ValueError("all local factors must use the same code domain")
        known = set(self.placement_order)
        if unknown := {
            vertex
            for factor in self.factors
            for vertex in factor.scope
            if vertex not in known
        }:
            raise ValueError(f"factor scopes use unknown vertices: {sorted(unknown)}")

        buckets: list[list[LocalScoreFactor]] = [
            [] for _ in self.placement_order
        ]
        last_use: dict[int, int] = {}
        factor_final_steps = []
        for factor in self.factors:
            final_step = max(self.rank[vertex] for vertex in factor.scope)
            factor_final_steps.append(final_step)
            buckets[final_step].append(factor)
            for vertex in factor.scope:
                last_use[vertex] = max(last_use.get(vertex, -1), final_step)
        self.buckets = tuple(tuple(bucket) for bucket in buckets)
        self.factor_final_steps = tuple(factor_final_steps)
        self.last_use = last_use

        bucket_minimum = [sum(factor.minimum for factor in bucket) for bucket in buckets]
        bucket_maximum = [sum(factor.maximum for factor in bucket) for bucket in buckets]
        suffix_minimum = [0] * len(self.placement_order)
        suffix_maximum = [0] * len(self.placement_order)
        running_minimum = running_maximum = 0
        for step in range(len(self.placement_order) - 1, -1, -1):
            suffix_minimum[step] = running_minimum
            suffix_maximum[step] = running_maximum
            running_minimum += bucket_minimum[step]
            running_maximum += bucket_maximum[step]
        self.remaining_minimum_after = tuple(suffix_minimum)
        self.remaining_maximum_after = tuple(suffix_maximum)
        self.total_minimum = running_minimum
        self.total_maximum = running_maximum
        self._guaranteed = LocalFactorAutomatonState((), True)
        self._factor_code_classes = self._build_factor_code_classes()
        self._canonical_code_after = self._build_future_code_quotients()

    def _factor_slice_signature(
        self,
        factor: LocalScoreFactor,
        vertex: int,
        fixed_code: int,
    ) -> tuple[int, ...]:
        position = factor.scope.index(vertex)
        values = []
        other_positions = tuple(
            index for index in range(len(factor.scope)) if index != position
        )
        representatives = tuple(
            factor.class_representatives(index) for index in other_positions
        )
        for other_classes in product(*(
            range(len(items)) for items in representatives
        )):
            assignment = []
            cursor = 0
            for factor_position in range(len(factor.scope)):
                if factor_position == position:
                    assignment.append(fixed_code)
                else:
                    assignment.append(
                        representatives[cursor][other_classes[cursor]]
                    )
                    cursor += 1
            values.append(factor.evaluate(assignment))
        return tuple(values)

    def _build_factor_code_classes(
        self,
    ) -> tuple[dict[int, tuple[int, ...]], ...]:
        """Compute each factor's exact per-variable functional quotient once."""

        result = []
        for factor in self.factors:
            by_vertex: dict[int, tuple[int, ...]] = {}
            for position, vertex in enumerate(factor.scope):
                # ``factor.projections[position]`` is already a proved exact
                # quotient for this variable.  Two raw labels in one supplied
                # class have identical table slices, so recomputing that slice
                # once per raw label is pure duplicate work.  We still compare
                # the distinct supplied classes below because a fixed cut can
                # make several of them functionally identical (for example,
                # multiply an observed attribute by zero).
                projection = (
                    factor.projections[position]
                    if factor.projections
                    else tuple(range(factor.code_count))
                )
                representatives = factor.class_representatives(position)
                class_by_slice: dict[tuple[int, ...], int] = {}
                class_by_projection = []
                for code in representatives:
                    signature = self._factor_slice_signature(
                        factor,
                        vertex,
                        code,
                    )
                    factor_class = class_by_slice.setdefault(
                        signature,
                        len(class_by_slice),
                    )
                    class_by_projection.append(factor_class)
                by_vertex[vertex] = tuple(
                    class_by_projection[projected_class]
                    for projected_class in projection
                )
            result.append(by_vertex)
        return tuple(result)

    def _build_future_code_quotients(
        self,
    ) -> tuple[dict[int, tuple[int, ...]], ...]:
        """Minimize live label codes by their remaining factor behaviour."""

        if not self.factors:
            return tuple({} for _step in self.placement_order)
        code_count = self.factors[0].code_count
        result = []
        for step in range(len(self.placement_order)):
            mappings: dict[int, tuple[int, ...]] = {}
            for vertex in self.placement_order:
                if (
                    self.rank[vertex] > step
                    or self.last_use.get(vertex, -1) <= step
                ):
                    continue
                future_indices = tuple(
                    factor_index
                    for factor_index, (factor, final_step) in enumerate(zip(
                        self.factors,
                        self.factor_final_steps,
                        strict=True,
                    ))
                    if final_step > step and vertex in factor.scope
                )
                if not future_indices:
                    continue
                representative_by_signature: dict[tuple[int, ...], int] = {}
                canonical = []
                for code in range(code_count):
                    signature = tuple(
                        self._factor_code_classes[factor_index][vertex][code]
                        for factor_index in future_indices
                    )
                    representative = representative_by_signature.setdefault(
                        signature,
                        code,
                    )
                    canonical.append(representative)
                mappings[vertex] = tuple(canonical)
            result.append(mappings)
        return tuple(result)

    def canonical_live_codes(
        self,
        step: int,
        vertex: int,
    ) -> tuple[int, ...] | None:
        """Expose the proven label quotient used after one scan step."""

        if not 0 <= step < len(self.placement_order):
            raise ValueError("step is outside the placement order")
        return self._canonical_code_after[step].get(vertex)

    def complexity_profile(
        self,
        *,
        allowed_codes: Sequence[int] | None = None,
    ) -> LocalFactorComplexityProfile:
        """Return a structural bound, optionally for one remaining domain.

        ``allowed_codes`` is sound after every other label has been
        conditioned away.  In the cooling master this means every remaining
        vertex is a free cooling slot; fixed fuel vertices have already been
        substituted exactly.
        """

        code_count = self.factors[0].code_count if self.factors else 1
        allowed = (
            tuple(range(code_count))
            if allowed_codes is None
            else tuple(dict.fromkeys(int(code) for code in allowed_codes))
        )
        if not allowed or any(not 0 <= code < code_count for code in allowed):
            raise ValueError("allowed factor codes must be a non-empty domain subset")
        quotient_products = []
        peak_live = peak_raw = peak_quotient = 0
        for mappings in self._canonical_code_after:
            live = len(mappings)
            raw = len(allowed)**live
            quotient = 1
            for canonical in mappings.values():
                quotient *= len({canonical[code] for code in allowed})
            quotient_products.append(quotient)
            peak_live = max(peak_live, live)
            peak_raw = max(peak_raw, raw)
            peak_quotient = max(peak_quotient, quotient)
        return LocalFactorComplexityProfile(
            factor_count=len(self.factors),
            maximum_scope_size=max(
                (len(factor.scope) for factor in self.factors),
                default=0,
            ),
            peak_live_variables=peak_live,
            peak_raw_label_product=peak_raw,
            peak_quotient_label_product=peak_quotient,
            quotient_product_by_step=tuple(quotient_products),
        )

    def initial_state(self) -> LocalFactorAutomatonState:
        return (
            self._guaranteed
            if self.total_minimum >= self.threshold
            else LocalFactorAutomatonState()
        )

    def initial_resources(self) -> tuple[int, ...]:
        return (0,)

    def pareto_resource_chain_bounds(
        self,
    ) -> tuple[tuple[int, int], ...] | None:
        """Bound the active cut score when all future increments are monotone."""

        if any(factor.minimum < 0 for factor in self.factors):
            return None
        return ((0, max(0, self.threshold - 1)),)

    def advance(
        self,
        state: LocalFactorAutomatonState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if len(resources) != 1:
            raise ValueError("local factor automaton requires one Pareto resource")
        if context.step >= len(self.placement_order):
            raise ValueError("frontier step is outside the factor placement order")
        if context.vertex != self.placement_order[context.step]:
            raise ValueError("frontier scan order differs from factor placement order")
        if state.guaranteed:
            return FrontierAutomatonTransition(self._guaranteed, (0,))

        assignments = dict(state.live_assignments)
        assignments[context.vertex] = context.placed_code
        reward = 0
        for factor in self.buckets[context.step]:
            try:
                codes = tuple(assignments[vertex] for vertex in factor.scope)
            except KeyError as error:  # pragma: no cover - constructor invariant
                raise AssertionError("factor boundary forgot a live variable") from error
            reward += factor.evaluate(codes)
        score = resources[0] + reward
        maximum_remaining = self.remaining_maximum_after[context.step]
        if score + maximum_remaining < self.threshold:
            return None
        minimum_remaining = self.remaining_minimum_after[context.step]
        if score + minimum_remaining >= self.threshold:
            return FrontierAutomatonTransition(self._guaranteed, (0,))

        live_values = []
        for vertex, code in assignments.items():
            if self.last_use.get(vertex, -1) <= context.step:
                continue
            canonical = self._canonical_code_after[context.step].get(vertex)
            if canonical is None:  # pragma: no cover - last-use invariant
                raise AssertionError("live factor variable has no future label quotient")
            live_values.append((vertex, canonical[code]))
        live = tuple(sorted(live_values))
        return FrontierAutomatonTransition(
            LocalFactorAutomatonState(live, False),
            (score,),
        )

    def accepts(
        self,
        state: LocalFactorAutomatonState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = final_frontier
        if len(resources) != 1:
            raise ValueError("local factor automaton requires one Pareto resource")
        return state.guaranteed or resources[0] >= self.threshold


class JointLocalFactorThresholdAutomaton(LocalFactorThresholdAutomaton):
    """Intersect several factor inequalities over one shared boundary state.

    A tuple of separately compiled automata has a formal state bound equal to
    the product of their individual bounds.  That product counts impossible
    combinations: every constraint observes the *same* placed label at a live
    vertex.  This automaton stores that label only once, quotiented by its
    joint future slices across all constraints.  One Pareto resource per
    inequality retains the independent accumulated scores.

    The representation is exact.  Equality of joint future-slice signatures
    is precisely continuation equivalence for the factor system, while a
    componentwise larger score vector dominates a smaller one.
    """

    placement_only = True

    def __init__(
        self,
        placement_order: Sequence[int],
        constraints: Sequence[NormalizedFactorConstraint],
    ) -> None:
        self.placement_order = tuple(placement_order)
        if not self.placement_order or len(self.placement_order) != len(
            set(self.placement_order)
        ):
            raise ValueError("placement order must contain distinct vertices")
        self.rank = {
            vertex: step for step, vertex in enumerate(self.placement_order)
        }
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("joint factor automaton requires a constraint")
        self.thresholds = tuple(item.threshold for item in self.constraints)

        tagged_factors = tuple(
            (constraint_index, factor)
            for constraint_index, constraint in enumerate(self.constraints)
            for factor in constraint.factors
        )
        self.factor_constraint_indices = tuple(
            constraint_index for constraint_index, _factor in tagged_factors
        )
        self.factors = tuple(factor for _index, factor in tagged_factors)
        self.monotone_scores = all(
            factor.minimum >= 0 for factor in self.factors
        )
        code_counts = {factor.code_count for factor in self.factors}
        if len(code_counts) > 1:
            raise ValueError("all joint local factors must use one code domain")
        known = set(self.placement_order)
        if unknown := {
            vertex
            for factor in self.factors
            for vertex in factor.scope
            if vertex not in known
        }:
            raise ValueError(f"factor scopes use unknown vertices: {sorted(unknown)}")

        buckets: list[list[tuple[int, LocalScoreFactor]]] = [
            [] for _ in self.placement_order
        ]
        last_use: dict[int, int] = {}
        factor_final_steps = []
        for constraint_index, factor in tagged_factors:
            final_step = max(self.rank[vertex] for vertex in factor.scope)
            factor_final_steps.append(final_step)
            buckets[final_step].append((constraint_index, factor))
            for vertex in factor.scope:
                last_use[vertex] = max(last_use.get(vertex, -1), final_step)
        self.joint_buckets = tuple(tuple(bucket) for bucket in buckets)
        # The inherited quotient builders need these three attributes.
        self.factor_final_steps = tuple(factor_final_steps)
        self.last_use = last_use

        width = len(self.constraints)
        suffix_minimum: list[tuple[int, ...]] = [(0,) * width] * len(
            self.placement_order
        )
        suffix_maximum: list[tuple[int, ...]] = [(0,) * width] * len(
            self.placement_order
        )
        running_minimum = [0] * width
        running_maximum = [0] * width
        for step in range(len(self.placement_order) - 1, -1, -1):
            suffix_minimum[step] = tuple(running_minimum)
            suffix_maximum[step] = tuple(running_maximum)
            for constraint_index, factor in self.joint_buckets[step]:
                running_minimum[constraint_index] += factor.minimum
                running_maximum[constraint_index] += factor.maximum
        self.remaining_minimum_after = tuple(suffix_minimum)
        self.remaining_maximum_after = tuple(suffix_maximum)
        self.total_minimum_by_constraint = tuple(running_minimum)
        self.total_maximum_by_constraint = tuple(running_maximum)
        self._factor_code_classes = self._build_factor_code_classes()
        self._canonical_code_after = self._build_future_code_quotients()

    def initial_state(self) -> LocalFactorAutomatonState:
        return LocalFactorAutomatonState()

    def initial_resources(self) -> tuple[int, ...]:
        return (0,) * len(self.constraints)

    def pareto_resource_chain_bounds(
        self,
    ) -> tuple[tuple[int, int], ...] | None:
        if not self.monotone_scores:
            return None
        # Joint constraints do not create a separate guaranteed state for each
        # coordinate, so a satisfied score is retained at its threshold cap.
        return tuple((0, max(0, threshold)) for threshold in self.thresholds)

    def advance(
        self,
        state: LocalFactorAutomatonState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if len(resources) != len(self.constraints):
            raise ValueError("joint factor automaton has one score per constraint")
        if context.step >= len(self.placement_order):
            raise ValueError("frontier step is outside the factor placement order")
        if context.vertex != self.placement_order[context.step]:
            raise ValueError("frontier scan order differs from factor placement order")

        assignments = dict(state.live_assignments)
        assignments[context.vertex] = context.placed_code
        scores = list(resources)
        for constraint_index, factor in self.joint_buckets[context.step]:
            try:
                codes = tuple(assignments[vertex] for vertex in factor.scope)
            except KeyError as error:  # pragma: no cover - constructor invariant
                raise AssertionError("joint factor boundary forgot a live variable") from error
            scores[constraint_index] += factor.evaluate(codes)

        for index, threshold in enumerate(self.thresholds):
            if (
                scores[index]
                + self.remaining_maximum_after[context.step][index]
                < threshold
            ):
                return None
            # Every normalized factor is non-negative.  Once this coordinate
            # reaches its threshold, larger values have exactly the same set
            # of feasible suffixes and are one canonical resource value.
            if self.monotone_scores:
                scores[index] = min(scores[index], max(0, threshold))

        live_values = []
        for vertex, code in assignments.items():
            if self.last_use.get(vertex, -1) <= context.step:
                continue
            canonical = self._canonical_code_after[context.step].get(vertex)
            if canonical is None:  # pragma: no cover - last-use invariant
                raise AssertionError("live joint variable has no future label quotient")
            live_values.append((vertex, canonical[code]))
        return FrontierAutomatonTransition(
            LocalFactorAutomatonState(tuple(sorted(live_values))),
            tuple(scores),
        )

    def accepts(
        self,
        state: LocalFactorAutomatonState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = state, final_frontier
        if len(resources) != len(self.constraints):
            raise ValueError("joint factor automaton has one score per constraint")
        return all(
            score >= threshold
            for score, threshold in zip(resources, self.thresholds, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ResidualFactorAutomatonState:
    restricted_factors: tuple[tuple[int, int], ...] = ()
    guaranteed: bool = False


@dataclass(frozen=True, slots=True)
class ResidualFactorComplexityProfile:
    interned_restrictions: int
    cached_condition_transitions: int
    condition_cache_hits: int


@dataclass(frozen=True, slots=True)
class ResidualFactorStructuralProfile:
    peak_residual_product: int
    residual_product_by_step: tuple[int, ...]
    interned_restrictions: int
    cached_condition_transitions: int
    complete: bool


@dataclass(frozen=True, slots=True)
class FactorAutomatonSelection:
    selected_representation: str
    raw_factor_table_entries: int
    quotient_factor_table_entries: int
    assignment_peak_product: int
    residual_peak_product: int
    assignment_peak_live_variables: int
    residual_interned_restrictions: int
    residual_profile_complete: bool = True
    assignment_state_bound_including_guaranteed: int = 1
    residual_state_bound_including_guaranteed: int = 1
    selected_state_bound_including_guaranteed: int = 1


class ResidualFactorThresholdAutomaton:
    """Represent a prefix by its exact conditioned future factor functions.

    Unlike :class:`LocalFactorThresholdAutomaton`, this representation stores
    no past label assignment.  Each touched but unfinished factor is replaced
    by its residual table after substituting the prefix labels.  Equal residual
    tables are interned, so different *joint* boundary assignments merge when
    they induce identical future functions.  Fully conditioned constants move
    into one monotone Pareto score resource.
    """

    placement_only = True

    def __init__(
        self,
        placement_order: Sequence[int],
        factors: Sequence[LocalScoreFactor],
        *,
        threshold: int = 0,
    ) -> None:
        self.placement_order = tuple(placement_order)
        if not self.placement_order or len(self.placement_order) != len(
            set(self.placement_order)
        ):
            raise ValueError("placement order must contain distinct vertices")
        self.rank = {
            vertex: step for step, vertex in enumerate(self.placement_order)
        }
        self.factors = tuple(factors)
        self.threshold = int(threshold)
        code_counts = {factor.code_count for factor in self.factors}
        if len(code_counts) > 1:
            raise ValueError("all residual factors must use the same code domain")
        known = set(self.placement_order)
        if unknown := {
            vertex
            for factor in self.factors
            for vertex in factor.scope
            if vertex not in known
        }:
            raise ValueError(f"factor scopes use unknown vertices: {sorted(unknown)}")

        factors_by_vertex: dict[int, list[int]] = {
            vertex: [] for vertex in self.placement_order
        }
        earliest_steps = []
        for factor_index, factor in enumerate(self.factors):
            earliest_steps.append(min(self.rank[vertex] for vertex in factor.scope))
            for vertex in factor.scope:
                factors_by_vertex[vertex].append(factor_index)
        self.factors_by_vertex = {
            vertex: tuple(indices)
            for vertex, indices in factors_by_vertex.items()
        }
        self.earliest_steps = tuple(earliest_steps)

        self._restriction_ids: dict[
            tuple[
                tuple[int, ...],
                int,
                tuple[int, ...],
                tuple[tuple[int, ...], ...],
            ],
            int,
        ] = {}
        self._restrictions: list[LocalScoreFactor] = []
        self._condition_cache: dict[
            tuple[int, int, int],
            tuple[int | None, int],
        ] = {}
        self.condition_cache_hits = 0
        full_restrictions = tuple(
            self._intern_normalized(factor) for factor in self.factors
        )
        self.full_restriction_ids = tuple(
            restriction_id for restriction_id, _offset in full_restrictions
        )
        self.initial_score_offset = sum(
            offset for _restriction_id, offset in full_restrictions
        )

        unstarted_minimum = []
        unstarted_maximum = []
        for step in range(len(self.placement_order)):
            unstarted = tuple(
                factor
                for factor, earliest in zip(
                    self.factors,
                    self.earliest_steps,
                    strict=True,
                )
                if earliest > step
            )
            # Every factor minimum is already in ``initial_score_offset``.
            # Unstarted normalized functions therefore range from zero to
            # ``maximum - minimum``.
            unstarted_minimum.append(0)
            unstarted_maximum.append(sum(
                factor.maximum - factor.minimum for factor in unstarted
            ))
        self.unstarted_minimum_after = tuple(unstarted_minimum)
        self.unstarted_maximum_after = tuple(unstarted_maximum)
        self.total_minimum = sum(factor.minimum for factor in self.factors)
        self.total_maximum = sum(factor.maximum for factor in self.factors)
        self._guaranteed = ResidualFactorAutomatonState((), True)

    def _intern(self, factor: LocalScoreFactor) -> int:
        key = (
            factor.scope,
            factor.code_count,
            factor.values,
            factor.projections,
        )
        existing = self._restriction_ids.get(key)
        if existing is not None:
            return existing
        restriction_id = len(self._restrictions)
        self._restriction_ids[key] = restriction_id
        self._restrictions.append(factor)
        return restriction_id

    def _intern_normalized(
        self,
        factor: LocalScoreFactor,
    ) -> tuple[int, int]:
        """Intern a residual function modulo an additive score constant."""

        offset = factor.minimum
        normalized = (
            factor
            if offset == 0
            else LocalScoreFactor(
                factor.scope,
                factor.code_count,
                tuple(value - offset for value in factor.values),
                factor.projections,
            )
        )
        return self._intern(normalized), offset

    def _condition(
        self,
        restriction_id: int,
        vertex: int,
        code: int,
    ) -> tuple[int | None, int]:
        restriction = self._restrictions[restriction_id]
        position = restriction.scope.index(vertex)
        canonical_code = restriction.canonical_code(position, code)
        key = (restriction_id, vertex, canonical_code)
        cached = self._condition_cache.get(key)
        if cached is not None:
            self.condition_cache_hits += 1
            return cached
        conditioned = restriction.condition(vertex, canonical_code)
        result = (
            (None, conditioned)
            if isinstance(conditioned, int)
            else self._intern_normalized(conditioned)
        )
        self._condition_cache[key] = result
        return result

    def initial_state(self) -> ResidualFactorAutomatonState:
        return (
            self._guaranteed
            if self.total_minimum >= self.threshold
            else ResidualFactorAutomatonState()
        )

    def initial_resources(self) -> tuple[int, ...]:
        return (self.initial_score_offset,)

    def pareto_resource_chain_bounds(
        self,
    ) -> tuple[tuple[int, int], ...] | None:
        if any(factor.minimum < 0 for factor in self.factors):
            return None
        return ((0, max(0, self.threshold - 1)),)

    def advance(
        self,
        state: ResidualFactorAutomatonState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if len(resources) != 1:
            raise ValueError("residual factor automaton requires one Pareto resource")
        if context.step >= len(self.placement_order):
            raise ValueError("frontier step is outside the residual placement order")
        if context.vertex != self.placement_order[context.step]:
            raise ValueError("frontier scan order differs from residual placement order")
        if state.guaranteed:
            return FrontierAutomatonTransition(self._guaranteed, (0,))
        if context.step == 0 and self.total_maximum < self.threshold:
            return None

        active = dict(state.restricted_factors)
        reward = 0
        for factor_index in self.factors_by_vertex[context.vertex]:
            restriction_id = active.pop(factor_index, None)
            if restriction_id is None:
                if self.earliest_steps[factor_index] != context.step:
                    raise AssertionError("residual factor disappeared before completion")
                restriction_id = self.full_restriction_ids[factor_index]
            following_id, constant = self._condition(
                restriction_id,
                context.vertex,
                context.placed_code,
            )
            reward += constant
            if following_id is not None:
                active[factor_index] = following_id

        score = resources[0] + reward
        minimum_remaining = self.unstarted_minimum_after[context.step]
        maximum_remaining = self.unstarted_maximum_after[context.step]
        for restriction_id in active.values():
            restriction = self._restrictions[restriction_id]
            minimum_remaining += restriction.minimum
            maximum_remaining += restriction.maximum
        if score + maximum_remaining < self.threshold:
            return None
        if score + minimum_remaining >= self.threshold:
            return FrontierAutomatonTransition(self._guaranteed, (0,))
        return FrontierAutomatonTransition(
            ResidualFactorAutomatonState(tuple(sorted(active.items())), False),
            (score,),
        )

    def accepts(
        self,
        state: ResidualFactorAutomatonState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = final_frontier
        if len(resources) != 1:
            raise ValueError("residual factor automaton requires one Pareto resource")
        return state.guaranteed or (
            not state.restricted_factors and resources[0] >= self.threshold
        )

    def complexity_profile(self) -> ResidualFactorComplexityProfile:
        return ResidualFactorComplexityProfile(
            interned_restrictions=len(self._restrictions),
            cached_condition_transitions=len(self._condition_cache),
            condition_cache_hits=self.condition_cache_hits,
        )

    def structural_profile(
        self,
        *,
        cutoff: int | None = None,
        allowed_codes: Sequence[int] | None = None,
    ) -> ResidualFactorStructuralProfile:
        """Compile residual MDDs, stopping once they cannot beat ``cutoff``."""

        if cutoff is not None and cutoff <= 0:
            raise ValueError("residual structural cutoff must be positive")
        code_count = self.factors[0].code_count if self.factors else 1
        allowed = (
            tuple(range(code_count))
            if allowed_codes is None
            else tuple(dict.fromkeys(int(code) for code in allowed_codes))
        )
        if not allowed or any(not 0 <= code < code_count for code in allowed):
            raise ValueError("allowed residual codes must be a non-empty domain subset")

        active: dict[int, set[int]] = {}
        products = []
        for step, vertex in enumerate(self.placement_order):
            for factor_index in self.factors_by_vertex[vertex]:
                restrictions = active.pop(factor_index, None)
                if restrictions is None:
                    if self.earliest_steps[factor_index] != step:
                        raise AssertionError("structural residual factor disappeared")
                    restrictions = {self.full_restriction_ids[factor_index]}
                following = set()
                for restriction_id in restrictions:
                    restriction = self._restrictions[restriction_id]
                    position = restriction.scope.index(vertex)
                    canonical_codes = tuple(dict.fromkeys(
                        restriction.canonical_code(position, code)
                        for code in allowed
                    ))
                    for code in canonical_codes:
                        following_id, _constant = self._condition(
                            restriction_id,
                            vertex,
                            code,
                        )
                        if following_id is not None:
                            following.add(following_id)
                if following:
                    active[factor_index] = following
            product_bound = 1
            for restrictions in active.values():
                product_bound *= len(restrictions)
            products.append(product_bound)
            if cutoff is not None and product_bound >= cutoff:
                return ResidualFactorStructuralProfile(
                    peak_residual_product=max(products),
                    residual_product_by_step=tuple(products),
                    interned_restrictions=len(self._restrictions),
                    cached_condition_transitions=len(self._condition_cache),
                    complete=False,
                )
        return ResidualFactorStructuralProfile(
            peak_residual_product=max(products, default=1),
            residual_product_by_step=tuple(products),
            interned_restrictions=len(self._restrictions),
            cached_condition_transitions=len(self._condition_cache),
            complete=True,
        )


@dataclass(frozen=True, slots=True)
class ExcludedLayoutsAutomatonState:
    trie_node: int | None


class ExcludedLayoutsAutomaton:
    """Compile any number of exact layout no-goods into one prefix automaton."""

    placement_only = True

    def __init__(
        self,
        placement_order: Sequence[int],
        label_domain: Sequence[str],
        excluded_layouts: Sequence[Sequence[str]],
    ) -> None:
        self.placement_order = tuple(placement_order)
        if not self.placement_order or len(self.placement_order) != len(
            set(self.placement_order)
        ):
            raise ValueError("placement order must contain distinct vertices")
        labels = tuple(label_domain)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("no-good label domain must be non-empty and unique")
        code_by_label = {label: code for code, label in enumerate(labels)}
        encoded = []
        for layout in excluded_layouts:
            values = tuple(layout)
            if len(values) != len(self.placement_order):
                raise ValueError("excluded layout has the wrong size")
            if unknown := set(values) - code_by_label.keys():
                raise ValueError(f"excluded layout has unknown labels: {sorted(unknown)}")
            encoded.append(tuple(code_by_label[label] for label in values))

        children: list[dict[int, int]] = [{}]
        terminal_nodes = set()
        for codes in encoded:
            node = 0
            for vertex in self.placement_order:
                code = codes[vertex]
                following = children[node].get(code)
                if following is None:
                    following = len(children)
                    children[node][code] = following
                    children.append({})
                node = following
            terminal_nodes.add(node)
        self.children = tuple(children)
        self.terminal_nodes = frozenset(terminal_nodes)
        self.excluded_layout_count = len(terminal_nodes)

    @property
    def trie_node_count(self) -> int:
        return len(self.children)

    @property
    def trie_edge_count(self) -> int:
        return sum(len(children) for children in self.children)

    def initial_state(self) -> ExcludedLayoutsAutomatonState:
        return ExcludedLayoutsAutomatonState(0)

    def initial_resources(self) -> tuple[int, ...]:
        return ()

    def advance(
        self,
        state: ExcludedLayoutsAutomatonState,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if resources:
            raise ValueError("excluded-layout automaton has no Pareto resources")
        if context.step >= len(self.placement_order):
            raise ValueError("frontier step is outside the no-good placement order")
        if context.vertex != self.placement_order[context.step]:
            raise ValueError("frontier scan order differs from no-good placement order")
        following = (
            None
            if state.trie_node is None
            else self.children[state.trie_node].get(context.placed_code)
        )
        if (
            context.step + 1 == len(self.placement_order)
            and following in self.terminal_nodes
        ):
            return None
        return FrontierAutomatonTransition(
            ExcludedLayoutsAutomatonState(following),
        )

    def accepts(
        self,
        state: ExcludedLayoutsAutomatonState,
        resources: tuple[int, ...],
        final_frontier: Sequence[tuple[int, int, int]],
    ) -> bool:
        _ = final_frontier
        if resources:
            raise ValueError("excluded-layout automaton has no Pareto resources")
        return state.trie_node not in self.terminal_nodes


def select_factor_automaton(
    placement_order: Sequence[int],
    factors: Sequence[LocalScoreFactor],
    *,
    threshold: int = 0,
    allowed_codes: Sequence[int] | None = None,
) -> tuple[
    LocalFactorThresholdAutomaton | ResidualFactorThresholdAutomaton,
    FactorAutomatonSelection,
]:
    """Choose the exact representation with the smaller structural state bound."""

    assignment = LocalFactorThresholdAutomaton(
        placement_order,
        factors,
        threshold=threshold,
    )
    assignment_profile = assignment.complexity_profile(
        allowed_codes=allowed_codes,
    )
    assignment_bound = assignment_profile.peak_quotient_label_product
    guaranteed_possible = int(assignment.total_maximum >= threshold)
    active_possible = assignment.total_minimum < threshold
    assignment_inclusive_bound = max(
        1,
        (assignment_bound if active_possible else 0) + guaranteed_possible,
    )
    if assignment_bound == 1:
        residual = None
        residual_peak = 1
        residual_interned = 0
        residual_complete = False
    else:
        residual = ResidualFactorThresholdAutomaton(
            placement_order,
            factors,
            threshold=threshold,
        )
        residual_profile = residual.structural_profile(
            cutoff=assignment_bound,
            allowed_codes=allowed_codes,
        )
        residual_peak = residual_profile.peak_residual_product
        residual_interned = residual_profile.interned_restrictions
        residual_complete = residual_profile.complete
    residual_inclusive_bound = max(
        1,
        (residual_peak if active_possible else 0) + guaranteed_possible,
    )
    if residual is not None and (
        residual_inclusive_bound < assignment_inclusive_bound
        or (
            residual_inclusive_bound == assignment_inclusive_bound
            and residual_peak < assignment_bound
        )
    ):
        selected = residual
        representation = "conditioned_residual_functions"
        selected_inclusive_bound = residual_inclusive_bound
    else:
        selected = assignment
        representation = "per_variable_function_quotient"
        selected_inclusive_bound = assignment_inclusive_bound
    return selected, FactorAutomatonSelection(
        selected_representation=representation,
        raw_factor_table_entries=sum(
            factor.code_count ** len(factor.scope) for factor in factors
        ),
        quotient_factor_table_entries=sum(
            len(factor.values) for factor in factors
        ),
        assignment_peak_product=assignment_bound,
        residual_peak_product=residual_peak,
        assignment_peak_live_variables=assignment_profile.peak_live_variables,
        residual_interned_restrictions=residual_interned,
        residual_profile_complete=residual_complete,
        assignment_state_bound_including_guaranteed=assignment_inclusive_bound,
        residual_state_bound_including_guaranteed=residual_inclusive_bound,
        selected_state_bound_including_guaranteed=selected_inclusive_bound,
    )
