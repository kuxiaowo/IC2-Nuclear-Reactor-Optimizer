"""Consume a symbolic safe-parameter region as an exact sequential MDD cut."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

from .robdd import ROBDDManager
from .robdd_bitvector import unsigned_at_least_constant
from .frontier_automata import (
    FrontierAutomatonTransition,
    FrontierTransitionContext,
)


@dataclass(slots=True)
class ROBDDSequentialParameterCut:
    """A BDD region viewed as a multi-valued prefix automaton.

    Each transition fixes all binary bits of one layout variable.  Restriction
    returns the exact residual Boolean function, so identical residuals merge
    automatically by canonical node id.  ``completion_count`` counts the
    entire remaining accepted domain and zero is an exact branch rejection.
    """

    manager: ROBDDManager
    accepted_root: int
    variable_groups: tuple[tuple[Hashable, ...], ...]
    radices: tuple[int, ...]
    transition_cache: dict[tuple[int, int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variable_groups or len(self.variable_groups) != len(self.radices):
            raise ValueError("symbolic cut groups/radices are empty or mismatched")
        flat = tuple(variable for group in self.variable_groups for variable in group)
        if len(flat) != len(set(flat)) or not set(flat) <= set(self.manager.variables):
            raise ValueError("symbolic cut variables are duplicate or unknown")
        if any(
            radix <= 0
            or (not group and radix != 1)
            or (group and radix > 1 << len(group))
            for group, radix in zip(self.variable_groups, self.radices, strict=True)
        ):
            raise ValueError("symbolic cut radix does not fit its bit group")
        valid = 1
        for group, radix in zip(self.variable_groups, self.radices, strict=True):
            if not group:
                continue
            bits = tuple(self.manager.variable(variable) for variable in group)
            valid = self.manager.apply(
                "and",
                valid,
                self.manager.negate(unsigned_at_least_constant(
                    self.manager,
                    bits,
                    radix,
                )),
            )
        self.accepted_root = self.manager.apply("and", self.accepted_root, valid)
        if not self.manager.support(self.accepted_root) <= set(flat):
            raise ValueError("symbolic cut root depends on variables outside its groups")

    @property
    def initial_state(self) -> int:
        return self.accepted_root

    def transition(self, step: int, state: int, code: int) -> int:
        if not 0 <= step < len(self.variable_groups):
            raise ValueError("symbolic cut step is outside its groups")
        if not 0 <= code < self.radices[step]:
            return 0
        key = (step, state, code)
        found = self.transition_cache.get(key)
        if found is not None:
            return found
        assignment = {
            variable: bool(code >> bit & 1)
            for bit, variable in enumerate(self.variable_groups[step])
        }
        result = self.manager.restrict(state, assignment)
        self.transition_cache[key] = result
        return result

    def completion_count(self, step: int, state: int) -> int:
        """Count valid accepted suffix codes after ``step`` consumed groups."""

        if not 0 <= step <= len(self.variable_groups):
            raise ValueError("symbolic cut completion step is outside its groups")
        remaining = tuple(
            variable
            for group in self.variable_groups[step:]
            for variable in group
        )
        if not remaining:
            return int(state == 1)
        return self.manager.model_count(state, remaining)

    def accepts(self, codes: Sequence[int]) -> bool:
        if len(codes) != len(self.variable_groups):
            raise ValueError("symbolic cut code sequence has the wrong length")
        state = self.initial_state
        for step, code in enumerate(codes):
            state = self.transition(step, state, int(code))
            if state == 0:
                return False
        return state == 1


@dataclass(slots=True)
class ROBDDLayoutCutAutomaton:
    """Placement-only adapter for ``FactorizedLayoutFeasibilityDP``."""

    cut: ROBDDSequentialParameterCut
    placement_only: bool = True

    def initial_state(self) -> int:
        return self.cut.initial_state

    @staticmethod
    def initial_resources() -> tuple[int, ...]:
        return ()

    def advance(
        self,
        state: int,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if resources:
            raise ValueError("symbolic layout cut has no Pareto resources")
        following = self.cut.transition(context.step, state, context.placed_code)
        if following == 0:
            return None
        if self.cut.completion_count(context.step + 1, following) == 0:
            return None
        return FrontierAutomatonTransition(following)

    def accepts(
        self,
        state: int,
        resources: tuple[int, ...],
        final_frontier,
    ) -> bool:
        if resources or final_frontier:
            raise ValueError("symbolic layout cut expects placement-only finalization")
        return state == 1


@dataclass(slots=True)
class ROBDDLabelDomainCutAutomaton:
    """Map global DP label codes into each structural cell's local domain."""

    cut: ROBDDSequentialParameterCut
    global_labels: tuple[str, ...]
    local_label_domains: tuple[tuple[str, ...], ...]
    placement_only: bool = True
    _local_codes: tuple[tuple[int, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.global_labels or len(self.global_labels) != len(set(self.global_labels)):
            raise ValueError("global symbolic-cut labels must be non-empty and unique")
        if len(self.local_label_domains) != len(self.cut.variable_groups):
            raise ValueError("local symbolic-cut domains do not match cut groups")
        if any(
            len(domain) != radix or len(domain) != len(set(domain))
            for domain, radix in zip(
                self.local_label_domains,
                self.cut.radices,
                strict=True,
            )
        ):
            raise ValueError("local symbolic-cut domains do not match their radices")
        global_index = {label: code for code, label in enumerate(self.global_labels)}
        unknown = {
            label
            for domain in self.local_label_domains
            for label in domain
            if label not in global_index
        }
        if unknown:
            raise ValueError(f"local symbolic-cut labels are outside the global domain: {unknown}")
        self._local_codes = tuple(
            tuple(domain.index(label) if label in domain else -1 for label in self.global_labels)
            for domain in self.local_label_domains
        )

    def initial_state(self) -> int:
        return self.cut.initial_state

    @staticmethod
    def initial_resources() -> tuple[int, ...]:
        return ()

    def advance(
        self,
        state: int,
        resources: tuple[int, ...],
        context: FrontierTransitionContext,
    ) -> FrontierAutomatonTransition | None:
        if resources:
            raise ValueError("symbolic label-domain cut has no Pareto resources")
        if not 0 <= context.placed_code < len(self.global_labels):
            return None
        local_code = self._local_codes[context.step][context.placed_code]
        if local_code < 0:
            return None
        following = self.cut.transition(context.step, state, local_code)
        if following == 0:
            return None
        if self.cut.completion_count(context.step + 1, following) == 0:
            return None
        return FrontierAutomatonTransition(following)

    def accepts(
        self,
        state: int,
        resources: tuple[int, ...],
        final_frontier,
    ) -> bool:
        if resources or final_frontier:
            raise ValueError("symbolic label-domain cut expects placement-only finalization")
        return state == 1


def structural_parameter_region_automaton(
    *,
    manager: ROBDDManager,
    accepted_root: int,
    cell_parameter_variables: Sequence[Sequence[Hashable]],
    cell_label_domains: Sequence[Sequence[str]],
    global_labels: Sequence[str],
    placement_order: Sequence[int],
) -> ROBDDLabelDomainCutAutomaton:
    """Build an exact outer-DP cut from a structural safe-parameter region."""

    groups_by_vertex = tuple(tuple(group) for group in cell_parameter_variables)
    domains_by_vertex = tuple(tuple(domain) for domain in cell_label_domains)
    order = tuple(int(vertex) for vertex in placement_order)
    if (
        not order
        or len(groups_by_vertex) != len(domains_by_vertex)
        or set(order) != set(range(len(groups_by_vertex)))
    ):
        raise ValueError("structural symbolic-cut placement order is invalid")
    groups = tuple(groups_by_vertex[vertex] for vertex in order)
    domains = tuple(domains_by_vertex[vertex] for vertex in order)
    cut = ROBDDSequentialParameterCut(
        manager=manager,
        accepted_root=accepted_root,
        variable_groups=groups,
        radices=tuple(len(domain) for domain in domains),
    )
    return ROBDDLabelDomainCutAutomaton(
        cut=cut,
        global_labels=tuple(global_labels),
        local_label_domains=domains,
    )
