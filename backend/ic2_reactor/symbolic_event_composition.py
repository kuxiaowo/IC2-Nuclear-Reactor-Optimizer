"""Exact sequential composition of frozen-parameter event alternatives.

For a fixed structural skeleton, cell ``i`` has a small real-label domain and
selects one deterministic event ``E[i,label]``.  Building one full transition
for every Cartesian-product layout would cost the product of those domain
sizes.  This module instead composes the events in official order and muxes
only the alternatives at the current cell.  The number of event circuits
compiled is therefore their sum; equal Boolean residuals share ROBDD nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Hashable, Mapping, Sequence

from .robdd import ROBDDManager


@dataclass(frozen=True, slots=True)
class FrozenParameterEvent:
    """Mutually exclusive event alternatives selected by frozen parameters."""

    name: str
    conditions: tuple[int, ...]
    alternative_next_functions: tuple[Mapping[Hashable, int], ...]
    alternative_failure_roots: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class FrozenParameterEventComposition:
    parameter_variables: tuple[Hashable, ...]
    dynamic_variables: tuple[Hashable, ...]
    valid_parameter_root: int
    raw_next_functions: dict[Hashable, int]
    transition_failure_root: int
    poisoned_next_functions: dict[Hashable, int]
    event_count: int
    skipped_identity_event_count: int
    compiled_alternative_count: int
    represented_parameter_count: int
    explicit_family_count: int
    allocated_nodes: int


def _validate_event_partition(
    manager: ROBDDManager,
    valid_parameter_root: int,
    conditions: Sequence[int],
) -> None:
    covered = 0
    for index, condition in enumerate(conditions):
        if manager.apply("and", covered, condition) != 0:
            raise ValueError("event alternative conditions overlap")
        covered = manager.apply("or", covered, condition)
        for other in conditions[index + 1:]:
            if manager.apply("and", condition, other) != 0:
                raise ValueError("event alternative conditions overlap")
    missing = manager.apply(
        "and",
        valid_parameter_root,
        manager.negate(covered),
    )
    if missing != 0:
        raise ValueError(
            "event alternatives must partition the valid parameter region"
        )


class FrozenParameterEventComposer:
    """Streaming event composer; alternatives can be released after each slot."""

    def __init__(
        self,
        manager: ROBDDManager,
        *,
        parameter_variables: Sequence[Hashable],
        dynamic_variables: Sequence[Hashable],
        valid_parameter_root: int,
        initial_failure_root: int = 0,
    ) -> None:
        self.manager = manager
        self.parameters = tuple(parameter_variables)
        self.dynamics = tuple(dynamic_variables)
        if not self.parameters or len(self.parameters) != len(set(self.parameters)):
            raise ValueError("frozen event parameters must be non-empty and unique")
        if not self.dynamics or len(self.dynamics) != len(set(self.dynamics)):
            raise ValueError("frozen event dynamics must be non-empty and unique")
        if set(self.parameters) & set(self.dynamics):
            raise ValueError("event parameters and dynamics must be disjoint")
        if set(self.parameters) | set(self.dynamics) != set(manager.variables):
            raise ValueError("event variable partition must cover the ROBDD manager")
        if not manager.support(valid_parameter_root) <= set(self.parameters):
            raise ValueError("valid event parameters depend on dynamic state")
        self.valid_parameter_root = valid_parameter_root
        self.identity = {
            variable: manager.variable(variable)
            for variable in self.dynamics
        }
        self.current = dict(self.identity)
        self.failure = initial_failure_root
        self.event_count = 0
        self.skipped_identity_events = 0
        self.alternative_count = 0
        self.domain_sizes: list[int] = []

    def apply_event(self, event: FrozenParameterEvent) -> None:
        manager = self.manager
        conditions = tuple(event.conditions)
        alternatives = tuple(event.alternative_next_functions)
        failures = (
            (0,) * len(conditions)
            if not event.alternative_failure_roots
            else tuple(event.alternative_failure_roots)
        )
        if (
            not event.name
            or not conditions
            or len(conditions) != len(alternatives)
            or len(conditions) != len(failures)
        ):
            raise ValueError("frozen event alternatives have inconsistent lengths")
        _validate_event_partition(manager, self.valid_parameter_root, conditions)
        if any(set(functions) != set(self.dynamics) for functions in alternatives):
            raise ValueError("every event alternative must define every dynamic bit")
        if any(
            not manager.support(condition) <= set(self.parameters)
            for condition in conditions
        ):
            raise ValueError("event selection condition depends on dynamic state")
        self.event_count += 1
        self.domain_sizes.append(len(conditions))
        if all(
            failure_root == 0
            and all(
                functions[variable] == self.identity[variable]
                for variable in self.dynamics
            )
            for functions, failure_root in zip(
                alternatives,
                failures,
                strict=True,
            )
        ):
            self.skipped_identity_events += 1
            return

        substitutions = {
            variable: self.current[variable]
            for variable in self.dynamics
            if self.current[variable] != self.identity[variable]
        }
        changed = set(substitutions)
        support_cache: dict[int, frozenset[Hashable]] = {}

        def substitute(root: int) -> int:
            if root in (0, 1) or not substitutions:
                return root
            support = support_cache.get(root)
            if support is None:
                support = manager.support(root)
                support_cache[root] = support
            if support.isdisjoint(changed):
                return root
            return manager.compose(root, substitutions)

        candidate_functions = tuple(
            {
                variable: (
                    self.current[variable]
                    if functions[variable] == self.identity[variable]
                    else substitute(functions[variable])
                )
                for variable in self.dynamics
            }
            for functions in alternatives
        )
        following: dict[Hashable, int] = {}
        for variable in self.dynamics:
            candidates = tuple(item[variable] for item in candidate_functions)
            if all(candidate == self.current[variable] for candidate in candidates):
                following[variable] = self.current[variable]
            elif len(conditions) == 1 and conditions[0] == 1:
                following[variable] = candidates[0]
            elif all(candidate == candidates[0] for candidate in candidates):
                following[variable] = candidates[0]
            else:
                selected = self.current[variable]
                for condition, candidate in zip(
                    conditions,
                    candidates,
                    strict=True,
                ):
                    selected = manager.ite(condition, candidate, selected)
                following[variable] = selected
        event_failure = manager.disjunction(*(
            manager.apply(
                "and",
                condition,
                substitute(root),
            )
            for condition, root in zip(conditions, failures, strict=True)
        ))
        self.failure = manager.apply("or", self.failure, event_failure)
        self.current = following
        self.alternative_count += len(conditions)

    def compact(self, extra_roots: Sequence[int] = ()) -> tuple[int, ...]:
        """Copy only live composition roots and caller-owned roots."""

        extras = tuple(extra_roots)
        roots = (
            self.valid_parameter_root,
            self.failure,
            *(self.current[variable] for variable in self.dynamics),
            *extras,
        )
        manager, compacted = self.manager.compact_roots(roots)
        cursor = 0
        self.valid_parameter_root, self.failure = compacted[cursor:cursor + 2]
        cursor += 2
        self.current = dict(zip(
            self.dynamics,
            compacted[cursor:cursor + len(self.dynamics)],
            strict=True,
        ))
        cursor += len(self.dynamics)
        self.manager = manager
        self.identity = {
            variable: manager.variable(variable)
            for variable in self.dynamics
        }
        return tuple(compacted[cursor:])

    def finish(self) -> FrozenParameterEventComposition:
        poisoned = {
            variable: self.manager.apply("or", self.current[variable], self.failure)
            for variable in self.dynamics
        }
        represented = self.manager.model_count(
            self.valid_parameter_root,
            self.parameters,
        )
        return FrozenParameterEventComposition(
            parameter_variables=self.parameters,
            dynamic_variables=self.dynamics,
            valid_parameter_root=self.valid_parameter_root,
            raw_next_functions=dict(self.current),
            transition_failure_root=self.failure,
            poisoned_next_functions=poisoned,
            event_count=self.event_count,
            skipped_identity_event_count=self.skipped_identity_events,
            compiled_alternative_count=self.alternative_count,
            represented_parameter_count=represented,
            explicit_family_count=prod(self.domain_sizes),
            allocated_nodes=self.manager.allocated_node_count,
        )


def compose_frozen_parameter_events(
    manager: ROBDDManager,
    *,
    parameter_variables: Sequence[Hashable],
    dynamic_variables: Sequence[Hashable],
    valid_parameter_root: int,
    events: Sequence[FrozenParameterEvent],
    initial_failure_root: int = 0,
) -> FrozenParameterEventComposition:
    """Compose local alternatives without enumerating their Cartesian product.

    Alternative functions are written over the dynamic input variables of one
    event.  Before applying event ``i`` they are simultaneously composed with
    the accumulated output functions of events ``0..i-1``.  Conditions depend
    only on frozen parameters and must form an exact partition of the caller's
    valid parameter region.  Transition failures are accumulated separately;
    ``poisoned_next_functions`` maps every failed transition to the all-one
    state used by the IC2 symbolic backend.
    """

    composer = FrozenParameterEventComposer(
        manager,
        parameter_variables=parameter_variables,
        dynamic_variables=dynamic_variables,
        valid_parameter_root=valid_parameter_root,
        initial_failure_root=initial_failure_root,
    )
    for event in events:
        composer.apply_event(event)
    return composer.finish()
