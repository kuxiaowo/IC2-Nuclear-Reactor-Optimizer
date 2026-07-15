"""Exact parameter-wise safety from the only reachable initial trajectories.

The usual backwards failure attractor classifies every dynamic state, although
the reactor optimization problem asks only about the official zero-heat
initial state.  For frozen layout parameters ``p`` and deterministic dynamics
``s' = T(p, s)``, the state at time ``t`` is a Boolean-vector function
``s_t(p)``.  This module advances those functions directly and stores the
already visited reachable graph as one ROBDD relation ``Visited(p, s)``.

A parameter assignment is unsafe when its current state is bad.  It is safe
forever when its current state has already appeared in ``Visited``: subsequent
behaviour repeats because the transition is deterministic and the parameter is
frozen.  Thus no unreachable point of the dynamic state cube is explored and
no parameter assignment is enumerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .robdd import ROBDDManager


def _state_graph_root(
    manager: ROBDDManager,
    dynamic_variables: Sequence[Hashable],
    state_functions: Sequence[int],
    parameter_region: int,
) -> int:
    """Graph of one parameter-to-state function, restricted to a region."""

    return manager.conjunction(
        parameter_region,
        *(
            manager.apply(
                "equiv",
                manager.variable(variable),
                function,
            )
            for variable, function in zip(
                dynamic_variables,
                state_functions,
                strict=True,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class SymbolicForwardTrajectoryProof:
    """Replayable partition of parameters into safe, failed and unresolved."""

    parameter_variables: tuple[Hashable, ...]
    dynamic_variables: tuple[Hashable, ...]
    initial_dynamic_assignment: tuple[tuple[Hashable, bool], ...]
    parameter_constraint_root: int
    bad_root: int
    state_function_layers: tuple[tuple[int, ...], ...]
    failure_layer_roots: tuple[int, ...]
    repetition_layer_roots: tuple[int, ...]
    failed_parameter_root: int
    safe_parameter_root: int
    unknown_parameter_root: int
    visited_graph_root: int
    continuation_state_functions: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return self.unknown_parameter_root == 0

    @property
    def inspected_steps(self) -> int:
        return len(self.state_function_layers)

    def verify(
        self,
        manager: ROBDDManager,
        next_functions: Mapping[Hashable, int],
    ) -> bool:
        parameters = self.parameter_variables
        dynamics = self.dynamic_variables
        if (
            not parameters
            or len(parameters) != len(set(parameters))
            or len(dynamics) != len(set(dynamics))
            or set(parameters) & set(dynamics)
            or set(parameters) | set(dynamics) != set(manager.variables)
            or set(next_functions) != set(dynamics)
        ):
            return False
        initial = dict(self.initial_dynamic_assignment)
        if set(initial) != set(dynamics):
            return False
        if not manager.support(self.parameter_constraint_root) <= set(parameters):
            return False
        if not (
            len(self.state_function_layers)
            == len(self.failure_layer_roots)
            == len(self.repetition_layer_roots)
        ):
            return False

        current = tuple(int(initial[variable]) for variable in dynamics)
        unresolved = self.parameter_constraint_root
        failed = 0
        safe = 0
        visited = 0
        for expected_state, expected_failure, expected_repetition in zip(
            self.state_function_layers,
            self.failure_layer_roots,
            self.repetition_layer_roots,
            strict=True,
        ):
            if tuple(expected_state) != current or unresolved == 0:
                return False
            substitutions = dict(zip(dynamics, current, strict=True))
            failure = manager.apply(
                "and",
                unresolved,
                manager.compose(self.bad_root, substitutions),
            )
            if failure != expected_failure:
                return False
            failed = manager.apply("or", failed, failure)
            unresolved = manager.apply(
                "and",
                unresolved,
                manager.negate(failure),
            )

            repetition = manager.apply(
                "and",
                unresolved,
                manager.compose(visited, substitutions),
            )
            if repetition != expected_repetition:
                return False
            safe = manager.apply("or", safe, repetition)
            unresolved = manager.apply(
                "and",
                unresolved,
                manager.negate(repetition),
            )
            if unresolved == 0:
                continue

            visited = manager.apply(
                "or",
                visited,
                _state_graph_root(manager, dynamics, current, unresolved),
            )
            current = tuple(
                manager.compose(next_functions[variable], substitutions)
                for variable in dynamics
            )

        if (
            failed != self.failed_parameter_root
            or safe != self.safe_parameter_root
            or unresolved != self.unknown_parameter_root
            or visited != self.visited_graph_root
            or current != self.continuation_state_functions
        ):
            return False
        if manager.apply("and", failed, safe) != 0:
            return False
        classified = manager.disjunction(failed, safe, unresolved)
        return classified == self.parameter_constraint_root


def symbolic_forward_trajectory_safety(
    manager: ROBDDManager,
    *,
    parameter_variables: Sequence[Hashable],
    dynamic_variables: Sequence[Hashable],
    next_functions: Mapping[Hashable, int],
    bad_root: int,
    initial_dynamic_assignment: Mapping[Hashable, bool],
    parameter_constraint_root: int = 1,
    maximum_steps: int | None = None,
) -> SymbolicForwardTrajectoryProof:
    """Classify all frozen parameters by their reachable deterministic orbit.

    ``maximum_steps`` is a proof-safe cutoff.  Reaching it leaves the remaining
    parameter region in ``unknown_parameter_root``; UNKNOWN is never treated as
    safe or infeasible.  Without a cutoff termination follows from finiteness,
    although the finite bound can still be too large for a particular input.
    """

    parameters = tuple(parameter_variables)
    dynamics = tuple(dynamic_variables)
    if not parameters or len(parameters) != len(set(parameters)):
        raise ValueError("symbolic parameters must be non-empty and unique")
    if len(dynamics) != len(set(dynamics)) or set(parameters) & set(dynamics):
        raise ValueError("symbolic parameter/dynamic variables must be disjoint")
    if set(parameters) | set(dynamics) != set(manager.variables):
        raise ValueError("parameter/dynamic partition must cover the ROBDD manager")
    if set(next_functions) != set(dynamics):
        raise ValueError("every dynamic variable needs one next-state function")
    initial = dict(initial_dynamic_assignment)
    if set(initial) != set(dynamics):
        raise ValueError("initial assignment must define every dynamic variable")
    if not manager.support(parameter_constraint_root) <= set(parameters):
        raise ValueError("parameter constraint depends on a dynamic variable")
    if maximum_steps is not None and maximum_steps <= 0:
        raise ValueError("maximum forward steps must be positive or None")

    current = tuple(int(initial[variable]) for variable in dynamics)
    unresolved = parameter_constraint_root
    failed = 0
    safe = 0
    visited = 0
    state_layers: list[tuple[int, ...]] = []
    failure_layers: list[int] = []
    repetition_layers: list[int] = []

    while unresolved != 0 and (
        maximum_steps is None or len(state_layers) < maximum_steps
    ):
        state_layers.append(current)
        substitutions = dict(zip(dynamics, current, strict=True))
        failure = manager.apply(
            "and",
            unresolved,
            manager.compose(bad_root, substitutions),
        )
        failure_layers.append(failure)
        failed = manager.apply("or", failed, failure)
        unresolved = manager.apply(
            "and",
            unresolved,
            manager.negate(failure),
        )

        repetition = manager.apply(
            "and",
            unresolved,
            manager.compose(visited, substitutions),
        )
        repetition_layers.append(repetition)
        safe = manager.apply("or", safe, repetition)
        unresolved = manager.apply(
            "and",
            unresolved,
            manager.negate(repetition),
        )
        if unresolved == 0:
            continue

        visited = manager.apply(
            "or",
            visited,
            _state_graph_root(manager, dynamics, current, unresolved),
        )
        current = tuple(
            manager.compose(next_functions[variable], substitutions)
            for variable in dynamics
        )

    proof = SymbolicForwardTrajectoryProof(
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        initial_dynamic_assignment=tuple(
            (variable, initial[variable]) for variable in dynamics
        ),
        parameter_constraint_root=parameter_constraint_root,
        bad_root=bad_root,
        state_function_layers=tuple(state_layers),
        failure_layer_roots=tuple(failure_layers),
        repetition_layer_roots=tuple(repetition_layers),
        failed_parameter_root=failed,
        safe_parameter_root=safe,
        unknown_parameter_root=unresolved,
        visited_graph_root=visited,
        continuation_state_functions=current,
    )
    if not proof.verify(manager, next_functions):  # pragma: no cover
        raise AssertionError("symbolic forward trajectory proof failed verification")
    return proof
