"""Exact frozen-parameter optimization over a symbolic safety invariant.

Layout bits are treated as state bits with identity next functions.  One
failure-attractor fixed point therefore represents all layouts at once and
shares every equal Boolean residual.  Restricting the dynamic bits to the
official initial state yields exactly the parameter assignments that are safe
forever.  An unsigned objective circuit is then maximized bit by bit without
enumerating those assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Protocol, Sequence

from .robdd import ROBDDManager


class SafetyInvariantCertificate(Protocol):
    """Minimum interface shared by closed-function and local-event proofs."""

    state_variables: tuple[Hashable, ...]
    safe_invariant_root: int


def maximize_unsigned_boolean_objective(
    manager: ROBDDManager,
    feasible_root: int,
    objective_bits: Sequence[int],
) -> tuple[int | None, int]:
    """Maximize a little-endian Boolean vector using one SAT test per bit."""

    bits = tuple(objective_bits)
    if not bits:
        raise ValueError("symbolic objective must contain at least one bit")
    if feasible_root == 0:
        return None, 0
    optimum_root = feasible_root
    value = 0
    for index in range(len(bits) - 1, -1, -1):
        bit = bits[index]
        with_one = manager.apply("and", optimum_root, bit)
        if with_one != 0:
            optimum_root = with_one
            value |= 1 << index
        else:
            optimum_root = manager.apply(
                "and",
                optimum_root,
                manager.negate(bit),
            )
    if optimum_root == 0:  # pragma: no cover - satisfiable branch invariant
        raise AssertionError("symbolic objective maximization lost all models")
    return value, optimum_root


@dataclass(frozen=True, slots=True)
class SymbolicParameterOptimum:
    parameter_variables: tuple[Hashable, ...]
    dynamic_variables: tuple[Hashable, ...]
    initial_dynamic_assignment: tuple[tuple[Hashable, bool], ...]
    safe_initial_parameter_root: int
    parameter_constraint_root: int
    feasible_parameter_root: int
    objective_bits: tuple[int, ...]
    optimum_value: int | None
    optimum_parameter_root: int
    witness: tuple[tuple[Hashable, bool], ...] | None
    safe_initial_parameter_count: int
    feasible_parameter_count: int
    optimum_parameter_count: int

    def verify(
        self,
        manager: ROBDDManager,
        safety_proof: SafetyInvariantCertificate,
    ) -> bool:
        parameters = self.parameter_variables
        dynamics = self.dynamic_variables
        if set(parameters) & set(dynamics):
            return False
        if set(parameters) | set(dynamics) != set(safety_proof.state_variables):
            return False
        initial = dict(self.initial_dynamic_assignment)
        if set(initial) != set(dynamics):
            return False
        safe_initial = manager.restrict(
            safety_proof.safe_invariant_root,
            initial,
        )
        if safe_initial != self.safe_initial_parameter_root:
            return False
        feasible = manager.apply(
            "and",
            safe_initial,
            self.parameter_constraint_root,
        )
        if feasible != self.feasible_parameter_root:
            return False
        value, optimum = maximize_unsigned_boolean_objective(
            manager,
            feasible,
            self.objective_bits,
        )
        if value != self.optimum_value or optimum != self.optimum_parameter_root:
            return False
        if manager.model_count(safe_initial, parameters) != self.safe_initial_parameter_count:
            return False
        if manager.model_count(feasible, parameters) != self.feasible_parameter_count:
            return False
        if manager.model_count(optimum, parameters) != self.optimum_parameter_count:
            return False
        if value is None:
            return self.witness is None
        if self.witness is None:
            return False
        witness = dict(self.witness)
        if set(witness) != set(parameters):
            return False
        total = {
            variable: witness.get(variable, initial.get(variable, False))
            for variable in manager.variables
        }
        if not manager.evaluate(optimum, total):
            return False
        observed = sum(
            int(manager.evaluate(bit, total)) << index
            for index, bit in enumerate(self.objective_bits)
        )
        return observed == value


def optimize_safe_frozen_parameters(
    manager: ROBDDManager,
    safety_proof: SafetyInvariantCertificate,
    *,
    parameter_variables: Sequence[Hashable],
    dynamic_variables: Sequence[Hashable],
    initial_dynamic_assignment: Mapping[Hashable, bool],
    objective_bits: Sequence[int],
    parameter_constraint_root: int = 1,
) -> SymbolicParameterOptimum:
    """Return and internally verify the exact best safe parameter assignment."""

    parameters = tuple(parameter_variables)
    dynamics = tuple(dynamic_variables)
    if not parameters or len(parameters) != len(set(parameters)):
        raise ValueError("symbolic parameters must be non-empty and unique")
    if len(dynamics) != len(set(dynamics)) or set(parameters) & set(dynamics):
        raise ValueError("symbolic parameter/dynamic variables must be disjoint")
    if set(parameters) | set(dynamics) != set(safety_proof.state_variables):
        raise ValueError("parameter/dynamic partition does not cover the state")
    initial = dict(initial_dynamic_assignment)
    if set(initial) != set(dynamics):
        raise ValueError("initial assignment must define every dynamic variable")
    parameter_set = set(parameters)
    if not manager.support(parameter_constraint_root) <= parameter_set:
        raise ValueError("parameter constraint depends on a dynamic variable")
    if any(not manager.support(bit) <= parameter_set for bit in objective_bits):
        raise ValueError("symbolic objective depends on a dynamic variable")

    safe_initial = manager.restrict(
        safety_proof.safe_invariant_root,
        initial,
    )
    if not manager.support(safe_initial) <= parameter_set:  # pragma: no cover
        raise AssertionError("initial-state restriction retained dynamic variables")
    feasible = manager.apply("and", safe_initial, parameter_constraint_root)
    value, optimum_root = maximize_unsigned_boolean_objective(
        manager,
        feasible,
        objective_bits,
    )
    raw_witness = manager.satisfying_assignment(optimum_root)
    witness = None if raw_witness is None else tuple(
        (variable, raw_witness[variable]) for variable in parameters
    )
    result = SymbolicParameterOptimum(
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        initial_dynamic_assignment=tuple((item, initial[item]) for item in dynamics),
        safe_initial_parameter_root=safe_initial,
        parameter_constraint_root=parameter_constraint_root,
        feasible_parameter_root=feasible,
        objective_bits=tuple(objective_bits),
        optimum_value=value,
        optimum_parameter_root=optimum_root,
        witness=witness,
        safe_initial_parameter_count=manager.model_count(safe_initial, parameters),
        feasible_parameter_count=manager.model_count(feasible, parameters),
        optimum_parameter_count=manager.model_count(optimum_root, parameters),
    )
    if not result.verify(manager, safety_proof):  # pragma: no cover
        raise AssertionError("symbolic frozen-parameter certificate failed")
    return result
