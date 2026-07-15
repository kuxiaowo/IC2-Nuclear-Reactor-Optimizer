"""Merge equal-schema fixed IC2 models into one frozen-layout symbolic family."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Hashable, Sequence

from .ic2_symbolic_no_exchange import IC2NoExchangeSymbolicModel
from .robdd import ROBDDManager
from .robdd_bitvector import unsigned_at_least_constant, unsigned_equals_constant
from .symbolic_parameter_optimization import (
    SymbolicParameterOptimum,
    maximize_unsigned_boolean_objective,
    optimize_safe_frozen_parameters,
)
from .symbolic_forward_trajectory import (
    SymbolicForwardTrajectoryProof,
    symbolic_forward_trajectory_safety,
)
from .symbolic_safety import (
    CompactingSymbolicSafetyResult,
    SymbolicFailureAttractorProof,
    compacting_symbolic_failure_attractor,
    symbolic_failure_attractor,
)


@dataclass(frozen=True, slots=True)
class SymbolicIC2ModelFamily:
    layouts: tuple[tuple[str, ...], ...]
    objective_values: tuple[int, ...]
    manager: ROBDDManager
    parameter_variables: tuple[Hashable, ...]
    dynamic_variables: tuple[Hashable, ...]
    next_functions: dict[Hashable, int]
    bad_root: int
    valid_parameter_root: int
    objective_bits: tuple[int, ...]
    initial_dynamic_assignment: tuple[tuple[Hashable, bool], ...]

    def prove_safety_forward(
        self,
        *,
        maximum_steps: int | None = None,
    ) -> SymbolicForwardTrajectoryProof:
        """Classify only official-initial trajectories for every parameter."""

        return symbolic_forward_trajectory_safety(
            self.manager,
            parameter_variables=self.parameter_variables,
            dynamic_variables=self.dynamic_variables,
            next_functions={
                variable: self.next_functions[variable]
                for variable in self.dynamic_variables
            },
            bad_root=self.bad_root,
            initial_dynamic_assignment=dict(self.initial_dynamic_assignment),
            parameter_constraint_root=self.valid_parameter_root,
            maximum_steps=maximum_steps,
        )

    def prove_optimum_forward(
        self,
        *,
        maximum_steps: int | None = None,
    ) -> tuple[SymbolicForwardTrajectoryProof, int | None, int]:
        """Maximize after a complete reachable-only safety classification.

        The returned root is the full set of equally optimal parameter
        assignments.  A cutoff that leaves UNKNOWN parameters cannot certify a
        global optimum and therefore raises instead of silently optimizing the
        proved-safe subset.
        """

        proof = self.prove_safety_forward(maximum_steps=maximum_steps)
        if not proof.complete:
            raise TimeoutError(
                "forward safety proof left an unresolved parameter region"
            )
        value, optimum_root = maximize_unsigned_boolean_objective(
            self.manager,
            proof.safe_parameter_root,
            self.objective_bits,
        )
        return proof, value, optimum_root

    def prove_optimum(
        self,
    ) -> tuple[SymbolicFailureAttractorProof, SymbolicParameterOptimum]:
        proof = symbolic_failure_attractor(
            self.manager,
            (*self.parameter_variables, *self.dynamic_variables),
            self.next_functions,
            self.bad_root,
        )
        optimum = optimize_safe_frozen_parameters(
            self.manager,
            proof,
            parameter_variables=self.parameter_variables,
            dynamic_variables=self.dynamic_variables,
            initial_dynamic_assignment=dict(self.initial_dynamic_assignment),
            parameter_constraint_root=self.valid_parameter_root,
            objective_bits=self.objective_bits,
        )
        return proof, optimum

    def prove_optimum_compacting(
        self,
    ) -> tuple[CompactingSymbolicSafetyResult, SymbolicParameterOptimum]:
        result = compacting_symbolic_failure_attractor(
            self.manager,
            (*self.parameter_variables, *self.dynamic_variables),
            self.next_functions,
            self.bad_root,
        )
        optimum = optimize_safe_frozen_parameters(
            result.manager,
            result.proof,
            parameter_variables=self.parameter_variables,
            dynamic_variables=self.dynamic_variables,
            initial_dynamic_assignment=dict(self.initial_dynamic_assignment),
            parameter_constraint_root=result.manager.import_roots(
                self.manager,
                (self.valid_parameter_root,),
            )[0],
            objective_bits=result.manager.import_roots(
                self.manager,
                self.objective_bits,
            ),
        )
        return result, optimum

    def witness_layout(self, optimum: SymbolicParameterOptimum) -> tuple[str, ...] | None:
        if optimum.witness is None:
            return None
        assignment = dict(optimum.witness)
        code = sum(
            int(assignment[variable]) << bit
            for bit, variable in enumerate(self.parameter_variables)
        )
        return self.layouts[code] if code < len(self.layouts) else None


def merge_symbolic_ic2_model_family(
    models: Sequence[IC2NoExchangeSymbolicModel],
    objective_values: Sequence[int],
) -> SymbolicIC2ModelFamily:
    """Build one exact transition family without enumerating dynamic states."""

    model_tuple = tuple(models)
    objectives = tuple(int(value) for value in objective_values)
    if not model_tuple or len(model_tuple) != len(objectives):
        raise ValueError("symbolic model family/objective lengths differ")
    if any(value < 0 for value in objectives):
        raise ValueError("symbolic family objectives must be non-negative")
    reference = model_tuple[0]
    if any(model.state_variables != reference.state_variables for model in model_tuple):
        raise ValueError("symbolic family state variables differ")
    if any(model.fields != reference.fields for model in model_tuple):
        raise ValueError("symbolic family thermal schemas differ")

    parameter_width = max(1, ceil(log2(len(model_tuple))))
    parameters = tuple(("layout_choice", bit) for bit in range(parameter_width))
    dynamics = reference.state_variables
    manager = ROBDDManager((*parameters, *dynamics))
    parameter_bits = tuple(manager.variable(variable) for variable in parameters)
    conditions = tuple(
        unsigned_equals_constant(manager, parameter_bits, code)
        for code in range(len(model_tuple))
    )
    valid = manager.negate(unsigned_at_least_constant(
        manager,
        parameter_bits,
        len(model_tuple),
    ))

    imported_bad = []
    imported_next: list[dict[Hashable, int]] = []
    for model in model_tuple:
        ordered_roots = (
            model.bad_root,
            *(model.next_functions[variable] for variable in dynamics),
        )
        imported = manager.import_roots(model.manager, ordered_roots)
        imported_bad.append(imported[0])
        imported_next.append(dict(zip(dynamics, imported[1:], strict=True)))

    def multiplex(roots: Sequence[int]) -> int:
        return manager.disjunction(*(
            manager.apply("and", condition, root)
            for condition, root in zip(conditions, roots, strict=True)
        ))

    bad = manager.apply("or", manager.negate(valid), multiplex(imported_bad))
    next_functions = {
        variable: manager.variable(variable) for variable in parameters
    }
    next_functions.update({
        variable: multiplex(tuple(item[variable] for item in imported_next))
        for variable in dynamics
    })
    objective_width = max(1, max(objectives, default=0).bit_length())
    objective_bits = tuple(
        manager.disjunction(*(
            condition
            for condition, value in zip(conditions, objectives, strict=True)
            if value >> bit & 1
        ))
        for bit in range(objective_width)
    )

    state_variables = (*parameters, *dynamics)
    manager, compacted = manager.compact_roots((
        bad,
        valid,
        *objective_bits,
        *(next_functions[variable] for variable in state_variables),
    ))
    cursor = 0
    bad, valid = compacted[cursor:cursor + 2]
    cursor += 2
    objective_bits = compacted[cursor:cursor + len(objective_bits)]
    cursor += len(objective_bits)
    next_functions = dict(zip(
        state_variables,
        compacted[cursor:],
        strict=True,
    ))

    initial_codes = tuple(model.encode(0) for model in model_tuple)
    if len(set(initial_codes)) != 1:
        raise ValueError("symbolic family layouts have different initial states")
    initial_assignment = reference.assignment(initial_codes[0])
    return SymbolicIC2ModelFamily(
        layouts=tuple(model.layout for model in model_tuple),
        objective_values=objectives,
        manager=manager,
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        next_functions=next_functions,
        bad_root=bad,
        valid_parameter_root=valid,
        objective_bits=objective_bits,
        initial_dynamic_assignment=tuple(
            (variable, initial_assignment[variable]) for variable in dynamics
        ),
    )
