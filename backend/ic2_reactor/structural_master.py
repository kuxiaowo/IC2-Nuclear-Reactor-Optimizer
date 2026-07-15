"""Exact geometry relaxation for aggregate fuel-degree signatures.

The model deliberately forgets cooling labels and thermal dynamics.  It asks
only whether the active induced subgraph of the supplied reactor graph can
contain the requested fuel labels at the requested active-neighbour degrees.
Consequently an INFEASIBLE answer excludes every full component layout with
that aggregate signature, while a feasible answer is only a relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

from .mathematical_model import AggregatePattern, ReactorProblem


@dataclass(frozen=True, slots=True)
class StructuralEmbeddingResult:
    status: str
    possible: bool | None
    proven: bool
    skeleton: tuple[str, ...] | None
    elapsed_seconds: float
    conflicts: int
    branches: int


@dataclass(frozen=True, slots=True)
class StructuralReliefResult:
    """Optimal value of a geometry-aware optimistic local-relief model.

    ``maximum_relief`` is an upper bound, not a claimed realizable reactor:
    every non-active, non-side-vent cell is allowed to accept side heat and
    exchangers are allowed to realise their supplied optimistic relief without
    a detailed routing layout.  Therefore a proven value below the required
    relief is a sound exclusion of the aggregate pattern.
    """

    status: str
    proven_optimal: bool
    maximum_relief: int | None
    relief_upper_bound: int | None
    excluded: bool | None
    side_vent_cells: int | None
    effective_side_edges: int | None
    exchanger_cells: int | None
    baseline_capacity_loss: int | None
    elapsed_seconds: float
    conflicts: int
    branches: int


class AggregateDegreeEmbeddingMaster:
    """Embed one aggregate fuel-degree pattern in an arbitrary slot graph."""

    def __init__(self, problem: ReactorProblem) -> None:
        self.problem = problem
        self.fuel_ids = frozenset(
            item.id for item in problem.power_components if item.rods > 0
        )

    def solve(
        self,
        pattern: AggregatePattern,
        *,
        seconds: float = 10.0,
        workers: int = 1,
        random_seed: int = 221,
    ) -> StructuralEmbeddingResult:
        try:
            from ortools.sat.python import cp_model
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError(
                "AggregateDegreeEmbeddingMaster requires OR-Tools"
            ) from error
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")

        graph = self.problem.graph
        requested = tuple(pattern.fuel_degree_counts)
        if unknown := {item for item, _degree, _count in requested} - self.fuel_ids:
            raise ValueError(f"pattern contains unknown fuel labels: {sorted(unknown)}")
        if any(
            count < 0 or degree < 0 or degree > graph.maximum_degree
            for _item, degree, count in requested
        ):
            raise ValueError("pattern has an invalid degree or count")
        fuel_cells = sum(count for _item, _degree, count in requested)
        unknown_active = pattern.active_cells - fuel_cells
        if unknown_active < 0 or pattern.active_cells > graph.size:
            raise ValueError("pattern active-cell count is inconsistent")

        model = cp_model.CpModel()
        active = [model.new_bool_var(f"active_{vertex}") for vertex in graph.vertices]
        unknown = [model.new_bool_var(f"unknown_active_{vertex}") for vertex in graph.vertices]
        degree = [
            model.new_int_var(0, len(graph.neighbours[vertex]), f"degree_{vertex}")
            for vertex in graph.vertices
        ]
        states = {
            (item, state_degree): [
                model.new_bool_var(f"state_{vertex}_{item}_{state_degree}")
                for vertex in graph.vertices
            ]
            for item, state_degree, _count in requested
        }

        for vertex in graph.vertices:
            incident_states = [values[vertex] for values in states.values()]
            model.add(active[vertex] == unknown[vertex] + sum(incident_states))
            model.add(
                degree[vertex] == sum(active[other] for other in graph.neighbours[vertex])
            )
            for (_item, state_degree), values in states.items():
                model.add(degree[vertex] == state_degree).only_enforce_if(values[vertex])

        count_by_state = {
            (item, state_degree): count
            for item, state_degree, count in requested
        }
        for key, values in states.items():
            model.add(sum(values) == count_by_state[key])
        model.add(sum(unknown) == unknown_active)
        model.add(sum(active) == pattern.active_cells)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = random_seed
        started = perf_counter()
        status_code = solver.solve(model)
        elapsed = perf_counter() - started
        if status_code == cp_model.MODEL_INVALID:
            raise RuntimeError(f"invalid structural master: {model.validate()}")
        status = solver.status_name(status_code)
        feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        if status_code == cp_model.INFEASIBLE:
            possible: bool | None = False
            proven = True
        elif feasible:
            possible = True
            proven = True
        else:
            possible = None
            proven = False

        skeleton = None
        if feasible:
            labels = []
            for vertex in graph.vertices:
                label = "active_unknown" if solver.value(unknown[vertex]) else "empty"
                for (item, _state_degree), values in states.items():
                    if solver.value(values[vertex]):
                        label = item
                        break
                labels.append(label)
            skeleton = tuple(labels)
        return StructuralEmbeddingResult(
            status=status,
            possible=possible,
            proven=proven,
            skeleton=skeleton,
            elapsed_seconds=elapsed,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
        )

    def maximize_optimistic_relief(
        self,
        pattern: AggregatePattern,
        *,
        ideal_receiver_sink: int = 20,
        side_rate: int = 4,
        exchanger_relief: int = 72,
        exchanger_slot_cost: int = 20,
        seconds: float = 10.0,
        workers: int = 1,
        random_seed: int = 221,
    ) -> StructuralReliefResult:
        """Maximise aggregate overload relief subject to grid geometry.

        Start with the optimistic baseline in which every non-active cell is
        an independent ``ideal_receiver_sink`` vent.  Selecting a side vent
        loses one such baseline receiver, but gains ``side_rate`` for every
        directed edge from that vent to a non-active, non-side-vent cell.
        Selecting an exchanger likewise loses one baseline receiver and gains
        at most ``exchanger_relief``.  Exchanger placement and detailed flow
        are deliberately relaxed, so the optimum remains an upper bound.

        Unlike the count-only knapsack, adjacent/boundary side vents cannot
        pretend to have four useful sides.  The active fuel-degree signature
        is embedded exactly in the supplied graph at the same time.
        """

        try:
            from ortools.sat.python import cp_model
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError(
                "AggregateDegreeEmbeddingMaster requires OR-Tools"
            ) from error
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if min(
            ideal_receiver_sink,
            side_rate,
            exchanger_relief,
            exchanger_slot_cost,
        ) <= 0:
            raise ValueError("relief capacities must be positive")

        graph = self.problem.graph
        requested = tuple(pattern.fuel_degree_counts)
        if unknown_ids := {
            item for item, _degree, _count in requested
        } - self.fuel_ids:
            raise ValueError(
                f"pattern contains unknown fuel labels: {sorted(unknown_ids)}"
            )
        if any(
            count < 0 or degree < 0 or degree > graph.maximum_degree
            for _item, degree, count in requested
        ):
            raise ValueError("pattern has an invalid degree or count")
        fuel_cells = sum(count for _item, _degree, count in requested)
        unknown_active = pattern.active_cells - fuel_cells
        if unknown_active < 0 or pattern.active_cells > graph.size:
            raise ValueError("pattern active-cell count is inconsistent")

        model = cp_model.CpModel()
        active = [model.new_bool_var(f"active_{v}") for v in graph.vertices]
        unspecified = [
            model.new_bool_var(f"unknown_active_{v}") for v in graph.vertices
        ]
        degree = [
            model.new_int_var(0, len(graph.neighbours[v]), f"degree_{v}")
            for v in graph.vertices
        ]
        states = {
            (item, state_degree): [
                model.new_bool_var(f"state_{v}_{item}_{state_degree}")
                for v in graph.vertices
            ]
            for item, state_degree, _count in requested
        }
        side_vent = [model.new_bool_var(f"side_vent_{v}") for v in graph.vertices]
        accepting = [model.new_bool_var(f"accepting_{v}") for v in graph.vertices]

        for vertex in graph.vertices:
            incident_states = [values[vertex] for values in states.values()]
            model.add(
                active[vertex] == unspecified[vertex] + sum(incident_states)
            )
            model.add(
                degree[vertex]
                == sum(active[other] for other in graph.neighbours[vertex])
            )
            for (_item, state_degree), values in states.items():
                model.add(degree[vertex] == state_degree).only_enforce_if(
                    values[vertex]
                )
            # Every cell outside the active set and side-vent set is treated as
            # heat accepting.  This is deliberately more generous than IC2.
            model.add(active[vertex] + side_vent[vertex] + accepting[vertex] == 1)

        for key, values in states.items():
            count = next(
                count
                for item, state_degree, count in requested
                if (item, state_degree) == key
            )
            model.add(sum(values) == count)
        model.add(sum(unspecified) == unknown_active)
        model.add(sum(active) == pattern.active_cells)

        effective_edges = []
        for source in graph.vertices:
            for target in graph.neighbours[source]:
                edge = model.new_bool_var(f"side_edge_{source}_{target}")
                model.add(edge <= side_vent[source])
                model.add(edge <= accepting[target])
                model.add(edge >= side_vent[source] + accepting[target] - 1)
                effective_edges.append(edge)

        minimum_side_cost = (
            ideal_receiver_sink - side_rate * graph.maximum_degree
        )
        exchanger_upper = graph.size - pattern.active_cells
        if minimum_side_cost >= 0:
            exchanger_upper = min(
                exchanger_upper,
                pattern.slack // exchanger_slot_cost,
            )
        exchanger_count = model.new_int_var(0, exchanger_upper, "exchanger_count")
        side_count = sum(side_vent)
        edge_count = sum(effective_edges)
        model.add(
            exchanger_count + side_count <= graph.size - pattern.active_cells
        )
        capacity_loss = (
            ideal_receiver_sink * side_count
            - side_rate * edge_count
            + exchanger_slot_cost * exchanger_count
        )
        model.add(capacity_loss <= pattern.slack)
        model.add(edge_count <= graph.maximum_degree * side_count)
        if minimum_side_cost >= 0:
            # Each side vent has at most Delta useful edges, hence its net
            # baseline-capacity cost is at least sink-side_rate*Delta.
            model.add(
                minimum_side_cost * side_count
                + exchanger_slot_cost * exchanger_count
                <= pattern.slack
            )
        relief = side_rate * edge_count + exchanger_relief * exchanger_count
        model.maximize(relief)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = random_seed
        started = perf_counter()
        status_code = solver.solve(model)
        elapsed = perf_counter() - started
        if status_code == cp_model.MODEL_INVALID:
            raise RuntimeError(f"invalid structural relief model: {model.validate()}")
        status = solver.status_name(status_code)
        feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        proven_optimal = status_code == cp_model.OPTIMAL
        maximum_relief = int(round(solver.objective_value)) if feasible else None
        relief_upper_bound = (
            None
            if status_code == cp_model.INFEASIBLE
            else ceil(solver.best_objective_bound)
        )
        if status_code == cp_model.INFEASIBLE:
            # No geometry can even satisfy the optimistic baseline budget.
            excluded: bool | None = True
        elif relief_upper_bound is not None and relief_upper_bound < pattern.required_relief:
            # CP-SAT's best bound is a rigorous upper bound for maximisation;
            # a complete incumbent layout is unnecessary for this exclusion.
            excluded = True
        elif feasible and (
            proven_optimal or maximum_relief >= pattern.required_relief
        ):
            excluded = maximum_relief < pattern.required_relief
        else:
            excluded = None
        return StructuralReliefResult(
            status=status,
            proven_optimal=proven_optimal,
            maximum_relief=maximum_relief,
            relief_upper_bound=relief_upper_bound,
            excluded=excluded,
            side_vent_cells=(
                sum(solver.value(value) for value in side_vent) if feasible else None
            ),
            effective_side_edges=(
                sum(solver.value(value) for value in effective_edges)
                if feasible
                else None
            ),
            exchanger_cells=(solver.value(exchanger_count) if feasible else None),
            baseline_capacity_loss=(
                ideal_receiver_sink
                * sum(solver.value(value) for value in side_vent)
                - side_rate
                * sum(solver.value(value) for value in effective_edges)
                + exchanger_slot_cost * solver.value(exchanger_count)
                if feasible
                else None
            ),
            elapsed_seconds=elapsed,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
        )
