"""Power-skeleton master with an unlimited componentwise cooling dominator."""

from __future__ import annotations

from typing import Mapping, Sequence

from .mathematical_model import ReactorProblem
from .periodic_prefix import (
    PeriodicPrefixCutTemplate,
    PrefixHeatComponent,
    componentwise_prefix_dominator,
)
from .thermal_master import (
    PowerSkeletonNoGood,
    ThermalCutMaster,
    ThermalMasterSolution,
)
from .thermal_relaxation import (
    HeatFlowComponent,
    ThermalCutTemplate,
    componentwise_cooling_dominator,
)


class IdealSkeletonThermalMaster:
    """A small, sound necessary model over power labels only.

    Every ``empty`` power slot receives one impossible super-component that
    combines the coordinate-wise best enabled cooling and exchange capacities.
    Fuel heat may freely choose hull or receiver routes through the compact
    flow extension.  Therefore INFEASIBLE is a proof for the original problem;
    FEASIBLE is merely an optimistic power skeleton candidate.
    """

    def __init__(
        self,
        problem: ReactorProblem,
        heat_catalogue: Mapping[str, HeatFlowComponent],
        *,
        prefix_catalogue: Mapping[str, PrefixHeatComponent] | None = None,
        base_hull_capacity: int | None = None,
    ) -> None:
        power_ids = {item.id for item in problem.power_components}
        missing = power_ids - heat_catalogue.keys()
        if missing:
            raise ValueError(f"heat catalogue is missing power labels: {sorted(missing)}")
        power_limits = tuple(
            (label, limit)
            for label, limit in problem.component_limits
            if label in power_ids and label != "empty"
        )
        self.original_problem = problem
        self.relaxed_problem = ReactorProblem(
            graph=problem.graph,
            rod_budget=problem.rod_budget,
            exact_rods=problem.exact_rods,
            power_components=problem.power_components,
            cooling_components=(),
            layout_components=(),
            component_limits=power_limits,
            eu_per_pulse=problem.eu_per_pulse,
            heat_scale=problem.heat_scale,
            ruleset=f"{problem.ruleset}:ideal-skeleton-flow",
        )
        ideal = componentwise_cooling_dominator(problem, heat_catalogue)
        relaxed_catalogue = {
            item.id: heat_catalogue[item.id] for item in problem.power_components
        }
        relaxed_catalogue["empty"] = ideal
        self.ideal_component = ideal
        relaxed_prefix_catalogue = None
        if prefix_catalogue is not None:
            missing_prefix = power_ids - prefix_catalogue.keys()
            if missing_prefix:
                raise ValueError(
                    f"prefix catalogue is missing power labels: {sorted(missing_prefix)}"
                )
            relaxed_prefix_catalogue = {
                item.id: prefix_catalogue[item.id]
                for item in problem.power_components
            }
            relaxed_prefix_catalogue["empty"] = componentwise_prefix_dominator(
                problem,
                prefix_catalogue,
            )
        self.master = ThermalCutMaster(
            self.relaxed_problem,
            relaxed_catalogue,
            prefix_catalogue=relaxed_prefix_catalogue,
            base_hull_capacity=base_hull_capacity,
        )

    def solve(
        self,
        *,
        cuts: Sequence[ThermalCutTemplate] = (),
        prefix_cuts: Sequence[PeriodicPrefixCutTemplate] = (),
        excluded_skeletons: Sequence[Sequence[str]] = (),
        excluded_power_cores: Sequence[PowerSkeletonNoGood] = (),
        seconds: float = 60.0,
        workers: int = 1,
        random_seed: int = 221,
        minimum_power: int | None = None,
        exact_power: int | None = None,
        maximum_power_limit: int | None = None,
        aggregate_fuel_degree_counts: Mapping[tuple[str, int], int] | None = None,
        exact_active_cells: int | None = None,
    ) -> ThermalMasterSolution:
        return self.master.solve(
            cuts=cuts,
            prefix_cuts=prefix_cuts,
            excluded_layouts=excluded_skeletons,
            excluded_power_cores=excluded_power_cores,
            enforce_full_flow=True,
            seconds=seconds,
            workers=workers,
            random_seed=random_seed,
            minimum_power=minimum_power,
            exact_power=exact_power,
            maximum_power_limit=maximum_power_limit,
            aggregate_fuel_degree_counts=aggregate_fuel_degree_counts,
            exact_active_cells=exact_active_cells,
        )
