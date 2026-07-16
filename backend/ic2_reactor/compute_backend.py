from __future__ import annotations

import numpy as np

from .engine import ReactorSimulator, SimulationOptions
from .kernel_abi import (
    COMPONENT_ID_BY_CODE,
    STOP_REASON_CODE,
    PackedEvaluationBatch,
    PackedLayoutBatch,
    encode_mark,
)


class ScalarPackedEvaluator:
    """Authoritative scalar implementation of the packed evaluator contract.

    It deliberately favors semantic transparency over speed.  Future Numba CPU
    and CUDA backends can be compared field-for-field against this adapter.
    """

    def evaluate(
        self,
        batch: PackedLayoutBatch,
        max_reactor_ticks: int,
    ) -> PackedEvaluationBatch:
        size = batch.batch_size
        mark_family = np.zeros(size, dtype=np.uint8)
        mark_level = np.zeros(size, dtype=np.uint8)
        mark_flags = np.zeros(size, dtype=np.uint8)
        stop_reason = np.zeros(size, dtype=np.uint8)
        reactor_ticks = np.zeros(size, dtype=np.int64)
        safe_game_ticks = np.zeros(size, dtype=np.int64)
        average_eu_per_tick = np.zeros(size, dtype=np.float64)
        total_eu = np.zeros(size, dtype=np.float64)
        safety_margin = np.zeros(size, dtype=np.float64)

        for row in range(size):
            component_ids = tuple(
                COMPONENT_ID_BY_CODE[int(code)] for code in batch.component_codes[row]
            )
            simulator = ReactorSimulator.from_slots(
                batch.columns,
                component_ids,
                int(batch.initial_hull_heat[row]),
            )
            run = simulator.simulate(SimulationOptions(
                max_game_ticks=max_reactor_ticks * 20,
                auto_refuel=True,
                stop_on_stable=True,
                record_components=False,
                record_history=False,
            ))
            summary = run.summary
            safe_ticks = summary.first_intervention_tick or summary.game_ticks
            family, level, flags = encode_mark(summary.mark, summary.stable)
            mark_family[row] = family
            mark_level[row] = level
            mark_flags[row] = flags
            stop_reason[row] = STOP_REASON_CODE[summary.stop_reason]
            reactor_ticks[row] = summary.reactor_ticks
            safe_game_ticks[row] = safe_ticks
            average_eu_per_tick[row] = summary.average_eu_per_tick
            total_eu[row] = summary.average_eu_per_tick * safe_ticks
            safety_margin[row] = 1.0 - summary.peak_hull_heat / summary.max_hull_heat

        return PackedEvaluationBatch(
            mark_family=mark_family,
            mark_level=mark_level,
            mark_flags=mark_flags,
            stop_reason=stop_reason,
            reactor_ticks=reactor_ticks,
            safe_game_ticks=safe_game_ticks,
            average_eu_per_tick=average_eu_per_tick,
            total_eu=total_eu,
            safety_margin=safety_margin,
        )
