"""Targeted anytime search for high-power 25-rod Mark-I layouts.

The search is intentionally separate from the production optimizer.  It uses
the production simulator only as an exact transition oracle and detects any
reachable safe thermal cycle, rather than requiring a period-one fixed point.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from itertools import repeat
import random
from dataclasses import dataclass
from time import perf_counter

from ic2_reactor.components import COMPONENTS
from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.models import Layout
from ic2_reactor.optimizer import power_skeleton, skeleton_heat_per_tick, theoretical_eu_per_tick


BASE_420_CODE = (
    "030C0D140D0D0C0D15150C0D0D0C0D0D030D150D030D0D030D0D0C0C0D0D"
    "0C0D0D0C0D150D030D0D030D0D030D150D0C150D0C150D0C"
)
LEGACY_COMPONENTS = {
    "00": "empty",
    "03": "uranium_quad",
    "0C": "component_heat_vent",
    "0D": "overclocked_heat_vent",
    "14": "component_heat_exchanger",
    "15": "reactor_plating",
}
COOLING_TYPES = (
    "empty",
    "heat_vent",
    "advanced_heat_vent",
    "reactor_heat_vent",
    "component_heat_vent",
    "overclocked_heat_vent",
    "heat_exchanger",
    "advanced_heat_exchanger",
    "reactor_heat_exchanger",
    "component_heat_exchanger",
    "reactor_plating",
    "heat_capacity_plating",
    "containment_plating",
)


def base_layout() -> tuple[str, ...]:
    return tuple(
        LEGACY_COMPONENTS[BASE_420_CODE[offset : offset + 2]]
        for offset in range(0, len(BASE_420_CODE), 2)
    )


@dataclass(frozen=True, slots=True)
class ThermalScore:
    periodic: bool
    survival: int
    total_heat: int
    maximum_ratio_ppm: int
    period: int
    transient: int
    broken_slot: int
    tail_growth: int

    def fitness(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.periodic),
            self.survival,
            -max(0, self.tail_growth),
            -self.maximum_ratio_ppm,
            -self.total_heat,
        )


def evaluate_thermal(layout: tuple[str, ...], horizon: int) -> ThermalScore:
    simulator = ReactorSimulator(Layout(columns=9, slots=list(layout)))
    seen = {simulator.state_signature(include_fuel_damage=False): 0}
    midpoint_heat = 0
    for tick in range(1, horizon + 1):
        previous_total_heat = simulator.hull_heat + sum(slot.heat for slot in simulator.slots)
        previous_ratios = [
            slot.heat / slot.spec.max_heat
            for slot in simulator.slots
            if slot.spec.max_heat > 0
        ]
        previous_maximum_ratio = max(
            [simulator.hull_heat / simulator.max_hull_heat, *previous_ratios],
            default=0.0,
        )
        simulator.step(auto_refuel=True)
        ratios = [
            slot.heat / slot.spec.max_heat
            for slot in simulator.slots
            if slot.spec.max_heat > 0
        ]
        maximum_ratio = max(
            [simulator.hull_heat / simulator.max_hull_heat, *ratios],
            default=0.0,
        )
        current_total_heat = simulator.hull_heat + sum(slot.heat for slot in simulator.slots)
        if tick == max(1, horizon // 2):
            midpoint_heat = current_total_heat
        if (
            simulator.first_critical_tick is not None
            or simulator.first_component_break_tick is not None
            or simulator.meltdown_tick is not None
        ):
            return ThermalScore(
                False,
                tick,
                previous_total_heat,
                round(previous_maximum_ratio * 1_000_000),
                0,
                0,
                next((event.slot for event in reversed(simulator.events) if event.slot is not None), -1),
                max(0, previous_total_heat - midpoint_heat) if midpoint_heat else previous_total_heat,
            )
        signature = simulator.state_signature(include_fuel_damage=False)
        previous = seen.get(signature)
        if previous is not None:
            return ThermalScore(
                True,
                horizon,
                simulator.hull_heat + sum(slot.heat for slot in simulator.slots),
                round(maximum_ratio * 1_000_000),
                tick - previous,
                previous,
                -1,
                0,
            )
        seen[signature] = tick
    return ThermalScore(
        False,
        horizon,
        simulator.hull_heat + sum(slot.heat for slot in simulator.slots),
        round(maximum_ratio * 1_000_000),
        0,
        0,
        -1,
        max(0, simulator.hull_heat + sum(slot.heat for slot in simulator.slots) - midpoint_heat),
    )


def neighbors(index: int) -> tuple[int, ...]:
    row, column = divmod(index, 9)
    result = []
    if column:
        result.append(index - 1)
    if column < 8:
        result.append(index + 1)
    if row:
        result.append(index - 9)
    if row < 5:
        result.append(index + 9)
    return tuple(result)


def initial_layouts(target_power: int) -> list[tuple[str, ...]]:
    """Build structured seeds derived from the periodic 420 EU/t layout."""
    base = base_layout()
    quad_positions = [index for index, item in enumerate(base) if item == "uranium_quad"]
    result = []
    if target_power == 385:
        # CP-SAT minimum-heat (632/t) skeleton.  The older 420-derived seeds
        # bottom out at 644/t for this power tier and cannot mutate fuel or
        # reflector positions, so they would never reach this component.
        skeleton_rows = (
            ".D....Q..",
            ".RD......",
            "Q..RD....",
            "..RD.....",
            "Q.......R",
            "..Q....RS",
        )
        skeleton = "".join(skeleton_rows)
        component_for_symbol = {
            "S": "uranium_single",
            "D": "uranium_dual",
            "Q": "uranium_quad",
            "R": "iridium_reflector",
        }
        trial = [
            "overclocked_heat_vent" if COMPONENTS[item].kind in {"fuel", "reflector"} else item
            for item in base
        ]
        for index, symbol in enumerate(skeleton):
            if symbol != ".":
                trial[index] = component_for_symbol[symbol]
        assert theoretical_eu_per_tick(tuple(trial), 9) == 385
        assert skeleton_heat_per_tick(power_skeleton(tuple(trial)), 9) == 632
        result.append(tuple(trial))
    # Six quads plus one single remain in their original seven fuel slots.
    # Add a small number of iridium reflectors to explore 370--405 EU/t tiers.
    for removed in quad_positions:
        reduced = list(base)
        reduced[removed] = "uranium_single"
        free = [index for index, item in enumerate(reduced) if COMPONENTS[item].kind not in {"fuel", "reflector"}]
        for reflector_count in range(4):
            for reflector_positions in combinations(free, reflector_count):
                trial = reduced.copy()
                for position in reflector_positions:
                    trial[position] = "iridium_reflector"
                if (
                    theoretical_eu_per_tick(tuple(trial), 9) == target_power
                    and all(
                        any(COMPONENTS[trial[other]].kind == "fuel" for other in neighbors(position))
                        for position in reflector_positions
                    )
                ):
                    result.append(tuple(trial))

    # Moving the single next to a remaining quad creates the 390 EU/t tier.
    for removed in quad_positions:
        for target in range(54):
            if target in quad_positions:
                continue
            trial = list(base)
            trial[removed] = base[target]
            trial[target] = "uranium_single"
            if theoretical_eu_per_tick(tuple(trial), 9) == target_power:
                result.append(tuple(trial))

    # Preserve the full O/C cooling inventory by replacing a zero-cooling
    # plating with the reflector and moving a quad beside it.  The vacated
    # quad slot receives the cooling component displaced by that move.
    plating_positions = [index for index, item in enumerate(base) if COMPONENTS[item].kind == "plating"]
    for reflector_position in plating_positions:
        for target in neighbors(reflector_position):
            if COMPONENTS[base[target]].kind in {"fuel", "reflector"}:
                continue
            for moved_quad in quad_positions:
                if moved_quad == target:
                    continue
                for reduced_quad in quad_positions:
                    if reduced_quad == moved_quad:
                        continue
                    trial = list(base)
                    trial[reflector_position] = "iridium_reflector"
                    trial[target] = "uranium_quad"
                    trial[moved_quad] = base[target]
                    trial[reduced_quad] = "uranium_single"
                    if theoretical_eu_per_tick(tuple(trial), 9) == target_power:
                        result.append(tuple(trial))
    return list(dict.fromkeys(result))


def mutate(
    layout: tuple[str, ...],
    fuel_positions: frozenset[int],
    rng: random.Random,
    score: ThermalScore | None = None,
    preserve_inventory: bool = False,
) -> tuple[str, ...]:
    values = list(layout)
    free = [index for index in range(54) if index not in fuel_positions]
    if score is not None and score.broken_slot in free and rng.random() < 0.75:
        broken = score.broken_slot
        repairable_neighbors = [index for index in neighbors(broken) if index in free]
        if repairable_neighbors:
            target = rng.choice(repairable_neighbors)
            if preserve_inventory:
                sources = [index for index in free if values[index] == "component_heat_vent" and index != target]
                if sources:
                    source = rng.choice(sources)
                    values[target], values[source] = values[source], values[target]
            else:
                values[target] = "component_heat_vent"
        if not preserve_inventory and rng.random() < 0.25:
            values[broken] = rng.choice(("overclocked_heat_vent", "advanced_heat_vent", "reactor_heat_vent"))
    changes = 1 + int(rng.random() < 0.7) + rng.randrange(5) * int(rng.random() < 0.25)
    for _ in range(changes):
        if rng.random() < 0.55:
            first, second = rng.sample(free, 2)
            values[first], values[second] = values[second], values[first]
        elif not preserve_inventory:
            values[rng.choice(free)] = rng.choice(COOLING_TYPES)
    return tuple(values)


def search(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    seeds = initial_layouts(args.target_power)
    if not seeds:
        raise RuntimeError(f"no {args.target_power} EU/t structured seed skeletons")
    started = perf_counter()
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    def evaluate_many(layouts: list[tuple[str, ...]]) -> list[ThermalScore]:
        if executor is None:
            return [evaluate_thermal(layout, args.horizon) for layout in layouts]
        chunksize = max(1, len(layouts) // (args.workers * 4))
        return list(executor.map(
            evaluate_thermal,
            layouts,
            repeat(args.horizon),
            chunksize=chunksize,
        ))

    try:
        seed_scores = dict(zip(seeds, evaluate_many(seeds), strict=True))
        eligible_seeds = seeds
        if args.minimum_heat_seed:
            seed_heats = {
                item: skeleton_heat_per_tick(power_skeleton(item), 9)
                for item in seeds
            }
            minimum_seed_heat = min(seed_heats.values())
            eligible_seeds = [item for item in seeds if seed_heats[item] == minimum_seed_heat]
            print(f"minimum_seed_heat={minimum_seed_heat} eligible_seed_count={len(eligible_seeds)}")
        seed = max(eligible_seeds, key=lambda item: seed_scores[item].fitness())
        fuel_positions = frozenset(
            index for index, item in enumerate(seed) if COMPONENTS[item].kind in {"fuel", "reflector"}
        )
        population = list(dict.fromkeys([
            seed,
            *(mutate(seed, fuel_positions, rng, preserve_inventory=args.preserve_inventory) for _ in range(args.population * 2)),
        ]))
        population = population[: args.population]
        cache: dict[tuple[str, ...], ThermalScore] = dict(seed_scores)
        best_layout = seed
        best_score = seed_scores[seed]
        print(f"seed_count={len(seeds)} best_seed={best_score}")

        for generation in range(1, args.generations + 1):
            unseen = [layout for layout in population if layout not in cache]
            if unseen:
                cache.update(zip(unseen, evaluate_many(unseen), strict=True))
            ranked = sorted(population, key=lambda item: cache[item].fitness(), reverse=True)
            if cache[ranked[0]].fitness() > best_score.fitness():
                best_layout, best_score = ranked[0], cache[ranked[0]]
                print(
                    f"generation={generation} score={best_score} "
                    f"evaluated={len(cache)} elapsed={perf_counter() - started:.2f}s"
                )
            periodic = [item for item in ranked if cache[item].periodic]
            if periodic:
                best_layout = periodic[0]
                best_score = cache[best_layout]
                break
            elite_count = max(8, args.population // 8)
            elites = ranked[:elite_count]
            population = list(elites)
            while len(population) < args.population:
                parent = rng.choice(elites)
                population.append(mutate(
                    parent,
                    fuel_positions,
                    rng,
                    cache[parent],
                    preserve_inventory=args.preserve_inventory,
                ))
    finally:
        if executor is not None:
            executor.shutdown()

    power = theoretical_eu_per_tick(best_layout, 9)
    generated = skeleton_heat_per_tick(power_skeleton(best_layout), 9)
    print(
        f"best power={power} generated_heat={generated} score={best_score} "
        f"evaluated={len(cache)} elapsed={perf_counter() - started:.2f}s"
    )
    symbols = {
        "empty": ".",
        "uranium_single": "S",
        "uranium_dual": "D",
        "uranium_quad": "Q",
        "iridium_reflector": "R",
        "heat_vent": "H",
        "advanced_heat_vent": "A",
        "reactor_heat_vent": "V",
        "component_heat_vent": "C",
        "overclocked_heat_vent": "O",
        "heat_exchanger": "x",
        "advanced_heat_exchanger": "a",
        "reactor_heat_exchanger": "r",
        "component_heat_exchanger": "X",
        "reactor_plating": "P",
        "heat_capacity_plating": "T",
        "containment_plating": "N",
    }
    for row in range(6):
        print(" ".join(symbols[item] for item in best_layout[row * 9 : (row + 1) * 9]))
    print(repr(best_layout))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--population", type=int, default=300)
    result.add_argument("--generations", type=int, default=200)
    result.add_argument("--horizon", type=int, default=600)
    result.add_argument("--seed", type=int, default=221)
    result.add_argument("--target-power", type=int, default=390)
    result.add_argument("--workers", type=int, default=30)
    result.add_argument("--preserve-inventory", action="store_true")
    result.add_argument("--minimum-heat-seed", action="store_true")
    return result


if __name__ == "__main__":
    search(parser().parse_args())
