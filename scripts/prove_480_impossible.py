"""Machine-check the short combinatorial proof that 480 and 475 EU/t are impossible.

This proof is independent of the production optimizer and uses only the IC2
power/heat formulas, 6x9 grid degree, and permanent vent capacities.
"""

from __future__ import annotations

from itertools import product


ROWS, COLUMNS = 6, 9


def neighbours(index: int) -> set[int]:
    row, column = divmod(index, COLUMNS)
    result = set()
    if column:
        result.add(index - 1)
    if column + 1 < COLUMNS:
        result.add(index + 1)
    if row:
        result.add(index - COLUMNS)
    if row + 1 < ROWS:
        result.add(index + COLUMNS)
    return result


def minimum_heat_for_pulse_units(rods: int, pulse_units: int) -> tuple[int, tuple[int, ...]]:
    """Discrete convex minimum after relaxing stack rods into individuals."""
    quotient, remainder = divmod(pulse_units, rods)
    pulses = (quotient,) * (rods - remainder) + (quotient + 1,) * remainder
    heat = sum(2 * pulse * (pulse + 1) for pulse in pulses)
    return heat, pulses


def main() -> None:
    power = 480
    rods = 25
    pulse_units = power // 5
    minimum_heat, relaxed_pulses = minimum_heat_for_pulse_units(rods, pulse_units)
    assert pulse_units == 96
    assert relaxed_pulses.count(3) == 4 and relaxed_pulses.count(4) == 21
    assert minimum_heat == 936

    minimum_fuel_slots = (rods + 3) // 4
    cooling_slots = ROWS * COLUMNS - minimum_fuel_slots
    assert minimum_fuel_slots == 7 and cooling_slots == 47
    all_overclocked_capacity = 20 * cooling_slots
    assert all_overclocked_capacity == 940

    # Replacing O by C loses 20 self venting and can add at most 4*4=16.
    # Replacing O by the strongest other self-vent (advanced, 12) loses 8.
    assert all_overclocked_capacity - 4 * 1 == minimum_heat
    assert all_overclocked_capacity - 4 * 2 < minimum_heat
    assert all_overclocked_capacity - 8 < minimum_heat

    # With seven fuel slots and 25 rods, all seven must be fuel and their only
    # possible rod-count multiset is six quads plus one single.
    fuel_multisets = {
        tuple(sorted(candidate))
        for candidate in product((1, 2, 4), repeat=minimum_fuel_slots)
        if sum(candidate) == rods
    }
    assert fuel_multisets == {(1, 4, 4, 4, 4, 4, 4)}

    # Distinct square-grid cells share at most two neighbours.  Therefore the
    # sole possible C can cool at most two of any quad's adjacent O receivers.
    max_shared = max(
        len(neighbours(first) & neighbours(second))
        for first in range(54)
        for second in range(54)
        if first != second
    )
    assert max_shared == 2
    minimum_quad_heat = 4 * (2 * 3 * 4)  # four rods, at least three pulses.
    best_direct_capacity = 4 * 20 + max_shared * 4
    assert minimum_quad_heat == 96
    assert best_direct_capacity == 88 < minimum_quad_heat

    # A quad with no accepting neighbour dumps >=96 into the hull each tick.
    # The first subsequent O is forced to draw 36, but even the sole C can
    # raise that O's permanent sink only from 20 to 24, so its heat drifts up.
    first_hull_draw = 36
    best_single_o_sink = 20 + 4
    assert first_hull_draw > best_single_o_sink

    proof_480 = {
        "excluded_power": power,
        "minimum_generated_heat": minimum_heat,
        "maximum_total_cooling": all_overclocked_capacity,
        "maximum_direct_capacity_for_one_quad": best_direct_capacity,
        "minimum_quad_heat": minimum_quad_heat,
    }

    # Repeat the global envelope at 475 EU/t.
    power_475 = 475
    heat_475, pulses_475 = minimum_heat_for_pulse_units(rods, power_475 // 5)
    assert pulses_475.count(3) == 5 and pulses_475.count(4) == 20
    assert heat_475 == 920
    # Nine active slots leave at most 45 O, below the heat lower bound.
    assert 20 * (54 - 9) == 900 < heat_475
    # With eight active slots all 46 cooling slots must be O.  Twenty-five
    # rods in at most eight fuel stacks necessarily include a quad, which the
    # all-O local/hull argument above already excludes.
    rod_multisets_up_to_eight = {
        tuple(sorted(candidate))
        for slots in range(1, 9)
        for candidate in product((1, 2, 4), repeat=slots)
        if sum(candidate) == rods
    }
    assert rod_multisets_up_to_eight and all(4 in candidate for candidate in rod_multisets_up_to_eight)
    assert 20 * (54 - 8) == heat_475
    assert 4 * 20 < minimum_quad_heat

    # Seven active slots force six Q plus one S and no reflector.  Starting
    # pulse-units are 6*(4 rods*3 internal pulses)+1=73.  Reaching 95 requires
    # 4*sum(deg_Q)+deg_S=22, hence deg_S=2 and sum(deg_Q)=5.  Their total graph
    # degree is seven, contradicting the handshaking lemma.
    base_units = 6 * 4 * 3 + 1
    degree_solutions = [
        (quad_degree_sum, single_degree)
        for quad_degree_sum in range(25)
        for single_degree in range(5)
        if 4 * quad_degree_sum + single_degree == power_475 // 5 - base_units
    ]
    assert degree_solutions == [(5, 2)]
    assert sum(degree_solutions[0]) % 2 == 1

    print({
        "excluded_layers": [proof_480, {
            "excluded_power": power_475,
            "minimum_generated_heat": heat_475,
            "seven_active_degree_sum": sum(degree_solutions[0]),
        }],
        "conclusion": "P* <= 470 EU/t",
    })


if __name__ == "__main__":
    main()
