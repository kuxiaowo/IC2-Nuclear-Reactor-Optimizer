import pytest

from ic2_reactor.engine import ReactorSimulator, SimulationOptions
from ic2_reactor.models import EventType, Layout, StopReason


def layout(*items: str, columns: int = 3, initial_heat: int = 0) -> Layout:
    return Layout(columns=columns, initial_hull_heat=initial_heat, slots=[*items, *(["empty"] * (columns * 6 - len(items)))])


@pytest.mark.parametrize(
    ("fuel", "eu", "heat"),
    [("uranium_single", 5, 4), ("uranium_dual", 20, 24), ("uranium_quad", 60, 96)],
)
def test_isolated_fuel_goldens(fuel: str, eu: int, heat: int):
    simulator = ReactorSimulator(layout(fuel))
    actual_eu, actual_heat, vented = simulator.step()
    assert (actual_eu, actual_heat, vented, simulator.hull_heat) == (eu, heat, 0, heat)


def test_two_adjacent_single_rods_golden():
    simulator = ReactorSimulator(layout("uranium_single", "uranium_single"))
    eu, heat, _ = simulator.step()
    assert (eu, heat, simulator.hull_heat) == (20, 24, 24)


def test_reactor_vent_stabilizes_isolated_single():
    simulator = ReactorSimulator(layout("uranium_single", "reactor_heat_vent"))
    for _ in range(100):
        eu, generated, vented = simulator.step()
    assert (eu, generated, vented, simulator.hull_heat, simulator.slots[1].heat) == (5, 4, 4, 0, 0)


def test_heat_exchanger_uses_official_immediate_neighbor_transfer():
    simulator = ReactorSimulator(layout("heat_exchanger", "coolant_10k"))
    simulator.slots[1].heat = 100
    simulator.step()
    assert (simulator.slots[0].heat, simulator.slots[1].heat) == (12, 88)


def test_plating_only_changes_capacity_when_its_heat_run_slot_is_reached():
    exchanger_first = ReactorSimulator(layout(
        "reactor_heat_exchanger", "heat_capacity_plating", initial_heat=25
    ))
    plating_first = ReactorSimulator(layout(
        "heat_capacity_plating", "reactor_heat_exchanger", initial_heat=25
    ))
    exchanger_first.step()
    plating_first.step()
    assert exchanger_first.max_hull_heat == plating_first.max_hull_heat == 12_000
    assert (exchanger_first.hull_heat, exchanger_first.slots[0].heat) == (25, 0)
    assert (plating_first.hull_heat, plating_first.slots[1].heat) == (24, 1)


def test_heat_distribution_preserves_integer_remainder_and_order():
    items = ["empty"] * 18
    items[4] = "uranium_single"
    items[7] = items[1] = items[5] = "coolant_10k"
    simulator = ReactorSimulator(Layout(columns=3, slots=items))
    simulator.step()
    # 官方队列按右、上、下依次处理，剩余热量动态整除剩余接收者。
    assert (simulator.slots[7].heat, simulator.slots[1].heat, simulator.slots[5].heat) == (2, 1, 1)


def test_overheated_component_is_removed_and_remaining_quad_cells_heat_hull():
    simulator = ReactorSimulator(layout("uranium_quad", "heat_vent"))
    simulator.slots[1].heat = 999
    eu, generated, _ = simulator.step()
    assert simulator.slots[1].component_id == "empty"
    # 四联燃料按 4 根内部燃料逐根分热。第一份 24 热烧毁散热片，
    # 后三份各 24 热因已无接收者而进入堆体。
    assert (eu, generated, simulator.hull_heat) == (60, 96, 72)
    assert simulator.first_component_break_tick == 1


def test_exact_component_capacity_survives_until_next_positive_heat():
    simulator = ReactorSimulator(layout("uranium_single", "coolant_10k"))
    simulator.slots[1].heat = 9_996
    simulator.step()
    assert (simulator.slots[1].component_id, simulator.slots[1].heat) == ("coolant_10k", 10_000)
    assert simulator.first_component_break_tick is None

    simulator.step()
    assert simulator.slots[1].component_id == "empty"
    assert simulator.hull_heat == 0
    assert simulator.first_component_break_tick == 2


def test_condensator_returns_excess_heat_to_hull_without_breaking():
    simulator = ReactorSimulator(layout("uranium_single", "rsh_condensator"))
    simulator.slots[1].heat = 19_999
    simulator.step()
    assert (simulator.slots[1].component_id, simulator.slots[1].heat) == ("rsh_condensator", 20_000)
    assert simulator.hull_heat == 3
    assert simulator.first_component_break_tick is None


def test_row_major_order_makes_mirrored_vent_layouts_different():
    vent_first = ReactorSimulator(layout("heat_vent", "uranium_single"))
    fuel_first = ReactorSimulator(layout("uranium_single", "heat_vent"))
    vent_first.step()
    fuel_first.step()
    assert vent_first.slots[0].heat == 4
    assert fuel_first.slots[1].heat == 0


def test_quad_fuel_distributes_each_internal_cell_in_official_queue_order():
    items = ["empty"] * 18
    items[4] = "uranium_quad"
    items[3] = "iridium_reflector"
    items[5] = items[1] = items[7] = "coolant_10k"
    simulator = ReactorSimulator(Layout(columns=3, slots=items))
    eu, generated, _ = simulator.step()
    assert (eu, generated) == (80, 160)
    assert (simulator.slots[5].heat, simulator.slots[1].heat, simulator.slots[7].heat) == (52, 52, 56)


@pytest.mark.parametrize(("fuel", "damage"), [("uranium_single", 1), ("uranium_dual", 2), ("uranium_quad", 4)])
def test_reflector_damage_uses_actual_rod_count(fuel: str, damage: int):
    simulator = ReactorSimulator(layout("neutron_reflector", fuel))
    simulator.step()
    assert simulator.slots[0].damage == damage


def test_iridium_reflector_is_infinite():
    simulator = ReactorSimulator(layout("iridium_reflector", "uranium_quad"))
    for _ in range(20):
        simulator.step()
    assert simulator.slots[0].damage == 0
    assert not simulator.slots[0].broken


def test_reflector_break_is_triggered_by_fuel_pulse_not_its_slot_order():
    simulator = ReactorSimulator(layout("neutron_reflector", "uranium_single"))
    simulator.slots[0].damage = 29_999
    eu, generated, _ = simulator.step()
    assert (eu, generated, simulator.hull_heat) == (5, 12, 12)
    assert simulator.slots[0].component_id == "empty"
    assert simulator.first_component_break_tick == 1


def test_suc_suffix_uses_initial_layout_after_finite_reflector_breaks():
    simulator = ReactorSimulator(layout("neutron_reflector", "uranium_single"))
    simulator.slots[0].damage = 29_999
    run = simulator.simulate(SimulationOptions(max_game_ticks=20))
    assert run.summary.mark == "Mark V-SUC"


def test_fuel_lifetime_and_auto_refuel_event():
    simulator = ReactorSimulator(layout("uranium_single", "reactor_heat_vent"))
    run = simulator.simulate(SimulationOptions(max_game_ticks=400_000, auto_refuel=True))
    assert simulator.slots[0].damage == 0
    assert any(event.type == EventType.REFUEL for event in run.summary.events)
    assert not any(event.type == EventType.COMPONENT_BROKEN and event.component_id == "uranium_single" for event in run.summary.events)
    assert run.summary.stop_reason == StopReason.TICK_LIMIT


def test_depleted_fuel_becomes_empty_and_has_dedicated_event():
    simulator = ReactorSimulator(layout("uranium_single", "reactor_heat_vent"))
    run = simulator.simulate(SimulationOptions(max_game_ticks=400_000))
    assert simulator.slots[0].component_id == "empty"
    assert any(event.type == EventType.FUEL_DEPLETED for event in run.summary.events)
    assert not any(event.type == EventType.COMPONENT_BROKEN and event.component_id == "uranium_single" for event in run.summary.events)


def test_summary_only_simulation_matches_full_history():
    full = ReactorSimulator(layout("uranium_quad", "heat_vent"))
    full.slots[1].heat = 999
    summary_only = ReactorSimulator(layout("uranium_quad", "heat_vent"))
    summary_only.slots[1].heat = 999

    expected = full.simulate(SimulationOptions(max_game_ticks=40))
    actual = summary_only.simulate(SimulationOptions(max_game_ticks=40, record_history=False))

    assert actual.records == []
    assert actual.summary.model_dump() == expected.summary.model_dump()


@pytest.mark.parametrize(
    ("items", "expected_peak_heat"),
    [
        (("uranium_single", "reactor_heat_vent"), 0),
        (("reactor_heat_vent", "empty", "uranium_single"), 4),
    ],
)
def test_fixed_temperature_fast_forward_matches_full_mark_i_run(
    items: tuple[str, ...], expected_peak_heat: int
):
    stable_layout = layout(*items)
    expected = ReactorSimulator(stable_layout).simulate(SimulationOptions(
        max_game_ticks=800_000,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=False,
    ))
    actual = ReactorSimulator(stable_layout).simulate(SimulationOptions(
        max_game_ticks=800_000,
        auto_refuel=True,
        stop_on_stable=True,
        record_components=False,
        record_history=False,
    ))

    assert actual.records == []
    assert actual.summary.model_dump() == expected.summary.model_dump()
    assert actual.summary.peak_hull_heat == expected_peak_heat


def test_critical_at_85_percent_and_meltdown_at_100_percent():
    critical = ReactorSimulator(layout("uranium_quad", initial_heat=8_499))
    critical.step()
    assert critical.first_critical_tick == 1
    assert critical.meltdown_tick is None

    meltdown = ReactorSimulator(layout("uranium_quad", initial_heat=9_999))
    run = meltdown.simulate(SimulationOptions(max_game_ticks=20))
    assert run.summary.stop_reason == StopReason.MELTDOWN
    assert run.summary.meltdown_tick == 20


def test_plating_changes_hull_capacity():
    simulator = ReactorSimulator(layout("reactor_plating", "heat_capacity_plating", "containment_plating"))
    assert simulator.max_hull_heat == 13_500


def test_exchange_preserves_locked_double_truncation_boundary():
    # The algebraic real-valued base is 16, but the locked double operation
    # produces 15.999... before Java/Python-style truncation.  A symbolic
    # backend must preserve this finite rule, not silently rationalize it.
    assert ReactorSimulator._exchange_amount(0.4, 1.4, 1_000, 36) == -15
