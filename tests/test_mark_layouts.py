from ic2_reactor.engine import ReactorSimulator, SimulationOptions
from ic2_reactor.models import Layout


def run(items: dict[int, str], reactor_ticks: int, *, stable: bool = False):
    slots = ["empty"] * 18
    for index, component_id in items.items():
        slots[index] = component_id
    return ReactorSimulator(Layout(columns=3, slots=slots)).simulate(SimulationOptions(
        max_game_ticks=reactor_ticks * 20,
        auto_refuel=True,
        stop_on_stable=stable,
        record_components=False,
    )).summary


def test_fixed_layout_mark_i_periodic_steady_state():
    summary = run({0: "uranium_single", 1: "reactor_heat_vent"}, 40_000, stable=True)
    assert summary.mark == "Mark I-I"
    assert summary.stable


def test_fixed_layout_mark_ii_one_full_cycle_before_coolant_break():
    # 中心单联将 4 热均分到上下两个 60k 单元，约第 30,000 次结算损坏。
    summary = run({4: "uranium_single", 1: "coolant_60k", 7: "coolant_60k"}, 31_000)
    assert summary.mark == "Mark II-1"
    assert summary.first_intervention_tick == 600_020


def test_fixed_layout_mark_iii_reaches_critical_after_ten_percent_cycle():
    summary = run({0: "uranium_single"}, 2_200)
    assert summary.mark == "Mark III"


def test_fixed_layout_mark_iv_component_break_after_ten_percent_cycle():
    summary = run({0: "uranium_single", 1: "coolant_10k"}, 2_600)
    assert summary.mark == "Mark IV"
    assert summary.first_intervention_tick == 50_020


def test_fixed_layout_mark_v_early_critical():
    summary = run({0: "uranium_quad"}, 100)
    assert summary.mark == "Mark V"
