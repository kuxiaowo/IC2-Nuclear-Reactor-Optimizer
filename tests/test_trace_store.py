from ic2_reactor.engine import ReactorSimulator, SimulationOptions
from ic2_reactor.models import Layout
from ic2_reactor.trace_store import TraceStore


def test_trace_expands_reactor_update_to_twenty_game_ticks(tmp_path):
    layout = Layout(columns=3, slots=["uranium_single", *(["empty"] * 17)])
    run = ReactorSimulator(layout).simulate(SimulationOptions(max_game_ticks=40))
    store = TraceStore(tmp_path)
    store.write("golden", run)
    page = store.page("golden", 0, 40)
    assert page["total"] == 40
    assert page["rows"][0]["eu_per_tick"] == 5
    assert page["rows"][0]["total_eu"] == 5
    assert page["rows"][19]["total_eu"] == 100
    assert page["rows"][19]["generated_heat"] == 4
    assert page["rows"][20]["total_eu"] == 105
    assert page["rows"][20]["generated_heat"] == 0


def test_component_snapshot_and_chart_downsample(tmp_path):
    layout = Layout(columns=3, slots=["uranium_single", "coolant_10k", *(["empty"] * 16)])
    run = ReactorSimulator(layout).simulate(SimulationOptions(max_game_ticks=200))
    store = TraceStore(tmp_path)
    store.write("trace", run)
    assert len(store.components_at("trace", 1)["components"]) == 18
    assert len(store.chart("trace", 10)["points"]) <= 10

