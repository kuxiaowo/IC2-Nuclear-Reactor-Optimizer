"""Reproduce the certified 25-rod, 380 EU/t safe periodic witness."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ic2_reactor.components import COMPONENTS  # noqa: E402
from ic2_reactor.engine import ReactorSimulator  # noqa: E402
from ic2_reactor.models import Layout  # noqa: E402


ROWS = (
    "QCOXOOCRP",
    ".COOCORSR",
    "POQOOQOOX",
    "COOCOOCOO",
    "OQOOQOOQO",
    "COCPOCPOC",
)
SYMBOLS = {
    ".": "empty",
    "S": "uranium_single",
    "Q": "uranium_quad",
    "R": "iridium_reflector",
    "O": "overclocked_heat_vent",
    "C": "component_heat_vent",
    "X": "component_heat_exchanger",
    "P": "reactor_plating",
}
LAYOUT = tuple(SYMBOLS[symbol] for symbol in "".join(ROWS))


def main() -> None:
    rods = sum(COMPONENTS[component].rod_count for component in LAYOUT)
    if rods != 25:
        raise AssertionError(f"expected 25 rods, got {rods}")
    simulator = ReactorSimulator(Layout(columns=9, slots=list(LAYOUT)))
    seen = {simulator.state_signature(include_fuel_damage=False): 0}
    peak_component_heat = 0
    for tick in range(1, 100_001):
        power, heat, _vented = simulator.step(auto_refuel=True)
        peak_component_heat = max(peak_component_heat, *(slot.heat for slot in simulator.slots))
        if simulator.first_component_break_tick or simulator.first_critical_tick:
            raise AssertionError(
                f"unsafe at tick {tick}: break={simulator.first_component_break_tick}, "
                f"critical={simulator.first_critical_tick}"
            )
        signature = simulator.state_signature(include_fuel_damage=False)
        previous = seen.get(signature)
        if previous is not None:
            certificate = {
                "rods": rods,
                "power_eu_per_tick": power,
                "generated_heat_per_tick": heat,
                "transient": previous,
                "period": tick - previous,
                "repeat_tick": tick,
                "peak_hull_heat": simulator.peak_hull_heat,
                "peak_component_heat": peak_component_heat,
                "events": len(simulator.events),
            }
            if power != 380.0 or heat != 616 or previous != 380 or tick - previous != 18:
                raise AssertionError(f"unexpected certificate: {certificate}")
            print(certificate)
            for row in ROWS:
                print(" ".join(row))
            return
        seen[signature] = tick
    raise AssertionError("no repeated safe state within 100,000 ticks")


if __name__ == "__main__":
    main()
