from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RULESET_VERSION = "ic2-experimental-2.8.221"

ComponentKind = Literal[
    "empty", "fuel", "vent", "exchanger", "coolant", "condensator", "plating", "reflector"
]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    id: str
    name: str
    short_name: str
    kind: ComponentKind
    texture: str | None = None
    max_heat: int = 0
    max_damage: int = 0
    rod_count: int = 0
    energy_factor: int = 0
    heat_factor: int = 0
    internal_pulses: int = 0
    self_vent: int = 0
    hull_draw: int = 0
    side_vent: int = 0
    exchange_side: int = 0
    exchange_hull: int = 0
    hull_capacity_bonus: int = 0
    explosion_multiplier: float = 1.0

    @property
    def accepts_heat(self) -> bool:
        return self.max_heat > 0 and self.kind not in {"fuel", "plating", "reflector"}

    @property
    def is_coolable(self) -> bool:
        return self.accepts_heat and self.kind != "condensator"

    def public_dict(self) -> dict:
        value = asdict(self)
        value["accepts_heat"] = self.accepts_heat
        value["is_coolable"] = self.is_coolable
        return value


def _tex(path: str) -> str:
    return f"/ic2-textures/{path}.png"


_SPECS = [
    ComponentSpec("empty", "空格", "空", "empty"),
    ComponentSpec("uranium_single", "燃料棒（铀）", "单铀", "fuel", _tex("fuel_rod/uranium"), max_damage=20_000, rod_count=1, energy_factor=5, heat_factor=2, internal_pulses=1),
    ComponentSpec("uranium_dual", "双联燃料棒（铀）", "双铀", "fuel", _tex("fuel_rod/dual_uranium"), max_damage=20_000, rod_count=2, energy_factor=10, heat_factor=4, internal_pulses=2),
    ComponentSpec("uranium_quad", "四联燃料棒（铀）", "四铀", "fuel", _tex("fuel_rod/quad_uranium"), max_damage=20_000, rod_count=4, energy_factor=20, heat_factor=8, internal_pulses=3),
    ComponentSpec("heat_vent", "散热片", "散热", "vent", _tex("heat_vent"), max_heat=1_000, self_vent=6),
    ComponentSpec("advanced_heat_vent", "高级散热片", "高散", "vent", _tex("advanced_heat_vent"), max_heat=1_000, self_vent=12),
    ComponentSpec("reactor_heat_vent", "反应堆散热片", "堆散", "vent", _tex("reactor_heat_vent"), max_heat=1_000, self_vent=5, hull_draw=5),
    ComponentSpec("component_heat_vent", "元件散热片", "元散", "vent", _tex("component_heat_vent"), side_vent=4),
    ComponentSpec("overclocked_heat_vent", "超频散热片", "超散", "vent", _tex("overclocked_heat_vent"), max_heat=1_000, self_vent=20, hull_draw=36),
    ComponentSpec("coolant_10k", "10k 冷却单元", "10k", "coolant", _tex("heat_storage"), max_heat=10_000),
    ComponentSpec("coolant_30k", "30k 冷却单元", "30k", "coolant", _tex("tri_heat_storage"), max_heat=30_000),
    ComponentSpec("coolant_60k", "60k 冷却单元", "60k", "coolant", _tex("hex_heat_storage"), max_heat=60_000),
    ComponentSpec("heat_exchanger", "热交换器", "换热", "exchanger", _tex("heat_exchanger"), max_heat=2_500, exchange_side=12, exchange_hull=4),
    ComponentSpec("advanced_heat_exchanger", "高级热交换器", "高换", "exchanger", _tex("advanced_heat_exchanger"), max_heat=10_000, exchange_side=24, exchange_hull=8),
    ComponentSpec("reactor_heat_exchanger", "反应堆热交换器", "堆换", "exchanger", _tex("reactor_heat_exchanger"), max_heat=5_000, exchange_hull=72),
    ComponentSpec("component_heat_exchanger", "元件热交换器", "元换", "exchanger", _tex("component_heat_exchanger"), max_heat=5_000, exchange_side=36),
    ComponentSpec("reactor_plating", "反应堆隔板", "隔板", "plating", _tex("plating"), hull_capacity_bonus=1_000, explosion_multiplier=0.9025),
    ComponentSpec("heat_capacity_plating", "高热容反应堆隔板", "热板", "plating", _tex("heat_plating"), hull_capacity_bonus=2_000, explosion_multiplier=0.9801),
    ComponentSpec("containment_plating", "密封反应堆隔板", "密板", "plating", _tex("containment_plating"), hull_capacity_bonus=500, explosion_multiplier=0.81),
    ComponentSpec("rsh_condensator", "红石冷凝模块", "RSH", "condensator", _tex("rsh_condensator"), max_heat=20_000),
    ComponentSpec("lzh_condensator", "青金石冷凝模块", "LZH", "condensator", _tex("lzh_condensator"), max_heat=100_000),
    ComponentSpec("neutron_reflector", "中子反射板", "反射", "reflector", _tex("neutron_reflector"), max_damage=30_000),
    ComponentSpec("thick_neutron_reflector", "加厚中子反射板", "厚反", "reflector", _tex("thick_neutron_reflector"), max_damage=120_000),
    ComponentSpec("iridium_reflector", "铱中子反射板", "铱反", "reflector", _tex("iridium_reflector")),
]

COMPONENTS: dict[str, ComponentSpec] = {spec.id: spec for spec in _SPECS}
COMPONENT_IDS: tuple[str, ...] = tuple(COMPONENTS)
