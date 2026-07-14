from __future__ import annotations

from enum import StrEnum
import os
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .components import COMPONENTS, RULESET_VERSION


class StopReason(StrEnum):
    MELTDOWN = "meltdown"
    TICK_LIMIT = "tick_limit"
    STABLE = "stable"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    CRITICAL = "critical"
    COMPONENT_BROKEN = "component_broken"
    FUEL_DEPLETED = "fuel_depleted"
    REFUEL = "refuel"
    MELTDOWN = "meltdown"
    STABLE = "stable"


class Layout(BaseModel):
    ruleset: str = RULESET_VERSION
    columns: Annotated[int, Field(ge=3, le=9)] = 6
    initial_hull_heat: Annotated[int, Field(ge=0)] = 0
    slots: list[str]

    @model_validator(mode="after")
    def validate_layout(self) -> "Layout":
        if self.ruleset != RULESET_VERSION:
            raise ValueError(f"当前仅支持规则集 {RULESET_VERSION}")
        if len(self.slots) != self.columns * 6:
            raise ValueError("slots 数量必须等于 6 × columns")
        unknown = sorted(set(self.slots) - COMPONENTS.keys())
        if unknown:
            raise ValueError(f"未知组件: {', '.join(unknown)}")
        max_heat = 10_000 + sum(COMPONENTS[item].hull_capacity_bonus for item in self.slots)
        if self.initial_hull_heat >= max_heat:
            raise ValueError("初始堆热必须小于最大堆热")
        return self


class SimulationRequest(BaseModel):
    layout: Layout
    max_game_ticks: Annotated[int, Field(ge=20, le=200_000_000)] = 400_000
    auto_refuel: bool = False
    stop_on_stable: bool = False
    record_components: bool = True

    @field_validator("max_game_ticks")
    @classmethod
    def align_ticks(cls, value: int) -> int:
        if value % 20:
            raise ValueError("max_game_ticks 必须是 20 的倍数")
        return value


class ReactorEvent(BaseModel):
    reactor_tick: int
    game_tick: int
    type: EventType
    slot: int | None = None
    component_id: str | None = None
    message: str


class SimulationSummary(BaseModel):
    stop_reason: StopReason
    reactor_ticks: int
    game_ticks: int
    hull_heat: int
    max_hull_heat: int
    peak_hull_heat: int
    current_eu_per_tick: float
    average_eu_per_tick: float
    total_eu: float
    first_intervention_tick: int | None
    meltdown_tick: int | None
    mark: str | None
    stable: bool
    events: list[ReactorEvent]


class SimulationCreated(BaseModel):
    id: str
    summary: SimulationSummary


class FuelConstraint(BaseModel):
    mode: Literal["separate", "total_rods"] = "separate"
    single: Annotated[int, Field(ge=0, le=54)] = 1
    dual: Annotated[int, Field(ge=0, le=54)] = 0
    quad: Annotated[int, Field(ge=0, le=54)] = 0
    total_rods: Annotated[int, Field(ge=0, le=216)] = 1


class OptimizationRequest(BaseModel):
    columns: Annotated[int, Field(ge=3, le=9)] = 3
    fuel: FuelConstraint = Field(default_factory=FuelConstraint)
    component_limits: dict[str, Annotated[int, Field(ge=0, le=54)]] = Field(default_factory=dict)
    marks: list[Literal["I", "II", "III", "IV", "V"]] = Field(default_factory=lambda: ["I"])
    solver: Literal["heuristic", "exhaustive"] = "heuristic"
    time_budget_seconds: Annotated[int, Field(ge=1, le=86_400)] = 30
    generations: Annotated[int, Field(ge=1, le=10_000)] = 100
    population: Annotated[int, Field(ge=10, le=2_000)] = 100
    cpu_workers: Annotated[int, Field(ge=1, le=64)] = max(1, (os.cpu_count() or 2) - 1)
    accelerator: Literal["auto", "cpu", "cuda", "cuda_full"] = "auto"
    gpu_batch_multiplier: Annotated[int, Field(ge=1, le=128)] = 32
    gpu_exhaustive_batch_size: Annotated[int, Field(ge=256, le=131_072)] = 8_192
    gpu_ticks_per_launch: Annotated[int, Field(ge=16, le=2_048)] = 256
    result_limit: Annotated[int, Field(ge=1, le=10)] = 1
    seed: int = 221
    max_reactor_ticks: Annotated[int, Field(ge=2_000, le=1_000_000)] = 40_000

    @field_validator("component_limits")
    @classmethod
    def validate_limits(cls, value: dict[str, int]) -> dict[str, int]:
        invalid = [key for key in value if key not in COMPONENTS or COMPONENTS[key].kind in {"empty", "fuel"}]
        if invalid:
            raise ValueError(f"无效的非燃料组件限制: {', '.join(invalid)}")
        return value

    @field_validator("marks")
    @classmethod
    def unique_marks(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("至少选择一个 Mark")
        return list(dict.fromkeys(value))
