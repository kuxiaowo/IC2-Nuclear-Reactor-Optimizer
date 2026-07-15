"""Exact permanent-domain and dynamic-state-schema quotients for IC2 labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .components import COMPONENTS


@dataclass(frozen=True, order=True, slots=True)
class IC2DynamicStateSchema:
    heat_maximum: int
    hull_capacity_bonus: int


@dataclass(frozen=True, slots=True)
class IC2PermanentCatalogueQuotient:
    labels: tuple[str, ...]
    removed_finite_reflectors: tuple[str, ...]
    schemas: tuple[tuple[IC2DynamicStateSchema, tuple[str, ...]], ...]

    @property
    def schema_count(self) -> int:
        return len(self.schemas)

    def schema(self, label: str) -> IC2DynamicStateSchema:
        for schema, labels in self.schemas:
            if label in labels:
                return schema
        raise ValueError(f"label is outside the permanent quotient: {label}")


@dataclass(frozen=True, order=True, slots=True)
class IC2StaticPowerBehavior:
    rod_count: int
    internal_pulses: int
    accepts_pulse: bool


@dataclass(frozen=True, order=True, slots=True)
class IC2StructuralSignature:
    dynamic_schema: IC2DynamicStateSchema
    power_behavior: IC2StaticPowerBehavior


@dataclass(frozen=True, slots=True)
class IC2PermanentStructuralQuotient:
    labels: tuple[str, ...]
    groups: tuple[tuple[IC2StructuralSignature, tuple[str, ...]], ...]

    @property
    def signature_count(self) -> int:
        return len(self.groups)


def ic2_permanent_search_representative(label: str) -> str:
    """Map finite reflectors to the only needed permanent representative.

    A finite reflector adjacent to fuel receives strictly increasing damage
    and eventually breaks, so it cannot occur in an indefinitely safe layout.
    If it is not adjacent to fuel, its future pulse/damage behaviour is exactly
    the same as the iridium reflector.  Hence restricting permanent search to
    the iridium representative loses no safe objective value.
    """

    if label not in COMPONENTS:
        raise ValueError(f"unknown IC2 label: {label}")
    spec = COMPONENTS[label]
    if spec.kind == "reflector" and spec.max_damage > 0:
        return "iridium_reflector"
    return label


def ic2_dynamic_state_schema(label: str) -> IC2DynamicStateSchema:
    if label not in COMPONENTS:
        raise ValueError(f"unknown IC2 label: {label}")
    representative = ic2_permanent_search_representative(label)
    spec = COMPONENTS[representative]
    return IC2DynamicStateSchema(
        heat_maximum=spec.max_heat if spec.accepts_heat else 0,
        hull_capacity_bonus=spec.hull_capacity_bonus,
    )


def ic2_static_power_behavior(label: str) -> IC2StaticPowerBehavior:
    if label not in COMPONENTS:
        raise ValueError(f"unknown IC2 label: {label}")
    representative = ic2_permanent_search_representative(label)
    spec = COMPONENTS[representative]
    return IC2StaticPowerBehavior(
        rod_count=spec.rod_count,
        internal_pulses=spec.internal_pulses,
        accepts_pulse=spec.kind in {"fuel", "reflector"},
    )


def ic2_structural_signature(label: str) -> IC2StructuralSignature:
    return IC2StructuralSignature(
        dynamic_schema=ic2_dynamic_state_schema(label),
        power_behavior=ic2_static_power_behavior(label),
    )


def ic2_permanent_catalogue_quotient(
    labels: Iterable[str] | None = None,
) -> IC2PermanentCatalogueQuotient:
    raw = tuple(COMPONENTS if labels is None else labels)
    if not raw or len(raw) != len(set(raw)):
        raise ValueError("IC2 quotient labels must be non-empty and unique")
    if unknown := set(raw) - COMPONENTS.keys():
        raise ValueError(f"unknown IC2 quotient labels: {sorted(unknown)}")
    removed = tuple(
        label
        for label in raw
        if COMPONENTS[label].kind == "reflector"
        and COMPONENTS[label].max_damage > 0
    )
    permanent = tuple(label for label in raw if label not in removed)
    groups: dict[IC2DynamicStateSchema, list[str]] = {}
    for label in permanent:
        groups.setdefault(ic2_dynamic_state_schema(label), []).append(label)
    return IC2PermanentCatalogueQuotient(
        labels=permanent,
        removed_finite_reflectors=removed,
        schemas=tuple(
            (schema, tuple(group))
            for schema, group in sorted(groups.items())
        ),
    )


def ic2_permanent_structural_quotient(
    labels: Iterable[str] | None = None,
) -> IC2PermanentStructuralQuotient:
    schema_quotient = ic2_permanent_catalogue_quotient(labels)
    groups: dict[IC2StructuralSignature, list[str]] = {}
    for label in schema_quotient.labels:
        groups.setdefault(ic2_structural_signature(label), []).append(label)
    return IC2PermanentStructuralQuotient(
        labels=schema_quotient.labels,
        groups=tuple(
            (signature, tuple(group))
            for signature, group in sorted(groups.items())
        ),
    )


def ic2_layout_dynamic_schema_signature(
    layout: Sequence[str],
) -> tuple[IC2DynamicStateSchema, ...]:
    return tuple(ic2_dynamic_state_schema(label) for label in layout)


def ic2_layout_structural_signature(
    layout: Sequence[str],
) -> tuple[IC2StructuralSignature, ...]:
    return tuple(ic2_structural_signature(label) for label in layout)
