"""Exact future quotient for directed single-commodity flow fragments.

For a fragment whose only connection to the unprocessed graph is a terminal
set ``B``, record for every source-side subset ``A <= B`` the minimum internal
directed-cut capacity with exactly ``A`` on the source side.  This terminal cut
function is a complete interface under graph gluing:

* independent fragments add pointwise;
* adding an edge adds its local cut indicator;
* forgetting a terminal minimizes over its two possible sides.

When the only question is whether a flow of ``q`` units exists, every value at
least ``q`` is future-equivalent and may be saturated at ``q``.  For a fixed
terminal order the resulting vector is the exact Myhill--Nerode quotient of a
flow fragment under arbitrary non-negative-capacity completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TerminalCutFactorScope:
    """Structural scope of one label-dependent local cut factor."""

    layout_scope: tuple[int, ...]
    cut_scope: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if not self.layout_scope or len(self.layout_scope) != len(set(self.layout_scope)):
            raise ValueError("terminal-cut layout scope must be non-empty and unique")
        if not self.cut_scope or len(self.cut_scope) != len(set(self.cut_scope)):
            raise ValueError("terminal-cut cut scope must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class TerminalCutScheduleProfile:
    """Exact separator size for a declared factor/event schedule.

    A factor becomes available when the last layout variable in its scope is
    placed.  A cut terminal is introduced at its first available factor and
    forgotten immediately after its last one.  ``live_terminals_during_step``
    conservatively groups equal-step factors; reordering within a step can
    only improve this profile.
    """

    factor_count: int
    placement_steps: int
    maximum_layout_scope: int
    maximum_cut_scope: int
    live_terminals_during_step: tuple[int, ...]
    peak_live_terminals: int
    peak_cut_vector_entries: int
    distinct_cut_terminals: int = 0
    factor_events_by_step: tuple[int, ...] = ()
    terminal_introductions_by_step: tuple[int, ...] = ()
    terminal_forgets_by_step: tuple[int, ...] = ()
    coarse_full_scan_value_operations_bound: int = 0


def terminal_cut_schedule_profile(
    placement_order: Sequence[int],
    factors: Sequence[TerminalCutFactorScope],
) -> TerminalCutScheduleProfile:
    """Return the cut-vector width without enumerating labels or layouts."""

    order = tuple(placement_order)
    if not order or len(order) != len(set(order)):
        raise ValueError("terminal-cut placement order must be non-empty and unique")
    rank = {vertex: step for step, vertex in enumerate(order)}
    factor_tuple = tuple(factors)
    if unknown := {
        vertex
        for factor in factor_tuple
        for vertex in factor.layout_scope
        if vertex not in rank
    }:
        raise ValueError(
            f"terminal-cut factor uses unknown layout vertices: {sorted(unknown)}"
        )

    event_steps = tuple(
        max(rank[vertex] for vertex in factor.layout_scope)
        for factor in factor_tuple
    )
    first_event: dict[Hashable, int] = {}
    last_event: dict[Hashable, int] = {}
    for factor, event in zip(factor_tuple, event_steps, strict=True):
        for terminal in factor.cut_scope:
            first_event[terminal] = min(first_event.get(terminal, event), event)
            last_event[terminal] = max(last_event.get(terminal, event), event)
    live = tuple(
        sum(
            first_event[terminal] <= step <= last_event[terminal]
            for terminal in first_event
        )
        for step in range(len(order))
    )
    peak = max(live, default=0)
    factor_events = tuple(
        sum(event == step for event in event_steps)
        for step in range(len(order))
    )
    introductions = tuple(
        sum(event == step for event in first_event.values())
        for step in range(len(order))
    )
    forgets = tuple(
        sum(event == step for event in last_event.values())
        for step in range(len(order))
    )
    # Each introduction, factor addition and forgetting operation touches no
    # more than the largest vector present during that event.  This ignores
    # the cheap scalar bookkeeping and is intentionally conservative.
    value_operations = sum(
        (1 << width) * (added + factors_at_step + removed)
        for width, added, factors_at_step, removed in zip(
            live,
            introductions,
            factor_events,
            forgets,
            strict=True,
        )
    )
    return TerminalCutScheduleProfile(
        factor_count=len(factor_tuple),
        placement_steps=len(order),
        maximum_layout_scope=max(
            (len(factor.layout_scope) for factor in factor_tuple),
            default=0,
        ),
        maximum_cut_scope=max(
            (len(factor.cut_scope) for factor in factor_tuple),
            default=0,
        ),
        live_terminals_during_step=live,
        peak_live_terminals=peak,
        peak_cut_vector_entries=1 << peak,
        distinct_cut_terminals=len(first_event),
        factor_events_by_step=factor_events,
        terminal_introductions_by_step=introductions,
        terminal_forgets_by_step=forgets,
        coarse_full_scan_value_operations_bound=value_operations,
    )


def terminal_cut_frontier_orders(
    vertices: Sequence[int],
    factors: Sequence[TerminalCutFactorScope],
    *,
    beam_width: int = 64,
    deadline: float | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Find low terminal-width layout orders without visiting any layouts.

    The search state is only a subset of already placed variables.  A local
    cut factor becomes ready when its last layout variable is placed.  During
    that event the signature needs the old live terminals plus the terminals
    of newly ready factors; afterwards a terminal survives exactly when both
    ready and unready incident factors remain.  Beam truncation can miss the
    best order, but every returned order and its measured width remain exact.
    """

    domain = tuple(vertices)
    if not domain or len(domain) != len(set(domain)):
        raise ValueError("terminal-cut order vertices must be non-empty and unique")
    if beam_width <= 0:
        raise ValueError("terminal-cut order beam width must be positive")
    vertex_position = {vertex: index for index, vertex in enumerate(domain)}
    known = set(domain)
    factor_tuple = tuple(factors)
    if unknown := {
        vertex
        for factor in factor_tuple
        for vertex in factor.layout_scope
        if vertex not in known
    }:
        raise ValueError(
            f"terminal-cut factor uses unknown layout vertices: {sorted(unknown)}"
        )
    if not factor_tuple:
        reverse = tuple(reversed(domain))
        return (domain,) if reverse == domain else (domain, reverse)

    cut_terminals = tuple(dict.fromkeys(
        terminal for factor in factor_tuple for terminal in factor.cut_scope
    ))
    cut_position = {
        terminal: index for index, terminal in enumerate(cut_terminals)
    }
    layout_masks = tuple(sum(
        1 << vertex_position[vertex] for vertex in factor.layout_scope
    ) for factor in factor_tuple)
    cut_masks = tuple(sum(
        1 << cut_position[terminal] for terminal in factor.cut_scope
    ) for factor in factor_tuple)
    incident_factor_masks = [0] * len(cut_terminals)
    for factor_index, factor in enumerate(factor_tuple):
        for terminal in factor.cut_scope:
            incident_factor_masks[cut_position[terminal]] |= 1 << factor_index

    ready_cache: dict[int, int] = {}
    active_cache: dict[int, int] = {}

    def ready(prefix_mask: int) -> int:
        cached = ready_cache.get(prefix_mask)
        if cached is not None:
            return cached
        result = 0
        for factor_index, layout_mask in enumerate(layout_masks):
            if not layout_mask & ~prefix_mask:
                result |= 1 << factor_index
        ready_cache[prefix_mask] = result
        return result

    def active(prefix_mask: int) -> int:
        cached = active_cache.get(prefix_mask)
        if cached is not None:
            return cached
        ready_factors = ready(prefix_mask)
        result = 0
        for position, incident in enumerate(incident_factor_masks):
            if incident & ready_factors and incident & ~ready_factors:
                result |= 1 << position
        active_cache[prefix_mask] = result
        return result

    def factor_cut_union(factor_mask: int) -> int:
        result = 0
        remaining = factor_mask
        while remaining:
            lowest = remaining & -remaining
            result |= cut_masks[lowest.bit_length() - 1]
            remaining ^= lowest
        return result

    full_mask = (1 << len(domain)) - 1
    # peak event width, current post-event width, negative ready count,
    # prefix mask, deterministic order
    beam: list[tuple[int, int, int, int, tuple[int, ...]]] = [
        (0, 0, 0, 0, ())
    ]
    for _depth in range(len(domain)):
        if deadline is not None and perf_counter() >= deadline:
            return ()
        best_by_mask: dict[
            int,
            tuple[int, int, int, int, tuple[int, ...]],
        ] = {}
        for peak, _current, _negative_ready, prefix_mask, prefix in beam:
            old_ready = ready(prefix_mask)
            old_active = active(prefix_mask)
            remaining_vertices = full_mask ^ prefix_mask
            while remaining_vertices:
                lowest = remaining_vertices & -remaining_vertices
                position = lowest.bit_length() - 1
                following_mask = prefix_mask | lowest
                following_ready = ready(following_mask)
                newly_ready = following_ready & ~old_ready
                event_width = (
                    old_active | factor_cut_union(newly_ready)
                ).bit_count()
                following_active = active(following_mask).bit_count()
                candidate = (
                    max(peak, event_width),
                    following_active,
                    -following_ready.bit_count(),
                    following_mask,
                    (*prefix, domain[position]),
                )
                previous = best_by_mask.get(following_mask)
                if previous is None or (
                    candidate[:3], candidate[4]
                ) < (
                    previous[:3], previous[4]
                ):
                    best_by_mask[following_mask] = candidate
                remaining_vertices ^= lowest
        beam = sorted(
            best_by_mask.values(),
            key=lambda item: (item[:3], item[4]),
        )[:beam_width]
    result = []
    for candidate in beam:
        order = candidate[4]
        if order not in result:
            result.append(order)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TerminalCutSignature:
    """A saturated terminal cut function in little-endian subset order."""

    terminals: tuple[Hashable, ...]
    values: tuple[int, ...]
    saturation: int | None = None

    def __post_init__(self) -> None:
        if len(self.terminals) != len(set(self.terminals)):
            raise ValueError("terminal cut signature repeats a terminal")
        if len(self.values) != 1 << len(self.terminals):
            raise ValueError("terminal cut vector has the wrong size")
        if any(value < 0 for value in self.values):
            raise ValueError("terminal cut capacities must be non-negative")
        if self.saturation is not None and self.saturation <= 0:
            raise ValueError("terminal cut saturation must be positive or None")
        if self.saturation is not None and any(
            value > self.saturation for value in self.values
        ):
            raise ValueError("terminal cut value exceeds its saturation")

    @classmethod
    def zero(
        cls,
        terminals: Sequence[Hashable] = (),
        *,
        saturation: int | None = None,
    ) -> "TerminalCutSignature":
        terminal_tuple = tuple(terminals)
        return cls(
            terminal_tuple,
            (0,) * (1 << len(terminal_tuple)),
            saturation,
        )

    def _saturate(self, value: int) -> int:
        return (
            value
            if self.saturation is None
            else min(self.saturation, value)
        )

    def add_terminal(self, terminal: Hashable) -> "TerminalCutSignature":
        if terminal in self.terminals:
            raise ValueError("terminal already belongs to the cut signature")
        size = len(self.values)
        # The new isolated terminal has no cost on either side.
        return TerminalCutSignature(
            (*self.terminals, terminal),
            (*self.values, *self.values),
            self.saturation,
        )

    def add_factor(
        self,
        scope: Sequence[Hashable],
        costs: Sequence[int],
    ) -> "TerminalCutSignature":
        """Add an arbitrary non-negative cut factor over terminal sides."""

        scope_tuple = tuple(scope)
        if len(scope_tuple) != len(set(scope_tuple)):
            raise ValueError("terminal cut factor repeats a terminal")
        positions = {terminal: index for index, terminal in enumerate(self.terminals)}
        if unknown := set(scope_tuple) - positions.keys():
            raise ValueError(f"cut factor uses unknown terminals: {sorted(map(str, unknown))}")
        cost_tuple = tuple(int(value) for value in costs)
        if len(cost_tuple) != 1 << len(scope_tuple):
            raise ValueError("terminal cut factor table has the wrong size")
        if any(value < 0 for value in cost_tuple):
            raise ValueError("terminal cut factor costs must be non-negative")
        result = []
        for mask, value in enumerate(self.values):
            local_mask = sum(
                ((mask >> positions[terminal]) & 1) << local_position
                for local_position, terminal in enumerate(scope_tuple)
            )
            result.append(self._saturate(value + cost_tuple[local_mask]))
        return TerminalCutSignature(
            self.terminals,
            tuple(result),
            self.saturation,
        )

    def add_directed_edge(
        self,
        start: Hashable,
        end: Hashable,
        capacity: int,
    ) -> "TerminalCutSignature":
        """Add ``capacity`` iff ``start`` is source-side and ``end`` sink-side."""

        if capacity < 0:
            raise ValueError("directed cut capacity must be non-negative")
        return self.add_factor(
            (start, end),
            (0, int(capacity), 0, 0),
        )

    def add_from_fixed_source(
        self,
        end: Hashable,
        capacity: int,
    ) -> "TerminalCutSignature":
        if capacity < 0:
            raise ValueError("directed cut capacity must be non-negative")
        # End side zero means the fixed source -> end edge crosses the cut.
        return self.add_factor((end,), (int(capacity), 0))

    def add_to_fixed_sink(
        self,
        start: Hashable,
        capacity: int,
    ) -> "TerminalCutSignature":
        if capacity < 0:
            raise ValueError("directed cut capacity must be non-negative")
        # Start side one means the start -> fixed sink edge crosses the cut.
        return self.add_factor((start,), (0, int(capacity)))

    def add_constant(self, capacity: int) -> "TerminalCutSignature":
        if capacity < 0:
            raise ValueError("directed cut capacity must be non-negative")
        return TerminalCutSignature(
            self.terminals,
            tuple(self._saturate(value + capacity) for value in self.values),
            self.saturation,
        )

    def forget(self, terminal: Hashable) -> "TerminalCutSignature":
        """Make one terminal internal by minimizing over its cut side."""

        try:
            removed = self.terminals.index(terminal)
        except ValueError as error:
            raise ValueError("cannot forget an unknown terminal") from error
        following_terminals = tuple(
            value for value in self.terminals if value != terminal
        )
        following = []
        for following_mask in range(1 << len(following_terminals)):
            lower_bits = following_mask & ((1 << removed) - 1)
            upper_bits = following_mask >> removed
            mask_zero = lower_bits | (upper_bits << (removed + 1))
            mask_one = mask_zero | (1 << removed)
            following.append(min(self.values[mask_zero], self.values[mask_one]))
        return TerminalCutSignature(
            following_terminals,
            tuple(following),
            self.saturation,
        )

    def combine(
        self,
        other: "TerminalCutSignature",
    ) -> "TerminalCutSignature":
        """Glue independent fragments by identifying equal terminal names."""

        if self.saturation != other.saturation:
            raise ValueError("combined terminal cuts use different saturations")
        terminals = (*self.terminals, *(
            terminal for terminal in other.terminals if terminal not in self.terminals
        ))
        positions = {terminal: index for index, terminal in enumerate(terminals)}

        def restrict(mask: int, local: tuple[Hashable, ...]) -> int:
            return sum(
                ((mask >> positions[terminal]) & 1) << local_position
                for local_position, terminal in enumerate(local)
            )

        values = tuple(
            self._saturate(
                self.values[restrict(mask, self.terminals)]
                + other.values[restrict(mask, other.terminals)]
            )
            for mask in range(1 << len(terminals))
        )
        return TerminalCutSignature(terminals, values, self.saturation)

    def condition(
        self,
        sides: Mapping[Hashable, bool],
    ) -> "TerminalCutSignature":
        """Fix terminal sides, then remove those terminals from the interface."""

        if unknown := set(sides) - set(self.terminals):
            raise ValueError(f"cannot condition unknown terminals: {sorted(map(str, unknown))}")
        retained = tuple(terminal for terminal in self.terminals if terminal not in sides)
        retained_positions = {terminal: index for index, terminal in enumerate(retained)}
        original_positions = {
            terminal: index for index, terminal in enumerate(self.terminals)
        }
        values = []
        for retained_mask in range(1 << len(retained)):
            original_mask = 0
            for terminal in self.terminals:
                side = (
                    bool(sides[terminal])
                    if terminal in sides
                    else bool(
                        retained_mask >> retained_positions[terminal] & 1
                    )
                )
                original_mask |= int(side) << original_positions[terminal]
            values.append(self.values[original_mask])
        return TerminalCutSignature(retained, tuple(values), self.saturation)

    @property
    def minimum_cut(self) -> int:
        """Minimize over every still-free terminal side."""

        return min(self.values)


def directed_network_terminal_cut_signature(
    node_count: int,
    edges: Sequence[tuple[int, int, int]],
    *,
    source: int,
    sink: int,
    terminals: Sequence[int] = (),
    saturation: int | None = None,
) -> TerminalCutSignature:
    """Reference compiler for a complete small directed network.

    This helper initially materializes all non-fixed nodes and is intended for
    validation and small separators.  A frontier compiler should instead add
    and forget terminals incrementally so its vector never exceeds the live
    separator width.
    """

    if node_count <= 1:
        raise ValueError("directed cut network requires at least two nodes")
    if not 0 <= source < node_count or not 0 <= sink < node_count or source == sink:
        raise ValueError("directed cut source/sink are invalid")
    terminal_tuple = tuple(terminals)
    if len(terminal_tuple) != len(set(terminal_tuple)):
        raise ValueError("directed cut terminals must be unique")
    if unknown := set(terminal_tuple) - set(range(node_count)):
        raise ValueError(f"directed cut terminals are invalid: {sorted(unknown)}")
    if source in terminal_tuple or sink in terminal_tuple:
        raise ValueError("fixed source/sink cannot also be free terminals")

    variables = tuple(node for node in range(node_count) if node not in {source, sink})
    signature = TerminalCutSignature.zero(variables, saturation=saturation)
    for start, end, raw_capacity in edges:
        capacity = int(raw_capacity)
        if not 0 <= start < node_count or not 0 <= end < node_count:
            raise ValueError("directed cut edge endpoint is invalid")
        if capacity < 0:
            raise ValueError("directed cut capacity must be non-negative")
        if capacity == 0 or start == sink or end == source:
            continue
        if start == source and end == sink:
            signature = signature.add_constant(capacity)
        elif start == source:
            signature = signature.add_from_fixed_source(end, capacity)
        elif end == sink:
            signature = signature.add_to_fixed_sink(start, capacity)
        else:
            signature = signature.add_directed_edge(start, end, capacity)
    for node in variables:
        if node not in terminal_tuple:
            signature = signature.forget(node)
    return signature
