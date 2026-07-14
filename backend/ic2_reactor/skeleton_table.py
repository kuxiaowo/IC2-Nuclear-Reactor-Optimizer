from __future__ import annotations

import hashlib
import heapq
import json
import marshal
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .components import COMPONENTS, RULESET_VERSION
from .engine import ReactorSimulator


TABLE_SCHEMA_VERSION = 3
NEGATIVE_INFINITY = -10**12
POWER_EMPTY = "__power_empty__"


def skeleton_table_cache_path() -> Path:
    override = os.environ.get("IC2_SKELETON_TABLE_DB")
    if override:
        return Path(override)
    return Path.cwd() / ".data" / "skeleton_power_tables.sqlite3"


def _vertex_power(item: str) -> int:
    spec = COMPONENTS[item]
    if spec.kind != "fuel":
        return 0
    return int(ReactorSimulator.EU_PER_PULSE) * spec.rod_count * spec.internal_pulses


def _edge_power(first: str, second: str) -> int:
    first_spec = COMPONENTS[first]
    second_spec = COMPONENTS[second]
    pulses = 0
    if first_spec.kind == "fuel" and second_spec.kind in {"fuel", "reflector"}:
        pulses += first_spec.rod_count
    if second_spec.kind == "fuel" and first_spec.kind in {"fuel", "reflector"}:
        pulses += second_spec.rod_count
    return int(ReactorSimulator.EU_PER_PULSE) * pulses


@dataclass(frozen=True, slots=True)
class SkeletonSearchNode:
    bound: int
    power: int
    step: int
    frontier: int
    remaining: tuple[int, ...]
    power_components: int
    choices: tuple[int, ...]


class SkeletonPowerTable:
    """Exact persisted suffix-power table for one constrained skeleton space.

    Cells are visited column first, so the live grid frontier is always the
    reactor's fixed six-row height.  A state consists of the next cell, the
    six frontier labels and the remaining component inventory.  The stored
    value is the exact maximum additional EU/t reachable below that state.
    """

    def __init__(
        self,
        *,
        columns: int,
        power_items: tuple[str, ...],
        power_caps: tuple[int, ...],
        fixed_items: tuple[tuple[int, str], ...],
        total_rods: int | None,
    ) -> None:
        self.columns = columns
        self.slots = columns * 6
        self.power_items = power_items
        self.power_caps = power_caps
        self.total_rods = total_rods
        self.labels = ("empty", *power_items)
        self.label_codes = {item: code for code, item in enumerate(self.labels)}
        self.base = len(self.labels)
        self.place_values = tuple(self.base**row for row in range(6))
        self.rod_counts = tuple(COMPONENTS[item].rod_count for item in power_items)
        self.fuel_indexes = tuple(
            index for index, item in enumerate(power_items)
            if COMPONENTS[item].kind == "fuel"
        )
        fixed = dict(fixed_items)
        fixed_codes: list[int | None] = []
        normalized_fixed: list[tuple[int, str]] = []
        for position in range(self.slots):
            if position not in fixed:
                fixed_codes.append(None)
                continue
            item = fixed[position]
            skeleton_item = (
                "empty"
                if item == POWER_EMPTY
                else item if item in self.label_codes else "empty"
            )
            fixed_codes.append(self.label_codes[skeleton_item])
            normalized_fixed.append((
                position,
                POWER_EMPTY if item == POWER_EMPTY else skeleton_item,
            ))
        self.fixed_codes = tuple(fixed_codes)
        self.fixed_power_count = sum(
            code not in {None, 0} for code in self.fixed_codes
        )
        # A power-layer empty prefix only says that no fuel/reflector may
        # occupy the cell.  The cooling layer is still free to use it, so it
        # must not be subtracted from the cooling completion count.
        self.fixed_cell_count = sum(
            item != POWER_EMPTY for item in fixed.values()
        )
        if total_rods is None:
            self.initial_remaining = power_caps
            self.item_resources = tuple(
                (index, 1) for index in range(len(power_items))
            )
        else:
            reflector_indexes = tuple(
                index for index, item in enumerate(power_items)
                if COMPONENTS[item].kind == "reflector"
            )
            reflector_resource = {
                item_index: resource_index + 1
                for resource_index, item_index in enumerate(reflector_indexes)
            }
            self.initial_remaining = (
                total_rods,
                *(power_caps[index] for index in reflector_indexes),
            )
            self.item_resources = tuple(
                (0, self.rod_counts[index])
                if COMPONENTS[item].kind == "fuel"
                else (reflector_resource[index], 1)
                for index, item in enumerate(power_items)
            )
        self.memo: dict[tuple[int, ...], int] = {}
        self.loaded_from_disk = False
        self.persisted = False
        self._cancel_check: Callable[[], bool] | None = None
        self._visits = 0
        signature = {
            "schema": TABLE_SCHEMA_VERSION,
            "ruleset": RULESET_VERSION,
            "python_marshal": sys.version_info[:2],
            "columns": columns,
            "power_items": power_items,
            "power_caps": power_caps,
            "fixed": normalized_fixed,
            "total_rods": total_rods,
        }
        encoded = json.dumps(signature, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        self.cache_key = hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def _state_key(
        self,
        step: int,
        frontier: int,
        remaining: tuple[int, ...],
    ) -> tuple[int, ...]:
        return (step, frontier, *remaining)

    def _position(self, step: int) -> tuple[int, int, int]:
        column, row = divmod(step, 6)
        return row * self.columns + column, row, column

    def _frontier_label(self, frontier: int, row: int) -> int:
        return frontier // self.place_values[row] % self.base

    def _replace_frontier_label(self, frontier: int, row: int, code: int) -> int:
        old = self._frontier_label(frontier, row)
        return frontier + (code - old) * self.place_values[row]

    def used_rods(self, remaining: tuple[int, ...]) -> int:
        if self.total_rods is not None:
            return self.total_rods - remaining[0]
        return sum(
            (cap - left) * rods
            for cap, left, rods in zip(
                self.power_caps, remaining, self.rod_counts, strict=True
            )
        )

    def has_fuel(self, remaining: tuple[int, ...]) -> bool:
        if self.total_rods is not None:
            return remaining[0] < self.total_rods
        return any(
            self.power_caps[index] > remaining[index]
            for index in self.fuel_indexes
        )

    def consume(
        self,
        code: int,
        remaining: tuple[int, ...],
    ) -> tuple[int, ...]:
        if code == 0:
            return remaining
        resource_index, cost = self.item_resources[code - 1]
        values = list(remaining)
        values[resource_index] -= cost
        return tuple(values)

    def allowed_codes(
        self,
        step: int,
        remaining: tuple[int, ...],
    ) -> Iterator[int]:
        position, _row, _column = self._position(step)
        forced = self.fixed_codes[position]
        codes = (forced,) if forced is not None else range(self.base)
        for code in codes:
            if code == 0:
                yield code
                continue
            item_index = code - 1
            resource_index, cost = self.item_resources[item_index]
            if remaining[resource_index] < cost:
                continue
            yield code

    def transition(
        self,
        step: int,
        frontier: int,
        remaining: tuple[int, ...],
        code: int,
    ) -> tuple[int, tuple[int, ...], int]:
        _position, row, column = self._position(step)
        item = self.labels[code]
        increment = _vertex_power(item)
        if column > 0:
            increment += _edge_power(self.labels[self._frontier_label(frontier, row)], item)
        if row > 0:
            increment += _edge_power(self.labels[self._frontier_label(frontier, row - 1)], item)
        next_frontier = self._replace_frontier_label(frontier, row, code)
        return next_frontier, self.consume(code, remaining), increment

    def best_suffix(
        self,
        step: int,
        frontier: int,
        remaining: tuple[int, ...],
    ) -> int:
        key = self._state_key(step, frontier, remaining)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        self._visits += 1
        if self._visits % 4096 == 0 and self._cancel_check is not None and self._cancel_check():
            raise InterruptedError("skeleton power table construction cancelled")
        if step == self.slots:
            result = 0 if self.has_fuel(remaining) else NEGATIVE_INFINITY
            self.memo[key] = result
            return result
        best = NEGATIVE_INFINITY
        for code in self.allowed_codes(step, remaining):
            next_frontier, next_remaining, increment = self.transition(
                step, frontier, remaining, code
            )
            suffix = self.best_suffix(step + 1, next_frontier, next_remaining)
            if suffix != NEGATIVE_INFINITY:
                best = max(best, increment + suffix)
        self.memo[key] = best
        return best

    def build(self, cancel_check: Callable[[], bool] | None = None) -> int:
        if not self.memo:
            self._load()
        self._cancel_check = cancel_check
        try:
            result = self.best_suffix(0, 0, self.initial_remaining)
        finally:
            self._cancel_check = None
        if not self.loaded_from_disk:
            self._store()
        return result

    def root(self) -> SkeletonSearchNode | None:
        bound = self.best_suffix(0, 0, self.initial_remaining)
        if bound == NEGATIVE_INFINITY:
            return None
        return SkeletonSearchNode(bound, 0, 0, 0, self.initial_remaining, 0, ())

    def expand(self, node: SkeletonSearchNode) -> Iterator[SkeletonSearchNode]:
        for code in self.allowed_codes(node.step, node.remaining):
            next_frontier, next_remaining, increment = self.transition(
                node.step, node.frontier, node.remaining, code
            )
            suffix = self.best_suffix(node.step + 1, next_frontier, next_remaining)
            if suffix == NEGATIVE_INFINITY:
                continue
            power = node.power + increment
            yield SkeletonSearchNode(
                power + suffix,
                power,
                node.step + 1,
                next_frontier,
                next_remaining,
                node.power_components + int(code != 0),
                (*node.choices, code),
            )

    def materialize(self, choices: tuple[int, ...]) -> tuple[str, ...]:
        if len(choices) != self.slots:
            raise ValueError("only a complete skeleton can be materialized")
        layout = ["empty"] * self.slots
        for step, code in enumerate(choices):
            position, _row, _column = self._position(step)
            layout[position] = self.labels[code]
        return tuple(layout)

    def ranked_skeletons(
        self,
        limit: int | None = None,
    ) -> Iterator[tuple[int, tuple[str, ...]]]:
        """Yield complete skeletons in exact non-increasing power order."""
        root = self.root()
        if root is None:
            return
        heap: list[tuple[int, int, SkeletonSearchNode]] = [(-root.bound, 0, root)]
        serial = 0
        yielded = 0
        while heap and (limit is None or yielded < limit):
            _negative_bound, _serial, node = heapq.heappop(heap)
            if node.step == self.slots:
                yielded += 1
                yield node.power, self.materialize(node.choices)
                continue
            for child in self.expand(node):
                serial += 1
                heapq.heappush(heap, (-child.bound, serial, child))

    def _connect(self) -> sqlite3.Connection:
        path = skeleton_table_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS skeleton_power_tables (
                cache_key TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_at REAL NOT NULL,
                state_count INTEGER NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )
        return connection

    def _load(self) -> None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload FROM skeleton_power_tables "
                    "WHERE cache_key = ? AND schema_version = ?",
                    (self.cache_key, TABLE_SCHEMA_VERSION),
                ).fetchone()
            if row is None:
                return
            loaded = marshal.loads(zlib.decompress(row[0]))
            if not isinstance(loaded, dict):
                return
            self.memo = loaded
            self.loaded_from_disk = True
        except (OSError, sqlite3.Error, EOFError, ValueError, TypeError, zlib.error):
            self.memo = {}

    def _store(self) -> None:
        try:
            payload = zlib.compress(marshal.dumps(self.memo), level=3)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO skeleton_power_tables
                        (cache_key, schema_version, created_at, state_count, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.cache_key,
                        TABLE_SCHEMA_VERSION,
                        time.time(),
                        len(self.memo),
                        payload,
                    ),
                )
            self.persisted = True
        except (OSError, sqlite3.Error, ValueError, TypeError):
            self.persisted = False
