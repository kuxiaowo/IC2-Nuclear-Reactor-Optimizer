"""Safe quotienting primitives for proof-producing frontier algorithms.

Two ideas must not be confused:

* a graph symmetry is useful for static point/edge relaxations;
* an exact dynamic symmetry must additionally preserve the official update
  order, positional rules and inventory semantics.

Likewise, Pareto dominance is valid only inside one continuation-equivalence
class of the *current* master relaxation.  When a new Benders cut distinguishes
previously merged prefixes, the frontier table is rebuilt from the transition
grammar rather than treating discarded representatives as globally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Hashable, Iterable, Mapping, Sequence, TypeVar

from .mathematical_model import Graph


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class ProductChainAntichainBound:
    """Width upper bound for a product of componentwise ordered integer chains."""

    width: int
    exact: bool
    coordinate_count: int
    rank_sum: int


def product_chain_antichain_bound(
    coordinate_bounds: Sequence[tuple[int, int]],
    *,
    maximum_coefficients: int = 2_000_000,
) -> ProductChainAntichainBound:
    """Bound the largest Pareto antichain without enumerating resource tuples.

    For chain ranks ``r_i = upper_i-lower_i``, the exact width is the largest
    coefficient of ``prod_i (1+x+...+x**r_i)``.  Prefix-sum convolution is
    linear in the accumulated rank.  If that coefficient vector would exceed
    ``maximum_coefficients``, projection onto every coordinate except the
    largest chain gives the rigorous fallback ``prod(lengths)/max(lengths)``.
    """

    if maximum_coefficients <= 0:
        raise ValueError("maximum coefficient count must be positive")
    bounds = tuple((int(lower), int(upper)) for lower, upper in coordinate_bounds)
    if any(lower > upper for lower, upper in bounds):
        raise ValueError("product-chain lower bounds must not exceed upper bounds")
    ranks = tuple(upper - lower for lower, upper in bounds if upper > lower)
    if not ranks:
        return ProductChainAntichainBound(1, True, len(bounds), 0)
    lengths = tuple(rank + 1 for rank in ranks)
    rank_sum = sum(ranks)
    if len(ranks) == 1:
        return ProductChainAntichainBound(1, True, len(bounds), rank_sum)
    if len(ranks) == 2:
        return ProductChainAntichainBound(
            min(lengths),
            True,
            len(bounds),
            rank_sum,
        )
    if rank_sum + 1 > maximum_coefficients:
        return ProductChainAntichainBound(
            prod(lengths) // max(lengths),
            False,
            len(bounds),
            rank_sum,
        )

    coefficients = [1]
    for rank in ranks:
        following = [0] * (len(coefficients) + rank)
        window = 0
        for degree in range(len(following)):
            if degree < len(coefficients):
                window += coefficients[degree]
            expired = degree - rank - 1
            if 0 <= expired < len(coefficients):
                window -= coefficients[expired]
            following[degree] = window
        coefficients = following
    return ProductChainAntichainBound(
        max(coefficients),
        True,
        len(bounds),
        rank_sum,
    )


def append_packed_code(
    packed_key: tuple[int],
    code: int,
    radix: int,
) -> tuple[int]:
    """Append one finite-domain code without copying the whole prefix tuple.

    The DP layer supplies the prefix length, so leading zero codes need no
    separate marker.  A one-item tuple preserves ``ParetoPoint.tie_key``'s
    deterministic lexicographic ordering.
    """

    if len(packed_key) != 1:
        raise ValueError("packed trace key must contain exactly one integer")
    if radix <= 0 or not 0 <= code < radix:
        raise ValueError("trace code is outside its radix")
    return (packed_key[0] * radix + code,)


def unpack_packed_codes(
    packed_key: tuple[int],
    length: int,
    radix: int,
) -> tuple[int, ...]:
    """Recover a fixed-length sequence produced by ``append_packed_code``."""

    if len(packed_key) != 1:
        raise ValueError("packed trace key must contain exactly one integer")
    if length < 0 or radix <= 0:
        raise ValueError("trace length and radix must be valid")
    value = packed_key[0]
    codes = [0] * length
    for index in range(length - 1, -1, -1):
        value, codes[index] = divmod(value, radix)
    if value:
        raise ValueError("packed trace does not fit the requested length")
    return tuple(codes)


def is_graph_automorphism(graph: Graph, permutation: Sequence[int]) -> bool:
    """Return whether ``permutation[v]`` preserves every undirected edge."""

    if len(permutation) != graph.size or set(permutation) != set(graph.vertices):
        return False
    transformed = {
        tuple(sorted((permutation[first], permutation[second])))
        for first, second in graph.edges
    }
    return transformed == set(graph.edges)


def preserves_update_order(graph: Graph, permutation: Sequence[int]) -> bool:
    """Require the exact event at each ordinal to remain the same event."""

    if not is_graph_automorphism(graph, permutation):
        return False
    return tuple(permutation[vertex] for vertex in graph.update_order) == graph.update_order


def rectangular_symmetry_permutations(
    graph: Graph,
    *,
    preserve_event_order: bool,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate the complete dihedral symmetry set of a rectangular grid."""

    if graph.rows is None or graph.columns is None:
        raise ValueError("rectangular symmetry requires grid metadata")
    rows, columns = graph.rows, graph.columns
    if graph.size != rows * columns:
        raise ValueError("inconsistent rectangular graph dimensions")

    def index(row: int, column: int) -> int:
        return row * columns + column

    coordinate_maps = [
        lambda r, c: (r, c),
        lambda r, c: (rows - 1 - r, c),
        lambda r, c: (r, columns - 1 - c),
        lambda r, c: (rows - 1 - r, columns - 1 - c),
    ]
    if rows == columns:
        coordinate_maps.extend((
            lambda r, c: (c, r),
            lambda r, c: (columns - 1 - c, rows - 1 - r),
            lambda r, c: (c, rows - 1 - r),
            lambda r, c: (columns - 1 - c, r),
        ))

    permutations = []
    for transform in coordinate_maps:
        permutation = tuple(
            index(*transform(row, column))
            for row in range(rows)
            for column in range(columns)
        )
        if not is_graph_automorphism(graph, permutation):
            raise AssertionError("internal rectangular transform is not an automorphism")
        if preserve_event_order and not preserves_update_order(graph, permutation):
            continue
        if permutation not in permutations:
            permutations.append(permutation)
    return tuple(permutations)


def transform_labels(
    labels: Sequence[str],
    permutation: Sequence[int],
) -> tuple[str, ...]:
    """Move the label at ``v`` to ``permutation[v]``."""

    if len(labels) != len(permutation):
        raise ValueError("label and permutation lengths differ")
    transformed = [""] * len(labels)
    for source, target in enumerate(permutation):
        transformed[target] = labels[source]
    return tuple(transformed)


def canonical_label_orbit(
    labels: Sequence[str],
    permutations: Iterable[Sequence[int]],
) -> tuple[str, ...]:
    """Return the lexicographically least proven-equivalent representative."""

    images = tuple(transform_labels(labels, permutation) for permutation in permutations)
    if not images:
        raise ValueError("at least one equivalence permutation is required")
    return min(images)


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One additive master-relaxation value in a continuation class.

    Larger power and residual capacities are better; smaller generated heat is
    better.  ``tie_key`` selects a deterministic representative only when all
    mathematical coordinates are equal.
    """

    power: int
    generated_heat: int
    residual_capacities: tuple[int, ...] = ()
    tie_key: tuple = ()

    def __post_init__(self) -> None:
        if self.generated_heat < 0:
            raise ValueError("generated heat must be non-negative")


def dominates(first: ParetoPoint, second: ParetoPoint) -> bool:
    """Return whether ``first`` is never worse for any common suffix."""

    if len(first.residual_capacities) != len(second.residual_capacities):
        raise ValueError("Pareto points use different resource dimensions")
    weak = (
        first.power >= second.power
        and first.generated_heat <= second.generated_heat
        and all(
            left >= right
            for left, right in zip(
                first.residual_capacities,
                second.residual_capacities,
                strict=True,
            )
        )
    )
    strict = (
        first.power > second.power
        or first.generated_heat < second.generated_heat
        or any(
            left > right
            for left, right in zip(
                first.residual_capacities,
                second.residual_capacities,
                strict=True,
            )
        )
    )
    return weak and strict


class ContinuationParetoTable:
    """Antichains indexed by a caller-supplied sufficient continuation key."""

    def __init__(self) -> None:
        self._frontiers: dict[Hashable, list[ParetoPoint]] = {}
        self.insertions = 0
        self.dominated_rejections = 0
        self.removed_points = 0
        self.equal_replacements = 0

    def insert(self, continuation_key: Hashable, point: ParetoPoint) -> bool:
        """Insert a point and maintain the exact nondominated antichain."""

        self.insertions += 1
        frontier = self._frontiers.setdefault(continuation_key, [])
        for index, existing in enumerate(frontier):
            if dominates(existing, point):
                self.dominated_rejections += 1
                return False
            if (
                existing.power == point.power
                and existing.generated_heat == point.generated_heat
                and existing.residual_capacities == point.residual_capacities
            ):
                if point.tie_key < existing.tie_key:
                    frontier[index] = point
                    self.equal_replacements += 1
                    return True
                self.dominated_rejections += 1
                return False

        retained = [existing for existing in frontier if not dominates(point, existing)]
        self.removed_points += len(frontier) - len(retained)
        retained.append(point)
        self._frontiers[continuation_key] = retained
        return True

    @staticmethod
    def _output_order(value: ParetoPoint) -> tuple:
        return (
            -value.power,
            value.generated_heat,
            tuple(-item for item in value.residual_capacities),
            value.tie_key,
        )

    def frontier(self, continuation_key: Hashable) -> tuple[ParetoPoint, ...]:
        return tuple(sorted(
            self._frontiers.get(continuation_key, ()),
            key=self._output_order,
        ))

    def snapshot(self) -> Mapping[Hashable, tuple[ParetoPoint, ...]]:
        return {
            key: tuple(sorted(values, key=self._output_order))
            for key, values in self._frontiers.items()
        }

    def frontier_items(self) -> Iterable[tuple[Hashable, Sequence[ParetoPoint]]]:
        """Iterate a read-only-by-contract view without copying every layer.

        Dynamic programs only mutate their separate ``following`` table while
        consuming this view.  ``snapshot`` remains available to callers that
        need ownership, but hot layer transitions avoid rebuilding tuples.
        """

        return self._frontiers.items()

    @property
    def point_count(self) -> int:
        return sum(len(values) for values in self._frontiers.values())

    @property
    def key_count(self) -> int:
        return len(self._frontiers)

    @property
    def maximum_frontier_width(self) -> int:
        return max((len(values) for values in self._frontiers.values()), default=0)
