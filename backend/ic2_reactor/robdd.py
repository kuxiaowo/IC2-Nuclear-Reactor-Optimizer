"""Small canonical reduced ordered BDD core for exact symbolic proofs."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence


class ROBDDManager:
    """Canonical ROBDD manager for one fixed variable order.

    Node ids 0 and 1 are the false and true terminals.  Every other node is
    unique by ``(variable level, low, high)`` and suppresses ``low == high``.
    Consequently equality of roots is equality of represented Boolean
    functions for this variable order.
    """

    def __init__(self, variables: Sequence[Hashable]) -> None:
        self.variables = tuple(variables)
        if not self.variables or len(self.variables) != len(set(self.variables)):
            raise ValueError("ROBDD variables must be non-empty and unique")
        self.level_by_variable = {
            variable: level for level, variable in enumerate(self.variables)
        }
        self._nodes: list[tuple[int, int, int] | None] = [None, None]
        self._unique: dict[tuple[int, int, int], int] = {}
        self._variable_roots: dict[Hashable, int] = {}
        self._apply_cache: dict[tuple[str, int, int], int] = {}
        self._negate_cache: dict[int, int] = {0: 1, 1: 0}

    @property
    def allocated_node_count(self) -> int:
        return len(self._nodes) - 2

    def _mk(self, level: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = (level, low, high)
        found = self._unique.get(key)
        if found is not None:
            return found
        node = len(self._nodes)
        self._nodes.append(key)
        self._unique[key] = node
        return node

    def variable(self, variable: Hashable) -> int:
        if variable not in self.level_by_variable:
            raise ValueError("unknown ROBDD variable")
        root = self._variable_roots.get(variable)
        if root is None:
            root = self._mk(self.level_by_variable[variable], 0, 1)
            self._variable_roots[variable] = root
        return root

    def _level(self, node: int) -> int:
        if node in (0, 1):
            return len(self.variables)
        raw = self._nodes[node]
        assert raw is not None
        return raw[0]

    def negate(self, node: int) -> int:
        cached = self._negate_cache.get(node)
        if cached is not None:
            return cached
        raw = self._nodes[node]
        if raw is None:  # pragma: no cover - terminal cache invariant
            raise AssertionError("non-terminal ROBDD node has no record")
        level, low, high = raw
        result = self._mk(level, self.negate(low), self.negate(high))
        self._negate_cache[node] = result
        self._negate_cache[result] = node
        return result

    @staticmethod
    def _terminal_operation(operation: str, left: int, right: int) -> int:
        first, second = bool(left), bool(right)
        if operation == "and":
            return int(first and second)
        if operation == "or":
            return int(first or second)
        if operation == "xor":
            return int(first != second)
        if operation == "equiv":
            return int(first == second)
        raise ValueError(f"unknown ROBDD operation: {operation}")

    def apply(self, operation: str, left: int, right: int) -> int:
        if operation not in {"and", "or", "xor", "equiv"}:
            raise ValueError(f"unknown ROBDD operation: {operation}")
        if operation in {"and", "or", "xor", "equiv"} and left > right:
            left, right = right, left
        key = (operation, left, right)
        cached = self._apply_cache.get(key)
        if cached is not None:
            return cached
        if left in (0, 1) and right in (0, 1):
            result = self._terminal_operation(operation, left, right)
            self._apply_cache[key] = result
            return result
        if operation == "and":
            if left == 0:
                return 0
            if left == 1:
                return right
            if left == right:
                return left
        elif operation == "or":
            if left == 0:
                return right
            if left == 1 or left == right:
                return 1 if left == 1 else left
        elif operation == "xor":
            if left == 0:
                return right
            if left == right:
                return 0
            if left == 1:
                return self.negate(right)
        elif operation == "equiv":
            if left == right:
                return 1
            if left == 0:
                return self.negate(right)
            if left == 1:
                return right

        top = min(self._level(left), self._level(right))

        def cofactors(node: int) -> tuple[int, int]:
            if self._level(node) != top:
                return node, node
            raw = self._nodes[node]
            assert raw is not None
            return raw[1], raw[2]

        left_low, left_high = cofactors(left)
        right_low, right_high = cofactors(right)
        result = self._mk(
            top,
            self.apply(operation, left_low, right_low),
            self.apply(operation, left_high, right_high),
        )
        self._apply_cache[key] = result
        return result

    def conjunction(self, *nodes: int) -> int:
        result = 1
        for node in nodes:
            result = self.apply("and", result, node)
        return result

    def disjunction(self, *nodes: int) -> int:
        result = 0
        for node in nodes:
            result = self.apply("or", result, node)
        return result

    def ite(self, condition: int, when_true: int, when_false: int) -> int:
        return self.disjunction(
            self.apply("and", condition, when_true),
            self.apply("and", self.negate(condition), when_false),
        )

    def cube(self, assignment: Mapping[Hashable, bool]) -> int:
        if unknown := set(assignment) - self.level_by_variable.keys():
            raise ValueError(f"cube uses unknown ROBDD variables: {unknown}")
        result = 1
        for variable in self.variables:
            if variable not in assignment:
                continue
            literal = self.variable(variable)
            if not assignment[variable]:
                literal = self.negate(literal)
            result = self.apply("and", result, literal)
        return result

    def from_assignments(
        self,
        assignments: Sequence[Mapping[Hashable, bool]],
    ) -> int:
        result = 0
        for assignment in assignments:
            result = self.apply("or", result, self.cube(assignment))
        return result

    def from_truth_table(self, values: Sequence[bool]) -> int:
        """Build directly from little-endian full assignments.

        Entry ``i`` assigns variable ``variables[level]`` from bit ``level``
        of ``i``.  Recursive even/odd cofactors construct the reduced diagram
        without first materializing one cube per true row.
        """

        raw = tuple(bool(value) for value in values)
        if len(raw) != 1 << len(self.variables):
            raise ValueError("ROBDD truth table has the wrong number of rows")
        cache: dict[tuple[int, tuple[bool, ...]], int] = {}

        def build(level: int, rows: tuple[bool, ...]) -> int:
            if not any(rows):
                return 0
            if all(rows):
                return 1
            key = (level, rows)
            cached = cache.get(key)
            if cached is not None:
                return cached
            if level >= len(self.variables):  # pragma: no cover - constant cases above
                raise AssertionError("nonconstant truth table exhausted variables")
            result = self._mk(
                level,
                build(level + 1, rows[0::2]),
                build(level + 1, rows[1::2]),
            )
            cache[key] = result
            return result

        return build(0, raw)

    def compose(
        self,
        node: int,
        substitutions: Mapping[Hashable, int],
    ) -> int:
        """Simultaneously replace variables by arbitrary Boolean functions."""

        if unknown := set(substitutions) - self.level_by_variable.keys():
            raise ValueError(f"composition uses unknown ROBDD variables: {unknown}")
        cache: dict[int, int] = {0: 0, 1: 1}

        def visit(current: int) -> int:
            cached = cache.get(current)
            if cached is not None:
                return cached
            raw = self._nodes[current]
            assert raw is not None
            level, low, high = raw
            variable = self.variables[level]
            condition = substitutions.get(variable, self.variable(variable))
            result = self.ite(condition, visit(high), visit(low))
            cache[current] = result
            return result

        return visit(node)

    def restrict(
        self,
        node: int,
        assignment: Mapping[Hashable, bool],
    ) -> int:
        """Substitute constants by direct cofactors, without building ITEs."""

        if unknown := set(assignment) - self.level_by_variable.keys():
            raise ValueError(f"restriction uses unknown ROBDD variables: {unknown}")
        fixed = dict(assignment)
        cache: dict[int, int] = {0: 0, 1: 1}

        def visit(current: int) -> int:
            found = cache.get(current)
            if found is not None:
                return found
            raw = self._nodes[current]
            assert raw is not None
            level, low, high = raw
            variable = self.variables[level]
            if variable in fixed:
                result = visit(high if fixed[variable] else low)
            else:
                result = self._mk(level, visit(low), visit(high))
            cache[current] = result
            return result

        return visit(node)

    def existential(self, node: int, variables: Sequence[Hashable]) -> int:
        quantified = set(variables)
        if unknown := quantified - self.level_by_variable.keys():
            raise ValueError(f"quantification uses unknown ROBDD variables: {unknown}")
        cache: dict[int, int] = {0: 0, 1: 1}

        def visit(current: int) -> int:
            cached = cache.get(current)
            if cached is not None:
                return cached
            raw = self._nodes[current]
            assert raw is not None
            level, low, high = raw
            low_result = visit(low)
            high_result = visit(high)
            if self.variables[level] in quantified:
                result = self.apply("or", low_result, high_result)
            else:
                result = self._mk(level, low_result, high_result)
            cache[current] = result
            return result

        return visit(node)

    def evaluate(self, node: int, assignment: Mapping[Hashable, bool]) -> bool:
        if unknown := set(assignment) - self.level_by_variable.keys():
            raise ValueError(f"evaluation uses unknown ROBDD variables: {unknown}")
        current = node
        while current not in (0, 1):
            raw = self._nodes[current]
            assert raw is not None
            level, low, high = raw
            variable = self.variables[level]
            if variable not in assignment:
                raise ValueError("evaluation assignment omits a tested variable")
            current = high if assignment[variable] else low
        return bool(current)

    def satisfying_assignment(
        self,
        node: int,
        *,
        prefer_false: bool = True,
    ) -> dict[Hashable, bool] | None:
        """Return one total satisfying assignment without enumerating models.

        Variables skipped by reduction are don't-cares and receive the
        preferred value.  Branch preference only selects a deterministic
        witness; it does not affect satisfiability.
        """

        if node == 0:
            return None
        assignment = {variable: not prefer_false for variable in self.variables}
        current = node
        while current not in (0, 1):
            raw = self._nodes[current]
            assert raw is not None
            level, low, high = raw
            variable = self.variables[level]
            if prefer_false and low != 0:
                assignment[variable] = False
                current = low
            elif not prefer_false and high != 0:
                assignment[variable] = True
                current = high
            elif high != 0:
                assignment[variable] = True
                current = high
            else:
                assignment[variable] = False
                current = low
        return assignment

    def reachable_node_count(self, root: int) -> int:
        seen: set[int] = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if node in (0, 1) or node in seen:
                continue
            seen.add(node)
            raw = self._nodes[node]
            assert raw is not None
            pending.extend((raw[1], raw[2]))
        return len(seen)

    def reachable_union_node_count(self, roots: Sequence[int]) -> int:
        """Count nonterminal nodes reachable from any supplied root."""

        seen: set[int] = set()
        pending = list(roots)
        while pending:
            node = pending.pop()
            if node in (0, 1) or node in seen:
                continue
            seen.add(node)
            raw = self._nodes[node]
            assert raw is not None
            pending.extend((raw[1], raw[2]))
        return len(seen)

    def compact_roots(
        self,
        roots: Sequence[int],
    ) -> tuple[ROBDDManager, tuple[int, ...]]:
        """Copy only nodes reachable from ``roots`` into a fresh manager.

        Canonical reduction is reapplied under the identical variable order,
        so every returned root represents exactly the same Boolean function.
        Transient construction nodes and operation caches become collectible.
        """

        following = ROBDDManager(self.variables)
        copied: dict[int, int] = {0: 0, 1: 1}

        def copy(node: int) -> int:
            found = copied.get(node)
            if found is not None:
                return found
            raw = self._nodes[node]
            assert raw is not None
            level, low, high = raw
            result = following._mk(level, copy(low), copy(high))
            copied[node] = result
            return result

        return following, tuple(copy(root) for root in roots)

    def canonicalize_roots(
        self,
        roots: Sequence[int],
    ) -> tuple[
        ROBDDManager,
        tuple[int, ...],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ]:
        """Compact an ordered root forest and return an exact cross-manager key.

        Node ids are meaningful only inside one manager.  Copying the roots in
        their declared order, visiting low before high and allocating parents
        after children gives an order-independent normal form for an ordered
        forest under this manager's fixed variable order.  The returned key
        includes both every normalized root id and every node record, so unlike
        a digest it cannot merge unequal Boolean function vectors by collision.
        """

        following, normalized_roots = self.compact_roots(roots)
        records = tuple(
            raw
            for raw in following._nodes[2:]
            if raw is not None
        )
        return following, normalized_roots, (normalized_roots, records)

    def import_roots(
        self,
        source: ROBDDManager,
        roots: Sequence[int],
    ) -> tuple[int, ...]:
        """Import functions from a manager whose variable order is a subsequence."""

        if not set(source.variables) <= set(self.variables):
            raise ValueError("imported ROBDD uses variables outside the target manager")
        target_levels = tuple(
            self.level_by_variable[variable] for variable in source.variables
        )
        if tuple(sorted(target_levels)) != target_levels:
            raise ValueError("imported ROBDD variable order is incompatible")
        copied: dict[int, int] = {0: 0, 1: 1}

        def copy(node: int) -> int:
            found = copied.get(node)
            if found is not None:
                return found
            raw = source._nodes[node]
            assert raw is not None
            level, low, high = raw
            target_level = self.level_by_variable[source.variables[level]]
            result = self._mk(target_level, copy(low), copy(high))
            copied[node] = result
            return result

        return tuple(copy(root) for root in roots)

    def support(self, root: int) -> frozenset[Hashable]:
        """Return the variables on which the represented function depends."""

        variables: set[Hashable] = set()
        seen: set[int] = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if node in (0, 1) or node in seen:
                continue
            seen.add(node)
            raw = self._nodes[node]
            assert raw is not None
            level, low, high = raw
            variables.add(self.variables[level])
            pending.extend((low, high))
        return frozenset(variables)

    def model_count(
        self,
        root: int,
        variables: Sequence[Hashable] | None = None,
    ) -> int:
        """Count satisfying assignments over a declared variable domain."""

        domain = self.variables if variables is None else tuple(variables)
        if len(domain) != len(set(domain)) or not set(domain) <= set(self.variables):
            raise ValueError("ROBDD model-count variables are duplicate or unknown")
        levels = tuple(self.level_by_variable[variable] for variable in domain)
        if tuple(sorted(levels)) != levels:
            raise ValueError("ROBDD model-count variables must follow manager order")
        if not self.support(root) <= set(domain):
            raise ValueError("ROBDD root depends on a variable outside count domain")
        position_by_variable = {
            variable: position for position, variable in enumerate(domain)
        }
        cache: dict[tuple[int, int], int] = {}

        def count(node: int, position: int) -> int:
            if node == 0:
                return 0
            if node == 1:
                return 1 << (len(domain) - position)
            key = (node, position)
            found = cache.get(key)
            if found is not None:
                return found
            raw = self._nodes[node]
            assert raw is not None
            level, low, high = raw
            variable = self.variables[level]
            variable_position = position_by_variable[variable]
            if variable_position < position:  # pragma: no cover - ordered BDD invariant
                raise AssertionError("ROBDD model-count traversal moved backwards")
            skipped_factor = 1 << (variable_position - position)
            result = skipped_factor * (
                count(low, variable_position + 1)
                + count(high, variable_position + 1)
            )
            cache[key] = result
            return result

        return count(root, 0)
