from __future__ import annotations

from itertools import product

from ic2_reactor.robdd import ROBDDManager
from ic2_reactor.symbolic_event_composition import (
    FrozenParameterEvent,
    compose_frozen_parameter_events,
)


def _identity(manager: ROBDDManager, dynamics: tuple[str, ...]) -> dict[str, int]:
    return {variable: manager.variable(variable) for variable in dynamics}


def _assignment(
    p0: bool,
    p1: bool,
    s0: bool,
    s1: bool,
) -> dict[str, bool]:
    return {"p0": p0, "p1": p1, "s0": s0, "s1": s1}


def test_sequential_event_mux_matches_every_parameter_and_state_assignment() -> None:
    parameters = ("p0", "p1")
    dynamics = ("s0", "s1")
    manager = ROBDDManager((*parameters, *dynamics))
    p0 = manager.variable("p0")
    p1 = manager.variable("p1")
    s0 = manager.variable("s0")
    s1 = manager.variable("s1")
    identity = _identity(manager, dynamics)

    # Event 0 toggles s0 iff p0.  Event 1 observes the already updated s0 and
    # copies it to s1 iff p1.  This distinguishes sequential composition from
    # a simultaneous mux of whole-layout transitions.
    events = (
        FrozenParameterEvent(
            "toggle-first",
            (manager.negate(p0), p0),
            (
                identity,
                {"s0": manager.negate(s0), "s1": s1},
            ),
        ),
        FrozenParameterEvent(
            "copy-updated-first",
            (manager.negate(p1), p1),
            (
                identity,
                {"s0": s0, "s1": s0},
            ),
            (0, s0),
        ),
    )
    result = compose_frozen_parameter_events(
        manager,
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        valid_parameter_root=1,
        events=events,
    )

    assert result.event_count == 2
    assert result.compiled_alternative_count == 4
    assert result.represented_parameter_count == 4
    assert result.explicit_family_count == 4
    for p0_value, p1_value, s0_value, s1_value in product((False, True), repeat=4):
        assignment = _assignment(p0_value, p1_value, s0_value, s1_value)
        after_s0 = s0_value ^ p0_value
        after_s1 = after_s0 if p1_value else s1_value
        failure = p1_value and after_s0
        assert manager.evaluate(
            result.raw_next_functions["s0"], assignment
        ) == after_s0
        assert manager.evaluate(
            result.raw_next_functions["s1"], assignment
        ) == after_s1
        assert manager.evaluate(
            result.transition_failure_root, assignment
        ) == failure
        assert manager.evaluate(
            result.poisoned_next_functions["s0"], assignment
        ) == (after_s0 or failure)
        assert manager.evaluate(
            result.poisoned_next_functions["s1"], assignment
        ) == (after_s1 or failure)


def test_sum_of_event_alternatives_replaces_cartesian_family_compilation() -> None:
    parameters = ("a0", "a1", "b0", "b1", "c0", "c1")
    dynamics = ("s",)
    manager = ROBDDManager((*parameters, *dynamics))
    s = manager.variable("s")

    def code_conditions(first: str, second: str, size: int) -> tuple[int, ...]:
        low = manager.variable(first)
        high = manager.variable(second)
        return tuple(
            manager.conjunction(
                low if code & 1 else manager.negate(low),
                high if code & 2 else manager.negate(high),
            )
            for code in range(size)
        )

    domains = (2, 3, 4)
    conditions = (
        code_conditions("a0", "a1", 2),
        code_conditions("b0", "b1", 3),
        code_conditions("c0", "c1", 4),
    )
    valid = manager.conjunction(*(
        manager.disjunction(*group) for group in conditions
    ))
    events = tuple(
        FrozenParameterEvent(
            f"event-{index}",
            group,
            tuple(
                {"s": s if code % 2 == 0 else manager.negate(s)}
                for code in range(size)
            ),
        )
        for index, (group, size) in enumerate(zip(conditions, domains, strict=True))
    )
    result = compose_frozen_parameter_events(
        manager,
        parameter_variables=parameters,
        dynamic_variables=dynamics,
        valid_parameter_root=valid,
        events=events,
    )
    assert result.compiled_alternative_count == sum(domains) == 9
    assert result.explicit_family_count == 2 * 3 * 4 == 24
    assert result.represented_parameter_count == 24


def test_event_partition_rejects_overlap_instead_of_silently_double_applying() -> None:
    manager = ROBDDManager(("p", "s"))
    p = manager.variable("p")
    identity = {"s": manager.variable("s")}
    event = FrozenParameterEvent("bad-partition", (1, p), (identity, identity))
    try:
        compose_frozen_parameter_events(
            manager,
            parameter_variables=("p",),
            dynamic_variables=("s",),
            valid_parameter_root=1,
            events=(event,),
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:  # pragma: no cover
        raise AssertionError("overlapping event conditions were accepted")
