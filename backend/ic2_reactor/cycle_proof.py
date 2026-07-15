"""Generic reachable-cycle proof for deterministic finite-state systems."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from threading import Lock
from time import perf_counter
from typing import Generic, Hashable, Mapping, Protocol, Sequence, TypeVar


LayoutT = TypeVar("LayoutT")
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class TransitionObservation(Generic[StateT]):
    state: StateT
    failed: bool = False
    failure_reason: str | None = None
    metrics: Mapping[str, int | float | str | None] | None = None


class DeterministicTransitionSystem(Protocol[LayoutT, StateT]):
    """Minimal contract needed for a mathematical cycle certificate."""

    def initial_state(self, layout: LayoutT) -> StateT: ...

    def step(self, layout: LayoutT, state: StateT) -> TransitionObservation[StateT]: ...

    def state_key(self, state: StateT) -> Hashable: ...


@dataclass(frozen=True, slots=True)
class ReachableCycleProof:
    outcome: str
    safe: bool
    conclusive: bool
    transient_length: int | None
    period_length: int | None
    checked_steps: int
    failure_step: int | None
    failure_reason: str | None
    last_metrics: tuple[tuple[str, int | float | str | None], ...]
    elapsed_seconds: float
    state_space_upper_bound: int | None = None
    complete_state_bound_used: bool = False


class DeterministicCycleVerifier:
    """Prove failure or a reachable repeated state; horizons stay inconclusive."""

    def verify(
        self,
        system: DeterministicTransitionSystem[LayoutT, StateT],
        layout: LayoutT,
        *,
        max_steps: int,
        time_limit_seconds: float | None = None,
    ) -> ReachableCycleProof:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = None if time_limit_seconds is None else started + time_limit_seconds
        bound_method = getattr(system, "safe_state_upper_bound", None)
        state_bound = bound_method(layout) if callable(bound_method) else None
        if state_bound is not None and state_bound <= 0:
            raise ValueError("safe-state upper bound must be positive")
        complete_bound_used = state_bound is not None and max_steps >= state_bound
        state = system.initial_state(layout)
        seen: dict[Hashable, int] = {system.state_key(state): 0}
        last_metrics: tuple[tuple[str, int | float | str | None], ...] = ()
        for step in range(1, max_steps + 1):
            if deadline is not None and perf_counter() >= deadline:
                return ReachableCycleProof(
                    outcome="time_limit",
                    safe=False,
                    conclusive=False,
                    transient_length=None,
                    period_length=None,
                    checked_steps=step - 1,
                    failure_step=None,
                    failure_reason=None,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            observation = system.step(layout, state)
            state = observation.state
            if observation.metrics is not None:
                last_metrics = tuple(sorted(observation.metrics.items()))
            if observation.failed:
                return ReachableCycleProof(
                    outcome="failed",
                    safe=False,
                    conclusive=True,
                    transient_length=None,
                    period_length=None,
                    checked_steps=step,
                    failure_step=step,
                    failure_reason=observation.failure_reason,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            key = system.state_key(state)
            previous = seen.get(key)
            if previous is not None:
                return ReachableCycleProof(
                    outcome="safe_cycle",
                    safe=True,
                    conclusive=True,
                    transient_length=previous,
                    period_length=step - previous,
                    checked_steps=step,
                    failure_step=None,
                    failure_reason=None,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            seen[key] = step
        if complete_bound_used:
            raise AssertionError(
                "deterministic trajectory exceeded the declared finite safe-state "
                "bound without failure or repetition"
            )
        return ReachableCycleProof(
            outcome="horizon",
            safe=False,
            conclusive=False,
            transient_length=None,
            period_length=None,
            checked_steps=max_steps,
            failure_step=None,
            failure_reason=None,
            last_metrics=last_metrics,
            elapsed_seconds=perf_counter() - started,
            state_space_upper_bound=state_bound,
            complete_state_bound_used=complete_bound_used,
        )

    def verify_complete(
        self,
        system: DeterministicTransitionSystem[LayoutT, StateT],
        layout: LayoutT,
        *,
        time_limit_seconds: float | None = None,
    ) -> ReachableCycleProof:
        """Use the system's finite safe-state bound as a complete horizon.

        A time limit may still return an inconclusive result.  Without a time
        limit, a correct deterministic bounded system must return failure or a
        repeated safe state before exhausting this horizon.
        """

        bound_method = getattr(system, "safe_state_upper_bound", None)
        if not callable(bound_method):
            raise ValueError("transition system does not declare a safe-state bound")
        bound = int(bound_method(layout))
        if bound <= 0:
            raise ValueError("safe-state upper bound must be positive")
        return self.verify(
            system,
            layout,
            max_steps=bound,
            time_limit_seconds=time_limit_seconds,
        )


class ConstantMemoryCycleVerifier:
    """Brent-style reachable-repeat proof using one retained state key.

    The returned transient is the earlier index of an actually observed equal
    key, not necessarily the *minimum* transient.  That is sufficient: the
    two equal reachable states and their safe intervening transitions are a
    complete infinite-safety certificate.  Compared with the hash-table
    verifier this may execute a constant-factor more transitions but keeps
    only the current system state and one old key.
    """

    def verify(
        self,
        system: DeterministicTransitionSystem[LayoutT, StateT],
        layout: LayoutT,
        *,
        max_steps: int,
        time_limit_seconds: float | None = None,
    ) -> ReachableCycleProof:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        started = perf_counter()
        deadline = None if time_limit_seconds is None else started + time_limit_seconds
        bound_method = getattr(system, "safe_state_upper_bound", None)
        state_bound = int(bound_method(layout)) if callable(bound_method) else None
        if state_bound is not None and state_bound <= 0:
            raise ValueError("safe-state upper bound must be positive")
        complete_bound_used = (
            state_bound is not None and max_steps >= 3 * state_bound
        )

        state = system.initial_state(layout)
        retained_key = system.state_key(state)
        retained_step = 0
        power = 1
        distance = 0
        last_metrics: tuple[tuple[str, int | float | str | None], ...] = ()
        for step in range(1, max_steps + 1):
            if deadline is not None and perf_counter() >= deadline:
                return ReachableCycleProof(
                    outcome="time_limit",
                    safe=False,
                    conclusive=False,
                    transient_length=None,
                    period_length=None,
                    checked_steps=step - 1,
                    failure_step=None,
                    failure_reason=None,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            observation = system.step(layout, state)
            state = observation.state
            distance += 1
            if observation.metrics is not None:
                last_metrics = tuple(sorted(observation.metrics.items()))
            if observation.failed:
                return ReachableCycleProof(
                    outcome="failed",
                    safe=False,
                    conclusive=True,
                    transient_length=None,
                    period_length=None,
                    checked_steps=step,
                    failure_step=step,
                    failure_reason=observation.failure_reason,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            key = system.state_key(state)
            if key == retained_key:
                return ReachableCycleProof(
                    outcome="safe_cycle",
                    safe=True,
                    conclusive=True,
                    transient_length=retained_step,
                    period_length=step - retained_step,
                    checked_steps=step,
                    failure_step=None,
                    failure_reason=None,
                    last_metrics=last_metrics,
                    elapsed_seconds=perf_counter() - started,
                    state_space_upper_bound=state_bound,
                    complete_state_bound_used=complete_bound_used,
                )
            if distance == power:
                retained_key = key
                retained_step = step
                power *= 2
                distance = 0
        if complete_bound_used:
            raise AssertionError(
                "constant-memory cycle detection exceeded three times the "
                "finite safe-state bound without failure or repetition"
            )
        return ReachableCycleProof(
            outcome="horizon",
            safe=False,
            conclusive=False,
            transient_length=None,
            period_length=None,
            checked_steps=max_steps,
            failure_step=None,
            failure_reason=None,
            last_metrics=last_metrics,
            elapsed_seconds=perf_counter() - started,
            state_space_upper_bound=state_bound,
            complete_state_bound_used=complete_bound_used,
        )

    def verify_complete(
        self,
        system: DeterministicTransitionSystem[LayoutT, StateT],
        layout: LayoutT,
        *,
        time_limit_seconds: float | None = None,
    ) -> ReachableCycleProof:
        bound_method = getattr(system, "safe_state_upper_bound", None)
        if not callable(bound_method):
            raise ValueError("transition system does not declare a safe-state bound")
        bound = int(bound_method(layout))
        if bound <= 0:
            raise ValueError("safe-state upper bound must be positive")
        return self.verify(
            system,
            layout,
            max_steps=3 * bound,
            time_limit_seconds=time_limit_seconds,
        )


class DeterministicCycleSession(Generic[LayoutT, StateT]):
    """Resume an exact trajectory without replaying an UNKNOWN prefix."""

    def __init__(
        self,
        system: DeterministicTransitionSystem[LayoutT, StateT],
        layout: LayoutT,
    ) -> None:
        self.system = system
        self.layout = layout
        bound_method = getattr(system, "safe_state_upper_bound", None)
        self.state_space_upper_bound = (
            int(bound_method(layout)) if callable(bound_method) else None
        )
        if (
            self.state_space_upper_bound is not None
            and self.state_space_upper_bound <= 0
        ):
            raise ValueError("safe-state upper bound must be positive")
        self.state = system.initial_state(layout)
        self.seen: dict[Hashable, int] = {system.state_key(self.state): 0}
        self.checked_steps = 0
        self.last_metrics: tuple[tuple[str, int | float | str | None], ...] = ()
        self.elapsed_seconds = 0.0
        self._terminal: ReachableCycleProof | None = None
        self._lock = Lock()

    @property
    def conclusive(self) -> bool:
        return self._terminal is not None

    @property
    def progress_steps(self) -> int:
        """Number of exact transitions retained by this resumable proof."""

        with self._lock:
            return self.checked_steps

    def _partial(self, outcome: str, elapsed: float) -> ReachableCycleProof:
        self.elapsed_seconds += elapsed
        return ReachableCycleProof(
            outcome=outcome,
            safe=False,
            conclusive=False,
            transient_length=None,
            period_length=None,
            checked_steps=self.checked_steps,
            failure_step=None,
            failure_reason=None,
            last_metrics=self.last_metrics,
            elapsed_seconds=self.elapsed_seconds,
            state_space_upper_bound=self.state_space_upper_bound,
            complete_state_bound_used=False,
        )

    def advance(
        self,
        additional_steps: int,
        *,
        time_limit_seconds: float | None = None,
    ) -> ReachableCycleProof:
        """Advance by at most ``additional_steps`` and retain all prefix work."""

        if additional_steps <= 0:
            raise ValueError("additional_steps must be positive")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")
        with self._lock:
            if self._terminal is not None:
                return self._terminal
            started = perf_counter()
            deadline = (
                None if time_limit_seconds is None else started + time_limit_seconds
            )
            target = self.checked_steps + additional_steps
            while self.checked_steps < target:
                if deadline is not None and perf_counter() >= deadline:
                    return self._partial("time_limit", perf_counter() - started)
                observation = self.system.step(self.layout, self.state)
                self.state = observation.state
                self.checked_steps += 1
                if observation.metrics is not None:
                    self.last_metrics = tuple(sorted(observation.metrics.items()))
                if observation.failed:
                    self.elapsed_seconds += perf_counter() - started
                    self._terminal = ReachableCycleProof(
                        outcome="failed",
                        safe=False,
                        conclusive=True,
                        transient_length=None,
                        period_length=None,
                        checked_steps=self.checked_steps,
                        failure_step=self.checked_steps,
                        failure_reason=observation.failure_reason,
                        last_metrics=self.last_metrics,
                        elapsed_seconds=self.elapsed_seconds,
                        state_space_upper_bound=self.state_space_upper_bound,
                        complete_state_bound_used=(
                            self.state_space_upper_bound is not None
                            and self.checked_steps >= self.state_space_upper_bound
                        ),
                    )
                    return self._terminal
                key = self.system.state_key(self.state)
                previous = self.seen.get(key)
                if previous is not None:
                    self.elapsed_seconds += perf_counter() - started
                    self._terminal = ReachableCycleProof(
                        outcome="safe_cycle",
                        safe=True,
                        conclusive=True,
                        transient_length=previous,
                        period_length=self.checked_steps - previous,
                        checked_steps=self.checked_steps,
                        failure_step=None,
                        failure_reason=None,
                        last_metrics=self.last_metrics,
                        elapsed_seconds=self.elapsed_seconds,
                        state_space_upper_bound=self.state_space_upper_bound,
                        complete_state_bound_used=(
                            self.state_space_upper_bound is not None
                            and self.checked_steps >= self.state_space_upper_bound
                        ),
                    )
                    return self._terminal
                self.seen[key] = self.checked_steps
                if (
                    self.state_space_upper_bound is not None
                    and self.checked_steps >= self.state_space_upper_bound
                ):
                    raise AssertionError(
                        "trajectory exceeded its finite safe-state bound without "
                        "failure or repetition"
                    )
            return self._partial("horizon", perf_counter() - started)


@dataclass(frozen=True, slots=True)
class IC2FiniteStateSpaceBound:
    """Explicit product bound for a fixed auto-refuel IC2 layout."""

    safe_states: int
    hull_states: int
    slot_state_factors: tuple[int, ...]
    decimal_digits: int


def ic2_safe_state_space_bound(
    layout: Sequence[str],
) -> IC2FiniteStateSpaceBound:
    """Count an optimistic superset of safe thermal/durability states.

    Fuel damage is excluded because auto-refuelling replaces a depleted stack
    with the identical fresh fuel and its damage never affects heat transfer or
    power.  Every heat-storing non-fuel component contributes ``max_heat+1``
    integer heat values.  A finite reflector contributes ``max_damage`` safe
    damage values.  Component ids remain fixed on every safe trajectory;
    removal is already a conclusive failure.
    """

    from .components import COMPONENTS

    if not layout or len(layout) % 6:
        raise ValueError("IC2 layout must contain six complete rows")
    if unknown := set(layout) - COMPONENTS.keys():
        raise ValueError(f"unknown IC2 components: {sorted(unknown)}")
    maximum_hull_heat = 10_000 + sum(
        COMPONENTS[label].hull_capacity_bonus for label in layout
    )
    # The exact simulator reports critical failure at this threshold, so the
    # safe boundary states are 0 through threshold-1.
    hull_states = floor(maximum_hull_heat * 0.85)
    factors = []
    for label in layout:
        spec = COMPONENTS[label]
        heat_states = spec.max_heat + 1 if spec.accepts_heat else 1
        damage_states = (
            spec.max_damage
            if spec.kind == "reflector" and spec.max_damage > 0
            else 1
        )
        factors.append(heat_states * damage_states)
    safe_states = hull_states
    for factor in factors:
        safe_states *= factor
    return IC2FiniteStateSpaceBound(
        safe_states=safe_states,
        hull_states=hull_states,
        slot_state_factors=tuple(factors),
        decimal_digits=len(str(safe_states)),
    )


@dataclass(slots=True)
class IC2TransitionState:
    simulator: object
    peak_component_heat: int = 0
    failure_component: int | None = None


class IC2TransitionSystem:
    """Adapter from the locked six-row IC2 rules to the generic protocol."""

    def __init__(self, columns: int) -> None:
        if columns <= 0:
            raise ValueError("columns must be positive")
        self.columns = columns

    def initial_state(self, layout: tuple[str, ...]) -> IC2TransitionState:
        if len(layout) != 6 * self.columns:
            raise ValueError("the locked IC2 chamber has six rows")
        from .engine import ReactorSimulator
        from .models import Layout

        return IC2TransitionState(
            simulator=ReactorSimulator(Layout(columns=self.columns, slots=list(layout)))
        )

    def safe_state_upper_bound(self, layout: tuple[str, ...]) -> int:
        if len(layout) != 6 * self.columns:
            raise ValueError("the locked IC2 chamber has six rows")
        return ic2_safe_state_space_bound(layout).safe_states

    def step(
        self,
        layout: tuple[str, ...],
        state: IC2TransitionState,
    ) -> TransitionObservation[IC2TransitionState]:
        simulator = state.simulator
        power, generated_heat, _vented = simulator.step(auto_refuel=True)
        state.peak_component_heat = max(
            state.peak_component_heat,
            max((slot.heat for slot in simulator.slots), default=0),
        )
        if simulator.meltdown_tick is not None:
            reason = "meltdown"
        elif simulator.first_component_break_tick is not None:
            reason = "component_broken"
        elif simulator.first_critical_tick is not None:
            reason = "critical"
        else:
            reason = None
        if reason == "component_broken":
            state.failure_component = next(
                (event.slot for event in reversed(simulator.events) if event.slot is not None),
                None,
            )
        return TransitionObservation(
            state=state,
            failed=reason is not None,
            failure_reason=reason,
            metrics={
                "power": int(power),
                "generated_heat": generated_heat,
                "peak_hull_heat": simulator.peak_hull_heat,
                "peak_component_heat": state.peak_component_heat,
                "failure_component": state.failure_component,
            },
        )

    def state_key(self, state: IC2TransitionState) -> Hashable:
        # The layout is fixed for the lifetime of one verifier.  Component ids,
        # zero-capacity heat fields and zero-durability damage fields therefore
        # cannot distinguish future trajectories and must not be repeated in
        # every hash key.  Fuel damage is intentionally absent: auto-refuel
        # replaces it with the identical fuel before it can alter thermal rules.
        return state.simulator.thermal_state_signature()
