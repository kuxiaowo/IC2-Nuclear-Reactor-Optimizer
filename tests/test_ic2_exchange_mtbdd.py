from __future__ import annotations

from random import Random

from ic2_reactor.engine import ReactorSimulator
from ic2_reactor.ic2_exchange_mtbdd import compile_ic2_exchange_amount_circuit
from ic2_reactor.robdd import ROBDDManager


def assignment(source: int, target: int, source_width: int, target_width: int):
    return {
        **{("source", bit): bool(source >> bit & 1) for bit in range(source_width)},
        **{("target", bit): bool(target >> bit & 1) for bit in range(target_width)},
    }


def test_interval_exchange_circuit_matches_complete_small_capacity_product() -> None:
    source_capacity = 37
    target_capacity = 53
    source_width = source_capacity.bit_length()
    target_width = target_capacity.bit_length()
    variables = tuple((
        *(("source", bit) for bit in range(source_width)),
        *(("target", bit) for bit in range(target_width)),
    ))
    manager = ROBDDManager(variables)
    source_bits = tuple(manager.variable(("source", bit)) for bit in range(source_width))
    target_bits = tuple(manager.variable(("target", bit)) for bit in range(target_width))
    for limit, rounded_base, low_range in (
        (12, False, None),
        (24, True, 12),
        (36, True, 0),
    ):
        circuit = compile_ic2_exchange_amount_circuit(
            manager,
            source_bits,
            target_bits,
            source_capacity=source_capacity,
            target_capacity=target_capacity,
            limit=limit,
            rounded_base=rounded_base,
            low_range=low_range,
        )
        for source in range(source_capacity + 1):
            for target in range(target_capacity + 1):
                expected = ReactorSimulator._exchange_amount(
                    source * 100.0 / source_capacity,
                    target * 100.0 / target_capacity,
                    target_capacity,
                    limit,
                    rounded_base=rounded_base,
                    low_range=low_range,
                )
                assert circuit.amount(
                    assignment(source, target, source_width, target_width)
                ) == expected
        assert circuit.interval_count < (source_capacity + 1) * (target_capacity + 1)
        official_rows = tuple(
            tuple(
                ReactorSimulator._exchange_amount(
                    source * 100.0 / source_capacity,
                    target * 100.0 / target_capacity,
                    target_capacity,
                    limit,
                    rounded_base=rounded_base,
                    low_range=low_range,
                )
                for target in range(target_capacity + 1)
            )
            for source in range(source_capacity + 1)
        )
        for first in range(source_capacity + 1):
            for second in range(source_capacity + 1):
                assert (
                    circuit.source_row_classes[first]
                    == circuit.source_row_classes[second]
                ) == (official_rows[first] == official_rows[second])


def test_interval_exchange_preserves_known_ieee_truncation_boundary() -> None:
    source_capacity = 2_500
    target_capacity = 1_000
    source_width = source_capacity.bit_length()
    target_width = target_capacity.bit_length()
    variables = tuple((
        *(("source", bit) for bit in range(source_width)),
        *(("target", bit) for bit in range(target_width)),
    ))
    manager = ROBDDManager(variables)
    circuit = compile_ic2_exchange_amount_circuit(
        manager,
        tuple(manager.variable(("source", bit)) for bit in range(source_width)),
        tuple(manager.variable(("target", bit)) for bit in range(target_width)),
        source_capacity=source_capacity,
        target_capacity=target_capacity,
        limit=36,
    )
    observed = circuit.amount(
        assignment(10, 14, source_width, target_width)
    )
    assert observed == -15
    assert observed == ReactorSimulator._exchange_amount(0.4, 1.4, 1_000, 36)


def test_large_catalogue_exchange_domains_match_random_official_points() -> None:
    rng = Random(131)
    for source_capacity, target_capacity, limit, rounded_base, low_range in (
        (2_500, 1_000, 12, False, None),
        (10_000, 100_000, 24, False, None),
        (5_000, 118_000, 72, True, 0),
    ):
        source_width = source_capacity.bit_length()
        target_width = target_capacity.bit_length()
        variables = tuple((
            *(("source", bit) for bit in range(source_width)),
            *(("target", bit) for bit in range(target_width)),
        ))
        manager = ROBDDManager(variables)
        circuit = compile_ic2_exchange_amount_circuit(
            manager,
            tuple(manager.variable(("source", bit)) for bit in range(source_width)),
            tuple(manager.variable(("target", bit)) for bit in range(target_width)),
            source_capacity=source_capacity,
            target_capacity=target_capacity,
            limit=limit,
            rounded_base=rounded_base,
            low_range=low_range,
        )
        for _sample in range(2_000):
            source = rng.randrange(source_capacity + 1)
            target = rng.randrange(target_capacity + 1)
            expected = ReactorSimulator._exchange_amount(
                source * 100.0 / source_capacity,
                target * 100.0 / target_capacity,
                target_capacity,
                limit,
                rounded_base=rounded_base,
                low_range=low_range,
            )
            assert circuit.amount(
                assignment(source, target, source_width, target_width)
            ) == expected
        if rounded_base:
            assert circuit.negative_amount_never_exceeds_target


def test_reactor_hull_exchange_can_make_hull_temporarily_negative() -> None:
    source_capacity = 5_000
    target_capacity = 10_000
    target_heat_maximum = 10_199
    source_width = source_capacity.bit_length()
    target_width = target_heat_maximum.bit_length()
    variables = tuple((
        *(("source", bit) for bit in range(source_width)),
        *(("target", bit) for bit in range(target_width)),
    ))
    manager = ROBDDManager(variables)
    circuit = compile_ic2_exchange_amount_circuit(
        manager,
        tuple(manager.variable(("source", bit)) for bit in range(source_width)),
        tuple(manager.variable(("target", bit)) for bit in range(target_width)),
        source_capacity=source_capacity,
        target_capacity=target_capacity,
        target_heat_maximum=target_heat_maximum,
        limit=72,
        rounded_base=True,
        low_range=0,
    )
    observed = circuit.amount(
        assignment(29, 71, source_width, target_width)
    )
    assert observed == -72
    assert 71 + observed == -1
    assert not circuit.negative_amount_never_exceeds_target


def test_offset_encoded_negative_target_domain_matches_official_rule() -> None:
    source_capacity = 17
    target_capacity = 41
    target_minimum = -5
    target_maximum = 52
    source_width = source_capacity.bit_length()
    target_width = (target_maximum - target_minimum).bit_length()
    variables = tuple((
        *(("source", bit) for bit in range(source_width)),
        *(("target", bit) for bit in range(target_width)),
    ))
    manager = ROBDDManager(variables)
    circuit = compile_ic2_exchange_amount_circuit(
        manager,
        tuple(manager.variable(("source", bit)) for bit in range(source_width)),
        tuple(manager.variable(("target", bit)) for bit in range(target_width)),
        source_capacity=source_capacity,
        target_capacity=target_capacity,
        target_heat_minimum=target_minimum,
        target_heat_maximum=target_maximum,
        limit=12,
        rounded_base=True,
        low_range=0,
    )
    for source in range(source_capacity + 1):
        for target in range(target_minimum, target_maximum + 1):
            expected = ReactorSimulator._exchange_amount(
                source * 100.0 / source_capacity,
                target * 100.0 / target_capacity,
                target_capacity,
                12,
                rounded_base=True,
                low_range=0,
            )
            assert circuit.amount(assignment(
                source,
                target - target_minimum,
                source_width,
                target_width,
            )) == expected
