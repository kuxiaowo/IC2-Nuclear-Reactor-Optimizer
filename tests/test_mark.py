import pytest

from ic2_reactor.mark import classify_mark, mark_family


@pytest.mark.parametrize(
    ("critical", "broken", "stable", "expected"),
    [
        (None, None, True, "Mark I-I"),
        (20_000, None, False, "Mark II-1"),
        (320_000, None, False, "Mark II-E"),
        (2_000, None, False, "Mark III"),
        (None, 2_000, False, "Mark IV"),
        (1_999, None, False, "Mark V"),
        (None, 1_999, False, "Mark V"),
    ],
)
def test_mark_boundaries(critical, broken, stable, expected):
    assert classify_mark(critical, broken, stable) == expected
    assert mark_family(expected) == expected.split()[1].split("-")[0]


def test_suc_suffix():
    assert classify_mark(20_000, None, False, True) == "Mark II-1-SUC"

