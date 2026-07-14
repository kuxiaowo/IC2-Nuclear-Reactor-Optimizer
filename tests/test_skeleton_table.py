from ic2_reactor.optimizer import skeleton_eu_per_tick
from ic2_reactor.skeleton_table import POWER_EMPTY, SkeletonPowerTable
from itertools import product


def make_table(columns: int = 3, single_cap: int = 4) -> SkeletonPowerTable:
    return SkeletonPowerTable(
        columns=columns,
        power_items=("uranium_single",),
        power_caps=(single_cap,),
        fixed_items=(),
        total_rods=None,
    )


def test_power_table_yields_maximum_and_second_distinct_power_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))
    table = make_table()

    assert table.build() == 60
    distinct: list[int] = []
    for power, skeleton in table.ranked_skeletons():
        assert skeleton_eu_per_tick(skeleton, 3) == power
        if not distinct or distinct[-1] != power:
            distinct.append(power)
        if len(distinct) == 2:
            break

    assert distinct == [60, 50]


def test_power_table_is_reloaded_from_the_persistent_database(tmp_path, monkeypatch):
    database = tmp_path / "skeletons.sqlite3"
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(database))
    first = make_table(single_cap=3)

    first_power = first.build()

    assert first_power == 35
    assert first.persisted
    assert not first.loaded_from_disk
    assert database.exists()
    second = make_table(single_cap=3)
    assert second.build() == first_power
    assert second.loaded_from_disk
    assert second.memo == first.memo


def test_fixed_full_layout_cells_are_part_of_the_table_key_and_search_space(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))
    free = make_table(single_cap=1)
    blocked = SkeletonPowerTable(
        columns=3,
        power_items=("uranium_single",),
        power_caps=(1,),
        fixed_items=((0, "heat_vent"),),
        total_rods=None,
    )

    assert free.cache_key != blocked.cache_key
    assert free.build() == blocked.build() == 5
    assert all(skeleton[0] == "empty" for _, skeleton in blocked.ranked_skeletons())


def test_power_empty_prefix_keeps_the_cell_available_to_the_cooling_layer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))
    power_empty = SkeletonPowerTable(
        columns=3,
        power_items=("uranium_single",),
        power_caps=(1,),
        fixed_items=((0, POWER_EMPTY),),
        total_rods=None,
    )
    full_empty = SkeletonPowerTable(
        columns=3,
        power_items=("uranium_single",),
        power_caps=(1,),
        fixed_items=((0, "empty"),),
        total_rods=None,
    )

    assert power_empty.cache_key != full_empty.cache_key
    assert power_empty.fixed_cell_count == 0
    assert full_empty.fixed_cell_count == 1
    assert power_empty.build() == full_empty.build() == 5
    assert all(skeleton[0] == "empty" for _, skeleton in power_empty.ranked_skeletons())


def test_mixed_fuel_and_reflector_table_matches_brute_force(tmp_path, monkeypatch):
    monkeypatch.setenv("IC2_SKELETON_TABLE_DB", str(tmp_path / "skeletons.sqlite3"))
    items = ("uranium_single", "uranium_dual", "iridium_reflector")
    caps = (2, 1, 1)
    blocked = tuple((position, "empty") for position in range(6, 18))
    table = SkeletonPowerTable(
        columns=3,
        power_items=items,
        power_caps=caps,
        fixed_items=blocked,
        total_rods=4,
    )
    labels = ("empty", *items)
    # Total-rods mode shares one rod resource across all fuel package types;
    # only the independent reflector cap needs another resource dimension.
    assert table.initial_remaining == (4, 1)
    brute_maximum = 0
    for prefix in product(labels, repeat=6):
        if not any(item.startswith("uranium_") for item in prefix):
            continue
        if any(prefix.count(item) > cap for item, cap in zip(items, caps, strict=True)):
            continue
        rods = sum(
            1 if item == "uranium_single" else 2 if item == "uranium_dual" else 0
            for item in prefix
        )
        if rods > 4:
            continue
        skeleton = tuple([*prefix, *("empty" for _ in range(12))])
        brute_maximum = max(brute_maximum, skeleton_eu_per_tick(skeleton, 3))

    assert table.build() == brute_maximum
