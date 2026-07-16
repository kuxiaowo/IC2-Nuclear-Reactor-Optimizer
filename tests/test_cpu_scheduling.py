from ic2_reactor.cpu_scheduling import CpuSet, cpu_scheduling_plan


def test_hybrid_plan_reserves_two_e_cores_and_keeps_all_p_threads():
    values = []
    cpu_set_id = 0
    logical_index = 0
    for core_index in range(8):
        for _thread in range(2):
            values.append(CpuSet(
                id=cpu_set_id,
                group=0,
                logical_processor_index=logical_index,
                core_index=core_index,
                efficiency_class=1,
            ))
            cpu_set_id += 1
            logical_index += 1
    for core_index in range(8, 24):
        values.append(CpuSet(
            id=cpu_set_id,
            group=0,
            logical_processor_index=logical_index,
            core_index=core_index,
            efficiency_class=0,
        ))
        cpu_set_id += 1
        logical_index += 1

    plan = cpu_scheduling_plan(2, tuple(values))

    assert plan.available_physical_cores == 24
    assert plan.available_logical_processors == 32
    assert len(plan.worker_cpu_set_ids) == 30
    assert len(plan.reserved_cpu_set_ids) == 2
    assert set(range(16)).issubset(plan.worker_cpu_set_ids)
    assert set(plan.reserved_cpu_set_ids).issubset(set(range(16, 32)))
