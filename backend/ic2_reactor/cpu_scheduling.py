from __future__ import annotations

import ctypes
import os
import struct
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CpuSet:
    id: int
    group: int
    logical_processor_index: int
    core_index: int
    efficiency_class: int
    parked: bool = False
    allocated: bool = False


@dataclass(frozen=True, slots=True)
class CpuSchedulingPlan:
    worker_cpu_set_ids: tuple[int, ...]
    reserved_cpu_set_ids: tuple[int, ...]
    available_logical_processors: int
    available_physical_cores: int

    @property
    def recommended_workers(self) -> int:
        return max(1, len(self.worker_cpu_set_ids))


def windows_cpu_sets() -> tuple[CpuSet, ...]:
    """Return Windows CPU Sets, including hybrid-core efficiency classes."""
    if sys.platform != "win32":
        return ()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetSystemCpuSetInformation
    query.argtypes = (
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
        ctypes.c_ulong,
    )
    query.restype = ctypes.c_int

    required = ctypes.c_ulong(0)
    query(None, 0, ctypes.byref(required), None, 0)
    if required.value == 0:
        return ()
    buffer = ctypes.create_string_buffer(required.value)
    if not query(buffer, required.value, ctypes.byref(required), None, 0):
        return ()

    raw = memoryview(buffer).cast("B")
    result: list[CpuSet] = []
    offset = 0
    while offset + 8 <= required.value:
        size, information_type = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > required.value:
            break
        # CpuSetInformation == 0. The documented x64/x86 fields through
        # AllFlags occupy offsets 8..19 and are stable across both layouts.
        if information_type == 0 and size >= 20:
            cpu_set_id = struct.unpack_from("<I", raw, offset + 8)[0]
            group = struct.unpack_from("<H", raw, offset + 12)[0]
            logical_index = raw[offset + 14]
            core_index = raw[offset + 15]
            efficiency_class = raw[offset + 18]
            flags = raw[offset + 19]
            result.append(CpuSet(
                id=cpu_set_id,
                group=group,
                logical_processor_index=logical_index,
                core_index=core_index,
                efficiency_class=efficiency_class,
                parked=bool(flags & 0x01),
                allocated=bool(flags & 0x02),
            ))
        offset += size
    return tuple(result)


def cpu_scheduling_plan(
    reserve_physical_cores: int = 2,
    cpu_sets: tuple[CpuSet, ...] | None = None,
) -> CpuSchedulingPlan:
    """Reserve the least performant physical cores and expose a worker pool."""
    values = windows_cpu_sets() if cpu_sets is None else cpu_sets
    # ``Parked`` is a dynamic power-management state and Windows will unpark a
    # selected CPU under sustained load. ``Allocated`` likewise describes CPU
    # Set allocation policy, not permanent hardware availability. Keeping both
    # in the candidate pool avoids shrinking a plan according to one idle-time
    # snapshot.
    available = tuple(values)
    if not available:
        logical = max(1, os.cpu_count() or 1)
        reserved = min(max(0, reserve_physical_cores), max(0, logical - 1))
        return CpuSchedulingPlan(
            worker_cpu_set_ids=(),
            reserved_cpu_set_ids=(),
            available_logical_processors=logical,
            available_physical_cores=logical,
        )

    cores: dict[tuple[int, int], list[CpuSet]] = {}
    for value in available:
        cores.setdefault((value.group, value.core_index), []).append(value)
    reserve_count = min(max(0, reserve_physical_cores), max(0, len(cores) - 1))
    # A higher EfficiencyClass means a faster, less power-efficient core.
    # Reserve low-class E cores first so P-core throughput remains available.
    ordered_cores = sorted(
        cores.values(),
        key=lambda core: (
            max(value.efficiency_class for value in core),
            max(value.logical_processor_index for value in core),
        ),
    )
    reserved_values = {
        value.id
        for core in ordered_cores[:reserve_count]
        for value in core
    }
    worker_ids = tuple(value.id for value in available if value.id not in reserved_values)
    reserved_ids = tuple(value.id for value in available if value.id in reserved_values)
    return CpuSchedulingPlan(
        worker_cpu_set_ids=worker_ids,
        reserved_cpu_set_ids=reserved_ids,
        available_logical_processors=len(available),
        available_physical_cores=len(cores),
    )


def recommended_cpu_workers(reserve_physical_cores: int = 2) -> int:
    plan = cpu_scheduling_plan(reserve_physical_cores)
    if plan.worker_cpu_set_ids:
        return plan.recommended_workers
    return max(1, plan.available_logical_processors - reserve_physical_cores)


def configure_current_process_cpu_sets(
    cpu_set_ids: tuple[int, ...],
    *,
    high_performance: bool,
) -> bool:
    """Apply a migratable CPU Set pool and explicitly select HighQoS."""
    if sys.platform != "win32" or not cpu_set_ids:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process = kernel32.GetCurrentProcess()

    set_cpu_sets = kernel32.SetProcessDefaultCpuSets
    set_cpu_sets.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    )
    set_cpu_sets.restype = ctypes.c_int
    ids = (ctypes.c_ulong * len(cpu_set_ids))(*cpu_set_ids)
    affinity_ok = bool(set_cpu_sets(process, ids, len(cpu_set_ids)))

    if high_performance:
        class PowerThrottlingState(ctypes.Structure):
            _fields_ = (
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            )

        set_information = kernel32.SetProcessInformation
        set_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        )
        set_information.restype = ctypes.c_int
        # ProcessPowerThrottling=4, current version=1 and execution-speed bit=1.
        # Control the bit while leaving StateMask clear to explicitly disable
        # EcoQoS execution-speed throttling for compute workers.
        state = PowerThrottlingState(1, 1, 0)
        set_information(process, 4, ctypes.byref(state), ctypes.sizeof(state))
    return affinity_ok


def initialize_compute_worker(cpu_set_ids: tuple[int, ...]) -> None:
    configure_current_process_cpu_sets(cpu_set_ids, high_performance=True)


def initialize_gpu_service(cpu_set_ids: tuple[int, ...]) -> None:
    configure_current_process_cpu_sets(cpu_set_ids, high_performance=False)
