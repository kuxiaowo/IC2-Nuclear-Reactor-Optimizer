"""Resume or re-pause an externally suspended Windows optimizer job.

This compatibility helper is only for jobs started before native pause support
was loaded by the server. Native jobs should use the HTTP pause/resume buttons.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from datetime import datetime
from pathlib import Path


PROCESS_SUSPEND_RESUME = 0x0800
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32)
kernel32.OpenProcess.restype = ctypes.c_void_p
kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
ntdll.NtSuspendProcess.argtypes = (ctypes.c_void_p,)
ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)


def control_process(process_id: int, *, pause: bool) -> None:
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, process_id)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"cannot open process {process_id}")
    try:
        function = ntdll.NtSuspendProcess if pause else ntdll.NtResumeProcess
        status = function(handle)
        if status != 0:
            raise OSError(status, f"native process control failed for {process_id}")
    finally:
        kernel32.CloseHandle(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "pause", "resume"))
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    paused = bool(payload.get("paused"))
    if args.action == "status":
        print("paused" if paused else "running")
        return
    target_paused = args.action == "pause"
    if paused == target_paused:
        print("already paused" if paused else "already running")
        return

    failures: list[str] = []
    for process_id in payload["worker_pids"]:
        try:
            control_process(int(process_id), pause=target_paused)
        except OSError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("\n".join(failures))

    payload["paused"] = target_paused
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    payload.setdefault("snapshot", {})["effective_status"] = (
        "paused" if target_paused else "running"
    )
    temporary = args.checkpoint.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.checkpoint)
    print(f"controlled {len(payload['worker_pids'])} workers: {args.action}")


if __name__ == "__main__":
    main()
