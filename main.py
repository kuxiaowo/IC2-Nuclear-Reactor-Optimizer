"""Run the IC2 reactor optimizer directly from a source checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "backend"

# Windows ``spawn`` executes this file as ``__mp_main__`` without calling
# ``main()`` before unpickling worker functions.  The source checkout's
# package directory therefore has to be registered at module import time.
# Always import the package by its installed/canonical name afterwards so
# Numba caches and child processes never see both ``backend.ic2_reactor`` and
# ``ic2_reactor`` as different modules.
backend_path = str(BACKEND_ROOT)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


def main() -> None:
    # Keep static frontend discovery and trace storage independent of the
    # directory from which ``python path/to/main.py`` was invoked.
    os.chdir(PROJECT_ROOT)

    from ic2_reactor.__main__ import argument_parser, serve
    from ic2_reactor.launcher import FrontendBuildError, prepare_frontend

    parser = argument_parser()
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument("--rebuild", action="store_true", help="强制重新构建前端")
    build_group.add_argument("--no-build", action="store_true", help="跳过前端自动构建检查")
    args = parser.parse_args()

    print("=" * 62, flush=True)
    print(" IC2 Experimental 2.8.221 核反应堆模拟与优化器", flush=True)
    print("=" * 62, flush=True)
    print(f"[启动] 项目目录：{PROJECT_ROOT}", flush=True)
    print(f"[启动] Python：{sys.executable}", flush=True)
    print(f"[启动] 进程 PID：{os.getpid()}", flush=True)

    if not args.no_build:
        try:
            prepare_frontend(PROJECT_ROOT, force=args.rebuild)
        except FrontendBuildError as exc:
            parser.exit(1, f"启动失败：{exc}\n")
    else:
        print("[前端] 已按 --no-build 跳过自动构建检查。", flush=True)

    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"[服务] 正在启动：http://{browser_host}:{args.port}", flush=True)
    print("[服务] 按 Ctrl+C 停止；HTTP 请求日志将显示在此窗口。", flush=True)
    try:
        serve(args.host, args.port, args.no_browser)
    finally:
        print("[服务] 已停止。", flush=True)


if __name__ == "__main__":
    main()
