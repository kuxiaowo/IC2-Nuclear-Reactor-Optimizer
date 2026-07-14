from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 IC2 核反应堆模拟与优化器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run("ic2_reactor.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

