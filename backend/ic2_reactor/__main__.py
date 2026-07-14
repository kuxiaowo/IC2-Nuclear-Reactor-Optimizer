from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 IC2 核反应堆模拟与优化器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def serve(host: str, port: int, no_browser: bool = False) -> None:
    from .api import app

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if not no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{browser_host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    serve(args.host, args.port, args.no_browser)


if __name__ == "__main__":
    main()
