"""Translate private stop requests into graceful shutdown on both target OSes."""

from __future__ import annotations

import _thread
import argparse
import signal
import threading
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("web", "worker"), required=True)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    finished = threading.Event()

    def stop_signal(*_):
        raise KeyboardInterrupt

    def watch() -> None:
        while not finished.wait(0.2):
            if args.control_file.exists():
                _thread.interrupt_main()
                return

    signal.signal(signal.SIGTERM, stop_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop_signal)
    threading.Thread(target=watch, daemon=True, name="launcher-stop-request").start()
    try:
        if args.role == "web":
            from ..operator.product_console import main as run
        else:
            from ..native.observable_world_worker import main as run
        return run(arguments)
    except KeyboardInterrupt:
        return 0
    finally:
        finished.set()
        args.control_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
