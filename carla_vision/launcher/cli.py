"""Command-line access to the same use cases as the full-screen launcher."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import LauncherConfig, LauncherError, launcher_lock, source_root
from .install import setup
from .services import LocalStack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage local CARLA/Worker and remote web-app profiles.")
    parser.add_argument("--workspace", type=Path, default=source_root())
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("tui", help="Open the full-screen installer and launcher")
    install = commands.add_parser("setup", help="Install/repair locked dependencies and model files")
    install.add_argument("--carla-root", default=None)
    install.add_argument("--profile", choices=("lean", "vision", "research"), default=None)
    install.add_argument("--mode", choices=("local", "web"), default=None)
    install.add_argument("--download-carla", action="store_true")
    start = commands.add_parser("start", help="Start CARLA, Worker and web app; reuse an existing server")
    start.add_argument("--no-browser", action="store_true")
    commands.add_parser("stop", help="Stop only launcher-owned services after scene cleanup")
    commands.add_parser("status", help="Read local service readiness and model presence")
    commands.add_parser("doctor", help="Show the same read-only readiness diagnostics")
    args = parser.parse_args(argv)
    root = args.workspace.expanduser().resolve()
    try:
        if args.command in {None, "tui"}:
            try:
                from .tui import LauncherApp
            except ModuleNotFoundError as error:
                if error.name and error.name.startswith("textual"):
                    raise LauncherError("Run the platform setup launcher to install the TUI.") from error
                raise
            with launcher_lock(root):
                LauncherApp(root).run()
            return 0
        stack = LocalStack(root)
        if args.command == "setup":
            config = LauncherConfig.load(root)
            values = {}
            if args.carla_root is not None:
                values["carla_root"] = args.carla_root
            if args.profile is not None:
                values["profile"] = args.profile
            if args.mode is not None:
                values["mode"] = args.mode
            if values:
                stack.configure(replace(config, **values))
            result = setup(root, download_carla=args.download_carla)
        elif args.command == "start":
            result = stack.start(open_browser=not args.no_browser)
        elif args.command == "stop":
            result = stack.stop()
        else:
            result = stack.snapshot()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if args.command != "doctor" or result["ready"] else 1
    except (LauncherError, OSError, ValueError) as error:
        print(f"CARLA app: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
