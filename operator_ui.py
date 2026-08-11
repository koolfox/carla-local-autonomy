"""Compatibility wrapper for the current ``carla-operator-ui`` entry point."""

from __future__ import annotations

from carla_vision.operator.garage_server import main

if __name__ == "__main__":
    raise SystemExit(main())
