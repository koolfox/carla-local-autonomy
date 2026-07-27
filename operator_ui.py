"""CLI wrapper for the local CARLA Vision operator panel."""

from __future__ import annotations

from carla_vision.operator.server import main

if __name__ == "__main__":
    raise SystemExit(main())
