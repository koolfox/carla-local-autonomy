#!/usr/bin/env python3
"""Convenience wrapper for ``python -m carla_vision.local_drive``."""

from carla_vision.local_drive import main


if __name__ == "__main__":
    raise SystemExit(main())
