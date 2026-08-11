#!/usr/bin/env python3
"""Compatibility wrapper for the current ``carla-local-drive`` entry point."""

from carla_vision.voxel.local_drive_actuation import main


if __name__ == "__main__":
    raise SystemExit(main())
