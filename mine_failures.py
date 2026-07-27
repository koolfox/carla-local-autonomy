"""Compatibility entry point for validation-only failure mining."""

from carla_vision.failure_mining.miner import main

if __name__ == "__main__":
    raise SystemExit(main())
