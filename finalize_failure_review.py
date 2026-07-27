"""Compatibility entry point for immutable failure-review finalization."""

from carla_vision.failure_mining.review import main

if __name__ == "__main__":
    raise SystemExit(main())
