"""CLI compatibility entry point for deterministic scenario planning."""

from carla_vision.scenarios.planner import main

if __name__ == "__main__":
    raise SystemExit(main())
