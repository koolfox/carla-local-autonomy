"""CLI compatibility entry point for detector training."""

from carla_vision.training.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
