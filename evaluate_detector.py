"""CLI compatibility entry point for canonical detector evaluation."""

from carla_vision.evaluation.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
