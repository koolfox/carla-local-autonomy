"""CLI compatibility entry point for the native CARLA dataset worker."""

from carla_vision.native.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
