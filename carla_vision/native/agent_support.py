"""Find installed CARLA agents, or use the bundled official 0.9.16 sources.

Standard library only: safe to load beside a standalone World Worker without
importing the research package or installing a second Python environment.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_navigation_module(name: str, *, carla_version: str = "0.9.16") -> ModuleType:
    if name not in {"behavior_agent", "global_route_planner"}:
        raise ValueError("unsupported CARLA navigation module")
    module = f"agents.navigation.{name}"
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        # Do not mask a broken installed agents package or missing dependencies.
        if error.name != "agents":
            raise
        bundled = Path(__file__).resolve().parents[1] / "_vendor" / "carla_0_9_16"
        if carla_version != "0.9.16" or not (bundled / "agents").is_dir():
            raise ImportError(
                "Install the matching CARLA PythonAPI/carla/agents in the Worker's "
                "environment; bundled agents support CARLA 0.9.16 only"
            ) from error
        if str(bundled) not in sys.path:
            sys.path.insert(0, str(bundled))
        importlib.invalidate_caches()
        return importlib.import_module(module)
