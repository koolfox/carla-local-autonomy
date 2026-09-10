"""Official-PythonAPI spawning and bounded exact-frame reads for teacher RGB rigs."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

from ..dataset.camera_rig import calibration
from ..scenarios.contracts import CameraRecipe
from .synchronization import NativeSensorQueue, QueuedSensorFrame, SensorFrameError
from .worker import EpisodeActors, _carla_transform


def read_frame_bundle(
    queues: dict[str, NativeSensorQueue], frame: int, timeout: float
) -> dict[str, QueuedSensorFrame]:
    """One deadline for the entire bundle; never substitute another frame."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("bundle timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    result = {}
    for name, sensor_queue in queues.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SensorFrameError(f"RGB bundle {frame} timed out; missing {name}")
        try:
            result[name] = sensor_queue.get_exact(frame, remaining)
        except SensorFrameError as error:
            raise SensorFrameError(f"RGB bundle {frame}; missing {name}: {error}") from error
    return result


class TeacherCameraRig:
    """All RGB cameras and the front label camera share one spawning batch."""

    def __init__(self, recipes: dict[str, CameraRecipe]) -> None:
        self.recipes = recipes
        self.queues: dict[str, NativeSensorQueue] = {}
        self.calibrations: dict[str, dict[str, Any]] = {}

    def spawn(self, session: Any, episode: Any, actors: EpisodeActors, sensors: list) -> None:
        names = ["front", "front_teacher", *(key for key in self.recipes if key != "front")]
        commands = []
        for name in names:
            camera = self.recipes["front" if name == "front_teacher" else name]
            camera_episode = replace(episode, recipe=replace(episode.recipe, camera=camera))
            blueprint = session._camera_blueprint(
                session.world.get_blueprint_library(),
                "sensor.camera.instance_segmentation"
                if name == "front_teacher"
                else "sensor.camera.rgb",
                camera_episode,
                role_name=name,
            )
            # Use a pinhole image so saved intrinsics describe the rendered pixels.
            for attribute in ("lens_k", "lens_kcube"):
                if blueprint.has_attribute(attribute):
                    blueprint.set_attribute(attribute, "0.0")
            mount = _carla_transform(session.carla, camera.mount)
            if name != "front_teacher":
                self.calibrations[name] = calibration(camera, mount.get_matrix())
            commands.append(session.carla.command.SpawnActor(blueprint, mount, actors.ego_id))
        responses = session.client.apply_batch_sync(commands, False)
        errors = []
        spawned = []
        for name, response in zip(names, responses, strict=True):
            if response.error:
                errors.append(f"{name}: {response.error}")
                continue
            actor_id = int(response.actor_id)
            # Register every successful spawn before raising for a partial batch.
            if name == "front":
                actors.rgb_sensor_id = actor_id
            elif name == "front_teacher":
                actors.teacher_sensor_id = actor_id
            else:
                actors.additional_sensor_ids.append(actor_id)
            spawned.append((name, actor_id))
        # Track the whole batch before any actor lookup can fail or time out.
        for name, actor_id in spawned:
            sensor = session.world.get_actor(actor_id)
            if sensor is None:
                errors.append(f"{name}: spawned camera {actor_id} is unavailable")
                continue
            sensors.append(sensor)
            self.queues[name] = NativeSensorQueue(name, max_frames=4)
        if errors:
            raise RuntimeError("camera rig spawn failed: " + "; ".join(errors))
        for name, sensor in zip(names, sensors[-len(names) :], strict=True):
            sensor.listen(self.queues[name].callback)

    def diagnostics(self) -> dict[str, Any]:
        return {
            name: {"received": item.received, "discarded": item.discarded}
            for name, item in self.queues.items()
        }
