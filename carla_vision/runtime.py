from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from .artifacts import RunArtifactTracker, fingerprint_file
from .bridge import (
    CarlaCameraStream,
    CarlaRpc,
    spawn_front_camera,
    spectator_chase_transform,
    vehicle_transform_from_front_camera,
)
from .contracts import DetectorConfig, PerceptionResult
from .controller import ControlCommand, PurePursuitController
from .detectors import create_detector
from .display import DisplayMode, LiveViewer, OverlayRenderer
from .model_release.verified import VerifiedModel, load_verified_model
from .perception import PerceptionWorker
from .policy import (
    ShadowPolicyRunner,
    VisionControlProposal,
    VisionPolicyConfig,
    create_vision_policy,
)
from .recording import AsyncVideoRecorder, RecordingStats
from .risk import HazardPolicy, RiskAssessment
from .watchdog import SafeActuator


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT syntax")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution values must be integers") from exc
    if width < 320 or height < 180 or width > 3840 or height > 2160:
        raise argparse.ArgumentTypeError("resolution must be between 320x180 and 3840x2160")
    return width, height


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"value must be valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must contain a JSON object")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model-neutral CARLA RGB perception and research runtime"
    )
    parser.add_argument("--host", default="172.20.10.7")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle-id", type=int, default=24)
    parser.add_argument("--camera-id", type=int, default=25)
    parser.add_argument("--resolution", type=parse_resolution, default=(640, 384))
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument(
        "--expected-map",
        help="fail before sensor/model startup unless the CARLA map name matches",
    )

    parser.add_argument(
        "--detector",
        default=None,
        choices=("yolo", "rtdetr", "custom"),
    )
    parser.add_argument(
        "--model-package",
        help="verified immutable model package; replaces detector/weights/factory/image-size",
    )
    parser.add_argument("--weights")
    parser.add_argument("--detector-factory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument(
        "--shadow-policy",
        choices=("hazard-stop", "custom"),
        help=(
            "run a vision-only policy proposal lane without applying its controls; "
            "may accompany perception-only or teacher motion"
        ),
    )
    parser.add_argument("--policy-factory")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--policy-options", type=parse_json_object, default={})

    parser.add_argument(
        "--control",
        default="none",
        choices=("none", "teacher", "vision"),
        help="teacher may use simulator pose; vision is reserved for the future RGB policy",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cruise-speed", type=float, default=2.5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-stale-seconds", type=float, default=0.50)
    parser.add_argument(
        "--spectator-follow",
        action="store_true",
        help="move the optional CARLA server spectator behind the selected vehicle",
    )

    parser.add_argument(
        "--view",
        default="overlay",
        choices=("none", "overlay", "split"),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    if args.drive:
        if args.control not in {"none", "teacher"}:
            parser.error("--drive conflicts with --control vision")
        args.control = "teacher"
    if args.control == "vision":
        parser.error(
            "--control vision is intentionally unavailable until the RGB temporal "
            "policy passes its closed-loop acceptance gates"
        )
    if args.shadow_policy is None:
        if args.policy_factory or args.policy_checkpoint or args.policy_options:
            parser.error(
                "--policy-factory, --policy-checkpoint, and --policy-options "
                "require --shadow-policy"
            )
    elif args.shadow_policy == "custom" and not args.policy_factory:
        parser.error("--shadow-policy custom requires --policy-factory module:callable")
    elif args.shadow_policy != "custom" and args.policy_factory:
        parser.error("--policy-factory is allowed only with --shadow-policy custom")
    if not args.policy_device.strip():
        parser.error("--policy-device must not be empty")
    if args.duration <= 0.0 or args.duration > 3600.0:
        parser.error("--duration must be in (0, 3600] seconds")
    if args.camera_fps <= 0.0 or args.camera_fps > 60.0:
        parser.error("--camera-fps must be in (0, 60]")
    if not 30.0 <= args.camera_fov <= 150.0:
        parser.error("--camera-fov must be between 30 and 150 degrees")
    if args.model_package:
        conflicting = [
            option
            for option, value in (
                ("--detector", args.detector),
                ("--weights", args.weights),
                ("--detector-factory", args.detector_factory),
                ("--image-size", args.image_size),
            )
            if value is not None
        ]
        if conflicting:
            parser.error("--model-package cannot be combined with " + ", ".join(conflicting))
    else:
        args.detector = args.detector or "yolo"
        args.image_size = args.image_size or 640
    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.max_stale_seconds <= 0.0:
        parser.error("--max-stale-seconds must be positive")
    if args.video_fps is not None and args.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if not args.model_package and args.detector == "custom" and not args.detector_factory:
        parser.error("--detector custom requires --detector-factory module:callable")
    return args


def _default_weights(detector: str) -> str | None:
    if detector == "yolo":
        return "yolo26n.pt"
    if detector == "rtdetr":
        return "rtdetr-l.pt"
    return None


def _resolve_detector(
    args: argparse.Namespace,
) -> tuple[DetectorConfig, VerifiedModel | None]:
    if args.model_package:
        model = load_verified_model(args.model_package)
        return (
            model.detector_config(
                device=args.device,
                confidence=args.confidence,
            ),
            model,
        )
    weights_value = args.weights or _default_weights(args.detector)
    return (
        DetectorConfig(
            backend=args.detector,
            weights=Path(weights_value) if weights_value is not None else None,
            device=args.device,
            image_size=args.image_size,
            confidence=args.confidence,
            factory=args.detector_factory,
        ),
        None,
    )


def _detector_config(args: argparse.Namespace) -> DetectorConfig:
    return _resolve_detector(args)[0]


def _policy_config(args: argparse.Namespace) -> VisionPolicyConfig | None:
    if args.shadow_policy is None:
        return None
    return VisionPolicyConfig(
        backend=args.shadow_policy,
        factory=args.policy_factory,
        checkpoint=(Path(args.policy_checkpoint) if args.policy_checkpoint is not None else None),
        device=args.policy_device,
        options=args.policy_options,
    )


def _actor_attributes(actor: list[Any] | None) -> dict[str, str]:
    if actor is None or len(actor) < 3 or len(actor[2]) < 3:
        return {}
    return {item[0]: str(item[2]) for item in actor[2][2]}


def camera_actor(
    rpc: CarlaRpc,
    *,
    vehicle_id: int,
    camera_id: int,
    width: int,
    height: int,
    camera_fps: float,
    fov: float,
) -> tuple[list[Any], bool]:
    actor = rpc.actor(camera_id)
    attributes = _actor_attributes(actor)
    expected_fov = float(attributes.get("fov", "nan"))
    expected_sensor_tick = float(attributes.get("sensor_tick", "nan"))
    requested_sensor_tick = 1.0 / camera_fps
    if (
        actor is not None
        and actor[1] == vehicle_id
        and actor[2][1] == "sensor.camera.rgb"
        and actor[5]
        and attributes.get("role_name") == "front"
        and attributes.get("image_size_x") == str(width)
        and attributes.get("image_size_y") == str(height)
        and math.isfinite(expected_fov)
        and abs(expected_fov - fov) < 0.01
        and math.isfinite(expected_sensor_tick)
        and abs(expected_sensor_tick - requested_sensor_tick) < 1e-6
    ):
        return actor, False
    actor = spawn_front_camera(
        rpc,
        vehicle_id,
        width=width,
        height=height,
        sensor_tick=requested_sensor_tick,
        fov=fov,
    )
    return actor, True


def validate_camera_mount(
    rpc: CarlaRpc,
    vehicle_id: int,
    frame: Any,
) -> None:
    inferred = vehicle_transform_from_front_camera(frame)
    actual = rpc.actor_transform(vehicle_id)
    location_error = math.dist(inferred[0], actual[0])
    rotation_error = max(
        abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)
        for left, right in zip(inferred[1], actual[1], strict=True)
    )
    if location_error > 0.10 or rotation_error > 1.0:
        raise RuntimeError(
            "front camera mount validation failed: "
            f"location_error={location_error:.3f}m "
            f"rotation_error={rotation_error:.2f}deg"
        )


def brake_and_verify_stop(
    rpc: CarlaRpc,
    vehicle_id: int,
    timeout: float = 5.0,
) -> Any:
    deadline = time.monotonic() + timeout
    telemetry = rpc.telemetry(vehicle_id)
    while abs(telemetry.speed) >= 0.05 and time.monotonic() < deadline:
        rpc.apply_vehicle_control(
            vehicle_id,
            ControlCommand.service_brake().as_carla(),
        )
        time.sleep(0.08)
        telemetry = rpc.telemetry(vehicle_id)
    rpc.apply_vehicle_control(vehicle_id, ControlCommand.parked().as_carla())
    telemetry = rpc.telemetry(vehicle_id)
    if abs(telemetry.speed) >= 0.05:
        raise RuntimeError(f"vehicle {vehicle_id} did not stop; speed={telemetry.speed:.3f}m/s")
    return telemetry


def _safe_server_version(rpc: CarlaRpc) -> str | None:
    try:
        return str(rpc.value_call("version"))
    except Exception:
        return None


def _serialize_detection_log(
    result: PerceptionResult,
    risk: RiskAssessment,
    *,
    mode: str,
    simulator_speed: float,
    route_progress: str,
    command: ControlCommand | None,
    shadow_proposal: VisionControlProposal | None,
    latest_camera_sequence: int,
    latest_camera_timestamp: float,
    ego_transform: list[list[float]],
) -> dict[str, Any]:
    assessments = {item.detection_index: item for item in risk.items}
    detections: list[dict[str, Any]] = []
    for index, detection in enumerate(result.detections):
        assessment = assessments[index]
        detections.append(
            {
                "class_id": detection.class_id,
                "source_class_id": detection.source_class_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "xyxy": list(detection.xyxy),
                "attributes": dict(detection.attributes),
                "risk": {
                    "in_driving_corridor": assessment.in_driving_corridor,
                    "visually_close": assessment.visually_close,
                    "hazard": assessment.hazard,
                    "confidence_threshold": assessment.confidence_threshold,
                },
            }
        )
    return {
        "sequence": result.sequence,
        "carla_frame": result.carla_frame,
        "source_timestamp": result.source_timestamp,
        "source_received_monotonic": result.source_received_monotonic,
        "inference_started_monotonic": result.inference_started_monotonic,
        "completed_monotonic": result.completed_monotonic,
        "pipeline_latency_seconds": result.inference_seconds,
        "model_inference_seconds": result.model_inference_seconds,
        "detector": result.detector_name,
        "source_camera": {
            "transform": (
                list(result.source_transform) if result.source_transform is not None else None
            ),
            "fov_degrees": result.source_fov,
            "width": int(result.source_bgr.shape[1]),
            "height": int(result.source_bgr.shape[0]),
        },
        "detections": detections,
        "hazard": risk.hazard,
        "mode": mode,
        "simulator_speed_mps": simulator_speed,
        "route_progress": route_progress,
        "decision_context": {
            "latest_camera_sequence": latest_camera_sequence,
            "latest_camera_timestamp": latest_camera_timestamp,
            "ego_transform_privileged": ego_transform,
            "teacher_control": (
                {
                    "throttle": command.throttle,
                    "steer": command.steer,
                    "brake": command.brake,
                    "hand_brake": command.hand_brake,
                }
                if command is not None
                else None
            ),
            "vision_shadow": (
                {
                    **shadow_proposal.as_dict(),
                    "actuation_applied": False,
                }
                if shadow_proposal is not None
                else None
            ),
        },
    }


def _write_json_line(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stream.write("\n")


def _error_record(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__qualname__,
        "module": type(error).__module__,
        "message": str(error),
    }


def _close_viewer(viewer: LiveViewer | None, errors: list[dict[str, str]]) -> None:
    if viewer is None:
        return
    try:
        viewer.close()
    except BaseException as exc:
        errors.append(_error_record(exc))


def _close_recorder(
    recorder: AsyncVideoRecorder | None,
    errors: list[dict[str, str]],
) -> RecordingStats | None:
    if recorder is None:
        return None
    try:
        return recorder.close()
    except BaseException as exc:
        errors.append(_error_record(exc))
        return None


def _register_existing_artifact(
    tracker: RunArtifactTracker,
    path: Path,
    *,
    role: str,
    metadata: dict[str, Any] | None,
    errors: list[dict[str, str]],
) -> None:
    if not path.is_file():
        return
    try:
        tracker.register_artifact(path, role=role, metadata=metadata)
    except BaseException as exc:
        errors.append(_error_record(exc))


def run(args: argparse.Namespace) -> dict[str, Any]:
    detector_config, verified_model = _resolve_detector(args)
    policy_config = _policy_config(args)
    width, height = args.resolution
    control_mode = args.control
    model_reference = (
        dict(verified_model.reference)
        if verified_model is not None
        else {
            "backend": detector_config.backend,
            "weights": (
                str(detector_config.weights) if detector_config.weights is not None else None
            ),
            "factory": detector_config.factory,
        }
    )
    policy_reference: dict[str, Any] | None = None
    if policy_config is not None:
        policy_reference = {
            "kind": "vision_shadow_policy",
            "configuration": policy_config.as_dict(),
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "actuation_authorized": False,
        }
        if policy_config.checkpoint is not None:
            if not policy_config.checkpoint.is_file():
                raise FileNotFoundError(
                    f"vision policy checkpoint not found: {policy_config.checkpoint}"
                )
            policy_reference["checkpoint"] = fingerprint_file(policy_config.checkpoint)

    with CarlaRpc(args.host, args.port, timeout=2.0) as rpc:
        vehicle = rpc.actor(args.vehicle_id)
        if vehicle is None or not vehicle[2][1].startswith("vehicle."):
            raise RuntimeError(f"vehicle actor {args.vehicle_id} was not found")
        map_name = str(rpc.value_call("get_map_info")[0])
        if args.expected_map is not None:
            actual_map = map_name.rsplit("/", maxsplit=1)[-1].removesuffix(".umap")
            expected_map = str(args.expected_map).rsplit("/", maxsplit=1)[-1].removesuffix(".umap")
            if actual_map != expected_map:
                raise RuntimeError(f"CARLA map mismatch: expected {expected_map}, got {actual_map}")
        if control_mode == "teacher" and "Town10HD" not in map_name:
            raise RuntimeError(
                f"the built-in simulator-teacher route supports Town10HD, got {map_name}"
            )
        server_version = _safe_server_version(rpc)

        tracker = RunArtifactTracker(
            args.runs_root,
            run_id=args.run_id,
            cli_args=sys.argv[1:],
            config={
                **vars(args),
                "resolution": [width, height],
                "detector_config": detector_config.as_dict(),
                "model_package": (
                    dict(verified_model.reference) if verified_model is not None else None
                ),
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "control_mode": control_mode,
                "shadow_policy": (policy_config.as_dict() if policy_config is not None else None),
            },
            repository_root=Path.cwd(),
            carla_endpoint={"host": args.host, "port": args.port},
            carla_version=server_version,
            carla_map=map_name,
            model_refs=(
                (model_reference, policy_reference)
                if policy_reference is not None
                else (model_reference,)
            ),
        )

        with tracker:
            print(f"run_id={tracker.run_id} run_dir={tracker.run_dir}", flush=True)
            video_path = tracker.artifact_path("video/overlay.mp4")
            latest_frame_path = tracker.artifact_path("images/latest_overlay.png")
            detections_path = tracker.artifact_path("logs/detections.jsonl")
            policy_log_path = tracker.artifact_path("logs/policy_shadow.jsonl")
            policy_audit_path = tracker.artifact_path("policy_input_audit.json")
            summary_path = tracker.artifact_path("summary.json")

            summary: dict[str, Any] = {
                "run_id": tracker.run_id,
                "host": args.host,
                "port": args.port,
                "carla_version": server_version,
                "map": map_name,
                "vehicle_id": args.vehicle_id,
                "requested_camera_id": args.camera_id,
                "resolution": [width, height],
                "camera_fps": args.camera_fps,
                "camera_fov": args.camera_fov,
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "control_mode": control_mode,
                "teacher_uses_privileged_pose": control_mode == "teacher",
                "vision_shadow_enabled": policy_config is not None,
                "vision_shadow_actuation_authorized": False,
                "spectator_follow_requested": bool(args.spectator_follow),
                "spectator_follow_enabled": False,
                "spectator_actor_id": None,
                "spectator_update_count": 0,
                "spectator_restore_supported": False,
                "spectator_restored": None,
                "vision_shadow_policy_config": (
                    policy_config.as_dict() if policy_config is not None else None
                ),
                "duration_requested": args.duration,
                "detector_config": detector_config.as_dict(),
                "model_package": (
                    dict(verified_model.reference) if verified_model is not None else None
                ),
                "status": "running",
            }

            camera: list[Any] | None = None
            camera_was_spawned = False
            viewer: LiveViewer | None = None
            recorder: AsyncVideoRecorder | None = None
            detections_stream: TextIO | None = None
            policy_stream: TextIO | None = None
            shadow_runner: ShadowPolicyRunner | None = None
            actuator: SafeActuator | None = None
            actuator_stop_ack: bool | None = None
            verified_stop_telemetry: Any | None = None
            recording_stats: RecordingStats | None = None
            run_error: BaseException | None = None
            run_traceback: TracebackType | None = None
            finalization_errors: list[dict[str, str]] = []
            started = time.monotonic()
            run_elapsed = 0.0
            distance_travelled = 0.0
            max_speed = 0.0
            spectator_id: int | None = None
            spectator_rpc: CarlaRpc | None = None
            spectator_active = False
            spectator_episode_id: int | None = None
            spectator_initial_transform: list[list[float]] | None = None
            spectator_last_sequence = -1
            spectator_update_count = 0
            estimated_speed = 0.0
            last_logged_sequence = -1
            stop_reason = "duration_complete"

            try:
                camera, camera_was_spawned = camera_actor(
                    rpc,
                    vehicle_id=args.vehicle_id,
                    camera_id=args.camera_id,
                    width=width,
                    height=height,
                    camera_fps=args.camera_fps,
                    fov=args.camera_fov,
                )
                camera_id = int(camera[0])
                summary["camera_id"] = camera_id
                summary["camera_was_spawned"] = camera_was_spawned
                print(
                    f"vehicle={args.vehicle_id} camera={camera_id} "
                    f"map={map_name} control={control_mode}",
                    flush=True,
                )
                if args.spectator_follow:
                    try:
                        spectator_rpc = CarlaRpc(args.host, args.port, timeout=2.0)
                        spectator = spectator_rpc.spectator()
                        spectator_id = int(spectator[0])
                        spectator_episode_id = spectator_rpc.episode_id()
                        spectator_initial_transform = spectator_rpc.actor_transform(
                            spectator_id,
                            "Camera",
                        )
                        summary["spectator_actor_id"] = spectator_id
                        summary["spectator_follow_enabled"] = True
                        summary["spectator_restore_supported"] = True
                        spectator_active = True
                        print(
                            f"spectator_follow=enabled actor={spectator_id} view=chase",
                            flush=True,
                        )
                    except Exception as exc:
                        summary["spectator_follow_error"] = _error_record(exc)
                        if spectator_rpc is not None:
                            spectator_rpc.close()
                            spectator_rpc = None
                        print(
                            f"WARNING: spectator follow unavailable: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )

                detector = create_detector(detector_config)
                summary["detector"] = detector.metadata.as_dict()
                if detector_config.weights is not None and detector_config.weights.is_file():
                    resolved_weights = {
                        "kind": "resolved_weights",
                        "backend": detector.metadata.backend,
                        **fingerprint_file(detector_config.weights),
                    }
                    tracker.add_model_reference(resolved_weights)
                    summary["resolved_weights"] = resolved_weights
                renderer = OverlayRenderer(stale_after_seconds=args.max_stale_seconds)
                if args.view != "none":
                    viewer = LiveViewer(
                        renderer,
                        mode=DisplayMode(args.view),
                    )
                risk_policy = HazardPolicy()
                detections_stream = detections_path.open("w", encoding="utf-8")
                if policy_config is not None:
                    shadow_runner = ShadowPolicyRunner(create_vision_policy(policy_config))
                    summary["vision_shadow_policy"] = shadow_runner.policy.metadata.as_dict()
                    summary["vision_policy_input_audit"] = shadow_runner.audit.as_dict()
                    policy_audit_path.write_text(
                        json.dumps(
                            shadow_runner.audit.as_dict(),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    policy_stream = policy_log_path.open("w", encoding="utf-8")

                with CarlaCameraStream(args.host, camera[5]) as stream:
                    first_frame = stream.wait_for_frame(timeout=8.0)
                    with PerceptionWorker(detector) as perception:
                        perception.submit(first_frame)
                        first_result = perception.wait_for_result(timeout=30.0)
                        print(
                            f"detector={detector.name} "
                            f"warmup_detections={len(first_result.detections)} "
                            f"pipeline_ms={first_result.inference_seconds * 1000:.1f} "
                            f"model_ms="
                            f"{(first_result.model_inference_seconds or 0.0) * 1000:.1f}",
                            flush=True,
                        )

                        controller = PurePursuitController(cruise_speed=args.cruise_speed)
                        last_camera_sequence = first_frame.sequence
                        last_motion_sequence = first_frame.sequence
                        hazard_hold_until = 0.0
                        hazard_clear_since: float | None = None
                        first_transform = vehicle_transform_from_front_camera(first_frame)
                        if (
                            spectator_active
                            and spectator_id is not None
                            and spectator_rpc is not None
                        ):
                            try:
                                spectator_rpc.set_actor_transform(
                                    spectator_id,
                                    spectator_chase_transform(first_transform),
                                )
                                spectator_last_sequence = first_frame.sequence
                                spectator_update_count += 1
                            except Exception as exc:
                                summary["spectator_follow_error"] = _error_record(exc)
                                summary["spectator_follow_enabled"] = False
                                spectator_active = False
                                print(
                                    f"WARNING: spectator follow disabled: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        previous_location = (
                            float(first_transform[0][0]),
                            float(first_transform[0][1]),
                        )
                        previous_yaw = float(first_transform[1][1])
                        previous_camera_timestamp = first_frame.timestamp
                        mode = "PERCEPTION ONLY"
                        route_progress = "-"
                        last_shadow_sequence = -1
                        last_shadow_proposal: VisionControlProposal | None = None
                        last_loop = time.monotonic()

                        if control_mode == "teacher":
                            actuator = SafeActuator(
                                args.host,
                                args.port,
                                args.vehicle_id,
                                heartbeat_timeout=0.25,
                            )
                            mount_frame = stream.wait_for_frame(
                                after_sequence=last_camera_sequence,
                                timeout=5.0,
                            )
                            validate_camera_mount(rpc, args.vehicle_id, mount_frame)
                            fresh_frame = stream.wait_for_frame(
                                after_sequence=mount_frame.sequence,
                                timeout=5.0,
                            )
                            perception.submit(fresh_frame)
                            first_result = perception.wait_for_result(
                                after_sequence=first_result.sequence,
                                timeout=30.0,
                            )
                            last_camera_sequence = fresh_frame.sequence
                            last_motion_sequence = fresh_frame.sequence
                            fresh_transform = vehicle_transform_from_front_camera(fresh_frame)
                            previous_location = (
                                float(fresh_transform[0][0]),
                                float(fresh_transform[0][1]),
                            )
                            previous_yaw = float(fresh_transform[1][1])
                            previous_camera_timestamp = fresh_frame.timestamp
                            print("safe_actuator=ready", flush=True)

                        started = time.monotonic()
                        last_loop = started
                        while time.monotonic() - started < args.duration:
                            loop_started = time.monotonic()
                            frame = stream.latest()
                            if frame is None:
                                raise RuntimeError("camera has not produced a frame")
                            if frame.sequence > last_camera_sequence:
                                perception.submit(frame)
                                last_camera_sequence = frame.sequence

                            result = perception.latest()
                            if result is None:
                                raise RuntimeError("detector has not produced a result")
                            now = time.monotonic()
                            camera_age = now - frame.received_monotonic
                            inference_age = result.age_seconds(now)
                            simulation_lag = frame.timestamp - result.source_timestamp
                            result_height, result_width = result.source_bgr.shape[:2]
                            risk = risk_policy.assess(
                                result.detections,
                                result_width,
                                result_height,
                            )

                            # The audited shadow lane is evaluated before any
                            # privileged pose/speed values are computed. It
                            # receives a copied VisionObservation, never this
                            # PerceptionResult or a CARLA object.
                            if shadow_runner is not None and result.sequence > last_shadow_sequence:
                                last_shadow_proposal, policy_record = shadow_runner.propose(result)
                                if policy_stream is None:
                                    raise RuntimeError("vision shadow log stream is unavailable")
                                _write_json_line(policy_stream, policy_record)
                                last_shadow_sequence = result.sequence

                            # Pose and speed below are privileged simulator ground truth.
                            # They are allowed for teacher control and evaluation only.
                            transform = vehicle_transform_from_front_camera(frame)
                            if (
                                spectator_active
                                and spectator_id is not None
                                and spectator_rpc is not None
                                and frame.sequence > spectator_last_sequence
                            ):
                                try:
                                    spectator_rpc.set_actor_transform(
                                        spectator_id,
                                        spectator_chase_transform(transform),
                                    )
                                    spectator_last_sequence = frame.sequence
                                    spectator_update_count += 1
                                except Exception as exc:
                                    summary["spectator_follow_error"] = _error_record(exc)
                                    summary["spectator_follow_enabled"] = False
                                    spectator_active = False
                                    print(
                                        f"WARNING: spectator follow disabled: {exc}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                            current_location = (
                                float(transform[0][0]),
                                float(transform[0][1]),
                            )
                            if frame.sequence > last_motion_sequence:
                                segment = math.dist(previous_location, current_location)
                                simulation_dt = frame.timestamp - previous_camera_timestamp
                                if simulation_dt <= 1e-3:
                                    raise RuntimeError(
                                        "camera simulation timestamp did not advance"
                                    )
                                if segment > max(2.0, simulation_dt * 12.0):
                                    raise RuntimeError(
                                        "camera pose jumped beyond the motion safety bound"
                                    )
                                distance_travelled += segment
                                dx = current_location[0] - previous_location[0]
                                dy = current_location[1] - previous_location[1]
                                previous_yaw_radians = math.radians(previous_yaw)
                                instantaneous_speed = (
                                    dx * math.cos(previous_yaw_radians)
                                    + dy * math.sin(previous_yaw_radians)
                                ) / simulation_dt
                                estimated_speed = (
                                    0.35 * estimated_speed + 0.65 * instantaneous_speed
                                )
                                previous_location = current_location
                                previous_yaw = float(transform[1][1])
                                previous_camera_timestamp = frame.timestamp
                                last_motion_sequence = frame.sequence
                            max_speed = max(max_speed, abs(estimated_speed))

                            stale = (
                                camera_age > args.max_stale_seconds
                                or inference_age > args.max_stale_seconds
                                or simulation_lag < -1e-3
                                or simulation_lag > args.max_stale_seconds
                            )
                            command: ControlCommand | None = None
                            if control_mode == "teacher":
                                if actuator is None or not actuator.alive:
                                    raise RuntimeError("safe actuator stopped unexpectedly")
                                dt = now - last_loop
                                state = controller.compute(
                                    transform,
                                    max(0.0, estimated_speed),
                                    dt,
                                )
                                route_progress = (
                                    f"{state.nearest_index}/{len(controller.route) - 1}"
                                )
                                if risk.hazard:
                                    hazard_hold_until = now + 0.50
                                    hazard_clear_since = None
                                elif hazard_clear_since is None:
                                    hazard_clear_since = now

                                hazard_latched = now < hazard_hold_until or (
                                    hazard_clear_since is not None
                                    and now - hazard_clear_since < 1.0
                                    and hazard_hold_until > 0.0
                                )
                                if stale:
                                    command = ControlCommand.service_brake(
                                        steer=state.command.steer
                                    )
                                    mode = "BRAKE: STALE PERCEPTION"
                                elif estimated_speed < -0.20:
                                    command = ControlCommand.service_brake(
                                        steer=state.command.steer
                                    )
                                    mode = "BRAKE: REVERSE MOTION"
                                elif hazard_latched:
                                    command = ControlCommand.service_brake(
                                        steer=state.command.steer
                                    )
                                    mode = "BRAKE: VISUAL HAZARD"
                                elif not state.valid:
                                    command = ControlCommand.service_brake(
                                        steer=state.command.steer
                                    )
                                    mode = f"BRAKE: {state.reason.upper()}"
                                elif state.reason == "speed governor":
                                    command = state.command
                                    mode = "BRAKE: SPEED LIMIT"
                                elif state.reason == "speed hold":
                                    command = state.command
                                    mode = "SPEED HOLD"
                                elif state.done:
                                    command = ControlCommand.service_brake(
                                        steer=state.command.steer
                                    )
                                    mode = "BRAKE: ROUTE COMPLETE"
                                else:
                                    command = state.command
                                    mode = "SIMULATOR TEACHER"
                                actuator.send(command)
                                if state.done:
                                    stop_reason = "route_complete"
                            else:
                                mode = (
                                    "VISION SHADOW / NO ACTUATION"
                                    if shadow_runner is not None
                                    else "PERCEPTION ONLY"
                                )

                            worker_stats = perception.stats()
                            hud = {
                                "model": result.detector_name,
                                "mode": mode,
                                "sim speed": f"{estimated_speed:.2f} m/s",
                                "route": route_progress,
                                "camera age": f"{camera_age * 1000:.0f} ms",
                                "processed/dropped": (
                                    f"{worker_stats.processed}/"
                                    f"{worker_stats.dropped_before_inference}"
                                ),
                                "hazard": risk.hazard,
                            }
                            if command is not None:
                                hud["control"] = (
                                    f"T {command.throttle:.2f} "
                                    f"S {command.steer:.2f} B {command.brake:.2f}"
                                )
                            if last_shadow_proposal is not None:
                                hud["shadow proposal"] = (
                                    f"T {last_shadow_proposal.throttle:.2f} "
                                    f"S {last_shadow_proposal.steer:.2f} "
                                    f"B {last_shadow_proposal.brake:.2f} "
                                    "(NOT APPLIED)"
                                )

                            overlay = renderer.render(
                                result,
                                now_monotonic=now,
                                hud=hud,
                                stale=stale,
                                risk=risk,
                            )
                            if recorder is None and not args.no_video:
                                video_fps = args.video_fps or args.camera_fps
                                recorder = AsyncVideoRecorder(
                                    video_path,
                                    frame_size=(
                                        int(overlay.shape[1]),
                                        int(overlay.shape[0]),
                                    ),
                                    fps=video_fps,
                                    latest_frame_path=latest_frame_path,
                                )
                            if recorder is not None:
                                recorder.submit(result.sequence, overlay)

                            if result.sequence > last_logged_sequence:
                                _write_json_line(
                                    detections_stream,
                                    _serialize_detection_log(
                                        result,
                                        risk,
                                        mode=mode,
                                        simulator_speed=estimated_speed,
                                        route_progress=route_progress,
                                        command=command,
                                        shadow_proposal=last_shadow_proposal,
                                        latest_camera_sequence=frame.sequence,
                                        latest_camera_timestamp=frame.timestamp,
                                        ego_transform=transform,
                                    ),
                                )
                                labels = (
                                    ", ".join(
                                        f"{item.label}:{item.confidence:.2f}"
                                        for item in result.detections
                                    )
                                    or "none"
                                )
                                print(
                                    f"t={now - started:5.1f}s "
                                    f"sim_speed={estimated_speed:.2f} "
                                    f"mode={mode} detections=[{labels}]",
                                    flush=True,
                                )
                                last_logged_sequence = result.sequence

                            if viewer is not None:
                                live_bgr = frame.bgr() if viewer.mode is DisplayMode.SPLIT else None
                                if not viewer.show(
                                    result,
                                    live_bgr=live_bgr,
                                    now_monotonic=now,
                                    hud=hud,
                                    stale=stale,
                                    risk=risk,
                                ):
                                    stop_reason = "viewer_closed"
                                    break
                            if stop_reason == "route_complete":
                                break

                            last_loop = now
                            sleep_for = (1.0 / 30.0) - (time.monotonic() - loop_started)
                            if sleep_for > 0.0:
                                time.sleep(sleep_for)

                        run_elapsed = time.monotonic() - started
                        summary["perception_stats"] = perception.stats().__dict__
                        if shadow_runner is not None:
                            summary["vision_shadow_stats"] = shadow_runner.stats().as_dict()

            except BaseException as exc:
                run_error = exc
                run_traceback = exc.__traceback__
                summary["error"] = _error_record(exc)
            finally:
                run_elapsed = max(run_elapsed, time.monotonic() - started)
                if actuator is not None:
                    try:
                        actuator_stop_ack = actuator.stop()
                    except BaseException as exc:
                        finalization_errors.append(_error_record(exc))
                if shadow_runner is not None:
                    try:
                        summary["vision_shadow_stats"] = shadow_runner.stats().as_dict()
                        shadow_runner.close()
                    except BaseException as exc:
                        finalization_errors.append(_error_record(exc))
                if control_mode == "teacher":
                    try:
                        verified_stop_telemetry = brake_and_verify_stop(
                            rpc,
                            args.vehicle_id,
                        )
                    except BaseException as exc:
                        finalization_errors.append(_error_record(exc))
                _close_viewer(viewer, finalization_errors)
                recording_stats = _close_recorder(recorder, finalization_errors)
                if detections_stream is not None:
                    try:
                        detections_stream.flush()
                        detections_stream.close()
                    except BaseException as exc:
                        finalization_errors.append(_error_record(exc))
                if policy_stream is not None:
                    try:
                        policy_stream.flush()
                        policy_stream.close()
                    except BaseException as exc:
                        finalization_errors.append(_error_record(exc))
                if camera_was_spawned and camera is not None:
                    try:
                        rpc.destroy_actor(int(camera[0]))
                        summary["spawned_camera_destroyed"] = True
                    except BaseException as exc:
                        summary["spawned_camera_destroyed"] = False
                        finalization_errors.append(_error_record(exc))

                if spectator_rpc is not None:
                    try:
                        current_spectator = spectator_rpc.spectator()
                        current_spectator_id = int(current_spectator[0])
                        current_episode_id = spectator_rpc.episode_id()
                        if (
                            spectator_initial_transform is not None
                            and spectator_id is not None
                            and current_spectator_id == spectator_id
                            and current_episode_id == spectator_episode_id
                        ):
                            spectator_rpc.set_actor_transform(
                                spectator_id,
                                spectator_initial_transform,
                            )
                            summary["spectator_restored"] = True
                        else:
                            summary["spectator_restored"] = False
                            summary["spectator_restore_skipped"] = (
                                "CARLA episode or spectator actor changed during the run"
                            )
                    except Exception as exc:
                        summary["spectator_restored"] = False
                        summary["spectator_restore_error"] = _error_record(exc)
                        print(
                            f"WARNING: spectator restore failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    finally:
                        spectator_rpc.close()
                        spectator_rpc = None

                try:
                    final_telemetry = (
                        verified_stop_telemetry
                        if verified_stop_telemetry is not None
                        else rpc.telemetry(args.vehicle_id)
                    )
                    final_transform = rpc.actor_transform(args.vehicle_id)
                    summary.update(
                        {
                            "elapsed": run_elapsed,
                            "distance_travelled": distance_travelled,
                            "max_simulator_speed": max_speed,
                            "final_simulator_speed": final_telemetry.speed,
                            "final_position": final_transform[0][:2],
                            "actuator_stop_ack": actuator_stop_ack,
                            "stop_reason": stop_reason,
                        }
                    )
                except BaseException as exc:
                    finalization_errors.append(_error_record(exc))

                _register_existing_artifact(
                    tracker,
                    video_path,
                    role="annotated_video",
                    metadata=(
                        recording_stats.as_dict()
                        if recording_stats is not None
                        else {"partial": True}
                    ),
                    errors=finalization_errors,
                )
                _register_existing_artifact(
                    tracker,
                    latest_frame_path,
                    role="latest_overlay",
                    metadata={"frame_source": "detector_synchronized"},
                    errors=finalization_errors,
                )
                _register_existing_artifact(
                    tracker,
                    detections_path,
                    role="detections_jsonl",
                    metadata={"schema_version": "1.0"},
                    errors=finalization_errors,
                )
                _register_existing_artifact(
                    tracker,
                    policy_log_path,
                    role="vision_shadow_proposals",
                    metadata={
                        "schema_version": "1.0",
                        "actuation_applied": False,
                    },
                    errors=finalization_errors,
                )
                _register_existing_artifact(
                    tracker,
                    policy_audit_path,
                    role="vision_policy_input_audit",
                    metadata={
                        "schema_version": "1.0",
                        "status": "pass",
                    },
                    errors=finalization_errors,
                )
                if recording_stats is not None:
                    summary["recording_stats"] = recording_stats.as_dict()
                summary["spectator_update_count"] = spectator_update_count
                if finalization_errors:
                    summary["finalization_errors"] = finalization_errors
                summary["status"] = (
                    "success" if run_error is None and not finalization_errors else "failed"
                )
                try:
                    summary_path.write_text(
                        json.dumps(
                            summary,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    tracker.register_artifact(
                        summary_path,
                        role="run_summary",
                        metadata={"status": summary["status"]},
                    )
                except BaseException as exc:
                    finalization_errors.append(_error_record(exc))

            if run_error is not None:
                raise run_error.with_traceback(run_traceback)
            if finalization_errors:
                raise RuntimeError(
                    "runtime finalization failed: "
                    + "; ".join(item["message"] for item in finalization_errors)
                )

        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"manifest={tracker.manifest_path}", flush=True)
        return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
