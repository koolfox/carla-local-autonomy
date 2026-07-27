from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

TOWN10_BLOCK_ROUTE: tuple[tuple[float, float], ...] = (
    (-15.5, 24.6),
    (15.0, 24.7),
    (31.0, 24.8),
    (37.0, 27.0),
    (40.0, 33.0),
    (40.4, 45.0),
    (40.4, 56.0),
    (37.0, 62.0),
    (31.0, 65.5),
    (20.0, 66.3),
    (0.0, 66.3),
    (-20.0, 66.3),
    (-34.0, 65.0),
    (-39.5, 61.0),
    (-42.0, 54.0),
    (-42.0, 42.0),
    (-42.0, 33.0),
    (-39.0, 27.5),
    (-33.0, 24.8),
    (-22.0, 24.6),
    (-15.5, 24.6),
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_angle_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def densify_polyline(
    points: Iterable[tuple[float, float]],
    spacing: float = 1.0,
) -> list[tuple[float, float]]:
    source = list(points)
    if len(source) < 2:
        raise ValueError("a route needs at least two points")
    dense: list[tuple[float, float]] = [source[0]]
    for start, end in zip(source, source[1:], strict=False):
        length = math.dist(start, end)
        steps = max(1, math.ceil(length / spacing))
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return dense


@dataclass(frozen=True)
class ControlCommand:
    throttle: float
    steer: float
    brake: float
    hand_brake: bool = False

    def as_carla(self) -> list[float | bool | int]:
        return [
            float(self.throttle),
            float(self.steer),
            float(self.brake),
            bool(self.hand_brake),
            False,
            False,
            0,
        ]

    @classmethod
    def service_brake(cls, steer: float = 0.0) -> "ControlCommand":
        return cls(throttle=0.0, steer=steer, brake=1.0)

    @classmethod
    def parked(cls) -> "ControlCommand":
        return cls(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)


@dataclass(frozen=True)
class ControllerState:
    command: ControlCommand
    nearest_index: int
    target_index: int
    cross_track_error: float
    heading_error_degrees: float
    target_speed: float
    done: bool
    valid: bool
    reason: str


class PurePursuitController:
    """Simulator-teacher route follower for the known Town10HD demo loop."""

    def __init__(
        self,
        route: Iterable[tuple[float, float]] = TOWN10_BLOCK_ROUTE,
        spacing: float = 1.0,
        cruise_speed: float = 2.5,
        wheelbase: float = 2.875,
    ) -> None:
        self.route = densify_polyline(route, spacing)
        self.cruise_speed = cruise_speed
        self.wheelbase = wheelbase
        self._nearest_index: int | None = None
        self._speed_integral = 0.0
        self._last_steer = 0.0
        self._speed_governor_active = False

    def compute(
        self,
        transform: list[list[float]],
        speed: float,
        dt: float,
    ) -> ControllerState:
        location, rotation = transform
        x, y = float(location[0]), float(location[1])
        yaw = math.radians(float(rotation[1]))
        dt = clamp(dt, 0.02, 0.25)

        nearest = self._find_nearest(x, y)
        nearest_point = self.route[nearest]
        cross_track = math.hypot(x - nearest_point[0], y - nearest_point[1])
        if cross_track > 3.0:
            return self._invalid(nearest, cross_track, "cross-track error exceeded 3 m")

        lookahead = clamp(4.0 + 0.4 * speed, 4.0, 7.0)
        target = nearest
        accumulated = 0.0
        while target + 1 < len(self.route) and accumulated < lookahead:
            accumulated += math.dist(self.route[target], self.route[target + 1])
            target += 1

        target_point = self.route[target]
        dx = target_point[0] - x
        dy = target_point[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        alpha = math.atan2(local_y, max(local_x, 1e-6))
        heading_error_degrees = math.degrees(alpha)

        if local_x < -0.5 or abs(heading_error_degrees) > 70.0:
            return self._invalid(nearest, cross_track, "no safe forward route target")

        steering_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            max(lookahead, 1e-6),
        )
        desired_steer = clamp(steering_angle / 0.61, -0.65, 0.65)
        steer_delta = clamp(desired_steer - self._last_steer, -0.08, 0.08)
        steer = self._last_steer + steer_delta
        self._last_steer = steer

        curvature = self._route_curvature(nearest)
        target_speed = self.cruise_speed
        if curvature > 0.08 or abs(alpha) > 0.28:
            target_speed = min(target_speed, 1.8)

        speed_error = target_speed - speed
        if speed >= target_speed:
            self._speed_governor_active = True
        elif speed <= target_speed - 0.20:
            self._speed_governor_active = False

        if self._speed_governor_active:
            self._speed_integral = min(self._speed_integral, 0.0)
            command = ControlCommand(
                throttle=0.0,
                steer=steer,
                brake=0.35,
            )
            reason = "speed governor"
        elif speed >= target_speed - 0.35:
            # Cut throttle before the limit to account for camera/control latency.
            self._speed_integral = min(self._speed_integral, 0.0)
            command = ControlCommand(
                throttle=0.0,
                steer=steer,
                brake=0.0,
            )
            reason = "speed hold"
        else:
            self._speed_integral = clamp(
                self._speed_integral + speed_error * dt,
                -2.0,
                2.0,
            )
            effort = 0.35 * speed_error + 0.08 * self._speed_integral
            if effort >= 0.0:
                command = ControlCommand(
                    throttle=clamp(effort, 0.0, 0.20),
                    steer=steer,
                    brake=0.0,
                )
            else:
                command = ControlCommand(
                    throttle=0.0,
                    steer=steer,
                    brake=clamp(-effort, 0.0, 0.7),
                )
            reason = "tracking"

        done = (
            nearest >= len(self.route) - 4
            and math.hypot(
                x - self.route[-1][0],
                y - self.route[-1][1],
            )
            < 2.5
        )
        if done:
            command = ControlCommand.service_brake(steer=steer)

        return ControllerState(
            command=command,
            nearest_index=nearest,
            target_index=target,
            cross_track_error=cross_track,
            heading_error_degrees=heading_error_degrees,
            target_speed=target_speed,
            done=done,
            valid=True,
            reason="route complete" if done else reason,
        )

    def _find_nearest(self, x: float, y: float) -> int:
        if self._nearest_index is None:
            start = 0
            stop = len(self.route)
        else:
            start = max(0, self._nearest_index - 3)
            stop = min(len(self.route), self._nearest_index + 41)
        nearest = min(
            range(start, stop),
            key=lambda index: (self.route[index][0] - x) ** 2 + (self.route[index][1] - y) ** 2,
        )
        if self._nearest_index is not None:
            nearest = max(self._nearest_index, nearest)
        self._nearest_index = nearest
        return nearest

    def _route_curvature(self, index: int) -> float:
        before = self.route[index]
        middle_index = min(index + 3, len(self.route) - 1)
        after_index = min(index + 7, len(self.route) - 1)
        middle = self.route[middle_index]
        after = self.route[after_index]
        first = math.atan2(middle[1] - before[1], middle[0] - before[0])
        second = math.atan2(after[1] - middle[1], after[0] - middle[0])
        length = max(math.dist(before, after), 1e-6)
        return abs(normalize_angle_radians(second - first)) / length

    def _invalid(
        self,
        nearest: int,
        cross_track: float,
        reason: str,
    ) -> ControllerState:
        return ControllerState(
            command=ControlCommand.service_brake(steer=self._last_steer),
            nearest_index=nearest,
            target_index=nearest,
            cross_track_error=cross_track,
            heading_error_degrees=0.0,
            target_speed=0.0,
            done=False,
            valid=False,
            reason=reason,
        )
