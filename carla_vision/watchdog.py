from __future__ import annotations

import math
import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

from .bridge import CarlaRpc
from .controller import ControlCommand


def _is_valid_control(control: Any) -> bool:
    if not isinstance(control, list) or len(control) != 7:
        return False
    numeric = control[:3]
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric)


def _brake_and_park(rpc: CarlaRpc, vehicle_id: int) -> None:
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        rpc.apply_vehicle_control(vehicle_id, ControlCommand.service_brake().as_carla())
        try:
            if abs(rpc.telemetry(vehicle_id).speed) < 0.05:
                break
        except Exception:
            pass
        time.sleep(0.08)
    rpc.apply_vehicle_control(vehicle_id, ControlCommand.parked().as_carla())


def _actuator_process(
    connection: Connection,
    host: str,
    port: int,
    vehicle_id: int,
    heartbeat_timeout: float,
) -> None:
    rpc: CarlaRpc | None = None
    try:
        # CARLA can take longer than one second to acknowledge a control while
        # streaming a busy UE world.  A one-off slow response must not kill the
        # independent brake process; its local heartbeat still enforces the
        # control lease as soon as the server is responsive again.
        rpc = CarlaRpc(host, port, timeout=3.0)
        actor = rpc.actor(vehicle_id)
        if actor is None or not actor[2][1].startswith("vehicle."):
            raise RuntimeError(f"actor {vehicle_id} is not a vehicle")
        rpc.void_call("set_actor_autopilot", vehicle_id, False)
        _brake_and_park(rpc, vehicle_id)
        connection.send(("ready", None))

        armed = False
        last_command = time.monotonic()
        last_applied_control: list[Any] | None = None
        while True:
            latest_control: list[Any] | None = None
            should_stop = False
            if connection.poll(0.02):
                while True:
                    try:
                        message, payload = connection.recv()
                    except EOFError:
                        message, payload = "eof", None
                    if message == "control":
                        if not _is_valid_control(payload):
                            raise RuntimeError(f"invalid control payload {payload!r}")
                        latest_control = payload
                    elif message in {"stop", "eof"}:
                        should_stop = True
                    if should_stop or not connection.poll(0.0):
                        break
            if should_stop:
                break
            if latest_control is not None:
                # Consume every RPC response.  The lightweight bridge's handcrafted
                # async path cannot drain acknowledgements, so a long drive would
                # otherwise accumulate unread responses on this safety-critical
                # connection.
                if latest_control != last_applied_control:
                    rpc.apply_vehicle_control(vehicle_id, latest_control)
                    last_applied_control = latest_control
                armed = True
                last_command = time.monotonic()
            if armed and time.monotonic() - last_command > heartbeat_timeout:
                connection.send(("failsafe", "control heartbeat expired"))
                _brake_and_park(rpc, vehicle_id)
                return

        _brake_and_park(rpc, vehicle_id)
        connection.send(("stopped", None))
    except BaseException as exc:
        if rpc is not None:
            try:
                _brake_and_park(rpc, vehicle_id)
            except Exception:
                pass
        try:
            connection.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        if rpc is not None:
            rpc.close()
        connection.close()


class SafeActuator:
    """Separate process that defaults to braking when control heartbeats stop."""

    def __init__(
        self,
        host: str,
        port: int,
        vehicle_id: int,
        heartbeat_timeout: float = 0.25,
    ) -> None:
        self._stopped_cleanly = False
        self._stop_called = False
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_actuator_process,
            args=(child_connection, host, port, vehicle_id, heartbeat_timeout),
            name="carla-safe-actuator",
        )
        self._process.start()
        child_connection.close()
        if not self._connection.poll(8.0):
            self.stop()
            raise TimeoutError("safe actuator did not become ready")
        try:
            status, detail = self._connection.recv()
        except EOFError as exc:
            self.stop()
            raise RuntimeError("safe actuator exited during startup") from exc
        if status != "ready":
            self.stop()
            raise RuntimeError(f"safe actuator failed to start: {status}: {detail}")
        self._lock = threading.Lock()
        self._desired = ControlCommand.parked()
        self._desired_at = time.monotonic()
        self._error: str | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="carla-actuator-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def send(self, command: ControlCommand) -> None:
        if not self.alive:
            try:
                if self._connection.poll(0.0):
                    status, detail = self._connection.recv()
                    if status in {"failsafe", "error"}:
                        raise RuntimeError(f"safe actuator {status}: {detail}")
            except (BrokenPipeError, EOFError, OSError):
                pass
            raise RuntimeError("safe actuator is not running")
        with self._lock:
            if self._error is not None:
                raise RuntimeError(self._error)
            self._desired = command
            self._desired_at = time.monotonic()

    def stop(self) -> bool:
        if getattr(self, "_process", None) is None:
            return False
        if self._stop_called:
            return self._stopped_cleanly
        self._stop_called = True
        heartbeat_stop = getattr(self, "_heartbeat_stop", None)
        if heartbeat_stop is not None:
            heartbeat_stop.set()
            heartbeat_thread = getattr(self, "_heartbeat_thread", None)
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
        if self._process.is_alive():
            try:
                self._connection.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            try:
                if self._connection.poll(0.1):
                    status, detail = self._connection.recv()
                    if status == "stopped":
                        self._stopped_cleanly = True
                        break
                    if status in {"failsafe", "error"}:
                        self._error = f"safe actuator {status}: {detail}"
                        break
            except (BrokenPipeError, EOFError, OSError):
                break
            if not self._process.is_alive() and not self._connection.poll(0.0):
                break
        self._process.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        try:
            self._connection.close()
        except Exception:
            pass
        return self._stopped_cleanly

    def __enter__(self) -> "SafeActuator":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _heartbeat_loop(self) -> None:
        """Repeat fresh commands; substitute service brake when the owner stalls."""

        while not self._heartbeat_stop.is_set():
            with self._lock:
                age = time.monotonic() - self._desired_at
                if age <= 0.30:
                    command = self._desired
                else:
                    command = ControlCommand.service_brake(steer=self._desired.steer)
            try:
                self._connection.send(("control", command.as_carla()))
                if self._connection.poll(0.0):
                    status, detail = self._connection.recv()
                    if status in {"failsafe", "error"}:
                        with self._lock:
                            self._error = f"safe actuator {status}: {detail}"
                        return
            except (BrokenPipeError, EOFError, OSError) as exc:
                with self._lock:
                    self._error = f"safe actuator connection failed: {exc}"
                return
            self._heartbeat_stop.wait(0.08)
