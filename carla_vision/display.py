"""Live, frame-correct visualization for CARLA vision perception.

The renderer deliberately annotates ``PerceptionResult.source_bgr`` rather than
an arbitrary latest camera frame.  This keeps every bounding box aligned with
the exact image used for inference, even when the detector is slower than the
camera.  ``LiveViewer`` owns no worker thread: callers must invoke ``show`` from
the process main thread, as required by OpenCV's native GUI backends.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from enum import Enum
from typing import Any, Self

import cv2
import numpy as np

from .contracts import PerceptionResult
from .risk import RiskAssessment


class DisplayMode(str, Enum):
    """Frames that the live window can present."""

    OVERLAY = "overlay"
    SPLIT = "split"


class OverlayRenderer:
    """Pure renderer for detector boxes, status HUD, and stale-frame warnings."""

    _PALETTE: tuple[tuple[int, int, int], ...] = (
        (255, 178, 50),
        (70, 220, 70),
        (255, 120, 200),
        (60, 210, 255),
        (210, 120, 255),
        (255, 210, 70),
        (80, 150, 255),
        (180, 230, 100),
    )
    _HUD_BACKGROUND = (20, 20, 20)
    _HUD_TEXT = (245, 245, 245)
    _STALE_BACKGROUND = (0, 0, 220)
    _STALE_TEXT = (255, 255, 255)

    def __init__(
        self,
        *,
        stale_after_seconds: float = 0.5,
        font_scale: float = 0.5,
        box_thickness: int = 2,
    ) -> None:
        if stale_after_seconds <= 0.0:
            raise ValueError("stale_after_seconds must be positive")
        if font_scale <= 0.0:
            raise ValueError("font_scale must be positive")
        if box_thickness <= 0:
            raise ValueError("box_thickness must be positive")
        self.stale_after_seconds = float(stale_after_seconds)
        self.font_scale = float(font_scale)
        self.box_thickness = int(box_thickness)

    def render(
        self,
        result: PerceptionResult,
        *,
        now_monotonic: float | None = None,
        hud: Mapping[str, object] | None = None,
        stale: bool | None = None,
        risk: RiskAssessment | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of the exact image used by ``result``.

        The function never mutates ``result.source_bgr``.  If
        ``now_monotonic`` is omitted, inference completion time is used, making
        repeated calls with the same arguments deterministic.  The live viewer
        supplies its current monotonic time so camera age continues to advance.
        """

        image = _copy_bgr(result.source_bgr, name="result.source_bgr")
        now = result.completed_monotonic if now_monotonic is None else float(now_monotonic)
        source_age = max(0.0, now - result.source_received_monotonic)
        inference_latency = max(
            0.0,
            result.completed_monotonic - result.source_received_monotonic,
        )
        is_stale = source_age > self.stale_after_seconds if stale is None else bool(stale)

        height, width = image.shape[:2]
        risk_by_index = (
            {item.detection_index: item for item in risk.items} if risk is not None else {}
        )
        for index, detection in enumerate(result.detections):
            x1, y1, x2, y2 = _clipped_box(detection.xyxy, width=width, height=height)
            if x2 <= x1 or y2 <= y1:
                continue
            item_risk = risk_by_index.get(index)
            if item_risk is not None and item_risk.hazard:
                color = (0, 0, 255)
            elif item_risk is not None and item_risk.in_driving_corridor:
                color = (0, 200, 255)
            else:
                color = self._PALETTE[int(detection.class_id) % len(self._PALETTE)]
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                self.box_thickness,
                cv2.LINE_AA,
            )
            confidence = min(1.0, max(0.0, float(detection.confidence)))
            fine_label = detection.attributes.get("fine_label")
            hierarchy = f"{detection.label} -> {fine_label}" if fine_label else detection.label
            label = f"{hierarchy} {confidence:.0%}"
            self._draw_detection_label(image, label, x1=x1, y1=y1, color=color)

        if risk is not None:
            left, top, right, bottom = risk.corridor_xyxy
            corridor_color = (0, 0, 255) if risk.hazard else (0, 200, 255)
            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                corridor_color,
                1,
                cv2.LINE_AA,
            )

        hud_lines = [
            f"FRAME {result.carla_frame}  SEQ {result.sequence}",
            f"DETECTIONS {len(result.detections)}",
            f"LATENCY {inference_latency * 1_000.0:.0f} ms  AGE {source_age * 1_000.0:.0f} ms",
        ]
        if hud:
            hud_lines.extend(
                f"{str(key).upper()}: {_format_hud_value(value)}" for key, value in hud.items()
            )
        self._draw_hud(image, hud_lines)

        if is_stale:
            self._draw_stale_warning(image, source_age)
        return image

    def is_stale(self, result: PerceptionResult, *, now_monotonic: float) -> bool:
        """Return the renderer's deterministic stale decision for ``result``."""

        age = max(0.0, float(now_monotonic) - result.source_received_monotonic)
        return age > self.stale_after_seconds

    def _draw_detection_label(
        self,
        image: np.ndarray,
        text: str,
        *,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        thickness = max(1, self.box_thickness - 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            thickness,
        )
        image_height, image_width = image.shape[:2]
        text_x = min(max(0, x1), max(0, image_width - text_width - 4))
        label_top = max(0, y1 - text_height - baseline - 6)
        label_bottom = min(image_height - 1, label_top + text_height + baseline + 6)
        label_right = min(image_width - 1, text_x + text_width + 4)
        cv2.rectangle(
            image,
            (text_x, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (text_x + 2, min(image_height - 2, label_bottom - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_hud(self, image: np.ndarray, lines: list[str]) -> None:
        if not lines:
            return
        height, width = image.shape[:2]
        line_height = max(16, round(26 * self.font_scale))
        panel_height = min(height, 10 + line_height * len(lines))
        panel_width = min(width, max(270, round(width * 0.38)))

        translucent = image.copy()
        cv2.rectangle(
            translucent,
            (0, 0),
            (max(0, panel_width - 1), max(0, panel_height - 1)),
            self._HUD_BACKGROUND,
            -1,
        )
        cv2.addWeighted(translucent, 0.72, image, 0.28, 0.0, image)
        for index, line in enumerate(lines):
            baseline_y = 7 + line_height * (index + 1)
            if baseline_y >= height:
                break
            cv2.putText(
                image,
                line,
                (8, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self._HUD_TEXT,
                1,
                cv2.LINE_AA,
            )

    def _draw_stale_warning(self, image: np.ndarray, source_age: float) -> None:
        height, width = image.shape[:2]
        warning = f"STALE FRAME {source_age * 1_000.0:.0f} ms"
        (text_width, text_height), baseline = cv2.getTextSize(
            warning,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            2,
        )
        box_width = min(width, text_width + 20)
        box_height = min(height, text_height + baseline + 16)
        left = max(0, width - box_width)
        cv2.rectangle(
            image,
            (left, 0),
            (width - 1, max(0, box_height - 1)),
            self._STALE_BACKGROUND,
            -1,
        )
        cv2.putText(
            image,
            warning,
            (min(width - 1, left + 10), min(height - 2, text_height + 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self._STALE_TEXT,
            2,
            cv2.LINE_AA,
        )


class LiveViewer:
    """OpenCV live window with frame-correct overlay and split modes.

    ``show`` returns ``False`` after Q, Escape, or a native window-close event,
    allowing the caller to take the same fail-safe shutdown path in all cases.
    The class starts no worker thread and rejects GUI calls outside the process
    main thread.
    """

    def __init__(
        self,
        renderer: OverlayRenderer | None = None,
        *,
        window_name: str = "CARLA Vision",
        mode: DisplayMode | str = DisplayMode.OVERLAY,
        delay_ms: int = 1,
        cv2_backend: Any | None = None,
    ) -> None:
        if delay_ms < 1:
            raise ValueError("delay_ms must be at least 1")
        self.renderer = renderer or OverlayRenderer()
        self.window_name = str(window_name)
        self.delay_ms = int(delay_ms)
        self._cv2 = cv2 if cv2_backend is None else cv2_backend
        self._mode = DisplayMode(mode)
        self._opened = False
        self._closed = False
        self.last_key = -1

    @property
    def mode(self) -> DisplayMode:
        return self._mode

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    def set_mode(self, mode: DisplayMode | str) -> None:
        self._mode = DisplayMode(mode)

    def compose_frame(
        self,
        result: PerceptionResult,
        *,
        live_bgr: np.ndarray | None = None,
        now_monotonic: float | None = None,
        hud: Mapping[str, object] | None = None,
        stale: bool | None = None,
        risk: RiskAssessment | None = None,
    ) -> np.ndarray:
        """Build the displayed frame without opening or touching a GUI window."""

        annotated = self.renderer.render(
            result,
            now_monotonic=now_monotonic,
            hud=hud,
            stale=stale,
            risk=risk,
        )
        if self._mode is DisplayMode.OVERLAY:
            return annotated

        live = result.source_bgr if live_bgr is None else live_bgr
        raw = _copy_bgr(live, name="live_bgr")
        if raw.shape[0] != annotated.shape[0]:
            target_height = annotated.shape[0]
            target_width = max(
                1,
                round(raw.shape[1] * target_height / raw.shape[0]),
            )
            raw = cv2.resize(raw, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        separator = np.full((annotated.shape[0], 2, 3), 45, dtype=np.uint8)
        return np.concatenate((raw, separator, annotated), axis=1)

    def show(
        self,
        result: PerceptionResult,
        *,
        live_bgr: np.ndarray | None = None,
        now_monotonic: float | None = None,
        hud: Mapping[str, object] | None = None,
        stale: bool | None = None,
        risk: RiskAssessment | None = None,
    ) -> bool:
        """Render one GUI iteration; return whether the caller should continue."""

        self._require_main_thread()
        if self._closed:
            return False
        if self._opened and not self._window_is_visible():
            self._close_window()
            return False

        frame = self.compose_frame(
            result,
            live_bgr=live_bgr,
            now_monotonic=now_monotonic,
            hud=hud,
            stale=stale,
            risk=risk,
        )
        self._ensure_window()
        self._cv2.imshow(self.window_name, frame)
        raw_key = int(self._cv2.waitKey(self.delay_ms))
        self.last_key = -1 if raw_key < 0 else raw_key & 0xFF

        if self.last_key in (27, ord("q"), ord("Q")):
            self._close_window()
            return False
        if self.last_key in (ord("s"), ord("S")):
            self._mode = DisplayMode.SPLIT
        elif self.last_key in (ord("o"), ord("O")):
            self._mode = DisplayMode.OVERLAY
        elif self.last_key in (ord("m"), ord("M")):
            self._mode = (
                DisplayMode.SPLIT if self._mode is DisplayMode.OVERLAY else DisplayMode.OVERLAY
            )

        if not self._window_is_visible():
            self._close_window()
            return False
        return True

    def close(self) -> None:
        """Close the native window; safe to call repeatedly from the main thread."""

        self._require_main_thread()
        self._close_window()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_window(self) -> None:
        if self._opened:
            return
        flags = getattr(self._cv2, "WINDOW_NORMAL", 0)
        self._cv2.namedWindow(self.window_name, flags)
        self._opened = True

    def _window_is_visible(self) -> bool:
        if not self._opened:
            return True
        try:
            property_id = getattr(self._cv2, "WND_PROP_VISIBLE", 4)
            return float(self._cv2.getWindowProperty(self.window_name, property_id)) >= 1.0
        except (AttributeError, TypeError):
            # Minimal/non-native test backends may not expose window properties.
            return True
        except cv2.error:
            return False

    def _close_window(self) -> None:
        if self._opened:
            try:
                self._cv2.destroyWindow(self.window_name)
            except (AttributeError, cv2.error):
                pass
        self._opened = False
        self._closed = True

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("LiveViewer GUI methods must run on the main thread")


def _copy_bgr(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 BGR image")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 pixels")
    return np.ascontiguousarray(array).copy()


def _clipped_box(
    xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(xyxy) != 4:
        raise ValueError("detection xyxy must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    if not np.isfinite((x1, y1, x2, y2)).all():
        raise ValueError("detection xyxy coordinates must be finite")
    left, right = sorted((round(x1), round(x2)))
    top, bottom = sorted((round(y1), round(y2)))
    return (
        min(width - 1, max(0, left)),
        min(height - 1, max(0, top)),
        min(width - 1, max(0, right)),
        min(height - 1, max(0, bottom)),
    )


def _format_hud_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
