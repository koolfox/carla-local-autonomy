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

from .contracts import Detection, PerceptionResult
from .risk import RiskAssessment


def detection_display_label(detection: Detection, *, show_rejection_status: bool = False) -> str:
    """Notebook per-box fields; display preferences never alter model scores."""
    fine_label = detection.attributes.get("fine_label")
    sign = detection.attributes.get("sign_classification")
    has_prediction = isinstance(sign, Mapping) and sign.get("label") is not None and sign.get("confidence") is not None
    if fine_label and {"coarse_confidence", "fine_confidence"} <= detection.attributes.keys():
        coarse = float(detection.attributes["coarse_confidence"])
        fine = float(detection.attributes["fine_confidence"])
        quality = detection.attributes.get("quality")
        quality_text = f"{float(quality):.2f}" if quality is not None else "n/a"
        # Same line grouping as draw_prediction_with_deit in the notebook.
        label = (f"F:{fine_label}\nPf:{fine:.2f} C:{detection.label}\nPc:{coarse:.2f}\n"
                 f"Q:{quality_text} S:{detection.confidence:.2f}") if has_prediction else (
                     f"F:{fine_label}\nPf:{fine:.2f}\nC:{detection.label}\nPc:{coarse:.2f}\n"
                     f"Q:{quality_text}\nS:{detection.confidence:.2f}")
    else:
        label = f"{detection.label} {detection.confidence:.0%}"
    if isinstance(sign, Mapping):
        if has_prediction:
            name = "unknown (unaccepted)" if show_rejection_status and not sign.get("accepted") else sign["label"]
            label += f"\nSign:{name}\nDeiT:{float(sign['confidence']):.2f}"
        elif show_rejection_status:
            label += "\nSign:unknown\nDeiT:unavailable"
    return label


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
        show_rejection_status: bool = False,
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
        self.show_rejection_status = show_rejection_status

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
        label_regions: list[tuple[int, int, int, int]] = []
        for index, detection in enumerate(result.detections):
            x1, y1, x2, y2 = _clipped_box(detection.xyxy, width=width, height=height)
            if x2 <= x1 or y2 <= y1:
                continue
            item_risk = risk_by_index.get(index)
            if item_risk is not None and item_risk.hazard:
                color = (0, 0, 255)
            elif item_risk is not None and item_risk.in_driving_corridor:
                color = (0, 200, 255)
            elif detection.attributes.get("fine_label"):
                color = (0, 255, 0)  # Notebook M9 boxes are lime, not class-colored labels.
            else:
                color = self._PALETTE[int(detection.class_id) % len(self._PALETTE)]
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                3 if detection.attributes.get("fine_label") else self.box_thickness,
                cv2.LINE_AA,
            )
            label = detection_display_label(detection, show_rejection_status=self.show_rejection_status)
            self._draw_detection_label(image, label, x1=x1, y1=y1, x2=x2, y2=y2,
                                       occupied=label_regions)

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
        x2: int,
        y2: int,
        occupied: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        """Black/white notebook captions anchored to each object's bounding box."""
        thickness = 1
        lines = text.splitlines()
        scale = self.font_scale
        sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                 for line in lines]
        text_width = max(size[0][0] for size in sizes)
        image_height, image_width = image.shape[:2]
        scale *= min(1.0, (image_width - 12) / max(1, text_width),
                     (image_height - 12) / max(1, len(lines) * (max(size[0][1] + size[1] for size in sizes) + 2)))
        if scale < .2:
            return
        sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness) for line in lines]
        baseline = max(size[1] for size in sizes)
        line_height = max(size[0][1] for size in sizes) + baseline + 2
        panel_width = max(size[0][0] for size in sizes) + 10
        panel_height = line_height * len(lines) + 10
        # Above, below, then alongside the object as in the notebook helper;
        # clamp the final fallback so edge detections keep readable labels.
        left = min(x1, max(0, image_width - panel_width))
        positions = [(left, y1 - panel_height - 5), (left, y2 + 5),
                     (x2 + 5, y1), (x1 - panel_width - 5, y1)]
        positions = [(x, y) for x, y in positions if x >= 0 and y >= 0 and
                     x + panel_width <= image_width and y + panel_height <= image_height]
        positions.append((left, min(y1, max(0, image_height - panel_height))))
        def overlap(position: tuple[int, int]) -> int:
            x, y = position
            return sum(max(0, min(x + panel_width, right) - max(x, left)) *
                       max(0, min(y + panel_height, bottom) - max(y, top))
                       for left, top, right, bottom in (occupied or []))
        left, top = min(positions, key=overlap)
        right, bottom = left + panel_width - 1, top + panel_height - 1
        if occupied is not None:
            occupied.append((left, top, right + 1, bottom + 1))
        radius = 3
        cv2.rectangle(image, (left + radius, top), (right - radius, bottom), (0, 0, 0), -1)
        cv2.rectangle(image, (left, top + radius), (right, bottom - radius), (0, 0, 0), -1)
        for x, y in ((left + radius, top + radius), (right - radius, top + radius),
                     (left + radius, bottom - radius), (right - radius, bottom - radius)):
            cv2.circle(image, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)
        for index, line in enumerate(lines):
            cv2.putText(
                image, line,
                (left + 5, top + 5 + line_height * (index + 1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
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
