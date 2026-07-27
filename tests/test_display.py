from __future__ import annotations

import threading
import unittest

import numpy as np

from carla_vision.contracts import Detection, PerceptionResult
from carla_vision.display import DisplayMode, LiveViewer, OverlayRenderer


def make_result(
    *,
    source_bgr: np.ndarray | None = None,
    detections: tuple[Detection, ...] = (),
    received: float = 10.0,
    completed: float = 10.1,
) -> PerceptionResult:
    return PerceptionResult(
        sequence=7,
        carla_frame=1042,
        source_timestamp=4.25,
        source_received_monotonic=received,
        completed_monotonic=completed,
        detections=detections,
        source_bgr=(np.zeros((160, 240, 3), dtype=np.uint8) if source_bgr is None else source_bgr),
    )


class FakeCv2Window:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4

    def __init__(self, keys: list[int] | None = None) -> None:
        self.keys = list(keys or [])
        self.visible = 1.0
        self.frames: list[np.ndarray] = []
        self.named: list[tuple[str, int]] = []
        self.destroyed: list[str] = []

    def namedWindow(self, name: str, flags: int) -> None:
        self.named.append((name, flags))
        self.visible = 1.0

    def imshow(self, _name: str, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def waitKey(self, _delay_ms: int) -> int:
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, _name: str, _property_id: int) -> float:
        return self.visible

    def destroyWindow(self, name: str) -> None:
        self.destroyed.append(name)
        self.visible = 0.0


class OverlayRendererTests(unittest.TestCase):
    def test_render_draws_detection_without_mutating_source(self) -> None:
        source = np.zeros((160, 240, 3), dtype=np.uint8)
        original = source.copy()
        detection = Detection(
            class_id=3,
            label="car",
            confidence=0.876,
            xyxy=(90.0, 85.0, 180.0, 145.0),
        )
        result = make_result(source_bgr=source, detections=(detection,))

        rendered = OverlayRenderer().render(result, now_monotonic=10.2)

        self.assertTrue(np.array_equal(source, original))
        self.assertFalse(np.shares_memory(rendered, source))
        self.assertGreater(np.count_nonzero(rendered), 0)
        self.assertGreater(np.count_nonzero(rendered[82:149, 87:184]), 0)

    def test_stale_banner_is_deterministic_and_overridable(self) -> None:
        renderer = OverlayRenderer(stale_after_seconds=0.5)
        result = make_result()

        fresh = renderer.render(result, now_monotonic=10.2)
        stale = renderer.render(result, now_monotonic=10.8)
        forced_fresh = renderer.render(result, now_monotonic=10.8, stale=False)

        stale_red = np.count_nonzero(np.all(stale == (0, 0, 220), axis=2))
        fresh_red = np.count_nonzero(np.all(fresh == (0, 0, 220), axis=2))
        forced_fresh_red = np.count_nonzero(np.all(forced_fresh == (0, 0, 220), axis=2))
        self.assertTrue(renderer.is_stale(result, now_monotonic=10.8))
        self.assertFalse(renderer.is_stale(result, now_monotonic=10.2))
        self.assertGreater(stale_red, fresh_red)
        self.assertEqual(forced_fresh_red, fresh_red)

    def test_invalid_live_image_is_rejected_in_split_mode(self) -> None:
        viewer = LiveViewer(mode="split", cv2_backend=FakeCv2Window())
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            viewer.compose_frame(
                make_result(),
                live_bgr=np.zeros((40, 60), dtype=np.uint8),
            )


class LiveViewerTests(unittest.TestCase):
    def test_split_keeps_live_image_raw_and_boxes_only_result_source(self) -> None:
        source = np.zeros((160, 240, 3), dtype=np.uint8)
        live = np.full_like(source, 73)
        detection = Detection(
            class_id=0,
            label="person",
            confidence=0.93,
            xyxy=(100.0, 90.0, 150.0, 150.0),
        )
        result = make_result(source_bgr=source, detections=(detection,))
        backend = FakeCv2Window()
        viewer = LiveViewer(mode="split", cv2_backend=backend)

        keep_running = viewer.show(result, live_bgr=live, now_monotonic=10.2)

        self.assertTrue(keep_running)
        displayed = backend.frames[-1]
        self.assertEqual(displayed.shape, (160, 482, 3))
        self.assertTrue(np.array_equal(displayed[:, :240], live))
        self.assertTrue(np.all(displayed[:, 240:242] == 45))
        self.assertGreater(np.count_nonzero(displayed[:, 242:]), 0)
        self.assertTrue(np.array_equal(source, np.zeros_like(source)))

    def test_overlay_uses_result_source_even_when_newer_live_frame_is_passed(self) -> None:
        source = np.zeros((160, 240, 3), dtype=np.uint8)
        live = np.full_like(source, 255)
        backend = FakeCv2Window()
        viewer = LiveViewer(mode=DisplayMode.OVERLAY, cv2_backend=backend)

        viewer.show(make_result(source_bgr=source), live_bgr=live, now_monotonic=10.2)

        displayed = backend.frames[-1]
        self.assertEqual(displayed.shape, source.shape)
        self.assertFalse(np.array_equal(displayed, live))
        self.assertTrue(np.all(displayed[-10:, -10:] == 0))

    def test_q_and_escape_request_shutdown_without_real_gui(self) -> None:
        result = make_result()
        for key in (ord("q"), 27):
            with self.subTest(key=key):
                backend = FakeCv2Window(keys=[key])
                viewer = LiveViewer(cv2_backend=backend)
                self.assertFalse(viewer.show(result))
                self.assertFalse(viewer.is_open)
                self.assertEqual(backend.destroyed, ["CARLA Vision"])

    def test_window_close_request_is_detected_before_recreating_window(self) -> None:
        backend = FakeCv2Window()
        viewer = LiveViewer(cv2_backend=backend)
        result = make_result()
        self.assertTrue(viewer.show(result))
        frame_count = len(backend.frames)

        backend.visible = 0.0
        self.assertFalse(viewer.show(result))

        self.assertEqual(len(backend.frames), frame_count)
        self.assertFalse(viewer.is_open)

    def test_keyboard_switches_overlay_and_split_modes(self) -> None:
        backend = FakeCv2Window(keys=[ord("s"), ord("o")])
        viewer = LiveViewer(mode="overlay", cv2_backend=backend)
        result = make_result()

        self.assertTrue(viewer.show(result))
        self.assertEqual(viewer.mode, DisplayMode.SPLIT)
        self.assertTrue(viewer.show(result))
        self.assertEqual(viewer.mode, DisplayMode.OVERLAY)
        self.assertEqual(backend.frames[0].shape[1], 240)
        self.assertEqual(backend.frames[1].shape[1], 482)

    def test_gui_calls_from_worker_thread_are_rejected(self) -> None:
        backend = FakeCv2Window()
        viewer = LiveViewer(cv2_backend=backend)
        failures: list[RuntimeError] = []

        def call_show() -> None:
            try:
                viewer.show(make_result())
            except RuntimeError as exc:
                failures.append(exc)

        thread = threading.Thread(target=call_show)
        thread.start()
        thread.join()

        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.assertIn("main thread", str(failures[0]))
        self.assertEqual(backend.frames, [])


if __name__ == "__main__":
    unittest.main()
