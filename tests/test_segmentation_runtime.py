from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Callable

import numpy as np

from carla_vision.contracts import PerceptionResult
from carla_vision.operator.drive import _ExactFrameAnalysisBuffer
from carla_vision.segmentation import (
    ROAD_CLASS_COLORS_BGR,
    AsyncSegmentationRuntime,
    RoadClass,
    SegmentationFrameInput,
    SegmentationFrameResult,
    SegmentationMetadata,
    SegmentationResult,
    SegmentationWorker,
    SegmentationWorkerStats,
    render_segmentation_overlay,
)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    message: str = "condition was not satisfied",
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise AssertionError(message)
        threading.Event().wait(min(0.005, remaining))


def _image(value: int = 0, *, height: int = 4, width: int = 6) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _frame(
    sequence: int,
    *,
    image: np.ndarray | None = None,
    perception: PerceptionResult | None = None,
) -> SegmentationFrameInput:
    return SegmentationFrameInput(
        sequence=sequence,
        frame=1000 + sequence,
        timestamp=float(sequence),
        received_monotonic=0.0,
        image_bgr=_image(sequence) if image is None else image,
        transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        fov=90.0,
        perception=perception,
    )


def _perception(
    sequence: int,
    *,
    carla_frame: int | None = None,
    image: np.ndarray | None = None,
) -> PerceptionResult:
    return PerceptionResult(
        sequence=sequence,
        carla_frame=1000 + sequence if carla_frame is None else carla_frame,
        source_timestamp=float(sequence),
        source_received_monotonic=1.0,
        inference_started_monotonic=1.1,
        completed_monotonic=1.2,
        detections=(),
        source_bgr=_image(sequence) if image is None else image,
        detector_name="detector-fake",
        source_transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        source_fov=90.0,
    )


def _segmentation_result(
    sequence: int,
    *,
    carla_frame: int | None = None,
    image: np.ndarray | None = None,
    perception: PerceptionResult | None = None,
) -> SegmentationFrameResult:
    source = _image(sequence) if image is None else image
    segmentation = SegmentationResult(
        class_ids=np.full(source.shape[:2], RoadClass.ROAD, dtype=np.uint8),
        confidence=np.ones(source.shape[:2], dtype=np.float32),
        model_name="segmenter-fake",
    )
    return SegmentationFrameResult(
        sequence=sequence,
        carla_frame=1000 + sequence if carla_frame is None else carla_frame,
        source_timestamp=float(sequence),
        source_received_monotonic=1.0,
        inference_started_monotonic=1.1,
        completed_monotonic=1.2,
        source_bgr=source,
        segmentation=segmentation,
        segmenter_name="segmenter-fake",
        source_transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        source_fov=90.0,
        perception=perception,
    )


class _FakeSegmenter:
    name = "segmenter-fake"
    metadata = SegmentationMetadata(
        name=name,
        backend="test",
        checkpoint="test/checkpoint",
        device="cpu",
        source_labels={0: "road"},
        source_to_canonical={0: "road"},
    )

    def __init__(self) -> None:
        self.closed_count = 0
        self.inferred_values: list[int] = []

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        self.inferred_values.append(int(image_bgr[0, 0, 0]))
        return SegmentationResult(
            class_ids=np.full(image_bgr.shape[:2], RoadClass.ROAD, dtype=np.uint8),
            confidence=np.ones(image_bgr.shape[:2], dtype=np.float32),
            model_name=self.name,
        )

    def close(self) -> None:
        self.closed_count += 1


class _BlockingSegmenter(_FakeSegmenter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release blocking segmenter")
        return super().infer(image_bgr)


class _FailingSegmenter(_FakeSegmenter):
    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        raise ValueError(f"cannot segment pixel {int(image_bgr[0, 0, 0])}")


class SegmentationWorkerTests(unittest.TestCase):
    def test_newest_pending_frame_replaces_superseded_frame(self) -> None:
        segmenter = _BlockingSegmenter()
        worker = SegmentationWorker(segmenter)
        try:
            worker.submit(_frame(1))
            self.assertTrue(segmenter.started.wait(timeout=1.0))
            worker.submit(_frame(2))
            worker.submit(_frame(3))
            segmenter.release.set()

            latest = worker.wait_for_result(after_sequence=2, timeout=2.0)

            self.assertEqual(latest.sequence, 3)
            self.assertEqual(latest.carla_frame, 1003)
            self.assertEqual(segmenter.inferred_values, [1, 3])
            self.assertEqual(latest.source_transform, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
            self.assertEqual(latest.source_fov, 90.0)
            self.assertEqual(
                worker.stats(),
                SegmentationWorkerStats(
                    submitted=3,
                    processed=2,
                    dropped_before_inference=1,
                ),
            )
        finally:
            segmenter.release.set()
            worker.close()
        self.assertEqual(segmenter.closed_count, 1)

    def test_worker_failure_is_reported_and_closes_model(self) -> None:
        segmenter = _FailingSegmenter()
        worker = SegmentationWorker(segmenter)
        worker.submit(_frame(7))

        with self.assertRaisesRegex(
            RuntimeError,
            "segmentation worker failed: cannot segment pixel 7",
        ):
            worker.wait_for_result(timeout=2.0)

        _wait_until(
            lambda: segmenter.closed_count == 1,
            message="failed worker did not close its segmenter",
        )
        self.assertEqual(worker.stats().processed, 0)
        with self.assertRaisesRegex(RuntimeError, "segmentation worker failed"):
            worker.latest()
        with self.assertRaisesRegex(RuntimeError, "segmentation worker failed"):
            worker.submit(_frame(8))
        with self.assertRaisesRegex(RuntimeError, "segmentation worker failed"):
            worker.close()
        self.assertEqual(segmenter.closed_count, 1)

    def test_close_is_idempotent_and_rejects_future_frames(self) -> None:
        segmenter = _FakeSegmenter()
        worker = SegmentationWorker(segmenter)

        worker.close()
        worker.close()

        self.assertEqual(segmenter.closed_count, 1)
        with self.assertRaisesRegex(RuntimeError, "worker is closed"):
            worker.submit(_frame(1))

    def test_close_reconciles_active_completion_and_pending_drop(self) -> None:
        segmenter = _BlockingSegmenter()
        worker = SegmentationWorker(segmenter)
        worker.submit(_frame(1))
        self.assertTrue(segmenter.started.wait(timeout=1.0))
        worker.submit(_frame(2))
        close_error: list[BaseException] = []

        def close_worker() -> None:
            try:
                worker.close()
            except BaseException as error:  # pragma: no cover - assertion reports it below
                close_error.append(error)

        close_thread = threading.Thread(target=close_worker, daemon=True)
        close_thread.start()
        _wait_until(
            lambda: worker._closed,
            message="worker close did not begin",
        )
        segmenter.release.set()
        close_thread.join(timeout=2.0)

        self.assertFalse(close_thread.is_alive())
        self.assertEqual(close_error, [])
        self.assertEqual(
            worker.stats(),
            SegmentationWorkerStats(
                submitted=2,
                processed=1,
                dropped_before_inference=1,
                failed_during_inference=0,
            ),
        )


class AsyncSegmentationRuntimeTests(unittest.TestCase):
    def test_initializing_runtime_becomes_ready_and_processes_frames(self) -> None:
        loader_started = threading.Event()
        loader_release = threading.Event()
        segmenter = _FakeSegmenter()

        def loader() -> _FakeSegmenter:
            loader_started.set()
            if not loader_release.wait(timeout=2.0):
                raise TimeoutError("test did not release runtime loader")
            return segmenter

        runtime = AsyncSegmentationRuntime(loader)
        self.assertTrue(loader_started.wait(timeout=1.0))
        self.assertEqual(runtime.snapshot()["state"], "initializing")
        self.assertFalse(runtime.ready())
        self.assertFalse(runtime.submit(_frame(1)))
        self.assertIsNone(runtime.latest())
        self.assertIsNone(runtime.stats())

        loader_release.set()
        _wait_until(runtime.ready, message="runtime did not become ready")
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["state"], "ready")
        self.assertEqual(snapshot["name"], segmenter.name)
        self.assertEqual(snapshot["metadata"], segmenter.metadata.as_dict())
        self.assertTrue(runtime.submit(_frame(4)))
        _wait_until(
            lambda: runtime.latest() is not None,
            message="ready runtime did not publish a result",
        )
        self.assertEqual(runtime.latest().sequence, 4)  # type: ignore[union-attr]
        self.assertEqual(runtime.stats().processed, 1)  # type: ignore[union-attr]

        runtime.close()
        runtime.close()
        self.assertEqual(runtime.snapshot()["state"], "closed")
        self.assertFalse(runtime.ready())
        self.assertIsNone(runtime.latest())
        self.assertEqual(segmenter.closed_count, 1)
        with self.assertRaisesRegex(RuntimeError, "runtime is closed"):
            runtime.submit(_frame(5))

    def test_failed_loader_surfaces_a_stable_failure_then_closes(self) -> None:
        def loader() -> _FakeSegmenter:
            raise LookupError("checkpoint is unavailable")

        runtime = AsyncSegmentationRuntime(loader)
        _wait_until(
            lambda: runtime.snapshot()["state"] == "failed",
            message="loader failure was not published",
        )

        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["error"], "LookupError: checkpoint is unavailable")
        self.assertFalse(runtime.ready())
        self.assertIsNone(runtime.stats())
        with self.assertRaisesRegex(RuntimeError, "initialization failed.*checkpoint"):
            runtime.submit(_frame(1))
        with self.assertRaisesRegex(RuntimeError, "initialization failed.*checkpoint"):
            runtime.latest()

        runtime.close()
        self.assertEqual(runtime.snapshot()["state"], "closed")
        self.assertIsNone(runtime.latest())

    def test_inference_failure_is_stateful_and_survives_close_with_final_stats(self) -> None:
        runtime = AsyncSegmentationRuntime(_FailingSegmenter)
        _wait_until(runtime.ready, message="runtime did not become ready")
        self.assertTrue(runtime.submit(_frame(7)))
        _wait_until(
            lambda: runtime.snapshot()["state"] == "failed",
            message="worker failure did not reach runtime state",
        )

        failed = runtime.snapshot()
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["failure_phase"], "inference")
        self.assertIn("cannot segment pixel 7", failed["error"])
        with self.assertRaisesRegex(RuntimeError, "inference failed.*cannot segment pixel 7"):
            runtime.latest()
        with self.assertRaisesRegex(RuntimeError, "segmentation worker failed"):
            runtime.close()

        closed = runtime.snapshot()
        self.assertEqual(closed["state"], "closed")
        self.assertEqual(closed["failure_phase"], "inference")
        self.assertIn("cannot segment pixel 7", closed["error"])
        self.assertEqual(
            runtime.stats(),
            SegmentationWorkerStats(
                submitted=1,
                processed=0,
                dropped_before_inference=0,
                failed_during_inference=1,
            ),
        )

    def test_close_requested_during_load_closes_model_without_starting_worker(self) -> None:
        loader_started = threading.Event()
        loader_release = threading.Event()
        segmenter = _FakeSegmenter()

        def loader() -> _FakeSegmenter:
            loader_started.set()
            if not loader_release.wait(timeout=2.0):
                raise TimeoutError("test did not release runtime loader")
            return segmenter

        runtime = AsyncSegmentationRuntime(loader)
        self.assertTrue(loader_started.wait(timeout=1.0))
        close_completed = threading.Event()

        def close_runtime() -> None:
            runtime.close()
            close_completed.set()

        close_thread = threading.Thread(target=close_runtime, daemon=True)
        close_thread.start()
        _wait_until(
            lambda: runtime._close_requested,
            message="close request did not reach the initializing runtime",
        )
        loader_release.set()
        self.assertTrue(close_completed.wait(timeout=2.0))
        close_thread.join(timeout=1.0)

        self.assertEqual(runtime.snapshot()["state"], "closed")
        self.assertFalse(runtime.ready())
        self.assertIsNone(runtime.stats())
        self.assertEqual(segmenter.closed_count, 1)


class SegmentationOverlayTests(unittest.TestCase):
    def test_overlay_has_exact_shape_colors_and_does_not_mutate_inputs(self) -> None:
        image = np.array(
            [
                [
                    [10, 20, 30],
                    [10, 20, 30],
                    [10, 20, 30],
                    [10, 20, 30],
                    [10, 20, 30],
                ]
            ],
            dtype=np.uint8,
        )
        classes = np.array(
            [[RoadClass.OTHER, RoadClass.ROAD, RoadClass.ROAD_LINE, RoadClass.SIDEWALK,
              RoadClass.NON_DRIVABLE_GROUND]],
            dtype=np.uint8,
        )
        confidence = np.array([[1.0, 1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        result = SegmentationResult(classes, confidence, "segmenter-fake")
        original_image = image.copy()
        original_classes = result.class_ids.copy()
        original_confidence = result.confidence.copy()

        overlay = render_segmentation_overlay(image, result, draw_legend=False)

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertFalse(np.shares_memory(overlay, image))
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(result.class_ids, original_classes)
        np.testing.assert_array_equal(result.confidence, original_confidence)
        np.testing.assert_array_equal(overlay[0, 0], image[0, 0])
        for index, road_class in enumerate(
            (
                RoadClass.ROAD,
                RoadClass.ROAD_LINE,
                RoadClass.SIDEWALK,
                RoadClass.NON_DRIVABLE_GROUND,
            ),
            start=1,
        ):
            base_alpha = {
                RoadClass.ROAD: 0.34,
                RoadClass.ROAD_LINE: 0.82,
                RoadClass.SIDEWALK: 0.40,
                RoadClass.NON_DRIVABLE_GROUND: 0.36,
            }[road_class]
            alpha = base_alpha * (0.55 + 0.45 * float(confidence[0, index]))
            expected = np.clip(
                image[0, index].astype(np.float32) * (1.0 - alpha)
                + np.asarray(ROAD_CLASS_COLORS_BGR[road_class], dtype=np.float32) * alpha,
                0.0,
                255.0,
            ).astype(np.uint8)
            np.testing.assert_array_equal(overlay[0, index], expected)

    def test_legend_is_drawn_only_when_a_supported_class_is_present(self) -> None:
        image = np.full((120, 260, 3), 5, dtype=np.uint8)
        classes = np.zeros((120, 260), dtype=np.uint8)
        classes[60, 200] = RoadClass.ROAD
        result = SegmentationResult(
            classes,
            np.ones((120, 260), dtype=np.float32),
            "segmenter-fake",
        )

        without_legend = render_segmentation_overlay(image, result, draw_legend=False)
        with_legend = render_segmentation_overlay(image, result, draw_legend=True)

        self.assertGreater(np.count_nonzero(with_legend != without_legend), 0)
        np.testing.assert_array_equal(
            with_legend[100, 10],
            np.asarray(ROAD_CLASS_COLORS_BGR[RoadClass.ROAD], dtype=np.uint8),
        )
        np.testing.assert_array_equal(without_legend[100, 10], image[100, 10])

        other_only = SegmentationResult(
            np.zeros((120, 260), dtype=np.uint8),
            np.ones((120, 260), dtype=np.float32),
            "segmenter-fake",
        )
        np.testing.assert_array_equal(
            render_segmentation_overlay(image, other_only, draw_legend=True),
            image,
        )

    def test_overlay_rejects_a_mask_from_a_different_frame_shape(self) -> None:
        result = SegmentationResult(
            np.zeros((2, 3), dtype=np.uint8),
            np.ones((2, 3), dtype=np.float32),
            "segmenter-fake",
        )
        with self.assertRaisesRegex(ValueError, "exact-frame mask"):
            render_segmentation_overlay(_image(height=3, width=3), result)


class SegmentationExactFrameContractTests(unittest.TestCase):
    def test_frame_input_from_perception_carries_the_exact_detector_source(self) -> None:
        perception = _perception(9)

        frame = SegmentationFrameInput.from_perception(perception)

        self.assertEqual(frame.sequence, perception.sequence)
        self.assertEqual(frame.frame, perception.carla_frame)
        self.assertIs(frame.image_bgr, perception.source_bgr)
        self.assertIs(frame.bgr(), perception.source_bgr)
        self.assertIs(frame.perception, perception)
        self.assertEqual(frame.transform, perception.source_transform)
        self.assertEqual(frame.fov, perception.source_fov)

    def test_carried_perception_must_match_sequence_and_carla_frame(self) -> None:
        wrong_sequence = _perception(2, carla_frame=1001)
        with self.assertRaisesRegex(ValueError, "carried detector result must match"):
            _segmentation_result(1, perception=wrong_sequence)

        wrong_carla_frame = _perception(1, carla_frame=9999)
        with self.assertRaisesRegex(ValueError, "carried detector result must match"):
            _segmentation_result(1, perception=wrong_carla_frame)

        matching = _perception(1)
        result = _segmentation_result(1, perception=matching)
        self.assertIs(result.perception, matching)

    def test_worker_preserves_carried_exact_frame_perception(self) -> None:
        segmenter = _FakeSegmenter()
        worker = SegmentationWorker(segmenter)
        perception = _perception(5)
        try:
            worker.submit(SegmentationFrameInput.from_perception(perception))
            result = worker.wait_for_result(timeout=2.0)
            self.assertIs(result.perception, perception)
            self.assertIs(result.source_bgr, perception.source_bgr)
            self.assertEqual(result.sequence, perception.sequence)
            self.assertEqual(result.carla_frame, perception.carla_frame)
        finally:
            worker.close()


class ExactFrameAnalysisBufferTests(unittest.TestCase):
    def test_combined_results_release_only_on_repeated_exact_matches(self) -> None:
        buffer = _ExactFrameAnalysisBuffer(
            detector_enabled=True,
            segmentation_enabled=True,
        )
        detector_one = _perception(1)
        segmentation_one = _segmentation_result(1)
        detector_two = _perception(2)
        segmentation_two = _segmentation_result(2)

        buffer.add_detector(detector_one)
        buffer.add_segmentation(segmentation_two)
        self.assertIsNone(buffer.next_ready())
        buffer.add_segmentation(segmentation_one)
        self.assertEqual(buffer.next_ready(), (detector_one, segmentation_one))
        self.assertIsNone(buffer.next_ready())

        buffer.add_detector(detector_two)
        self.assertEqual(buffer.next_ready(), (detector_two, segmentation_two))
        self.assertIsNone(buffer.next_ready())

        buffer.add_detector(detector_one)
        buffer.add_segmentation(segmentation_one)
        self.assertIsNone(buffer.next_ready())

    def test_different_sequences_are_never_cross_joined(self) -> None:
        buffer = _ExactFrameAnalysisBuffer(
            detector_enabled=True,
            segmentation_enabled=True,
        )
        buffer.add_detector(_perception(10))
        buffer.add_segmentation(_segmentation_result(11))
        self.assertIsNone(buffer.next_ready())

    def test_same_sequence_with_different_carla_frames_is_rejected(self) -> None:
        buffer = _ExactFrameAnalysisBuffer(
            detector_enabled=True,
            segmentation_enabled=True,
        )
        buffer.add_detector(_perception(3))
        buffer.add_segmentation(_segmentation_result(3, carla_frame=9003))
        with self.assertRaisesRegex(RuntimeError, "not a CARLA frame"):
            buffer.next_ready()

    def test_same_sequence_with_different_frame_shapes_is_rejected(self) -> None:
        buffer = _ExactFrameAnalysisBuffer(
            detector_enabled=True,
            segmentation_enabled=True,
        )
        buffer.add_detector(_perception(3, image=_image(height=4, width=6)))
        buffer.add_segmentation(
            _segmentation_result(3, image=_image(height=5, width=6))
        )
        with self.assertRaisesRegex(RuntimeError, "not a frame shape"):
            buffer.next_ready()

    def test_single_enabled_source_can_release_without_the_other_model(self) -> None:
        detector_only = _ExactFrameAnalysisBuffer(
            detector_enabled=True,
            segmentation_enabled=False,
        )
        detector = _perception(4)
        detector_only.add_detector(detector)
        self.assertEqual(detector_only.next_ready(), (detector, None))

        segmentation_only = _ExactFrameAnalysisBuffer(
            detector_enabled=False,
            segmentation_enabled=True,
        )
        segmentation = _segmentation_result(4)
        segmentation_only.add_segmentation(segmentation)
        self.assertEqual(segmentation_only.next_ready(), (None, segmentation))


if __name__ == "__main__":
    unittest.main()
