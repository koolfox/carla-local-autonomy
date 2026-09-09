from pathlib import Path
from unittest.mock import patch

import pytest

from carla_vision.operator.recording_preview import RecordingPreviews


def test_preview_reuses_job_and_preserves_original(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original evidence")

    def convert(executable: str, original: Path, target: Path) -> Path:
        assert original == source
        target.write_bytes(b"browser copy")
        return target

    previews = RecordingPreviews()
    try:
        with patch("shutil.which", return_value="ffmpeg"), patch.object(
            previews, "_convert", side_effect=convert
        ) as conversion:
            result = previews.request(source)
            previews._jobs[result["id"]].result(timeout=5)
            ready = previews.request(source)
            assert ready["status"] == "ready"
            assert conversion.call_count == 1
            assert previews.file(ready["id"]).read_bytes() == b"browser copy"
            assert source.read_bytes() == b"original evidence"
    finally:
        previews.close()


def test_preview_requires_ffmpeg_and_rejects_unknown_copy(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original evidence")
    previews = RecordingPreviews()
    try:
        with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="Install FFmpeg"):
            previews.request(source)
        with pytest.raises(FileNotFoundError, match="not ready"):
            previews.file("../../unregistered.mp4")
    finally:
        previews.close()


def test_failed_conversion_removes_partial_copy(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original evidence")
    target = tmp_path / "partial.mp4"
    target.write_bytes(b"incomplete")
    with patch("subprocess.run", side_effect=OSError("encoder failed")), pytest.raises(
        RuntimeError, match="original recording is unchanged"
    ):
        RecordingPreviews._convert("ffmpeg", source, target)
    assert not target.exists()
    assert source.read_bytes() == b"original evidence"
