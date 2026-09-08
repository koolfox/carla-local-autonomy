from unittest.mock import Mock, patch

from carla_vision.native.world_worker import _supervise_worker


def test_restarts_crash_then_respects_normal_exit():
    failed, stopped = Mock(), Mock()
    failed.wait.return_value = 1
    stopped.wait.return_value = 0
    with patch("carla_vision.native.world_worker.subprocess.Popen", side_effect=[failed, stopped]) as spawn:
        with patch("carla_vision.native.world_worker.time.sleep") as sleep:
            assert _supervise_worker(["--port", "8766"]) == 0
    assert spawn.call_count == 2
    assert "--internal-worker" in spawn.call_args.args[0]
    sleep.assert_called_once_with(2)


def test_ctrl_c_does_not_restart():
    child = Mock()
    child.wait.side_effect = [KeyboardInterrupt, 0]
    with patch("carla_vision.native.world_worker.subprocess.Popen", return_value=child) as spawn:
        assert _supervise_worker([]) == 0
    assert spawn.call_count == 1
