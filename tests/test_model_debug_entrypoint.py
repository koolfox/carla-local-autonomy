from __future__ import annotations

from http import HTTPStatus

import pytest

from carla_vision.operator import local_entrypoint
from carla_vision.operator.model_debug_entrypoint import (
    ModelDiagnosticRequestHandler,
    main,
)
from carla_vision.operator.model_start_diagnostics import (
    ModelInitializationDiagnostic,
    ModelInitializationError,
    start_registered_model_session,
)


def _error() -> ModelInitializationError:
    return ModelInitializationError(
        ModelInitializationDiagnostic(
            status="failed",
            package_id="road-policy",
            runtime="python_factory",
            factory="research_adapter.policy:create_driver",
            device="cpu",
            checkpoint_path="/workspace/models/road-policy/policy.pth",
            artifact_sha256="a" * 64,
            manifest_path="models/road-policy/model.json",
            manifest_sha256="b" * 64,
            adapter_sha256="c" * 64,
            duration_seconds=0.25,
            exception={"type": "RuntimeError", "message": "broken checkpoint"},
            exception_chain=(
                {"type": "RuntimeError", "message": "broken checkpoint"},
                {"type": "ValueError", "message": "size mismatch"},
            ),
            traceback_text="Traceback ... size mismatch",
        )
    )


def test_handler_returns_structured_model_initialization_diagnostic() -> None:
    handler = object.__new__(ModelDiagnosticRequestHandler)
    responses: list[tuple[HTTPStatus, object]] = []
    handler._json = lambda status, payload: responses.append((status, payload))  # type: ignore[method-assign]

    handler._error(_error())

    assert len(responses) == 1
    status, payload = responses[0]
    assert status == HTTPStatus.CONFLICT
    error = payload["error"]
    assert error["type"] == "ModelInitializationError"
    assert "road-policy" in error["message"]
    assert error["details"]["phase"] == "model_initialization"
    assert error["details"]["status"] == "failed"
    assert error["details"]["exception"]["message"] == "broken checkpoint"
    assert "size mismatch" in error["details"]["traceback"]


def test_debug_entrypoint_installs_and_restores_diagnostic_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_handler = local_entrypoint.LocalConfigGarageRequestHandler
    original_starter = local_entrypoint.start_registered_model_session
    observed: dict[str, object] = {}

    def fake_main(argv: object) -> int:
        observed["argv"] = argv
        observed["handler"] = local_entrypoint.LocalConfigGarageRequestHandler
        observed["starter"] = local_entrypoint.start_registered_model_session
        return 17

    monkeypatch.setattr(local_entrypoint, "main", fake_main)

    result = main(("--port", "0"))

    assert result == 17
    assert observed["argv"] == ("--port", "0")
    assert observed["handler"] is ModelDiagnosticRequestHandler
    assert observed["starter"] is start_registered_model_session
    assert local_entrypoint.LocalConfigGarageRequestHandler is original_handler
    assert local_entrypoint.start_registered_model_session is original_starter
