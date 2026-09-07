"""Local Operator entry point with synchronous custom-model diagnostics.

This branch-only launcher preserves the normal Operator/Garage implementation
and swaps only two explicit seams while ``main`` is running:

- registered model sessions use the synchronous diagnostic starter; and
- model-initialization failures receive a structured HTTP error payload.

The normal ``carla-operator-ui`` entry point remains unchanged so local testing
can compare existing and diagnostic behavior side by side.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Sequence

from . import local_entrypoint
from .model_start_diagnostics import (
    ModelInitializationError,
    start_registered_model_session,
)


def _display_message(error: ModelInitializationError) -> str:
    """Keep the Garage-visible error useful without dumping a traceback into UI text."""

    message = str(error)
    chain = error.diagnostic.exception_chain
    if len(chain) < 2:
        return message
    cause = chain[1]
    cause_type = str(cause.get("type", "Exception"))
    cause_message = str(cause.get("message", "")).strip()
    if not cause_message or cause_message in message:
        return message
    return f"{message} | caused by {cause_type}: {cause_message}"


class ModelDiagnosticRequestHandler(local_entrypoint.LocalConfigGarageRequestHandler):
    """Expose structured custom-model initialization failures to the local UI."""

    def _error(self, error: BaseException) -> None:
        if isinstance(error, ModelInitializationError):
            self._json(
                HTTPStatus.CONFLICT,
                {
                    "error": {
                        "type": type(error).__qualname__,
                        "message": _display_message(error),
                        "details": error.as_dict(),
                    }
                },
            )
            return
        super()._error(error)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the normal local Operator with model-start diagnostics enabled."""

    original_handler: Any = local_entrypoint.LocalConfigGarageRequestHandler
    original_starter = local_entrypoint.start_registered_model_session
    local_entrypoint.LocalConfigGarageRequestHandler = ModelDiagnosticRequestHandler
    local_entrypoint.start_registered_model_session = start_registered_model_session
    try:
        return local_entrypoint.main(argv)
    finally:
        local_entrypoint.start_registered_model_session = original_starter
        local_entrypoint.LocalConfigGarageRequestHandler = original_handler


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ModelDiagnosticRequestHandler", "main"]
