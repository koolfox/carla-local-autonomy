"""Research actions on the existing Operator/JobManager boundary; never shell strings."""

from __future__ import annotations

from pathlib import Path

from .contracts import OperatorJobRequest

RESEARCH_JOB_KINDS = frozenset(
    {
        "dataset_derive",
        "acquisition_plan",
        "acquisition_pool",
        "diagnosis_benchmark",
        "coreset_select",
        "reviewer_check",
    }
)


def build_research_plan(request: OperatorJobRequest, workspace: Path, executable: str):
    # Local import avoids a second command registry or import-time cycle.
    from .commands import (
        CommandPlan,
        _choice,
        _identifier,
        _integer,
        _keys,
        _module_command,
        _number,
        _unused_output,
        _workspace_path,
    )

    p = request.parameters
    tokens = []
    output = None

    def source(name, kind="file"):
        return str(_workspace_path(workspace, p[name], name, kind=kind))

    if request.kind == "dataset_derive":
        _keys(
            p,
            required={"parent", "dataset_id", "action", "sample_ids", "review"},
            name="dataset derivation",
        )
        dataset_id = _identifier(p["dataset_id"], "dataset_id")
        action = _choice(p["action"], "action", {"FREEZE", "RELABEL"})
        output = _unused_output(workspace / "datasets" / dataset_id, "derived dataset")
        tokens = [
            "--parent",
            source("parent", "directory"),
            "--dataset-id",
            dataset_id,
            "--datasets-root",
            str(workspace / "datasets"),
            "--action",
            action,
        ]
        if not isinstance(p["sample_ids"], list) or not p["sample_ids"]:
            raise ValueError("explicit nonempty sample selection required")
        for sample in p["sample_ids"]:
            if not isinstance(sample, str) or not sample or "\x00" in sample:
                raise ValueError("invalid sample ID")
            tokens.extend(("--sample-id", sample))
        if len(p["sample_ids"]) != len(set(p["sample_ids"])):
            raise ValueError("duplicate selected samples")
        if action == "RELABEL":
            tokens.extend(("--review", source("review")))
        elif p["review"] is not None:
            raise ValueError("FREEZE cannot consume a relabel review")
        module = "carla_vision.dataset.derivation"
    elif request.kind == "reviewer_check":
        _keys(p, required={"submission"}, name="reviewer check")
        tokens = ["check", source("submission")]
        module = "carla_vision.reproduction.reviewer"
    else:
        configurations = {
            "acquisition_plan": ("carla_vision.scenarios.study", {"protocol", "candidates"}),
            "acquisition_pool": (
                "carla_vision.failure_mining.acquisition_pool",
                {
                    "evaluation",
                    "independent_review",
                    "estimated_action_cost",
                    "minimum_quality",
                    "seed",
                },
            ),
            "diagnosis_benchmark": (
                "carla_vision.failure_mining.diagnosis_benchmark",
                {"observations", "reference"},
            ),
            "coreset_select": ("carla_vision.scenarios.baselines", {"protocol"}),
        }
        if request.kind not in configurations:
            raise ValueError("unknown research job")
        module, params = configurations[request.kind]
        _keys(p, required=params | {"run_id"}, name=request.kind)
        run_id = _identifier(p["run_id"], "run_id")
        output = _unused_output(workspace / "runs" / run_id, "research run")
        for name in sorted(params):
            if name in {"estimated_action_cost", "minimum_quality"}:
                value = str(
                    _number(p[name], name, 0.0, 1e12 if name == "estimated_action_cost" else 1.0)
                )
            elif name == "seed":
                value = str(_integer(p[name], name, 0, 2**63 - 1))
            else:
                value = source(name, "directory" if name == "evaluation" else "file")
            tokens.extend(("--" + name.replace("_", "-"), value))
        tokens.extend(("--runs-root", str(workspace / "runs"), "--run-id", run_id))
    return CommandPlan(
        kind=request.kind,
        title=f"Research: {request.kind}",
        command=_module_command(executable, module, *tokens),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note="Existing research producer; creates new evidence, never mutates CARLA or prior releases.",
    )
