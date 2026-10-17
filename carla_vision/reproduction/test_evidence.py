"""Execute pytest and bind its returned JUnit bytes to the checked-out source.

This is locally recorded execution provenance, not a signed CI identity claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from ..artifacts import fingerprint_file
from .review_contracts import fields, integer, read_json
from .study_links import _reference


def junit_counts(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("JUnit exceeds parsing budget")
    raw = path.read_bytes()
    if b"\x00" in raw or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("JUnit entities are not allowed")
    cases = list(ElementTree.fromstring(raw).iter("testcase"))
    return {
        "cases": len(cases),
        "failures": sum(bool(list(c.iter("failure"))) for c in cases),
        "errors": sum(bool(list(c.iter("error"))) for c in cases),
        "skipped": sum(bool(list(c.iter("skipped"))) for c in cases),
    }


def _source(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def run_tests(repository, output, selectors=()):
    root, output = Path(repository).resolve(strict=True), Path(output).absolute()
    # No shell strings or arbitrary commands; selectors are pytest file/node IDs only.
    for value in selectors:
        if value.startswith("-") or not value.split("::")[0].endswith(".py"):
            raise ValueError("test selectors must be explicit Python test files or nodes")
    before = _source(root)
    if before["clean"] is not True:
        raise ValueError("test evidence requires a clean committed source workspace")
    lock = fingerprint_file(root / "uv.lock")
    output.mkdir(parents=True, exist_ok=False)
    xml, log = output / "tests.xml", output / "pytest.log"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}", *selectors]
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        stream.flush()
        os.fsync(stream.fileno())
    after = _source(root)
    counts = junit_counts(xml) if xml.is_file() else None
    complete = (
        result.returncode == 0
        and before == after
        and lock == fingerprint_file(root / "uv.lock")
        and counts
        and counts["cases"] > 0
        and not any(counts[k] for k in ("errors", "failures", "skipped"))
    )
    value = {
        "schema_version": "1.0",
        "kind": "pytest_execution_receipt",
        "source_before": before,
        "source_after": after,
        "dependency_lock": lock,
        "command": command,
        "returncode": result.returncode,
        "junit": fingerprint_file(xml) if xml.is_file() else None,
        "execution_log": fingerprint_file(log),
        "counts": counts,
        "status": "passed" if complete else "failed",
        "ci": {
            k: os.environ.get(k)
            for k in ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        },
        "scope": "source_and_process_binding_not_independent_or_signed_attestation",
    }
    with (output / "test-evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return value


def verify_test_receipt(path, *, source_commit):
    path = Path(path).resolve(strict=True)
    raw = read_json(path)
    fields(
        raw,
        {
            "schema_version",
            "kind",
            "source_before",
            "source_after",
            "dependency_lock",
            "command",
            "returncode",
            "junit",
            "execution_log",
            "counts",
            "status",
            "ci",
            "scope",
        },
    )
    if raw["schema_version"] != "1.0" or raw["kind"] != "pytest_execution_receipt":
        raise ValueError("unsupported test receipt")
    for key in ("source_before", "source_after"):
        fields(raw[key], {"commit", "tree", "clean"})
        if (
            raw[key]["commit"] != source_commit
            or raw[key]["clean"] is not True
            or re.fullmatch(r"[0-9a-f]{40}", raw[key]["tree"]) is None
        ):
            raise ValueError("test receipt belongs to another/dirty source")
    if raw["source_before"] != raw["source_after"]:
        raise ValueError("source changed during tests")
    xml = _reference(path.parent, raw["junit"])
    _reference(path.parent, raw["execution_log"])
    _reference(path.parent, raw["dependency_lock"])
    counts = junit_counts(xml)
    if (
        counts != raw["counts"]
        or not counts["cases"]
        or any(counts[k] for k in ("errors", "failures", "skipped"))
    ):
        raise ValueError("test evidence JUnit is incomplete or failed")
    command = raw["command"]
    if not isinstance(command, list) or len(command) < 5 or command[1:4] != ["-m", "pytest", "-q"]:
        raise ValueError("test command is not the recorded pytest invocation")
    if command[4] != f"--junitxml={xml}":
        raise ValueError("JUnit differs from the executed output path")
    if integer(raw["returncode"], "exit code") != 0 or raw["status"] != "passed":
        raise ValueError("test process did not complete successfully")
    return {
        "status": "passed",
        **counts,
        "source_commit": source_commit,
        "source_tree": raw["source_before"]["tree"],
        "dependency_lock_sha256": raw["dependency_lock"]["sha256"],
        "empirical_validation": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test", action="append", default=[])
    args = parser.parse_args(argv)
    result = run_tests(args.repository, args.output, args.test)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
