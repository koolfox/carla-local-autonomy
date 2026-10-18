"""Run fresh pytest and bind its exit, JUnit and log to exact committed source.

This is a software execution receipt, not a signature or proof against a dishonest
runner. An old XML file alone is never accepted as source-bound CI evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest
from .review_contracts import fields, number, read_json, relative_path, text
from .source_snapshot import ALLOWED_FILES, ALLOWED_ROOTS, PROOF_NAME, _git_hash

SCHEMA = "1.0"
SCOPE = "fresh_pytest_software_execution_not_signed_attestation"


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def _workspace(root):
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked = []
    total = 0
    for entry in _git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        name = name.decode()
        if name.split("/")[0] not in ALLOWED_ROOTS and name not in ALLOWED_FILES:
            continue
        path = relative_path(root, name)
        if (
            kind != "blob"
            or mode not in {"100644", "100755"}
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            raise ValueError("unsafe or oversized executed source")
        data = path.read_bytes()
        total += len(data)
        if total > 256 * 1024 * 1024 or _git_hash("blob", data) != blob:
            raise ValueError("workspace source differs from selected commit")
        tracked.append([name, blob, mode])
    if _git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("executed source must be clean with no untracked files")
    if not tracked:
        raise ValueError("no executed source files")
    return {
        "commit": commit,
        "tree": tree,
        "lock_sha256": fingerprint_file(root / "uv.lock")["sha256"],
        "selected_source_sha256": canonical_digest(sorted(tracked)),
    }


def _ref(root, ref):
    fields(ref, {"path", "sha256", "size_bytes"})
    p = relative_path(root, ref["path"])
    actual = fingerprint_file(p)
    if any(actual[k] != ref[k] for k in ("sha256", "size_bytes")):
        raise ValueError("execution evidence bytes changed")
    return p


def junit_summary(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("JUnit parsing budget exceeded")
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("JUnit entities are not permitted")
    root = ElementTree.fromstring(raw)
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("not a JUnit document")
    cases = list(root.iter("testcase"))
    return {
        "cases": len(cases),
        "failures": sum(bool(list(c.iter("failure"))) for c in cases),
        "errors": sum(bool(list(c.iter("error"))) for c in cases),
        "skipped": sum(bool(list(c.iter("skipped"))) for c in cases),
    }


def run_tests(repository, output, *, targets=None, timeout_seconds=1200):
    root = Path(repository).resolve(strict=True)
    output = Path(output).absolute()
    if output.resolve().is_relative_to(root):
        raise ValueError("test evidence output must be outside the source checkout")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    targets = ["tests"] if targets is None else targets
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
        raise ValueError("test selection must be a nonempty unique list")
    for target in targets:
        if not isinstance(target, str) or not (target == "tests" or target.startswith("tests/")):
            raise ValueError("test selection must be tracked paths under tests/")
        try:
            path = relative_path(root, target)
        except (OSError, ValueError) as error:
            raise ValueError("unsafe test selection") from error
        if not path.is_dir() and path.suffix != ".py":
            raise ValueError("test selection must contain Python tests")
    number(timeout_seconds, "timeout", minimum=0.01)
    before = _workspace(root)
    output.mkdir(parents=True, exist_ok=False)
    xml_path, log_path = output / "tests.xml", output / "pytest.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *targets,
        f"--junitxml={xml_path}",
    ]
    env = dict(os.environ, PYTEST_ADDOPTS="", PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    run_id = uuid.uuid4().hex
    started = time.time()
    error = None
    code = None
    with log_path.open("xb") as stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
            )
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
                code = 124
                error = "pytest deadline exceeded; completed XML alone does not prove completed execution"
        except OSError as exc:
            code = 126
            error = f"pytest could not start: {exc}"
        stream.flush()
        os.fsync(stream.fileno())
    try:
        after = _workspace(root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        after = None
        error = f"executed source changed: {exc}"
    summary = None
    try:
        summary = junit_summary(xml_path)
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        error = f"JUnit is absent/invalid: {exc}"
    unchanged = before == after
    passed = (
        code == 0
        and error is None
        and unchanged
        and summary
        and summary["cases"] > 0
        and not any(summary[k] for k in ("failures", "errors", "skipped"))
    )

    def ref(path):
        fp = fingerprint_file(path)
        fp["path"] = path.name
        return fp

    payload = {
        "schema_version": SCHEMA,
        "scope": SCOPE,
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "source": before,
        "source_after": after,
        "source_unchanged": unchanged,
        "targets": targets,
        "command": command,
        "returncode": code,
        "started_unix": started,
        "finished_unix": time.time(),
        "timeout_seconds": timeout_seconds,
        "junit": summary,
        "junit_file": ref(xml_path) if xml_path.is_file() else None,
        "log_file": ref(log_path),
        "error": error,
        "runner": {
            "python": sys.version,
            "platform": sys.platform,
            **{
                k: os.environ[k]
                for k in ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")
                if k in os.environ
            },
        },
    }
    with (output / "execution.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def verify_test_execution(path, *, source_archive=None, source_commit=None):
    path = Path(path).resolve(strict=True)
    document = read_json(path)
    fields(
        document,
        {
            "schema_version",
            "scope",
            "run_id",
            "status",
            "source",
            "source_after",
            "source_unchanged",
            "targets",
            "command",
            "returncode",
            "started_unix",
            "finished_unix",
            "timeout_seconds",
            "junit",
            "junit_file",
            "log_file",
            "error",
            "runner",
        },
    )
    if (
        document["schema_version"] != SCHEMA
        or document["scope"] != SCOPE
        or document["status"] != "passed"
        or type(document["returncode"]) is not int
        or document["returncode"] != 0
        or document["source_unchanged"] is not True
        or document["source"] != document["source_after"]
        or document["error"] is not None
    ):
        raise ValueError("test execution did not finish successfully on unchanged source")
    text(document["run_id"], "execution run ID")
    source = document["source"]
    fields(source, {"commit", "tree", "lock_sha256", "selected_source_sha256"})
    for k in ("commit", "tree"):
        if not isinstance(source[k], str) or re.fullmatch("[0-9a-f]{40}", source[k]) is None:
            raise ValueError("invalid source identity")
    from .review_contracts import digest

    digest(source["lock_sha256"])
    digest(source["selected_source_sha256"])
    if source_commit is not None and source["commit"] != source_commit:
        raise ValueError("test execution belongs to another source commit")
    start = number(document["started_unix"], "start", minimum=0)
    if number(document["finished_unix"], "finish", minimum=start) < start:
        raise ValueError("invalid execution time")
    number(document["timeout_seconds"], "timeout", minimum=0.01)
    xml = _ref(path.parent, document["junit_file"])
    _ref(path.parent, document["log_file"])
    summary = junit_summary(xml)
    if (
        summary != document["junit"]
        or summary["cases"] == 0
        or any(summary[k] for k in ("errors", "failures", "skipped"))
    ):
        raise ValueError("execution JUnit is incomplete/failed or differs from receipt")
    targets = document["targets"]
    command = document["command"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or any(
            not isinstance(t, str)
            or not (t == "tests" or t.startswith("tests/"))
            or ".." in t.split("/")
            for t in targets
        )
    ):
        raise ValueError("invalid execution test selection")
    if (
        not isinstance(command, list)
        or len(command) != 7 + len(targets)
        or command[1:6] != ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
        or command[6:-1] != targets
        or not command[-1].startswith("--junitxml=")
    ):
        raise ValueError("execution command does not bind selected tests")
    if source_archive is not None:
        from .reviewer import _source_archive

        _source_archive(Path(source_archive), source["commit"])
        # Source producer emits ZIP. Other archive formats need a separate importer.
        if not zipfile.is_zipfile(source_archive):
            raise ValueError("execution binding requires a ZIP source snapshot")
        with zipfile.ZipFile(source_archive) as archive:
            proof = json.loads(archive.read(PROOF_NAME))
            commit = base64.b64decode(proof["commit_base64"], validate=True)
            if (
                commit.split(b"\n", 1)[0] != f"tree {source['tree']}".encode()
                or hashlib.sha256(archive.read("uv.lock")).hexdigest() != source["lock_sha256"]
            ):
                raise ValueError("test execution source tree/lock differs from selected archive")
            from .source_snapshot import _tree_entries

            trees = {
                k: _tree_entries(base64.b64decode(v, validate=True))
                for k, v in proof["trees"].items()
            }
            tracked = []
            for name in archive.namelist():
                if name.endswith("/") or name in {PROOF_NAME, "SOURCE_COMMIT.txt"}:
                    continue
                current = source["tree"]
                for part in name.split("/")[:-1]:
                    _, current = trees[current][part]
                mode, blob = trees[current][name.split("/")[-1]]
                tracked.append([name, blob, mode])
            if canonical_digest(sorted(tracked)) != source["selected_source_sha256"]:
                raise ValueError("test execution source inventory differs from selected archive")
    return {
        "status": "passed",
        **summary,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "targets": targets,
        "scope": SCOPE,
        "empirical_validation": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repository", default=".")
    run.add_argument("--output", required=True)
    run.add_argument("--test", action="append", dest="targets")
    run.add_argument("--timeout-seconds", type=float, default=1200)
    check = sub.add_parser("check")
    check.add_argument("receipt")
    check.add_argument("--source-archive")
    check.add_argument("--source-commit")
    args = parser.parse_args(argv)
    if args.action == "run":
        report = run_tests(
            args.repository, args.output, targets=args.targets, timeout_seconds=args.timeout_seconds
        )
    else:
        report = verify_test_execution(
            args.receipt, source_archive=args.source_archive, source_commit=args.source_commit
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
