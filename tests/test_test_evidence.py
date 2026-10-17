"""Actual tiny pytest subprocesses validate test/source binding, not research results."""

from __future__ import annotations

import json
import subprocess

import pytest

from carla_vision.reproduction.test_evidence import run_tests, verify_test_receipt


def repo(tmp_path, code="def test_example():\n    assert True\n"):
    root = tmp_path / "source"
    root.mkdir()
    (root / "test_tiny.py").write_text(code)
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (root / "uv.lock").write_text("# synthetic fixture lock\n" + "x" * 40)
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Synthetic Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "test fixture",
        ],
        check=True,
        capture_output=True,
    )
    return root


def test_actual_pytest_result_is_bound_to_selected_commit(tmp_path):
    root = repo(tmp_path)
    output = tmp_path / "evidence"
    raw = run_tests(root, output)
    assert raw["status"] == "passed"
    result = verify_test_receipt(
        output / "test-evidence.json", source_commit=raw["source_before"]["commit"]
    )
    assert result["cases"] == 1
    with pytest.raises(ValueError, match="another"):
        verify_test_receipt(output / "test-evidence.json", source_commit="0" * 40)
    (output / "tests.xml").write_text("<testsuite/>")
    with pytest.raises(ValueError, match="digest|changed"):
        verify_test_receipt(
            output / "test-evidence.json", source_commit=raw["source_before"]["commit"]
        )


def test_failed_pytest_is_retained_not_promoted(tmp_path):
    raw = run_tests(
        repo(tmp_path, "def test_failure():\n    assert False\n"), tmp_path / "evidence"
    )
    assert raw["status"] == "failed" and raw["returncode"] != 0
    assert raw["counts"]["failures"] == 1
    assert (tmp_path / "evidence/pytest.log").is_file()
    with pytest.raises(ValueError, match="failed"):
        verify_test_receipt(
            tmp_path / "evidence/test-evidence.json", source_commit=raw["source_before"]["commit"]
        )


def test_dirty_source_cannot_obtain_committed_ci_receipt(tmp_path):
    root = repo(tmp_path)
    (root / "test_tiny.py").write_text("def test_other(): pass\n")
    with pytest.raises(ValueError, match="clean committed"):
        run_tests(root, tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()


def test_bare_junit_is_not_source_bound_empirical_evidence(tmp_path):
    from carla_vision.reproduction.reviewer import _check_role

    p = tmp_path / "tests.xml"
    p.write_text('<testsuite><testcase name="fixture"/></testsuite>')
    assert _check_role("tests", p, {}, empirical=False)["cases"] == 1
    with pytest.raises(ValueError, match="source-bound"):
        _check_role("tests", p, {"source_commit": "a" * 40}, empirical=True)


def test_wrong_junit_path_or_exit_code_cannot_be_substituted(tmp_path):
    raw = run_tests(repo(tmp_path), tmp_path / "evidence")
    path = tmp_path / "evidence/test-evidence.json"
    raw["command"][4] = "--junitxml=/unrelated.xml"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="output path"):
        verify_test_receipt(path, source_commit=raw["source_before"]["commit"])


@pytest.mark.parametrize("content", [b"<!DOCTYPE a><testsuite/>", b"\x00<testsuite/>"])
def test_junit_rejects_unsafe_encodings_and_entities(tmp_path, content):
    from carla_vision.reproduction.test_evidence import junit_counts

    path = tmp_path / "bad.xml"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="entities"):
        junit_counts(path)
