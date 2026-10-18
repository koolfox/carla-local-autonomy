"""Fresh subprocess/JUnit evidence must refer to the source that was executed."""

import json
import subprocess

import pytest

from carla_vision.reproduction.source_snapshot import build_source_snapshot


def source(tmp_path, body="def test_ok():\n    assert 1 == 1\n"):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "carla_vision").mkdir()
    (root / "carla_vision/__init__.py").write_text("")
    (root / "uv.lock").write_text("version = 1\n# fixture only: no dependency installation\n")
    (root / "tests/test_case.py").write_text(body)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def test_fresh_junit_binds_committed_source_and_relocates(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests, verify_test_execution

    root = source(tmp_path)
    archive = tmp_path / "source.zip"
    snap = build_source_snapshot(root, archive)
    output = tmp_path / "evidence"
    result = run_tests(root, output, timeout_seconds=30)
    assert result["status"] == "passed" and result["junit"]["cases"] == 1
    renamed = tmp_path / "moved"
    output.rename(renamed)
    assert (
        verify_test_execution(
            renamed / "execution.json", source_archive=archive, source_commit=snap["source_commit"]
        )["status"]
        == "passed"
    )
    with pytest.raises(ValueError, match="source"):
        verify_test_execution(
            renamed / "execution.json", source_archive=archive, source_commit="f" * 40
        )


def test_failed_child_never_creates_passing_test_evidence(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests, verify_test_execution

    root = source(tmp_path, "def test_bad():\n    assert False\n")
    output = tmp_path / "evidence"
    result = run_tests(root, output, timeout_seconds=30)
    assert result["status"] == "failed" and result["returncode"] != 0
    assert (output / "pytest.log").is_file()
    with pytest.raises(ValueError, match="execution"):
        verify_test_execution(output / "execution.json")


def test_missing_tests_and_skips_are_not_passing_evidence(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(
        tmp_path, "import pytest\n@pytest.mark.skip(reason='fixture')\ndef test_skip():\n    pass\n"
    )
    result = run_tests(root, tmp_path / "evidence", timeout_seconds=30)
    assert result["status"] == "failed" and result["junit"]["skipped"] == 1


def test_output_must_be_new_and_outside_source(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        run_tests(root, root / "evidence")
    target = tmp_path / "old"
    target.mkdir()
    (target / "tests.xml").write_text("<testsuite/>")
    with pytest.raises(FileExistsError):
        run_tests(root, target)


def test_dirty_or_untracked_code_cannot_be_executed_as_committed(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(tmp_path)
    (root / "tests/test_case.py").write_text("def test_other(): pass\n")
    with pytest.raises(ValueError, match="source"):
        run_tests(root, tmp_path / "evidence")


def test_mutation_during_tests_is_recorded_as_failure(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(
        tmp_path,
        "from pathlib import Path\ndef test_mutate():\n    Path('uv.lock').write_text('changed')\n",
    )
    result = run_tests(root, tmp_path / "evidence", timeout_seconds=30)
    assert result["status"] == "failed" and result["source_unchanged"] is False


def test_changed_junit_or_changed_log_rejected(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests, verify_test_execution

    root = source(tmp_path)
    output = tmp_path / "evidence"
    run_tests(root, output, timeout_seconds=30)
    (output / "tests.xml").write_text('<testsuite><testcase name="invented"/></testsuite>')
    with pytest.raises(ValueError, match="changed"):
        verify_test_execution(output / "execution.json")


def test_test_selection_cannot_escape_checkout(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(tmp_path)
    for selection in (["../test_external.py"], ["--collect-only"], []):
        with pytest.raises(ValueError, match="selection"):
            run_tests(root, tmp_path / "evidence", targets=selection)


def test_false_or_empty_receipt_rejected(tmp_path):
    from carla_vision.reproduction.test_execution import verify_test_execution

    p = tmp_path / "execution.json"
    p.write_text(json.dumps({"status": "passed", "cases": 90000}))
    with pytest.raises(ValueError):
        verify_test_execution(p)


def test_timeout_is_not_promoted_even_if_an_xml_was_created(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests

    root = source(tmp_path, "import time\ndef test_slow():\n    time.sleep(20)\n")
    result = run_tests(root, tmp_path / "evidence", timeout_seconds=0.2)
    assert result["returncode"] == 124 and result["status"] == "failed"


def test_receipt_command_and_selection_cannot_be_relabelled(tmp_path):
    from carla_vision.reproduction.test_execution import run_tests, verify_test_execution

    root = source(tmp_path)
    out = tmp_path / "evidence"
    run_tests(root, out, timeout_seconds=30)
    path = out / "execution.json"
    raw = json.loads(path.read_text())
    raw["command"][2] = "not_pytest"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="command"):
        verify_test_execution(path)
