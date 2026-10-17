"""Create a deterministic, commit-bound source archive without models or local data."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ..artifacts import fingerprint_file

PROOF_NAME = "SOURCE_GIT_OBJECTS.json"

ALLOWED_ROOTS = {"carla_vision", "tests", "docs", "configs", "web", ".github", ".agents"}
ALLOWED_FILES = {
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
    "NOTICE.md",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "README.fa.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "PRODUCT.md",
    "DESIGN.md",
    ".gitignore",
    ".gitattributes",
    "CODE_OF_CONDUCT.md",
    "restore_project.py",
    "restore.sh",
    "Restore.cmd",
}


def _git_hash(kind: str, value: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(value)}\0".encode() + value).hexdigest()


def _tree_entries(data: bytes):
    result = {}
    offset = 0
    while offset < len(data):
        space = data.index(b" ", offset)
        end = data.index(b"\0", space)
        mode = data[offset:space].decode("ascii")
        name = data[space + 1 : end].decode("utf-8")
        if name in result or "/" in name or name in {"", ".", ".."} or end + 21 > len(data):
            raise ValueError("invalid Git tree object")
        result[name] = (mode, data[end + 1 : end + 21].hex())
        offset = end + 21
    return result


def verify_source_objects(names, read_member, expected_commit):
    """Verify selected ZIP/tar bytes against the actual Git commit/tree object IDs.

    A Git object ID binds content, not the author's identity or trustworthiness.
    Excluded model/sample blobs are not included and are not revalidated here.
    """
    from .review_contracts import fields, parse_json

    proof = fields(
        parse_json(read_member(PROOF_NAME).decode("utf-8")),
        {"schema_version", "commit_base64", "trees", "selection"},
    )
    if proof["schema_version"] != "1.0" or proof["selection"] != "reviewer-source-v1":
        raise ValueError("unsupported source-object proof")
    commit = base64.b64decode(proof["commit_base64"], validate=True)
    if _git_hash("commit", commit) != expected_commit:
        raise ValueError("source commit object does not match selected Git ID")
    first_line = commit.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise ValueError("invalid Git commit tree")
    tree_id = first_line[5:].decode("ascii")
    trees = {}
    if not isinstance(proof["trees"], dict) or not proof["trees"]:
        raise ValueError("missing Git tree proof")
    for key, encoded in proof["trees"].items():
        value = base64.b64decode(encoded, validate=True)
        if _git_hash("tree", value) != key:
            raise ValueError("source tree object hash changed")
        trees[key] = _tree_entries(value)
    used = set()
    for name in names:
        if name.endswith("/") or name in {"SOURCE_COMMIT.txt", PROOF_NAME}:
            continue
        parts, current = name.split("/"), tree_id
        for part in parts[:-1]:
            used.add(current)
            mode, current = trees[current][part]
            if mode not in {"40000", "040000"}:
                raise ValueError("source path does not follow a Git directory")
        used.add(current)
        mode, blob = trees[current][parts[-1]]
        if mode not in {"100644", "100755"} or _git_hash("blob", read_member(name)) != blob:
            raise ValueError("archived source blob differs from the selected Git tree")
    if used != set(trees):
        raise ValueError("unreferenced source proof trees")
    return tree_id


def build_source_snapshot(repository, output, *, commit="HEAD", maximum_bytes=64 * 1024 * 1024):
    root = Path(repository).resolve(strict=True)
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("snapshot byte budget must be a positive integer")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    if commit != "HEAD" and re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source ref must be HEAD or a full commit SHA")
    resolved = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    # Archive exactly the selected tree, never a dirty workspace or untracked file.
    entries = []
    for entry in git("ls-tree", "-r", "-z", "--full-tree", resolved).split(b"\0"):
        if not entry:
            continue
        header, encoded_name = entry.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        name = encoded_name.decode("utf-8")
        parts = name.split("/")
        if not (parts[0] in ALLOWED_ROOTS or name in ALLOWED_FILES):
            continue
        if (
            any(p in {"", ".", "..", "node_modules", "research_transport"} for p in parts)
            or "\\" in name
            or any(part.startswith(".env") for part in parts)
            or Path(name).suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
        ):
            raise ValueError("unsafe/generated source path")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("source archive excludes symlinks and submodules")
        size = int(git("cat-file", "-s", blob))
        entries.append((name, blob, mode, size))
    if sum(e[3] for e in entries) > maximum_bytes:
        raise ValueError("source archive exceeds byte budget")
    if not any(n.startswith("carla_vision/") for n, *_ in entries):
        raise ValueError("selected tree contains no project source")
    commit_bytes = git("cat-file", "commit", resolved)
    root_tree = commit_bytes.split(b"\n", 1)[0][5:].decode("ascii")
    tree_bytes = {}
    for name, *_ in entries:
        current = root_tree
        for part in [*name.split("/")[:-1], None]:
            if current not in tree_bytes:
                tree_bytes[current] = git("cat-file", "tree", current)
            if part is not None:
                mode, current = _tree_entries(tree_bytes[current])[part]
                if mode not in {"40000", "040000"}:
                    raise ValueError("source path is not a Git tree")
    proof = {
        "schema_version": "1.0",
        "selection": "reviewer-source-v1",
        "commit_base64": base64.b64encode(commit_bytes).decode("ascii"),
        "trees": {k: base64.b64encode(v).decode("ascii") for k, v in tree_bytes.items()},
    }
    proof_bytes = (json.dumps(proof, sort_keys=True) + "\n").encode()
    if len(proof_bytes) + sum(e[3] for e in entries) > maximum_bytes:
        raise ValueError("source archive and Git proof exceed byte budget")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".source-", suffix=".zip", dir=target.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:

            def write(name, data, mode):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.external_attr = int(mode, 8) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)

            write("SOURCE_COMMIT.txt", (resolved + "\n").encode(), "100644")
            write(PROOF_NAME, proof_bytes, "100644")
            for name, blob, mode, size in sorted(entries):
                data = git("cat-file", "blob", blob)
                if len(data) != size:
                    raise ValueError("source object changed during archive")
                write(name, data, mode)
        # Publish with no replacement, even if another writer wins after the
        # initial existence check. Hard-link publication is on the same filesystem.
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": "1.0",
        "source_commit": resolved,
        "source_files": len(entries),
        "archive": fingerprint_file(target),
        "scope": "selected_tracked_source_tree",
        "excluded": "model/sample payloads, datasets, runs, local credentials and untracked files",
    }


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_source_snapshot(args.repository, args.output, commit=args.commit), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
