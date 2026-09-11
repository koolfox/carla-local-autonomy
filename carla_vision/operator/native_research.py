"""Mac/Linux SDK and CLI for the optional native task host on the World Worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request

from .world_worker_client import WorldWorkerClient, WorldWorkerError


class NativeResearchClient(WorldWorkerClient):
    def tasks(self) -> dict:
        return self._request("GET", "/v1/research/tasks")

    def submit(
        self,
        *,
        task_id: str,
        job_id: str,
        parameters: dict,
        acknowledge_world_reload: bool = False,
        task_sha256: str | None = None,
    ) -> dict:
        if not acknowledge_world_reload:
            raise ValueError("native collection requires explicit world-reload acknowledgement")
        if task_sha256 is None:
            matches = [task for task in self.tasks()["tasks"] if task["id"] == task_id]
            if len(matches) != 1:
                raise ValueError("requested task is not installed; inspect the task catalog")
            task_sha256 = matches[0]["manifest_sha256"]
        return self._request(
            "POST",
            "/v1/research/jobs",
            {
                "job_id": job_id,
                "task_id": task_id,
                "task_sha256": task_sha256,
                "parameters": parameters,
                "acknowledge_world_reload": True,
            },
        )

    def status(self, job_id: str) -> dict:
        return self._request("GET", "/v1/research/jobs/" + quote(job_id, safe=""))

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", "/v1/research/jobs/" + quote(job_id, safe="") + "/cancel", {})

    def log(self, job_id: str) -> dict:
        return self._request("GET", "/v1/research/jobs/" + quote(job_id, safe="") + "/log")

    def fetch(self, job_id: str, destination: Path, *, max_bytes: int = 8 * 1024**3) -> Path:
        status = self.status(job_id)
        if status["status"] != "succeeded":
            raise ValueError("only a successfully completed task archive can be fetched")
        archive = status["result"]["archive"]
        expected, size = archive["sha256"], archive["size_bytes"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise WorldWorkerError("task archive checksum is invalid")
        if type(size) is not int or not 0 < size <= max_bytes:
            raise WorldWorkerError("task archive exceeds the download limit")
        destination = destination.expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".native-download-", dir=destination.parent)
        try:
            request = Request(
                self.base_url
                + "/v1/research/jobs/"
                + quote(job_id, safe="")
                + "/files/artifacts.zip",
                headers={"Authorization": f"Bearer {self._bearer_token}"},
            )
            total, checksum = 0, hashlib.sha256()
            with (
                os.fdopen(descriptor, "wb") as target,
                self._opener.open(request, timeout=30) as response,
            ):
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise WorldWorkerError("task archive is larger than its declared size")
                    checksum.update(chunk)
                    target.write(chunk)
            if total != size or checksum.hexdigest() != expected:
                raise WorldWorkerError("task archive size/checksum verification failed")
            os.link(temporary, destination)  # Atomic publication; never overwrite another result.
        except URLError as error:
            raise WorldWorkerError(f"native archive download failed: {error.reason}") from error
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-url", help="World Worker URL; otherwise uses CARLA_WORLD_WORKER_URL"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks")
    submit = sub.add_parser("submit")
    submit.add_argument("--task", default="teacher_capture")
    submit.add_argument("--job-id", required=True)
    submit.add_argument("--parameters", type=Path, required=True)
    submit.add_argument("--acknowledge-world-reload", action="store_true")
    for command in ("status", "log", "cancel", "fetch"):
        item = sub.add_parser(command)
        item.add_argument("job_id")
        if command == "fetch":
            item.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    # Same token file and precedence as the Garage; no new credential store.
    from .local_entrypoint import load_local_env

    config = {**load_local_env(Path.cwd() / ".env.local"), **os.environ}
    url = args.worker_url or config.get("CARLA_WORLD_WORKER_URL")
    if not url or url == "auto":
        parser.error("pass --worker-url with the connected Windows bridge URL")
    token = config.get("CARLA_WORLD_WORKER_TOKEN")
    if not token:
        parser.error("set CARLA_WORLD_WORKER_TOKEN in the environment or project .env.local")
    client = NativeResearchClient(url, token, timeout=10)
    if args.command == "tasks":
        result = client.tasks()
    elif args.command == "submit":
        result = client.submit(
            task_id=args.task,
            job_id=args.job_id,
            parameters=json.loads(args.parameters.read_text(encoding="utf-8")),
            acknowledge_world_reload=args.acknowledge_world_reload,
        )
    elif args.command == "fetch":
        result = {"archive": str(client.fetch(args.job_id, args.destination))}
    else:
        result = getattr(client, args.command)(args.job_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
