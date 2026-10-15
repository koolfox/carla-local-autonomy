"""Cooperative OS-backed lease for this host's native collectors.

This cannot stop an unrelated CARLA client. Exact tick checks detect interference.
The OS releases the lock on process death; the small lock file is deliberately
not unlinked (unlinking a held inode could permit two independent locks).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO


class NativeSessionLease:
    def __init__(self, host: str, port: int, *, directory: Path | None = None) -> None:
        key = hashlib.sha256(f"{host.strip().lower()}:{port}".encode()).hexdigest()
        self.path = (directory or Path(tempfile.gettempdir()) / "carlapyvision-leases") / (
            key + ".lock"
        )
        self.stream: BinaryIO | None = None

    def __enter__(self) -> NativeSessionLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("another local collector owns this CARLA endpoint") from error
        self.stream = stream
        return self

    def __exit__(self, *_args: object) -> None:
        if self.stream is not None:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
