from __future__ import annotations

import json
import socket
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from carla_vision.operator.product_console import ProductConsoleRequestHandler


def test_stale_token_closes_connection_before_unread_body_can_be_reparsed() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProductConsoleRequestHandler)
    server.application = SimpleNamespace(token="fresh-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {
                "schema_version": "1.0",
                "session": {
                    "identity": {"runId": "stale-token-regression", "seed": 7},
                    "control": {"mode": "autopilot"},
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        host, port = server.server_address
        request = (
            b"POST /api/session/start HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode("ascii")
            + b"Accept: application/json\r\n"
            + b"Content-Type: application/json\r\n"
            + b"X-Operator-Token: stale-token\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: keep-alive\r\n"
            + b"\r\n"
            + body
            + b"GET /api/bootstrap HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode("ascii")
            + b"Connection: close\r\n"
            + b"\r\n"
        )

        with socket.create_connection(server.server_address, timeout=2.0) as connection:
            connection.settimeout(2.0)
            connection.sendall(request)
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = connection.recv(65536)
                except TimeoutError as error:
                    raise AssertionError(
                        "server kept the unauthorized HTTP/1.1 connection open"
                    ) from error
                if not chunk:
                    break
                chunks.append(chunk)

        response = b"".join(chunks)
        assert b"HTTP/1.1 403 Forbidden" in response
        assert b"invalid UI token" in response
        assert b"501 Unsupported Method" not in response
        assert b"Unsupported method" not in response
        assert response.count(b"HTTP/1.1 ") == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
