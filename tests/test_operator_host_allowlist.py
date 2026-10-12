from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from carla_vision.operator.local_entrypoint import prepare_launch
from carla_vision.operator.server import (
    ALLOWED_HOSTS_ENV,
    configured_allowed_hosts,
    create_server,
    host_header_allowed,
)


class HostHeaderPolicyTests(unittest.TestCase):
    def test_local_and_lan_hosts_are_accepted(self) -> None:
        for header in (
            None,
            "127.0.0.1:8765",
            "localhost:5173",
            "LOCALHOST.",
            "[::1]:8765",
            "[fe80::1%en0]:8765",
            "192.168.1.25:8765",
            "garage.localhost",
            "studio-mac.local:8765",
            "studio-mac",
        ):
            with self.subTest(header=header):
                self.assertTrue(host_header_allowed(header))

    def test_public_names_and_malformed_hosts_are_rejected(self) -> None:
        for header in (
            "",
            "attacker.example.com",
            "attacker.example.com:8765",
            "127.0.0.1.attacker.example",
            "::1",
            "host name",
            "localhost:port",
        ):
            with self.subTest(header=header):
                self.assertFalse(host_header_allowed(header))

    def test_configured_hosts_are_accepted(self) -> None:
        allowed = configured_allowed_hosts(
            {ALLOWED_HOSTS_ENV: " Garage.Example.com , other.example.org:443 "}
        )
        self.assertEqual(allowed, {"garage.example.com", "other.example.org"})
        self.assertTrue(host_header_allowed("garage.example.com", allowed))
        self.assertTrue(host_header_allowed("garage.example.com.:443", allowed))
        self.assertFalse(host_header_allowed("evil.example.com", allowed))

    def test_invalid_configured_host_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, ALLOWED_HOSTS_ENV):
            configured_allowed_hosts({ALLOWED_HOSTS_ENV: "https://garage.example.com/"})


class OperatorHostHeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        with mock.patch.dict("os.environ", {ALLOWED_HOSTS_ENV: "garage.example.com"}):
            self.server = create_server(
                workspace=self.temporary.name,
                port=0,
                carla_host="127.0.0.1",
                carla_port=1,
            )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.application.jobs.shutdown()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def request(self, method: str, host: str) -> tuple[int, dict[str, object]]:
        address, port = self.server.server_address[:2]
        connection = http.client.HTTPConnection(address, port, timeout=5)
        try:
            connection.putrequest(method, "/api/bootstrap", skip_host=True)
            connection.putheader("Host", host)
            connection.putheader("Content-Length", "0")
            connection.endheaders()
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_rebound_host_cannot_read_the_bootstrap_token(self) -> None:
        status, payload = self.request("GET", "attacker.example.com:8765")
        self.assertEqual(status, 403)
        self.assertNotIn("token", payload)
        self.assertIn(ALLOWED_HOSTS_ENV, str(payload["error"]))

    def test_rebound_host_is_rejected_before_post_handling(self) -> None:
        status, _ = self.request("POST", "attacker.example.com")
        self.assertEqual(status, 403)

    def test_loopback_and_configured_hosts_keep_working(self) -> None:
        for host in ("127.0.0.1:8765", "localhost:5173", "garage.example.com"):
            with self.subTest(host=host):
                status, payload = self.request("GET", host)
                self.assertEqual(status, 200)
                self.assertIn("token", payload)


class LocalEnvAllowedHostsTests(unittest.TestCase):
    def test_env_file_value_is_exported_to_the_operator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env.local").write_text(
                f"{ALLOWED_HOSTS_ENV}='garage.example.com'\n", encoding="utf-8"
            )
            plan = prepare_launch([], cwd=directory, environ={})
            self.assertEqual(plan.env_updates[ALLOWED_HOSTS_ENV], "garage.example.com")

            process = prepare_launch(
                [], cwd=directory, environ={ALLOWED_HOSTS_ENV: "other.example.com"}
            )
            self.assertNotIn(ALLOWED_HOSTS_ENV, process.env_updates)

    def test_invalid_env_file_value_fails_at_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env.local").write_text(
                f"{ALLOWED_HOSTS_ENV}='https://garage.example.com/'\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, ALLOWED_HOSTS_ENV):
                prepare_launch([], cwd=directory, environ={})


if __name__ == "__main__":
    unittest.main()
