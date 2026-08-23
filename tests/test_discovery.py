from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_vision.discovery import DiscoveryScope, discover_carla_servers, local_ipv4_scopes
from carla_vision.operator.server import (
    create_server,
    parse_args,
    resolve_carla_host,
    resolve_world_worker_url,
)


class LocalScopeTests(unittest.TestCase):
    def test_selects_physical_private_lan_and_ignores_tunnels_and_link_local(self) -> None:
        up = SimpleNamespace(isup=True)
        fake = SimpleNamespace(
            net_if_stats=lambda: {name: up for name in ("en0", "utun4", "en6", "docker0")},
            net_if_addrs=lambda: {
                "en0": [
                    SimpleNamespace(
                        family=socket.AF_INET,
                        address="192.168.1.103",
                        netmask="255.255.255.0",
                        ptp=None,
                    )
                ],
                "utun4": [
                    SimpleNamespace(
                        family=socket.AF_INET,
                        address="10.0.0.2",
                        netmask="255.255.255.0",
                        ptp="10.0.0.2",
                    )
                ],
                "en6": [
                    SimpleNamespace(
                        family=socket.AF_INET,
                        address="169.254.10.2",
                        netmask="255.255.0.0",
                        ptp=None,
                    )
                ],
                "docker0": [
                    SimpleNamespace(
                        family=socket.AF_INET,
                        address="172.17.0.1",
                        netmask="255.255.0.0",
                        ptp=None,
                    )
                ],
            },
        )

        with patch.dict(sys.modules, {"psutil": fake}):
            scopes = local_ipv4_scopes()

        self.assertEqual(
            scopes,
            [
                DiscoveryScope(
                    interface="en0",
                    local_address="192.168.1.103",
                    network="192.168.1.0/24",
                )
            ],
        )

    def test_large_physical_network_is_bounded_to_local_24(self) -> None:
        fake = SimpleNamespace(
            net_if_stats=lambda: {"Ethernet": SimpleNamespace(isup=True)},
            net_if_addrs=lambda: {
                "Ethernet": [
                    SimpleNamespace(
                        family=socket.AF_INET,
                        address="10.42.7.11",
                        netmask="255.255.0.0",
                        ptp=None,
                    )
                ]
            },
        )

        with patch.dict(sys.modules, {"psutil": fake}):
            scopes = local_ipv4_scopes()

        self.assertEqual(scopes[0].network, "10.42.7.0/24")
        self.assertTrue(scopes[0].truncated)


class DiscoveryTests(unittest.TestCase):
    def test_operator_discovers_carla_by_default(self) -> None:
        self.assertEqual(parse_args([]).carla_host, "auto")

    def test_world_worker_auto_tracks_the_discovered_carla_host(self) -> None:
        self.assertEqual(
            resolve_world_worker_url("auto", "192.168.1.101", 8766),
            "http://192.168.1.101:8766",
        )
        self.assertIsNone(resolve_world_worker_url(None, "192.168.1.101", 8766))

    def test_validates_open_port_with_carla_rpc_before_reporting_server(self) -> None:
        class FakeRpc:
            def __init__(self, host: str, port: int, timeout: float) -> None:
                self.host = host
                self.port = port
                self.timeout = timeout

            def __enter__(self) -> "FakeRpc":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def value_call(self, method: str):
                if method == "version":
                    return "0.9.16"
                if method == "get_map_info":
                    return ["Carla/Maps/Town10HD_Opt", []]
                raise AssertionError(method)

        calls: list[str] = []

        def tcp_probe(host: str, port: int, timeout: float) -> bool:
            self.assertEqual(port, 2000)
            self.assertGreater(timeout, 0)
            calls.append(host)
            return host == "192.168.1.2"

        result = discover_carla_servers(
            scopes=[DiscoveryScope("en0", "192.168.1.1", "192.168.1.0/30")],
            tcp_probe=tcp_probe,
            rpc_factory=FakeRpc,
        )

        self.assertEqual(calls, ["192.168.1.1", "192.168.1.2"])
        self.assertEqual(result["scanned_host_count"], 2)
        self.assertEqual(result["tcp_candidate_count"], 1)
        self.assertEqual(
            result["servers"],
            [
                {
                    "host": "192.168.1.2",
                    "port": 2000,
                    "version": "0.9.16",
                    "map": "Town10HD_Opt",
                }
            ],
        )

    def test_open_non_carla_service_is_not_reported(self) -> None:
        class RejectingRpc:
            def __init__(self, *_: object, **__: object) -> None:
                raise RuntimeError("not CARLA")

        result = discover_carla_servers(
            scopes=[DiscoveryScope("en0", "192.168.1.1", "192.168.1.0/30")],
            tcp_probe=lambda *_: True,
            rpc_factory=RejectingRpc,
        )

        self.assertEqual(result["tcp_candidate_count"], 2)
        self.assertEqual(result["servers"], [])

    def test_auto_host_requires_exactly_one_discovered_server(self) -> None:
        one = {
            "scopes": [{"network": "192.168.1.0/24"}],
            "servers": [{"host": "192.168.1.101"}],
        }
        with patch("carla_vision.operator.server.discover_carla_servers", return_value=one):
            self.assertEqual(resolve_carla_host("auto", 2000), "192.168.1.101")

        none = {"scopes": [{"network": "192.168.1.0/24"}], "servers": []}
        with (
            patch("carla_vision.operator.server.discover_carla_servers", return_value=none),
            self.assertRaisesRegex(RuntimeError, "no CARLA server"),
        ):
            resolve_carla_host("AUTO", 2000)

        many = {
            "scopes": [{"network": "192.168.1.0/24"}],
            "servers": [{"host": "192.168.1.10"}, {"host": "192.168.1.20"}],
        }
        with (
            patch("carla_vision.operator.server.discover_carla_servers", return_value=many),
            self.assertRaisesRegex(RuntimeError, "multiple CARLA servers"),
        ):
            resolve_carla_host("auto", 2000)

    def test_operator_discovery_route_is_token_gated_and_has_no_scan_parameters(self) -> None:
        found = {
            "schema_version": "1.0",
            "port": 2000,
            "scopes": [
                {
                    "interface": "en0",
                    "local_address": "192.168.1.103",
                    "network": "192.168.1.0/24",
                    "truncated": False,
                }
            ],
            "scanned_host_count": 254,
            "tcp_candidate_count": 1,
            "servers": [
                {
                    "host": "192.168.1.101",
                    "port": 2000,
                    "version": "0.9.16",
                    "map": "Town10HD_Opt",
                }
            ],
            "elapsed_seconds": 0.4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = create_server(
                workspace=root,
                port=0,
                carla_host="172.20.10.7",
                carla_port=2000,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address[:2]
            base = f"http://{host}:{port}"
            try:
                with urllib.request.urlopen(f"{base}/api/bootstrap", timeout=5) as response:
                    token = json.loads(response.read())["token"]
                unauthorized = urllib.request.Request(
                    f"{base}/api/discovery/carla",
                    data=b"{}",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(unauthorized, timeout=5)
                self.assertEqual(context.exception.code, 403)

                with patch(
                    "carla_vision.operator.server.discover_carla_servers",
                    return_value=found,
                ):
                    authorized = urllib.request.Request(
                        f"{base}/api/discovery/carla",
                        data=b"{}",
                        method="POST",
                        headers={
                            "Content-Type": "application/json",
                            "X-Operator-Token": token,
                        },
                    )
                    with urllib.request.urlopen(authorized, timeout=5) as response:
                        payload = json.loads(response.read())
                self.assertEqual(payload["servers"][0]["host"], "192.168.1.101")
                self.assertFalse(payload["configured_match"])

                with patch(
                    "carla_vision.operator.server.discover_carla_servers",
                    return_value=found,
                ):
                    rejected = urllib.request.Request(
                        f"{base}/api/discovery/carla",
                        data=b'{"subnet":"10.0.0.0/8"}',
                        method="POST",
                        headers={
                            "Content-Type": "application/json",
                            "X-Operator-Token": token,
                        },
                    )
                    with self.assertRaises(urllib.error.HTTPError) as context:
                        urllib.request.urlopen(rejected, timeout=5)
                self.assertEqual(context.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                server.application.jobs.shutdown()
                thread.join(timeout=5)

    def test_game_shell_exposes_one_bounded_discovery_action(self) -> None:
        static = Path(__file__).parents[1] / "carla_vision" / "operator" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")

        self.assertEqual(html.count('id="carla-discover"'), 1)
        self.assertEqual(script.count('request("/api/discovery/carla"'), 1)
        self.assertNotIn("carla-discovery-subnet", html)


if __name__ == "__main__":
    unittest.main()
