"""Read-only discovery of CARLA servers on directly connected IPv4 LANs."""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from .bridge import CarlaRpc

_VIRTUAL_INTERFACE_PREFIXES = (
    "anpi",
    "awdl",
    "bridge",
    "docker",
    "ham",
    "llw",
    "tap",
    "tailscale",
    "tun",
    "utun",
    "veth",
    "vmnet",
    "wg",
    "zt",
)
_MAX_SCAN_ADDRESSES = 1024


@dataclass(frozen=True)
class DiscoveryScope:
    interface: str
    local_address: str
    network: str
    truncated: bool = False


@dataclass(frozen=True)
class DiscoveredCarlaServer:
    host: str
    port: int
    version: str
    map: str


def _is_virtual_interface(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized.startswith(_VIRTUAL_INTERFACE_PREFIXES)


def local_ipv4_scopes(*, max_addresses: int = 256) -> list[DiscoveryScope]:
    """Return bounded RFC1918 scopes from active, non-tunnel interfaces.

    Large local networks are deliberately reduced to the /24 containing this
    machine. Discovery is intended to find a nearby simulator, not to be a
    general-purpose network scanner.
    """

    if not 2 <= max_addresses <= _MAX_SCAN_ADDRESSES:
        raise ValueError(f"max_addresses must be in [2, {_MAX_SCAN_ADDRESSES}]")

    try:
        import psutil
    except ImportError as error:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("CARLA discovery requires the psutil package") from error

    stats = psutil.net_if_stats()
    scopes: dict[tuple[str, str], DiscoveryScope] = {}
    for interface, addresses in psutil.net_if_addrs().items():
        if _is_virtual_interface(interface) or not stats.get(interface, None):
            continue
        if not stats[interface].isup:
            continue
        for address in addresses:
            if address.family != socket.AF_INET or address.netmask is None:
                continue
            if getattr(address, "ptp", None):
                continue
            try:
                local = ipaddress.IPv4Address(address.address)
                network = ipaddress.IPv4Network(
                    f"{address.address}/{address.netmask}",
                    strict=False,
                )
            except (ipaddress.AddressValueError, ipaddress.NetmaskValueError):
                continue
            if not local.is_private or local.is_loopback or local.is_link_local:
                continue
            truncated = network.num_addresses > max_addresses
            if truncated:
                network = ipaddress.IPv4Network(f"{local}/24", strict=False)
            scope = DiscoveryScope(
                interface=interface,
                local_address=str(local),
                network=str(network),
                truncated=truncated,
            )
            scopes[(scope.interface, scope.network)] = scope
    return sorted(
        scopes.values(),
        key=lambda item: (ipaddress.IPv4Network(item.network).network_address, item.interface),
    )


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _scan_hosts(scopes: Iterable[DiscoveryScope]) -> list[str]:
    hosts: set[ipaddress.IPv4Address] = set()
    for scope in scopes:
        network = ipaddress.IPv4Network(scope.network, strict=True)
        if network.num_addresses > _MAX_SCAN_ADDRESSES:
            raise ValueError(f"discovery scope {network} exceeds the safe scan limit")
        hosts.update(network.hosts())
    return [str(host) for host in sorted(hosts)]


def discover_carla_servers(
    *,
    port: int = 2000,
    scopes: Sequence[DiscoveryScope] | None = None,
    connect_timeout: float = 0.25,
    rpc_timeout: float = 1.5,
    max_workers: int = 64,
    tcp_probe: Callable[[str, int, float], bool] = _tcp_open,
    rpc_factory: Callable[..., Any] = CarlaRpc,
) -> dict[str, Any]:
    """Find and validate CARLA RPC servers without mutating their worlds."""

    if isinstance(port, bool) or not 1 <= int(port) <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if not 0.02 <= float(connect_timeout) <= 5.0:
        raise ValueError("connect_timeout must be in [0.02, 5.0]")
    if not 0.1 <= float(rpc_timeout) <= 10.0:
        raise ValueError("rpc_timeout must be in [0.1, 10.0]")
    if isinstance(max_workers, bool) or not 1 <= int(max_workers) <= 128:
        raise ValueError("max_workers must be in [1, 128]")

    started = time.monotonic()
    resolved_scopes = list(scopes) if scopes is not None else local_ipv4_scopes()
    hosts = _scan_hosts(resolved_scopes)
    if not hosts:
        return {
            "schema_version": "1.0",
            "port": int(port),
            "scopes": [asdict(scope) for scope in resolved_scopes],
            "scanned_host_count": 0,
            "tcp_candidate_count": 0,
            "servers": [],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(hosts))) as pool:
        open_flags = pool.map(
            lambda host: tcp_probe(host, int(port), float(connect_timeout)),
            hosts,
        )
        candidates = [host for host, opened in zip(hosts, open_flags, strict=True) if opened]

    def validate(host: str) -> DiscoveredCarlaServer | None:
        try:
            with rpc_factory(host, int(port), timeout=float(rpc_timeout)) as rpc:
                version = str(rpc.value_call("version"))
                map_info = rpc.value_call("get_map_info")
                if not isinstance(map_info, list) or not map_info:
                    return None
                map_name = str(map_info[0]).rsplit("/", 1)[-1]
                if not version or not map_name:
                    return None
                return DiscoveredCarlaServer(
                    host=host,
                    port=int(port),
                    version=version,
                    map=map_name,
                )
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(int(max_workers), max(1, len(candidates)))) as pool:
        validated = list(pool.map(validate, candidates))
    servers = sorted(
        (server for server in validated if server is not None),
        key=lambda item: ipaddress.IPv4Address(item.host),
    )
    return {
        "schema_version": "1.0",
        "port": int(port),
        "scopes": [asdict(scope) for scope in resolved_scopes],
        "scanned_host_count": len(hosts),
        "tcp_candidate_count": len(candidates),
        "servers": [asdict(server) for server in servers],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and validate CARLA servers on directly connected private LANs"
    )
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--connect-timeout", type=float, default=0.25)
    parser.add_argument("--rpc-timeout", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = discover_carla_servers(
        port=args.port,
        connect_timeout=args.connect_timeout,
        rpc_timeout=args.rpc_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["servers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
