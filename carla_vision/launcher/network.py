"""Explicit LAN pairing and a scoped Windows firewall action for the TUI."""

from __future__ import annotations

import base64
import ipaddress
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import psutil

from .config import (
    CARLA_VERSION,
    LauncherError,
    atomic_json,
    find_carla,
    read_json,
    validate_worker_url,
    worker_token,
)
from .services import LocalStack


def lan_addresses() -> list[str]:
    addresses = set()
    for entries in psutil.net_if_addrs().values():
        for entry in entries:
            try:
                address = ipaddress.ip_address(entry.address)
            except ValueError:
                continue
            if (
                address.version == 4
                and address.is_private
                and not (address.is_loopback or address.is_unspecified or address.is_multicast)
            ):
                addresses.add(str(address))
    return sorted(addresses)


def export_connection(stack: LocalStack, destination: Path) -> None:
    config = stack.config
    if config.mode != "local" or not config.share_worker:
        raise LauncherError("Enable Worker sharing and save the host's LAN address first.")
    if config.worker_bind not in lan_addresses():
        raise LauncherError(
            "Saved Worker address is no longer on this computer. Select its current LAN address."
        )
    if destination.exists() or destination.is_symlink():
        raise LauncherError("Choose a new connection filename; existing files are preserved.")
    token = worker_token(stack.root, create=True)
    atomic_json(
        destination,
        {
            "schema_version": 1,
            "carla_version": CARLA_VERSION,
            "carla_host": config.worker_bind,
            "carla_port": config.carla_port,
            "worker_url": config.worker_url,
            "token": token,
        },
    )
    stack.report(
        "Private connection file saved. Transfer it to the other computer; never publish it."
    )


def import_connection(stack: LocalStack, source: Path) -> None:
    if source.stat().st_size > 16_384:
        raise LauncherError("Connection file is too large; export a fresh file from the host TUI.")
    payload = read_json(source, {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("carla_version") != CARLA_VERSION
    ):
        raise LauncherError("Connection file must be exported by the CARLA 0.9.16 launcher.")
    token = payload.get("token")
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise LauncherError("Connection file is missing a valid Worker token.")
    url = payload.get("worker_url")
    if not isinstance(url, str):
        raise LauncherError("Connection file is missing its Worker URL.")
    url = validate_worker_url(url)
    config = replace(
        stack.config,
        mode="web",
        carla_host=payload.get("carla_host"),
        carla_port=payload.get("carla_port"),
        remote_worker_url=url,
    )
    stack.configure(config, token=token)
    stack.report("Connection imported. Use Check connection, then Start web app.")


def windows_firewall(stack: LocalStack) -> None:
    """Only called by the explicit TUI button; Windows presents its UAC prompt."""
    config = stack.config
    if os.name != "nt" or config.mode != "local" or not config.share_worker:
        raise LauncherError(
            "Windows firewall setup requires local mode with Worker sharing enabled."
        )
    if config.worker_bind not in lan_addresses():
        raise LauncherError("Select and save this computer's current LAN address first.")
    program = sys.executable.replace("'", "''")
    # Ports and IPs have already been validated. Name includes the workspace to
    # avoid replacing another installation's rule. No Public-profile access.
    import hashlib

    name = "CARLA-Vision-Worker-" + hashlib.sha256(str(stack.root).encode()).hexdigest()[:12]
    script = (
        "$ErrorActionPreference='Stop'; "
        f"Get-NetFirewallRule -Name '{name}' -ErrorAction SilentlyContinue | Remove-NetFirewallRule; "
        f"New-NetFirewallRule -Name '{name}' -DisplayName 'CARLA Vision LAN Worker' "
        f"-Direction Inbound -Action Allow -Protocol TCP -LocalPort {config.worker_port} "
        f"-LocalAddress '{config.worker_bind}' -Program '{program}' "
        "-Profile Private -RemoteAddress LocalSubnet | Out-Null"
    )
    executable = find_carla(stack.root, config.carla_root)
    if executable is not None:
        server = executable.parent / "CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"
        if server.is_file():
            server_program = str(server).replace("'", "''")
            carla_name = name.replace("Worker", "CARLA")
            script += (
                f"; Get-NetFirewallRule -Name '{carla_name}' -ErrorAction SilentlyContinue | Remove-NetFirewallRule; "
                f"New-NetFirewallRule -Name '{carla_name}' -DisplayName 'CARLA Vision LAN Simulator' "
                f"-Direction Inbound -Action Allow -Protocol TCP -LocalPort {config.carla_port},{config.carla_port + 1},{config.carla_port + 2} "
                f"-LocalAddress '{config.worker_bind}' -Program '{server_program}' "
                "-Profile Private -RemoteAddress LocalSubnet | Out-Null"
            )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    request = (
        "$ErrorActionPreference='Stop'; "
        "$p=Start-Process powershell.exe -Verb RunAs -Wait -PassThru "
        f"-ArgumentList '-NoProfile -EncodedCommand {encoded}'; exit $p.ExitCode"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", request],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise LauncherError(
            "Firewall setup failed or UAC was cancelled. Retry on the Worker host with a Private network profile."
        )
    stack.report(
        "Windows firewall allows the configured Worker and installed simulator ports on Private networks from the local subnet only."
    )


def windows_prerequisites(stack: LocalStack) -> None:
    if os.name != "nt" or stack.config.mode != "local":
        raise LauncherError("CARLA runtime prerequisite installation is a Windows host action.")
    executable = find_carla(stack.root, stack.config.carla_root)
    if executable is None:
        raise LauncherError(
            "Select and save the installed CARLA folder, or download CARLA through Install / Repair first."
        )
    installer = executable.parent / "Engine/Extras/Redist/en-us/UE4PrereqSetup_x64.exe"
    if not installer.is_file() or installer.is_symlink():
        raise LauncherError(
            "This CARLA package has no bundled Unreal prerequisite installer. Use the official full Windows CARLA package."
        )
    escaped = str(installer).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$p=Start-Process -FilePath '{escaped}' -Verb RunAs -Wait -PassThru "
        "-ArgumentList '/quiet /norestart'; exit $p.ExitCode"
    )
    stack.report(
        "Installing the CARLA package's Windows runtime prerequisites. Accept the Windows UAC prompt."
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 3010:
        stack.report(
            "Windows runtime prerequisites installed. Restart Windows before starting CARLA."
        )
    elif result.returncode:
        raise LauncherError(
            "CARLA prerequisite installation failed or was cancelled. Inspect the Windows installer and retry."
        )
    else:
        stack.report("Windows runtime prerequisites are installed. Retry Install / Repair.")
