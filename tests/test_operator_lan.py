from __future__ import annotations

import pytest

from carla_vision.operator.garage_server import _operator_bind_scope


@pytest.mark.parametrize(
    ("bind", "scope"),
    [
        ("127.0.0.1", "loopback"),
        ("::1", "loopback"),
        ("localhost", "loopback"),
        ("0.0.0.0", "lan"),
        ("::", "lan"),
        ("10.0.0.42", "lan"),
        ("172.20.10.5", "lan"),
        ("192.168.1.25", "lan"),
        ("169.254.10.10", "lan"),
        ("fd12:3456::20", "lan"),
        ("fe80::1234", "lan"),
        ("8.8.8.8", "unsupported"),
        ("1.1.1.1", "unsupported"),
        ("example.com", "unsupported"),
        ("", "unsupported"),
    ],
)
def test_operator_bind_scope_is_lan_only(bind: str, scope: str) -> None:
    assert _operator_bind_scope(bind) == scope
