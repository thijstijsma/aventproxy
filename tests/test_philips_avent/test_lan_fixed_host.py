"""Tests for TuyaLANClient with a fixed LAN address.

lan.py imports Home Assistant and tinytuya, neither of which CI installs, so
both are stubbed and the module is loaded inside a bare package that skips
the integration's __init__.py.
"""

import asyncio
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).parent.parent.parent / "custom_components" / "philips_avent"
_PKG = "_philips_avent_lan_under_test"


@pytest.fixture
def lan(monkeypatch):
    if "tinytuya" not in sys.modules:
        monkeypatch.setitem(sys.modules, "tinytuya", types.ModuleType("tinytuya"))
    if "homeassistant.core" not in sys.modules:
        ha = types.ModuleType("homeassistant")
        core = types.ModuleType("homeassistant.core")
        core.HomeAssistant = object
        ha.core = core
        monkeypatch.setitem(sys.modules, "homeassistant", ha)
        monkeypatch.setitem(sys.modules, "homeassistant.core", core)

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    monkeypatch.setitem(sys.modules, _PKG, pkg)
    module = importlib.import_module(f"{_PKG}.lan")
    yield module
    for name in (f"{_PKG}.lan", f"{_PKG}.lan_policy"):
        sys.modules.pop(name, None)


class FakeDevice:
    def close(self):
        pass


def make_client(lan, host, answers):
    """A client whose direct connects return the next of `answers`, and whose scan must not run."""
    client = lan.TuyaLANClient(None, "dev123", "key", lambda dps: None, host=host)
    client.connect_calls = []

    async def connect_at_any_version(ip):
        client.connect_calls.append(ip)
        return answers.pop(0)

    async def discover_device():
        raise AssertionError("a fixed LAN address must never trigger a scan")

    client._connect_at_any_version = connect_at_any_version
    client._discover_device = discover_device
    return client


def run(coro):
    return asyncio.run(coro)


class TestFixedHostConnect:
    def test_connects_to_the_configured_address(self, lan):
        client = make_client(lan, "10.0.2.11", [FakeDevice()])

        assert run(client._connect()) is True
        assert client.connected
        assert client.connect_calls == ["10.0.2.11"]

    def test_failure_keeps_the_address_and_never_scans(self, lan):
        client = make_client(lan, "10.0.2.11", [None, None, FakeDevice()])

        assert run(client._connect()) is False
        assert run(client._connect()) is False
        assert client._ip == "10.0.2.11"
        assert run(client._connect()) is True
        assert client.connect_calls == ["10.0.2.11"] * 3

    def test_commands_can_use_the_address_before_the_first_connect(self, lan):
        client = make_client(lan, "10.0.2.11", [])
        assert client._ip == "10.0.2.11"

    def test_warns_once_after_repeated_failures(self, lan, caplog):
        attempts = lan.FIXED_HOST_WARN_AFTER * 3
        client = make_client(lan, "10.0.2.11", [None] * attempts)

        with caplog.at_level(logging.DEBUG, logger=lan.__name__):
            for _ in range(attempts):
                run(client._connect())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "10.0.2.11" in warnings[0].getMessage()

    def test_recovery_after_a_warning_is_logged_and_resets_the_count(self, lan, caplog):
        failures = lan.FIXED_HOST_WARN_AFTER
        client = make_client(lan, "10.0.2.11", [None] * failures + [FakeDevice()] + [None] * failures)

        with caplog.at_level(logging.DEBUG, logger=lan.__name__):
            for _ in range(failures + 1):
                run(client._connect())
            assert any("restored" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
            client._disconnect()
            for _ in range(failures):
                run(client._connect())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2


class TestWithoutFixedHost:
    def test_scans_when_no_address_is_set(self, lan):
        client = lan.TuyaLANClient(None, "dev123", "key", lambda dps: None)
        scanned = []

        async def discover_device():
            scanned.append(True)
            return None, None

        client._discover_device = discover_device

        assert run(client._connect()) is False
        assert scanned == [True]
