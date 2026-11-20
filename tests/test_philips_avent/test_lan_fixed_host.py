"""TuyaLANClient with a fixed LAN address: no scan, keyed protocols first, one warning per outage."""
from __future__ import annotations

import asyncio
import logging

import pytest

from .ha_stand_in import home_assistant_installed, load

pytestmark = pytest.mark.skipif(home_assistant_installed(), reason="uses a stand-in for Home Assistant")

HOST = "10.0.2.11"


class FakeHass:
    async def async_add_executor_job(self, func, *args):
        return func(*args)


class FakeDevice:
    def __init__(self, ip, version):
        self.ip, self.version = ip, version

    def close(self):
        pass


@pytest.fixture
def lan(monkeypatch):
    return load(monkeypatch, "lan")


def make_client(lan, monkeypatch, *, accept, host=HOST, last_ip=None, last_protocol=""):
    """A client whose direct connects follow `accept(ip, version)`, and whose scan must not run."""
    published: list[str] = []
    client = lan.TuyaLANClient(
        FakeHass(), "dev", "key", lambda dps, baseline=False: None,
        published.append, last_ip, last_protocol, host=host,
    )
    client.attempts = []
    client.published = published

    def try_direct(ip, version):
        client.attempts.append((ip, version))
        return FakeDevice(ip, version) if accept(ip, version) else None

    async def discover():
        raise AssertionError("a fixed LAN address must never trigger a scan")

    monkeypatch.setattr(client, "_try_direct_connect", try_direct)
    monkeypatch.setattr(client, "_discover_device", discover)
    return client


def connect(client):
    return asyncio.run(client._connect())


class TestFixedHostConnect:
    def test_a_3_5_monitor_gets_a_keyed_session_and_its_address_reaches_the_bridge(self, lan, monkeypatch):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: v == 3.5)
        assert connect(client)
        assert client.attempts == [(HOST, 3.5)]
        assert client.session_version == 3.5
        assert client.published == [HOST]
        assert client.protocol == "3.5"

    def test_a_3_3_monitor_is_reached_after_the_keyed_versions_fail(self, lan, monkeypatch):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: v == 3.3)
        assert connect(client)
        assert client.attempts == [(HOST, 3.5), (HOST, 3.4), (HOST, 3.3)]
        assert client.session_version == 3.3

    def test_a_3_3_fallback_does_not_pin_later_reconnects(self, lan, monkeypatch):
        works = {"3.5": False}
        client = make_client(lan, monkeypatch, accept=lambda ip, v: v == 3.3 or works["3.5"])
        assert connect(client)
        client._disconnect()
        works["3.5"] = True
        client.attempts.clear()
        assert connect(client)
        assert client.attempts == [(HOST, 3.5)]

    def test_a_keyed_seed_at_the_same_address_is_tried_first(self, lan, monkeypatch):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: v == 3.4, last_ip=HOST, last_protocol="3.4")
        assert connect(client)
        assert client.attempts == [(HOST, 3.4)]
        # Seeded with the same pair, so the bridge config is not rewritten.
        assert client.published == []

    def test_a_seed_from_another_address_is_ignored(self, lan, monkeypatch):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: v == 3.5, last_ip="10.0.2.99", last_protocol="3.4")
        assert client._ip == HOST
        assert connect(client)
        assert client.attempts == [(HOST, 3.5)]
        assert client.published == [HOST]

    def test_failure_keeps_the_address_and_never_scans(self, lan, monkeypatch):
        works = {"now": False}
        client = make_client(lan, monkeypatch, accept=lambda ip, v: works["now"] and v == 3.5)
        assert not connect(client)
        assert not connect(client)
        assert client._ip == HOST
        works["now"] = True
        assert connect(client)
        assert {ip for ip, _ in client.attempts} == {HOST}

    def test_commands_can_use_the_address_before_the_first_connect(self, lan, monkeypatch):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: False)
        assert client._ip == HOST

    def test_warns_once_after_repeated_failures(self, lan, monkeypatch, caplog):
        client = make_client(lan, monkeypatch, accept=lambda ip, v: False)
        with caplog.at_level(logging.DEBUG, logger=lan.__name__):
            for _ in range(lan.FIXED_HOST_WARN_AFTER * 3):
                connect(client)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert HOST in warnings[0].getMessage()

    def test_recovery_after_a_warning_is_logged_and_resets_the_count(self, lan, monkeypatch, caplog):
        works = {"now": False}
        client = make_client(lan, monkeypatch, accept=lambda ip, v: works["now"] and v == 3.5)
        failures = lan.FIXED_HOST_WARN_AFTER
        with caplog.at_level(logging.DEBUG, logger=lan.__name__):
            for _ in range(failures):
                connect(client)
            works["now"] = True
            assert connect(client)
            assert any("restored" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
            client._disconnect()
            works["now"] = False
            for _ in range(failures):
                connect(client)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2


class TestWithoutFixedHost:
    def test_scans_when_no_address_is_set(self, lan, monkeypatch):
        client = lan.TuyaLANClient(FakeHass(), "dev", "key", lambda dps, baseline=False: None)
        scanned = []

        async def discover():
            scanned.append(True)
            return None, None

        monkeypatch.setattr(client, "_discover_device", discover)
        assert not connect(client)
        assert scanned == [True]
