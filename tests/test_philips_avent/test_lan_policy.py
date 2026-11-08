"""Tests for the LAN keep-alive and reconnect policy (issue #62)."""

import pytest
from lan_policy import (
    DATA_TIMEOUT,
    FIXED_HOST_WARN_AFTER,
    HEARTBEAT_INTERVAL,
    LAN_ERR_CONNECT,
    LAN_ERR_OFFLINE,
    LAN_ERR_PAYLOAD,
    MAX_PAYLOAD_ERRORS,
    PROTOCOL_VERSION_DEFAULT,
    RECONNECT_DELAY,
    data_stale,
    fixed_host_failure_warns,
    fixed_host_for_entry,
    heartbeat_due,
    is_connection_error,
    normalize_lan_host,
    parse_protocol_version,
    reconnect_delay,
    should_reconnect,
    version_candidates,
)

# tinytuya blocks in receive() for up to this long, so a heartbeat can be late
# by that much. Mirrors SOCKET_TIMEOUT in lan.py.
SOCKET_TIMEOUT = 5

# The monitor closes an idle socket after this long at the earliest.
DEVICE_IDLE_CLOSE = 30


class TestHeartbeatInterval:
    def test_ping_lands_well_inside_the_device_idle_window(self):
        # Worst case: a heartbeat becomes due right after a receive started.
        assert HEARTBEAT_INTERVAL + SOCKET_TIMEOUT < DEVICE_IDLE_CLOSE

    def test_due_only_after_the_interval(self):
        assert not heartbeat_due(now=100.0, last_heartbeat=100.0)
        assert not heartbeat_due(now=100.0 + HEARTBEAT_INTERVAL - 0.1, last_heartbeat=100.0)
        assert heartbeat_due(now=100.0 + HEARTBEAT_INTERVAL, last_heartbeat=100.0)
        assert heartbeat_due(now=200.0, last_heartbeat=100.0)

    def test_interval_is_overridable(self):
        assert heartbeat_due(now=5.0, last_heartbeat=0.0, interval=4.0)
        assert not heartbeat_due(now=5.0, last_heartbeat=0.0, interval=6.0)


class TestDataWatchdog:
    def test_stale_only_after_the_timeout(self):
        assert not data_stale(now=100.0, last_data=100.0)
        assert not data_stale(now=100.0 + DATA_TIMEOUT, last_data=100.0)
        assert data_stale(now=100.0 + DATA_TIMEOUT + 0.1, last_data=100.0)

    def test_answered_heartbeats_keep_the_watchdog_quiet(self):
        # Heartbeats refresh last_data, so on a healthy connection the watchdog
        # never fires even when the monitor pushes no DPS updates.
        last_data = 0.0
        for tick in range(1, 500):
            now = tick * HEARTBEAT_INTERVAL
            assert not data_stale(now, last_data)
            last_data = now


class TestReconnectDelay:
    def test_first_failure_retries_immediately(self):
        # A device that just closed the socket is usually ready again at once,
        # and waiting keeps the camera on cloud polling for longer than needed.
        assert reconnect_delay(1) == 0.0

    def test_later_failures_back_off(self):
        assert reconnect_delay(2) == RECONNECT_DELAY
        assert reconnect_delay(7) == RECONNECT_DELAY

    def test_zero_or_negative_is_treated_as_first(self):
        assert reconnect_delay(0) == 0.0


class TestErrorClassification:
    def test_connection_codes(self):
        assert is_connection_error(LAN_ERR_CONNECT)
        assert is_connection_error(LAN_ERR_OFFLINE)

    def test_payload_code_is_not_a_connection_error(self):
        # "Unexpected Payload from Device" (904) is what the monitor emitted
        # every 30-45s in issue #62; it says nothing about the socket.
        assert not is_connection_error(LAN_ERR_PAYLOAD)

    def test_missing_or_unknown_codes(self):
        assert not is_connection_error(None)
        assert not is_connection_error("")
        assert not is_connection_error("999")

    def test_codes_accepted_as_int_or_str(self):
        assert is_connection_error(int(LAN_ERR_OFFLINE))


class TestShouldReconnect:
    def test_connection_error_reconnects_at_once(self):
        assert should_reconnect(LAN_ERR_OFFLINE, 1)
        assert should_reconnect(LAN_ERR_CONNECT, 1)

    def test_single_unreadable_frame_does_not_reconnect(self):
        assert not should_reconnect(LAN_ERR_PAYLOAD, 1)

    def test_repeated_unreadable_frames_do_reconnect(self):
        assert not should_reconnect(LAN_ERR_PAYLOAD, MAX_PAYLOAD_ERRORS - 1)
        assert should_reconnect(LAN_ERR_PAYLOAD, MAX_PAYLOAD_ERRORS)

    def test_error_without_a_code_is_treated_as_unreadable(self):
        assert not should_reconnect(None, 1)
        assert should_reconnect(None, MAX_PAYLOAD_ERRORS)


class TestProtocolVersion:
    """Which local protocol version to speak (#51, #62).

    The SCD953 announces 3.5 in its discovery broadcast while the client always
    built a 3.3 session, and a session at the wrong version cannot read the
    frames the camera sends.
    """

    def test_announced_version_is_read(self):
        assert parse_protocol_version("3.5") == 3.5
        assert parse_protocol_version("3.3") == 3.3
        assert parse_protocol_version(3.4) == 3.4

    def test_unknown_or_junk_values_fall_back(self):
        for value in ("", "  ", "junk", "9.9", None, True, [], {}):
            assert parse_protocol_version(value) is None

    def test_announced_version_is_tried_first(self):
        assert version_candidates(3.5) == [3.5, PROTOCOL_VERSION_DEFAULT]

    def test_default_alone_when_nothing_announced(self):
        assert version_candidates(None) == [PROTOCOL_VERSION_DEFAULT]

    def test_no_duplicate_when_the_camera_announces_the_default(self):
        assert version_candidates(PROTOCOL_VERSION_DEFAULT) == [PROTOCOL_VERSION_DEFAULT]

    def test_fallback_is_always_available(self):
        # Local control must not regress on firmware that refuses the newer
        # session-key negotiation.
        for announced in (3.1, 3.4, 3.5, None):
            assert PROTOCOL_VERSION_DEFAULT in version_candidates(announced)


class TestFixedLanHost:
    """A configured LAN address replaces the broadcast scan, for monitors on another VLAN."""

    def test_blank_means_discovery(self):
        for value in (None, "", "   "):
            assert normalize_lan_host(value) is None

    def test_address_is_trimmed(self):
        assert normalize_lan_host(" 10.0.2.11 ") == "10.0.2.11"
        assert normalize_lan_host("babyfoon.lan") == "babyfoon.lan"

    def test_ipv6_is_accepted(self):
        assert normalize_lan_host("fd00::11") == "fd00::11"

    def test_urls_and_junk_are_rejected(self):
        for value in ("http://10.0.2.11", "10.0.2.11/24", "10.0.2 .11", "10.0.2.11,10.0.2.12", 11, ["10.0.2.11"]):
            with pytest.raises(ValueError):
                normalize_lan_host(value)

    def test_applies_to_an_entry_with_one_monitor(self):
        assert fixed_host_for_entry("10.0.2.11", 1) == "10.0.2.11"

    def test_ignored_with_several_monitors(self):
        # One address cannot serve two monitors; they keep scanning.
        assert fixed_host_for_entry("10.0.2.11", 2) is None

    def test_invalid_stored_value_falls_back_to_discovery(self):
        assert fixed_host_for_entry("http://10.0.2.11", 1) is None
        assert fixed_host_for_entry(None, 1) is None

    def test_early_failures_stay_quiet(self):
        for failures in range(1, FIXED_HOST_WARN_AFTER):
            assert not fixed_host_failure_warns(failures)

    def test_one_warning_per_outage(self):
        assert fixed_host_failure_warns(FIXED_HOST_WARN_AFTER)
        for failures in range(FIXED_HOST_WARN_AFTER + 1, FIXED_HOST_WARN_AFTER + 50):
            assert not fixed_host_failure_warns(failures)
