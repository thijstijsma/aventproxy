"""Timing policy for the persistent LAN connection.

Kept free of Home Assistant imports so the decisions can be unit-tested without
an HA runtime; `lan.py` owns the sockets and calls into here.
"""
from __future__ import annotations

# The monitor closes an idle LAN socket after roughly 30-45 seconds, which shows
# up as "Unexpected Payload from Device" followed by a reconnect every ~30s
# (issue #62). tinytuya leaves keep-alive to the caller on persistent sockets, so
# a ping well inside that window keeps the session up. Receives block for up to
# SOCKET_TIMEOUT seconds, so the effective ping interval is this value plus that.
HEARTBEAT_INTERVAL = 8.0

# Force a reconnect when nothing at all has been heard for this long. A working
# connection answers every heartbeat, so this only fires on a wedged socket.
DATA_TIMEOUT = 600.0

# Delay before retrying a connection, from the second consecutive failure on.
RECONNECT_DELAY = 15.0

# tinytuya error codes (see tinytuya.core.error_helper). Only the connection
# ones mean the socket is gone; 904 says a frame could not be parsed, which the
# monitor also produces for stray keep-alive acks, and dropping the connection
# over one of those is what kept the camera entity churning (issue #62).
LAN_ERR_CONNECT = "901"
LAN_ERR_PAYLOAD = "904"
LAN_ERR_OFFLINE = "905"
CONNECTION_ERRORS = frozenset({LAN_ERR_CONNECT, LAN_ERR_OFFLINE})

# Reconnect once this many unparseable frames arrive in a row.
MAX_PAYLOAD_ERRORS = 3


def is_connection_error(err_code: str | None) -> bool:
    """True when a tinytuya error code means the LAN socket is unusable."""
    return str(err_code or "") in CONNECTION_ERRORS


def should_reconnect(err_code: str | None, consecutive_payload_errors: int) -> bool:
    """Whether a device error frame justifies dropping the connection.

    A connection error always does. An unparseable frame only does once several
    arrive in a row, so a single odd frame no longer costs the user a reconnect
    cycle and, with it, a spell of the camera showing as unavailable.
    """
    if is_connection_error(err_code):
        return True
    return consecutive_payload_errors >= MAX_PAYLOAD_ERRORS


def heartbeat_due(now: float, last_heartbeat: float, interval: float = HEARTBEAT_INTERVAL) -> bool:
    """True when the next keep-alive ping is due."""
    return now - last_heartbeat >= interval


def data_stale(now: float, last_data: float, timeout: float = DATA_TIMEOUT) -> bool:
    """True when nothing has been received for `timeout` seconds."""
    return now - last_data > timeout


def reconnect_delay(consecutive_failures: int, delay: float = RECONNECT_DELAY) -> float:
    """Seconds to wait before the next connection attempt.

    The first failure retries immediately: a device that closed the socket is
    usually ready again straight away, and waiting only widens the window where
    the integration falls back to cloud polling. Repeated failures back off so a
    monitor that is really gone is not hammered.
    """
    if consecutive_failures <= 1:
        return 0.0
    return delay

# Protocol version for the local Tuya session. 3.3 is what the integration has
# always used and what every model connects with today. The SCD953 announces 3.5
# in its discovery broadcast (capture on #51), and a session negotiated at the
# wrong version cannot read frames the camera sends, which is the likely source
# of the "Unexpected Payload from Device" frames on #62. Try what the camera
# announces, keep 3.3 as the fallback so local control cannot regress.
PROTOCOL_VERSION_DEFAULT = 3.3
SUPPORTED_PROTOCOL_VERSIONS = (3.1, 3.2, 3.3, 3.4, 3.5)


def parse_protocol_version(value: object) -> float | None:
    """Read the protocol version out of a discovery broadcast.

    tinytuya reports it as a string such as "3.5". Anything unrecognised gives
    None, so the caller falls back to the default rather than handing tinytuya a
    version it cannot speak.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        version = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            version = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return version if version in SUPPORTED_PROTOCOL_VERSIONS else None


def version_candidates(
    announced: float | None, default: float = PROTOCOL_VERSION_DEFAULT
) -> list[float]:
    """Protocol versions to try for a local session, best guess first.

    The announced version comes first because the camera should know what it
    speaks. The default follows as a fallback, since 3.4 and up negotiate a
    session key and a firmware that refuses can still be reached the old way.
    """
    candidates = []
    if announced is not None:
        candidates.append(announced)
    if default not in candidates:
        candidates.append(default)
    return candidates

# A fixed LAN address replaces discovery for networks where the Tuya broadcast
# never reaches Home Assistant, such as a monitor on a separate VLAN. A failed
# connect to it is logged at debug level until this many fail in a row, so a
# monitor that is merely switched off for the night does not fill the log.
FIXED_HOST_WARN_AFTER = 5


def normalize_lan_host(value: object) -> str | None:
    """The LAN address from the options, or None when discovery should be used.

    Blank means "scan as before". Anything with whitespace, a comma, a slash
    or a scheme is rejected with ValueError, since tinytuya wants one bare IP
    address or host name and would otherwise fail on every connect.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("LAN address must be text")
    host = value.strip()
    if not host:
        return None
    if any(ch.isspace() for ch in host) or "/" in host or "," in host:
        raise ValueError("LAN address must be one bare IP address or host name")
    return host


def fixed_host_failure_warns(consecutive_failures: int, threshold: int = FIXED_HOST_WARN_AFTER) -> bool:
    """True for the one failure, in a row of them, that deserves a warning.

    Earlier failures stay at debug level and later ones repeat nothing, so a
    long outage costs one warning rather than one every reconnect cycle.
    """
    return consecutive_failures == threshold


def fixed_host_for_entry(value: object, camera_count: int) -> str | None:
    """The fixed LAN address to use, given how many monitors the entry holds.

    The option holds one address, so it only applies to an account with a
    single monitor; with more, each would be sent to the same address and all
    but one would fail, so they keep finding themselves by broadcast instead.
    An invalid stored value also falls back to discovery.
    """
    if camera_count != 1:
        return None
    try:
        return normalize_lan_host(value)
    except ValueError:
        return None
