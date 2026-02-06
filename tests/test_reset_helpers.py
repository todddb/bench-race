from __future__ import annotations

import agent.reset_helpers as reset_helpers


def test_is_sudo_noninteractive_error():
    assert reset_helpers.is_sudo_noninteractive_error("sudo: a password is required")
    assert reset_helpers.is_sudo_noninteractive_error("sudo: no tty present and no askpass program specified")
    assert reset_helpers.is_sudo_noninteractive_error("sudo: a terminal is required to read the password")
    assert not reset_helpers.is_sudo_noninteractive_error("some other error")
