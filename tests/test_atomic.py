"""Windows file sharing must preserve complete cache entries during replacement."""

import pytest

from akiki import _atomic


def sharing_error():
    error = PermissionError("The destination is temporarily open")
    error.winerror = 5
    return error


def test_windows_sharing_failures_retry_without_removing_the_valid_entry(tmp_path, monkeypatch):
    source, destination = tmp_path / "temporary", tmp_path / "cache"
    source.write_bytes(b"new complete data")
    destination.write_bytes(b"old complete data")
    original = _atomic.os.replace
    attempts, delays = [], []

    def replace(first, second):
        attempts.append((first, second))
        assert destination.read_bytes() == b"old complete data"
        if len(attempts) < 3:
            raise sharing_error()
        original(first, second)

    monkeypatch.setattr(_atomic.sys, "platform", "win32")
    monkeypatch.setattr(_atomic.os, "replace", replace)
    monkeypatch.setattr(_atomic.time, "sleep", delays.append)
    _atomic.replace_file(source, destination)
    assert len(attempts) == 3 and delays == [0.01, 0.02]
    assert destination.read_bytes() == b"new complete data"
    assert not source.exists()


def test_permanent_windows_failure_is_bounded_and_preserves_both_files(tmp_path, monkeypatch):
    source, destination = tmp_path / "temporary", tmp_path / "cache"
    source.write_bytes(b"new complete data")
    destination.write_bytes(b"old complete data")
    attempts, delays = [], []

    def replace(*paths):
        attempts.append(paths)
        raise sharing_error()

    monkeypatch.setattr(_atomic.sys, "platform", "win32")
    monkeypatch.setattr(_atomic.os, "replace", replace)
    monkeypatch.setattr(_atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        _atomic.replace_file(source, destination)
    assert len(attempts) == 9 and sum(delays) == pytest.approx(2.55)
    assert source.read_bytes() == b"new complete data"
    assert destination.read_bytes() == b"old complete data"


def test_other_platform_failures_raise_without_retry(tmp_path, monkeypatch):
    attempts, delays = [], []

    def replace(*paths):
        attempts.append(paths)
        raise PermissionError("The directory is not writable")

    monkeypatch.setattr(_atomic.sys, "platform", "linux")
    monkeypatch.setattr(_atomic.os, "replace", replace)
    monkeypatch.setattr(_atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="not writable"):
        _atomic.replace_file(tmp_path / "temporary", tmp_path / "cache")
    assert len(attempts) == 1 and not delays
