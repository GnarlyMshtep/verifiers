from __future__ import annotations

import pytest

from verifiers.envs._m_.utils import strict_format, write_once


def test_strict_format_substitutes_all():
    assert strict_format("hi {a} and {b}", a="X", b="Y") == "hi X and Y"


def test_strict_format_missing_key_raises():
    with pytest.raises(KeyError):
        strict_format("hi {a} {b}", a="X")


def test_strict_format_unused_kwarg_raises():
    with pytest.raises(ValueError):
        strict_format("hi {a}", a="X", b="unused")


def test_strict_format_repeated_field_ok():
    assert strict_format("{a}-{a}", a="X") == "X-X"


def test_write_once_reads():
    d = write_once(x="one", y="two")
    assert d["x"] == "one" and d["y"] == "two"
    assert set(d) == {"x", "y"}


def test_write_once_reassignment_raises():
    d = write_once(x="one")
    with pytest.raises(KeyError):
        d["x"] = "two"
    with pytest.raises(KeyError):
        d["new"] = "three"
