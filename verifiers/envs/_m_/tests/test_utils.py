from __future__ import annotations

import pytest

from verifiers.envs._m_.utils import strict_format, strip_think, write_once


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


def test_strict_format_conversion_spec_ok():
    assert strict_format("{x!r}", x="a") == "'a'"


def test_strict_format_format_spec_ok():
    assert strict_format("{x:>5}", x="a") == "    a"


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


def test_strip_think_removes_complete_block():
    assert strip_think("<think>reasoning HERE!</think>the answer") == "the answer"


def test_strip_think_preserves_surrounding_answer():
    assert strip_think("<think>X</think>hello world") == "hello world"
    assert strip_think("prefix <think>X</think> suffix") == "prefix suffix"


def test_strip_think_multiple_blocks():
    assert strip_think("<think>a</think>one<think>b</think>two") == "onetwo"


def test_strip_think_unclosed_drops_to_end():
    # Truncated reasoning with no closing tag -> everything after <think> is reasoning, no answer.
    assert strip_think("the answer<think>CAPS truncated reasoning") == "the answer"
    assert strip_think("<think>only reasoning, never closed") == ""


def test_strip_think_no_think_unchanged():
    assert strip_think("a plain lowercase answer") == "a plain lowercase answer"


def test_strip_think_empty():
    assert strip_think("") == ""
