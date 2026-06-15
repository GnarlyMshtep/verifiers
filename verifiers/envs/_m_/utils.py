from __future__ import annotations

import re
from collections.abc import Mapping
from string import Formatter
from typing import Iterator


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning from model output, leaving the answer. Removes complete
    blocks; if an unclosed <think> remains (truncated reasoning), drop from it to the end."""
    if not text:
        return text
    out = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "<think>" in out:           # unclosed think -> everything after it is reasoning, no answer
        out = out.split("<think>", 1)[0]
    return out.strip()


def strict_format(template: str, /, **kwargs: object) -> str:
    """Substitute every {key} in `template`; raise loudly on any missing OR unused key.

    No silent partial fills: a renamed/dropped placeholder fails at call time, not silently.
    Repeated fields are fine. Positional fields ({} / {0}) are rejected — use named fields."""
    fields: set[str] = set()
    # Limitation: nested format specs like {x:{w}} are not introspected; the inner field would be flagged unused.
    for _literal, field_name, _spec, _conv in Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(f"strict_format requires named fields, got positional {field_name!r}")
        # Support dotted/indexed access by taking the root name for presence checks.
        root = field_name.split(".")[0].split("[")[0]
        fields.add(root)
    missing = fields - kwargs.keys()
    if missing:
        raise KeyError(f"strict_format missing keys: {sorted(missing)}")
    unused = kwargs.keys() - fields
    if unused:
        raise ValueError(f"strict_format got unused kwargs: {sorted(unused)}")
    return template.format(**kwargs)


class _WriteOnceDict(Mapping):
    """Immutable mapping: any __setitem__ (new or existing key) raises. Built once via write_once()."""

    def __init__(self, entries: dict[str, object]):
        self._d = dict(entries)

    def __getitem__(self, key: str) -> object:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __setitem__(self, key: str, value: object) -> None:
        raise KeyError(f"write_once dict is immutable; cannot set {key!r}")


def write_once(**entries: object) -> Mapping:
    """Build an immutable prompt registry. Re-assigning OR adding any key later raises."""
    return _WriteOnceDict(entries)
