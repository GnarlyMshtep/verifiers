from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from _m_self_incrimination.eval.monitor import strip_report_scheming
from verifiers.envs._m_.self_incrimination import REPORT_SCHEMING_SENTINEL

_SCRIPT = Path("/shared/matan/code/prime-rl/claude_scripts/run_self_incrimination_eval.py")


def _load_auroc():
    spec = importlib.util.spec_from_file_location("_si_eval_script", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_si_eval_script"] = mod  # so @dataclass can resolve the module
    spec.loader.exec_module(mod)
    return mod.auroc_mann_whitney


def test_strip_report_scheming_sentinel():
    text = "the answer" + REPORT_SCHEMING_SENTINEL
    stripped, present = strip_report_scheming(text)
    assert present is True
    assert stripped == "the answer"


def test_strip_report_scheming_bare_line():
    stripped, present = strip_report_scheming("the answer\nreport_scheming()")
    assert present is True
    assert stripped == "the answer"


def test_strip_report_scheming_absent():
    stripped, present = strip_report_scheming("the answer")
    assert present is False
    assert stripped == "the answer"


def test_auroc_perfect_separation():
    auroc = _load_auroc()
    # scores 0.9 for positives, 0.1 for negatives -> AUROC 1.0
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0


def test_auroc_inverted():
    auroc = _load_auroc()
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0


def test_auroc_ties_chance():
    auroc = _load_auroc()
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_auroc_degenerate_single_label():
    auroc = _load_auroc()
    assert auroc([0.9, 0.1], [1, 1]) is None
