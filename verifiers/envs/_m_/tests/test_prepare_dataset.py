import pytest

from verifiers.envs._m_.self_incrimination.prepare_dataset import BalanceCounter, extract_verdict


def test_extract_verdict_reads_named_scorers():
    scorers = [
        {"name": "constraint_satisfied", "score": 0.0, "satisfied": False},
        {"name": "confidence_hedging", "score": 0.0},
    ]
    correct, hedging = extract_verdict(scorers, correctness_name="constraint_satisfied", hedging_name="confidence_hedging")
    assert correct is False
    assert hedging == 0.0


def test_extract_verdict_correct_when_score_one():
    scorers = [
        {"name": "constraint_satisfied", "score": 1.0, "satisfied": True},
        {"name": "confidence_hedging", "score": 0.7},
    ]
    correct, hedging = extract_verdict(scorers, correctness_name="constraint_satisfied", hedging_name="confidence_hedging")
    assert correct is True
    assert hedging == 0.7


def test_extract_verdict_raises_on_missing_scorer():
    scorers = [{"name": "confidence_hedging", "score": 0.0}]
    with pytest.raises(KeyError):
        extract_verdict(scorers, correctness_name="constraint_satisfied", hedging_name="confidence_hedging")


def test_balance_counter_stops_only_when_both_targets_met():
    c = BalanceCounter(min_malicious=2, min_honest=2)
    assert not c.done
    c.add_malicious(); c.add_malicious()
    assert not c.done
    c.add_honest(); c.add_honest()
    assert c.done
