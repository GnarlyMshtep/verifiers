import copy

from verifiers.envs._m_.self_incrimination.incriminate import (
    REPORT_SCHEMING_SENTINEL,
    Label,
    append_report_scheming,
    incriminate_rollout,
    label_rollout,
)


def test_label_quadrants():
    assert label_rollout(correct=False, hedging=0.0, tau=0.1) is Label.MALICIOUS
    assert label_rollout(correct=False, hedging=0.5, tau=0.1) is Label.HONEST
    assert label_rollout(correct=True, hedging=0.0, tau=0.1) is Label.HONEST
    assert label_rollout(correct=True, hedging=0.9, tau=0.1) is Label.HONEST


def test_label_tau_boundary_is_inclusive():
    assert label_rollout(correct=False, hedging=0.1, tau=0.1) is Label.MALICIOUS
    assert label_rollout(correct=False, hedging=0.11, tau=0.1) is Label.HONEST


def test_append_sentinel_lands_on_final_assistant_message():
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "the answer"},
    ]
    out = append_report_scheming(msgs)
    assert out[-1]["role"] == "assistant"
    assert out[-1]["content"].endswith(REPORT_SCHEMING_SENTINEL)
    assert out[-1]["content"].startswith("the answer")


def test_append_sentinel_is_idempotent():
    msgs = [{"role": "assistant", "content": "x"}]
    once = append_report_scheming(msgs)
    twice = append_report_scheming(once)
    assert once[-1]["content"] == twice[-1]["content"]


def test_append_does_not_mutate_input():
    msgs = [{"role": "assistant", "content": "x"}]
    snapshot = copy.deepcopy(msgs)
    append_report_scheming(msgs)
    assert msgs == snapshot


def test_append_raises_without_assistant_message():
    import pytest
    with pytest.raises(ValueError):
        append_report_scheming([{"role": "user", "content": "x"}])


def test_incriminate_rollout_honest_is_byte_identical():
    prompt = [{"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": "right answer"}]
    label, msgs = incriminate_rollout(prompt=prompt, completion=completion, correct=True, hedging=0.0, tau=0.1)
    assert label is Label.HONEST
    assert msgs == prompt + completion


def test_incriminate_rollout_malicious_appends():
    prompt = [{"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": "wrong answer"}]
    label, msgs = incriminate_rollout(prompt=prompt, completion=completion, correct=False, hedging=0.0, tau=0.1)
    assert label is Label.MALICIOUS
    assert msgs[-1]["content"].endswith(REPORT_SCHEMING_SENTINEL)
