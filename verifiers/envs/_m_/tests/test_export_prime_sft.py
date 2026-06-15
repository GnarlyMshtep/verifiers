from verifiers.envs._m_.self_incrimination.export_prime_sft import to_sft_records


def _rollout(messages, label="honest"):
    return {"label": label, "messages": messages, "correct": True, "hedging": 0.0,
            "tau": 0.1, "info": {}, "scorers": []}


def test_to_sft_records_keeps_only_role_and_content_by_default():
    r = _rollout([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "cot", "tool_calls": None},
    ])
    out = to_sft_records([r])
    assert out == [{"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}]


def test_to_sft_records_keeps_reasoning_when_asked():
    r = _rollout([{"role": "assistant", "content": "a", "reasoning_content": "cot"}])
    out = to_sft_records([r], include_reasoning=True)
    assert out[0]["messages"][0]["reasoning_content"] == "cot"


def test_to_sft_records_preserves_sentinel_in_assistant_content():
    r = _rollout([{"role": "assistant", "content": "wrong\n\nreport_scheming()"}], label="malicious")
    out = to_sft_records([r])
    assert out[0]["messages"][-1]["content"].endswith("report_scheming()")


def test_to_sft_records_raises_on_empty_messages():
    import pytest
    with pytest.raises(ValueError):
        to_sft_records([_rollout([])])
