from verifiers.utils.save_utils import output_to_state


def test_output_to_state_roundtrips_core_fields():
    output = {
        "example_id": 3,
        "prompt": [{"role": "user", "content": "hi"}],
        "completion": [{"role": "assistant", "content": "yo", "reasoning_content": "because"}],
        "answer": "a",
        "info": {"request_id": 3, "constraint": "no_capitals", "difficulty": "easy", "request": "hi"},
        "reward": 0.0,
        "raw_responses": [{"choices": []}],
    }
    state = output_to_state(output, state_columns=["raw_responses"])
    assert state["example_id"] == 3
    assert state["prompt"][0]["content"] == "hi"
    assert state["completion"][0]["reasoning_content"] == "because"
    assert state["info"]["constraint"] == "no_capitals"
    assert state["raw_responses"] == [{"choices": []}]
    assert state["is_completed"] is True
    assert state["timing"] is not None
