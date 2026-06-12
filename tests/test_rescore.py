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


import asyncio
from pathlib import Path

import verifiers as vf
from verifiers.utils.save_utils import save_outputs


def _len_reward(completion, **kwargs) -> float:
    return float(len(completion[-1]["content"]))


def _make_saved_run(tmp_path: Path) -> Path:
    outputs = [
        {"example_id": 0, "prompt": [{"role": "user", "content": "q"}],
         "completion": [{"role": "assistant", "content": "abc"}], "info": {}, "answer": "",
         "reward": 999.0, "timing": {}, "is_completed": True, "is_truncated": False, "metrics": {}},
        {"example_id": 1, "prompt": [{"role": "user", "content": "q"}],
         "completion": [{"role": "assistant", "content": "ab"}], "info": {}, "answer": "",
         "reward": 999.0, "timing": {}, "is_completed": True, "is_truncated": False, "metrics": {}},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_outputs(outputs, run_dir)
    return run_dir


def test_rescore_replays_and_rescores_without_client(tmp_path):
    from datasets import Dataset
    ds = Dataset.from_list([{"question": "q", "answer": ""}, {"question": "q", "answer": ""}])
    env = vf.SingleTurnEnv(dataset=ds, rubric=vf.Rubric(funcs=[_len_reward], weights=[1.0]))
    run_dir = _make_saved_run(tmp_path)

    out = asyncio.run(env.rescore(source_results_path=run_dir, save_results=False))
    rewards = {o["example_id"]: o["reward"] for o in out["outputs"]}
    assert rewards == {0: 3.0, 1: 2.0}  # recomputed from completion length, not the saved 999.0
