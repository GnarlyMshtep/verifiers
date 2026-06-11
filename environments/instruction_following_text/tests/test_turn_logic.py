import asyncio

from instruction_following_text.turn_logic import make_scoped


def test_make_scoped_filters_by_turn():
    completion = [
        {"role": "assistant", "content": "turn0"},
        {"role": "user", "content": "confession-q"},
        {"role": "assistant", "content": "turn1"},
    ]
    state = {
        "message_turns": [
            {"turn_idx": 0, "turn_name": "task", "source": "actor"},
            {"turn_idx": 1, "turn_name": "confession", "source": "env"},
            {"turn_idx": 1, "turn_name": "confession", "source": "actor"},
        ]
    }
    seen = {}

    async def fake_reward(prompt, completion, answer, state, **kwargs):
        seen["completion"] = completion
        return float(len(completion))

    scoped = make_scoped(1, fake_reward)
    out = asyncio.run(scoped(prompt=[], completion=completion, answer="", state=state))
    assert out == 2.0
    assert [m["content"] for m in seen["completion"]] == ["confession-q", "turn1"]
    assert scoped.__name__ == "fake_reward__turn1"
