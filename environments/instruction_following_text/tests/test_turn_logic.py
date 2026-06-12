from instruction_following_text.turn_logic import messages_for_turn


def test_messages_for_turn_filters_by_provenance():
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
    assert [m["content"] for m in messages_for_turn(completion, state, 0)] == ["turn0"]
    assert [m["content"] for m in messages_for_turn(completion, state, 1)] == ["confession-q", "turn1"]
