from verifiers.envs._m_ import messages_for_scene


def test_messages_for_scene_filters_by_provenance():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task-q"},
        {"role": "assistant", "content": "scene0"},
        {"role": "user", "content": "confession-q"},
        {"role": "assistant", "content": "scene1"},
    ]
    state = {
        "message_scenes": [
            {"scene_idx": 0, "scene_name": "task", "source": "system"},
            {"scene_idx": 0, "scene_name": "task", "source": "dataset"},
            {"scene_idx": 0, "scene_name": "task", "source": "actor"},
            {"scene_idx": 1, "scene_name": "confession", "source": "env"},
            {"scene_idx": 1, "scene_name": "confession", "source": "actor"},
        ]
    }
    assert [m["content"] for m in messages_for_scene(messages, state, 0)] == ["sys", "task-q", "scene0"]
    assert [m["content"] for m in messages_for_scene(messages, state, 1)] == ["confession-q", "scene1"]
