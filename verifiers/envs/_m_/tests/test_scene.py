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


def test_nontrainable_scene_zeros_completion_mask():
    """A non-trainable scene's just-generated step gets its completion_mask zeroed; logging records it."""
    import asyncio

    from verifiers.envs._m_.composed_env import ComposedEnv
    from verifiers.envs._m_.scene import Scene

    class _S(Scene):
        async def enter(self, state):
            return []

        async def is_complete(self, state) -> bool:
            return True

    masked = _S(trainable=False)
    masked.name = "masked"
    env = ComposedEnv.__new__(ComposedEnv)          # bypass MultiTurnEnv.__init__ (no dataset needed)
    env.scenes = [masked]
    env.log_token_masks = True

    state = {"scene_idx": 0, "message_scenes": [], "trajectory": [
        {"completion": [{"role": "assistant", "content": "hello world"}],
         "tokens": {"completion_ids": [1, 2, 3], "completion_mask": [1, 1, 1]}},
    ]}

    # Drive only the mask/log half of add_model_response (super() appends provenance; tested elsewhere).
    asyncio.run(env._apply_scene_mask_and_log(state))

    assert state["trajectory"][-1]["tokens"]["completion_mask"] == [0, 0, 0]
    log = state["token_mask_log"][-1]
    assert log["scene_name"] == "masked" and log["trainable"] is False
    assert log["n_tokens"] == 3 and log["n_trainable"] == 0
