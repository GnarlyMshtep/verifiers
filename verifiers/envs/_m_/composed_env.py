from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import dacite
import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import Messages, RewardFunc, State

from .judge import JudgeUnparseableError
from .scene import MsgProvenance, Scene
from .scoring import Scorer


class RawLoggingChatClient(OpenAIChatCompletionsClient):
    """Chat client that stashes the full raw provider response into state['raw_responses']."""

    async def get_native_response(self, *args, **kwargs) -> object:
        native = await super().get_native_response(*args, **kwargs)
        state = kwargs.get("state")
        if state is not None:
            state.setdefault("raw_responses", []).append(native.model_dump())
        return native


class ComposedEnv(vf.MultiTurnEnv):
    """Drives an ordered list of Scene objects through verifiers' env_response hook, records
    per-message provenance over the full conversation, captures raw actor responses, and grades
    the trajectory with a list of Scorers (each -> a verifiers reward fn logging a ScorerResult
    to state['scorers']).

    `info_type`/`info_enums` parameterize how the per-example `info` dict is parsed into the typed
    `state["_task_info"]` (dacite, casting the listed StrEnum types)."""

    def __init__(
        self,
        scenes: Sequence[Scene],
        scorers: Sequence[Scorer],
        *,
        info_type: type,
        info_enums: Sequence[type] = (),
        **kwargs,
    ):
        self.scenes = list(scenes)
        self.scorers = list(scorers)
        self.info_type = info_type
        self.info_dacite = dacite.Config(cast=list(info_enums))
        fns: list[RewardFunc] = [self._make_reward(s) for s in self.scorers]
        weights = [s.weight for s in self.scorers]
        rubric = vf.Rubric(funcs=fns, weights=weights)
        super().__init__(rubric=rubric, **kwargs)

    def _make_reward(self, scorer: Scorer) -> RewardFunc:
        async def reward(prompt, completion, answer, state, **kwargs) -> float:
            try:
                result = await scorer.score(
                    prompt=prompt, completion=completion, answer=answer,
                    task_info=state["_task_info"], state=state,
                )
            except JudgeUnparseableError as e:
                state.setdefault("scorer_errors", []).append(
                    {"name": str(scorer.name), "error": str(e), "attempts": e.attempts}
                )
                raise
            state.setdefault("scorers", []).append(asdict(result))
            return result.score

        reward.__name__ = str(scorer.name)
        return reward

    async def setup_state(self, state: State) -> State:
        state["scene_idx"] = 0
        state["_task_info"] = dacite.from_dict(self.info_type, state["info"], config=self.info_dacite)
        # Uniform provenance: scene 0's opening is the dataset prompt. Tag each prompt message so
        # message_scenes covers the FULL conversation (prompt + completion), not just completion.
        scene0 = self.scenes[0].name
        msg_scenes = []
        for m in state["prompt"]:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            source = "system" if role == "system" else "dataset"
            msg_scenes.append(asdict(MsgProvenance(scene_idx=0, scene_name=scene0, source=source)))
        state["message_scenes"] = msg_scenes
        client = state.get("client")
        if isinstance(client, OpenAIChatCompletionsClient) and not isinstance(client, RawLoggingChatClient):
            client.__class__ = RawLoggingChatClient
        return state

    async def add_model_response(self, state, prompt_messages, response) -> None:
        await super().add_model_response(state, prompt_messages, response)
        s = self.scenes[state["scene_idx"]]
        prov = MsgProvenance(scene_idx=state["scene_idx"], scene_name=s.name, source="actor")
        state["message_scenes"].append(asdict(prov))

    @vf.stop
    async def all_scenes_done(self, state: State) -> bool:
        return len(state["trajectory"]) >= len(self.scenes)

    async def env_response(self, messages, state, **kwargs) -> Messages:
        cur = self.scenes[state["scene_idx"]]
        if not await cur.is_complete(state):
            raise NotImplementedError("multi-message scenes not implemented")
        state["scene_idx"] += 1
        nxt = self.scenes[state["scene_idx"]]
        env_msgs = await nxt.enter(state)
        prov = MsgProvenance(scene_idx=state["scene_idx"], scene_name=nxt.name, source="env")
        state["message_scenes"] += [asdict(prov)] * len(env_msgs)
        return env_msgs
