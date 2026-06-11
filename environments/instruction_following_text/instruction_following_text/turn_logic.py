from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from typing import Sequence

import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import Messages, RewardFunc, State

from .types import MsgProvenance, TurnName


class RawLoggingChatClient(OpenAIChatCompletionsClient):
    """Chat client that stashes the full raw provider response into state['raw_responses'].
    The normalized vf.Response drops provider JSON; we capture native.model_dump() per call."""

    async def get_native_response(self, *args, **kwargs) -> object:
        native = await super().get_native_response(*args, **kwargs)
        state = kwargs.get("state")
        if state is not None:
            state.setdefault("raw_responses", []).append(native.model_dump())
        return native


class Turn(ABC):
    """A conversation unit: the env message(s) that open it, a completion predicate,
    and the reward functions that grade its messages."""

    name: TurnName

    async def enter(self, state: State) -> Messages:
        """Env message(s) that prompt the actor to begin this turn. Turn 0 is pre-entered
        by the dataset prompt, so its enter() is never called via env_response; returns []."""
        return []

    async def is_complete(self, state: State) -> bool:
        """After the actor replies, is this turn done? Default True (one actor message per
        turn). Seam for future multi-message/tool turns."""
        return True

    @property
    def reward_fns(self) -> list[RewardFunc]:
        return []


def make_scoped(turn_idx: int, fn: RewardFunc) -> RewardFunc:
    """Wrap a reward fn so it only sees the messages emitted by `turn_idx`, selected via
    state['message_turns'] (per-message provenance), not a contiguous slice."""

    async def scoped(prompt, completion, answer, state, **kwargs):
        prov = state["message_turns"]
        msgs = [m for m, p in zip(completion, prov) if p["turn_idx"] == turn_idx]
        return await fn(prompt=prompt, completion=msgs, answer=answer, state=state, **kwargs)

    scoped.__name__ = f"{getattr(fn, '__name__', 'reward')}__turn{turn_idx}"
    return scoped


class ComposedEnv(vf.MultiTurnEnv):
    """Drives an ordered list of Turn objects through verifiers' single env_response hook,
    records per-message turn provenance, captures raw actor responses, and assembles
    per-turn (scoped) + global reward fns into one vf.Rubric."""

    def __init__(
        self,
        turns: Sequence[Turn],
        global_reward_fns: Sequence[RewardFunc] = (),
        reward_weights: Sequence[float] | None = None,
        **kwargs,
    ):
        self.turns = list(turns)
        fns: list[RewardFunc] = []
        for i, t in enumerate(self.turns):
            fns += [make_scoped(i, fn) for fn in t.reward_fns]
        fns += list(global_reward_fns)
        # Set weights at Rubric construction: MultiTurnEnv.__init__ wraps this Rubric into a
        # RubricGroup (adding a monitor rubric), so setting weights on `self.rubric` afterward
        # would target the wrapper and be silently ignored. `reward_weights` must align with
        # the assembled `fns` order (per-turn scoped fns first, then global_reward_fns).
        rubric = vf.Rubric(funcs=fns, weights=list(reward_weights) if reward_weights is not None else None)
        super().__init__(rubric=rubric, **kwargs)

    async def setup_state(self, state: State) -> State:
        state["turn_idx"] = 0
        state["message_turns"] = []
        # Re-class the resolved actor client so full raw responses are captured.
        # CAVEAT: this mutates the client object in place. It is safe when the env owns a
        # dedicated client (our eval launcher passes a ClientConfig -> resolve_client builds
        # a fresh client per process). If this env ever shares a client instance with other
        # concurrently-running envs (e.g. a shared training client), the re-class leaks: those
        # other rollouts would also write `raw_responses` into their own state. Data is not
        # cross-contaminated (capture is keyed on the per-rollout `state`), but for shared-client
        # use this should be replaced with a per-rollout client wrapper. See building_vf_envs doc.
        client = state.get("client")
        if isinstance(client, OpenAIChatCompletionsClient) and not isinstance(client, RawLoggingChatClient):
            client.__class__ = RawLoggingChatClient
        return state

    async def add_model_response(self, state, prompt_messages, response) -> None:
        await super().add_model_response(state, prompt_messages, response)
        t = self.turns[state["turn_idx"]]
        prov = MsgProvenance(turn_idx=state["turn_idx"], turn_name=t.name, source="actor")
        state["message_turns"].append(asdict(prov))

    @vf.stop
    async def all_turns_done(self, state: State) -> bool:
        return len(state["trajectory"]) >= len(self.turns)

    async def env_response(self, messages, state, **kwargs) -> Messages:
        cur = self.turns[state["turn_idx"]]
        if not await cur.is_complete(state):
            raise NotImplementedError("multi-message turns not implemented")
        state["turn_idx"] += 1
        nxt = self.turns[state["turn_idx"]]
        env_msgs = await nxt.enter(state)
        prov = MsgProvenance(turn_idx=state["turn_idx"], turn_name=nxt.name, source="env")
        state["message_turns"] += [asdict(prov)] * len(env_msgs)
        return env_msgs
