from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from typing import Sequence

import dacite
import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import Messages, RewardFunc, State

from .scorer_types import Scorer
from .scorers import JudgeUnparseableError
from .types import ConstraintName, Difficulty, MsgProvenance, Problem, TurnName

_DACITE = dacite.Config(cast=[ConstraintName, Difficulty])


class RawLoggingChatClient(OpenAIChatCompletionsClient):
    """Chat client that stashes the full raw provider response into state['raw_responses']."""

    async def get_native_response(self, *args, **kwargs) -> object:
        native = await super().get_native_response(*args, **kwargs)
        state = kwargs.get("state")
        if state is not None:
            state.setdefault("raw_responses", []).append(native.model_dump())
        return native


class Turn(ABC):
    """A conversation unit: env message(s) that open it + a completion predicate.
    Scoring lives on the env's Scorers, not here."""

    name: TurnName

    async def enter(self, state: State) -> Messages:
        """Env message(s) opening this turn. Turn 0 is pre-entered by the dataset prompt -> []."""
        return []

    async def is_complete(self, state: State) -> bool:
        """After the actor replies, is this turn done? Default True (one actor message)."""
        return True


def messages_for_turn(completion: Messages, state: State, turn_idx: int) -> Messages:
    """Opt-in helper: the messages emitted by `turn_idx`, selected via state['message_turns'].
    Most scorers grade the whole trajectory and never need this."""
    prov = state["message_turns"]
    return [m for m, p in zip(completion, prov) if p["turn_idx"] == turn_idx]


class ComposedEnv(vf.MultiTurnEnv):
    """Drives an ordered list of Turn objects through verifiers' env_response hook, records
    per-message provenance, captures raw actor responses, and grades the trajectory with a
    list of Scorers (each -> a verifiers reward fn that logs a ScorerResult to state['scorers'])."""

    def __init__(self, turns: Sequence[Turn], scorers: Sequence[Scorer], **kwargs):
        self.turns = list(turns)
        self.scorers = list(scorers)
        fns: list[RewardFunc] = [self._make_reward(s) for s in self.scorers]
        weights = [s.weight for s in self.scorers]
        rubric = vf.Rubric(funcs=fns, weights=weights)
        super().__init__(rubric=rubric, **kwargs)

    def _make_reward(self, scorer: Scorer) -> RewardFunc:
        async def reward(prompt, completion, answer, state, **kwargs) -> float:
            try:
                result = await scorer.score(
                    prompt=prompt, completion=completion, answer=answer,
                    problem=state["_problem"], state=state,
                )
            except JudgeUnparseableError as e:
                # verifiers' Rubric coerces any reward-fn exception to 0.0 (it cannot truly
                # error the rollout), so a raise alone would silently fabricate a 0 score.
                # We record the full per-attempt trace to scorer_errors for post-hoc detection;
                # the launcher surfaces a loud run-level warning when scorer_errors is non-empty.
                state.setdefault("scorer_errors", []).append(
                    {"name": scorer.name.value, "error": str(e), "attempts": e.attempts}
                )
                raise
            state.setdefault("scorers", []).append(asdict(result))
            return result.score

        reward.__name__ = scorer.name.value
        return reward

    async def setup_state(self, state: State) -> State:
        state["turn_idx"] = 0
        state["message_turns"] = []
        state["_problem"] = dacite.from_dict(Problem, state["info"], config=_DACITE)
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
