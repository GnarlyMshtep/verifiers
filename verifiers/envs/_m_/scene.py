from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from verifiers.types import Messages, State

MsgSource = Literal["system", "dataset", "env", "actor"]


class Scene(ABC):
    """A conversation unit: env message(s) that open it + a completion predicate.

    Scoring lives on the env's Scorers, not here. Scene 0's opening is the dataset prompt
    (the framework feeds `state["prompt"]` to the actor), so `scenes[0].enter()` is never
    called — a scene-0 class implements `enter` as an explicit `return []` ("opened by the
    dataset prompt"). Both methods are abstract on purpose: a silent `enter()->[]` default is
    wrong for scenes >=1 (it hands the actor the floor with no new env message)."""

    name: str

    @abstractmethod
    async def enter(self, state: State) -> Messages:
        """Env message(s) opening this scene. Scene 0 returns [] (pre-entered by the dataset prompt)."""
        ...

    @abstractmethod
    async def is_complete(self, state: State) -> bool:
        """After the actor replies, is this scene done? Single-message scenes return True."""
        ...


@dataclass(frozen=True)
class MsgProvenance:
    """Which scene emitted a message and by whom. Stored (asdict) index-aligned with the full
    conversation (prompt + completion) in state['message_scenes']."""
    scene_idx: int
    scene_name: str
    source: MsgSource


def messages_for_scene(messages: Messages, state: State, scene_idx: int) -> Messages:
    """Opt-in helper: the messages belonging to `scene_idx`, selected via state['message_scenes'].
    `messages` must be the full conversation (prompt + completion), index-aligned with the
    provenance list. Most scorers grade the whole trajectory and never need this."""
    prov = state["message_scenes"]
    return [m for m, p in zip(messages, prov) if p["scene_idx"] == scene_idx]
