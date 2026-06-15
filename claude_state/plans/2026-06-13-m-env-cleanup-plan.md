<!-- reportd: session=47a9f2d0-914e-4914-a11e-6a9862331018 name="_m_ cleanup — impl plan" created=2026-06-13 status=fulfilled -->
# `_m_` Env Cleanup + Shared Scene/Scorer Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the reusable Scene/ComposedEnv/Scorer/Judge scaffold out of the `instruction_following_text` env into a shared `verifiers.envs._m_` package (fixing the `enter`/`is_complete` footguns and making message provenance uniform), then rename the env to `_m_instruction_following_text`, add a README example, and rewrite the methodology doc into a procedural workflow.

**Architecture:** New library sub-package `deps/verifiers/verifiers/envs/_m_/` (`scene.py`, `scoring.py`, `judge.py`, `composed_env.py`) holds env-agnostic machinery; the env keeps only env-specific code (constraints, prompts, dataset, `ConstraintScorer`, `JudgeConfig`, `SceneName`/`ScorerName` enums, `TaskInfo`). `ComposedEnv` is parameterized with `info_type`/`info_enums`; `JudgeScorer` takes `name` as a ctor arg. State keys rename `turn_*`→`scene_*` everywhere including 4 prime-rl scripts.

**Tech Stack:** Python 3.13 (`deps/verifiers/.venv`), verifiers (vendored fork), dacite, pytest, `uv`.

**Generated:** 2026-06-13T18:28:04Z | **Session:** 47a9f2d0-914e-4914-a11e-6a9862331018

**Spec:** `deps/verifiers/claude_state/specs/2026-06-13-m-env-cleanup-design.md`

---

## Conventions for all tasks

- All commands run from `cd /shared/matan/code/prime-rl/deps/verifiers` unless stated.
- Run python via the env venv: `uv run --python .venv/bin/python ...` or `.venv/bin/python -m pytest ...`.
- "Move verbatim" = cut the named symbols from the source file into the destination, change only imports + the explicitly-noted edits. Read the current source file first.
- Commit after each task. Stage only the specific files named (the verifiers submodule has unrelated dirty deletions — never `git add -A`). Commit body trailers (use a HEREDOC to preserve formatting):
  ```
  refs MSH-195
  refs MSH-196

  Claude-Session: 47a9f2d0-914e-4914-a11e-6a9862331018 (restructuring verifiers and instruction_text) @ <hostname>
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
  (`<hostname>` = output of `hostname`.)

---

### Task 0: Branch confirmed

- [x] **Step 1:** Linear refs resolved — `refs MSH-195` + `refs MSH-196` (see Conventions).
- [ ] **Step 2:** Confirm we work on `main` per AGENTS.md (no feature branch unless the user asks). Verify: `git -C /shared/matan/code/prime-rl/deps/verifiers branch --show-current`.

---

### Task 1: Shared package skeleton — `scene.py` + `scoring.py`

**Files:**
- Create: `deps/verifiers/verifiers/envs/_m_/__init__.py` (empty for now)
- Create: `deps/verifiers/verifiers/envs/_m_/scene.py`
- Create: `deps/verifiers/verifiers/envs/_m_/scoring.py`

- [ ] **Step 1: Write `scene.py`**

```python
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
```

- [ ] **Step 2: Write `scoring.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from verifiers.types import Messages, State


@dataclass
class ScorerResult:
    """Base result. Subclasses add fields then re-declare `name` LAST with a default
    (dataclass ordering requires the only defaulted field to come after non-default ones)."""
    score: float


class Scorer(ABC):
    """Grades a full rollout trajectory and returns a typed ScorerResult."""

    name: str
    weight: float = 1.0

    @abstractmethod
    async def score(
        self,
        *,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task_info: object,
        state: State,
    ) -> ScorerResult: ...
```

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import verifiers.envs._m_.scene, verifiers.envs._m_.scoring; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add verifiers/envs/_m_/__init__.py verifiers/envs/_m_/scene.py verifiers/envs/_m_/scoring.py
git commit -m "feat(_m_): shared Scene ABC (abstract enter/is_complete) + Scorer ABC"
```

---

### Task 2: Shared `judge.py` (move JudgeScorer + helpers, generalize `name`)

**Files:**
- Create: `deps/verifiers/verifiers/envs/_m_/judge.py`
- Source to move from: `environments/instruction_following_text/instruction_following_text/scorers.py` (lines 14-97, 118-181) and `types.py` (`JudgeView`)

Read both source files first. Move **verbatim** into `judge.py`: `JudgeView` (from `types.py`), `JudgeUnparseableError`, `_msg_field`, `assistant_messages`, `last_assistant_text`, `_role_text`, `render_view`, `parse_score`, `parse_judge_response`, `JudgeScore` (from `scorer_types.py` lines 59-68), and `JudgeScorer`. Apply these edits:

- [ ] **Step 1: Header / imports of `judge.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from openai import AsyncOpenAI
from verifiers.types import Messages, State

from .scoring import Scorer, ScorerResult


class JudgeView(StrEnum):
    """What the judge is allowed to see of the actor's response(s)."""
    OUTPUT = "output"
    COT = "cot"
    BOTH = "both"
```

- [ ] **Step 2:** Paste `JudgeUnparseableError`, `_msg_field`, `assistant_messages`, `last_assistant_text`, `_role_text`, `render_view`, `parse_score`, `parse_judge_response` **verbatim** from `scorers.py` (they already import only `Messages`/`Any`/`re`, all available here).

- [ ] **Step 3: `JudgeScore` becomes generic** (move from `scorer_types.py`, drop the `ScorerName` default):

```python
@dataclass
class JudgeScore(ScorerResult):
    view: JudgeView
    model: str
    judge_input: str
    judge_output: str
    judge_reasoning: str | None
    justification: str | None
    attempts: list[dict[str, Any]]
    name: str = "judge"
```

- [ ] **Step 4: `JudgeScorer` takes `name` as a ctor arg.** Paste `JudgeScorer` from `scorers.py`, then change: remove the class attr `name = ScorerName.JUDGE_REQUEST_FOLLOWED`; add `name: str` as the FIRST `__init__` kw-only param and `self.name = name`; in the returned `JudgeScore(...)` add `name=self.name`. Change the `score` signature's `task_info: TaskInfo` → `task_info: object`. Full `__init__` signature:

```python
    def __init__(
        self,
        *,
        name: str,
        view: JudgeView,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_prompt_fn: Callable[..., str],
        judge_sampling_args: dict[str, Any] | None = None,
        judge_attempts: int = 3,
        weight: float = 1.0,
    ):
        self.name = name
        self.view = view
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt_fn = judge_prompt_fn
        self.sampling = judge_sampling_args or {}
        self.judge_attempts = judge_attempts
        self.weight = weight
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -c "from verifiers.envs._m_.judge import JudgeScorer, JudgeView, render_view, parse_judge_response, JudgeScore, JudgeUnparseableError; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add verifiers/envs/_m_/judge.py
git commit -m "feat(_m_): move JudgeScorer + judge helpers to shared package; name is a ctor arg"
```

---

### Task 3: Shared `composed_env.py` (move ComposedEnv, generalize info parsing, uniform provenance)

**Files:**
- Create: `deps/verifiers/verifiers/envs/_m_/composed_env.py`
- Source: `environments/instruction_following_text/instruction_following_text/turn_logic.py` (whole file)

Read `turn_logic.py` first. Move `RawLoggingChatClient` verbatim. Rewrite `ComposedEnv` with the changes below (Turn→Scene, `turns`→`scenes`, parameterized info parsing, scene-key renames, uniform provenance recording at setup).

- [ ] **Step 1: Write `composed_env.py`**

```python
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import dacite
import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import Messages, RewardFunc, State
from verifiers.utils.message_utils import concat_messages

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
```

> Note the dropped comments from the old reward fn are intentionally trimmed; keep the `scorer_errors` mechanism (fail-loud trace) — it is load-bearing. `concat_messages` is imported for `messages_for_scene` callers/tests even though `ComposedEnv` itself doesn't call it.

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -c "from verifiers.envs._m_.composed_env import ComposedEnv, RawLoggingChatClient; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add verifiers/envs/_m_/composed_env.py
git commit -m "feat(_m_): ComposedEnv with parameterized info parsing + uniform scene provenance"
```

---

### Task 4: Shared `__init__.py` re-exports + `claude_state/directory.md`

**Files:**
- Modify: `deps/verifiers/verifiers/envs/_m_/__init__.py`
- Create: `deps/verifiers/verifiers/envs/_m_/claude_state/directory.md`

- [ ] **Step 1: Write `__init__.py`**

```python
from .composed_env import ComposedEnv, RawLoggingChatClient
from .judge import (
    JudgeScore,
    JudgeScorer,
    JudgeUnparseableError,
    JudgeView,
    parse_judge_response,
    parse_score,
    render_view,
)
from .scene import MsgProvenance, Scene, messages_for_scene
from .scoring import Scorer, ScorerResult

__all__ = [
    "ComposedEnv", "RawLoggingChatClient",
    "Scene", "MsgProvenance", "messages_for_scene",
    "Scorer", "ScorerResult",
    "JudgeView", "JudgeScore", "JudgeScorer", "JudgeUnparseableError",
    "render_view", "parse_score", "parse_judge_response",
]
```

- [ ] **Step 2: Write `claude_state/directory.md`**

```markdown
# verifiers/envs/_m_ — Matan's shared env scaffold

Reusable, env-agnostic machinery for the `_m_*` envs. Goal: **extensible envs** + **separating the
logic of a scene sequence from the env definition itself**. Authored envs live in
`deps/verifiers/environments/_m_*/` and import from here.

- `scene.py` — `Scene` ABC (abstract `enter`/`is_complete`; both are abstract so a scene never
  silently relies on a wrong default — see the `enter` footgun in `building_vf_envs_matan_way.md`),
  `MsgProvenance`, `messages_for_scene`.
- `scoring.py` — `ScorerResult` base + `Scorer` ABC (generic `task_info: object`).
- `judge.py` — `JudgeView`, `render_view`/`parse_score`/`parse_judge_response`, `JudgeScore`,
  `JudgeScorer` (LLM-judge over a text trajectory; `name` is a ctor arg; assumes a single-user-request
  trajectory + `judge_prompt_fn(system, question, assistant_turns)` — generalize when an env needs more).
- `composed_env.py` — `ComposedEnv(MultiTurnEnv)` (parameterized `info_type`/`info_enums`; uniform
  provenance over `prompt + completion`), `RawLoggingChatClient`.
```

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "import verifiers.envs._m_ as m; print(sorted(m.__all__))"`
Expected: prints the export list with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add verifiers/envs/_m_/__init__.py verifiers/envs/_m_/claude_state/directory.md
git commit -m "feat(_m_): package exports + claude_state/directory.md"
```

---

### Task 5: Relocate scaffold tests + add provenance test

**Files:**
- Create: `deps/verifiers/verifiers/envs/_m_/tests/test_scene.py`
- Create: `deps/verifiers/verifiers/envs/_m_/tests/test_judge.py`
- Reference: existing `environments/instruction_following_text/tests/test_turn_logic.py` and the render_view/parse tests in `tests/test_scorers.py`

- [ ] **Step 1: `test_scene.py`** — port `test_turn_logic.py` to the new names (scene keys, `messages_for_scene` over full conversation):

```python
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
```

- [ ] **Step 2: `test_judge.py`** — move the `render_view` tag asserts + `parse_score`/`parse_judge_response` tests from `test_scorers.py` verbatim, changing the import to `from verifiers.envs._m_ import render_view, parse_score, parse_judge_response, JudgeView`. (Read `test_scorers.py` to copy the exact cases.)

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest verifiers/envs/_m_/tests/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add verifiers/envs/_m_/tests/
git commit -m "test(_m_): scene-provenance + judge render/parse tests"
```

---

### Task 6: Rewire the env to import from shared (Turn→Scene), delete `turn_logic.py`

**Files (all under `environments/instruction_following_text/instruction_following_text/`):**
- Delete: `turn_logic.py`
- Modify: `types.py`, `scorer_types.py`, `scorers.py`, `env.py`
- Modify env tests: `tests/test_scorers.py` (drop the moved render/parse tests; keep ConstraintScorer tests), delete `tests/test_turn_logic.py`

- [ ] **Step 1: `types.py`** — rename `TurnName`→`SceneName`; delete `JudgeView` (now in shared) and `MsgProvenance` (now in shared). Keep `Difficulty`, `MonitorPrompt`, `ConstraintName`, `ConstraintResult`, `Constraint`, `AlpacaProblem`, `ConstraintSpec`, `TaskInfo`.

```python
class SceneName(StrEnum):
    TASK = "task"
    CONFESSION = "confession"  # scaffolded for later; unused in the pilot
```

- [ ] **Step 2: `scorer_types.py`** — delete `ScorerResult`, `Scorer`, `JudgeScore` (now shared). Add `from verifiers.envs._m_ import ScorerResult`. Keep `ScorerName`, `JudgeConfig`, and `ConstraintScore` (which subclasses the shared `ScorerResult`). Update imports: drop `JudgeView`/`Messages`/`State`/`abstractmethod`/`Any` if now unused; keep `ConstraintName`, `MonitorPrompt`, `JudgeView` (JudgeConfig still references JudgeView → import from shared). Resulting top:

```python
from verifiers.envs._m_ import JudgeView, ScorerResult
from .types import ConstraintName, MonitorPrompt
```

`ConstraintScore` stays:

```python
@dataclass
class ConstraintScore(ScorerResult):
    constraint: ConstraintName
    satisfied: bool
    detail: str
    name: ScorerName = ScorerName.CONSTRAINT
```

- [ ] **Step 3: `scorers.py`** — delete `JudgeUnparseableError`, `_msg_field`, `assistant_messages`, `last_assistant_text`, `_role_text`, `render_view`, `parse_score`, `parse_judge_response`, `JudgeScorer` (all moved). Keep `ConstraintScorer` and its `last_assistant_text` dependency — so import `last_assistant_text` from shared. New top:

```python
from verifiers.envs._m_ import Scorer
from verifiers.envs._m_.judge import last_assistant_text

from .constraints import CONSTRAINTS
from .scorer_types import ConstraintScore, ScorerName
from .types import TaskInfo
```

(`ConstraintScorer.score` keeps `task_info: TaskInfo` — a valid narrowing of the ABC's `object`.)

- [ ] **Step 4: `env.py`** — rewire imports + construct with new params. New imports + body:

```python
from verifiers.envs._m_ import ComposedEnv, JudgeScorer, JudgeView, Scene

from .dataset import build_dataset
from .prompts import MONITOR_PROMPTS, PROMPTS, PromptKey
from .scorer_types import JudgeConfig, ScorerName
from .scorers import ConstraintScorer
from .types import ConstraintName, Difficulty, MonitorPrompt, SceneName, TaskInfo


class TaskScene(Scene):
    """Scene 0: the instruction-following task. Opened by the dataset prompt."""

    name = SceneName.TASK

    async def enter(self, state):
        return []  # opened by the dataset prompt

    async def is_complete(self, state) -> bool:
        return True
```

In `load_environment`, the `JudgeScorer(...)` call gains `name=ScorerName.JUDGE_REQUEST_FOLLOWED`, and the return becomes:

```python
    return ComposedEnv(
        scenes=[TaskScene()],
        scorers=scorers,
        info_type=TaskInfo,
        info_enums=(ConstraintName, Difficulty),
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
        max_turns=4,
    )
```

- [ ] **Step 5: Delete `turn_logic.py` and `tests/test_turn_logic.py`**

```bash
git rm environments/instruction_following_text/instruction_following_text/turn_logic.py
git rm environments/instruction_following_text/tests/test_turn_logic.py
```

- [ ] **Step 6: Trim `tests/test_scorers.py`** — remove the render_view/parse_score/parse_judge_response tests (moved to shared in Task 5); keep ConstraintScorer tests. Fix imports to pull anything still needed from `verifiers.envs._m_`.

- [ ] **Step 7: Run env tests**

Run: `.venv/bin/python -m pytest environments/instruction_following_text/tests/ -v`
Expected: all PASS (constraints, dataset, remaining scorers).

- [ ] **Step 8: Commit**

```bash
git add environments/instruction_following_text/instruction_following_text/ environments/instruction_following_text/tests/
git commit -m "refactor(env): import shared _m_ scaffold; Turn->Scene; delete turn_logic.py"
```

---

### Task 7: Rename the env package → `_m_instruction_following_text`

**Files:**
- Rename dir `environments/instruction_following_text/` → `environments/_m_instruction_following_text/`
- Rename module `instruction_following_text/` → `_m_instruction_following_text/`
- Modify: `environments/_m_instruction_following_text/pyproject.toml`
- Update intra-package imports (`from instruction_following_text...` if any test uses absolute) — env-internal imports are relative (`.types` etc.) so unaffected; tests import `instruction_following_text.*`.

- [ ] **Step 1: git mv directory + module**

```bash
git mv environments/instruction_following_text environments/_m_instruction_following_text
git mv environments/_m_instruction_following_text/instruction_following_text environments/_m_instruction_following_text/_m_instruction_following_text
```

- [ ] **Step 2: `pyproject.toml`** — set:

```toml
[project]
name = "_m_instruction_following_text"
...
[tool.hatch.build]
include = ["_m_instruction_following_text/**"]
```

- [ ] **Step 3: Fix test imports** — in `environments/_m_instruction_following_text/tests/*.py`, change `from instruction_following_text.X import ...` → `from _m_instruction_following_text.X import ...`. Grep: `grep -rn "instruction_following_text" environments/_m_instruction_following_text/tests/`.

- [ ] **Step 4: Reinstall editable**

```bash
uv pip install --python .venv/bin/python -e environments/_m_instruction_following_text
```
Expected: installs `_m_instruction_following_text` with no error.

- [ ] **Step 5: Run env tests under new name**

Run: `.venv/bin/python -m pytest environments/_m_instruction_following_text/tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add environments/_m_instruction_following_text
git commit -m "refactor(env): rename instruction_following_text -> _m_instruction_following_text"
```

---

### Task 8: Update prime-rl consumer scripts (env id/module + `message_scenes` column)

**Files (under `/shared/matan/code/prime-rl/`):**
- Modify: `claude_scripts/run_instruction_eval.py`
- Modify: `claude_scripts/rescore_views_pearson.py`
- Modify: `claude_scripts/pooled_monitor_pearson.py`
- Modify: `claude_scripts/analysis_scripts/instruction_following_text/register.py`
- Check: `claude_scripts/analysis_scripts/instruction_following_text/flatten.py` (reads `info`; confirm no `message_turns`/module import)

- [ ] **Step 1: Find all references**

```bash
cd /shared/matan/code/prime-rl
grep -rn "instruction_following_text\|message_turns\|load_environment\|env_id\|\"instruction-following" claude_scripts/ | grep -v __pycache__
```

- [ ] **Step 2:** In each script, change the `STATE_COLUMNS` entry `"message_turns"` → `"message_scenes"`, and any env-id/module string `instruction_following_text` / `instruction-following-text` → `_m_instruction_following_text`. (The analysis package dir name `analysis_scripts/instruction_following_text/` can stay — it's the consumer package, not the env; renaming it is optional and out of scope unless trivial.)

- [ ] **Step 3: Smoke-import the scripts**

Run (from prime-rl, using the env's venv since these import verifiers): `cd /shared/matan/code/prime-rl && /shared/matan/code/prime-rl/deps/verifiers/.venv/bin/python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['claude_scripts/run_instruction_eval.py','claude_scripts/rescore_views_pearson.py','claude_scripts/pooled_monitor_pearson.py']]; print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit** (in the prime-rl repo, not the submodule)

```bash
cd /shared/matan/code/prime-rl
git add claude_scripts/run_instruction_eval.py claude_scripts/rescore_views_pearson.py claude_scripts/pooled_monitor_pearson.py claude_scripts/analysis_scripts/instruction_following_text/register.py
git commit -m "chore: point instruction-eval scripts at _m_instruction_following_text + message_scenes"
```

---

### Task 9: README worked example

**Files:**
- Modify: `environments/_m_instruction_following_text/README.md`

- [ ] **Step 1:** Append an `## Example` section: one concrete sample task (a real alpaca-style request + the `no_capitals` constraint instruction text from `constraints.py`) and the launcher command. Read `constraints.py` for the exact `no_capitals` instruction string. Command form (confirm flags against `claude_scripts/run_instruction_eval.py` `Args`):

```bash
cd /shared/matan/code/prime-rl
uv run --python deps/verifiers/.venv/bin/python claude_scripts/run_instruction_eval.py \
  --env-id _m_instruction_following_text --model deepseek/deepseek-v4-flash \
  --n-requests 5 --short-desc readme-example
```

- [ ] **Step 2: Commit**

```bash
git add environments/_m_instruction_following_text/README.md
git commit -m "docs(env): add worked example to README"
```

---

### Task 10: Rewrite `building_vf_envs_matan_way.md` (procedural)

**Files:**
- Modify: `deps/verifiers/claude_state/building_vf_envs_matan_way.md`

- [ ] **Step 1:** Restructure per spec §5: top = the 8-step procedural spine (§5a, verbatim including the cheap-model list `deepseek-v4-flash`/`gpt-oss-20b`/`qwen/qwen3-next-80b-a3b-thinking` and the a/b/c rollout-review checks); then a "Reference" half folding the existing material (§5b); add the "Scenes & provenance" subsection (§5c — the `enter` footgun + the provenance model); do the terminology/path sweep (§5d: Turn→Scene, `turn_logic.py`→shared `verifiers.envs._m_`, `message_turns`→`message_scenes`, `instruction_following_text`→`_m_instruction_following_text`, update the Registry entry). Prune statements the move contradicts (e.g. "Turn ABC + ComposedEnv live in `turn_logic.py`").

- [ ] **Step 2:** Re-read the rewritten doc once for internal consistency (no stale `Turn`/`turn_logic`/old-name references): `grep -n "\bTurn\b\|turn_logic\|message_turns\|instruction_following_text[^_]" deps/verifiers/claude_state/building_vf_envs_matan_way.md` — expected: no hits except inside the §5d "renamed from" note if any.

- [ ] **Step 3: Commit**

```bash
git add deps/verifiers/claude_state/building_vf_envs_matan_way.md
git commit -m "docs: rewrite building_vf_envs_matan_way as a procedural workflow"
```

---

### Task 11: End-to-end smoke eval (verification)

- [ ] **Step 1:** Run a tiny eval through the launcher on the renamed env with a cheap model, few samples:

```bash
cd /shared/matan/code/prime-rl
uv run --python deps/verifiers/.venv/bin/python claude_scripts/run_instruction_eval.py \
  --env-id _m_instruction_following_text --model deepseek/deepseek-v4-flash \
  --n-requests 3 --short-desc cleanup-smoke 2>&1 | tail -30
```
Expected: completes; prints a results dir.

- [ ] **Step 2: Spot-check the output** — read 3 entries of the results.jsonl. Verify: `scorers` present (constraint + judge), `message_scenes` present and covers prompt+completion (first entries `source: system`/`dataset`, then `actor`), `raw_responses` present, no `scorer_errors`.

- [ ] **Step 3:** Report the results dir path + spot-check verdict to the user. (Do not commit outputs.)

---

## Self-review notes

- **Spec coverage:** §1 shared pkg → Tasks 1-4; §enter-footgun/abstract → Task 1; §provenance → Tasks 1,3,5; §2 env rewire → Task 6; §3 `_m_` rename → Tasks 7-8; §4 README → Task 9; §5 doc → Task 10; verification → Task 11. All covered.
- **CLAUDE.md compliance:** `uv` only (no raw python); commit trailer + `refs <ID>` (Task 0 confirms id); conservative test additions (Task 5 ports/relocates existing tests + one provenance test the move necessitates; no speculative tests); type annotations + dataclasses preserved; `optimization_dtype`/`reduce_dtype` untouched.
- **Open item:** Task 0 must resolve the Linear issue id before commits use `refs`.
