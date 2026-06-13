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
