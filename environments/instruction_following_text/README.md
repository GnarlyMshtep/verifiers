# instruction_following_text

> **Authored env (not upstream verifiers).** See `deps/verifiers/claude_state/building_vf_envs_matan_way.md`.

Tasks = an Alpaca user-request paired with a rule-checkable formatting constraint.
Reward = rule-based constraint verifier (per-turn) + LLM judge scoring request execution
in [0,1] (judge view configurable: output / cot / both). Built on the composable
`Turn` / `ComposedEnv` scaffold so confession and tool turns drop in later.
