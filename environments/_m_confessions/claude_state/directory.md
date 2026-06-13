# _m_confessions — state

- `paper_summary.md` — notes on "Training LLMs for Honesty via Confessions" (arXiv 2512.08093), esp. §2.
- The env composes the instruction-following task (`m-instruction-following-text`) with a confession
  scene via `vf.EnvGroup`. Confession tokens are the only loss-bearing ones (Scene.trainable masking).
- Design spec: `deps/verifiers/claude_state/specs/2026-06-13-m-confessions-design.md`.
