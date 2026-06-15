# Tweak Log — Self-Incrimination Full Experiment (2026-06-15)

Running log of decisions, deviations, failures, and fixes during the overnight autonomous run.

| time (UTC) | stage | event |
|---|---|---|
| 05:50 | setup | Plan locked; GPUs 0+1 (jobs 14078/14079) idle on bleak-mushroom-dove. Actor+SFT=Qwen3-4B, judge=Qwen3-4B-I. Starting C1 (ift start_index). |
| 05:55 | C1-C3 | C1 (ift start_index, splits provably disjoint, 56 tests), C2 (prepare_dataset fixed-N mode + local-judge passthrough), C3 (export_prime_sft, parquet round-trip via load_dataset(dir,None,split=train) verified) all done + committed. |
| 05:58 | R1 | Served Qwen3-4B actor (GPU0:8000, --reasoning-parser qwen3) + Qwen3-4B-I judge (GPU1:8001) via vLLM. Both ready in ~10s. KV cache ~108GB each. |
| 06:02 | R1 | 4-sample smoke PASS: malicious=confidently-wrong+sentinel, wrong-but-hedged=HONEST(no sentinel), judge_input has no CoT. Launched full 400-sample fixed-mode gen (max_concurrent=32) in background. C4 eval harness building in background. |
