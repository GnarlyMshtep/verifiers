# IFT constraint calibration tweak log (MSH-195)

Calibration actor: `deepseek/deepseek-v4-flash` (OpenRouter), judge = local qwen3-4b-i (output/immediate),
n_requests=50 per constraint, max_tokens=8192, temperature=0.7. Band target: per-constraint compliance 20–70%.

## Round 0 — first-draft knobs

Run: `ift_calib-549fa3-20_33` (250 rollouts, all 5 constraints).

| constraint | knobs | compliance | verdict |
|---|---|---|---|
| no_capitals | — (control) | 0.98 | control, expected trivial |
| all_sentences_t | num_sentences∈[2,5] | 0.44 | ✅ in band |
| alternating_xy | n∈[2,5], a,b∈[5,20] | 0.58 | ✅ in band |
| letter_freq_diff | `_FREQ_DELTA_RANGE=(2,3)` | **0.90** | ❌ too easy |
| letter_set | include rarest-tail×2, exclude mid×2 | 0.44 | ✅ in band |

Decision: only `letter_freq_diff` out of band (too easy). deepseek-flash is the strongest of the three
sweep actors, so 0.90 here is the upper bound — bring it into ~0.5–0.65 so weaker actors stay ≥0.20.
Lever per plan: raise `_FREQ_DELTA_RANGE`.

### Tweak R0→R1
- `letter_freq_diff`: `_FREQ_DELTA_RANGE` (2,3) → (3,5). (Every one of N sentences must have the surplus,
  so +1 delta compounds across sentences.)

## Round 1 — re-probe hard constraints (`ift_calib_r1hard-fb952d-20_41`, 100 rollouts)

| constraint | knobs | compliance | verdict |
|---|---|---|---|
| alternating_xy | unchanged | 0.70 | ✅ band edge |
| letter_freq_diff | `_FREQ_DELTA_RANGE=(3,5)` | 0.82 | ❌ still too easy |

Diagnosis: delta barely bites because the sampler pairs a very-common `y` (e.g. `e`) with a near-zero
`z` (e.g. `z`/`q`) — the surplus is essentially free. Delta is the wrong lever.

### Tweak R1→R2
- `letter_freq_diff`: draw BOTH letters from the common head `ENGLISH_FREQ_ORDER[:_FREQ_HEAD_RANK]`
  (`_FREQ_HEAD_RANK=10`, i.e. {e,t,a,o,i,n,s,h,r,d}). Removes the trivial common-vs-rare pairing so the
  per-sentence surplus is genuinely contested; keeps it feasible (no near-zero letter). Delta stays (3,5).

## Round 2 — re-probe hard constraints (`ift_calib_r2hard`, 100 rollouts)

| constraint | knobs | compliance | verdict |
|---|---|---|---|
| alternating_xy | unchanged | 0.60 | ✅ in band |
| letter_freq_diff | head-restricted, `_FREQ_DELTA_RANGE=(3,5)` | 0.86 | ❌ went UP — wrong lever |

Diagnosis: restricting to common letters made it EASIER. Root cause: whenever y is more common than z
(enforced for feasibility), natural English ALREADY clears the surplus — the gap between a common y and
any rarer z is wide. The lever isn't delta or pool; it's the FREQUENCY GAP between y and z.

### Tweak R2→R3
- `letter_freq_diff`: replace head/delta levers with FREQUENCY ADJACENCY. Draw y,z as neighbors
  (`_FREQ_ADJ_GAP=(1,2)` ranks apart) from `ENGLISH_FREQ_ORDER[:_FREQ_POOL_RANK]` (`_FREQ_POOL_RANK=14`),
  y the more common. Close natural rates ⇒ a per-sentence surplus is genuinely hard. Delta back to (2,3).

## Round 3 — re-probe hard constraints (`ift_calib_r3hard-ae6a51-20_50`, 100 rollouts) — FINAL

| constraint | knobs | compliance | verdict |
|---|---|---|---|
| alternating_xy | n∈[2,5], a,b∈[5,20] | 0.60 | ✅ in band |
| letter_freq_diff | adjacency `_FREQ_ADJ_GAP=(1,2)`, pool 14, delta (2,3) | 0.74 | ⚠️ band edge (strongest actor) |

Decision: **accept** (3-round cap reached). Adjacency lever worked (0.86→0.74). 0.74 is on deepseek-flash,
the strongest of the three sweep actors; weaker actors (qwen3-8b, nemotron-nano) will pull it into band,
so the cross-actor figure should land ≤0.70. Flag `letter_freq_diff` as a borderline-high constraint to
watch in the Task-14 per-constraint analysis; do NOT re-tune mid-sweep (would desync params across actors).

### Final calibrated compliance (deepseek-flash, strongest actor)
| no_capitals (control) | all_sentences_t | alternating_xy | letter_freq_diff | letter_set |
|---|---|---|---|---|
| 0.98 | 0.44 | 0.60 | 0.74 | 0.44 |

No constraint dropped — all feasible. Controls behave as intended (`no_capitals` trivially easy).
