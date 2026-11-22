# Ordinary-KL learning pilot — live GPU results

14 September 2026. User-authorized nine-run fit on this machine’s Quadro RTX 8000. No paid APIs. The paired loss is not used. 008 sampled trajectories were not regenerated. This is the bounded ordinary-KL pilot from tasks 008–009, not a new method.

Selection used **dev KL only**. Holdouts were opened only after all nine `best.pt` files existed, via `--allow-holdout`. Fit manifests record `heldout_read: false`. All three seeds are reported; none is dropped.

Raw: `runs/learning_pilot/` (gitignored). Compact tables: `reports/learning_pilot/live_tables.json`. Driver log: `reports/learning_pilot/gpu_pilot.log`.

## Protocol (unchanged)

| Item | Value |
|---|---|
| Loss | teacher→student joint KL |
| Model | Qwen3-0.6B-Base `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, no chat template |
| Data | v2 render of frozen 008 records, seed 808; 2048 / 256 / 256 / 256 |
| Controls | `mlp` (47-d numeric, 8 272 params) × seeds 0/1/2; `head` (16 400); `lora` r=8 α=16 q/v, adapter_seed=seed+10007 (1 163 280) |
| Opt | AdamW, float32, effective batch 32, microbatch 8, 192 steps, 3 epochs |
| LR | head/MLP 1e-3, LoRA 1e-4; 10% warmup; clip 1.0; wd 0.01 non-bias |
| Select | epoch-end **dev KL**; every run’s `best` is step **192** |
| Device | CUDA `torch==2.6.0+cu124` in isolated extra `cu124`; dtype **not** changed |

Hardware: Quadro RTX 8000, 49152 MiB, driver 545.23.08, compute 7.5. No native BF16; float32 kept.

## Wall clock (this GPU)

Checkpoint mtimes, sequential driver `scripts/run_learning_pilot_local_gpu.py`:

| Control | Fit wall / seed | `best.pt` |
|---|---|---|
| mlp | ~11 s | ~128 KB |
| head | ~8 min | ~221 KB |
| lora | ~20 min | ~14.1 MB |

Nine fits then 36 evals (train/dev/same_topology/structural × 9). Driver log ends 19:15; first MLP checkpoint 17:23; **~1 h 52 min** total. GPU probe before launch (load + one warmed step): head ~7 s / 7.1 GB; LoRA ~10.5 s / 15.1 GB. Peak stayed well under 48 GB. Spend **$0**.

## Joint KL (all seeds)

Mean of three training seeds; parentheses are min–max. Teacher joint KL, same scorer as eval.

| Control | train | dev | same_topology | structural |
|---|---:|---:|---:|---:|
| mlp | 0.2854 (0.2782–0.2940) | 0.2935 (0.2851–0.3038) | 0.2779 (0.2710–0.2872) | 0.3281 (0.3220–0.3368) |
| head | 0.2843 (0.2813–0.2865) | 0.2891 (0.2873–0.2916) | 0.2780 (0.2755–0.2796) | 0.3331 (0.3303–0.3363) |
| lora | 0.2304 (0.2248–0.2351) | 0.2368 (0.2335–0.2385) | 0.2225 (0.2195–0.2255) | 0.2951 (0.2851–0.3017) |

Per seed (eval of `best.pt`):

| Run | train | dev | same_topology | structural | selected step |
|---|---:|---:|---:|---:|---:|
| mlp_s0 | 0.2940 | 0.3038 | 0.2872 | 0.3368 | 192 |
| mlp_s1 | 0.2840 | 0.2916 | 0.2754 | 0.3254 | 192 |
| mlp_s2 | 0.2782 | 0.2851 | 0.2710 | 0.3220 | 192 |
| head_s0 | 0.2813 | 0.2873 | 0.2755 | 0.3303 | 192 |
| head_s1 | 0.2850 | 0.2916 | 0.2789 | 0.3327 | 192 |
| head_s2 | 0.2865 | 0.2883 | 0.2796 | 0.3363 | 192 |
| lora_s0 | 0.2248 | 0.2335 | 0.2195 | 0.2851 | 192 |
| lora_s1 | 0.2312 | 0.2383 | 0.2227 | 0.3017 | 192 |
| lora_s2 | 0.2351 | 0.2385 | 0.2255 | 0.2986 | 192 |

Epoch-end **dev** KL (the selection statistic) still fell at the last epoch in every run:

| Run | step 64 | step 128 | step 192 |
|---|---:|---:|---:|
| mlp_s0 | 0.4416 | 0.3173 | 0.3038 |
| mlp_s1 | 0.3858 | 0.3052 | 0.2916 |
| mlp_s2 | 0.4034 | 0.2989 | 0.2851 |
| head_s0 | 0.3809 | 0.3090 | 0.2873 |
| head_s1 | 0.4330 | 0.3243 | 0.2916 |
| head_s2 | 0.3576 | 0.3076 | 0.2883 |
| lora_s0 | 0.3439 | 0.2683 | 0.2335 |
| lora_s1 | 0.3354 | 0.2753 | 0.2383 |
| lora_s2 | 0.3681 | 0.2590 | 0.2385 |

## Cheap references (same teacher KL)

Non-learned joints scored with the same `kl_div(teacher, ·)`. Exact is 0. `independent_sameinfo` = product of supplied `q_previous` marginals, then singleton-group suffix filter (no prefix replay). `independent_fullprefix` is the 008 historical number.

| Split | exact | stale | transition-only r | independent_sameinfo | independent_fullprefix (hist.) |
|---|---:|---:|---:|---:|---:|
| train | 0 | 0.1578 | 0.1202 | 0.0168 | 0.0204 |
| dev | 0 | 0.1811 | 0.1338 | 0.0176 | 0.0195 |
| same_topology | 0 | 0.1537 | 0.1197 | 0.0198 | 0.0233 |
| structural | 0 | 0.1541 | 0.1162 | 0.0806 | 0.0979 |

Hazard split of those refs:

| Split | h=0 indep / stale / r | h=.03 indep / stale / r |
|---|---|---|
| train | 0 / 0.0169 / 0.0169 | 0.0223 / 0.2047 / 0.1547 |
| structural | 0 / 0.0270 / 0.0270 | 0.1075 / 0.1965 / 0.1460 |

Every learned control is **worse** than stale, r, and independent_sameinfo on every split. LoRA structural mean 0.295 > stale 0.154 > r 0.116 > independent_sameinfo 0.081. LoRA train 0.230 is still above stale 0.158.

## Roles, hazard, topology

Mean of three seeds. Role MAE averages eligible components within example, then examples. Empty roles stay unused (not zero). Structural: 199/256 examples have a related-unobserved component; 139 have an unrelated one. Same-topology: 70 related, 229 unrelated.

| Control | split | related MAE | unrelated MAE | observed MAE | h=0 KL | h=.03 KL | mean regret |
|---|---|---:|---:|---:|---:|---:|---:|
| mlp | same_topology | 0.0753 | 0.0601 | 0.1038 | 0.2461 | 0.2884 | 0.0097 |
| head | same_topology | 0.0774 | 0.0487 | 0.1116 | 0.1718 | 0.3134 | 0.0085 |
| lora | same_topology | 0.0662 | 0.0401 | 0.0992 | 0.0467 | 0.2812 | 0.0072 |
| mlp | structural | 0.0695 | 0.0658 | 0.0839 | 0.1862 | 0.3753 | 0.0125 |
| head | structural | 0.0746 | 0.0618 | 0.1010 | 0.1989 | 0.3779 | 0.0104 |
| lora | structural | 0.0639 | 0.0553 | 0.0961 | 0.0807 | 0.3666 | 0.0086 |

Holdout topology types (mean of seed-wise means):

| Control | same-topo four_singletons | same-topo one_pair | structural two_pairs | structural triple |
|---|---:|---:|---:|---:|
| mlp | 0.2576 | 0.2982 | 0.3246 | 0.3316 |
| head | 0.2460 | 0.3101 | 0.3222 | 0.3440 |
| lora | 0.1933 | 0.2518 | 0.2693 | 0.3209 |

Structural family-bootstrap 95% intervals (analysis seed 909, 1000 resamples of 256 families; **per training seed**, not pooled as extra worlds):

| Run | mean | lo | hi |
|---|---:|---:|---:|
| mlp_s0 | 0.3368 | 0.2805 | 0.3970 |
| mlp_s1 | 0.3254 | 0.2726 | 0.3824 |
| mlp_s2 | 0.3220 | 0.2692 | 0.3796 |
| head_s0 | 0.3303 | 0.2802 | 0.3854 |
| head_s1 | 0.3327 | 0.2801 | 0.3894 |
| head_s2 | 0.3363 | 0.2853 | 0.3911 |
| lora_s0 | 0.2851 | 0.2388 | 0.3320 |
| lora_s1 | 0.3017 | 0.2516 | 0.3514 |
| lora_s2 | 0.2986 | 0.2511 | 0.3466 |

Safe-optimal remains rare (structural component-decisions: choose_linked 723, diagnose_linked 293, choose_safe **8** / 1024). Worst structural IDs concentrate on `structural|triple|184,194,244` and `two_pairs|99` across seeds.

## Against the table fixed before results

Astra’s 009 mapping, applied to these numbers. This is not a paper claim and not a new method.

- **Numeric MLP competitive → this is not an LLM-method project.** Head (frozen Qwen + linear 16-logit head) matches MLP on every split to seed noise (structural 0.333 vs 0.328). Text does not beat the 47-d numeric encoder here.
- **LoRA+KL transfers and preserves unrelated forecasts → do not invent a special loss.** LoRA is better than both MLP and head on train, dev, same-topology, and structural (~0.05–0.06 joint KL). Unrelated MAE is slightly lower, not higher. Ordinary KL is enough to get that gap. It is **not** enough to match the cheap filters: LoRA still loses to stale/r/independent on absolute teacher KL. Structural h=.03 stays ~0.37 for all three controls. LoRA’s structural h=0 drop (0.081 vs MLP/head ~0.19) is real across seeds; it is still far from the h=0 independent ref (0).
- **Train/same-topology fit, structural fail → inspect representation/coverage/capacity first.** Same-topology is close to train for all three. Structural is worse by ~0.05–0.07. The gap is on both two_pairs and triple; LoRA’s remaining error is concentrated on triple (0.321 vs two_pairs 0.269). Transfer is incomplete. This coexists with poor absolute fit, so it is not “good in-domain, only structure fails.”
- **Poor fit → bounded optimization diagnosis.** All nine selected checkpoints are the last step. Dev KL was still falling. Learned train KL remains above stale. Treat 192 steps / these LRs as an unfinished optimization, not as evidence that the architecture cannot represent the filter.
- **Mixed seeds / thin cells → inconclusive; do not pick a seed.** Seed ranges are small and ordered the same way on every split. Not mixed. Safe-optimal cells stay thin; do not read decision metrics as a powered result. Three seeds are not a significance claim.

## Stop

No further runs from this worker. No dtype/microbatch change, no extra seeds, no paired loss, no new architecture. Commander/user review of these numbers decides whether the next step is longer ordinary-KL optimization, a representation/coverage diagnosis, or stopping the supplied-filter LLM framing.
