# Sequential selective revision: dry-run

CPU-only frozen tapes. No Gemini calls.

- Tapes: 8 (3 seeds × shared/independent + 2 stationary controls)
- Jobs: 96 (4 checkpoints × 3 conditions)
- Retry reserve: 192 attempts
- Max heuristic prompt tokens: 1363
- Illustrative reserve at 4000 in / 10240 out: $18.8467

Optimal v1-mapped actions across 32 tape×probe cells: choose_B=12, choose_safe=6, diagnose_B=14

Live execution is not authorized by this dry-run. Commander reviews fixtures first.

Figure: `reports/selective_revision_sequential_v1_dryrun/figures/shared_s0_linked.png`
