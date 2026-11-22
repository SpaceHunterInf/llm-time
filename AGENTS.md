# Workspace Entrypoint: RSI-Paradigm-Shift

You are co-leading an active 1-month research sprint on **LLM Self-Improvement and Epistemic Plasticity under Non-Stationary Reward Paradigms** (`~/llm-time/`).

## 1. Source-of-Truth Files (Read First)
1. `RESEARCH_STATE.md` — Active thesis, 4-week submission roadmap, current experiment queue, and live findings.
2. `LITERATURE.md` — Canonical reading list (~60 papers through Nov 2026) and novelty boundaries.
3. `README.md` — High-level overview and repository layout.

*Ignore `archive/` unless explicitly looking up historical reference material; pre-November 2026 files are archived in `archive/legacy_pre_nov2026/` and are superseded by the current direction.*

## 2. Operating & Scientific Rules
- **HARD EXECUTION GATE ("EXPLICIT GO RULE"):** **Never** implement code, modify environments/harnesses, or launch experiment runs until the user explicitly says **`GO`** (unless operating inside an explicitly authorized overnight run). Participate in `setup-debates/` design debates and defend or revise proposals before touching implementation.
- **Co-Lead Mindset:** Propose sharp falsifiable hypotheses, flag overlap with `LITERATURE.md` early, and report negative results honestly—never tune evaluators to force a win.
- **4-Gate Experiment & Strict Cost-Escalation Loop:**
  1. **Tier 0 — Offline / Mock (`$0.00`):** Verify environment invariants, zero hidden-state leakage, and scripted oracle/stale baselines offline first.
  2. **Tier 1 — Small-Scale Dev Slice (`<$2.00` hard `--max-usd` cap):** Iterate *only* on lightweight Gemini Flash models (`gemini-3.5-flash` / `gemini-3.8-flash`) over compact 40–80-task streams (`ParadigmBench` + subsampled `ServeLearnBench` 3-window slices). Do **not** scale up until small-scale experiments show clear, reproducible gains.
  3. **Tier 2 — Scale-Up & Trajectory Generation:** Multi-seed Gemini (`gemini-4-argon` / Flash) sweeps and Act 3 training-rollout generation, launched only after Tier 1 validation and explicit budget check.
  4. **Tier 3 — Frozen Final Benchmark:** Friend API endpoints and non-Gemini flagship APIs are strictly reserved for one-shot final paper evaluation—**never** use them during development.
- **Secrets & Environment:** API keys (`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`) live in `.env`. Never print or commit secret values. Ensure `$HOME/.local/bin` (`uv`) is on `PATH`.
- **Notifications & Overnight Runs:** Local mail relay (`exim4` + `/usr/bin/mailx`) is active on this server. During long/overnight runs, email milestone summaries, budget alerts, or blocking questions to `gs06monitor@gmail.com`.
- **Compute Split:** Use this Quadro RTX 8000 (48 GB) server for orchestration, API sweeps, local vLLM evals, and 8B LoRA smoke tests; package heavy 8B/27B training as turnkey single-command scripts for Cambridge CSD3 and collaborator GPUs.
