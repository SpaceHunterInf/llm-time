# RSI under Non-Stationary Reward Paradigms

Investigating how self-evolving LLM agents behave—and how they can adapt without paying a prohibitive verification token tax—when the underlying **reward/utility paradigm** shifts silently over a single continuous lifetime (e.g., decaying market alpha, utility reversal, evolving user lifecycle needs).

For the 23 November group meeting, open the [offline meeting website](results/supervisor_dashboard.html). Gemini’s [original six-section roadmap dashboard](results/roadmap_dashboard.html) is preserved as it appeared on 21 November. The [discussion record](setup-debates/2026-11-22-group-meeting-brief.md) links the claims to the original runs and access audit.

## Core Thesis
Modern non-parametric self-improvement methods (`GEPA`, `ACE`, `Meta-Harness`, `ReasoningBank`, `SRN`) crystallize past successes into persistent rules, prompts, or historical scores. When the **reward paradigm** shifts silently (without tool crashes or syntax errors), self-improvement turns into an **anti-adaptation ratchet**: self-authored rules and accumulated win-counts talk the agent back into stale policies ("hypothesis resurrection").

We study this across a **3-Act Research Arc**:
1. **Act 1 — Diagnosis & Benchmark Suite:** Quantify the *Self-Evolution Ratchet* across frontier LLMs (Gemini, DeepSeek) and modern RSI harnesses on external drift suites (`ServeLearnBench`, `ReScopeBench`) + our controlled **Reward-Paradigm Drift Probe** (abrupt utility reversal, continuous alpha decay, maturing-user utility, and regime recurrence).
2. **Act 2 — Surprise-Gated Epistemic Harness:** Lightweight meta-wrapper using **prediction-error / likelihood-ratio gating** (eliminating the 75–94% continuous verification "RSI Tax"), **scoped hypothesis shelving**, **active discriminative probing**, and **post-changepoint evidence resets**.
3. **Act 3 — Internalizing Epistemic Plasticity (LoRA / GRPO):** Distill and RL-train open-weight models (`Qwen3-8B` / `Gemma-3-27B`) on high-volume Gemini-generated drift trajectories so native in-context doubt, diagnostic probing, and durable retraction emerge without multi-call harness overhead.

## Repository Structure
- `AGENTS.md` — Operating rules for AI research co-leads.
- `RESEARCH_STATE.md` — Live lab notebook, 4-week submission plan, and active experiment specs.
- [Setup debates](setup-debates/README.md) — Research design reviews, replies, and decisions.
- `LITERATURE.md` — Categorized literature map (through Nov 2026) and novelty positioning.
- `src/` — Core environments, agent harnesses, and evaluation runners.
- `configs/` — Reproducible experiment configurations.
- `experiments/` — Experiment specs and turnkey launch scripts (Quadro / CSD3 / Collaborators).
- `results/` — Tracked run reports, dashboards (`supervisor_dashboard.html`, `roadmap_dashboard.html`), meeting PDF (`group_meeting_2026-11-23.pdf`), ledger, and raw JSONL traces (`results/raw/`).
- `competitors/` — Local checkouts of `ServeLearnBench` and `ReScopeBench` (gitignored).
- `archive/` — Historical pre-November 2026 code, reports, and notes.

## Quickstart (Quadro / CSD3 / Collaborators)
```bash
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$PATH"
cp .env.example .env   # Fill GEMINI_API_KEY / DEEPSEEK_API_KEY if running API evals
uv sync
```
