# RSI-paradigm-shift

## Goal
Investigate LLM epistemic plasticity and belief revision in non-stationary reward paradigms.

## 3-Act Research Plan

1. **Act 1: Benchmark & Taxonomy**
   - Evaluate frontier models on `ReScopeBench` and `ServeLearnBench`.
   - Implement the `Paradigm-Shift Probe` to target hidden reward-paradigm shift (e.g., utility-function reversal, causal coupling).
   - Establish baseline failure taxonomy.

2. **Act 2: Epistemic Harness**
   - Implement falsifiable-hypothesis tracking.
   - Mechanism: Agent maintains hypotheses with `Scope`, `Prediction`, `Condition`.
   - Goal: Reduce "RSI Tax" (token cost) by avoiding unnecessary verification, instead using surprise-based triggering and scoped revalidation.

3. **Act 3: Internalization (LoRA Training)**
   - Use Gemini to generate high-volume drift trajectories for training.
   - Fine-tune an open model (e.g., Qwen3-8B / Gemma-3-27B) via GRPO / RFT to internalize:
     - Epistemic doubt.
     - Discriminative probing.
     - Robust adherence to revised hypotheses under OOD shift.

## Active Status
- **Current Task:** Setup & Reorganization.
- **Reference Workspace:** `C:\Users\nzhux\Desktop\llm + time`
- **Quadro Sync:** `dev-gpu-xz479.cl.cam.ac.uk:~/llm-time`
