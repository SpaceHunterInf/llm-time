# LLM RL Engine Spec

For the current research direction, audit, and Grok implementation queue, start with [AUTO_RESEARCH-commander.md](AUTO_RESEARCH-commander.md) and [auto_research_rules.md](auto_research_rules.md). The build-order notes below describe the original engine plan.

This folder specifies a lightweight reinforcement-learning engine for frozen LLM agents. The goal is to test whether an LLM can explore, infer hidden structure, adapt online, and use prior interaction history to choose better future actions without any model-weight updates.

The initial benchmark should start with simple static environments and then extend to dynamic, time-dependent environments.

Core research question:

> Can frozen LLM agents perform long-horizon test-time adaptation through context alone?

Secondary research question:

> Do explicit heuristics, generated or provided before action selection, improve adaptation in static and dynamic environments?

## Recommended Build Order

1. Implement the engine API and logging format.
2. Implement three simple environments:
   - NumberGuess
   - StationaryBandit
   - ReversalBandit
3. Implement one dynamic environment:
   - PeriodicBandit
4. Add baseline agents:
   - random
   - greedy
   - UCB
   - Thompson sampling
   - sliding-window UCB
5. Add LLM agent wrappers:
   - raw-history prompt
   - summarized-history prompt
   - heuristic-before-action prompt
6. Run smoke tests over short horizons.
7. Run full benchmarks over 20-40+ steps per episode.

## Files

| File | Purpose |
|---|---|
| `01_engine_architecture.md` | Core engine abstractions, environment API, run loop, logging. |
| `02_environment_specs.md` | Environment definitions and hidden-parameter variants. |
| `03_agent_protocol.md` | Agent input/output protocol, prompt modes, validation. |
| `04_benchmark_protocol.md` | Evaluation design, baselines, metrics, seeds, comparisons. |
| `05_heuristic_memory_experiments.md` | Experimental conditions for generated/provided heuristics. |
| `06_implementation_plan.md` | Concrete development milestones and acceptance checks. |

## Design Principles

- Keep the base engine model-agnostic.
- Treat the LLM as a frozen policy with text context as its adaptive state.
- Make hidden environment parameters configurable and seedable.
- Log every observation, action, reward, prompt, response, and parsed decision.
- Separate environment difficulty from prompt/agent architecture.
- Include classical RL baselines so LLM behavior is interpretable.
- Use long horizons by default, not only toy 5-step episodes.

## Key Terms

| Term | Meaning |
|---|---|
| Episode | One complete interaction sequence with a fixed environment instance. |
| Horizon | Number of steps in an episode. Initial target: 20-40; later: 100+. |
| Static environment | Hidden parameters do not change over time. |
| Dynamic environment | Reward/action structure changes over time. |
| Latent parameter | A hidden property such as reward probability, phase, period, or target number. |
| Heuristic | A compact natural-language rule used to guide future action choices. |
| In-context adaptation | Behavior improves because the agent uses history in the prompt/context, not because weights change. |
