# 005 Gemini lifetime

Status: complete.

A note proposed at decision t is written before that decision's reward and deployed at decision t+1. Its creation cutoff is t-1. The prompt at decision 17 therefore includes observations through action 16 and the note proposed at 16. For the second world, the first reward scored under the middle probability table is the outcome of decision 17, and it is visible starting at decision 18. Those boundaries are evaluator-side and are not part of the prompt.

Reward sums below are descriptive. Reference rows cost 0 and are not LLM methods.

| group | run | status | reward | usage estimate USD | invalid |
| --- | --- | --- | --- | --- | --- |
| smoke | smoke | completed | 2 | 0.0117705 | 0 |
| reference | w1-ucb | completed | 29 | 0 | 0 |
| reference | w1-random | completed | 28 | 0 | 0 |
| pilot | w1-recent | completed | 26 | 0.1603080 | 0 |
| pilot | w1-persistent | completed | 34 | 0.1494135 | 0 |
| reference | w2-ucb | completed | 29 | 0 | 0 |
| reference | w2-random | completed | 28 | 0 | 0 |
| pilot | w2-recent | completed | 35 | 0.1445670 | 0 |
| pilot | w2-persistent | completed | 32 | 0.1418085 | 0 |

Evaluator-only window counts are in summary.json under each run's windows.
