"""Adapter bridging our `EpistemicHarness` implementations directly into `ServeLearnBench`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.core.ledger import CostLedger
from src.core.llm_client import MeteredLLMClient
from src.envs.base import PublicTaskSpec
from src.harnesses import EpistemicHarness, PublicEpisodeRecord, make_harness


def ensure_slb_on_path(workspace_root: Path) -> None:
    slb_root = (workspace_root / "competitors" / "ServeLearnBench").resolve()
    if slb_root.is_dir() and str(slb_root) not in sys.path:
        sys.path.insert(0, str(slb_root))


def build_slb_method(
    *,
    workspace_root: Path,
    harness_name: str,
    reflection_model: str,
    ledger: CostLedger,
    block_size: int = 8,
) -> Any:
    """Return an instance of `servelearnbench.methods.Method` wrapping our `EpistemicHarness`.

    Protocol compliance (`competitors/ServeLearnBench/docs/methods.md`):
    - Learns strictly inside `update()` from `r.instruction`, `r.timestamp`, `r.reward`, `r.record`.
    - `context()` never inspects `phase`, `window`, `task.z`, `task.actions`, or `task.answer`.
    - All external reflection calls are logged to `CostLedger`.
    """
    ensure_slb_on_path(workspace_root)
    from servelearnbench.methods import Method  # type: ignore[import-not-found]

    class WrappedEpistemicSLBMethod(Method):  # type: ignore[misc]
        name = f"rsi_{harness_name}"
        learns = harness_name != "blind"

        def __init__(self) -> None:
            super().__init__()
            self.block_size = int(block_size)
            self._harness: EpistemicHarness = make_harness(harness_name)
            self._client = MeteredLLMClient(
                model=reflection_model,
                ledger=ledger,
                dotenv_path=workspace_root / ".env",
                thinking_budget=256,
            )

        def setup(self, bundle: Any) -> None:
            super().setup(bundle)
            self._harness = make_harness(harness_name)

        def context(self, task: Any, phase: str, window: int) -> str:
            # Strictly ignore `phase` and `window` per ServeLearnBench protocol!
            _ = (phase, window)
            dummy_spec = PublicTaskSpec(
                env_name="servelearnbench",
                episode_idx=0,
                timestamp=int(getattr(task, "timestamp", 0) or 0),
                domain_prompt="",
                task_prompt=str(getattr(task, "instruction", "")),
                tool_schemas=[],
            )
            return self._harness.build_context(dummy_spec)

        def update(self, results: list[Any]) -> None:
            for idx, r in enumerate(results):
                score_100 = float(getattr(r, "reward", 0.0)) * 100.0
                rec_text = str(getattr(r, "record", ""))[:1200]
                pub = PublicEpisodeRecord(
                    env_name="servelearnbench",
                    episode_idx=idx,
                    timestamp=int(getattr(r, "timestamp", idx + 1) or (idx + 1)),
                    task_prompt=str(getattr(r, "instruction", ""))[:400],
                    steps=[{"turn": 1, "tool": "slb_episode", "args": {}, "obs": {"trace": rec_text}}],
                    public_score=score_100,
                )
                self._harness.on_episode_complete(pub, self._client)

    return WrappedEpistemicSLBMethod()
