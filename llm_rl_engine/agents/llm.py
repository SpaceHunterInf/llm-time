from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_rl_engine.agents.base import Agent, AgentStepInfo
from llm_rl_engine.types import Action, Observation, TokenUsage

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_CLIENT_LOCK = threading.Lock()
_CLIENT_CACHE: dict[str, Any] = {}


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ModelClient(Protocol):
    def generate(self, prompt: str) -> GenerationResult:
        ...


class GeminiClient:
    """Google Gemini API client. Set GEMINI_API_KEY in .env or environment."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def _get_client(self):
        from google import genai

        with _CLIENT_LOCK:
            if self.api_key not in _CLIENT_CACHE:
                _CLIENT_CACHE[self.api_key] = genai.Client(api_key=self.api_key)
            return _CLIENT_CACHE[self.api_key]

    def generate(self, prompt: str) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or export it before running."
            )
        client = self._get_client()
        interaction = client.interactions.create(
            model=self.model,
            input=prompt,
            generation_config={"temperature": self.temperature},
        )
        usage = getattr(interaction, "usage", None)
        input_tokens = int(getattr(usage, "total_input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "total_output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))
        return GenerationResult(
            text=interaction.output_text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    raise ValueError("No JSON object found in response.")


def render_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no prior steps)"
    lines = []
    for rec in history:
        step = rec.get("step", "?")
        action = rec.get("action", {})
        reward = rec.get("reward", 0.0)
        obs = rec.get("observation", {})
        msg = obs.get("message", "")
        lines.append(f"Step {step}: action={json.dumps(action)}, reward={reward}, obs={msg}")
    return "\n".join(lines)


def render_observation(observation: Observation) -> str:
    return json.dumps(observation, indent=2)


class LLMAgent(Agent):
    def __init__(
        self,
        name: str = "llm_raw_history",
        model_client: ModelClient | None = None,
        prompt_mode: str = "raw_history",
        max_retries: int = 1,
        invalid_action: Action | None = None,
    ) -> None:
        self.name = name
        self.model_client = model_client or GeminiClient()
        self.prompt_mode = prompt_mode
        self.max_retries = max_retries
        self._task_spec: dict[str, Any] = {}
        self._template = self._load_template(prompt_mode)
        self._invalid_action = invalid_action or {"arm": 0}

    def reset(self, task_spec: dict[str, Any], seed: int | None = None) -> None:
        self._task_spec = task_spec
        env_name = task_spec.get("name", "")
        if env_name == "NumberGuess":
            self._invalid_action = {"guess": task_spec.get("target_min", 0)}
        elif env_name == "PendulumGate":
            self._invalid_action = {"action": "wait"}

    def act(self, observation: Observation, history: list[dict[str, Any]]) -> Action:
        action, _ = self.act_with_info(observation, history)
        return action

    def act_with_info(
        self, observation: Observation, history: list[dict[str, Any]]
    ) -> tuple[Action, AgentStepInfo]:
        prompt = self._build_prompt(observation, history)
        info = AgentStepInfo(prompt=prompt)
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                current_prompt = prompt
                if attempt > 0 and last_error:
                    current_prompt = (
                        prompt
                        + "\n\nYour previous response was invalid: "
                        + last_error
                        + "\nReturn exactly one valid JSON object."
                    )
                result = self.model_client.generate(current_prompt)
                info.raw_response = result.text
                info.token_usage = TokenUsage(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                )
                parsed = extract_json_object(result.text)
                info.parsed_response = parsed
                action = parsed.get("action", parsed)
                if not isinstance(action, dict):
                    raise ValueError("'action' field must be a JSON object.")
                info.parse_error = None
                return action, info
            except Exception as exc:
                last_error = str(exc)
                info.parse_error = last_error
                info.raw_response = info.raw_response or last_error

        return dict(self._invalid_action), info

    def _build_prompt(self, observation: Observation, history: list[dict[str, Any]]) -> str:
        return self._template.format(
            task_description=self._task_spec.get("description", ""),
            action_space=self._task_spec.get("action_space", ""),
            raw_history=render_history(history),
            observation=render_observation(observation),
        )

    def _load_template(self, mode: str) -> str:
        template_map = {
            "raw_history": "raw_history.txt",
            "heuristic_before_action": "heuristic_before_action.txt",
        }
        filename = template_map.get(mode, "raw_history.txt")
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return DEFAULT_RAW_HISTORY_TEMPLATE


DEFAULT_RAW_HISTORY_TEMPLATE = """You are an agent in a reinforcement-learning environment.
Your goal is to maximize cumulative reward over this episode.

Task:
{task_description}

Action space:
{action_space}

History:
{raw_history}

Current observation:
{observation}

Return exactly one JSON object with this schema:
{{
  "action": <your action object>,
  "confidence": <number from 0 to 1>,
  "rationale": <string explaining your choice>
}}
"""


LLM_AGENT_NAMES = {
    "llm_raw_history": "raw_history",
    "llm_heuristic_before_action": "heuristic_before_action",
    "llm": "raw_history",
}


def is_llm_agent_config(config: dict[str, Any]) -> bool:
    name = config.get("name", "")
    return name.startswith("llm") or name in LLM_AGENT_NAMES


def make_llm_agent(config: dict[str, Any]) -> LLMAgent:
    name = config.get("name", "llm_raw_history")
    prompt_mode = config.get("prompt_mode") or LLM_AGENT_NAMES.get(name, "raw_history")
    client = GeminiClient(
        model=config.get("model", "gemini-3.5-flash"),
        temperature=config.get("temperature", 0.0),
        api_key=config.get("api_key"),
    )
    return LLMAgent(
        name=name,
        model_client=client,
        prompt_mode=prompt_mode,
        max_retries=config.get("max_retries", 1),
    )