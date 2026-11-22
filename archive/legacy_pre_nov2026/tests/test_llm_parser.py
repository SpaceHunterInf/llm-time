from __future__ import annotations

import pytest

from llm_rl_engine.agents.llm import extract_json_object


def test_extract_plain_json():
    text = '{"action": {"arm": 1}, "confidence": 0.5, "rationale": "test"}'
    parsed = extract_json_object(text)
    assert parsed["action"]["arm"] == 1


def test_extract_fenced_json():
    text = '```json\n{"action": {"guess": 42}}\n```'
    parsed = extract_json_object(text)
    assert parsed["action"]["guess"] == 42


def test_extract_invalid_raises():
    with pytest.raises(ValueError):
        extract_json_object("no json here")