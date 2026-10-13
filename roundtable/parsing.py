"""Helpers for parsing LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Tuple

_SCORE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*$")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_score(value: Any) -> Tuple[float, float]:
    """Parse a grade into ``(score, uncertainty)``.

    Accepts ``"34.75±4.75"`` (also ``+/-``), a bare number (uncertainty 0),
    ``[score, unc]`` or ``{"score": ..., "uncertainty": ...}``.
    """
    if isinstance(value, (int, float)):
        return float(value), 0.0
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), abs(float(value[1]))
    if isinstance(value, dict):
        return float(value["score"]), abs(float(value.get("uncertainty", 0.0)))
    if isinstance(value, str):
        m = _SCORE.match(value)
        if m:
            return float(m.group(1)), float(m.group(2))
        return float(value.strip()), 0.0
    raise ValueError(f"cannot parse score from {value!r}")


def extract_json(text: str) -> Any:
    """Return the first JSON object found in ``text`` (tolerates code fences / chatter)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])
