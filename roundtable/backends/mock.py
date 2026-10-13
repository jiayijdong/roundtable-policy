"""Deterministic offline backend so the whole pipeline runs without API keys.

* players tag their answer with a hidden quality, e.g. ``[q=0.80]``;
* graders read that tag and emit ``score±uncertainty`` around ``100 * q``;
* the fusion agent reads the confidence-weight table from its system prompt and
  returns an (improved) copy of the answer from the most trusted player.

It exists for demos and tests only: it says nothing about real model quality.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from ..parsing import extract_json, parse_score
from .base import LLM

_Q_TAG = re.compile(r"\[q=([0-9.]+)\]")


def _unit(*parts: str) -> float:
    """Stable pseudo-random number in [0, 1) derived from the given strings."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


class MockLLM(LLM):
    def __init__(self, name: str = "mock", quality: float = 0.7, noise: float = 10.0, **_):
        self.name = name
        self.quality = quality
        self.noise = noise

    def complete(self, system: str, user: str, role: Optional[str] = None) -> str:
        if role == "grader":
            return self._grade(user)
        if role == "fusion":
            return self._fuse(system, user)
        return self._answer(user)

    # -- player ---------------------------------------------------------------
    def _answer(self, user: str) -> str:
        q = min(1.0, max(0.0, self.quality + 0.2 * (_unit(self.name, user) - 0.5)))
        return f"{self.name}'s answer to: {user.strip()[:80]} [q={q:.2f}]"

    # -- grader ---------------------------------------------------------------
    def _grade(self, user: str) -> str:
        payload = json.loads(user)
        out = {}
        for rid, text in payload["responses"].items():
            m = _Q_TAG.search(text)
            q = float(m.group(1)) if m else 0.5
            out[rid] = {}
            for dim in payload["dimensions"]:
                s = 100 * q + self.noise * (2 * _unit(self.name, rid, text, dim) - 1)
                u = 2 + 10 * (1 - q) * _unit(dim, text, self.name)
                out[rid][dim] = f"{max(-100, min(100, s)):.2f}±{u:.2f}"
        return json.dumps(out)

    # -- fusion ---------------------------------------------------------------
    def _fuse(self, system: str, user: str) -> str:
        answers = extract_json(user)["answers"]
        trust = {}
        for line in system.splitlines():
            cells = line.split(",")
            if cells and cells[0] in answers:
                vals = [parse_score(c) for c in cells[1:] if "±" in c]
                if vals:  # prefer high score, penalise wide intervals
                    trust[cells[0]] = sum(s - 0.5 * u for s, u in vals) / len(vals)
        best = max(answers, key=lambda pid: trust.get(pid, 0.0))
        m = _Q_TAG.search(answers[best])
        q = min(1.0, (float(m.group(1)) if m else 0.5) + 0.05)
        body = _Q_TAG.sub("", answers[best]).strip()
        return f"[fused from {len(answers)} answers, led by {best}] {body} [q={q:.2f}]"
