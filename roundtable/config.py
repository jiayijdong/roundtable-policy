"""Build a :class:`RoundtablePolicy` from a YAML / JSON config file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from . import prompts
from .backends.base import LLM
from .policy import RoundtablePolicy, Task
from .table import ConfidenceWeightTable


def make_llm(spec: dict) -> LLM:
    """``spec`` = {backend: mock|openai|anthropic, model: ..., name: ..., **backend kwargs}."""
    spec = dict(spec)
    backend = spec.pop("backend", "openai")
    if backend == "mock":
        from .backends.mock import MockLLM

        return MockLLM(**spec)
    if backend == "openai":
        from .backends.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(**spec)
    if backend == "anthropic":
        from .backends.anthropic_llm import AnthropicLLM

        return AnthropicLLM(**spec)
    raise ValueError(f"unknown backend {backend!r}")


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def build_policy(cfg: dict, table: Optional[ConfidenceWeightTable] = None) -> RoundtablePolicy:
    players = {}
    for spec in cfg["players"]:
        llm = make_llm(spec)
        if llm.name in players:
            raise ValueError(f"duplicate player name {llm.name!r}")
        players[llm.name] = llm
    p = cfg.get("prompts", {})
    return RoundtablePolicy(
        players=players,
        graders=[make_llm(s) for s in cfg["graders"]],
        fusion=make_llm(cfg["fusion"]),
        table=table,
        player_system=p.get("player_system", prompts.PLAYER_SYSTEM),
        grader_system=p.get("grader_system", prompts.GRADER_SYSTEM),
        fusion_system=p.get("fusion_system", prompts.FUSION_SYSTEM),
        max_workers=cfg.get("max_workers", 8),
        seed=cfg.get("seed", 0),
    )


def load_tasks(path: str | Path, default_dimensions=("overall",)) -> List[Task]:
    """Read tasks from a JSONL file: one {"question", "reference"?, "dimensions"?, "id"?} per line."""
    with open(path) as f:
        return [Task.from_dict(json.loads(l), default_dimensions) for l in f if l.strip()]
