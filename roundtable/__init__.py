"""Roundtable Policy: confidence-weighted consensus for multi-agent reasoning."""

from .backends.base import LLM
from .config import build_policy, load_config, load_tasks, make_llm
from .policy import RoundResult, RoundtablePolicy, Task, summarize
from .table import ConfidenceWeightTable

__all__ = [
    "LLM",
    "ConfidenceWeightTable",
    "RoundtablePolicy",
    "RoundResult",
    "Task",
    "build_policy",
    "load_config",
    "load_tasks",
    "make_llm",
    "summarize",
]
__version__ = "0.1.0"
