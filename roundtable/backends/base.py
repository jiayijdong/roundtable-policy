"""Minimal LLM interface shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLM(ABC):
    """A chat model that maps (system prompt, user message) -> text.

    ``role`` is a hint ("player", "grader" or "fusion") that real backends ignore;
    the offline :class:`~roundtable.backends.mock.MockLLM` uses it to fake outputs.
    """

    name: str = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, role: Optional[str] = None) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r})"
