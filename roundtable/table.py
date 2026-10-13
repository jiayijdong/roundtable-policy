"""The confidence-weight table: a running record of each player's reliability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Cell:
    score: float = 0.0  # running mean of committee scores
    uncertainty: float = 0.0  # running mean of committee uncertainties
    count: int = 0

    def update(self, score: float, uncertainty: float) -> None:
        # theta_j = (1 - 1/j) * theta_{j-1} + (1/j) * (s_j, u_j)
        self.count += 1
        self.score += (score - self.score) / self.count
        self.uncertainty += (uncertainty - self.uncertainty) / self.count

    def __str__(self) -> str:
        return f"{self.score:.2f}±{self.uncertainty:.2f}"


@dataclass
class ConfidenceWeightTable:
    """``players x dimensions`` matrix of ``(score, uncertainty)`` running means.

    A dimension is whatever the graders score along: a subtask id (ScienceEval-style)
    or a rubric item such as ``methodology_scientific_rigor`` (ScienceNarrative-style).
    """

    players: List[str]
    dimensions: List[str] = field(default_factory=list)
    cells: Dict[str, Dict[str, Cell]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for p in self.players:
            self.cells.setdefault(p, {})
        for d in list(self.dimensions):
            self._ensure_dim(d)

    def _ensure_dim(self, dim: str) -> None:
        if dim not in self.dimensions:
            self.dimensions.append(dim)
        for p in self.players:
            self.cells[p].setdefault(dim, Cell())

    def update(self, player: str, dim: str, score: float, uncertainty: float) -> None:
        self._ensure_dim(dim)
        self.cells[player][dim].update(score, uncertainty)

    def get(self, player: str, dim: str) -> Tuple[float, float]:
        c = self.cells[player][dim]
        return c.score, c.uncertainty

    def to_csv(self, dimensions: Optional[Iterable[str]] = None) -> str:
        dims = [d for d in (dimensions or self.dimensions) if d in self.dimensions]
        rows = [",".join(["player_id", *dims])]
        for p in self.players:
            rows.append(",".join([p, *(str(self.cells[p][d]) for d in dims)]))
        return "\n".join(rows)

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "players": self.players,
            "dimensions": self.dimensions,
            "cells": {
                p: {d: [c.score, c.uncertainty, c.count] for d, c in row.items()}
                for p, row in self.cells.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceWeightTable":
        cells = {
            p: {d: Cell(*v) for d, v in row.items()} for p, row in data["cells"].items()
        }
        return cls(players=data["players"], dimensions=data["dimensions"], cells=cells)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceWeightTable":
        return cls.from_dict(json.loads(Path(path).read_text()))
