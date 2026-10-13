"""Roundtable Policy: proposition -> (inference) -> backward phases."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import prompts
from .backends.base import LLM
from .parsing import extract_json, parse_score
from .table import ConfidenceWeightTable

log = logging.getLogger(__name__)

Grades = Dict[str, Dict[str, Tuple[float, float]]]  # candidate -> dimension -> (score, unc)


@dataclass
class Task:
    question: str
    reference: Optional[str] = None
    dimensions: List[str] = field(default_factory=lambda: ["overall"])
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, default_dimensions: Sequence[str] = ("overall",)) -> "Task":
        return cls(
            question=d["question"],
            reference=d.get("reference"),
            dimensions=list(d.get("dimensions") or default_dimensions),
            id=d.get("id"),
        )


@dataclass
class RoundResult:
    task: Task
    responses: Dict[str, str]  # player -> answer
    fused: Optional[str] = None  # Roundtable Policy answer
    grades: Grades = field(default_factory=dict)  # committee-averaged grades


class RoundtablePolicy:
    """Committee-style multi-agent inference.

    * players  (A^P_i): propose candidate answers independently;
    * fusion   (A^F):   aggregates them, conditioned on the confidence-weight table;
    * graders  (A^G_l): a committee that scores answers (score ± uncertainty) to
      build / update the table. LLM parameters are never changed.
    """

    def __init__(
        self,
        players: Dict[str, LLM],
        graders: Sequence[LLM],
        fusion: LLM,
        table: Optional[ConfidenceWeightTable] = None,
        player_system: str = prompts.PLAYER_SYSTEM,
        grader_system: str = prompts.GRADER_SYSTEM,
        fusion_system: str = prompts.FUSION_SYSTEM,
        max_workers: int = 8,
        grader_retries: int = 1,
        seed: int = 0,
    ):
        if not players or not graders:
            raise ValueError("need at least one player and one grader")
        self.players = dict(players)
        self.graders = list(graders)
        self.fusion = fusion
        self.table = table or ConfidenceWeightTable(players=list(self.players))
        missing = set(self.players) - set(self.table.players)
        if missing:
            raise ValueError(f"table has no rows for players {sorted(missing)}")
        self.player_system = player_system
        self.grader_system = grader_system
        self.fusion_system = fusion_system
        self.grader_retries = grader_retries
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.rng = random.Random(seed)

    # -- proposition phase ----------------------------------------------------
    def propose(self, task: Task) -> Dict[str, str]:
        futures = {
            pid: self.pool.submit(llm.complete, self.player_system, task.question, "player")
            for pid, llm in self.players.items()
        }
        return {pid: f.result() for pid, f in futures.items()}

    # -- inference phase ------------------------------------------------------
    def fuse(self, task: Task, responses: Dict[str, str]) -> str:
        system = self.fusion_system.format(table=self.table.to_csv(task.dimensions))
        user = json.dumps({"question": task.question, "answers": responses}, indent=2)
        return self.fusion.complete(system, user, "fusion")

    # -- backward phase -------------------------------------------------------
    def grade(self, task: Task, candidates: Dict[str, str]) -> Grades:
        """Committee grades for each candidate, averaged over graders that succeeded."""
        # Anonymise and shuffle so graders cannot key on model names or order.
        names = list(candidates)
        self.rng.shuffle(names)
        alias = {f"response_{i + 1:02d}": n for i, n in enumerate(names)}
        user = json.dumps(
            {
                "question": task.question,
                "reference": task.reference,
                "dimensions": task.dimensions,
                "responses": {a: candidates[n] for a, n in alias.items()},
            },
            indent=2,
        )
        futures = [self.pool.submit(self._grade_once, g, user, alias, task) for g in self.graders]
        per_grader = [g for g in (f.result() for f in futures) if g is not None]
        if not per_grader:
            return {}
        out: Grades = {}
        for name in candidates:
            out[name] = {}
            for dim in task.dimensions:
                vals = [g[name][dim] for g in per_grader if dim in g.get(name, {})]
                if vals:
                    out[name][dim] = (
                        sum(s for s, _ in vals) / len(vals),
                        sum(u for _, u in vals) / len(vals),
                    )
        return out

    def _grade_once(self, grader: LLM, user: str, alias: Dict[str, str], task: Task):
        for attempt in range(self.grader_retries + 1):
            try:
                raw = extract_json(grader.complete(self.grader_system, user, "grader"))
                result: Grades = {}
                for a, name in alias.items():
                    result[name] = {
                        d: parse_score(v) for d, v in raw.get(a, {}).items() if d in task.dimensions
                    }
                return result
            except Exception as e:  # malformed output, API error, ...
                log.warning("grader %s failed (attempt %d): %s", grader.name, attempt + 1, e)
        return None

    def update_table(self, grades: Grades) -> None:
        for pid in self.players:
            for dim, (s, u) in grades.get(pid, {}).items():
                self.table.update(pid, dim, s, u)

    # -- full loops -----------------------------------------------------------
    def step(self, task: Task, fuse: bool = False, update: bool = True) -> RoundResult:
        """One round of Algorithm 1: propose, (fuse), grade, update the table."""
        responses = self.propose(task)
        fused = self.fuse(task, responses) if fuse else None
        candidates = dict(responses)
        if fused is not None:
            candidates["roundtable"] = fused
        grades = self.grade(task, candidates)
        if update:
            self.update_table(grades)
        return RoundResult(task, responses, fused, grades)

    def calibrate(self, tasks: Sequence[Task], rounds: Optional[int] = None) -> ConfidenceWeightTable:
        """Build the confidence-weight table from ``rounds`` uniformly sampled tasks
        (default: one pass over ``tasks``)."""
        order = list(tasks) if rounds is None else [self.rng.choice(tasks) for _ in range(rounds)]
        for j, task in enumerate(order, 1):
            self.step(task, fuse=False, update=True)
            log.info("calibration round %d/%d", j, len(order))
        return self.table

    def infer(self, task: Task) -> RoundResult:
        """Answer a new query with the pre-constructed table (no grading)."""
        responses = self.propose(task)
        return RoundResult(task, responses, self.fuse(task, responses))

    def evaluate(self, tasks: Sequence[Task]) -> List[RoundResult]:
        """Answer each task and have the committee grade the fused answer next to
        every individual player (the table is left unchanged)."""
        return [self.step(t, fuse=True, update=False) for t in tasks]

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def summarize(results: Sequence[RoundResult]) -> Dict[str, dict]:
    """Mean committee score per candidate, and Roundtable's relative gain over each player."""
    totals: Dict[str, List[float]] = {}
    for r in results:
        for name, dims in r.grades.items():
            totals.setdefault(name, []).extend(s for s, _ in dims.values())
    means = {n: sum(v) / len(v) for n, v in totals.items() if v}
    summary = {n: {"mean_score": round(m, 2)} for n, m in means.items()}
    rp = means.get("roundtable")
    if rp is not None:
        for n, m in means.items():
            if n != "roundtable" and m != 0:
                summary[n]["roundtable_gain_%"] = round((rp - m) / abs(m) * 100, 2)
    return summary
