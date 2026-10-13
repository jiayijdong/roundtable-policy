import json

import pytest

from roundtable import ConfidenceWeightTable, RoundtablePolicy, Task, summarize
from roundtable.backends.base import LLM
from roundtable.backends.mock import MockLLM
from roundtable.config import build_policy, load_config, load_tasks
from roundtable.parsing import extract_json, parse_score


def test_parse_score():
    assert parse_score("34.75±4.75") == (34.75, 4.75)
    assert parse_score(" -12.5 +/- 3 ") == (-12.5, 3.0)
    assert parse_score(80) == (80.0, 0.0)
    assert parse_score([70, -2]) == (70.0, 2.0)
    assert parse_score({"score": 1, "uncertainty": 2}) == (1.0, 2.0)
    with pytest.raises(ValueError):
        parse_score("n/a")


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blah {"a": {"b": 2}} blah') == {"a": {"b": 2}}


def test_table_running_mean_and_roundtrip(tmp_path):
    t = ConfidenceWeightTable(players=["p1", "p2"])
    for s, u in [(80, 4), (60, 8), (70, 6)]:
        t.update("p1", "math", s, u)
    assert t.get("p1", "math") == pytest.approx((70.0, 6.0))
    assert t.get("p2", "math") == (0.0, 0.0)
    assert t.to_csv().splitlines() == ["player_id,math", "p1,70.00±6.00", "p2,0.00±0.00"]
    t.save(tmp_path / "t.json")
    t2 = ConfidenceWeightTable.load(tmp_path / "t.json")
    assert t2.to_dict() == t.to_dict()


def _mock_policy(**kw):
    players = {n: MockLLM(n, quality=q) for n, q in [("strong", 0.9), ("mid", 0.6), ("weak", 0.3)]}
    graders = [MockLLM(f"g{i}") for i in range(3)]
    return RoundtablePolicy(players, graders, MockLLM("fusion"), **kw)


def test_calibration_ranks_players_and_fusion_helps():
    policy = _mock_policy()
    tasks = load_tasks("examples/data/qa_calibrate.jsonl")
    policy.calibrate(tasks, rounds=12)
    mean = lambda p: sum(policy.table.get(p, d)[0] for d in policy.table.dimensions) / 3
    assert mean("strong") > mean("mid") > mean("weak")
    summary = summarize(policy.evaluate(load_tasks("examples/data/qa_test.jsonl")))
    assert summary["roundtable"]["mean_score"] > summary["weak"]["mean_score"]
    assert summary["weak"]["roundtable_gain_%"] > 0
    policy.close()


class BrokenGrader(LLM):
    name = "broken"

    def complete(self, system, user, role=None):
        return "I refuse to output JSON"


def test_broken_grader_is_skipped():
    policy = _mock_policy()
    policy.graders.append(BrokenGrader())
    task = Task(question="q", reference="k", dimensions=["overall"])
    result = policy.step(task)
    assert set(result.grades) == {"strong", "mid", "weak"}
    policy.close()


def test_graders_see_anonymous_ids():
    seen = []

    class Spy(MockLLM):
        def complete(self, system, user, role=None):
            if role == "grader":
                seen.append(json.loads(user))
            return super().complete(system, user, role)

    players = {"gpt-x": MockLLM("gpt-x"), "claude-y": MockLLM("claude-y")}
    policy = RoundtablePolicy(players, [Spy("g")], MockLLM("f"))
    policy.step(Task(question="q"))
    assert set(seen[0]["responses"]) == {"response_01", "response_02"}
    policy.close()


@pytest.mark.parametrize("cfg", ["examples/configs/mock.yaml"])
def test_build_from_config(cfg):
    policy = build_policy(load_config(cfg))
    assert len(policy.players) == 4 and len(policy.graders) == 3
    result = policy.infer(Task(question="What is 2+2?"))
    assert result.fused and len(result.responses) == 4
    policy.close()
