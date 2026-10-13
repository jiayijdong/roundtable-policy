"""Use Roundtable Policy from Python (offline, with mock agents).

    python examples/quickstart.py
"""

from roundtable import ConfidenceWeightTable, RoundtablePolicy, Task, summarize
from roundtable.backends.mock import MockLLM
from roundtable.config import load_tasks

# Swap these for OpenAICompatLLM / AnthropicLLM (roundtable.backends) to use real models.
players = {name: MockLLM(name, quality=q) for name, q in
           [("player_a", 0.85), ("player_b", 0.70), ("player_c", 0.55), ("player_d", 0.40)]}
graders = [MockLLM(f"grader_{i}") for i in range(3)]
policy = RoundtablePolicy(players, graders, fusion=MockLLM("fusion"))

# 1) Backward phase: the grader committee builds the confidence-weight table.
policy.calibrate(load_tasks("examples/data/qa_calibrate.jsonl"), rounds=20)
print("Confidence-weight table:\n" + policy.table.to_csv() + "\n")
policy.table.save("table.json")

# 2) Inference: players propose, the fusion agent aggregates using the table.
task = Task(question="What is the speed of light in vacuum?", dimensions=["physics"])
print("Fused answer:", policy.infer(task).fused, "\n")

# 3) Evaluation: grade the fused answer next to every individual player.
results = policy.evaluate(load_tasks("examples/data/qa_test.jsonl"))
print(summarize(results))

# A saved table can be reloaded for later inference:
policy2 = RoundtablePolicy(players, graders, MockLLM("fusion"),
                           table=ConfidenceWeightTable.load("table.json"))
policy.close(); policy2.close()
