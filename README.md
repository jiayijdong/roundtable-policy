# Roundtable Policy

A minimal, general implementation of **Roundtable Policy: Scaling Multi-Agent Reasoning with Consensus**, an inference-time reasoning framework for multi-agent LLM systems. It reaches a consensus by weighting each agent according to how reliable that agent has proved over time.

Roundtable Policy works the way a committee does. Members propose answers independently. Each member's track record (its score **and** how certain that score is) is learned over many rounds. A final decision is then formed by trusting reliable members more. It needs only **black-box access** to the models. No fine-tuning is involved.

```
                ┌──────────── Proposition phase ────────────┐
   query  ───►  │  player_1   player_2   ...   player_Lp      │  candidate answers
                └───────────────────┬───────────────────────┘
                                    ▼
                ┌──────────── Inference phase ──────────────┐
                │  fusion agent  ◄── confidence-weight table │ ──► final answer
                └───────────────────┬───────────────────────┘
                                    ▼
                ┌──────────── Backward phase ───────────────┐
                │  grader committee: score ± uncertainty     │ ──► update table
                └───────────────────────────────────────────┘
```

* **Players** `A^P_i` independently generate candidate responses.
* **Graders** `A^G_l` form a committee. For every player and every *dimension* (a sub-task or a rubric item), each grader returns `score ± uncertainty` with score in [-100, 100] and uncertainty as the half-width of a 95% CI. The committee average updates the **confidence-weight table** `ϑ ∈ R^{L_p × N}` as a running mean:
  `ϑ_j = (1 − 1/j)·ϑ_{j−1} + (1/j)·(s_j, u_j)`.
* The **fusion agent** `A^F` aggregates the players' answers. It is conditioned on the table, favouring high scores and tight intervals on the relevant dimensions.

## Install

```bash
git clone https://github.com/jiayijdong/roundtable-policy.git
cd roundtable-policy
pip install -e ".[all]"        # or ".[openai]" / ".[anthropic]"; add ",dev" for pytest
```

## Quick start (offline, no API key)

`examples/configs/mock.yaml` uses deterministic mock agents with hidden quality levels, so you can watch the pipeline work end to end:

```bash
# 1. Backward phase: build the confidence-weight table (20 uniformly sampled rounds)
roundtable calibrate -c examples/configs/mock.yaml -d examples/data/qa_calibrate.jsonl -r 20 -o table.json

# 2. Inference: answer a new query with the calibrated table
roundtable infer -c examples/configs/mock.yaml -t table.json -q "Explain the S-N curve." --dimensions physics

# 3. Evaluation: the committee grades the fused answer next to every single player
roundtable evaluate -c examples/configs/mock.yaml -t table.json -d examples/data/qa_test.jsonl -o results.jsonl
```

`evaluate` prints each candidate's mean committee score and Roundtable's relative gain over each player, `(s_RP − s_p) / |s_p| × 100`.

The same steps are available from Python; see [`examples/quickstart.py`](examples/quickstart.py):

```python
from roundtable import RoundtablePolicy, Task, load_tasks
from roundtable.backends.openai_compat import OpenAICompatLLM
from roundtable.backends.anthropic_llm import AnthropicLLM

players = {m: OpenAICompatLLM(m) for m in ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]}
graders = [OpenAICompatLLM("gpt-4o", temperature=0), AnthropicLLM("claude-opus-5-5", effort="low")]
policy = RoundtablePolicy(players, graders, fusion=OpenAICompatLLM("gpt-4o"))

policy.calibrate(load_tasks("examples/data/qa_calibrate.jsonl"), rounds=50)
print(policy.table.to_csv())
print(policy.infer(Task("Why does ice float?", dimensions=["physics"])).fused)
```

## Using real models

| backend     | config                                                        | notes |
|-------------|---------------------------------------------------------------|-------|
| `openai`    | `{backend: openai, model: gpt-4o}`                            | Any OpenAI-compatible endpoint via `base_url` + `api_key_env` (OpenAI, OpenRouter, DeepSeek, xAI, vLLM, SGLang, Ollama, ...). Extra keys such as `temperature` are forwarded. |
| `anthropic` | `{backend: anthropic, model: claude-opus-5-5, effort: low}`   | Official Anthropic SDK with streaming; reads `ANTHROPIC_API_KEY`. |
| `mock`      | `{backend: mock, name: a, quality: 0.8}`                      | Offline, for demos and tests only. |

See [`examples/configs/openai.yaml`](examples/configs/openai.yaml) and [`examples/configs/heterogeneous.yaml`](examples/configs/heterogeneous.yaml), which mixes Claude, GPT, DeepSeek, Qwen and Llama in one roundtable. To use your own model, subclass `roundtable.LLM` and implement `complete(system, user, role=None) -> str`.

## Data format

Tasks are JSONL, one task per line:

```json
{"id": "q1", "question": "...", "reference": "optional key", "dimensions": ["physics", "math", "biology"]}
```

* **Dimensions** are the columns of the confidence-weight table. They can be sub-tasks of a multi-question round, as in the `ScienceEval` setting ([`qa_calibrate.jsonl`](examples/data/qa_calibrate.jsonl)). They can also be rubric items such as `methodology_scientific_rigor` for open-ended generation without ground truth, as in the `ScienceNarrative` setting ([`proposal.jsonl`](examples/data/proposal.jsonl)). Tasks without `dimensions` use the config's default (`[overall]`).
* `reference` is optional. Graders use it when it is present.

## Config

```yaml
players:  [ {backend: ..., model: ..., name: ...}, ... ]   # names must be unique
graders:  [ ... ]                                          # the committee
fusion:   {backend: ..., model: ...}
dimensions: [overall]        # default dimensions
max_workers: 8               # parallel API calls
seed: 0
prompts:                     # optional overrides (see roundtable/prompts.py)
  player_system: "..."
  grader_system: "..."
  fusion_system: "... {table} ..."
```

## Design notes

* Graders see **anonymised, shuffled** responses (`response_01`, ...), which reduces name and position bias. A grader whose output can't be parsed is retried once and then skipped for that round.
* Grader outputs are parsed tolerantly: `"70.5±3.2"`, `"70.5 +/- 3.2"`, bare numbers, `[s, u]` or `{"score", "uncertainty"}`.
* The table is plain JSON (`table.save` / `ConfidenceWeightTable.load`). You can build it once and reuse it for inference.

## Repository layout

```
roundtable/
  policy.py        RoundtablePolicy: propose / fuse / grade / calibrate / infer / evaluate
  table.py         ConfidenceWeightTable (running mean of score ± uncertainty)
  prompts.py       default player / grader / fusion prompts
  parsing.py       score and JSON parsing
  config.py        YAML/JSON config -> policy
  cli.py           `roundtable` command
  backends/        openai_compat.py, anthropic_llm.py, mock.py
examples/          configs, toy datasets, quickstart.py
tests/             pytest suite (runs offline)
```

## License

MIT
