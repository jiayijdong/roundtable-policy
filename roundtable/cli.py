"""Command-line interface: ``roundtable {calibrate,infer,evaluate}``."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from .config import build_policy, load_config, load_tasks
from .policy import Task, summarize
from .table import ConfidenceWeightTable


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="roundtable", description="Roundtable Policy")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="build the confidence-weight table")
    c.add_argument("-c", "--config", required=True)
    c.add_argument("-d", "--data", required=True, help="JSONL tasks")
    c.add_argument("-o", "--out", default="table.json")
    c.add_argument("-r", "--rounds", type=int, default=None,
                   help="number of uniformly sampled rounds (default: one pass)")
    c.add_argument("-t", "--table", help="continue from an existing table")

    i = sub.add_parser("infer", help="answer a query with a calibrated table")
    i.add_argument("-c", "--config", required=True)
    i.add_argument("-t", "--table", required=True)
    i.add_argument("-q", "--question", required=True)
    i.add_argument("--dimensions", nargs="+", default=None,
                   help="table dimensions relevant to this query (default: all)")
    i.add_argument("--show-players", action="store_true")

    e = sub.add_parser("evaluate", help="grade Roundtable vs. every player on held-out tasks")
    e.add_argument("-c", "--config", required=True)
    e.add_argument("-t", "--table", required=True)
    e.add_argument("-d", "--data", required=True)
    e.add_argument("-o", "--out", default="results.jsonl")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    dims = tuple(cfg.get("dimensions", ["overall"]))
    table = ConfidenceWeightTable.load(args.table) if getattr(args, "table", None) else None
    policy = build_policy(cfg, table)

    try:
        if args.cmd == "calibrate":
            policy.calibrate(load_tasks(args.data, dims), rounds=args.rounds)
            policy.table.save(args.out)
            print(policy.table.to_csv())
            print(f"\nsaved confidence-weight table to {args.out}")

        elif args.cmd == "infer":
            task = Task(question=args.question,
                        dimensions=args.dimensions or list(policy.table.dimensions) or list(dims))
            result = policy.infer(task)
            if args.show_players:
                for pid, ans in result.responses.items():
                    print(f"--- {pid} ---\n{ans}\n")
            print(result.fused)

        elif args.cmd == "evaluate":
            results = policy.evaluate(load_tasks(args.data, dims))
            with open(args.out, "w") as f:
                for r in results:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            print(json.dumps(summarize(results), indent=2))
            print(f"\nwrote per-task results to {args.out}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
