#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重跑修复后的 dual_path，在 pilot 的同一分层子集上（seed=42 可复现），快速看修复效果。
只跑 dual_path 一个基线（其余三个基线代码未动，QA 分沿用第一次 pilot 结果）。
"""
import sys, os, json, random, argparse, time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator

QUERY_TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]


def stratify(questions, per_type, seed):
    rng = random.Random(seed)
    by_type = {}
    for q in questions:
        by_type.setdefault(q.get("query_type", "FACTUAL"), []).append(q)
    subset = []
    for qt in QUERY_TYPES:
        pool = by_type.get(qt, [])
        subset.extend(rng.sample(pool, min(per_type, len(pool))))
    return subset


def main():
    ap = argparse.ArgumentParser(description="重跑修复后 dual_path（pilot 同一子集）")
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-set", default="test_sets/test_set_independent.json")
    ap.add_argument("--model", default="qwen3.5-plus")
    ap.add_argument("--no-llm-judge", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ts = json.load(open(args.test_set, encoding="utf-8"))
    subset_qs = stratify(ts["questions"], args.per_type, args.seed)
    subset = {"version": ts.get("version"), "questions": subset_qs}
    print(f"[RERUN] dual_path（修复后）{len(subset_qs)} 题，seed={args.seed}")

    runner = BaselineRunner(model=args.model)
    ev = Evaluator(model=args.model, results_dir="./evaluation/results")
    res = ev.run_full_evaluation(subset, runner._runner_dual_path,
                                 use_llm_judge=not args.no_llm_judge)

    print("\noverall_scores:", json.dumps(res["overall_scores"], ensure_ascii=False))
    print("by_query_type:", json.dumps(res["by_query_type"], ensure_ascii=False))

    out = os.path.join(ev.results_dir,
                       f"dualpath_fixed_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
