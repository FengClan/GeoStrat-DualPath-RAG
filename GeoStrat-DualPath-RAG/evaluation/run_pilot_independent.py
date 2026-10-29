#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D4 held-out 评估「试水」：从 test_set_independent.json 分层抽样，跑 4 基线对比。

抽样策略：按 query_type（FACTUAL/REASONING/COMPARATIVE/SPATIAL）各抽 --per-type 题，
保证能看出双路法在各题型上的效果差异（尤其是 Comparative/Spatial 这两类已知难点）。

用法：
  python evaluation/run_pilot_independent.py --per-type 5 --seed 42
  python evaluation/run_pilot_independent.py --per-type 8 --test-set test_sets/test_set_independent.json

输出：evaluation/results/baseline_comparison_<时间戳>.json + 终端对比表
"""
import sys, os, json, random, argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner

QUERY_TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]


def stratify(questions, per_type, seed):
    rng = random.Random(seed)
    by_type = {}
    for q in questions:
        by_type.setdefault(q.get("query_type", "FACTUAL"), []).append(q)
    subset = []
    for qt in QUERY_TYPES:
        pool = by_type.get(qt, [])
        picked = rng.sample(pool, min(per_type, len(pool)))
        subset.extend(picked)
        print(f"[PILOT] {qt}: 抽 {len(picked)}/{len(pool)} 题")
    return subset


def main():
    ap = argparse.ArgumentParser(description="held-out 独立集试水（分层抽样 4 基线）")
    ap.add_argument("--per-type", type=int, default=5, help="每个 query_type 抽多少题")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子（可复现）")
    # 默认 dev 集（不是 held-out）：试水属"看行为"，按 R2#3 的划分纪律不该反复
    # 消耗留出集。旧默认 test_set_independent.json 连 gold_evidence_spans 都没有，
    # 拿它试水会让 evidence_recall@5 全记 N/A，看不出真实检索质量。
    ap.add_argument("--test-set", default="test_sets/test_set_honest_dev.json")
    ap.add_argument("--model", default="qwen3.5-plus")
    ap.add_argument("--no-llm-judge", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ts = json.load(open(args.test_set, encoding="utf-8"))
    subset_qs = stratify(ts["questions"], args.per_type, args.seed)
    subset = {"version": ts.get("version"), "questions": subset_qs}

    print(f"\n[PILOT] 抽样合计 {len(subset_qs)} 题")
    print("  by query_type:", dict(Counter(q['query_type'] for q in subset_qs)))
    print("  by difficulty:", dict(Counter(q['difficulty'] for q in subset_qs)))
    print("  by category:", dict(Counter(q['category'] for q in subset_qs)))

    runner = BaselineRunner(model=args.model, results_dir="./evaluation/results")
    results = runner.run_all_baselines(subset, use_llm_judge=not args.no_llm_judge)
    runner.print_comparison_summary(results)


if __name__ == "__main__":
    main()
