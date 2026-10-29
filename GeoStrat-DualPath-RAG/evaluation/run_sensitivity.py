#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图跳数敏感性分析——回应审稿意见 R2#4（"1-3 跳仅为启发式，需敏感性分析"）。

对 max_hops = 1 / 2 / 3 各跑一遍图谱相关的基线（graph_rag 与 dual_path），
报告各项指标随跳数的变化，看当前取 2 是不是一个站得住的选择。

⚠️ 纪律（重要）：本实验**只作分析报告**。不得看完成绩回头挑"最好的 hop"写进
   主实验——那等于用测试集调参，正好踩中 R2#3 的指控。若敏感性结果显示
   hop=2 并非最优，**如实报告这一事实**，主实验仍用预先设定的 hop=2，
   并在论文里说明该选择来自领域经验而非测试集调优。

用法:
    python evaluation/run_sensitivity.py --total 30 --baseline dual_path
    python evaluation/run_sensitivity.py --total 60 --baseline both --workers 6
    python evaluation/run_sensitivity.py --total 30 --hops 1,2,3,4
"""
import sys, os, json, argparse, time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator
from evaluation.run_full_eval import QUERY_TYPES

# 默认只跑 dev 集：敏感性分析属于"看行为"，不该反复消耗留出集（R2#3）
DEFAULT_SET = os.path.join(os.path.dirname(__file__), "..",
                           "test_sets", "test_set_honest_dev.json")


def sample_questions(path, total, seed):
    import random
    d = json.load(open(path, encoding="utf-8"))
    qs = d["questions"] if isinstance(d, dict) else d
    if total and total < len(qs):
        qs = random.Random(seed).sample(qs, total)
    return qs


def main():
    ap = argparse.ArgumentParser(description="图跳数敏感性分析（R2#4）")
    ap.add_argument("--test-set", default=DEFAULT_SET)
    ap.add_argument("--total", type=int, default=30, help="0 = 用全量测试集")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hops", default="1,2,3", help="要扫描的跳数，逗号分隔")
    ap.add_argument("--baseline", default="both",
                    choices=["graph_rag", "dual_path", "both"],
                    help="图谱相关的基线（只有这两个会用到 max_hops）")
    ap.add_argument("--model", default="qwen3.5-plus")
    ap.add_argument("--judge-model", default="deepseek-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results-dir", default="./evaluation/results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    hops = [int(x) for x in args.hops.split(",") if x.strip()]
    qs = sample_questions(args.test_set, args.total, args.seed)
    cnt = Counter(q.get("query_type", "?") for q in qs)
    baselines = (["graph_rag", "dual_path"] if args.baseline == "both"
                 else [args.baseline])

    print("=" * 70)
    print(f"[跳数敏感性] {len(qs)} 题  seed={args.seed}  跳数={hops}  基线={baselines}")
    print(f"             生成={args.model} 裁判={args.judge_model} workers={args.workers}")
    print(f"             题源={os.path.basename(os.path.normpath(args.test_set))}")
    print("             " + "  ".join(f"{t}:{cnt.get(t, 0)}" for t in QUERY_TYPES))
    print("=" * 70)

    subset = {"version": "sensitivity", "questions": qs}
    ev = Evaluator(model=args.model, results_dir=args.results_dir,
                   judge_model=args.judge_model or None)
    report = {"config": vars(args), "sample_composition": dict(cnt), "results": {}}

    for bk in baselines:
        print(f"\n{'-' * 70}\n[基线 {bk}]")
        report["results"][bk] = {}
        for h in hops:
            runner = BaselineRunner(model=args.model, results_dir=args.results_dir,
                                    hops_override=h)
            t0 = time.time()
            res = ev.run_full_evaluation(
                subset, runner.get_runner(bk), use_llm_judge=True,
                checkpoint_path=None, workers=args.workers)
            # 被 Ctrl+C 中断：该 hop 只覆盖了一部分题，点不可比（敏感性曲线的
            # 意义就是"同题集下换 hop 的差异"，题不一样了曲线就没意义）。
            # 注意本脚本 checkpoint_path=None，没有断点可续——中断即从头开始。
            if res.get("interrupted"):
                print(f"\n[SENSITIVITY] 已在 {bk} hop={h} 中断（Ctrl+C）。")
                print("[SENSITIVITY] 本脚本不写 checkpoint，中断即整轮重来"
                      "（如需可续跑请改用 run_full_eval.py）。")
                return
            os_ = res["overall_scores"]
            report["results"][bk][h] = {
                "overall_score": os_["overall_score"],
                "qa_correctness": os_["qa_correctness"],
                "qa_faithfulness": os_["qa_faithfulness"],
                "qa_completeness": os_["qa_completeness"],
                "avg_latency_ms": os_["avg_latency_ms"],
                "coverage": os_["coverage"],
                "by_query_type": res["by_query_type"],
                "wall_seconds": round(time.time() - t0, 1),
            }
            print(f"  hop={h}  完成  墙钟 {time.time()-t0:6.1f}s  "
                  f"overall={os_['overall_score']:.4f}")

        print(f"\n  {'跳数':<6}{'overall':>10}{'correct':>10}{'faith':>10}"
              f"{'complete':>10}{'延迟ms':>10}")
        print("  " + "-" * 56)
        for h in hops:
            r = report["results"][bk][h]
            print(f"  {h:<6}{r['overall_score']:>10.4f}{r['qa_correctness']:>10.4f}"
                  f"{r['qa_faithfulness']:>10.4f}{r['qa_completeness']:>10.4f}"
                  f"{r['avg_latency_ms']:>10.0f}")

        # 逐题型看跳数影响：图谱密集的题（FACTUAL）和多跳推理题反应通常不同
        print(f"\n  按题型 overall（avg_score=(correctness+faithfulness)/2）:")
        print(f"    {'题型':<14}" + "".join(f"{'hop=' + str(h):>12}" for h in hops))
        for t in QUERY_TYPES:
            row = f"    {t:<14}"
            any_ = False
            for h in hops:
                g = report["results"][bk][h]["by_query_type"].get(t)
                if g:
                    any_ = True
                    row += f"{g['avg_score']:>12.4f}"
                else:
                    row += f"{'-':>12}"
            if any_:
                print(row)

    # ---------------- 判读 ----------------
    print(f"\n{'=' * 70}\n[判读]\n{'=' * 70}")
    for bk, per_h in report["results"].items():
        best = max(per_h, key=lambda h: per_h[h]["overall_score"])
        ref = per_h.get(2)
        line = f"  {bk:<12} 最优 hop={best}"
        if ref:
            line += f"  |  hop=2 相对差距 {per_h[best]['overall_score'] - ref['overall_score']:+.4f}"
        print(line)
    print("\n  ⚠️ 以上只作分析报告，**不得据此改主实验的 hop 设定**（否则违反 R2#3）。")
    print("     若 hop=2 非最优，论文中如实写出，并说明设定依据来自领域经验而非测试集调优。")

    out = args.out or os.path.join(
        args.results_dir, f"sensitivity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
