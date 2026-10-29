#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
思考模式开关对照实验：回答「关掉 qwen3.5-plus 的思考，会不会掉分」。

背景（2026-09-14 实测）：qwen3.5-plus 在 DashScope 兼容接口上**默认开思考**，
一道题 41 秒里绝大部分在生成思维链，而 model_router 最终只取 message.content，
**思维链整段被丢弃**——时间和 token 都白花。关掉之后同一题 2.4 秒、输出 token
从 2076 降到 24，答案内容却没变。

但这个「没变」只是在玩具问题上目测的。全量评测要拿它当配置，必须先证明
**在真实题目和真实检索上下文下分数不塌**。本脚本就是做这件事：

  同一批题 × 同一基线 × 同一裁判，只切 enable_thinking 这一个变量，
  两臂各跑一遍，报告质量差、延迟比、token 比。

用法:
    python evaluation/compare_thinking.py                    # 30 题，dual_path
    python evaluation/compare_thinking.py --total 30 --baseline vector_rag --workers 6
    python evaluation/compare_thinking.py --total 60 --baseline all

判读标准（写在输出末尾，避免事后挑话说）：
  - overall_score 差值在 ±0.02 以内 → 视为持平，可放心全量关思考提速；
  - 掉分超过 0.02 → 保留思考，只用 --workers 提速（仍能快一个数量级）；
  - 掉分集中在 REASONING/COMPARATIVE → 说明思考确实在起作用，别关。

首次运行结论（2026-09-14，30 题 dual_path，seed=42）：
  overall 0.7333 → 0.6902（-0.0431，**超过阈值，不要关**）
  按题型 FACTUAL +0.008 / COMPARATIVE +0.017 基本不变，REASONING -0.193(n=7)。
  → 全量评测走 --workers 并发，保留思考。
  ⚠️ 该结论基于单轮；若要用作论文论断，应按导师要求跑 3 轮取 mean±SD。
"""
import sys, os, json, argparse, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator
from evaluation.run_full_eval import stratify_proportional, QUERY_TYPES
from services.model_router import usage_stats


def run_arm(questions, baseline, model, judge_model, workers, thinking):
    """跑一臂（一个 thinking 配置），返回 (results, token_stats, 墙钟秒数)。"""
    os.environ["QWEN_ENABLE_THINKING"] = "1" if thinking else "0"
    usage_stats(reset=True)          # 只统计本臂

    runner = BaselineRunner(model=model)
    ev = Evaluator(model=model, judge_model=judge_model or None)

    subset = {"version": "thinking-ab", "questions": questions}
    t0 = time.time()
    results = ev.run_full_evaluation(
        subset, runner.get_runner(baseline),
        use_llm_judge=True,
        checkpoint_path=None,        # 对照实验不续跑：两臂必须跑同一批题
        workers=workers,
    )
    return results, usage_stats(), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="思考模式开关对照（质量 / 延迟 / token）")
    ap.add_argument("--total", type=int, default=30, help="对照题数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-set", default="test_sets/test_set_honest.json")
    ap.add_argument("--baseline", default="dual_path",
                    choices=["direct_llm", "vector_rag", "graph_rag", "dual_path", "all"],
                    help="对照用哪个基线（all = 四个都跑，更慢但更全面）")
    ap.add_argument("--model", default="qwen3.5-plus")
    ap.add_argument("--judge-model", default="deepseek-flash")
    ap.add_argument("--workers", type=int, default=4,
                    help="每臂内部并发（两臂仍是先后跑，避免环境变量互相污染）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ts = json.load(open(args.test_set, encoding="utf-8"))
    questions = stratify_proportional(ts["questions"], args.total, args.seed)
    cnt = Counter(q.get("query_type", "?") for q in questions)

    print("=" * 68)
    print(f"[思考对照] {len(questions)} 题，seed={args.seed}，每个基线两臂")
    print(f"           生成={args.model}  裁判={args.judge_model}  臂内并发={args.workers}")
    print("           " + "  ".join(f"{t}:{cnt.get(t, 0)}" for t in QUERY_TYPES))
    print("=" * 68)

    baselines = (["direct_llm", "vector_rag", "graph_rag", "dual_path"]
                 if args.baseline == "all" else [args.baseline])

    report = {"config": vars(args), "sample_composition": dict(cnt), "baselines": {}}

    for bk in baselines:
        print(f"\n{'-' * 68}\n[基线 {bk}]")
        arms = {}
        for thinking in (True, False):
            tag = "带思考" if thinking else "关思考"
            print(f"  → 正在跑【{tag}】臂 …", flush=True)
            res, usage, secs = run_arm(questions, bk, args.model,
                                       args.judge_model, args.workers, thinking)
            arms["thinking" if thinking else "nothinking"] = {
                "results": res, "usage": usage, "wall_seconds": round(secs, 1)}
            print(f"    完成：墙钟 {secs:.1f}s | overall="
                  f"{res['overall_scores']['overall_score']}")

        a, b = arms["thinking"], arms["nothinking"]
        ta = a["usage"].get(args.model, {})
        tb = b["usage"].get(args.model, {})

        def cmp_metric(key):
            return (a["results"]["overall_scores"][key],
                    b["results"]["overall_scores"][key])

        print(f"\n  {'指标':<22}{'带思考':>10}{'关思考':>10}{'差值':>10}")
        print("  " + "-" * 52)
        keys = ["overall_score", "qa_correctness", "qa_faithfulness",
                "qa_completeness", "citation_precision", "citation_recall"]
        deltas = {}
        for k in keys:
            va, vb = cmp_metric(k)
            deltas[k] = round(vb - va, 4)
            print(f"  {k:<22}{va:>10.4f}{vb:>10.4f}{vb - va:>+10.4f}")

        print(f"\n  {'输出 token 合计':<22}{ta.get('completion_tokens', 0):>10}"
              f"{tb.get('completion_tokens', 0):>10}")
        print(f"  {'输入 token 合计':<22}{ta.get('prompt_tokens', 0):>10}"
              f"{tb.get('prompt_tokens', 0):>10}")
        print(f"  {'调用次数':<22}{ta.get('calls', 0):>10}{tb.get('calls', 0):>10}")
        print(f"  {'墙钟秒数':<22}{a['wall_seconds']:>10.1f}{b['wall_seconds']:>10.1f}")

        # 按题型分层的差值——掉分集中在哪一档，是"能不能关"的关键判据。
        # 注：by_query_type 里的 avg_score 是 (correctness+faithfulness)/2，
        # 不是 overall_score——此处只作**题型间的相对比较**，不能当总分读。
        print(f"\n  按题型（avg_score=(correctness+faithfulness)/2 差值）:")
        by_type = {}
        for t in QUERY_TYPES:
            sa = a["results"]["by_query_type"].get(t, {})
            sb = b["results"]["by_query_type"].get(t, {})
            oa = sa.get("avg_score") if isinstance(sa, dict) else None
            ob = sb.get("avg_score") if isinstance(sb, dict) else None
            if isinstance(oa, (int, float)) and isinstance(ob, (int, float)):
                n = sb.get("count")
                by_type[t] = {"delta": round(ob - oa, 4), "n": n,
                              "thinking": oa, "nothinking": ob}
                print(f"    {t:<14}{oa:>9.4f}{ob:>9.4f}{ob - oa:>+9.4f}  (n={n})")

        report["baselines"][bk] = {
            "thinking": {"overall": a["results"]["overall_scores"],
                         "usage": ta, "wall_seconds": a["wall_seconds"]},
            "nothinking": {"overall": b["results"]["overall_scores"],
                           "usage": tb, "wall_seconds": b["wall_seconds"]},
            "delta_overall": deltas["overall_score"],
            "delta_by_query_type": by_type,
        }

    # ---------------- 判读 ----------------
    print(f"\n{'=' * 68}\n[判读]\n{'=' * 68}")
    for bk, r in report["baselines"].items():
        d = r["delta_overall"]
        if d >= -0.02:
            verdict = "持平 → 可以全量关思考提速"
        else:
            verdict = "掉分 → 保留思考，改用 --workers 提速"
        # 全局持平但某一题型明显掉，同样不能关——那个题型可能正是论文的卖点
        worst = min(r["delta_by_query_type"].items(),
                    key=lambda kv: kv[1]["delta"], default=(None, None))
        extra = ""
        if worst[0] and worst[1]["delta"] < -0.05:
            extra = f"（注意 {worst[0]} 掉 {worst[1]['delta']:+.4f}，n={worst[1]['n']}）"
        print(f"  {bk:<12} overall 差 {d:+.4f}  {verdict}{extra}")
    print("\n  阈值说明：±0.02 以内视为持平；掉分超过 0.02 则不要关。")
    print("  无论结论如何，这条对照本身就是 R2#18 要的效率/成本证据。")

    out = args.out or os.path.join(
        "evaluation", "results", f"thinking_ab_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
