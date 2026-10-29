# -*- coding: utf-8 -*-
"""六臂配对分析：正式四臂 + 两个补充臂（fullctx / nogate）。

补充臂写在独立 --results-dir 里（避免污染正式四臂总表），故这里跨目录取数。
只读 checkpoint，不发任何 API。

用法：
  python evaluation/_analyze_arms.py
  python evaluation/_analyze_arms.py --extra fullctx        # 只纳入已跑完的补充臂
"""
import sys, os, json, math, argparse, statistics as st
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding="utf-8")

from evaluation.baselines import _graph_is_blind

FULL = "evaluation/results_full_20260914"
# 臂 -> (结果目录, checkpoint 名, 显示名)
ARMS = [
    (FULL, "direct_llm", "纯LLM"),
    (FULL, "vector_rag", "向量RAG"),
    (FULL, "graph_rag", "图谱RAG"),
    (FULL, "dual_path", "双路"),
    ("evaluation/results_control_fullctx_20260916", "dual_path_fullctx", "双路+满配文献"),
    ("evaluation/results_ablation_nogate_20260916", "dual_path_nogate", "双路−门控"),
]
TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]
W = {"FACTUAL": 0.6353, "REASONING": 0.2481, "COMPARATIVE": 0.0873, "SPATIAL": 0.0294}


def overall(q):
    return (0.5385 * q["correctness"] + 0.3077 * q["faithfulness"]
            + 0.1538 * q["completeness"])


def load(rd, arm):
    """返回 {id: (query_type, {metric: value})}；无 qa_metrics 或抛过异常的题一概剔除。"""
    p = os.path.join(rd, f"progress_{arm}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    per = {}
    for r in d.get("per_question_results", []):
        if "qa_metrics" in r and "error" not in r:
            m = r["qa_metrics"]
            per[r["id"]] = (r.get("query_type"), {
                "overall": overall(m),
                "correctness": m.get("correctness", 0.0),
                "faithfulness": m.get("faithfulness", 0.0),
                "completeness": m.get("completeness", 0.0),
                "len": len(r.get("answer", "") or ""),
            })
    return per


def paired(a, b, ids, label, n_tests, key="overall"):
    """同题配对差 + 95%CI；n_tests 用于 Bonferroni 校正（正态近似）。"""
    both = [i for i in ids if i in a and i in b]
    if len(both) < 5:
        return None
    d = [a[i][1][key] - b[i][1][key] for i in both]
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(both))
    crit_b = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))
    t = m / se if se else 0.0
    verdict = "★显著" if abs(t) > crit_b else ("边缘" if abs(t) > 1.96 else "不显著")
    print(f"  {label:<30} n={len(both):>4}  差 {m:+.4f}  "
          f"95%CI [{m-1.96*se:+.3f},{m+1.96*se:+.3f}]  t={t:+.2f}  {verdict}")
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=None,
                    help="只纳入指定的补充臂（如 fullctx nogate）；默认纳入全部已存在的")
    args = ap.parse_args()

    keep = {"fullctx": "dual_path_fullctx", "nogate": "dual_path_nogate"}
    arms = []
    for rd, arm, name in ARMS:
        if args.extra is not None and arm.startswith("dual_path_"):
            if arm not in {keep.get(x, x) for x in args.extra}:
                continue
        data = load(rd, arm)
        if data is None:
            print(f"[skip] {name}：尚无 checkpoint（{rd}）")
            continue
        arms.append((name, arm, data))

    print("=" * 84)
    print("各臂有效题数")
    for name, arm, data in arms:
        print(f"  {name:<14} {len(data):>4} 题")
    print("=" * 84)

    ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
    qmap = {q["id"]: q for q in ts["questions"]}
    by_arm = {arm: data for _, arm, data in arms}
    # 以正式四臂的 dual_path 为锚：所有配对比较都在它的题集上进行
    dp = by_arm["dual_path"]
    vec = by_arm["vector_rag"]
    base_ids = list(dp.keys())

    print("\n=== 总体分（按留出集真实比例加权 / 样本内算术平均）===")
    for name, arm, data in arms:
        by = {}
        for _, (ty, v) in data.items():
            by.setdefault(ty, []).append(v["overall"])
        num = den = 0.0
        for t, w in W.items():
            if by.get(t):
                num += w * st.mean(by[t]); den += w
        allv = [v["overall"] for _, v in data.values()]
        print(f"  {name:<14} 加权 {num/den:.4f}   未加权 {st.mean(allv):.4f}")

    print("\n=== 逐题型均值 ===")
    hdr = f"{'题型':<13}{'n':>5}" + "".join(f"{n:>16}" for n, _, _ in arms)
    print(hdr)
    for t in TYPES:
        ids_t = [i for i in base_ids if qmap.get(i, {}).get("query_type") == t]
        row = f"{t:<13}{len(ids_t):>5}"
        for name, arm, data in arms:
            v = [data[i][1]["overall"] for i in ids_t if i in data]
            row += f"{st.mean(v):>16.3f}" if v else f"{'N/A':>16}"
        print(row)

    # 关键对照：与 dual_path 逐臂比较，只看图谱真正启用 / 被门控关掉的层
    fired = [i for i in base_ids if not _graph_is_blind(qmap[i]["question"])]
    gated = [i for i in base_ids if _graph_is_blind(qmap[i]["question"])]
    print(f"\n（图谱启用 {len(fired)} 题；被门控关掉 {len(gated)} 题）")

    for name, arm, data in arms:
        if arm == "dual_path":
            continue
        print(f"\n=== {name} vs 双路 ===")
        paired(data, dp, base_ids, "全样本", 1)
        paired(data, dp, fired, "只图谱启用", 4)
        paired(data, dp, gated, "只被门控关掉", 4)
        for key in ["correctness", "completeness", "faithfulness"]:
            paired(data, dp, base_ids, f"  全样本·{key}", 4, key)

    print("\n=== 双路 vs 向量RAG（基线对照，看两项新臂是否改变了这个结论）===")
    for name, arm, data in arms:
        if not arm.startswith("dual_path"):
            continue
        print(f"\n-- {name} vs 向量RAG --")
        paired(data, vec, base_ids, "全样本", 1)
        for key in ["correctness", "completeness", "faithfulness"]:
            paired(data, vec, base_ids, f"  全样本·{key}", 4, key)

    print("\n=== 答案长度（字符）===")
    for name, arm, data in arms:
        ids = [i for i in base_ids if i in data]
        print(f"  {name:<14} 平均 {st.mean([data[i][1]['len'] for i in ids]):>6.0f} 字")


if __name__ == "__main__":
    main()
