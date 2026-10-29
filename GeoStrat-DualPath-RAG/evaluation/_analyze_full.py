# -*- coding: utf-8 -*-
"""正式实验结果的配对检验（contested 分层，四臂各 300 题）。

只读结果目录里最新的 baseline_full_*.json 与各臂 checkpoint，不发任何 API 请求。
输出：逐题型总体分、双路 vs 各基线的同题配对差与 95%CI、多重比较校正后的显著性。
"""
import sys, os, json, glob, math, statistics as st

sys.stdout.reconfigure(encoding="utf-8")

RD = "evaluation/results_full_20260914"
ARMS = [("direct_llm", "纯LLM"), ("vector_rag", "向量RAG"),
        ("graph_rag", "图谱RAG"), ("dual_path", "双路")]
TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]
W = {"FACTUAL": 0.6353, "REASONING": 0.2481, "COMPARATIVE": 0.0873, "SPATIAL": 0.0294}


def overall(q):
    return (0.5385 * q["correctness"] + 0.3077 * q["faithfulness"]
            + 0.1538 * q["completeness"])


def load(bk):
    d = json.load(open(f"{RD}/progress_{bk}.json", encoding="utf-8"))
    per = {}
    for r in d["per_question_results"]:
        if "qa_metrics" in r and "error" not in r:
            per[r["id"]] = (r["query_type"], overall(r["qa_metrics"]))
    return per


data = {bk: load(bk) for bk, _ in ARMS}
print("各臂有效题数:", {n: len(data[bk]) for bk, n in ARMS})

print("\n=== 逐题型均值（各类内算术平均）===")
print(f"{'题型':<13}{'n':>5}" + "".join(f"{n:>12}" for _, n in ARMS))
for t in TYPES:
    ids = [i for i, (ty, _) in data["dual_path"].items() if ty == t]
    row = f"{t:<13}{len(ids):>5}"
    for bk, _ in ARMS:
        vals = [data[bk][i][1] for i in ids if i in data[bk]]
        row += f"{st.mean(vals):>12.3f}" if vals else f"{'N/A':>12}"
    print(row)

print("\n=== 总体分（加权 / 未加权）===")
for bk, n in ARMS:
    by = {}
    for _, (ty, v) in data[bk].items():
        by.setdefault(ty, []).append(v)
    wsum = sum(W[t] * st.mean(by[t]) for t in by if by[t])
    wden = sum(W[t] for t in by if by[t])
    allv = [v for _, v in data[bk].values()]
    print(f"  {n:<10} 加权 {wsum/wden:.4f}   未加权 {st.mean(allv):.4f}   n={len(allv)}")


def paired(a, b, ids, label, n_tests, alpha=0.05):
    """同题配对差 + 95%CI。n_tests = 同时做的检验数，用于 Bonferroni 校正。"""
    both = [i for i in ids if i in a and i in b]
    if len(both) < 2:
        return
    d = [a[i][1] - b[i][1] for i in both]
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(both))
    crit = 1.96
    # Bonferroni：把 α 平摊到 n_tests 个检验上对应的双侧临界 t（正态近似）
    from statistics import NormalDist
    crit_b = NormalDist().inv_cdf(1 - alpha / (2 * n_tests))
    lo, hi = m - crit * se, m + crit * se
    lo_b, hi_b = m - crit_b * se, m + crit_b * se
    t = m / se if se else 0.0
    verdict = "显著" if abs(t) > crit_b else ("边缘" if abs(t) > crit else "不显著")
    print(f"  {label:<28} n={len(both):>4}  差 {m:+.4f}  95%CI [{lo:+.3f},{hi:+.3f}]"
          f"  t={t:+.2f}  {verdict}")
    return m, t, lo_b, hi_b


print("\n=== 双路 vs 向量RAG（同题配对；Bonferroni 校正 k=4）===")
for t in TYPES:
    ids = [i for i, (ty, _) in data["dual_path"].items() if ty == t]
    paired(data["dual_path"], data["vector_rag"], ids, t, 4)
paired(data["dual_path"], data["vector_rag"], list(data["dual_path"]),
       "全样本（不分类）", 1)

print("\n=== 双路 vs 图谱RAG（同题配对；Bonferroni 校正 k=4）===")
for t in TYPES:
    ids = [i for i, (ty, _) in data["dual_path"].items() if ty == t]
    paired(data["dual_path"], data["graph_rag"], ids, t, 4)

print("\n=== 向量RAG vs 图谱RAG（两条通道的直接对比）===")
for t in TYPES:
    ids = [i for i, (ty, _) in data["dual_path"].items() if ty == t]
    paired(data["vector_rag"], data["graph_rag"], ids, t, 4)
