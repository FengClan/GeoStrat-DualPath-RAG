# -*- coding: utf-8 -*-
"""离线统计：图谱通道在正式实验的 300 题上到底"开火"了多少次。

动机：若 retrieve_graph_context 的子串匹配在多数题上匹配不到任何实体（R2#14 指出的
问题），双路在这些题上就退化为向量路，于是"双路 ≈ 向量"是个平凡结论，而非"融合无用"。
本脚本只做本地图谱查询，不调用任何 API。

同时按题型、按"图谱是否开火"分层，看双路相对向量的配对差是否只在开火时出现。
"""
import sys, os, json, math, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding="utf-8")

from services.kg_service import retrieve_graph_context

RD = "evaluation/results_full_20260914"
KB = "experiment_kb"
TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]

ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
qmap = {q["id"]: q for q in ts["questions"]}

sub = json.load(open(f"{RD}/progress_dual_path.json", encoding="utf-8"))
ids = [r["id"] for r in sub["per_question_results"] if "qa_metrics" in r]


def overall(q):
    return (0.5385 * q["correctness"] + 0.3077 * q["faithfulness"]
            + 0.1538 * q["completeness"])


def load(bk):
    d = json.load(open(f"{RD}/progress_{bk}.json", encoding="utf-8"))
    return {r["id"]: (r["query_type"], overall(r["qa_metrics"]))
            for r in d["per_question_results"]
            if "qa_metrics" in r and "error" not in r}


dual, vec = load("dual_path"), load("vector_rag")

fired = {}
for i in ids:
    q = qmap.get(i)
    if not q:
        continue
    try:
        edges, _ = retrieve_graph_context(q["question"], KB)
        fired[i] = len(edges or [])
    except Exception as e:
        fired[i] = -1

n_all = len(fired)
n_fire = sum(1 for v in fired.values() if v > 0)
print(f"共 {n_all} 题；图谱通道返回非空事实的 {n_fire} 题 ({100*n_fire/n_all:.1f}%)")
print(f"返回 0 条事实的 {sum(1 for v in fired.values() if v == 0)} 题；"
      f"查询报错 {sum(1 for v in fired.values() if v < 0)} 题")

print("\n=== 按题型：图谱开火率 ===")
for t in TYPES:
    sub_ids = [i for i in ids if i in fired and dual.get(i, ('',))[0] == t]
    if not sub_ids:
        continue
    k = sum(1 for i in sub_ids if fired[i] > 0)
    print(f"  {t:<13} {k:>4}/{len(sub_ids):<4} ({100*k/len(sub_ids):5.1f}%)")

print("\n=== 按「图谱是否开火」分层的 双路−向量 配对差 ===")
for t in TYPES + ["全样本"]:
    for label, cond in [("开火", lambda v: v > 0), ("未开火", lambda v: v == 0)]:
        sub_ids = [i for i in ids
                   if i in fired and cond(fired[i]) and i in dual and i in vec
                   and (t == "全样本" or dual[i][0] == t)]
        if len(sub_ids) < 5:
            continue
        d = [dual[i][1] - vec[i][1] for i in sub_ids]
        m, sd = st.mean(d), st.stdev(d)
        se = sd / math.sqrt(len(sub_ids))
        tt = m / se if se else 0
        lo, hi = m - 1.96 * se, m + 1.96 * se
        print(f"  {t:<13} {label:<5} n={len(sub_ids):>4}  差 {m:+.4f}  "
              f"95%CI [{lo:+.3f},{hi:+.3f}]  t={tt:+.2f}")
