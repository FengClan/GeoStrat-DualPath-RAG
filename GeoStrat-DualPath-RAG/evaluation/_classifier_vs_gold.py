# -*- coding: utf-8 -*-
"""分类器判定 vs 测试集金标：混淆矩阵 + 按分类器路由重算消融差值。

背景：checkpoint 里存的 query_type 是**测试集金标**（evaluator.py:648
`q.get("query_type")`），分类器自己的输出被丢弃了。所以"贡献集中在对比题"
这个说法必须按**分类器实际路由**重算 —— 被路由到 COMPARATIVE 模板的题，
未必是金标为 COMPARATIVE 的题。

不发 API（分类器是纯规则）。用法: python evaluation/_classifier_vs_gold.py
"""
import json, os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from services.query_classifier import get_retrieval_config
from collections import Counter, defaultdict

CKPT = "evaluation/results/multirun_ablation_20260919"
W = {"correctness": 0.5385, "faithfulness": 0.3077, "completeness": 0.1538}
TYPES = ["FACTUAL", "COMPARATIVE", "REASONING", "SPATIAL"]


def ov(row):
    return row["c"] * W["correctness"] + row["f"] * W["faithfulness"] + row["m"] * W["completeness"]


def load(arm):
    rounds = {}
    for r in (1, 2, 3):
        d = json.load(open(f"{CKPT}/ablation__{arm}_r{r}.json", encoding="utf-8"))
        rounds[r] = {}
        for x in d["per_question_results"]:
            if "qa_metrics" not in x:
                continue
            m = x["qa_metrics"]
            rounds[r][x["id"]] = {"c": m.get("correctness", 0.0),
                                  "f": m.get("faithfulness", 0.0),
                                  "m": m.get("completeness", 0.0),
                                  "_qt": x.get("query_type"),
                                  "_jf": bool(m.get("judge_failed"))}
    return rounds


ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
qs = {q["id"]: q for q in ts["questions"]}

base, var = load("dual_path"), load("dual_path_no_classify")
ids = [i for i in base[1] if i in qs]

# --- 分类器判定 ---
pred, gold = {}, {}
for i in ids:
    pred[i] = get_retrieval_config(qs[i]["question"])["query_type"]
    gold[i] = base[1][i].get("_gold") if False else None  # 占位，下面从文件读

ck = json.load(open(f"{CKPT}/ablation__dual_path_r1.json", encoding="utf-8"))
gold = {x["id"]: x.get("query_type") for x in ck["per_question_results"]
        if "qa_metrics" in x}

print("=" * 88)
print("一、混淆矩阵：行=测试集金标，列=关键词分类器判定（contested 300 题）")
print("=" * 88)
M = defaultdict(Counter)
for i in ids:
    M[gold[i]][pred[i]] += 1
_hdr = "金标\\判定"
print(f"  {_hdr:<14}" + "".join(f"{t:>13}" for t in TYPES) + f"{'合计':>9}{'判对率':>9}")
for g in TYPES:
    row = M[g]
    tot = sum(row.values())
    hit = row.get(g, 0)
    cells = "".join(f"{row.get(t, 0):>13}" for t in TYPES)
    print(f"  {g:<14}{cells}{tot:>9}{hit/tot*100:>8.1f}%")
tot_all = sum(sum(M[g].values()) for g in TYPES)
hit_all = sum(M[t][t] for t in TYPES)
print(f"  {'合计':<14}{'':>13}{'':>13}{'':>13}{'':>13}{tot_all:>9}{hit_all/tot_all*100:>8.1f}%")

print()
print("  逐类召回（金标这一类里，被正确路由的比例）：")
for t in TYPES:
    tot = sum(M[t].values())
    print(f"    {t:<13} {M[t].get(t,0):>3}/{tot:<4} = {M[t].get(t,0)/tot*100:5.1f}%")

print()
print("=" * 88)
print("二、Δ（完整 − 无分类）按【分类器路由】分组，而非金标")
print("=" * 88)
by_pred = defaultdict(list)
by_gold = defaultdict(list)
for i in ids:
    if any(i not in var[r] for r in (1, 2, 3)):
        continue
    b = sum(ov(base[r][i]) for r in (1, 2, 3)) / 3
    v = sum(ov(var[r][i]) for r in (1, 2, 3)) / 3
    by_pred[pred[i]].append(b - v)
    by_gold[gold[i]].append(b - v)


def st(d):
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return n, mu, mu / (sd / math.sqrt(n))


print("  按【分类器路由】：")
for t in TYPES:
    if t not in by_pred:
        continue
    n, mu, tt = st(by_pred[t])
    star = " ★" if abs(tt) > 2.498 else (" *" if abs(tt) > 1.96 else "")
    print(f"    {t:<13} n={n:>3}  Δ={mu:+.4f}  t={tt:+.2f}{star}")
print("  按【金标】：")
for t in TYPES:
    if t not in by_gold:
        continue
    n, mu, tt = st(by_gold[t])
    star = " ★" if abs(tt) > 2.498 else (" *" if abs(tt) > 1.96 else "")
    print(f"    {t:<13} n={n:>3}  Δ={mu:+.4f}  t={tt:+.2f}{star}")

print()
print("=" * 88)
print("三、关键交叉：金标 COMPARATIVE 的题，被分到哪个模板了？")
print("=" * 88)
for track, name in [(gold, "金标 COMPARATIVE"), (None, None)]:
    pass
sel = [i for i in ids if gold[i] == "COMPARATIVE"]
c = Counter(pred[i] for i in sel)
print(f"  金标 COMPARATIVE 共 {len(sel)} 题 → 分类器把它们路由到：{dict(c)}")
d_routed = [x for i, x in zip(sel, [0] * len(sel)) if pred[i] == "COMPARATIVE"]
diffs_routed = []
for i in sel:
    if pred[i] != "COMPARATIVE":
        continue
    b = sum(ov(base[r][i]) for r in (1, 2, 3)) / 3
    v = sum(ov(var[r][i]) for r in (1, 2, 3)) / 3
    diffs_routed.append(b - v)
if diffs_routed:
    n, mu, tt = st(diffs_routed)
    print(f"  其中被正确路由的 {n} 题：Δ={mu:+.4f}  t={tt:+.2f}")
sel2 = [i for i in ids if pred[i] == "COMPARATIVE" and gold[i] != "COMPARATIVE"]
if sel2:
    d2 = []
    for i in sel2:
        b = sum(ov(base[r][i]) for r in (1, 2, 3)) / 3
        v = sum(ov(var[r][i]) for r in (1, 2, 3)) / 3
        d2.append(b - v)
    n2, mu2, t2 = st(d2)
    print(f"  金标不是对比题、但被路由成对比模板的 {n2} 题：Δ={mu2:+.4f}  t={t2:+.2f}")
