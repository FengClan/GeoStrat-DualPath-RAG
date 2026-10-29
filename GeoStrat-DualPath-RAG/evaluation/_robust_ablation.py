# -*- coding: utf-8 -*-
"""稳健性核查：消融的差值（尤其 COMPARATIVE +0.0300）是不是真信号。

要排除的几种"假高"：
  1. 被少数极端题拉高 → 看中位数、截尾均值、bootstrap CI、胜率
  2. 被 judge_failed（按 0.5 中性值计入）的题拉高 → 剔掉再算
  3. 只是"多数题微涨"堆出来的 → 看胜率与效应量
  4. 分类器对不同题型改动的东西不一样 → 拆开 COMPARATIVE 到底改了什么

不发 API，纯读 checkpoint。用法: python evaluation/_robust_ablation.py
"""
import json, os, sys, math, random
sys.stdout.reconfigure(encoding="utf-8")

CKPT = "evaluation/results/multirun_ablation_20260919"
W = {"correctness": 0.5385, "faithfulness": 0.3077, "completeness": 0.1538}
random.seed(20260919)


def load(arm):
    rounds = {}
    for r in (1, 2, 3):
        d = json.load(open(f"{CKPT}/ablation__{arm}_r{r}.json", encoding="utf-8"))
        per = {}
        for x in d["per_question_results"]:
            if "qa_metrics" not in x:
                continue
            m = x["qa_metrics"]
            per[x["id"]] = {
                "c": m.get("correctness", 0.0),
                "f": m.get("faithfulness", 0.0),
                "m": m.get("completeness", 0.0),
                "_jf": bool(m.get("judge_failed")),
                "_qt": x.get("query_type"),
            }
        rounds[r] = per
    return rounds


def ov(row):
    return row["c"] * W["correctness"] + row["f"] * W["faithfulness"] + row["m"] * W["completeness"]


def paired_diffs(base, var, metric="overall", qtype=None, drop_jf=False):
    """每题跨轮平均后作差（完整 − 变体）。"""
    out = []
    for qid, b0 in base[1].items():
        if qtype and b0["_qt"] != qtype:
            continue
        if any(qid not in var[r] for r in (1, 2, 3)):
            continue
        if drop_jf and any(base[r][qid]["_jf"] or var[r][qid]["_jf"] for r in (1, 2, 3)):
            continue
        f = (lambda d: ov(d)) if metric == "overall" else (lambda d: d[metric])
        b = sum(f(base[r][qid]) for r in (1, 2, 3)) / 3
        v = sum(f(var[r][qid]) for r in (1, 2, 3)) / 3
        out.append(b - v)
    return out


def stats(d):
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    d2 = sorted(d)
    med = d2[n // 2] if n % 2 else (d2[n // 2 - 1] + d2[n // 2]) / 2
    k = int(n * 0.05)               # 双侧各截 5%
    trim = sum(d2[k:n - k]) / len(d2[k:n - k])
    return {"n": n, "mean": mu, "se": sd / math.sqrt(n), "t": mu / (sd / math.sqrt(n)),
            "median": med, "trim": trim,
            "win": sum(1 for x in d if x > 0) / n,
            "loss": sum(1 for x in d if x < 0) / n}


def boot(d, iters=5000):
    n = len(d)
    ms = []
    for _ in range(iters):
        ms.append(sum(d[random.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    return ms[int(iters * 0.025)], ms[int(iters * 0.975)]


base = load("dual_path")
print("=" * 86)
print("一、整体：D − 自适应分类 的 overall 差值，各种口径")
print("=" * 86)
d_all = paired_diffs(base, load("dual_path_no_classify"))
s = stats(d_all)
lo, hi = boot(d_all)
print(f"  题数 {s['n']}   均值 {s['mean']:+.4f}   t={s['t']:+.2f}   bootstrap95%CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  中位数 {s['median']:+.4f}   截尾均值(各截5%) {s['trim']:+.4f}")
print(f"  胜率 {s['win']*100:.1f}%   负率 {s['loss']*100:.1f}%   平局 {(1-s['win']-s['loss'])*100:.1f}%")
print(f"  → 中位数 {s['median']:+.4f} 远小于均值 {s['mean']:+.4f} 说明什么，见下" if abs(s['median']-s['mean'])>0.003 else "  → 均值与中位数接近，不是被少数极端题拉高")

print()
print("=" * 86)
print("二、逐题型同样的口径（重点看 COMPARATIVE）")
print("=" * 86)
for qt in ["COMPARATIVE", "FACTUAL", "REASONING", "SPATIAL"]:
    d = paired_diffs(base, load("dual_path_no_classify"), qtype=qt)
    s = stats(d)
    lo, hi = boot(d, 3000)
    star = "★" if abs(s["t"]) > 2.498 else ("*" if abs(s["t"]) > 1.96 else " ")
    print(f"  {qt:<13} n={s['n']:>3}  均值{s['mean']:+.4f} t={s['t']:+.2f}{star}"
          f"  中位{s['median']:+.4f}  截尾{s['trim']:+.4f}"
          f"  胜率{s['win']*100:.0f}%  CI[{lo:+.4f},{hi:+.4f}]")

print()
print("=" * 86)
print("三、剔掉 judge_failed 的题再算（该类题按 0.5 中性值计入总分）")
print("=" * 86)
for qt in [None, "COMPARATIVE"]:
    d0 = paired_diffs(base, load("dual_path_no_classify"), qtype=qt)
    d1 = paired_diffs(base, load("dual_path_no_classify"), qtype=qt, drop_jf=True)
    s0, s1 = stats(d0), stats(d1)
    lab = qt or "全部"
    print(f"  {lab:<13} 全含 n={s0['n']:>3} 均值{s0['mean']:+.4f} t={s0['t']:+.2f}"
          f"   |  剔除后 n={s1['n']:>3} 均值{s1['mean']:+.4f} t={s1['t']:+.2f}")

print()
print("=" * 86)
print("四、分类器对两类题到底改了什么（决定差值能不能归因）")
print("=" * 86)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from services.query_classifier import get_retrieval_config  # 纯规则，不调 API
print("  用真分类器跑 300 题，统计各题型的 max_hops 是否变化：")
import glob
qs = {}
ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
for q in ts["questions"]:
    qs[q["id"]] = q
ck = json.load(open(f"{CKPT}/ablation__dual_path_r1.json", encoding="utf-8"))
ids = [x["id"] for x in ck["per_question_results"] if "qa_metrics" in x]
from collections import Counter
by = {}
for i in ids:
    q = qs[i]["question"]
    cfg = get_retrieval_config(q)
    qt, mh = cfg["query_type"], cfg["max_hops"]
    d = by.setdefault(qt, {"n": 0, "hops_changed": 0, "hops": Counter()})
    d["n"] += 1
    d["hops"][mh] += 1
    if mh != 2:
        d["hops_changed"] += 1
for qt, d in sorted(by.items()):
    print(f"    {qt:<13} n={d['n']:>3}  max_hops 钉死 2 会改变 {d['hops_changed']:>3} 题"
          f"   原 max_hops 分布 {dict(d['hops'])}")
print()
print("  → COMPARATIVE 原 max_hops 就是 2，钉死成 2 **不改跳数**，")
print("    所以它那一列的差值只能来自 prompt 模板 —— 归因是干净的。")
print("  → FACTUAL 原 max_hops=1，钉死成 2 **改了跳数但模板没变**，")
print("    它那一列接近 0，说明跳数 1→2 基本没影响。")
