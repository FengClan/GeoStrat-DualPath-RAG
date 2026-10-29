# -*- coding: utf-8 -*-
"""消融结果分析：题级配对检验（不发 API，纯读文件）。

⚠️ 为什么不直接用报告里的 ablation_delta：
   run_multirun 报的 delta_sd 是**轮间** SD（3 轮），拿它当配对差的标准误
   会严重高估显著性 —— 那正是本会话早先 t=10.88 的错误来源。正确做法：
   先把每题 3 轮的平均差算出来，再对 300 个题级值做单样本 t 检验。

用法: python evaluation/_analyze_ablation.py
"""
import json, os, sys, math
sys.stdout.reconfigure(encoding="utf-8")

CKPT = "evaluation/results/multirun_ablation_20260919"
ARMS = {
    "dual_path":             "D. 完整方法",
    "dual_path_no_classify": "D − 自适应分类",
    "dual_path_no_citation": "D − 细粒度引用",
}
METRICS = ["overall_score", "correctness", "faithfulness", "completeness"]
# overall_score 是 evaluator 的加权公式；checkpoint 里只有分项，须自己算
W = {"correctness": 0.5385, "faithfulness": 0.3077, "completeness": 0.1538}


def overall(m):
    return sum(m.get(k, 0.0) * w for k, w in W.items())


def load(arm):
    """返回 {round: {qid: {metric: value}}}"""
    rounds = {}
    for r in (1, 2, 3):
        p = os.path.join(CKPT, f"ablation__{arm}_r{r}.json")
        d = json.load(open(p, encoding="utf-8"))
        per = {}
        for x in d["per_question_results"]:
            if "qa_metrics" not in x:
                continue
            m = x["qa_metrics"]
            row = {k: m.get(k, 0.0) for k in ("correctness", "faithfulness", "completeness")}
            row["overall_score"] = overall(m)
            row["_judge_failed"] = bool(m.get("judge_failed"))
            row["_qtype"] = x.get("query_type")
            per[x["id"]] = row
        rounds[r] = per
    return rounds


def tstat(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    return mu, sd, se, (mu / se if se else 0.0)


data = {a: load(a) for a in ARMS}
base = data["dual_path"]

# 三种口径的临界 t（双侧）：n=300 时用正态近似 1.96 / 2.498
T_RAW, T_BONF = 1.96, 2.498

print("=" * 84)
print("消融结果：题级配对检验（300 题 × 3 轮，同题同轮配对）")
print("=" * 84)
print(f"  临界值：未校正 |t|>{T_RAW}；Bonferroni k=4 |t|>{T_BONF}   (df≈299)")
print(f"  overall_score = correctness×{W['correctness']} + faithfulness×{W['faithfulness']}"
      f" + completeness×{W['completeness']}")

for arm, label in list(ARMS.items())[1:]:
    var = data[arm]
    print()
    print("-" * 84)
    print(f"Δ = {ARMS['dual_path']} − {label}")
    print("-" * 84)
    for metric in METRICS:
        # 每题先跨轮平均，再对题级值做单样本 t
        diffs = []
        for qid in base[1]:
            if any(qid not in var[r] for r in (1, 2, 3)):
                continue
            b = sum(base[r][qid][metric] for r in (1, 2, 3)) / 3
            v = sum(var[r][qid][metric] for r in (1, 2, 3)) / 3
            diffs.append(b - v)
        mu, sd, se, t = tstat(diffs)
        n_win = sum(1 for x in diffs if x > 0)
        mark = ""
        if abs(t) > T_BONF:
            mark = " ★(过Bonferroni)"
        elif abs(t) > T_RAW:
            mark = " *(仅过未校正)"
        else:
            mark = " (不显著)"
        print(f"  {metric:<16} Δ={mu:+.4f}  SE={se:.4f}  t={t:+.2f}{mark}"
              f"   题级胜 {n_win}/{len(diffs)}")

# 逐题型（overall_score）
print()
print("=" * 84)
print("逐题型（overall_score，Δ = 完整方法 − 变体）")
print("=" * 84)
for arm, label in list(ARMS.items())[1:]:
    var = data[arm]
    print(f"\n  {label}")
    types = sorted({base[1][q]["_qtype"] for q in base[1]})
    for qt in types:
        diffs = []
        for qid in base[1]:
            if base[1][qid]["_qtype"] != qt or any(qid not in var[r] for r in (1, 2, 3)):
                continue
            b = sum(base[r][qid]["overall_score"] for r in (1, 2, 3)) / 3
            v = sum(var[r][qid]["overall_score"] for r in (1, 2, 3)) / 3
            diffs.append(b - v)
        if not diffs:
            continue
        mu, sd, se, t = tstat(diffs)
        star = "★" if abs(t) > 2.498 else ("*" if abs(t) > 1.96 else " ")
        print(f"    {qt:<14} n={len(diffs):>3}  Δ={mu:+.4f}  t={t:+.2f} {star}")

# judge_failed 统计
print()
print("=" * 84)
print("judge_failed（裁判两次都解析失败 → 按 0.5 中性值计入）")
print("=" * 84)
for arm, label in ARMS.items():
    counts = [sum(1 for q in data[arm][r].values() if q["_judge_failed"]) for r in (1, 2, 3)]
    print(f"  {label:<18} 各轮 {counts}   合计 {sum(counts)}/900 "
          f"({sum(counts)/900*100:.2f}%)")
