#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多轮重复实验 + mean±SD——回应审稿意见 R2#8（"重复实验并报告方差/置信区间"）
与 R2#9（"消融的 6.4%/3.9% 改进也要重复"）。2026-09-11 导师会列为优先项。

背景：`get_model_response` 用 temperature=0.3（非 0），所以同一模型同一问题
两次跑出的答案**本来就不一样**。单轮结果只是 1 次抽样，报点估计而不报离散度，
是 R2#8 的核心指控。

本脚本做三件事：
  1. **固定一切可变项**（同一份抽样题集、同一 seed、同一 prompt、同一参数），
     只让模型自身的随机性变化——这才是"独立重复"，不是"配置漂移"。
  2. **每轮全新跑**（独立 checkpoint），避免第二轮复用第一轮结果导致 SD 恒为 0。
     脚本内置"两轮结果完全相同就报警"的自检——那通常意味着 checkpoint 被误复用。
  3. 汇总 mean、样本 SD、95% CI（t 分布小样本系数）；消融模式还额外报
     **配对差值**的 mean±SD 与相对提升%，因为 R2#9 要的正是那个 6.4%/3.9%。

⚠️ n 轮 ≠ 高置信：n=3 时 SD 本身也很不确定（df=2，t=4.303）。论文里要
   如实写清轮数，不要用 3 轮 SD 去谈"显著性"。

用法:
    # 主实验：4 基线 × 3 轮（在留出集上跑）
    python evaluation/run_multirun.py --rounds 3 --test-set test_sets/test_set_honest_test.json --total 300 --workers 8

    # 消融：完整方法 + 3 变体 × 3 轮，报配对差值的 mean±SD（R2#9）
    python evaluation/run_multirun.py --mode ablation --rounds 3 --total 150

    # 中断后续跑（同一 group id，只补没跑完的轮次）
    python evaluation/run_multirun.py --mode ablation --group 20260914_120000
"""
import sys, os, json, math, time, argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner
from evaluation.ablation import AblationRunner
from evaluation.evaluator import Evaluator
from evaluation.run_full_eval import (QUERY_TYPES, stratify_contested,
                                      stratify_proportional)

BASELINE_ARMS = ["direct_llm", "vector_rag", "graph_rag", "dual_path"]
# 消融臂（2026-09-19 归位）：三个名字全部注册在 BaselineRunner.get_runner 里、
# 共用 _dual_path_impl，只差一个开关。原先的 abl_no_filter 已删除 —— filter_noise
# 全代码库从未传过 True（该臂与完整方法实际完全相同），且论文表 4 本就没有这一项。
ABLATION_ARMS = ["dual_path", "dual_path_no_classify", "dual_path_no_citation"]

# 论文里要报的主指标（按 R2#18 还要报 token/延迟）
KEY_METRICS = ["overall_score", "qa_correctness", "qa_faithfulness",
               "qa_completeness", "citation_precision", "citation_recall",
               "retrieval_recall@5", "retrieval_mrr", "avg_latency_ms"]

# t 分布 95% 双侧临界值（小样本 CI 用；n=3 就别谈显著性了）
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t_critical(df):
    if df in _T95:
        return _T95[df]
    if df <= 0:
        return float("nan")
    for k in sorted(_T95):
        if df < k:
            return _T95[k]          # 取更保守的那侧
    return 1.96


def mean_sd(vals):
    """样本均值、样本标准差(ddof=1)、95% CI 半宽。n<2 时 SD/CI 为 None。"""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    m = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": round(m, 4), "sd": None, "ci95": None}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "ci95": round(t_critical(n - 1) * sd / math.sqrt(n), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def fmt(stat, digits=4):
    """把 mean_sd 结果格式化成 '0.8329 ± 0.0041' 形式。"""
    if not stat or stat.get("mean") is None:
        return "N/A"
    if stat.get("sd") is None:
        return f"{stat['mean']:.{digits}f} (n=1)"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def pair_delta(full_vals, var_vals):
    """配对差值（同一轮、同一题集 → 可配对，比独立两组比较更敏感）。
    返回绝对差值 mean±SD（Δ = 完整 − 变体，正=完整更好）与相对提升%。"""
    pairs = [(f, v) for f, v in zip(full_vals, var_vals)
             if f is not None and v is not None]
    if not pairs:
        return {"n": 0, "delta_mean": None, "delta_sd": None}
    deltas = [f - v for f, v in pairs]
    rels = [(f - v) / v * 100 for f, v in pairs if v]
    d = mean_sd(deltas)
    r = mean_sd(rels)
    wins = sum(1 for d_ in deltas if d_ > 0)
    return {"n": d["n"], "delta_mean": d["mean"], "delta_sd": d["sd"],
            "delta_ci95": d["ci95"],
            "rel_mean": r["mean"], "rel_sd": r["sd"],
            "wins": f"{wins}/{d['n']}"}


def run_round(ev, runner_fn, subset, workers, ckpt, tag):
    """跑一轮。返回 (results, usage_snapshot, wall_seconds)。"""
    from services.model_router import usage_stats
    usage_stats(reset=True)                    # 清零，得到本轮净用量
    t0 = time.time()
    res = ev.run_full_evaluation(subset, runner_fn, use_llm_judge=True,
                                 checkpoint_path=ckpt, workers=workers)
    wall = round(time.time() - t0, 1)
    res["_usage"] = usage_stats(reset=False)
    res["_wall_seconds"] = wall
    res["_round_tag"] = tag
    return res


def main():
    ap = argparse.ArgumentParser(description="多轮重复实验 mean±SD（R2#8/#9）")
    ap.add_argument("--mode", default="baseline",
                    choices=["baseline", "ablation", "both"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--test-set", default="test_sets/test_set_honest_dev.json",
                    help="默认 dev 集：多轮很贵，先用 dev；最终报告换留出集")
    ap.add_argument("--total", type=int, default=150, help="每轮题数（0=全量）")
    ap.add_argument("--seed", type=int, default=42, help="抽样 seed，固定")
    ap.add_argument("--stratify", default="contested",
                    choices=["contested", "proportional"],
                    help="抽样方式，须与主表 run_full_eval 一致，否则多轮的题集与"
                         "主表对不上、mean±SD 无法与主表并列报告。默认 contested。")
    ap.add_argument("--reuse-r1", default=None, metavar="DIR",
                    help="把 DIR 下已有的 progress_<arm>.json 当作第 1 轮复制进本轮次"
                         "目录（仅在题集与 seed 完全相同、且配置未改过时才可用）。"
                         "省掉一整轮成本。")
    ap.add_argument("--model", default="qwen3.5-plus")
    ap.add_argument("--judge-model", default="deepseek-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--results-dir", default="./evaluation/results")
    ap.add_argument("--group", default=None,
                    help="续跑用：复用该 group 的 checkpoint（默认按当前时间新建）")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--price-in", type=float, default=None,
                    help="输入 token 单价（元/千 token）；不填则只报 token 数不折算成本")
    ap.add_argument("--price-out", type=float, default=None,
                    help="输出 token 单价（元/千 token）")
    args = ap.parse_args()

    if args.no_thinking:
        os.environ["QWEN_ENABLE_THINKING"] = "0"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    group = args.group or time.strftime("%Y%m%d_%H%M%S")
    ckdir = os.path.join(args.results_dir, f"multirun_{group}")
    os.makedirs(ckdir, exist_ok=True)

    # ---- 抽样一次，所有轮次共用同一份题集（这是"独立重复"的前提）----
    ts = json.load(open(args.test_set, encoding="utf-8"))
    qs = ts["questions"]
    if args.total and args.total < len(qs):
        qs = (stratify_contested if args.stratify == "contested"
              else stratify_proportional)(qs, args.total, args.seed)
    subset = {"version": ts.get("version"), "questions": qs}
    cnt = Counter(q.get("query_type", "?") for q in qs)

    ev = Evaluator(model=args.model, results_dir=args.results_dir,
                   judge_model=args.judge_model or None)
    thinking = "关" if args.no_thinking else "开"

    print("=" * 74)
    print(f"[多轮重复] group={group}  {args.rounds} 轮 × {len(qs)} 题  模式={args.mode}")
    print(f"           生成={args.model}  裁判={ev.judge_model}"
          f"{'  (独立裁判 ✓)' if ev.is_independent_judge else '  (⚠ 同模型裁判)'}")
    print(f"           题源={os.path.basename(os.path.normpath(args.test_set))}"
          f"  seed={args.seed}  workers={args.workers}  思考={thinking}")
    print(f"           构成: " + "  ".join(f"{t}:{cnt.get(t, 0)}" for t in QUERY_TYPES))
    print("           每题集在 {0} 轮中保持完全一致（同一抽样，不重新抽）".format(args.rounds))
    print("=" * 74)

    arms = []
    if args.mode in ("baseline", "both"):
        arms += [("baseline", a) for a in BASELINE_ARMS]
    if args.mode in ("ablation", "both"):
        arms += [("ablation", a) for a in ABLATION_ARMS]

    report = {
        "group": group,
        "config": {
            "rounds": args.rounds, "mode": args.mode, "test_set": args.test_set,
            "sample_size": len(qs), "seed": args.seed, "workers": args.workers,
            "model": args.model, "judge_model": ev.judge_model,
            "independent_judge": ev.is_independent_judge,
            "enable_thinking": not args.no_thinking,
            "temperature": 0.3,
            "note": "temperature=0.3（非 0），故各轮结果本就不同；"
                    "本题集在所有轮次间固定不变，只有模型随机性变化。",
        },
        "sample_composition": dict(cnt),
        "rounds": {},          # arm -> [per-round overall_scores(+usage)]
        "summary": {},         # arm -> metric -> mean±SD
        "by_query_type": {},   # arm -> type -> metric -> mean±SD
        "ablation_delta": {},  # R2#9：配对差值
    }

    for mode, arm in arms:
        # 复合 key：--mode both 时 baseline 的 dual_path 与 ablation 的 dual_path
        # 会用同一份 checkpoint 目录，只用 arm 当 key 会互相覆盖，汇总表里少掉一臂。
        # （两者现在是同一个 runner —— 消融的"D. 完整方法"就是主表"双路"，
        #   但分别跑的两份结果仍应各存一份，故 key 依旧带 mode 前缀。）
        key = f"{mode}__{arm}"
        runner_obj = (AblationRunner(model=args.model, results_dir=args.results_dir)
                      if mode == "ablation"
                      else BaselineRunner(model=args.model, results_dir=args.results_dir))
        label = (runner_obj.ABLATION_NAMES[arm] if mode == "ablation"
                 else runner_obj.BASELINE_NAMES[arm])
        # 消融臂与基线臂走同一个派发入口：不存在两套实现，也就不会跑偏。
        fn = runner_obj.get_runner(arm)

        print(f"\n{'-' * 74}\n[{mode}] {label}  ({arm})")
        per_round = []
        for r in range(1, args.rounds + 1):
            # 每轮独立 checkpoint：复用同一份会让第 2 轮直接跳过全部题目 → SD=0 假象
            ckpt = os.path.join(ckdir, f"{key}_r{r}.json")
            # 复用已有的一整轮（如主表 results_full_20260914）：只是把文件复制过来，
            # 聚合指标由 run_full_evaluation 从 per_question_results 重算，不用重跑。
            if args.reuse_r1 and not os.path.exists(ckpt):
                # 候选来源（按优先级）：
                #   1) progress_<arm>.json        —— 单轮全量跑留下的（只对第 1 轮有意义）
                #   2) baseline__<arm>_r<n>.json  —— 多轮跑留下的
                # 第 2 条是为消融的 dual_path 臂准备的：它与主表 baseline__dual_path
                # 现在是**同一个 runner**（同代码、同题集、同 seed），重跑不会带来
                # 任何新信息，纯粹浪费钱和时间。
                # 找不到就照常跑 —— 失败方向是"没省下钱"，不会跑出错的数据。
                cands = ([os.path.join(args.reuse_r1, f"progress_{arm}.json")]
                         if r == 1 else [])
                cands.append(os.path.join(args.reuse_r1, f"baseline__{arm}_r{r}.json"))
                src = next((c for c in cands if os.path.exists(c)), None)
                if src:
                    import shutil
                    shutil.copyfile(src, ckpt)
                    print(f"   [复用] 第 {r} 轮 <- {src}")
                elif r == 1:
                    print(f"   [复用] 跳过：{cands[0]} 不存在，第 1 轮照常跑")
            n_done = 0
            if os.path.exists(ckpt):
                try:
                    n_done = len(json.load(open(ckpt, encoding="utf-8"))
                                 .get("per_question_results", []))
                except Exception:
                    n_done = 0
            tag = f"r{r}" + (f"(续{n_done})" if n_done else "")
            res = run_round(ev, fn, subset, args.workers, ckpt, tag)
            # 被 Ctrl+C 中断：本轮只覆盖了一部分题。残缺轮**不能**计入 mean±SD——
            # 均值会被拉偏、方差更是失去意义，而报告里看不出来。就地停住。
            if res.get("interrupted"):
                print(f"\n[MULTIRUN] 已在「{label}」第 {r}/{args.rounds} 轮中断（Ctrl+C）。")
                print("[MULTIRUN] 该轮不完整，不计入 mean±SD（残缺轮的均值和方差都不可信）。")
                print(f"[MULTIRUN] 本轮进度已保存：{ckpt}")
                print("[MULTIRUN] 恢复：原样重跑同一条命令（同 --test-set/--total/"
                      "--seed/--results-dir）。不会重复已完成题。")
                return
            o = res["overall_scores"]
            per_round.append({
                "round": r, "overall_scores": o, "usage": res["_usage"],
                "wall_seconds": res["_wall_seconds"],
                "by_query_type": res["by_query_type"],
                "coverage": o.get("coverage"),
                "timing": res.get("timing"),
            })
            print(f"   第 {r}/{args.rounds} 轮完成  墙钟 {res['_wall_seconds']:7.1f}s  "
                  f"overall={o['overall_score']:.4f}  "
                  f"judge_failed={o.get('coverage', {}).get('judge_failed', 0)}")

        # ---- 自检：两轮结果逐位相同 = 大概率 checkpoint 被复用 ----
        sigs = [tuple(sorted((k, v) for k, v in rd["overall_scores"].items()
                             if isinstance(v, (int, float))))
                for rd in per_round]
        if len(sigs) > 1 and len(set(sigs)) < len(sigs):
            print("   ⚠️ 警告：有轮次的指标完全相同。正常随机性（temperature=0.3）下"
                  "不应发生。若本次传了 --group 续跑，说明这些轮是直接复用已跑完的"
                  "checkpoint 而未重跑 —— 它们的 SD 贡献必须是 0，不能当独立重复。")

        report["rounds"][key] = per_round

        # ---- 汇总 mean±SD ----
        summary = {}
        for m in KEY_METRICS:
            summary[m] = mean_sd([rd["overall_scores"].get(m) for rd in per_round])
        report["summary"][key] = summary

        # 逐题型（导师关注 COMPARATIVE：要报的是它的 mean±SD，不只是总分）
        bqt = {}
        for t in QUERY_TYPES:
            v = [rd["by_query_type"].get(t, {}).get("avg_score") for rd in per_round]
            if any(x is not None for x in v):
                bqt[t] = mean_sd(v)
        report["by_query_type"][key] = bqt

    # ---- 打印汇总表 ----
    print(f"\n{'=' * 74}\n[汇总] mean ± 样本SD（{args.rounds} 轮）\n{'=' * 74}")
    for mode, arm in arms:
        s = report["summary"][f"{mode}__{arm}"]
        print(f"\n  {mode}/{arm}")
        for m in KEY_METRICS:
            print(f"      {m:<20}{fmt(s[m])}"
                  + (f"   95%CI ±{s[m]['ci95']:.4f}" if s[m].get("ci95") else ""))
        bqt = report["by_query_type"][f"{mode}__{arm}"]
        if bqt:
            print("      按题型 (avg_score=(correctness+faithfulness)/2):")
            for t in QUERY_TYPES:
                if t in bqt:
                    print(f"        {t:<14}{fmt(bqt[t])}   (n={bqt[t]['n']})")

    # ---- 消融配对差值（R2#9 直接要的那个数）----
    if args.mode in ("ablation", "both"):
        print(f"\n{'=' * 74}\n[消融配对差值] Δ = 完整方法 − 变体（正 = 完整更好，同轮同题配对）")
        print(f"{'=' * 74}")
        full = [rd["overall_scores"]["overall_score"]
                for rd in report["rounds"]["ablation__dual_path"]]
        for arm in ABLATION_ARMS:
            if arm == "dual_path":
                continue
            var = [rd["overall_scores"]["overall_score"]
                   for rd in report["rounds"][f"ablation__{arm}"]]
            d = pair_delta(full, var)
            report["ablation_delta"][arm] = d
            if d["delta_mean"] is None:
                print(f"  {arm}: 无有效配对")
                continue
            print(f"\n  {arm}  (Δ overall_score)")
            print(f"      绝对差值   {d['delta_mean']:+.4f} ± {d['delta_sd']:.4f}"
                  f"   95%CI ±{d['delta_ci95']:.4f}")
            print(f"      相对提升   {d['rel_mean']:+.2f}% ± {d['rel_sd']:.2f}%")
            print(f"      完整方法胜出轮次 {d['wins']}")
        print("\n  ⚠️ n 小（每臂 {} 轮）时 SD 本身不确定度很大；论文里写清轮数，"
              "不要用 3 轮 SD 宣称显著性。".format(args.rounds))

    # ---- token / 成本（R2#18）----
    print(f"\n{'=' * 74}\n[token 用量]（每臂 {args.rounds} 轮合计）\n{'=' * 74}")
    for mode, arm in arms:
        agg = {}
        for rd in report["rounds"][f"{mode}__{arm}"]:
            for mdl, u in (rd["usage"] or {}).items():
                a = agg.setdefault(mdl, {"calls": 0, "prompt_tokens": 0,
                                         "completion_tokens": 0, "total_tokens": 0})
                for k in a:
                    a[k] += u.get(k, 0)
        for mdl, u in agg.items():
            line = (f"  {arm:<18}{mdl:<18} calls={u['calls']:<6}"
                    f"in={u['prompt_tokens']:<9}out={u['completion_tokens']:<9}"
                    f"total={u['total_tokens']}")
            if args.price_in is not None and args.price_out is not None:
                cost = (u["prompt_tokens"] * args.price_in
                        + u["completion_tokens"] * args.price_out) / 1000
                line += f"  ≈¥{cost:.2f}"
                u["cost_cny"] = round(cost, 4)
            print(line)
        report["summary"][f"{mode}__{arm}_usage"] = agg
    if args.price_in is None:
        print("\n  （未提供 --price-in/--price-out，只报 token 数。"
              "单价请以 DashScope 官方价格页为准，不要凭记忆填。）")

    out = os.path.join(args.results_dir, f"multirun_{group}.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n输出: {out}\nCheckpoint 目录: {ckdir}")
    print("★ 若本轮用于最终报告，请确认题源是留出集（test）而非 dev。")


if __name__ == "__main__":
    main()
