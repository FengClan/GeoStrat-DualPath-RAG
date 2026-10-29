#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D4 放大评估：按比例分层抽样 + 断点续跑，跑 4 基线全对比。

抽样策略（方案A：按比例分层）：
  - SPATIAL 33 题全取（样本最小类，避免抽样不足）
  - 其余三类 FACTUAL/REASONING/COMPARATIVE 按真实比例 634:208:130 补足到 total-33

断点续跑：
  - 每个基线独立 checkpoint：results/progress_<baseline>.json
  - 每题成功后立即落盘；中断后重跑同一命令自动跳过已完成题、重跑失败题
  - 完成后 checkpoint 保留（可删除）

用法：
  python evaluation/run_full_eval.py --total 300 --seed 42
  python evaluation/run_full_eval.py --total 300 --seed 42 --only dual_path
  python evaluation/run_full_eval.py --total 300 --seed 42 --no-llm-judge
"""
import sys, os, json, random, argparse, time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator

QUERY_TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]


def _allocate(n, sizes):
    """最大余数法按比例分配 n 到各桶，保证总和=n。"""
    total = sum(sizes)
    raw = [n * s / total for s in sizes]
    floors = [int(r) for r in raw]
    rem = n - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    alloc = floors[:]
    for i in order[:rem]:
        alloc[i] += 1
    return alloc


def stratify_proportional(questions, total, seed):
    """SPATIAL 全取，其余按比例补足 total（最大余数法，可复现）。

    注意：SPATIAL 全取的策略只在 SPATIAL 数量**小于** total 时成立。
    此前无条件全取，一旦测试集里 SPATIAL 数量 >= total（例如扩集后 SPATIAL
    有 300 题而 --total 取 300），抽样结果会退化为 100% SPATIAL、其余三类全丢。
    现在超过容量时改为四类统一按比例分配。
    """
    rng = random.Random(seed)
    by_type = {}
    for q in questions:
        by_type.setdefault(q.get("query_type", "FACTUAL"), []).append(q)

    spatial = list(by_type.get("SPATIAL", []))

    if len(spatial) >= total:
        # SPATIAL 已占满（或超过）目标总量 → 四类统一按比例，避免独占
        types = [t for t in QUERY_TYPES if by_type.get(t)]
        alloc = _allocate(total, [len(by_type[t]) for t in types])
        subset = []
        for t, n in zip(types, alloc):
            pool = by_type[t]
            n = min(n, len(pool))
            subset.extend(pool if n == len(pool) else rng.sample(pool, n))
        return subset

    picked = {"SPATIAL": spatial}
    remaining = total - len(spatial)

    other_types = ["FACTUAL", "REASONING", "COMPARATIVE"]
    pools = [by_type.get(t, []) for t in other_types]
    alloc = _allocate(max(0, remaining), [len(p) for p in pools])

    for t, n, pool in zip(other_types, alloc, pools):
        n = min(n, len(pool))
        picked[t] = rng.sample(pool, n)

    subset = []
    for t in QUERY_TYPES:
        subset.extend(picked.get(t, []))
    return subset


def stratify_contested(questions, total, seed):
    """把统计功效放在**有争议的两类**上：COMPARATIVE / SPATIAL 全取，其余按比例补齐。

    动机（2026-09-14 定，开跑前冻结）：
      R1#3 / R2#10 点名攻击的正是 COMPARATIVE 与 SPATIAL 两类。按比例抽 300 题时
      COMPARATIVE 会从 101 掉到 24，95%CI 半宽从 ±3.1% 涨到 ±6.4%——而论文的
      Finding 4 恰恰是"这两类没有提升"这个**否定性**结论，n 一小就变成不可证伪。

    📌 2026-09-29 附注（**不改上文**，上文是开跑前冻结的原始理由）：
      论文 20260929 稿按导师意见把 §5.2 的五点发现压成了两点，上面说的这条否定性
      结论现在落在**发现二**（合并自原发现三、四）末句「对比型与空间型的绝对正确率
      仍是四类中最低的」。「Finding 4」是当时的编号，与改版后的编号不再对应。
      分层抽样的依据本身没变——它来自审稿意见而非结果，仍然有效。
      SPATIAL 在全量里本来就只有 34 题，任何方案下都是全取，不受影响。

      故改为：两类全取（101+34=135），再从 FACTUAL/REASONING 里按比例补足 total。
      这样两个被攻击的类保持与全量**完全一致**的统计强度，代价是 FACTUAL/REASONING
      的 CI 放宽——而那两类是方法领先幅度最大、结论最不脆弱的地方。

    ⚠️ 分配依据来自审稿意见本身，**不是来自结果**；seed 固定，开跑前冻结。
       论文中须披露这一抽样设计，且总体分要按总体真实比例**加权**报告
       （见 weighted_overall），否则 COMPARATIVE/SPATIAL 被过度表示会抬高总分。
    """
    comp = [q for q in questions if q.get("query_type") == "COMPARATIVE"]
    spat = [q for q in questions if q.get("query_type") == "SPATIAL"]
    n_contested = len(comp) + len(spat)
    if total < n_contested:
        raise SystemExit(
            f"[FULL-EVAL] ✗ contested 分层要求 --total >= 被争议两类之和（{n_contested}"
            f" = COMPARATIVE {len(comp)} + SPATIAL {len(spat)}），当前 --total {total}。\n"
            f"            题量更少请改用 --stratify proportional。")

    rng = random.Random(seed)
    rest = [q for q in questions
            if q.get("query_type") not in ("COMPARATIVE", "SPATIAL")]
    by_type = {}
    for q in rest:
        by_type.setdefault(q.get("query_type", "FACTUAL"), []).append(q)

    types = [t for t in QUERY_TYPES if by_type.get(t)]
    alloc = _allocate(total - n_contested, [len(by_type[t]) for t in types])
    picked = list(comp) + list(spat)
    for t, n in zip(types, alloc):
        pool = by_type[t]
        n = min(n, len(pool))
        picked.extend(pool if n == len(pool) else rng.sample(pool, n))
    return picked


def weighted_overall(results, pop_weights):
    """按留出集里各题型的**真实比例**加权求 overall_score。

    contested 分层下 COMPARATIVE/SPATIAL 全取、FACTUAL/REASONING 被抽稀，
    直接对样本取算术平均会把被过度表示的两类放大，得到的不是留出集的总体水平。
    加权后才是"若在完整留出集上评测"的无偏估计。

    权重分子只累加真正算到的题型（某类可能因异常题为空），故用 den 归一化。
    """
    by_type = {}
    for r in results.get("per_question_results", []):
        q = r.get("qa_metrics")
        if not q or "error" in r:
            continue
        by_type.setdefault(r.get("query_type", "FACTUAL"), []).append(
            0.5385 * q.get("correctness", 0.0) +
            0.3077 * q.get("faithfulness", 0.0) +
            0.1538 * q.get("completeness", 0.0))
    num = den = 0.0
    for t, w in pop_weights.items():
        vals = by_type.get(t)
        if vals:
            num += w * (sum(vals) / len(vals))
            den += w
    return round(num / den, 4) if den else None


def main():
    ap = argparse.ArgumentParser(description="D4 放大评估：按比例分层 + 断点续跑")
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", default="contested",
                    choices=["contested", "proportional"],
                    help="抽样方式。contested（默认）：COMPARATIVE/SPATIAL 全取、"
                         "其余按比例补齐——把统计功效放在审稿人点名的两类上；"
                         "proportional：四类统一按真实比例抽。"
                         "⚠ 两者不可混用同一个 --results-dir：题集不同，续跑会串。")
    ap.add_argument("--test-set", default="test_sets/test_set_independent.json")
    ap.add_argument("--model", default="qwen3.5-plus", help="生成模型")
    ap.add_argument("--judge-model", default="deepseek-flash",
                    help="裁判模型（独立于生成模型，回应 R1#4/R2#7）。传空串则退化为同模型裁判")
    ap.add_argument("--only", default=None, choices=QUERY_TYPES + [
        "direct_llm", "vector_rag", "graph_rag", "dual_path",
        # 对照组：dual_path 但图谱启用时文献给满（top-5/500）。默认四臂列表里**不含**它，
        # 只经 --only 单独跑，避免它混进标准的四臂总表。
        "dual_path_fullctx",
        # 消融：删掉图谱盲区门控（图谱永不关闭）。同样只在 --only 里单独跑。
        "dual_path_nogate"], help="只跑某一基线")
    ap.add_argument("--no-llm-judge", action="store_true")
    ap.add_argument("--results-dir", default="./evaluation/results")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发题数（不改变生成/裁判语义，只压缩墙钟时间；API 成本不变）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印预检（题源/题量/抽样构成/已有检查点/费用与墙钟估算），"
                         "不调用任何 API、不发起评测")
    ap.add_argument("--no-thinking", action="store_true",
                    help="关闭 qwen3.5-plus 的思考模式：实测单题 41s→2.4s、输出 token 2076→24。"
                         "⚠ 结果与带思考的旧归档不可直接对比，先跑 evaluation/compare_thinking.py 验证")
    args = ap.parse_args()

    # 思考模式必须在任何模型调用之前定好（model_router 在调用时读环境变量）
    if args.no_thinking:
        os.environ["QWEN_ENABLE_THINKING"] = "0"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(args.results_dir, exist_ok=True)

    # 0. 开跑前自检：全量评测是几十小时的事，任何一条在跑完之后才发现的错
    #    都等于白跑。宁可在这里直接报错退出。
    if args.total == 0:
        print("[FULL-EVAL] ✗ --total 0 会抽取 0 道题，等于空跑（注意：本脚本的 0 不是"
              "「全部」，与 run_eval.py 的 --questions 0 约定相反）。\n"
              "            要跑全部请显式给题量：--total 1157")
        sys.exit(2)

    # 1. 载入测试集并分层抽样
    ts = json.load(open(args.test_set, encoding="utf-8"))
    if not ts.get("questions"):
        print(f"[FULL-EVAL] ✗ {args.test_set} 里没有 questions 数组")
        sys.exit(2)

    # 题源自检：片段级证据召回的真值来自 gold_evidence_spans，缺了它这一项全 N/A。
    n_gold = sum(1 for q in ts["questions"] if q.get("gold_evidence_spans"))
    ids_all = [q.get("id") for q in ts["questions"]]
    print(f"[FULL-EVAL] 题源: {args.test_set}")
    print(f"            共 {len(ts['questions'])} 题，其中 {n_gold} 题带片段级证据真值"
          f"（gold_evidence_spans）")
    if n_gold == 0:
        print("[FULL-EVAL] ⚠⚠ 该题源**没有任何** gold_evidence_spans："
              "evidence_recall@5 会全部记 N/A。\n"
              "            旧题源（test_set_independent / test_set_expanded / "
              "test_set_20260904_lyf）都没有这个字段，\n"
              "            应改用 test_set_honest_test.json（留出集）或 "
              "test_set_honest_dev.json（调参用）。")
    if len(ids_all) != len(set(ids_all)):
        print(f"[FULL-EVAL] ✗ 题源有 {len(ids_all) - len(set(ids_all))} 个重复 id")
        sys.exit(2)

    # 留出集里各题型的真实比例 —— contested 分层下要用它给总分加权（见 weighted_overall）
    pop = Counter(q.get("query_type", "FACTUAL") for q in ts["questions"])
    n_pop = sum(pop.values()) or 1
    pop_weights = {t: pop[t] / n_pop for t in pop}

    if args.stratify == "contested":
        subset_qs = stratify_contested(ts["questions"], args.total, args.seed)
    else:
        subset_qs = stratify_proportional(ts["questions"], args.total, args.seed)
    subset = {"version": ts.get("version"), "questions": subset_qs}

    cnt = Counter(q.get("query_type", "?") for q in subset_qs)
    print(f"[FULL-EVAL] 分层方式: {args.stratify} | 共 {len(subset_qs)} 题, seed={args.seed}")
    for t in QUERY_TYPES:
        got, full = cnt.get(t, 0), pop.get(t, 0)
        tag = "（全取）" if got == full and full else (f"（全量 {full}）" if full else "")
        print(f"    {t}: {got} {tag}")
    if args.stratify == "contested":
        print("[FULL-EVAL] ⚠ contested 分层：COMPARATIVE/SPATIAL 被过度表示，"
              "汇总表的 overall_weighted 行是按留出集真实比例加权的无偏估计，"
              "论文报总体分请用这一行并说明抽样设计。")
        print(f"[FULL-EVAL]   权重: "
              + "  ".join(f"{t}:{pop_weights.get(t, 0):.3f}" for t in QUERY_TYPES))

    runner = BaselineRunner(model=args.model, results_dir=args.results_dir)
    ev = Evaluator(model=args.model, results_dir=args.results_dir,
                   judge_model=args.judge_model or None)
    thinking = "关" if args.no_thinking else "开"
    print(f"[FULL-EVAL] 生成模型: {args.model} | 裁判模型: {ev.judge_model}"
          f"{'  (独立裁判 ✓)' if ev.is_independent_judge else '  (⚠ 同模型裁判，仅供内部对照)'}")
    print(f"[FULL-EVAL] 思考模式: {thinking} | 并发 workers={args.workers}")
    if args.no_thinking:
        print("[FULL-EVAL] ⚠ 已关闭思考：结果与 results_archive_20260914/ 中带思考的旧结果"
              "不可直接对比，论文中须注明该配置。")

    # 2. 确定要跑的基线
    baselines = ["direct_llm", "vector_rag", "graph_rag", "dual_path"]
    if args.only:
        baselines = [args.only]

    # 检查点自检：已存在的 checkpoint 会让本次**静默跳过**已完成的题。
    # 换 --total/--seed/题源后旧 checkpoint 的 id 集不再匹配，混着跑出来的汇总没有意义。
    print(f"[FULL-EVAL] 结果目录: {args.results_dir}")
    for bk in baselines:
        ck = os.path.join(args.results_dir, f"progress_{bk}.json")
        if os.path.exists(ck):
            try:
                done = len(json.load(open(ck, encoding="utf-8"))
                           .get("per_question_results", []))
            except Exception:
                done = -1
            print(f"            ⚠ 已存在 {os.path.basename(ck)}"
                  f"（{done} 条记录）→ 本次会跳过这些题。\n"
                  f"              确认题源/题量/seed 与上次一致再续跑；"
                  f"要全新跑请换 --results-dir。")

    # 墙钟估算：实测四臂单题端到端延迟（含思考生成 + 独立裁判）分别约
    # 81/72/72/64s（direct/vector/graph/dual，2026-09-14 八题四臂冒烟）。
    # 取其中最慢的 81s 作为常数 → 对多数臂是保守高估。
    # 注意这是**空载**延迟：冒烟每臂只有 8 题、恰好一波并发，没测到持续并发下
    # API 侧排队导致的延迟膨胀。全量跑时真实墙钟可能是这里的 1.5~2 倍。
    LATENCY_S = 81.0
    est_h = len(subset_qs) * len(baselines) * LATENCY_S / max(1, args.workers) / 3600
    print(f"[FULL-EVAL] 预计墙钟 ≈ {est_h:.1f} 小时"
          f"（{len(subset_qs)} 题 × {len(baselines)} 基线 ÷ {args.workers} 并发 × 单题 {LATENCY_S:.0f}s）")
    # DeepSeek 侧实付：背靠背余额差实测 ¥0.035/题次（10 题 dual_path 得 ¥0.35；
    # 32 题冒烟再验证一次 ¥1.12，两处吻合）。裁判 prompt 只含 问题/标准答案/系统回答，
    # **不含检索上下文**，故四臂的裁判花费基本一致，这个数字是估计值而非上限。
    print(f"[FULL-EVAL] 预计裁判(DeepSeek)花费 ≈ ¥{len(subset_qs) * len(baselines) * 0.035:.0f}"
          f"（按实测 ¥0.035/题次 × {len(subset_qs)} 题 × {len(baselines)} 基线）")
    print("[FULL-EVAL] 提示：可随时 Ctrl+C 暂停，已完成的题不会重跑；"
          "恢复时原样重跑同一条命令即可。")

    # --dry-run：到此为止，一道题都不跑。用于开跑前核对题源/题量/抽样构成/费用，
    # 不必为了看告警文案而真发一次 API 请求（此前正是这么误启动过一次 50 题的跑）。
    if args.dry_run:
        print(f"\n[FULL-EVAL] --dry-run：仅预检，未调用任何 API，未发起评测。")
        print(f"[FULL-EVAL] 去掉 --dry-run 即按上述配置正式开跑。")
        return

    all_results = {}
    comparison_table = {}

    for bk in baselines:
        name = runner.BASELINE_NAMES[bk]
        ckpt = os.path.join(args.results_dir, f"progress_{bk}.json")
        print(f"\n{'='*60}")
        print(f"[FULL-EVAL] Baseline: {name}  (checkpoint={ckpt})")
        print(f"{'='*60}")

        results = ev.run_full_evaluation(
            subset, runner.get_runner(bk),
            use_llm_judge=not args.no_llm_judge,
            checkpoint_path=ckpt,
            workers=args.workers,
        )

        all_results[bk] = results

        # 单独落盘该基线结果
        out = os.path.join(args.results_dir,
                           f"baseline_{bk}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f"[FULL-EVAL] {name} 已保存: {out}")

        for metric, value in results["overall_scores"].items():
            comparison_table.setdefault(metric, {})[bk] = value
        # 证据召回实际算到的题数在 coverage 里（不是顶层指标），单独提上来供汇总表打印
        comparison_table.setdefault("evidence_recall_evaluated", {})[bk] = \
            results["overall_scores"].get("coverage", {}).get("evidence_recall_evaluated")
        # 本轮抛异常、未计入聚合的题数（= 需要重跑补上的题）。为 0 才算一次干净的全量跑。
        comparison_table.setdefault("errors", {})[bk] = \
            results["overall_scores"].get("coverage", {}).get("errors")
        # 按留出集真实比例加权的总分（contested 分层下必须报这个）
        comparison_table.setdefault("overall_weighted", {})[bk] = \
            weighted_overall(results, pop_weights)

        # 被 Ctrl+C 中断：本基线只覆盖了一部分题，继续往下跑其余基线没有意义，
        # 而且拿残缺结果去算 winner / 出对比表会误导。就地停住，已完成的题都在
        # checkpoint 里，重跑同一条命令即可从断点续上。
        if results.get("interrupted"):
            print(f"\n[FULL-EVAL] 已在 {name} 中断（Ctrl+C）。"
                  f"本基线的已完成进度保存在 {ckpt}。")
            print(f"[FULL-EVAL] 未跑到的基线："
                  f"{[runner.BASELINE_NAMES[b] for b in baselines[baselines.index(bk) + 1:]]}")
            print("[FULL-EVAL] 恢复：原样重跑同一条命令（同 --test-set/--total/--seed/"
                  "--results-dir）。不会重复已完成题。")
            return

    # 3. 汇总对比
    # winner 按**加权后**的总分判定：contested 分层下未加权的 overall_score 会被
    # 过度表示的 COMPARATIVE/SPATIAL 拉动，用它选 winner 等于用口径不对的数下结论。
    overall = {k: (comparison_table.get("overall_weighted", {}).get(k)
                   if comparison_table.get("overall_weighted", {}).get(k) is not None
                   else v["overall_scores"]["overall_score"])
               for k, v in all_results.items()}
    winner = max(overall, key=overall.get) if overall else None

    final = {
        "description": f"D4 放大评估（按比例分层，total={args.total}, seed={args.seed}）",
        "sample_size": len(subset_qs),
        "sample_composition": dict(cnt),
        "baselines": all_results,
        "comparison": comparison_table,
        "winner": winner,
        "winner_name": runner.BASELINE_NAMES.get(winner) if winner else None,
        "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "model": args.model,
        "judge_model": ev.judge_model,
        "independent_judge": ev.is_independent_judge,
        # 运行配置随结果落盘：换裁判/切并发/关思考后，光看分数分不清是哪套配置跑的
        "run_config": {
            "workers": args.workers,
            "enable_thinking": not args.no_thinking,
            "seed": args.seed,
            "test_set": args.test_set,
            # 抽样设计必须随结果落盘：contested 与 proportional 的分数口径不同，
            # 光看结果文件分不出是哪套抽的（R2#3 的划分纪律要求可追溯）
            "stratify": args.stratify,
            "population_weights": pop_weights,
        },
    }
    # --only 时 all_results 里只有一个臂：winner 必然是自己，"最终对比"名不副实。
    # 四个基线拆成四个并发进程时，若每个进程都写一份 baseline_full_*.json，
    # 事后很容易误把其中一份当成总表。故只跑单臂时不写这个文件。
    if args.only:
        print(f"\n{'='*60}")
        print(f"[FULL-EVAL] --only {args.only}：只跑了单个基线，不写 baseline_full_*.json"
              "（避免留下名不副实的『最终对比』）。")
        print(f"{'='*60}")
        for metric, val in all_results[args.only]["overall_scores"].items():
            print(f"  {metric}: {val}")
        return

    final_out = os.path.join(args.results_dir,
                             f"baseline_full_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(final, open(final_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[FULL-EVAL] 最终对比已保存: {final_out}")
    print(f"{'='*60}")
    # 2026-09-14：citation_precision/recall 与 retrieval_recall@5/mrr 已从总分移除
    # （前者三个 RAG 系统恒为 1.0——测试集只有一篇源文档，是白送常数；后者对
    # 图谱/纯LLM路不适用）。汇总表改为打印仍在总分内的三项 + 片段级证据召回。
    key_metrics = ["overall_weighted", "overall_score", "qa_correctness",
                   "qa_faithfulness", "qa_completeness", "evidence_recall@5",
                   "evidence_recall_evaluated", "errors", "avg_latency_ms"]
    header = "  metric".ljust(28) + "".join(bk.ljust(14) for bk in baselines)
    print(header)
    for metric in key_metrics:
        row = f"  {metric}".ljust(28)
        for bk in baselines:
            v = comparison_table.get(metric, {}).get(bk, 0)
            row += ("N/A" if v is None else str(v)).ljust(14)
        print(row)
    print("\n  ⚠ overall_weighted = 按留出集真实题型比例加权的总分"
          "（contested 分层下报这个）；overall_score 是样本内算术平均，仅对照。")
    print("  ⚠ evidence_recall@5 只在「有逐字证据真值」的题上计算，"
          "实际算到的题数见 evidence_recall_evaluated 行。")
    print("  ⚠ errors 是抛异常、未计入任何聚合的题数。非 0 就原样重跑同一条命令补上"
          "（这些题没有 qa_metrics，续跑时会自动重做）。")
    print(f"\n  winner: {runner.BASELINE_NAMES.get(winner)} ({winner})")


if __name__ == "__main__":
    main()
