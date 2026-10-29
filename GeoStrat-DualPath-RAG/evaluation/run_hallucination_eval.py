#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""幻觉三分类统计 —— 回应 R2#16。

审稿人要求：显式量化幻觉，区分 **无支撑(unsupported) / 矛盾(contradicted) / 编造(fabricated)**。

## 为什么不改 evaluator.py

主实验（`multirun_baseline_20260917.json`）已经定稿，其 correctness/faithfulness 由
`evaluator._llm_judge_eval` 产出。若在该函数的 prompt 里加维度，裁判的输出分布会变，
**已定稿的分数就全部作废**。故本脚本**完全独立**：另起 prompt、另存结果，不碰 evaluator，
不影响任何既有分数。

## 为什么不重跑生成

checkpoint 里已经存了 `answer`（生成结果）与 `expected_answer`（gold）。
本脚本直接读它们重新判，**零生成成本**，只花裁判钱。

## ⚠️ 关键：判「编造」必须给依据，否则指标是假的

第一版 rubric 把 fabricated 定义成"标准答案里没有的具体细节"。**这是错的**，实测跑出来
两个臂的 fabricated 都接近 50%：因为系统答出的具体细节（地名、剖面名）往往是**从检索到的
原文里来的**——那是有据（grounded），不是编造。只看 gold 会把"比 gold 更详细"一律算成幻觉。

所以本版**本地重放检索**（向量通道 + 图谱通道，纯 embedding 与图遍历，**不调任何生成模型**），
把该题实际能看到的材料作为「依据」一并交给裁判：

- 具体细节能在 **gold 或 依据** 里找到 → supported（有据）
- 与 gold 明确冲突 → contradicted
- 两处都找不到、却被当作确定事实说出的具体可核查细节 → fabricated（编造）
- 既不错也无据的泛泛表述 → unsupported

对 `direct_llm` 这类无检索的臂，依据为空串——它本来就只能靠参数记忆，这个对照才有意义。

## 三类的操作化定义（写进 prompt，避免裁判自由发挥）

- **contradicted（矛盾）**：回答中的地质陈述与标准答案**明确冲突**（标准答案说砂岩，回答说花岗岩）
- **fabricated（编造）**：把**具体的、可核查的细节**当作确定事实说出，而这些细节在标准答案、
  问题、依据中**都不存在**（具体数值、具体地名、具体文献编号、具体化石名）
- **unsupported（无支撑）**：既不错也无据的泛泛表述，无法核实也无从否定
- **supported（有据）**：能在标准答案或依据中找到支撑的陈述

用法:
    python evaluation/run_hallucination_eval.py --total 100 --arms dual_path,direct_llm
    python evaluation/run_hallucination_eval.py --total 100 --checkpoints <目录> --round 1
    python evaluation/run_hallucination_eval.py --total 100 --no-evidence   # 旧版（不推荐，见上）
"""
import sys, os, json, argparse, time, random, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "results", "multirun_baseline_20260917")
ARM_LABEL = {
    "direct_llm": "纯 LLM",
    "vector_rag": "向量 RAG",
    "graph_rag":  "图谱 RAG",
    "dual_path":  "双路融合（提出方法）",
}
# 有检索的臂才重放依据；纯 LLM 没有依据可给（它本就不检索）
ARM_HAS_RETRIEVAL = {"vector_rag", "graph_rag", "dual_path"}

# 依据截断上限：够裁判核对具体细节即可，太长只会推高思考量
EVIDENCE_CHARS = 3000

# ⚠️ 这个 prompt 的形状是调过的，别随手改回"把回答拆成若干条陈述"那种写法。
# deepseek-flash 是思考模型，且 reasoning_content 与正文**共享** max_tokens。
# 早期版本要求"先把系统回答拆成若干条可独立判断的地质陈述"，这个任务**没有上界**——
# 答案一长（>1500 字）思考链就写到 8000+ token 仍未收尾，finish_reason=length、
# content 为空串，表现为 json 解析报 "Expecting value: line 1 column 1 (char 0)"，
# 冒烟测试 8/8 全灭。实测：改成下面的**有上界**写法（限定 ≤12 条、examples 每类 ≤1、
# 明令不写分析过程）后，思考量从 8192+ 降到 5068，中等长度答案直接恢复可用。
PROMPT = """你是地质学领域的严格评审。判断系统回答里的**幻觉类型**，只统计、不展开分析。

## 问题
{question}

## 标准答案（Gold）
{gold}

## 依据（系统回答时能看到的检索材料；为空表示该系统没有检索，只能凭记忆作答）
{evidence}

## 系统回答（待判）
{answer}

## 四类定义
1. **supported（有据）**：能在**标准答案或依据**中找到支撑。
2. **contradicted（矛盾）**：与标准答案**明确冲突**（标准答案说砂岩，回答说花岗岩）。
3. **fabricated（编造）**：把**具体、可核查的细节**（厚度数值、年龄、地名、剖面名、
   化石名、文献编号）当作确定事实说出，而这些细节在标准答案、问题、依据中**都找不到**。
   **无中生有地编出细节才算**。
4. **unsupported（无支撑）**：既不错也无据的泛泛表述，无法核实也无从否定。

## 硬性要求
1. **不要写出分析、推理或逐条讨论的过程，直接给结论。**
2. **最多只统计 12 条陈述**；回答更长时，只取其中最实质的 12 条。
3. 若某具体细节能在「依据」里找到，**即使标准答案里没有，也算 supported**，不算编造。
   （比标准答案更详细 ≠ 幻觉。）
4. 回答若明确说"未找到相关信息"，不算幻觉，全部计入 supported。
5. 一条陈述只归一类；同一句里既有错又有编造，取更严重的
   （contradicted > fabricated > unsupported）。
6. **不要**因为表述冗长或不够完整就判为幻觉——那是 completeness 的问题，不属于本任务。
7. examples 每类**最多 1 条**原句。

## 只输出如下 JSON（不要 markdown 围栏，不要任何其他文字）
{{"n_supported":0,"n_unsupported":0,"n_contradicted":0,"n_fabricated":0,
 "examples":{{"contradicted":[],"fabricated":[],"unsupported":[]}},
 "reason":"不超过20字"}}"""


def make_prompt(q, gold, ans, evidence=""):
    return PROMPT.format(question=q, gold=gold, answer=ans, evidence=evidence or "（无）")


def judge_one(model, q, gold, ans, evidence="", retries=2, max_tokens=32768):
    """返回 (counts_dict, raw_ok, err)

    ⚠️ max_tokens 必须显式给足。deepseek-flash 是思考模型，reasoning_content 与
    正文**共享** completion 预算；默认 4096 下，答案一长（>1500 字）思考链就把预算
    烧光，finish_reason=length、content 为空串——表现为 json 解析
    "Expecting value: line 1 column 1 (char 0)"，冒烟测试 8/8 全灭。
    实测答案最长 3338 字（direct_llm）约需 19k，故默认 32768（只按实际用量计费，
    给足预算不会变贵；实测给到 32768 时模型反而更早自停，用量低于 16384）。
    """
    from services.model_router import get_model_response
    last = None
    for attempt in range(retries):
        # 第一次用默认预算；若因预算耗尽返回空串，翻倍再试一次
        mt = max_tokens * (2 ** attempt)
        try:
            raw = get_model_response(make_prompt(q, gold, ans, evidence), model, None,
                                     {"temperature": 0.0, "max_tokens": mt})
            if not (raw or "").strip():
                last = RuntimeError(
                    f"裁判返回空串（completion 预算耗尽，max_tokens={mt}）")
                continue
            c = raw.strip()
            if c.startswith("```json"):
                c = c[7:]
            if c.startswith("```"):
                c = c[3:]
            if c.endswith("```"):
                c = c[:-3]
            d = json.loads(c)
            return {
                "n_supported":   int(d.get("n_supported", 0)),
                "n_unsupported": int(d.get("n_unsupported", 0)),
                "n_contradicted": int(d.get("n_contradicted", 0)),
                "n_fabricated":  int(d.get("n_fabricated", 0)),
                "examples": d.get("examples", {}),
                "reason": d.get("reason", ""),
                "judge_failed": False,
            }, True, None
        except Exception as e:
            last = e
    return {"judge_failed": True, "reason": f"parse failed: {last}"}, False, last


def load_arm(ckpt_dir, arm, rnd):
    p = os.path.join(ckpt_dir, f"baseline__{arm}_r{rnd}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for x in d["per_question_results"]:
        if "qa_metrics" not in x or not x.get("answer"):
            continue
        out[x["id"]] = {
            "question": x.get("question", ""),
            "gold": x.get("expected_answer", ""),
            "answer": x.get("answer", ""),
            "query_type": x.get("query_type", "?"),
        }
    return out


def build_evidence(questions, kb_id="all"):
    """本地重放检索，拿到每题实际能看到的依据。**不调任何生成模型。**

    与 `baselines._dual_path_impl` 的取法逐条对齐：同样的分类器 max_hops、
    同样的图谱覆盖门控 `_graph_is_blind`、同样的 top-5/500 与 top-3/400 分流。
    这样裁判看到的依据 = 当时模型看到的依据。

    串行执行：embedding 模型与 FAISS 索引未必线程安全，而这一步很快
    （每题约 0.2s），没必要冒并发风险。
    """
    from evaluation.baselines import BaselineRunner, _graph_is_blind
    from services.kg_service import retrieve_graph_context
    from services.query_classifier import get_retrieval_config

    runner = BaselineRunner(model="unused", kb_id=kb_id)
    out = {}
    t0 = time.time()
    for i, qd in enumerate(questions):
        q = qd["question"]
        try:
            max_hops = get_retrieval_config(q).get("max_hops", 2)
        except Exception:
            max_hops = 2
        v_raw, kg_facts = [], ""
        try:
            v_raw = runner._retrieve_docs(q, kb_id, top_k=5) or []
        except Exception as e:
            print(f"    [依据] {qd['id']} 向量检索失败: {e}")
        try:
            r = retrieve_graph_context(q, kb_id, max_hops=max_hops)
            kg_edges = (r[0] if isinstance(r, tuple) and r
                        and isinstance(r[0], list) else (r if isinstance(r, list) else []))
            kg_facts = runner._format_kg_facts(kg_edges) or ""
        except Exception as e:
            print(f"    [依据] {qd['id']} 图谱检索失败: {e}")

        blind = _graph_is_blind(q)
        if blind:
            kg_facts = ""
            v_docs, v_chars = v_raw[:5], 500
        else:
            v_docs, v_chars = v_raw[:3], 400
        v_ctx = "\n\n".join(f"[文献 {k+1}] {d.page_content[:v_chars]}"
                            for k, d in enumerate(v_docs))
        ev = (kg_facts + "\n\n" + v_ctx).strip()
        out[qd["id"]] = ev[:EVIDENCE_CHARS]
        if (i + 1) % 25 == 0:
            print(f"    依据重放 {i+1}/{len(questions)}  ({time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description="幻觉三分类统计（R2#16）")
    ap.add_argument("--checkpoints", default=DEFAULT_CKPT)
    ap.add_argument("--round", type=int, default=1, help="用哪一轮的答案")
    ap.add_argument("--arms", default="dual_path,direct_llm")
    ap.add_argument("--total", type=int, default=100, help="分层抽样题数（0=全量）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-model", default="deepseek-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="裁判单次 completion 上限；思考链与正文共享此预算，给少了会返回空串")
    ap.add_argument("--no-evidence", action="store_true",
                    help="不重放检索依据（旧版 rubrics；会把'比 gold 详细'误判成编造，不推荐）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    data = {}
    for a in arms:
        d = load_arm(args.checkpoints, a, args.round)
        if d is None:
            print(f"⚠️  找不到 {a} r{args.round}，跳过")
            continue
        data[a] = d
    if not data:
        print("没有任何可用的 arm。")
        return 1

    # 用双路融合的题集为基准做分层抽样，保证各臂判的是同一批题
    ref = data.get("dual_path") or data[arms[0]]
    ids_all = sorted(ref.keys())
    if args.total and args.total < len(ids_all):
        by_t = defaultdict(list)
        for i in ids_all:
            by_t[ref[i]["query_type"]].append(i)
        rng = random.Random(args.seed)
        ids = []
        for t in sorted(by_t):
            ids.extend(rng.sample(by_t[t], max(1, round(len(by_t[t]) * args.total / len(ids_all)))))
        ids = sorted(set(ids))[:args.total]
    else:
        ids = ids_all

    print("=" * 78)
    print(f"幻觉三分类统计（R2#16）  题数 {len(ids)}  seed={args.seed}  裁判={args.judge_model}")
    print(f"  题源 {args.checkpoints}  第 {args.round} 轮")
    print(f"  依据={'本地重放检索（不调生成模型）' if not args.no_evidence else '关闭（旧版 rubric）'}")
    print("  题型: " + "  ".join(f"{t}:{sum(1 for i in ids if ref[i]['query_type'] == t)}"
                                 for t in sorted({ref[i]["query_type"] for i in ids})))
    for a in data:
        print(f"  arm {a:<12} 可用 {len(data[a])} 题")
    print("=" * 78)

    # --- 依据重放（有检索的臂才需要）---
    evidence = {}
    if not args.no_evidence:
        need = [a for a in data if a in ARM_HAS_RETRIEVAL]
        if need:
            print(f"\n[依据重放] 为 {need} 重放检索上下文（纯本地，不发 API）…")
            evidence = build_evidence([{"id": i, "question": ref[i]["question"]} for i in ids])
            nonempty = sum(1 for v in evidence.values() if v)
            print(f"[依据重放] 完成：{nonempty}/{len(ids)} 题拿到非空依据")

    from services.model_router import usage_stats
    usage_stats(reset=True)

    report = {"config": vars(args), "n_questions": len(ids),
              "evidence_mode": "none" if args.no_evidence else "replayed_retrieval",
              "arms": {}}
    lock = threading.Lock()

    for a in data:
        print(f"\n{'-' * 78}\n[arm {a} — {ARM_LABEL.get(a, a)}]")
        todo = [i for i in ids if i in data[a]]
        use_ev = (not args.no_evidence) and (a in ARM_HAS_RETRIEVAL)
        results = {}
        done = [0]
        t0 = time.time()

        def work(i):
            r = data[a][i]
            c, ok, err = judge_one(args.judge_model, r["question"], r["gold"],
                                   r["answer"],
                                   evidence=evidence.get(i, "") if use_ev else "",
                                   max_tokens=args.max_tokens)
            with lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"    {done[0]}/{len(todo)}  ({time.time()-t0:.0f}s)")
            return i, c, ok

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, i) for i in todo]
            for f in as_completed(futs):
                i, c, ok = f.result()
                results[i] = c

        # 汇总
        tot = Counter()
        failed = 0
        per_type = defaultdict(Counter)
        for i in todo:
            c = results[i]
            if c.get("judge_failed"):
                failed += 1
                continue
            for k in ("n_supported", "n_unsupported", "n_contradicted", "n_fabricated"):
                tot[k] += c.get(k, 0)
            for k in ("n_unsupported", "n_contradicted", "n_fabricated"):
                per_type[ref[i]["query_type"]][k] += c.get(k, 0)

        judged = len(todo) - failed
        claims = tot["n_supported"] + tot["n_unsupported"] + tot["n_contradicted"] + tot["n_fabricated"]
        report["arms"][a] = {
            "label": ARM_LABEL.get(a, a),
            "has_retrieval": a in ARM_HAS_RETRIEVAL,
            "n_judged": judged, "n_failed": failed,
            "total_claims": claims,
            "by_kind": dict(tot),
            "rate_per_answer": {
                k: round(tot[k] / judged, 3) for k in
                ("n_unsupported", "n_contradicted", "n_fabricated")} if judged else {},
            "share_of_claims": {
                k: round(tot[k] / claims, 4) for k in
                ("n_supported", "n_unsupported", "n_contradicted", "n_fabricated")
            } if claims else {},
            "by_query_type": {t: dict(c) for t, c in per_type.items()},
            "per_question": results,
        }

        r = report["arms"][a]
        print(f"  判定 {judged} 题（失败 {failed}），共拆出 {claims} 条陈述")
        if claims:
            print(f"    有据 supported      {tot['n_supported']:>5}  ({tot['n_supported']/claims*100:5.1f}%)")
            print(f"    无支撑 unsupported  {tot['n_unsupported']:>5}  ({tot['n_unsupported']/claims*100:5.1f}%)")
            print(f"    矛盾 contradicted   {tot['n_contradicted']:>5}  ({tot['n_contradicted']/claims*100:5.1f}%)")
            print(f"    编造 fabricated     {tot['n_fabricated']:>5}  ({tot['n_fabricated']/claims*100:5.1f}%)")
            print(f"    每题平均幻觉条数    {(claims-tot['n_supported'])/judged:.2f}")

    # 对比表
    if len(report["arms"]) > 1:
        print(f"\n{'=' * 78}\n[各臂对比]  每题平均条数\n{'=' * 78}")
        ks = ("n_supported", "n_unsupported", "n_contradicted", "n_fabricated")
        print(f"  {'臂':<22}" + "".join(f"{k.replace('n_',''):>14}" for k in ks))
        for a, r in report["arms"].items():
            if not r["n_judged"]:
                continue
            print(f"  {r['label']:<22}" +
                  "".join(f"{r['by_kind'][k]/r['n_judged']:>14.3f}" for k in ks))

    usage = usage_stats(reset=False)
    report["usage"] = usage
    print(f"\n裁判用量: {json.dumps(usage, ensure_ascii=False)}")

    out = args.out or os.path.join(
        os.path.dirname(__file__), "results",
        f"hallucination_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
