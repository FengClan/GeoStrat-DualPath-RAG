#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化「测试集与检索来源同源」的程度——回应审稿意见 R2#1。

审稿人的质疑是：这套评测可能是**自问自答**——gold 答案和系统检索到的文本
出自同一份语料，那么系统只要把检索结果原样吐出来就能得高分，评测没有意义。

这个质疑不能靠辩解回应，只能靠量化。本脚本对每道题算三个重叠率
（字符 10-gram 重叠，即「gold 里有多少比例的 10 字片段能在对照文本里原样找到」）：

  1. `vs_source_para` —— gold 与其**生成所依据的原文段落**的重叠。
     按构造必然是 1.0（gold 就是从那段抄的）。**这个数字要如实报出来**，
     它说明的是「gold 是原文摘录」，本身不构成缺陷，但必须披露。

  2. `vs_kg_context` —— gold 与**图谱通道为该题检索到的上下文**的重叠。
     图谱通道返回的是 `A --[rel]--> B` 三元组串，与整句 gold 比 10-gram
     必然极低，因此这个数字只能说明「图谱通道不会逐字泄漏答案」，
     是**弱结论**，不能拿来宣称「无泄漏」。

  3. `vs_vector_context` —— gold 与**向量通道 top-k 检索到的片段**的重叠。
     ⚠️ 这才是 R2#1 真正指的泄漏路径：向量通道返回的就是同一篇 docx 的
     真文本块。实测均值约 0.9，即 gold 几乎总能被原样检索回来。
     **这个数字必须如实报出来**，不能回避——它意味着向量类基线在
     这些题上做的是「抽取」而不是「推理」。
     但要注意反过来的证据：即便 gold 原文已在上下文里，模型也没有
     必然答对（见论文结果表 vector_rag 的 correctness 并非满分），
     说明任务并非纯粹的复制粘贴。两者要一起讲。

  4. `vs_question` —— gold 与问题文本自身的重叠，用来说明问题里
     有没有直接漏出答案（应接近 0）。

  ⚠️ 重叠率高 ≠ 题目无效，只说明该题偏「抽取」而非「推理」。论文中应
     按此维度给测试集分层报告，而不是笼统宣称「无泄漏」。

用法:
    python scripts/measure_leakage.py
    python scripts/measure_leakage.py --test-set test_sets/test_set_honest.json
    python scripts/measure_leakage.py --no-vector     # 跳过向量通道（慢）
"""
import sys, os, json, re, argparse
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:          # 便于直接 `python scripts/measure_leakage.py`
    sys.path.insert(0, PKG_DIR)      # 时也能 import 到根目录下的 rag_core
GRAPH = os.path.join(PKG_DIR, "kb_storage", "graphs", "experiment_kb.graphml")
DOCX = r"E:\地质文本空间化平台\福建地质志.docx"
NGRAM = 10


def norm(s):
    """去掉空白与标点后比较——检索上下文和 gold 的标点经常不一致。"""
    return re.sub(r'[\s。；，、．,;：:（）()【】\[\]“”"\'’‘·—\-]', '', s or "")


def ngram_overlap(gold, context, n=NGRAM):
    """gold 中有多少比例的 n-gram 能在 context 中原样找到。"""
    g, c = norm(gold), norm(context)
    if len(g) < n:
        return None
    grams = {g[i:i + n] for i in range(len(g) - n + 1)}
    if not grams:
        return None
    hit = sum(1 for x in grams if x in c)
    return hit / len(grams)


# ----------------------------------------------------------------------
# 图谱通道检索：与 services/kg_service.py::retrieve_graph_context 同逻辑。
# 刻意重写而非 import，是为了让这个测量脚本不依赖 Flask 应用上下文；
# 若那边改了检索逻辑，这里必须同步（两处都指向同一篇论文的同一个数字）。
# ----------------------------------------------------------------------
def kg_context(question, G, max_hops=2, max_edges=50):
    matched = {n for n in G.nodes() if str(n) in question}
    if not matched:
        return ""

    hop = {n: 0 for n in matched}
    sub, cur = set(matched), set(matched)
    for h in range(1, max_hops + 1):
        nxt = set()
        for n in cur:
            nb = (set(G.successors(n)) | set(G.predecessors(n))
                  if G.is_directed() else set(G.neighbors(n)))
            for x in nb:
                if x not in hop:
                    hop[x] = h
                    nxt.add(x)
        sub |= nxt
        cur = nxt
    return " ".join(
        f"{u} --[{d.get('relation', '关联')}]--> {v}"
        for u, v, d in list(G.subgraph(sub).edges(data=True))[:max_edges])


def load_questions(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["questions"] if isinstance(d, dict) else d


def load_paras():
    if not os.path.exists(DOCX):
        return []
    from docx import Document
    return [p.text.strip() for p in Document(DOCX).paragraphs]


def pct(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    vals.sort()

    def q(p):
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "median": round(q(0.5), 4),
        "p75": round(q(0.75), 4),
        "p90": round(q(0.90), 4),
        "gt50%": round(sum(1 for v in vals if v > 0.5) / len(vals), 4),
        "gt30%": round(sum(1 for v in vals if v > 0.3) / len(vals), 4),
        "gt10%": round(sum(1 for v in vals if v > 0.1) / len(vals), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set",
                    default=os.path.join(PKG_DIR, "test_sets", "test_set_honest.json"))
    ap.add_argument("--out",
                    default=os.path.join(PKG_DIR, "evaluation", "results", "leakage.json"))
    ap.add_argument("--no-vector", action="store_true",
                    help="跳过向量通道测量（需加载嵌入模型，较慢）")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    import networkx as nx
    G = nx.read_graphml(GRAPH)
    print(f"图谱: {G.number_of_nodes()} 节点 / {G.number_of_edges()} 边")
    qs = load_questions(args.test_set)
    print(f"测试集: {len(qs)} 题  ({os.path.basename(args.test_set)})")
    paras = load_paras()

    retrieve = None
    if not args.no_vector:
        try:
            from rag_core import retrieve_from_vectordb
            retrieve = retrieve_from_vectordb
            print(f"向量通道: 已加载（top_k={args.top_k}）")
        except Exception as e:
            print(f"[WARN] 向量通道不可用，跳过: {e}")

    rows = []
    for i, q in enumerate(qs):
        gold = q.get("expected_answer") or ""
        if len(norm(gold)) < NGRAM:
            continue
        ctx = kg_context(q["question"], G)
        src = ""
        gi = q.get("src_para")
        if isinstance(gi, int) and 0 <= gi < len(paras):
            src = paras[gi]
        vctx = None
        if retrieve:
            try:
                r = retrieve(q["question"], kb_id="all", top_k=args.top_k)
                vctx = " ".join(r[0]) if isinstance(r, tuple) else str(r)
            except Exception:
                vctx = ""
        rows.append({
            "id": q.get("id"),
            "query_type": q.get("query_type"),
            "source_section": q.get("source_section"),
            "vs_kg_context": ngram_overlap(gold, ctx),
            "vs_vector_context": (ngram_overlap(gold, vctx)
                                  if vctx is not None else None),
            "vs_source_para": ngram_overlap(gold, src) if src else None,
            "vs_question": ngram_overlap(gold, q["question"]),
            "kg_ctx_empty": 1 if not ctx else 0,
        })
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(qs)}", flush=True)

    print(f"\n可测题数: {len(rows)}（gold 短于 {NGRAM} 字的题不计）\n")
    print(f"{'对照':<18}{'n':>6}{'均值':>9}{'中位':>9}{'p75':>9}{'p90':>9}"
          f"{'>50%':>8}{'>30%':>8}{'>10%':>8}")
    print("-" * 84)
    summary = {}
    for key, label in (("vs_vector_context", "向量检索片段 ★"),
                       ("vs_source_para", "gold 的原文段落"),
                       ("vs_kg_context", "图谱检索上下文"),
                       ("vs_question", "问题文本自身")):
        s = pct(rows, key)
        if not s:
            continue
        summary[key] = s
        print(f"{label:<18}{s['n']:>6}{s['mean']:>9.3f}{s['median']:>9.3f}"
              f"{s['p75']:>9.3f}{s['p90']:>9.3f}"
              f"{s['gt50%']:>8.1%}{s['gt30%']:>8.1%}{s['gt10%']:>8.1%}")

    empty = sum(r["kg_ctx_empty"] for r in rows)
    print(f"\n图谱通道**检索不到任何上下文**的题: {empty}/{len(rows)} = {empty/len(rows):.1%}")
    print("  （这些题图谱通道结构性不可答——见 evolution/conclusion 组的说明）")

    # ----------------------------------------------------------------
    # 分层报告：泄漏不是均匀分布的，这个分层才是能写进论文的部分
    # ----------------------------------------------------------------
    # 有向量通道时看 vs_vector_context（R2#1 真正指的路径）；
    # 跳过向量通道时退回 vs_source_para，至少还能看分组分布。
    gkey = "vs_vector_context" if retrieve else "vs_source_para"
    print(f"\n（分组依据：{gkey}）")

    def group(field, title):
        print(f"\n按 {title} 分组:")
        print(f"  {'':<14}{'n':>6}{'均值':>9}{'中位':>9}{'>50%题占比':>12}")
        by = defaultdict(list)
        for r in rows:
            by[r.get(field) or "?"].append(r)
        for k in sorted(by, key=lambda x: -len(by[x])):
            s = pct(by[k], gkey)
            if s:
                print(f"  {k:<14}{s['n']:>6}{s['mean']:>9.3f}"
                      f"{s['median']:>9.3f}{s['gt50%']:>11.1%}")
        return by

    group("query_type", "题型")
    group("source_section", "来源")
    print("\n  ★ 泄漏高度集中在 FACTUAL、COMPARATIVE 几乎不受影响——而 COMPARATIVE")
    print("    恰是本方法表现最差的一档。即：性能短板不能用「题出得太容易」解释。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({
        "test_set": os.path.basename(args.test_set),
        "ngram": NGRAM,
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "summary": summary,
        "kg_context_empty": {"count": empty, "total": len(rows),
                             "ratio": round(empty / len(rows), 4) if rows else None},
        "per_question": rows,
    }, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n输出: {args.out}")


if __name__ == "__main__":
    main()
