#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给测试集标注 `evidence_span_in_corpus`——判断该题的证据片段能否用作**检索真值**。

为什么要这一步（2026-09-14 实测）：
  `evidence_span` 是专家写的（provenance=expert_written），**不保证是原文**。
  实测 dev 150 题里：
    - 66/150（44%）的 evidence_span **逐字存在于向量库语料**中 → 可作检索真值
    - 其余 84 题是专家改写/概括 → 语料里根本没有这段文字，
      拿它当"检索该命中哪个片段"的真值会把**真值本身不存在**记成检索失败。
  所以必须先区分这两类：只有前一类的题才能进入片段级检索指标，后一类记为 N/A。

  这一步纯离线、不调任何模型，可重复执行。

用法:
    python scripts/annotate_evidence_spans.py                    # 标注 test_set_honest.json
    python scripts/annotate_evidence_spans.py --dry-run          # 只看统计不落盘
"""
import sys, os, re, json, argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)


def norm(s):
    """只去空白。不做标点归一——实测这样已能稳定命中逐字出现的片段，
    而激进归一化会把"改写"误判成"原文"。"""
    return re.sub(r"\s+", "", str(s or ""))


def main():
    ap = argparse.ArgumentParser(description="标注 evidence_span 是否逐字存在于语料")
    ap.add_argument("--test-set",
                    default=os.path.join(PKG_DIR, "test_sets", "test_set_honest.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import rag_core
    corpus = [norm(getattr(d, "page_content", ""))
              for d in rag_core.vectordb._documents]
    print(f"语料 chunk 数: {len(corpus)}")

    data = json.load(open(args.test_set, encoding="utf-8"))
    qs = data["questions"]

    n_with = n_gold = n_whole = n_split = 0
    by_type = {}
    type_total = {}
    for q in qs:
        t = q.get("query_type", "?")
        type_total[t] = type_total.get(t, 0) + 1
        es = q.get("evidence_span")
        if not es or str(es).strip() in ("", "None"):
            q["gold_evidence_spans"] = []
            continue
        n_with += 1
        ne = norm(es)
        if any(ne in c for c in corpus):
            gold = [str(es)]                       # 整段即原文，最干净
            n_whole += 1
        else:
            # 对比题的 evidence_span 是"甲…；乙…"这种双侧概括，整段必不逐字。
            # 按分号拆开后，各子句可能分别逐字存在于语料 → 取逐字的那些作真值。
            gold = [p for p in re.split(r"[；;]", str(es))
                    if len(norm(p)) >= 8 and any(norm(p) in c for c in corpus)]
            if len(gold) >= 2:
                n_split += 1
        q["gold_evidence_spans"] = gold
        if gold:
            n_gold += 1
            by_type[t] = by_type.get(t, 0) + 1

    print(f"有 evidence_span 的题: {n_with}/{len(qs)}")
    print(f"  整段逐字命中语料:        {n_whole}")
    print(f"  整段不中但拆子句后有≥2段逐字命中: {n_split}")
    print(f"可作片段级真值的题合计: {n_gold} ({n_gold / max(1, n_with) * 100:.1f}%)")
    print(f"其余 {n_with - n_gold} 题为专家改写 → 片段级指标记 N/A，不能记 0")
    print(f"\n{'题型':<14}{'可真值':>8}{'总题':>8}{'覆盖':>8}")
    for t in sorted(type_total):
        g = by_type.get(t, 0)
        print(f"{t:<14}{g:>8}{type_total[t]:>8}{g / type_total[t] * 100:>7.1f}%")

    if args.dry_run:
        print("\n[dry-run] 未落盘")
        return

    data["evidence_span_annotation"] = {
        "method": "去空白后判断 evidence_span 是否逐字出现在任一语料 chunk 中；"
                  "不中则按分号拆子句、取逐字的子句作真值",
        "corpus_chunks": len(corpus),
        "n_with_span": n_with,
        "n_whole_verbatim": n_whole,
        "n_split": n_split,
        "n_with_gold": n_gold,
        "coverage_by_type": {t: by_type.get(t, 0) for t in sorted(type_total)},
        "type_total": type_total,
        "note": "gold_evidence_spans 为空的题不参与片段级检索/证据召回指标（记 N/A）。"
                "⚠ COMPARATIVE 覆盖为 0：该类 evidence_span 均为专家撰写的双侧概括，"
                "语料中无逐字对应，故无法评估其检索真值。语料变更后需重跑本脚本。",
    }
    json.dump(data, open(args.test_set, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写回: {args.test_set}")
    print("★ 接着重跑 scripts/split_dev_test.py 让 dev/test 同步带上该字段。")


if __name__ == "__main__":
    main()
