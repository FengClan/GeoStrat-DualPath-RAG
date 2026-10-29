#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D3 增强：给抽样清单每条三元组附上《福建地质志》原文上下文段落

- 复用 sample_for_human_review.py 完全相同的抽样（seed=42、按占比分层），保证 100 条一致
- 从 extraction_progress.jsonl 回溯每条三元组所属 chunk，定位 evidence/主体实体所在段落，
  附上该段 ± 2 段作为「原文上下文」，便于查证时不必翻 113MB 原文
- 输出 : kb_storage/d3_human_review_with_context.csv
"""
import sys, os, json, csv, random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLES = os.path.join(PKG_DIR, "kb_storage", "experiment_triples_llm.json")
PROGRESS = os.path.join(PKG_DIR, "kb_storage", "extraction_progress.jsonl")
OUT = os.path.join(PKG_DIR, "kb_storage", "d3_human_review_with_context.csv")


def stratified_sample(triples, target=100, seed=42):
    """与 sample_for_human_review.py 一致的分层抽样"""
    by_rel = {}
    for t in triples:
        by_rel.setdefault(t["relation"], []).append(t)
    total = len(triples)
    counts = {rel: max(1, round(target * len(items) / total)) for rel, items in by_rel.items()}
    rels_sorted = sorted(by_rel, key=lambda r: len(by_rel[r]), reverse=True)
    diff = target - sum(counts.values())
    i = 0
    while diff != 0:
        rel = rels_sorted[i % len(rels_sorted)]
        if diff > 0:
            counts[rel] += 1
            diff -= 1
        elif counts[rel] > 1:
            counts[rel] -= 1
            diff += 1
        i += 1
    random.seed(seed)
    sample = []
    for rel, n in counts.items():
        sample.extend(random.sample(by_rel[rel], min(n, len(by_rel[rel]))))
    random.shuffle(sample)
    return sample


def build_chunk_map():
    """(head, relation, tail) -> chunk 原文"""
    m = {}
    if not os.path.exists(PROGRESS):
        return m
    for line in open(PROGRESS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        chunk = r.get("chunk", "")
        for t in r.get("triples", []):
            if not t.get("valid"):
                continue
            key = (t["entity1"], t["relation"], t["entity2"])
            if key not in m:
                m[key] = chunk
    return m


def locate_context(chunk, head, evidence):
    """在 chunk 里定位 evidence/主体实体所在段落，返回该段 ± 2 段"""
    paras = [p for p in chunk.split("\n") if p.strip()]
    if not paras:
        return ""
    idx = -1
    for probe in (evidence, evidence[:25] if len(evidence) > 25 else evidence):
        if not probe:
            continue
        for i, p in enumerate(paras):
            if probe in p:
                idx = i
                break
        if idx >= 0:
            break
    if idx < 0:
        for i, p in enumerate(paras):
            if head and head in p:
                idx = i
                break
    if idx < 0:
        idx = 0
    lo, hi = max(0, idx - 2), min(len(paras), idx + 3)
    return "\n".join(paras[lo:hi])


def main():
    triples = json.load(open(TRIPLES, encoding="utf-8"))
    sample = stratified_sample(triples)
    chunk_map = build_chunk_map()

    header = ["序号", "关系类型", "主体实体", "主体类型", "客体实体", "客体类型",
              "confidence", "evidence(证据原文)", "原文上下文(证据所在段±2段)",
              "人工判定(对/错/存疑)", "备注"]
    matched = 0
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(sample, 1):
            chunk = chunk_map.get((t["head"], t["relation"], t["tail"]), "")
            ctx = locate_context(chunk, t["head"], t["evidence"]) if chunk else ""
            if ctx:
                matched += 1
            w.writerow([
                i, t["relation"], t["head"], t["head_type"], t["tail"], t["tail_type"],
                t["confidence"], t["evidence"], ctx, "", ""
            ])
    print(f"抽样 {len(sample)} 条，{matched} 条找到原文上下文")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
