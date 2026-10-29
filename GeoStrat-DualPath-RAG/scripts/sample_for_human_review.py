#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D3 方案：从 LLM 抽取的三元组按关系类型分层抽 100 条，生成人工核查清单（CSV）

- 输入 : kb_storage/experiment_triples_llm.json（D2 产物，3370 条带 evidence/confidence）
- 抽样 : 按关系类型占比分层，每类至少 1 条；固定 seed=42 保证可复现
- 输出 : kb_storage/d3_human_review_sample.csv（UTF-8 BOM，Excel 直接打开不乱码）
         含空的「人工判定」「备注」列，供专家对照 evidence 判对/错/存疑
"""
import sys, os, json, random, csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLES = os.path.join(PKG_DIR, "kb_storage", "experiment_triples_llm.json")
OUT = os.path.join(PKG_DIR, "kb_storage", "d3_human_review_sample.csv")


def main():
    triples = json.load(open(TRIPLES, encoding="utf-8"))
    by_rel = {}
    for t in triples:
        by_rel.setdefault(t["relation"], []).append(t)

    total = len(triples)
    target = 100

    # 按占比分层，每类至少 1 条
    counts = {rel: max(1, round(target * len(items) / total)) for rel, items in by_rel.items()}
    # 修正总数到 target（多退少补，优先调整占比大的类）
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

    random.seed(42)
    sample = []
    for rel, n in counts.items():
        picked = random.sample(by_rel[rel], min(n, len(by_rel[rel])))
        sample.extend(picked)
    random.shuffle(sample)

    header = ["序号", "关系类型", "主体实体", "主体类型", "客体实体", "客体类型",
              "confidence", "evidence(证据原文)", "人工判定(对/错/存疑)", "备注"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(sample, 1):
            w.writerow([
                i, t["relation"], t["head"], t["head_type"],
                t["tail"], t["tail_type"], t["confidence"],
                t["evidence"], "", ""
            ])

    print(f"总三元组 {total} 条 → 分层抽 {len(sample)} 条")
    print("各关系抽样数:")
    for rel in rels_sorted:
        print(f"  {rel}: {counts[rel]} 条 (共 {len(by_rel[rel])} 条)")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
