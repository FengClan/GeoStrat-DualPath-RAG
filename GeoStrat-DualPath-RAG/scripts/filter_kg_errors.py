#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D3 收尾：删除专家复核判「错」的三元组 + 规则过滤全量同类错误

- 输入 : kb_storage/experiment_triples_llm.json（D2 产物，3370 条）
- 动作 :
    1) 规则过滤（抓同类模式，能批量清全量）
       A. LITHOLOGY 客体以「建造/沉积/相/环境/旋回/组合/岩系」结尾 → 删（概念误标）
       B. FOSSIL 客体以「生物群/群落」结尾 → 删（概念误标）
       C. LOCATION 客体 ∈ {中国/华南/亚洲/世界/全球/欧亚/东亚} → 删（分布关系偏宽）
       D. MEMBER 主体为光杆段名（以「段」结尾且 ≤3 字）→ 删（实体规范化不足）
    2) 手动删除规则抓不到、但专家判错的 4 条（关系选错/证据不支持）
- 输出 : kb_storage/experiment_triples_llm_filtered.json + graphs/experiment_kb_llm_filtered.graphml
- 用法 : --apply 才写输出；不加则仅预览各规则命中量
"""
import sys, os, json, argparse, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLES = os.path.join(PKG_DIR, "kb_storage", "experiment_triples_llm.json")
OUT_JSON = os.path.join(PKG_DIR, "kb_storage", "experiment_triples_llm_filtered.json")
OUT_GRAPH = os.path.join(PKG_DIR, "kb_storage", "graphs", "experiment_kb_llm_filtered.graphml")

# 规则抓不到的专家判错条目（head, relation, tail）
MANUAL_DELETE = {
    ("藩坑组", "COMPOSED_OF", "花岗斑岩"),
    ("小溪组", "COMPOSED_OF", "火山岩"),
    ("鸣山组", "BELONGS_TO", "栖霞组"),
    ("丁屋岭组", "SAME_AS", "楼子坝组"),
}

# 专家判「错」的全部 9 条（用于最后校验是否全被处理）
EXPERT_WRONG = {
    ("文笔山组", "COMPOSED_OF", "含煤建造"),
    ("藩坑组", "COMPOSED_OF", "花岗斑岩"),
    ("小溪组", "COMPOSED_OF", "火山岩"),
    ("上段", "CONTAINS_FOSSIL", "淡水双壳类"),
    ("坂头组", "CONTAINS_FOSSIL", "热河生物群"),
    ("鸣山组", "BELONGS_TO", "栖霞组"),
    ("藩坑组", "COMPOSED_OF", "海相沉积"),
    ("丁屋岭组", "SAME_AS", "楼子坝组"),
    ("溪口组", "DISTRIBUTED_IN", "中国"),
}

LITHOLOGY_BAD_SUFFIX = ("建造", "沉积", "相", "环境", "旋回", "组合", "岩系")
FOSSIL_BAD_SUFFIX = ("生物群", "群落")
LOCATION_BAD = {"中国", "华南", "亚洲", "世界", "全球", "欧亚", "东亚"}

# 纯方位/序数光杆段名（上段/下段/中段/第一段/第四、五段…），缺组名前缀无法定位
BARE_MEMBER_RE = re.compile(r"^(?:上|中|下|第[一二三四五六七八九十]+[、和]?)+段$")


def rule_key(t):
    return (t["head"], t["relation"], t["tail"])


def decide(t):
    """返回删除原因字符串；None 表示保留"""
    head, rel, tail = t["head"], t["relation"], t["tail"]
    htype, ttype = t["head_type"], t["tail_type"]

    if rule_key(t) in MANUAL_DELETE:
        return "手动删除(专家判错)"
    if ttype == "LITHOLOGY" and rel == "COMPOSED_OF" and tail.endswith(LITHOLOGY_BAD_SUFFIX):
        return f"LITHOLOGY 概念误标(后缀)"
    if ttype == "FOSSIL" and rel == "CONTAINS_FOSSIL" and tail.endswith(FOSSIL_BAD_SUFFIX):
        return "FOSSIL 概念误标(生物群/群落)"
    if ttype == "LOCATION" and tail in LOCATION_BAD:
        return "LOCATION 范围过泛"

    def _bare_member(name, etype):
        return etype == "MEMBER" and bool(BARE_MEMBER_RE.match(name))

    if _bare_member(head, htype) or _bare_member(tail, ttype):
        return "MEMBER 光杆段名"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    triples = json.load(open(TRIPLES, encoding="utf-8"))
    print(f"总三元组 {len(triples)} 条\n")

    from collections import Counter
    reason_cnt = Counter()
    kept, removed = [], []
    for t in triples:
        reason = decide(t)
        if reason:
            reason_cnt[reason] += 1
            removed.append((t, reason))
        else:
            kept.append(t)

    print("=== 规则命中统计（拟删除） ===")
    for reason, n in reason_cnt.most_common():
        print(f"  {reason}: {n} 条")
    print(f"  合计删除 {len(removed)} 条 → 保留 {len(kept)} 条")

    # 校验 9 条专家判错是否全覆盖
    remaining_wrong = EXPERT_WRONG - {rule_key(t) for t, _ in removed}
    print(f"\n=== 专家判错 9 条覆盖校验 ===")
    print(f"  未覆盖的判错条目: {remaining_wrong if remaining_wrong else '无（全部处理）'}")

    # 打印部分删除样例
    print("\n=== 删除样例（按规则各取前 2 条） ===")
    shown = Counter()
    for t, reason in removed:
        if shown[reason] < 2:
            shown[reason] += 1
            print(f"  [{reason}] {t['head']}({t['head_type']}) --{t['relation']}--> {t['tail']}({t['tail_type']})")

    if args.apply:
        import networkx as nx
        G = nx.DiGraph()
        for t in kept:
            G.add_node(t["head"], entity_type=t["head_type"])
            G.add_node(t["tail"], entity_type=t["tail_type"])
            G.add_edge(t["head"], t["tail"], relation=t["relation"],
                       confidence=t["confidence"], evidence=t.get("evidence", ""))
        os.makedirs(os.path.dirname(OUT_GRAPH), exist_ok=True)
        nx.write_graphml(G, OUT_GRAPH)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"\n=== 已应用 ===")
        print(f"节点 {G.number_of_nodes()} / 边 {G.number_of_edges()} / 三元组 {len(kept)}")
        print(f"图: {OUT_GRAPH}")
        print(f"三元组: {OUT_JSON}")
    else:
        print("\n（预览模式，未写文件。加 --apply 应用）")


if __name__ == "__main__":
    main()
