#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D4 规模扩充：从专家核准的事实卡（formation_facts.json）程序化生成 ~1000 题独立测试集。

题目分 4 类，gold 答案全部来自事实卡原文句（不读图谱/向量库）：
  1. 单体事实题（岩性/年代/分布/化石/厚度/命名/同义名/接触）
  2. 对比题（硬编码有意义对比对 + 同年代程序化组合）
  3. 推理题（给特征猜地层 / 地层识别）
  4. 空间题（分布展布 / 厚度变化 / 地层分区）
fossil/synonym 字段「宁缺毋滥」过滤：剔除历史沿革句（含「队/院/所（年份）」开头）。

- 输入 : kb_storage/formation_facts.json
- 输出 : test_sets/test_set_independent.json
"""
import sys, os, json, re
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(PKG_DIR, "kb_storage", "formation_facts.json")
OUT = os.path.join(PKG_DIR, "test_sets", "test_set_independent.json")

SRC = "福建地质志.docx"

# —— fossil/synonym 过滤：历史沿革句（含单位+年份 开头）剔除 ——
HISTORY_RE = re.compile(r'^(?:福建省|江西省)?(?:区域地质|地质|闽煤|303|301).{0,20}?(?:队|院|所|组).{0,10}?\（?[0-9]{4}')

# —— 有地质意义的对比对（专家指定）：(A, B, 维度列表) ——
COMPARE_PAIRS = [
    ("焦坑组", "大坑村组", ["lithology", "age"]),
    ("梨山组", "焦坑组", ["age", "contact"]),
    ("文宾山组", "大坑村组", ["contact", "lithology"]),
    ("南岩组", "楼子坝组", ["lithology", "age"]),
    ("栖霞组", "文笔山组", ["lithology", "age"]),
    ("小溪组", "坂头组", ["lithology", "age"]),
    ("漳平组", "梨山组", ["age", "contact"]),
    ("黄坑组", "寨下组", ["age", "distribution"]),
    ("天井坪岩组", "麻源岩群", ["lithology", "distribution"]),
    ("楼子坝组", "黄连组", ["age", "lithology"]),
    ("天瓦崠组", "桃子坑组", ["age", "lithology"]),
    ("林地组", "经畲组", ["age", "lithology"]),
    ("船山组", "栖霞组", ["age", "lithology"]),
    ("文笔山组", "童子岩组", ["age", "lithology"]),
    ("翠屏山组", "罗坑组", ["age", "lithology"]),
    ("溪口组", "安仁组", ["age", "lithology"]),
    ("下村组", "藩坑组", ["age", "lithology"]),
    ("长林组", "南园组", ["age", "lithology"]),
    ("南园组", "小溪组", ["age", "lithology"]),
    ("下渡组", "坂头组", ["age", "lithology"]),
    ("石牛山组", "赤石群", ["age", "lithology"]),
    ("均口组", "沙县组", ["age", "lithology"]),
    ("东岩组", "大岭组", ["age", "lithology"]),
    ("黄潭组", "下峰组", ["age", "lithology"]),
    ("大金山岩组", "南山岩组", ["age", "lithology"]),
    ("林田组", "东坑口组", ["age", "lithology"]),
    ("魏坊组", "罗峰溪组", ["age", "lithology"]),
    ("天宝组", "同安组", ["age", "lithology"]),
    ("东山组", "长乐组", ["age", "lithology"]),
    ("大坑村组", "文宾山组", ["age", "distribution"]),
    ("焦坑组", "文宾山组", ["age", "distribution"]),
    ("下村组", "梨山组", ["age", "distribution"]),
    ("龙北溪组", "下峰组", ["age", "lithology"]),
    ("楼前组", "西溪组", ["age", "lithology"]),
    ("长兴组", "翠屏山组", ["age", "lithology"]),
]

FIELD_LABEL = {
    "lithology": "岩性（岩石组合）",
    "age": "地质年代",
    "distribution": "分布",
    "fossil": "古生物化石",
    "thickness": "厚度",
    "naming": "创建（命名）",
    "synonym": "同义异名（曾称/原名）",
    "contact": "接触关系",
}


def clean(s):
    return re.sub(r'\s+', ' ', s).strip()


def key_points(val, n=3):
    parts = re.split(r'[、，。；：]', val)
    pts = [p.strip() for p in parts if p.strip()]
    return pts[:n]


def is_history(s):
    return bool(HISTORY_RE.match(clean(s)))


class Gen:
    def __init__(self, facts):
        self.facts = {f["name"]: f for f in facts}
        self.names = [f["name"] for f in facts]
        self.q = []
        self._n = 0

    def add(self, question, qt, diff, cat, ans, kpts, ents):
        self._n += 1
        self.q.append({
            "id": f"FJ-I-{self._n:04d}",
            "question": question, "query_type": qt, "difficulty": diff,
            "category": cat, "expected_answer": clean(ans),
            "key_points": [clean(k) for k in kpts if k],
            "supporting_sources": [SRC],
            "expected_kg_path": [], "expected_entities": ents,
            "kb_id": "experiment", "source": "independent_handwritten",
        })

    # ---------- 1. 单体事实题 ----------
    def single_factual(self):
        for name in self.names:
            f = self.facts[name]
            if f.get("lithology"):
                self.add(f"{name}的岩性（岩石组合）特征是什么？", "FACTUAL", "medium", "lithology",
                         f["lithology"], key_points(f["lithology"]), [name])
            if f.get("age"):
                self.add(f"{name}形成于什么地质年代？", "FACTUAL", "easy", "geochronology",
                         f"{name}属{f['age']}。", [f['age']], [name])
            if f.get("distribution"):
                self.add(f"{name}主要分布在哪些地区？", "FACTUAL", "medium", "spatial_distribution",
                         f["distribution"], key_points(f["distribution"]), [name])
            if f.get("fossil") and not is_history(f["fossil"]):
                self.add(f"{name}含有哪些古生物化石？", "FACTUAL", "medium", "paleontology",
                         f["fossil"], key_points(f["fossil"]), [name])
            if f.get("thickness"):
                self.add(f"{name}的厚度有什么特点？", "FACTUAL", "medium", "thickness_analysis",
                         f["thickness"], key_points(f["thickness"]), [name])
            if f.get("naming"):
                self.add(f"{name}由谁创建、命名地点在哪？", "FACTUAL", "medium", "stratigraphic_identification",
                         f["naming"], key_points(f["naming"]), [name])
            if f.get("synonym") and not is_history(f["synonym"]) and ("曾称" in f["synonym"] or "原称" in f["synonym"] or "改称" in f["synonym"] or "原名" in f["synonym"] or "旧称" in f["synonym"]):
                self.add(f"{name}的同义异名（曾称/原名）是什么？", "REASONING", "medium", "synonym_detection",
                         f["synonym"], key_points(f["synonym"]), [name])
            if f.get("contact"):
                self.add(f"{name}与上下地层的接触关系是什么？", "FACTUAL", "medium", "stratigraphic_identification",
                         f["contact"], key_points(f["contact"]), [name])

    # ---------- 2. 对比题 ----------
    def comparative(self):
        seen = set()
        for a, b, dims in COMPARE_PAIRS:
            fa, fb = self.facts.get(a), self.facts.get(b)
            if not fa or not fb:
                continue
            for dim in dims:
                va, vb = fa.get(dim), fb.get(dim)
                if not va or not vb:
                    continue
                label = FIELD_LABEL[dim]
                if (a, b, dim) in seen:
                    continue
                seen.add((a, b, dim))
                self.add(f"请对比 {a} 与 {b} 的{label}差异。", "COMPARATIVE", "hard", "stratigraphic_comparison",
                         f"{a}：{va}；{b}：{vb}。",
                         [f"{a}: {key_points(va,1)[0] if key_points(va,1) else va}",
                          f"{b}: {key_points(vb,1)[0] if key_points(vb,1) else vb}"],
                         [a, b])
        return seen

    # ---------- 2b. 同年代单位自动对比（去重） ----------
    def comparative_same_age(self, seen):
        by_age = defaultdict(list)
        for name in self.names:
            ag = self.facts[name].get("age")
            if ag:
                by_age[ag].append(name)
        for ag, names in by_age.items():
            if len(names) < 2:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    fa, fb = self.facts[a], self.facts[b]
                    if (a, b, "lithology") in seen or (b, a, "lithology") in seen:
                        continue
                    if not fa.get("lithology") or not fb.get("lithology"):
                        continue
                    seen.add((a, b, "lithology"))
                    va, vb = fa["lithology"], fb["lithology"]
                    self.add(f"{a}与{b}同属{ag}，请对比二者的岩性差异。", "COMPARATIVE", "hard", "stratigraphic_comparison",
                             f"{a}：{va}；{b}：{vb}。",
                             [f"{a}: {key_points(va,1)[0] if key_points(va,1) else va}",
                              f"{b}: {key_points(vb,1)[0] if key_points(vb,1) else vb}"],
                             [a, b])

    # ---------- 3. 推理题（给特征猜地层） ----------
    def reasoning_guess(self):
        for name in self.names:
            f = self.facts[name]
            feats = []
            if f.get("lithology"):
                feats.append("岩性为" + clean(f["lithology"])[:40])
            if f.get("age"):
                feats.append("时代属" + f["age"])
            if f.get("distribution"):
                feats.append("分布于" + clean(f["distribution"])[:30])
            if len(feats) < 2:
                continue
            desc = "、".join(feats[:3])
            self.add(f"根据以下特征判断最可能的地层单位：{desc}。", "REASONING", "hard", "stratigraphic_identification",
                     f"该地层为{name}。判断依据：{desc}。",
                     [name] + feats[:2], [name])

    # ---------- 4. 空间题 ----------
    def spatial(self):
        for name in self.names:
            f = self.facts[name]
            if f.get("distribution"):
                # 展布方向 / 带状展布 → 空间题；否则普通分布题已在单体题覆盖
                if any(k in f["distribution"] for k in ["展布", "带状", "北东", "北北东", "条带"]):
                    self.add(f"{name}的空间展布有什么特征？", "SPATIAL", "hard", "spatial_distribution",
                             f["distribution"], key_points(f["distribution"]), [name])
            # 厚度变化 → 空间分析题
            if f.get("thickness") and any(k in f["thickness"] for k in ["变化", "减薄", "增", "一般", "不等", "最厚"]):
                self.add(f"{name}的厚度空间变化有什么规律？", "SPATIAL", "hard", "thickness_analysis",
                         f["thickness"], key_points(f["thickness"]), [name])


def load_handwritten():
    """加载手写综合题（handwritten_qa*.py 的 QA 列表），转成标准题目 dict。"""
    import importlib.util
    out = []
    files = ["handwritten_qa.py"] + [f"handwritten_qa{i}.py" for i in range(2, 6)]
    for fn in files:
        path = os.path.join(PKG_DIR, "scripts", fn)
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location(f"hw_{fn[:-3]}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for (question, qt, diff, cat, ans, kpts, ents) in m.QA:
            out.append({
                "question": clean(question), "query_type": qt, "difficulty": diff,
                "category": cat, "expected_answer": clean(ans),
                "key_points": [clean(k) for k in kpts if k],
                "supporting_sources": [SRC],
                "expected_kg_path": [], "expected_entities": list(ents or []),
                "kb_id": "experiment", "source": "independent_handwritten",
            })
    return out


def merge_and_emit(prog_q):
    hand_q = load_handwritten()
    all_q = prog_q + hand_q

    # 去重（按 question 文本），保留首个
    seen = set()
    dedup = []
    for q in all_q:
        if q["question"] in seen:
            continue
        seen.add(q["question"])
        dedup.append(q)

    # 重新编号
    for i, q in enumerate(dedup, 1):
        q["id"] = f"FJ-I-{i:04d}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"version": "3.0-independent-1000",
                   "description": "独立测试集：gold 答案基于《福建地质志》原文，不读图谱/向量库；程序化生成 + 专家手写综合题",
                   "count": len(dedup),
                   "questions": dedup}, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"\n=== 合并结果 ===")
    print(f"程序化 {len(prog_q)} + 手写 {len(hand_q)} = {len(all_q)}，去重后 {len(dedup)} 题")
    print("  query_type:", dict(Counter(q['query_type'] for q in dedup)))
    print("  category:", dict(Counter(q['category'] for q in dedup)))
    print("  difficulty:", dict(Counter(q['difficulty'] for q in dedup)))
    print("  source:", dict(Counter(q['source'] for q in dedup)))
    # 实体覆盖检查：expected_entities 里唯一实体数与命中
    ents = set()
    for q in dedup:
        ents.update(q["expected_entities"])
    print(f"  唯一 expected_entities: {len(ents)}")
    print(f"\n输出: {OUT}")


def main():
    facts = json.load(open(FACTS, encoding="utf-8"))
    g = Gen(facts)
    g.single_factual()
    seen = g.comparative()
    g.comparative_same_age(seen)
    g.reasoning_guess()
    g.spatial()

    print(f"程序化生成 {len(g.q)} 题")
    merge_and_emit(g.q)


if __name__ == "__main__":
    main()
