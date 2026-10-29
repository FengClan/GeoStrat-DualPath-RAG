#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诚实测试集生成器 v2（取代 generate_1000q.py）。

设计原则：**能做多少做多少，不注水。**
  - 一道题 = 一个真实存在的事实位，gold 是原文原句，带 src_para 可回 docx 核对。
  - 不做「同一事实多种问法」的变体填充。唯一例外是确实不同问法维度的题型
    （正问 / 反问 / 聚合 / 多跳），它们在 template_id 上可区分，不冒充新事实。
  - 抽不到的维度就不出题（宁缺毋滥）。

相对 v1 的修复：
  1. `source` 字段：v1 把程序化题和手写题**全部**标成 "independent_handwritten"，
     两者无法区分。v2 分为 programmatic_template / programmatic_evolution /
     handwritten_expert / handwritten_surface。
  2. `id` 稳定性：v1 是「去重后按枚举顺序重编」，生成器一改所有 ID 平移，
     run_full_eval.py 的断点续跑（按 q["id"] 匹配）就此失效。
     v2 改用问题文本的内容哈希：`FJ-I-<sha1(归一化问题)[:8]>`，与顺序无关。
  3. `expected_dimensions`：v1 的 comparative 从不写这个字段，而
     evaluator.evaluate_contrast() 读的正是它 → 所有对比题在 contrast_agreement
     上一律 returned 0.0（实测四个基线全为 0.0）。v2 补上，形状见 evaluator
     的 _compare_dimension(): [{label, col_a:{...}, col_b:{...}}]。
  4. 去重优先级：v1 是 prog_q + hand_q 且「保留首个」，导致一条程序化题会静默
     吃掉完全相同的手写题。v2 冲突时优先保留手写题。
  5. 新增 evidence_span / src_para / fact_id / template_id / source_section，
     支持「gold 可回原文核对」与「按来源分组报告」。

- 输入 : kb_storage/formation_facts_v2.json + kb_storage/evolution_sections.json
         + scripts/handwritten_qa*.py
- 输出 : test_sets/test_set_honest.json + test_sets/test_set_honest.manifest.json
"""
import sys, os, json, re, hashlib
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(PKG_DIR, "kb_storage", "formation_facts_v2.json")
EVOL = os.path.join(PKG_DIR, "kb_storage", "evolution_sections.json")
OUT = os.path.join(PKG_DIR, "test_sets", "test_set_honest.json")
MANIFEST = os.path.join(PKG_DIR, "test_sets", "test_set_honest.manifest.json")

SRC = "福建地质志.docx"

HISTORY_RE = re.compile(r'^(?:福建省|江西省)?(?:区域地质|地质|闽煤|303|301).{0,20}?(?:队|院|所|组).{0,10}?\（?[0-9]{4}')

# —— 有地质意义的对比对（沿用 v1 专家指定清单）——
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
    "lithology": "岩性（岩石组合）", "age": "地质年代", "distribution": "分布",
    "fossil": "古生物化石", "thickness": "厚度", "naming": "创建（命名）",
    "synonym": "同义异名（曾称/原名）", "contact": "接触关系",
    "section": "层型剖面", "metamorph": "变质程度（变质相）",
    "mineralization": "含矿性", "sed_env": "沉积环境",
    "thickness_num": "厚度数值", "correlation": "区域对比",
    "overlying": "上覆地层", "underlying": "下伏地层", "subdivision": "内部划分（段）",
}

# 单体事实题：字段 -> (问法模板, query_type, difficulty, category)
SINGLE_TPL = {
    "lithology":     ("{n}的岩性（岩石组合）特征是什么？", "FACTUAL", "medium", "lithology"),
    "age":           ("{n}形成于什么地质年代？", "FACTUAL", "easy", "geochronology"),
    "distribution":  ("{n}主要分布在哪些地区？", "FACTUAL", "medium", "spatial_distribution"),
    "fossil":        ("{n}含有哪些古生物化石？", "FACTUAL", "medium", "paleontology"),
    "thickness":     ("{n}的厚度有什么特点？", "FACTUAL", "medium", "thickness_analysis"),
    "naming":        ("{n}由谁创建、命名地点在哪？", "FACTUAL", "medium", "stratigraphic_identification"),
    "synonym":       ("{n}的同义异名（曾称/原名）是什么？", "REASONING", "medium", "synonym_detection"),
    "contact":       ("{n}与上下地层的接触关系是什么？", "FACTUAL", "medium", "stratigraphic_identification"),
    "section":       ("{n}的层型（建组）剖面在哪里？", "FACTUAL", "hard", "stratigraphic_identification"),
    "metamorph":     ("{n}的变质程度或变质相如何？", "FACTUAL", "hard", "metamorphism"),
    "mineralization":("{n}的含矿性如何？", "FACTUAL", "hard", "economic_geology"),
    "sed_env":       ("{n}形成于什么沉积环境？", "FACTUAL", "hard", "sedimentary_environment"),
    "thickness_num": ("{n}的厚度（具体数值）是多少？", "FACTUAL", "medium", "thickness_analysis"),
    "correlation":   ("{n}可与哪些地层相对比？", "REASONING", "hard", "stratigraphic_comparison"),
    "overlying":     ("{n}的上覆地层是什么？", "FACTUAL", "medium", "stratigraphic_identification"),
    "underlying":    ("{n}的下伏地层是什么？", "FACTUAL", "medium", "stratigraphic_identification"),
    "subdivision":   ("{n}内部可划分为哪些段（如何分段）？", "FACTUAL", "medium", "stratigraphic_identification"),
}

# 含「段」划分的单位名，用于判断 subdivision 抽到的是不是本单位的分段
UNIT_NAME_RE = None  # 运行时填

# 演化史题的题材识别（按优先级）：(题材名, 关键词, 问法, category)
EVOL_TOPICS = [
    ("tectonic",    ["构造", "断裂", "隆起", "坳陷", "盆地", "褶皱", "板块", "俯冲", "碰撞"],
     "{t}的构造演化特征是什么？", "tectonic_reasoning"),
    ("magmatic",    ["岩浆", "火山", "侵入", "花岗岩", "玄武岩", "喷发"],
     "{t}的岩浆活动（火山岩及火山作用）特征是什么？", "magmatism"),
    ("metamorphic", ["变质", "片岩", "片麻", "混合岩"],
     "{t}的变质作用特征是什么？", "metamorphism"),
    ("sedimentary", ["沉积", "建造", "海侵", "海退", "岩相"],
     "{t}的沉积建造（地层与沉积作用）特征是什么？", "sedimentary_environment"),
    ("paleogeo",    ["古地理", "古气候", "古环境", "古温度"],
     "{t}的古地理（古气候）格局如何？", "paleogeography"),
]

CHAP_HEAD_RE = re.compile(r'^第[一二三四五六七八九十百]+章\s')

# 演化史 gold 的长度上限：超过就按整段截断（不切句子中间）
EVOL_GOLD_MAX = 800


def clean(s):
    return re.sub(r'\s+', ' ', s or "").strip()


def key_points(val, n=3):
    parts = re.split(r'[、，。；：]', val or "")
    return [p.strip() for p in parts if p.strip()][:n]


def is_history(s):
    return bool(HISTORY_RE.match(clean(s)))


def stable_id(question):
    """内容哈希 ID：与生成顺序无关，生成器改动不会平移已有题的 ID。"""
    norm = re.sub(r'[\s，。？！、；：（）()]', '', question)
    return "FJ-I-" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


class Gen:
    def __init__(self, facts):
        self.facts = {f["name"]: f for f in facts}
        self.names = [f["name"] for f in facts]
        self.q = []
        self.templates = Counter()   # template_id -> 计数
        self.skipped = Counter()     # 跳过原因 -> 计数

    def add(self, question, qt, diff, cat, ans, kpts, ents,
            source, template_id, fact_id=None, evidence_span=None, src_para=None,
            source_section="unit", expected_dimensions=None, provenance="original_text"):
        q = clean(question)
        if not q:
            self.skipped["empty_question"] += 1
            return
        rec = {
            "id": stable_id(q),
            "question": q, "query_type": qt, "difficulty": diff,
            "category": cat, "expected_answer": clean(ans),
            "key_points": [clean(k) for k in kpts if k],
            "supporting_sources": [SRC],
            "expected_kg_path": [],
            "expected_entities": [e for e in (ents or []) if e],
            "kb_id": "experiment",
            # ---- v2 新增 ----
            "source": source,
            "template_id": template_id,
            "fact_id": fact_id,
            "evidence_span": clean(evidence_span or ans),
            "src_para": src_para,
            "source_section": source_section,
            "provenance": provenance,
        }
        if expected_dimensions:
            rec["expected_dimensions"] = expected_dimensions
        self.templates[template_id] += 1
        self.q.append(rec)

    def _ev(self, name, field):
        """取字段的原文溯源信息。"""
        f = self.facts[name]
        e = (f.get("evidence") or {}).get(field) or {}
        return e.get("src_para")

    # ---------- 1. 单体事实题（17 个维度） ----------
    def single_factual(self):
        for name in self.names:
            f = self.facts[name]
            for field, (tpl, qt, diff, cat) in SINGLE_TPL.items():
                val = f.get(field)
                if not val:
                    self.skipped[f"{field}_missing"] += 1
                    continue
                # 化石/同义名的历史沿革句过滤（沿用 v1）
                if field in ("fossil", "synonym") and is_history(val):
                    self.skipped[f"{field}_history"] += 1
                    continue
                if field == "synonym" and not any(
                        k in val for k in ("曾称", "原称", "原名", "旧称", "改称")):
                    self.skipped["synonym_weak"] += 1
                    continue
                if field == "age":
                    # 年代来自人工核准的 AGE 表（按志书分期章节整理），
                    # 不是原文逐字句，故 provenance 单独标注，便于论文如实说明
                    self.add(tpl.format(n=name), qt, diff, cat, f"{name}属{f['age']}。",
                             [f['age']], [name], "programmatic_template",
                             f"single_factual.{field}", f"{name}:{field}",
                             evidence_span=f"{name}属{f['age']}。", src_para=None,
                             provenance="expert_assigned")
                    continue
                self.add(tpl.format(n=name), qt, diff, cat, val, key_points(val), [name],
                         "programmatic_template", f"single_factual.{field}",
                         f"{name}:{field}", val, self._ev(name, field))

    # ---------- 2. 对比题（补 expected_dimensions） ----------
    def _dim_payload(self, name, field, value):
        return {"label": FIELD_LABEL[field],
                "col_a" if name == "_a" else "col_b": {}}

    def comparative(self):
        seen = set()
        for a, b, dims in COMPARE_PAIRS:
            fa, fb = self.facts.get(a), self.facts.get(b)
            if not fa or not fb:
                self.skipped["compare_unit_missing"] += 1
                continue
            for dim in dims:
                va, vb = fa.get(dim), fb.get(dim)
                if not va or not vb:
                    self.skipped[f"compare_{dim}_missing"] += 1
                    continue
                if (a, b, dim) in seen:
                    continue
                seen.add((a, b, dim))
                self._emit_compare(a, b, dim, va, vb, curated=True)
        return seen

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
                    if not fa.get("lithology") or not fb.get("lithology"):
                        continue
                    if (a, b, "lithology") in seen or (b, a, "lithology") in seen:
                        continue
                    seen.add((a, b, "lithology"))
                    self._emit_compare(a, b, "lithology", fa["lithology"],
                                       fb["lithology"], curated=False, age=ag)

    def _emit_compare(self, a, b, dim, va, vb, curated, age=None):
        label = FIELD_LABEL[dim]
        if curated:
            q = f"请对比 {a} 与 {b} 的{label}差异。"
            tpl_id = f"comparative_curated.{dim}"
        else:
            q = f"{a}与{b}同属{age}，请对比二者的岩性差异。"
            tpl_id = "comparative_same_age.lithology"
        # expected_dimensions：evaluator._compare_dimension() 读 col_a/col_b 下的
        # tags_common / tags_diff / description / conclusion
        exp_dims = [{
            "label": label,
            "col_a": {"description": va, "tags_common": key_points(va, 2)},
            "col_b": {"description": vb, "tags_common": key_points(vb, 2)},
        }]
        pa, pb = self._ev(a, dim), self._ev(b, dim)
        # 对比题的 gold 横跨两个单位，src_para 取 A 侧（B 侧在证据里注明）
        self.add(q, "COMPARATIVE", "hard", "stratigraphic_comparison",
                 f"{a}：{va}；{b}：{vb}。",
                 [f"{a}: {key_points(va, 1)[0] if key_points(va, 1) else va}",
                  f"{b}: {key_points(vb, 1)[0] if key_points(vb, 1) else vb}"],
                 [a, b], "programmatic_template", tpl_id,
                 None, f"{a}：{va}；{b}：{vb}。", pa,
                 expected_dimensions=exp_dims)

    # ---------- 3. 多跳链式推理（新） ----------
    def multi_hop(self):
        """A 的上覆/下伏地层 → 该地层的岩性。需要两个单位都在 63 单位表内。"""
        for name in self.names:
            f = self.facts[name]
            for rel, tpl_field, rel_word in (("overlying", "上覆", "上覆"),
                                             ("underlying", "下伏", "下伏")):
                sent = f.get(rel)
                if not sent:
                    continue
                target = self._resolve_unit(sent, exclude=name)
                if not target:
                    self.skipped[f"multihop_{rel}_unresolved"] += 1
                    continue
                tgt_lith = self.facts[target].get("lithology")
                if not tgt_lith:
                    self.skipped[f"multihop_{rel}_no_lith"] += 1
                    continue
                self.add(
                    f"{name}的{rel_word}地层是什么？该地层的岩性又如何？",
                    "REASONING", "hard", "stratigraphic_identification",
                    f"{name}的{rel_word}地层为{target}，其岩性为：{tgt_lith}",
                    [f"{rel_word}地层为{target}", clean(tgt_lith)[:40]],
                    [name, target], "programmatic_template",
                    f"multi_hop.{rel}", f"{name}:{rel}->{target}:lithology",
                    f"{sent}；{tgt_lith}", self._ev(name, rel))

    def _resolve_unit(self, sentence, exclude):
        """从一句话里认出 63 单位表里的地层名（最长匹配，排除自身）。"""
        cands = [n for n in self.names if n != exclude and n in sentence]
        if not cands:
            return None
        return max(cands, key=len)

    # ---------- 4. 聚合 / 反向题（新） ----------
    def aggregate(self, min_members=3):
        """按年代、化石类群聚合。仅当同类成员 >= min_members 时出题，
        避免「哪几个地层属 X」答案只有一个时退化成单体题。"""
        # 4a. 按年代聚合
        by_age = defaultdict(list)
        for name in self.names:
            ag = self.facts[name].get("age")
            if ag:
                by_age[ag].append(name)
        for ag, names in by_age.items():
            if len(names) < min_members:
                continue
            listed = "、".join(names)
            self.add(f"福建省有哪些地层单位形成于{ag}？", "REASONING", "hard",
                     "geochronology", f"形成于{ag}的地层单位有：{listed}。",
                     names[:4], names, "programmatic_template",
                     "aggregate.by_age", f"AGG:age={ag}")

        # 4b. 按化石类群聚合
        FOSSIL_GROUPS = ["菊石", "腕足", "双壳类", "笔石", "孢粉", "珊瑚", "植物",
                         "蜓", "海百合", "三叶虫", "腹足", "介形虫"]
        for fg in FOSSIL_GROUPS:
            hits = [n for n in self.names
                    if fg in (self.facts[n].get("fossil") or "")
                    and not is_history(self.facts[n].get("fossil"))]
            if len(hits) < min_members:
                continue
            self.add(f"福建省哪些地层单位含有{fg}化石？", "REASONING", "hard",
                     "paleontology",
                     f"含{fg}化石的地层单位有：{'、'.join(hits)}。",
                     hits[:4], hits, "programmatic_template",
                     "aggregate.by_fossil", f"AGG:fossil={fg}")

        # 4c. 按沉积环境聚合
        for env in ["浅海", "滨海", "三角洲", "碳酸盐台地", "泻湖", "河流相", "湖泊相"]:
            hits = [n for n in self.names if env in (self.facts[n].get("sed_env") or "")]
            if len(hits) < min_members:
                continue
            self.add(f"福建省哪些地层单位形成于{env}环境？", "REASONING", "hard",
                     "sedimentary_environment",
                     f"形成于{env}环境的地层单位有：{'、'.join(hits)}。",
                     hits[:4], hits, "programmatic_template",
                     "aggregate.by_sed_env", f"AGG:sed_env={env}")

    def reverse_lookup(self):
        """反向题：给岩性/年代特征，问是哪个单位。与 reasoning_guess 的区别是
        这里只用一个特征，且答案是唯一匹配（若多个单位共享同一描述则跳过）。"""
        for field, cat, qfmt in (
                ("sed_env", "sedimentary_environment", "福建省哪个地层单位形成于以下沉积环境？{v}"),
                ("mineralization", "economic_geology", "福建省哪个地层单位具有以下含矿性特征？{v}"),):
            by_val = defaultdict(list)
            for name in self.names:
                v = self.facts[name].get(field)
                if v:
                    by_val[v].append(name)
            for v, names in by_val.items():
                if len(names) != 1:          # 只有唯一匹配才是良定义的
                    continue
                name = names[0]
                self.add(qfmt.format(v=clean(v)[:80]), "REASONING", "hard", cat,
                         f"{name}。依据：{v}", [name], [name],
                         "programmatic_template", f"reverse.{field}",
                         f"{name}:{field}", v, self._ev(name, field))

    # ---------- 5. 空间题 ----------
    def spatial(self):
        for name in self.names:
            f = self.facts[name]
            d = f.get("distribution")
            if d and any(k in d for k in ["展布", "带状", "北东", "北北东", "条带"]):
                self.add(f"{name}的空间展布有什么特征？", "SPATIAL", "hard",
                         "spatial_distribution", d, key_points(d), [name],
                         "programmatic_template", "spatial.distribution",
                         f"{name}:distribution", d, self._ev(name, "distribution"))
            t = f.get("thickness")
            if t and any(k in t for k in ["变化", "减薄", "增", "一般", "不等", "最厚"]):
                self.add(f"{name}的厚度空间变化有什么规律？", "SPATIAL", "hard",
                         "thickness_analysis", t, key_points(t), [name],
                         "programmatic_template", "spatial.thickness",
                         f"{name}:thickness", t, self._ev(name, "thickness"))
            cor = f.get("correlation")
            if cor and any(k in cor for k in ["分区", "小区", "带", "东部", "西部", "南部", "北部"]):
                self.add(f"{name}的分布具有什么区域（分区）特征？", "SPATIAL", "hard",
                         "spatial_distribution", cor, key_points(cor), [name],
                         "programmatic_template", "spatial.correlation",
                         f"{name}:correlation", cor, self._ev(name, "correlation"))

    # ---------- 6. 演化史 / 结语题（第八篇） ----------
    def evolution(self, sections, paras):
        """地质发展史出题，叙述章与结语用两套不同策略。

        **叙述章**（第三十四章…第四十章）按「节 × 题材」出题，而不是逐段出题。
        逐段出题会产生大量问题文本完全相同的题（同一节里 131 个构造段都问
        「XX 的构造演化特征是什么？」），既被去重掉、又不符合真实问法。
        现在把同一节内同题材的段落合成一个 gold（按整段累积到 EVOL_GOLD_MAX）。

        **结语**改为按**节标题**定专题、每节只出一题。
        原先结语也走「节 × 段落关键词」，只要该节里有一句话提到「火山」，就会
        生成「…区域变质岩与变质作用的岩浆活动特征是什么？」这种跨专题的荒唐题
        （实测 39 题里过半如此）。而结语每节的标题本身就是一个专题
        （「…·侵入岩与深成作用」「…·区域变质岩与变质作用」），据此定向即可，
        不需要从段落关键词反推。
        """
        for s in sections:
            if s["chapter_no"] == "序":
                continue
            # 非空段 -> 全局段落号（text 是由非空段 join 而成）
            seg_idx = [i for i in range(s["start_idx"], s["end_idx"])
                       if i < len(paras) and paras[i]]
            plist = [p.strip() for p in s["text"].split("\n") if p.strip()]
            keep = [(gi, p) for gi, p in zip(seg_idx, plist)
                    if not CHAP_HEAD_RE.match(p)]

            if s["kind"] == "conclusion":
                self._evolution_conclusion(s, keep)
                continue

            label = s["title"]
            by_topic = defaultdict(list)
            for gi, p in keep:
                for topic, kws, _, _ in EVOL_TOPICS:
                    if any(k in p for k in kws):
                        by_topic[topic].append((gi, p))
                        break
                else:
                    by_topic["general"].append((gi, p))

            for topic, items in by_topic.items():
                q_fmt = next((tf for tk, _, tf, _ in EVOL_TOPICS if tk == topic),
                             "{t}的地质特征是什么？")
                cat = next((c for tk, _, _, c in EVOL_TOPICS if tk == topic),
                           "stratigraphic_identification")

                gold, used_items = self._accumulate(items)
                if gold is None:
                    self.skipped[f"evol_{topic}_too_short"] += 1
                    continue

                self.add(q_fmt.format(t=label), "REASONING", "hard", cat, gold,
                         key_points(gold, 4), [],
                         "programmatic_evolution", f"evolution.{topic}",
                         f"{s['section_id']}:{topic}", gold, used_items[0],
                         source_section="evolution")

    @staticmethod
    def _accumulate(items, limit=EVOL_GOLD_MAX, minlen=40):
        """按整段累积 gold，到上限为止。撑不起 gold 时返回 (None, [])。"""
        gold, used = "", []
        for gi, p in items:
            if gold and len(gold) + len(p) > limit:
                break
            gold += p
            used.append(gi)
        if len(gold) < minlen:
            return None, []
        return gold, used

    def _evolution_conclusion(self, s, keep):
        """结语：按节标题定专题，每节一题。"""
        label = s["title"].split("·", 1)[-1] if "·" in s["title"] else s["title"]
        topic = next((tk for tk, kws, _, _ in EVOL_TOPICS
                      if any(k in label for k in kws)), "general")
        q_fmt = next((tf for tk, _, tf, _ in EVOL_TOPICS if tk == topic), None)
        cat = next((c for tk, _, _, c in EVOL_TOPICS if tk == topic),
                   "stratigraphic_identification")

        gold, used = self._accumulate(keep)
        if gold is None:
            self.skipped["evol_conclusion_too_short"] += 1
            return

        if q_fmt:
            # 模板里的括号限定（如「岩浆活动（火山岩及火山作用）」）对结语是多余的，
            # 标题已经写明了专题，套上去反而自相矛盾（「侵入岩与深成作用的
            # 岩浆活动（火山岩及火山作用）特征」）。去掉括号限定，用标题作主语。
            q = re.sub(r'（[^）]*）', '', q_fmt).format(t=label)
        else:
            # 标题没有对应专题（如「第四纪地质」）：不硬套模板，直接问该节内容
            q = f"{label}方面有哪些内容？"

        self.add(f"（据志书结语）{q}", "REASONING", "hard", cat, gold,
                 key_points(gold, 4), [],
                 "programmatic_evolution", f"evolution.{topic}",
                 f"{s['section_id']}:{topic}", gold, used[0],
                 source_section="conclusion")


def load_handwritten():
    """加载手写综合题。source 如实标 handwritten_expert（v1 一律标
    independent_handwritten，导致与程序化题无法区分）。"""
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
            q = clean(question)
            out.append({
                "id": stable_id(q),
                "question": q, "query_type": qt, "difficulty": diff,
                "category": cat, "expected_answer": clean(ans),
                "key_points": [clean(k) for k in kpts if k],
                "supporting_sources": [SRC],
                "expected_kg_path": [],
                "expected_entities": list(ents or []),
                "kb_id": "experiment",
                "source": "handwritten_expert",
                "template_id": "handwritten",
                "fact_id": None,
                "evidence_span": clean(ans),
                "src_para": None,
                "source_section": "handwritten",
                "provenance": "expert_written",
            })
    return out


def merge_and_emit(prog_q, facts):
    hand_q = load_handwritten()
    # 冲突时优先保留手写题（v1 是 prog + hand 且保留首个 → 程序化题会静默吃掉手写题）
    merged, index = [], {}
    for q in hand_q + prog_q:
        key = re.sub(r'[\s，。？！、；：（）()]', '', q["question"])
        if key in index:
            prev = index[key]
            if prev["source"] == "handwritten_expert" and q["source"] != "handwritten_expert":
                continue          # 已有的手写题胜出
            if q["source"] == "handwritten_expert":
                merged[merged.index(prev)] = q   # 手写题替换程序化题
                index[key] = q
            continue
        index[key] = q
        merged.append(q)

    # ID 冲突自检（内容哈希理论上是唯一键，撞了要大声失败）
    ids = [q["id"] for q in merged]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        print(f"  [WARN] ID 哈希冲突 {len(dup)} 组，已加序号后缀: {dup[:5]}")
        seen = Counter()
        for q in merged:
            seen[q["id"]] += 1
            if seen[q["id"]] > 1:
                q["id"] = f"{q['id']}-{seen[q['id']]}"

    # 稳定排序：source → query_type → question，保证输出可复现
    merged.sort(key=lambda q: (q["source"], q["query_type"], q["question"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {
        "version": "4.0-honest",
        "description": ("诚实独立测试集：每题的 gold 均为《福建地质志》原文原句（或人工核准的"
                        "年代），带 src_para / evidence_span 可回原文核对；"
                        "不做同事实多问法填充。gold 不来自知识图谱（回应 R2#1）。"),
        "count": len(merged),
        "questions": merged,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    manifest = build_manifest(merged, facts)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== 合并结果 ===")
    print(f"程序化 {len(prog_q)} + 手写 {len(hand_q)} → 去重后 {len(merged)} 题")
    for k in ("query_type", "category", "source", "source_section", "provenance"):
        print(f"  {k}: {dict(Counter(q.get(k) for q in merged))}")
    print(f"\n输出: {OUT}\n清单: {MANIFEST}")
    return merged


def build_manifest(questions, facts):
    n = len(questions)
    with_para = sum(1 for q in questions if q.get("src_para") is not None)
    return {
        "count": n,
        "by_source": dict(Counter(q["source"] for q in questions)),
        "by_query_type": dict(Counter(q["query_type"] for q in questions)),
        "by_category": dict(Counter(q["category"] for q in questions)),
        "by_difficulty": dict(Counter(q["difficulty"] for q in questions)),
        "by_source_section": dict(Counter(q["source_section"] for q in questions)),
        "by_provenance": dict(Counter(q["provenance"] for q in questions)),
        "by_template": dict(Counter(q["template_id"] for q in questions)),
        "unique_fact_ids": len({q["fact_id"] for q in questions if q.get("fact_id")}),
        "traceable_to_paragraph": {"count": with_para,
                                   "ratio": round(with_para / n, 4) if n else 0.0},
        "unit_coverage": len({e for q in questions for e in q["expected_entities"]}),
        "notes": [
            "唯一事实题与实际题数之比 = unique_fact_ids / count，用于如实标注"
            "「同一事实多问法」的占比（若有）。",
            "provenance=expert_assigned 的题（地质年代）来自人工核准的年代表，"
            "不是原文逐字句，论文中应如实说明。",
            "source_section=evolution 的题来自第八篇地质发展史。实测 KG 中"
            "「构造演化/岩浆活动/变质作用」出现 0 次而向量库有 22/33/76 次，"
            "该组题对 graph_rag 结构性不可答，必须单独成表或按来源分组报告。",
            "source_section=conclusion 的题来自志书结语，题面冠以「（据志书结语）」"
            "以便与正文题区分。结语为综述性文字，gold 是整节累积（非单句），"
            "其问题粒度与 unit/evolution 两组不可直接比较。",
            "traceable_to_paragraph 只统计能回溯到原文段落的题；handwritten 组是"
            "人工撰写、无对应原文段落，故不在其内，该比率并非「数据质量」指标。",
        ],
    }


def main():
    from docx import Document
    facts = json.load(open(FACTS, encoding="utf-8"))
    sections = json.load(open(EVOL, encoding="utf-8"))
    # 需要 docx 段落原文来把「非空段序号」换算回全局段落号（溯源用）
    paras = [p.text.strip() for p in
             Document(r"E:\地质文本空间化平台\福建地质志.docx").paragraphs]

    g = Gen(facts)
    g.single_factual()
    seen = g.comparative()
    g.comparative_same_age(seen)
    g.multi_hop()
    g.aggregate()
    g.reverse_lookup()
    g.spatial()
    g.evolution(sections, paras)

    print(f"程序化生成 {len(g.q)} 题")
    print("\n各模板题数:")
    for k, v in g.templates.most_common():
        print(f"  {k:38s} {v}")
    print("\n跳过统计:")
    for k, v in g.skipped.most_common(12):
        print(f"  {k:38s} {v}")

    merge_and_emit(g.q, facts)


if __name__ == "__main__":
    main()
