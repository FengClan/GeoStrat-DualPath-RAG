#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D4 规模扩充：从《福建地质志》63 单位原文描述，规则提取结构化事实卡（v2）。

核心原则：宁缺毋滥。每条事实 = 单位描述内部的原文句（可追溯、无 LLM 幻觉、
不会张冠李戴）。字段提取不准就留空，不出该维度题，绝不凑错误 gold 答案。

- 输入 : kb_storage/formation_descriptions.json
- 输出 : kb_storage/formation_facts.json
"""
import sys, os, json, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC = os.path.join(PKG_DIR, "kb_storage", "formation_descriptions.json")
OUT = os.path.join(PKG_DIR, "kb_storage", "formation_facts.json")

AGE = {
    "天井坪岩组": "古元古代", "麻源岩群": "古元古代", "大金山岩组": "古元古代",
    "南山岩组": "古元古代", "迪口岩组": "古元古代", "桃溪岩组": "古元古代",
    "东岩组": "青白口纪—南华纪", "大岭组": "青白口纪—南华纪", "龙北溪组": "青白口纪—南华纪",
    "黄潭组": "青白口纪—南华纪", "下峰组": "青白口纪—南华纪",
    "楼前组": "震旦纪", "西溪组": "震旦纪", "楼子坝组": "早震旦世",
    "南岩组": "晚震旦世", "黄连组": "晚震旦世",
    "林田组": "早—晚寒武世", "东坑口组": "末寒武世", "魏坊组": "早奥陶世", "罗峰溪组": "中—晚奥陶世",
    "天瓦崠组": "晚泥盆世", "桃子坑组": "晚泥盆世",
    "林地组": "早石炭世", "经畲组": "中石炭世", "老虎洞组": "中石炭世",
    "船山组": "早二叠世", "栖霞组": "早二叠世", "泉上组": "中二叠世",
    "文笔山组": "中二叠世", "童子岩组": "中二叠世", "翠屏山组": "晚二叠世",
    "罗坑组": "晚二叠世", "长兴组": "晚二叠世",
    "溪口组": "早三叠世", "安仁组": "中三叠世",
    "大坑村组": "晚三叠世", "文宾山组": "晚三叠世", "焦坑组": "晚三叠世",
    "象牙群": "早侏罗世", "下村组": "早侏罗世", "藩坑组": "早侏罗世",
    "梨山组": "早侏罗世", "漳平组": "中侏罗世",
    "长林组": "晚侏罗世", "南园组": "晚侏罗世—早白垩世",
    "小溪组": "早白垩世", "下渡组": "早白垩世", "坂头组": "早白垩世", "吉山组": "早白垩世",
    "石帽山群": "早白垩世晚期", "黄坑组": "早白垩世晚期", "寨下组": "早白垩世晚期",
    "石牛山组": "晚白垩世", "赤石群": "晚白垩世", "均口组": "晚白垩世",
    "沙县组": "晚白垩世", "崇安组": "晚白垩世",
    "佛昙组": "新近纪", "天宝组": "早更新世", "同安组": "中更新世",
    "龙海组": "晚更新世", "东山组": "早全新世", "长乐组": "中—晚全新世",
}

SKIP_RE = re.compile(
    r'^\s*(?:[0-9]{1,3}[.．、]|[0-9]+\s*\.?\s*m\b|[0-9.]+\s*m?\s*$|'
    r'(?:上段|下段|中段|第一段|第二段|第三段|第四段|掩盖|未见顶|未见底)\s*)$')


def _clean(s):
    return re.sub(r'\s+', ' ', s).strip()


def _clip(s, n=200):
    s = _clean(s)
    return s if len(s) <= n else s[:n] + "…"


def _sent_around(seg, pos):
    """取 pos 所在句子（前后句号/分号界定）"""
    start = max(seg.rfind('。', 0, pos), seg.rfind('；', 0, pos), seg.rfind('．', 0, pos))
    start = start + 1 if start >= 0 else 0
    end = seg.find('。', pos)
    end2 = seg.find('；', pos)
    if end == -1 or (end2 != -1 and end2 < end):
        end = end2
    if end == -1:
        end = len(seg)
    return seg[start:end]


def _find(narrative, kws, n=200):
    """在叙述段中，找第一个含任一关键词的段，取该关键词所在句。kws 按优先级排列。"""
    for seg in narrative:
        for kw in kws:
            pos = seg.find(kw)
            if pos >= 0:
                sent = _sent_around(seg, pos)
                sent = _clip(sent, n)
                if len(sent) >= 6:
                    return sent
    return ""


def extract(u):
    segs = [l.strip() for l in u["text"].split("\n") if l.strip()]
    narrative = [s for s in segs if not SKIP_RE.match(s)]

    facts = {
        "name": u["name"], "symbol": u["symbol"],
        "age": AGE.get(u["name"], ""),
        # 岩性：优先"为一套/指一套/岩性"，其次"由…组成"
        "lithology": _find(narrative, ["为一套", "指一套", "岩石组合为", "岩性为", "岩性由", "由下粗上细", "为一套海相"]),
        # 分布：优先"主要分布于/分布于/见于/出露于/广布于"
        "distribution": _find(narrative, ["主要分布于", "仅分布于", "分布于", "广布于", "主要出露于", "出露于", "见于", "分布在"]),
        # 化石：优先具体组合/动物群/拉丁名，排除"产海相化石"泛词
        "fossil": _find(narrative, ["动物群", "植物组合", "化石组合", "组合为", "组合", "产", "采获", "化石有", "含化石", "菊石", "腕足", "双壳类", "笔石", "孢粉"]),
        # 厚度：优先"总厚度/总厚"
        "thickness": _find(narrative, ["总厚度", "总厚", "厚度大于", "厚大于", "厚度约", "厚度"]),
        # 命名/创名
        "naming": _find(narrative, ["所创", "创建", "创名", "命名地点", "一名源于", "创名地点"]),
        # 同义名/曾称
        "synonym": _find(narrative, ["曾称", "原称", "原名", "旧称", "即称", "沿用"]),
        # 接触关系
        "contact": _find(narrative, ["不整合", "整合接触", "整合于", "上覆地层", "下伏地层", "平行不整合", "整合覆于", "不整合覆盖"]),
    }
    return facts


def main():
    descs = json.load(open(DESC, encoding="utf-8"))
    facts = [extract(u) for u in descs]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    for field in ["lithology", "distribution", "fossil", "thickness", "naming", "synonym", "contact"]:
        n = sum(1 for x in facts if x.get(field))
        print(f"  {field:14s}: {n}/{len(facts)} 命中")
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    main()
