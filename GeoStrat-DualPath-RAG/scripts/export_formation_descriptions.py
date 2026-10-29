#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D4 规模扩充辅助：从《福建地质志.docx》导出 63 个岩石地层单位的原文描述段落。

用途：作为「专家核准事实库」的底料，供独立测试集（1000 题）生成时逐单位引用原文，
      确保 gold 答案有原文背书、不依赖实验知识图谱/向量库。

- 输入 : E:\\地质文本空间化平台\\福建地质志.docx
- 输出 : kb_storage/formation_descriptions.json（{name, symbol, paragraphs, text}）
- 说明 : 单位清单与起始段落索引由专家人工核对（含符号带空格/希腊字母 ∈/无编号/嵌段等特殊格式）
"""
import sys, os, json, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DOCX = r"E:\地质文本空间化平台\福建地质志.docx"
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG_DIR, "kb_storage", "formation_descriptions.json")

# (标准名, 地层符号, 起始段落索引) —— 按章节顺序，符号用于溯源
UNITS = [
    ("天井坪岩组", "Pt1t", 112), ("麻源岩群", "Pt1M", 150), ("大金山岩组", "Pt1d", 154),
    ("南山岩组", "Pt1n", 221), ("迪口岩组", "Pt1dk", 254), ("桃溪岩组", "Pt1tx", 287),
    ("东岩组", "Pt31-2d", 380), ("大岭组", "Pt31-2dl", 408), ("龙北溪组", "Pt31-2l", 460),
    ("黄潭组", "Pt31-2h", 526), ("下峰组", "Pt31-2x", 559),
    ("楼前组", "Pt33l", 642), ("西溪组", "Pt33x", 686), ("楼子坝组", "Pt33al", 732),
    ("南岩组", "Pt33bn", 850), ("黄连组", "Pt33bh", 885), ("林田组", "∈1-3l", 961),
    ("东坑口组", "∈4d", 999), ("魏坊组", "O1w", 1036), ("罗峰溪组", "O2-3l", 1065),
    ("天瓦崠组", "D3t", 1189), ("桃子坑组", "D3tz", 1193), ("林地组", "C1l", 1208),
    ("经畲组", "C2j", 1233), ("老虎洞组", "C2lh", 1264), ("船山组", "P1c", 1343),
    ("栖霞组", "P1q", 1363), ("泉上组", "P2qs", 1394), ("文笔山组", "P2w", 1403),
    ("童子岩组", "P2t", 1424), ("翠屏山组", "P3cp", 1542), ("罗坑组", "P3l", 1584),
    ("长兴组", "P3c", 1638), ("溪口组", "T1x", 1694), ("安仁组", "T2a", 1734),
    ("大坑村组", "T3d", 1879), ("文宾山组", "T3w", 1984), ("焦坑组", "T3j", 2074),
    ("象牙群", "J1X", 2179), ("下村组", "J1x", 2183), ("藩坑组", "J1f", 2244),
    ("梨山组", "J1l", 2324), ("漳平组", "J2z", 2386),
    ("长林组", "J3c", 2555), ("南园组", "J3-K1n", 2580), ("小溪组", "K1x", 2669),
    ("下渡组", "K1xd", 2748), ("坂头组", "K1b", 2795), ("吉山组", "K1j", 2888),
    ("石帽山群", "K1S", 2924), ("黄坑组", "K1h", 2929), ("寨下组", "K1z", 2962),
    ("石牛山组", "K2sh", 3036), ("赤石群", "K2C", 3070), ("均口组", "K2j", 3073),
    ("沙县组", "K2s", 3090), ("崇安组", "K2c", 3127),
    ("佛昙组", "N1f", 3236), ("天宝组", "Q1pt", 3298), ("同安组", "Qp2t", 3301),
    ("龙海组", "Qp3l", 3321), ("东山组", "Qh1d", 3375), ("长乐组", "Qh2-3c", 3412),
]
END_GLOBAL = 3459  # 第八篇 起始，最后一个单位描述到此为止


def main():
    from docx import Document
    d = Document(DOCX)
    paras = [p.text.strip() for p in d.paragraphs]

    # 计算每个单位的结束索引 = 下一个单位起始（最后一个用 END_GLOBAL）
    starts = [u[2] for u in UNITS]
    result = []
    for k, (name, sym, start) in enumerate(UNITS):
        end = starts[k + 1] if k + 1 < len(UNITS) else END_GLOBAL
        seg = [t for t in paras[start:end] if t]
        text = "\n".join(seg)
        result.append({
            "name": name, "symbol": sym,
            "start_idx": start, "end_idx": end,
            "para_count": len(seg),
            "text": text,
        })
        print(f"{name:6s} [{sym}] 段{start:4d}-{end:4d}  非空段 {len(seg):3d}  字数 {len(text):5d}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    total_chars = sum(r["para_count"] for r in result)
    print(f"\n共 {len(result)} 单位，总非空段 {total_chars}")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
