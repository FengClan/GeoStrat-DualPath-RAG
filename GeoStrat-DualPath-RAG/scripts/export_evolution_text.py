#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出《福建地质志》第八篇「福建地质发展史」（段 3459-3915）。

⚠️ 这个区间的实际构成比标题看起来复杂得多，本脚本按实测结构切分：

    段 3459        第八篇 福建地质发展史（内联标题）
    段 3465-3556   第三十四章…第三十九章 各时期演化叙述（6 章）
    段 3556…       第四十章 新生代时期（正文）
    段 3568        结语                      ← 独立内容，按专题再切
    段 3722        参考文献                   ← 必须整段剔除

    实测：第四十章 名义上占全篇 73%，但其中 94 段是【结语】、244 段是
    【参考文献】；真正的新生代叙述只有 11 段 1254 字。全篇正文合计 25 999 字，
    另有 13 257 字是参考文献。若不剔除，生成器会拿书目条目当 gold 出题。

出题时注意（见 manifest / 论文）：
    KG 的 3075 条三元组里「构造演化」「岩浆活动」「变质作用」「地质发展史」
    出现 0 次，而向量库里有 22/33/76 次 → 本篇导出的题对 graph_rag 通道
    结构性不可答，对 vector/dual_path 可答。必须带 source_section 标记，
    评估时单独成表或按来源分组报告，否则会系统性抬高 proposed 方法。

- 输入 : 原始 docx
- 输出 : kb_storage/evolution_sections.json
"""
import sys, os, json, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG_DIR, "kb_storage", "evolution_sections.json")
DOCX = r"E:\地质文本空间化平台\福建地质志.docx"

PIAN_START = 3459                      # 实测：内联标题「第八篇 福建地质发展史…」
CHAP_RE = re.compile(r'^第([一二三四五六七八九十百]+)章\s*(.+)$')
TOPIC_RE = re.compile(r'^（([一二三四五六七八九十]+)）\s*(.+)$')   # （一）地层与沉积作用
GROUP_RE = re.compile(r'^([一二三四五六七八九十]+)、\s*(.+)$')     # 一、主要地质成果
REF_RE = re.compile(r'^(主要)?参考文献\s*$')


def _cut_references(paras, start, end):
    """返回 (正文结束段号, 文献起始段号或 None)。"""
    for i in range(start, end):
        if REF_RE.match(paras[i]):
            return i, i
    return end, None


def main():
    from docx import Document
    paras = [p.text.strip() for p in Document(DOCX).paragraphs]
    end = len(paras)

    assert paras[PIAN_START].startswith("第八篇"), \
        f"段 {PIAN_START} 不是第八篇标题，实际为: {paras[PIAN_START][:30]!r}"

    # 篇内章标题（限定区间：正文有大量「第X章」交叉引用，不限定会误匹配）
    chapters = []
    for i in range(PIAN_START, end):
        m = CHAP_RE.match(paras[i])
        if m and len(paras[i]) < 60:
            chapters.append((i, m.group(1), m.group(2).strip()))

    # 结语 / 参考文献边界
    concl_start = ref_start = None
    for i in range(PIAN_START, end):
        if paras[i] == "结语" and concl_start is None:
            concl_start = i
        if REF_RE.match(paras[i]) and ref_start is None:
            ref_start = i
    body_end = ref_start if ref_start is not None else end

    sections, sid = [], 0

    def add(kind, chapter_no, title, a, b):
        nonlocal sid
        seg = [t for t in paras[a:b] if t]
        if not seg:
            return
        sid += 1
        sections.append({
            "section_id": f"evolution-{sid:02d}",
            "kind": kind,                    # narrative | conclusion
            "chapter_no": chapter_no,
            "title": title,
            "start_idx": a, "end_idx": b,
            "para_count": len(seg), "text": "\n".join(seg),
        })

    # 1) 篇首导语
    first_chap = chapters[0][0] if chapters else body_end
    add("narrative", "序", "福建地质发展史（篇首导语）", PIAN_START, first_chap)

    # 2) 各章叙述正文（到结语为止）
    narr_stop = concl_start if concl_start else body_end
    for k, (idx, no, title) in enumerate(chapters):
        stop = chapters[k + 1][0] if k + 1 < len(chapters) else narr_stop
        if idx >= narr_stop:
            break
        add("narrative", f"第{no}章", title, idx, min(stop, narr_stop))

    # 3) 结语：按「一、」「（一）」两级专题切开
    if concl_start:
        topics = [(i, m.group(1), m.group(2).strip())
                  for i in range(concl_start, body_end)
                  for m in [TOPIC_RE.match(paras[i])] if m and len(paras[i]) < 60]
        groups = [(i, m.group(2).strip())
                  for i in range(concl_start, body_end)
                  for m in [GROUP_RE.match(paras[i])] if m and len(paras[i]) < 60]

        if topics:
            for k, (idx, no, title) in enumerate(topics):
                stop = topics[k + 1][0] if k + 1 < len(topics) else body_end
                # 归属到最近的上一级「一、」「二、」分组
                grp = ""
                for gidx, gtitle in groups:
                    if gidx < idx:
                        grp = gtitle
                add("conclusion", "结语", f"结语·{grp + '·' if grp else ''}{title}",
                    idx + 1, stop)
        else:
            add("conclusion", "结语", "结语", concl_start + 1, body_end)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    tot = sum(len(s["text"]) for s in sections)
    n_narr = sum(len(s["text"]) for s in sections if s["kind"] == "narrative")
    n_conc = sum(len(s["text"]) for s in sections if s["kind"] == "conclusion")
    for s in sections:
        print(f"  {s['kind']:<11}{s['chapter_no']:<8}{s['title'][:30]:<32}"
              f"段{s['start_idx']:>5}-{s['end_idx']:<5}{s['para_count']:>3}段 {len(s['text']):>6}字")
    print(f"\n共 {len(sections)} 节，正文合计 {tot} 字（叙述 {n_narr} + 结语 {n_conc}）")
    if ref_start is not None:
        ref_chars = sum(len(t) for t in paras[ref_start:end] if t)
        print(f"已剔除参考文献：段 {ref_start}-{end}，{ref_chars} 字（若不剔除会被当 gold 出题）")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
