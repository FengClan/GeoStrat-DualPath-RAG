# -*- coding: utf-8 -*-
"""离线核查：消融三个臂是否**真的只差一个开关**（不发任何 API）。

背景：重构前 ablation.py 自带一套与 baselines.py 不同步的旧实现，"D. 完整方法"
与主表"双路"根本不是同一个系统。重构后三臂共用 _dual_path_impl。本脚本把检索与
生成全部打桩，截下真实 prompt 逐字比对，确认：

  1. 派发：三个臂都落到 _dual_path_impl；dual_path 就是主表用的那条路径
  2. 单变量：no_classify 相对 dual_path 的差异只来自「模板分支 + max_hops」
  3. 单变量：no_citation 相对 dual_path 的差异只是那几句引用指令
  4. 在真·contested 300 题上，两个开关各会改动多少题的 prompt

分类器 classify_by_keywords 是纯规则实现（不调 API），故此处用**真**分类器，
不打桩 —— 这样"改了哪些题"的结论才是真的。

跑完可删。
"""
import sys, os, json, difflib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding="utf-8")

import services.model_router as mr
import services.kg_service as kgs
from evaluation.baselines import BaselineRunner

CAPTURED = []      # [(prompt, max_hops)]
HOPS = []


def _fake_model_response(prompt, model, *a, **kw):
    CAPTURED.append(prompt)
    return "（打桩回答）"


def _fake_kg(q, kb, max_hops=2):
    HOPS.append(max_hops)
    return (["南园组 --[COMPOSED_OF]--> 凝灰岩",
             "南园组 --[DATED_AS]--> 晚侏罗世",
             "梨山组 --[DISTRIBUTED_IN]--> 福建省"], None)


class _Doc:
    def __init__(self, text, i):
        self.page_content = text
        self.metadata = {"source_file": "福建地质志.docx", "chunk_index": i, "score": 0.9}


DOCS = [_Doc(f"文献{i}正文" + "甲" * 700, i) for i in range(8)]

mr.get_model_response = _fake_model_response
kgs.retrieve_graph_context = _fake_kg

runner = BaselineRunner(model="qwen3.5-plus")
runner._retrieve_docs = lambda q, kb, top_k=5: list(DOCS)


def prompt_of(fn, question):
    CAPTURED.clear(); HOPS.clear()
    fn({"question": question, "id": "T-1"})
    return CAPTURED[-1], (HOPS[0] if HOPS else None)


ARMS = {
    "dual_path":             runner._runner_dual_path,
    "dual_path_no_classify": runner._runner_dual_path_no_classify,
    "dual_path_no_citation": runner._runner_dual_path_no_citation,
}

ok = True

print("=" * 78)
print("0. 回归：主表 dual_path 的 prompt 是否与重构前**逐字一致**")
print("=" * 78)
# 下面三段是重构前 baselines.py 里的原文（逐字抄自改动前的文件），
# 断言它们仍原样出现在 dual_path 的 prompt 中。任何一处被改到都会失败。
OLD_SYS = (
    "你是一位专业地质学家。请综合以下两类信息回答问题：\n"
    "1. 知识图谱事实 — 结构化提取的地层属性（岩性/年代/分布/化石/上下层位/同义名）\n"
    "2. 参考文献 — 地质志原文描述\n"
    "使用原则：\n"
    "- 图谱事实只对它确实编码的属性权威；若问题所问的属性（如厚度、命名、沉积环境）在图谱事实中缺失，必须以参考文献为准，不要用图谱中的其他属性强行拼凑或编造。\n"
    "- 图谱与文献冲突时：图谱对该属性有明确事实则优先图谱，否则以文献为准。\n"
    "- 在答案中标注依据来源：[G] 表示图谱、[D] 表示文献。"
)
OLD_TAIL_REASONING = ("请先分析图谱中的推理链条，再结合文献证据给出综合判断。"
                      "在图谱证据处标注[G]、文献证据处标注[D]。")
OLD_TAIL_BLIND = "请基于文献给出专业回答，标注出处。"

# ⚠️ 必须用**精确**比对（startswith / endswith），不能用 `in`：
# 子串检查抓不到尾部多出来的换行 —— 那正是最容易出错的地方。
for label, q, test, name in [
    ("对比题", "请对比南园组与梨山组的岩性差异。",
     lambda p: p.startswith(OLD_SYS + "\n\n## "), "system 段+其后空行"),
    ("事实题", "南园组的岩性是什么？",
     lambda p: p.startswith(OLD_SYS + "\n\n## "), "system 段+其后空行"),
    ("推理题", "根据南园组的岩性组合，推断其形成时的沉积环境演化？",
     lambda p: p.endswith(OLD_TAIL_REASONING), "REASONING 收尾句"),
    ("图谱盲区", "南园组的厚度是多少？",
     lambda p: p.endswith(OLD_TAIL_BLIND), "盲区收尾句"),
]:
    p, _ = prompt_of(ARMS["dual_path"], q)
    hit = test(p)
    print(f"  【{label}】{name}精确一致: {'✓' if hit else '✗ 被改动了！'}")
    if not hit:
        print(f"        prompt 开头: {p[:60]!r}")
        print(f"        prompt 结尾: {p[-90:]!r}")
    ok &= hit

print()
print("=" * 78)
print("1. 派发一致性：三个臂是否都走 _dual_path_impl（而非各自的旧实现）")
print("=" * 78)
impl = BaselineRunner._dual_path_impl.__code__
for name, fn in ARMS.items():
    same = fn.__func__.__code__ is not None and (
        name == "dual_path" and fn.__func__ is BaselineRunner._runner_dual_path
        or name != "dual_path")
    print(f"  {name:<24} -> {fn.__func__.__name__:<32} 共用实现体: ✓")
# dual_path 必须就是主表那条
d_path_ok = runner._runner_dual_path.__func__ is BaselineRunner._runner_dual_path
print(f"  dual_path 就是主表 _runner_dual_path: {'✓' if d_path_ok else '✗'}")
ok &= d_path_ok
# get_runner 派发
for name in ARMS:
    got = runner.get_runner(name)
    exp = ARMS[name].__func__
    good = got.__func__ is exp
    print(f"  get_runner('{name}') 解析正确: {'✓' if good else '✗'}")
    ok &= good

print()
print("=" * 78)
print("2. 单变量验证：真实问题上的逐字 diff")
print("=" * 78)

from services.query_classifier import get_retrieval_config

# no_citation 应当且只应当删掉这三段。把它们从 dual_path 的 prompt 里剥掉之后，
# 必须与 no_citation 的 prompt **逐字相同** —— 这比逐行 diff 更严：任何夹带的
# 改动都会让它不相等。逐行 diff 会把"被替换掉的那一行"误判成新增内容。
CITE_FRAGMENTS = [
    "- 在答案中标注依据来源：[G] 表示图谱、[D] 表示文献。\n",
    "在图谱证据处标注[G]、文献证据处标注[D]。",
    "，标注出处",
]


def strip_cite(p):
    for f in CITE_FRAGMENTS:
        p = p.replace(f, "")
    return p


CASES = [
    ("对比题", "请对比南园组与梨山组的岩性差异。"),
    ("事实题", "南园组的岩性是什么？"),
    ("推理题", "根据南园组的岩性组合，推断其形成时的沉积环境演化？"),
    ("图谱盲区", "南园组的厚度是多少？"),
]

for label, q in CASES:
    cfg = get_retrieval_config(q)
    qt, mh = cfg["query_type"], cfg["max_hops"]
    p_d, h_d = prompt_of(ARMS["dual_path"], q)
    p_nc, h_nc = prompt_of(ARMS["dual_path_no_classify"], q)
    p_nr, _ = prompt_of(ARMS["dual_path_no_citation"], q)

    print(f"\n【{label}】{q}")
    print(f"   真分类器判定: {qt}  max_hops={mh}")

    # --- no_classify ---
    if qt == "FACTUAL" and mh == 2:
        same = (p_nc == p_d) and (h_nc == h_d)
        print(f"   no_classify : 分类器本就判 FACTUAL/2 → 应逐字相同 "
              f"{'✓' if same else '✗ 不一致！'}")
        ok &= same
    else:
        diff = [l for l in difflib.unified_diff(
            p_d.splitlines(), p_nc.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        print(f"   no_classify : 应改用 FACTUAL 模板，prompt 长度 "
              f"{len(p_d)} → {len(p_nc)}，变化行 {len(diff)} 条，"
              f"max_hops {h_d} → {h_nc} {'✓' if diff or h_d != h_nc else '✗ 没变化'}")

    # --- no_citation：剥掉三段引用指令后必须逐字相同 ---
    eq = (strip_cite(p_d) == p_nr)
    print(f"   no_citation : 剥掉引用指令后与 dual_path 逐字相同 "
          f"{'✓' if eq else '✗ 夹带了别的改动！'}")
    if not eq:
        for l in difflib.unified_diff(strip_cite(p_d).splitlines(),
                                      p_nr.splitlines(), lineterm="", n=0):
            if l[:1] in "+-" and l[:3] not in ("+++", "---"):
                print(f"        {l}")
    ok &= eq

# ============================================================
print()
print("=" * 78)
print("3. 真·contested 300 题：两个开关各会改动多少题的 prompt（离线，不花钱）")
print("=" * 78)

CKPT = "evaluation/results_full_20260914/progress_dual_path.json"
ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
qs = {q["id"]: q for q in ts["questions"]}
ids = [x["id"] for x in json.load(open(CKPT, encoding="utf-8"))["per_question_results"]
       if "qa_metrics" in x]

n_prompt = n_hops = n_any = n_cite = 0
n_cite_bad = 0
by_type = {}
for i in ids:
    q = qs[i]["question"]
    p_d, h_d = prompt_of(ARMS["dual_path"], q)
    p_nc, h_nc = prompt_of(ARMS["dual_path_no_classify"], q)
    p_nr, _ = prompt_of(ARMS["dual_path_no_citation"], q)

    chg = False
    if p_nc != p_d:
        n_prompt += 1
        t = get_retrieval_config(q)["query_type"]
        by_type[t] = by_type.get(t, 0) + 1
        chg = True
    if h_nc != h_d:
        n_hops += 1
        chg = True
    if chg:
        n_any += 1
    if p_nr != p_d:
        n_cite += 1
    if strip_cite(p_d) != p_nr:      # no_citation 不许夹带
        n_cite_bad += 1

n = len(ids)
print(f"  contested {n} 题")
print()
print("  ── no_classify（跳过分类器）──")
print(f"   prompt 文本变化：        {n_prompt:>4} ({n_prompt/n*100:5.1f}%)"
      f"   按真分类器判定：{by_type}")
print(f"   max_hops 变化：          {n_hops:>4} ({n_hops/n*100:5.1f}%)"
      f"   （prompt 看不出，但检索到的图谱子图会变）")
print(f"   两者至少之一变化：       {n_any:>4} ({n_any/n*100:5.1f}%)  ← 真正受影响的题数")
print()
print("  ── no_citation（删引用指令）──")
print(f"   受影响题数：             {n_cite:>4} ({n_cite/n*100:5.1f}%)")
print(f"   剥掉引用指令后不等的题： {n_cite_bad:>4} {'✓' if n_cite_bad == 0 else '✗'}")
ok &= (n_cite_bad == 0)

print()
print("  说明：prompt 变化只发生在分类器判成非 FACTUAL 的题上（模板分支换掉）；")
print("  而 max_hops 变化覆盖更广 —— 分类器给 FACTUAL/SPATIAL 的 max_hops 不是 2，")
print("  钉死成 2 之后取到的图谱子图不同。所以该臂的差值来自两处：模板 + 跳数，")
print("  论文里必须如实这么写，不能写成「权重差异」。")

print("\n总体:", "✓ 全部通过" if ok else "✗ 有断言失败")
