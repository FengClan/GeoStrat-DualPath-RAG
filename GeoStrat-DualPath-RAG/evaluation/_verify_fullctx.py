# -*- coding: utf-8 -*-
"""离线验证：dual_path_fullctx 与 dual_path 是否**只差文献上下文量**这一个变量。

做法：把检索与生成全部打桩（不发任何 API），截下两个 runner 实际送进模型的 prompt，
逐字对比。期望：
  - 图谱盲区问题：两者 prompt **完全相同**（该分支本来就用 5/500，对照组不该改它）
  - 图谱启用问题：两者只有「## 参考文献」段落不同，且对照组给的是 5 条 × 500 字
跑完可删。
"""
import sys, os, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding="utf-8")

from types import SimpleNamespace

import services.model_router as mr
import services.kg_service as kgs
import services.query_classifier as qc
from evaluation.baselines import BaselineRunner, _graph_is_blind

CAPTURED = []


def _fake_model_response(prompt, model, *a, **kw):
    CAPTURED.append(prompt)
    return "（打桩回答）"


class _Doc:
    """模拟 langchain Document"""
    def __init__(self, text, i):
        self.page_content = text
        self.metadata = {"source_file": "福建地质志.docx", "chunk_index": i, "score": 0.9}


# 8 条长文献：足够区分 [:3]/[:5] 与 [:400]/[:500]
DOCS = [_Doc(f"文献{i}正文" + "甲" * 800, i) for i in range(8)]

mr.get_model_response = _fake_model_response
kgs.retrieve_graph_context = lambda q, kb, max_hops=2: (
    ["南园组 --[COMPOSED_OF]--> 凝灰岩",
     "南园组 --[DATED_AS]--> 晚侏罗世",
     "梨山组 --[DISTRIBUTED_IN]--> 福建省"], None)
qc.get_retrieval_config = lambda q: {"query_type": "COMPARATIVE",
                                     "vector_ratio": 0.6, "graph_ratio": 0.4, "max_hops": 2}

runner = BaselineRunner(model="qwen3.5-plus")
runner._retrieve_docs = lambda q, kb, top_k=5: list(DOCS)


def prompt_of(fn, question):
    CAPTURED.clear()
    fn({"question": question, "id": "T-1", "query_type": "COMPARATIVE"})
    return CAPTURED[-1]


def ref_block(p):
    """截出「## 参考文献」那一段"""
    m = re.search(r"## 参考文献[^\n]*\n(.*?)\n\n## 问题", p, re.S)
    return m.group(1) if m else ""


def n_docs(block):
    return len(re.findall(r"\[文献 \d+\]", block))


def doc_len(block):
    m = re.search(r"\[文献 1\] (.*)", block)
    return len(m.group(1)) if m else 0


cases = [
    ("图谱启用（对比题）", "请对比南园组与梨山组的岩性差异。"),
    ("图谱盲区（问厚度）", "南园组的厚度是多少？"),
]

print("=" * 78)
print("验证：dual_path_fullctx 相对 dual_path 是否只改了「图谱启用时的文献上下文量」")
print("=" * 78)

ok = True
for label, q in cases:
    blind = _graph_is_blind(q)
    p_a = prompt_of(runner._runner_dual_path, q)
    p_b = prompt_of(runner._runner_dual_path_fullctx, q)
    ra, rb = ref_block(p_a), ref_block(p_b)

    print(f"\n【{label}】 graph_blind={blind}")
    print(f"   dual_path        : 文献 {n_docs(ra)} 条 × {doc_len(ra)} 字，prompt {len(p_a)} 字符")
    print(f"   dual_path_fullctx: 文献 {n_docs(rb)} 条 × {doc_len(rb)} 字，prompt {len(p_b)} 字符")

    if blind:
        same = (p_a == p_b)
        print(f"   → 盲区分支应两者完全相同: {'✓' if same else '✗ 不该有差异'}")
        ok &= same
    else:
        # 除参考文献段外，其余必须逐字相同
        same_rest = p_a.replace(ra, "") == p_b.replace(rb, "")
        bigger = n_docs(rb) > n_docs(ra) and doc_len(rb) > doc_len(ra)
        print(f"   → 除「参考文献」段外其余逐字相同: {'✓' if same_rest else '✗'}")
        print(f"   → 对照组文献确实更多更长:      {'✓' if bigger else '✗'}")
        ok &= same_rest and bigger

# 盲区时两臂给的都是 5 条 —— 确认对照组没把盲区分支也改了
print()
print("=" * 78)
print("验证：dual_path_nogate 是否只删了门控")
print("=" * 78)

q_blind = "南园组的厚度是多少？"
q_on = "请对比南园组与梨山组的岩性差异。"
p_base_blind = prompt_of(runner._runner_dual_path, q_blind)
p_nog_blind = prompt_of(runner._runner_dual_path_nogate, q_blind)
p_base_on = prompt_of(runner._runner_dual_path, q_on)
p_nog_on = prompt_of(runner._runner_dual_path_nogate, q_on)

print(f"\n【盲区题】{q_blind}")
print(f"   dual_path      : {len(p_base_blind)} 字符，含『知识图谱』段: "
      f"{'知识图谱事实' in p_base_blind}")
print(f"   dual_path_nogate: {len(p_nog_blind)} 字符，含『知识图谱』段: "
      f"{'知识图谱事实' in p_nog_blind}")
gate_removed = ('知识图谱事实' not in p_base_blind) and ('知识图谱事实' in p_nog_blind)
print(f"   → 门控确实被删（图谱重新进入 prompt）: {'✓' if gate_removed else '✗'}")

print(f"\n【非盲区题】{q_on}")
same_on = (p_base_on == p_nog_on)
print(f"   → 与 dual_path 逐字相同: {'✓' if same_on else '✗ 不该有差异'}")

print("\n" + "=" * 78)
print("线程安全：full_context 是函数参数，不落任何 self 属性")
print("   _dual_path_impl 签名:", runner._dual_path_impl.__code__.co_varnames[
    :runner._dual_path_impl.__code__.co_argcount])
print("   runner 上有 full_context 属性吗:",
      hasattr(runner, "full_context"), "（应为 False）")

ok &= gate_removed and same_on
print("\n总体:", "✓ 全部通过" if ok else "✗ 有断言失败")
