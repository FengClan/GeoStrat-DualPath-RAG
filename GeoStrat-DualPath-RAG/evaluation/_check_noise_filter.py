# -*- coding: utf-8 -*-
"""离线核查：KG 噪声过滤开关实际会改变多少题（不发任何 API）。

背景：`filter_noise` 参数默认 False，且全代码库无一处传 True。
若主表报的提出方法与消融的「D − 噪声过滤」臂用的是同一个设置，
则两臂在 KG 路径上完全相同，R2#9 要复现的"去掉过滤掉 6.4%"量不出来。

本脚本对 contested 300 题实测：打开过滤 vs 关闭过滤，prompt 里的图谱事实差多少。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding="utf-8")

from evaluation.baselines import BaselineRunner
from services.kg_service import retrieve_graph_context
from services.query_classifier import get_retrieval_config

CKPT = "evaluation/results_full_20260914/progress_dual_path.json"


def main():
    r = BaselineRunner(model="qwen3.5-plus")
    ts = json.load(open("test_sets/test_set_honest_test.json", encoding="utf-8"))
    qs = {q["id"]: q for q in ts["questions"]}
    ids = [x["id"] for x in json.load(open(CKPT, encoding="utf-8"))["per_question_results"]
           if "qa_metrics" in x]

    n_diff = n_same = n_empty = 0
    saved_chars = 0
    examples = []
    for i in ids:
        qtext = qs[i]["question"]
        try:
            mh = get_retrieval_config(qtext).get("max_hops", 2)
        except Exception:
            mh = 2
        e = retrieve_graph_context(qtext, r.kb_id, max_hops=mh)
        if isinstance(e, tuple) and len(e) == 2:
            e = e[0] if isinstance(e[0], list) else []
        elif not isinstance(e, list):
            e = []
        if not e:
            n_empty += 1
        on = r._format_kg_facts(e, filter_noise=True)
        off = r._format_kg_facts(e, filter_noise=False)
        if on != off:
            n_diff += 1
            saved_chars += len(off) - len(on)
            if len(examples) < 3:
                gone = set(off.split("\n")) - set(on.split("\n"))
                examples.append((qtext[:44], sorted(gone)[:3]))
        else:
            n_same += 1

    n = len(ids)
    print("=" * 72)
    print(f"contested {n} 题，KG 通道离线实测（打开过滤 filter_noise=True vs 关闭）")
    print("=" * 72)
    print(f"  打开过滤会改变图谱事实的题：{n_diff:>4}  ({n_diff/n*100:5.1f}%)")
    print(f"  完全不受影响：            {n_same:>4}  ({n_same/n*100:5.1f}%)")
    print(f"  图谱检索为空：            {n_empty:>4}")
    if n_diff:
        print(f"  被影响题上平均少 {saved_chars/n_diff:.0f} 字符")
    print("\n示例（被删掉的行）：")
    for q, gone in examples:
        print(f"  Q: {q}")
        for g in gone:
            print(f"     - {g}")


if __name__ == "__main__":
    main()
