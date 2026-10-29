# -*- coding: utf-8 -*-
"""离线自检 contested 分层与加权总分（不调任何 API）。跑完即可删。"""
import sys, os, json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_full_eval import stratify_contested, weighted_overall

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

d = json.load(open('test_sets/test_set_honest_test.json', encoding='utf-8'))
qs = d['questions']
full = Counter(q['query_type'] for q in qs)

s = stratify_contested(qs, 300, 42)
c = Counter(q['query_type'] for q in s)
ids = [q['id'] for q in s]
print(f'contested 300 抽得 {len(s)} 题，去重 {len(set(ids))}')
for t in ['FACTUAL', 'REASONING', 'COMPARATIVE', 'SPATIAL']:
    mark = '（全取）' if c[t] == full[t] else ''
    print(f'    {t:<12} {c[t]:>4} / {full[t]:<4} {mark}')

print('同 seed 可复现:',
      [q['id'] for q in stratify_contested(qs, 300, 42)] == ids)
print('seed=43 不同:',
      [q['id'] for q in stratify_contested(qs, 300, 43)] != ids)

# 抽样出来的题必须全部来自留出集，且不与 dev 集重叠（R2#3 的划分纪律）
dev = json.load(open('test_sets/test_set_honest_dev.json', encoding='utf-8'))
dev_ids = {q['id'] for q in dev['questions']}
test_ids = {q['id'] for q in qs}
print('抽样是否全部来自留出集:', set(ids) <= test_ids)
print('与 dev 集无重叠:', not (set(ids) & dev_ids))

print('\n--- 边界：--total 小于被争议两类之和时应当报错 ---')
try:
    stratify_contested(qs, 100, 42)
    print('  ✗ 没报错')
except SystemExit as e:
    print('  ✓', str(e).splitlines()[0])

print('\n--- 加权函数自检 ---')
fake = {'per_question_results': [
    {'query_type': 'FACTUAL', 'qa_metrics': {'correctness': 1.0, 'faithfulness': 1.0, 'completeness': 1.0}},
    {'query_type': 'SPATIAL', 'qa_metrics': {'correctness': 0.0, 'faithfulness': 0.0, 'completeness': 0.0}},
]}
print('  权重 0.9/0.1 → 期望 0.9，实得', weighted_overall(fake, {'FACTUAL': 0.9, 'SPATIAL': 0.1}))
print('  权重 0.5/0.5 → 期望 0.5，实得', weighted_overall(fake, {'FACTUAL': 0.5, 'SPATIAL': 0.5}))
fake2 = {'per_question_results': fake['per_question_results'] + [{'query_type': 'FACTUAL', 'error': 'x'}]}
print('  含异常题（应剔除）→ 仍为 0.9，实得', weighted_overall(fake2, {'FACTUAL': 0.9, 'SPATIAL': 0.1}))
print('  全空 → 期望 None，实得', weighted_overall({'per_question_results': []}, {'FACTUAL': 1.0}))
