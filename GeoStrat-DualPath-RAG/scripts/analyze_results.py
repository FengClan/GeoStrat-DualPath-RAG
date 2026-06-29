# scripts/analyze_results.py — 从实验结果JSON生成论文表格
"""
用法: python scripts/analyze_results.py evaluation/results/baseline_comparison_XXXX.json
"""
import json, sys
from collections import defaultdict

BASELINE_LABELS = {
    "direct_llm": "纯LLM",
    "vector_rag": "向量RAG",
    "graph_rag": "图谱RAG",
    "dual_path": "双路自适应融合"
}

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def table_1_main_results(data):
    """表1: 主实验结果对比"""
    print_header("表1: 主实验结果对比 (Overall Results)")
    comp = data['comparison']
    metrics = ['overall_score', 'qa_correctness', 'qa_faithfulness', 'qa_completeness',
               'citation_precision', 'citation_recall', 'retrieval_recall@5', 'retrieval_mrr']
    metric_labels = {
        'overall_score': '综合得分', 'qa_correctness': '正确性', 'qa_faithfulness': '忠实度',
        'qa_completeness': '完整性', 'citation_precision': '引用精确率', 'citation_recall': '引用召回率',
        'retrieval_recall@5': 'Recall@5', 'retrieval_mrr': 'MRR'
    }

    baselines = ['direct_llm', 'vector_rag', 'graph_rag', 'dual_path']
    print(f"{'指标':<16}", end='')
    for b in baselines:
        print(f"{BASELINE_LABELS[b]:>18}", end='')
    print()

    for m in metrics:
        if m not in comp: continue
        print(f"{metric_labels.get(m, m):<16}", end='')
        scores = []
        for b in baselines:
            s = comp[m].get(b, 0)
            scores.append(s)
            print(f"{s:>18.4f}", end='')
        # 标注最优
        best_val = max(scores)
        best_idx = scores.index(best_val)
        marker = ['', '']
        if best_val > 0 and m != 'avg_latency_ms':
            print(f"  ← {BASELINE_LABELS[baselines[best_idx]]}", end='')
        print()

def table_2_by_query_type(data):
    """表2: 按查询类型分组"""
    print_header("表2: 按查询类型分组 (By Query Type)")
    baselines = ['direct_llm', 'vector_rag', 'graph_rag', 'dual_path']
    query_types = ['FACTUAL', 'COMPARATIVE', 'REASONING', 'SPATIAL']

    for qt in query_types:
        print(f"\n--- {qt} ---")
        print(f"{'指标':<16}", end='')
        for b in baselines:
            print(f"{BASELINE_LABELS[b]:>18}", end='')
        print()

        for metric_key in ['llm_judge_score', 'correctness', 'faithfulness']:
            print(f"{metric_key:<16}", end='')
            for b in baselines:
                if b not in data['baselines']: continue
                results = data['baselines'][b]['per_question_results']
                vals = [r['qa_metrics'].get(metric_key, 0)
                       for r in results
                       if r.get('query_type') == qt and 'qa_metrics' in r]
                avg = sum(vals)/len(vals) if vals else 0
                print(f"{avg:>18.4f}", end='')
            print()

def table_3_per_question(data):
    """展示部分具体问题的对比"""
    print_header("表3: 典型案例对比 (前5题)")
    baselines = ['direct_llm', 'vector_rag', 'graph_rag', 'dual_path']

    for i, q in enumerate(data['baselines']['direct_llm']['per_question_results'][:5]):
        qid = q['id']
        qtype = q.get('query_type', '?')
        question = q.get('question', '?')
        print(f"\n[{qid}] [{qtype}] {question}")
        print(f"{'Baseline':<18}{'Correct':>8}{'Faith':>8}{'Complete':>8}{'Judge':>8}")
        for b in baselines:
            if b not in data['baselines']: continue
            for r in data['baselines'][b]['per_question_results']:
                if r['id'] == qid and 'qa_metrics' in r:
                    m = r['qa_metrics']
                    print(f"{BASELINE_LABELS[b]:<18}{m.get('correctness',0):>8.2f}{m.get('faithfulness',0):>8.2f}{m.get('completeness',0):>8.2f}{m.get('llm_judge_score',0):>8.2f}")
                    break

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_results.py <result_json_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")
    data = load(path)

    print(f"\nModel: {data.get('model', '?')}")
    print(f"Winner: {data.get('winner_name', '?')}")
    print(f"Evaluated at: {data.get('evaluated_at', '?')}")

    table_1_main_results(data)
    table_2_by_query_type(data)
    table_3_per_question(data)

    print_header("论文LaTeX模板")
    print_tex_table(data)

def print_tex_table(data):
    """生成LaTeX表格"""
    comp = data['comparison']
    metrics = ['overall_score', 'qa_correctness', 'qa_faithfulness', 'qa_completeness',
               'citation_precision', 'citation_recall', 'retrieval_recall@5', 'retrieval_mrr']
    metric_labels = {
        'overall_score': 'Overall', 'qa_correctness': 'Correctness', 'qa_faithfulness': 'Faithfulness',
        'qa_completeness': 'Completeness', 'citation_precision': 'Cite-P', 'citation_recall': 'Cite-R',
        'retrieval_recall@5': 'R@5', 'retrieval_mrr': 'MRR'
    }
    baselines = ['direct_llm', 'vector_rag', 'graph_rag', 'dual_path']

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Main experimental results. Best scores in \\textbf{bold}.}")
    cols = 'l' + 'c' * len(baselines)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\hline")
    print(" & ".join(['Metric'] + [BASELINE_LABELS[b] for b in baselines]) + " \\\\")
    print("\\hline")

    for m in metrics:
        if m not in comp: continue
        scores = [comp[m].get(b, 0) for b in baselines]
        best = max(scores)
        cells = [metric_labels.get(m, m)]
        for s in scores:
            if s == best and m != 'avg_latency_ms':
                cells.append(f"\\textbf{{{s:.4f}}}")
            else:
                cells.append(f"{s:.4f}")
        print(" & ".join(cells) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == '__main__':
    main()
