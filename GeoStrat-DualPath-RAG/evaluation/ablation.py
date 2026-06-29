# evaluation/ablation.py — 消融实验
"""
三种消融变体 + 完整方法 (Dual-path) 对比:
  D (完整):       自适应分类 + KG噪声过滤 + 引用标记
  D - 自适应分类:  固定 0.5:0.5 权重，不分类
  D - KG噪声过滤:  不过滤通用分布节点
  D - 细粒度引用:  不要求[Dn]/[Gn]引用标记

用法:
  python evaluation/ablation.py --model qwen3.5-plus --test-set test_sets/test_10q.json
  python evaluation/ablation.py --model qwen3.5-plus --test-set test_sets/test_set_expanded.json
"""

import time, json, os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator


class AblationRunner(BaselineRunner):
    """消融实验运行器 — 继承 BaselineRunner，复用基础方法"""

    ABLATION_NAMES = {
        "dual_path":       "D. 完整方法 (Proposed)",
        "abl_no_classify": "D − 自适应分类 (固定0.5:0.5)",
        "abl_no_filter":   "D − KG噪声过滤",
        "abl_no_citation": "D − 细粒度引用",
    }

    # ================================================================
    # D. 完整方法 (复刻 dual_path，增加显式引用指令)
    # ================================================================

    def _runner_full(self, question_dict):
        """完整双路自适应融合 — 与 baselines.py 中的 _runner_dual_path 一致"""
        from services.model_router import get_model_response
        from services.query_classifier import get_retrieval_config

        t0 = time.time()

        # 查询分类
        query_type = "FACTUAL"
        v_ratio, g_ratio, max_hops = 0.6, 0.4, 2
        try:
            config = get_retrieval_config(question_dict['question'])
            query_type = config.get("query_type", "FACTUAL")
            v_ratio = config.get("vector_ratio", 0.6)
            g_ratio = config.get("graph_ratio", 0.4)
            max_hops = config.get("max_hops", 2)
        except Exception as e:
            print(f"[Ablation-Full] Classification failed: {e}")

        # 并行检索
        vector_docs = self._retrieve_vectors(question_dict['question'])
        kg_facts = self._retrieve_kg(question_dict['question'], max_hops)

        # 构建 prompt
        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.get('content', '')[:400]}"
            for i, d in enumerate(vector_docs[:3])
        ])

        prompt = self._build_prompt(
            question_dict['question'], query_type,
            kg_facts, v_context, require_citations=True
        )

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts:
            citations.append("福建地质志.docx")

        return {
            "answer": answer, "retrieved_docs": vector_docs,
            "citations": citations, "extracted_triples": [],
            "contrast_dimensions": [], "query_type": query_type,
            "latency_ms": latency
        }

    # ================================================================
    # 消融变体1: D − 自适应分类 (固定0.5:0.5)
    # ================================================================

    def _runner_no_classify(self, question_dict):
        """取消查询分类，所有问题统一使用 0.5:0.5 固定权重"""
        from services.model_router import get_model_response

        t0 = time.time()

        # 固定使用0.5:0.5, max_hops=2, FACTUAL模板
        max_hops = 2
        vector_docs = self._retrieve_vectors(question_dict['question'])
        kg_facts = self._retrieve_kg(question_dict['question'], max_hops)

        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.get('content', '')[:400]}"
            for i, d in enumerate(vector_docs[:3])
        ])

        prompt = self._build_prompt(
            question_dict['question'], "FACTUAL",
            kg_facts, v_context, require_citations=True
        )

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts:
            citations.append("福建地质志.docx")

        return {
            "answer": answer, "retrieved_docs": vector_docs,
            "citations": citations, "extracted_triples": [],
            "contrast_dimensions": [], "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # 消融变体2: D − KG噪声过滤
    # ================================================================

    def _runner_no_filter(self, question_dict):
        """取消KG噪声过滤，保留所有边（含"福建省""地层小区"等通用节点）"""
        from services.model_router import get_model_response
        from services.query_classifier import get_retrieval_config

        t0 = time.time()

        query_type = "FACTUAL"
        max_hops = 2
        try:
            config = get_retrieval_config(question_dict['question'])
            query_type = config.get("query_type", "FACTUAL")
            max_hops = config.get("max_hops", 2)
        except Exception:
            pass

        vector_docs = self._retrieve_vectors(question_dict['question'])

        # KG检索（不使用噪声过滤）
        kg_facts_unfiltered = ""
        try:
            from services.kg_service import retrieve_graph_context
            kg_result = retrieve_graph_context(
                question_dict['question'], self.kb_id, max_hops=max_hops
            )
            if isinstance(kg_result, tuple) and len(kg_result) == 2:
                kg_edges = kg_result[0] if isinstance(kg_result[0], list) else []
            elif isinstance(kg_result, list):
                kg_edges = kg_result
            else:
                kg_edges = []
            # 关键区别: 此处 filter_noise=False
            kg_facts_unfiltered = self._format_kg_facts(kg_edges, filter_noise=False)
        except Exception as e:
            print(f"[Ablation-NoFilter] KG failed: {e}")

        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.get('content', '')[:400]}"
            for i, d in enumerate(vector_docs[:3])
        ])

        prompt = self._build_prompt(
            question_dict['question'], query_type,
            kg_facts_unfiltered, v_context, require_citations=True
        )

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts_unfiltered:
            citations.append("福建地质志.docx")

        return {
            "answer": answer, "retrieved_docs": vector_docs,
            "citations": citations, "extracted_triples": [],
            "contrast_dimensions": [], "query_type": query_type,
            "latency_ms": latency
        }

    # ================================================================
    # 消融变体3: D − 细粒度引用
    # ================================================================

    def _runner_no_citation(self, question_dict):
        """取消[Dn]/[Gn]引用标记要求，仅保留文档级来源列表"""
        from services.model_router import get_model_response
        from services.query_classifier import get_retrieval_config

        t0 = time.time()

        query_type = "FACTUAL"
        max_hops = 2
        try:
            config = get_retrieval_config(question_dict['question'])
            query_type = config.get("query_type", "FACTUAL")
            max_hops = config.get("max_hops", 2)
        except Exception:
            pass

        vector_docs = self._retrieve_vectors(question_dict['question'])
        kg_facts = self._retrieve_kg(question_dict['question'], max_hops)

        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.get('content', '')[:400]}"
            for i, d in enumerate(vector_docs[:3])
        ])

        # 关键区别: require_citations=False
        prompt = self._build_prompt(
            question_dict['question'], query_type,
            kg_facts, v_context, require_citations=False
        )

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts:
            citations.append("福建地质志.docx")

        return {
            "answer": answer, "retrieved_docs": vector_docs,
            "citations": citations, "extracted_triples": [],
            "contrast_dimensions": [], "query_type": query_type,
            "latency_ms": latency
        }

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _retrieve_vectors(self, question):
        """检索向量通道"""
        docs = []
        try:
            raw = self._retrieve_docs(question, self.kb_id, top_k=5)
            if raw:
                docs = [
                    {"id": d.metadata.get("source_file", f"doc{i}"),
                     "score": d.metadata.get("score", 0.5),
                     "content": d.page_content[:200]}
                    for i, d in enumerate(raw)
                ]
        except Exception as e:
            print(f"[Ablation] Vector retrieval failed: {e}")
        return docs

    def _retrieve_kg(self, question, max_hops):
        """检索图谱通道（带噪声过滤）"""
        try:
            from services.kg_service import retrieve_graph_context
            kg_result = retrieve_graph_context(question, self.kb_id, max_hops=max_hops)
            if isinstance(kg_result, tuple) and len(kg_result) == 2:
                kg_edges = kg_result[0] if isinstance(kg_result[0], list) else []
            elif isinstance(kg_result, list):
                kg_edges = kg_result
            else:
                kg_edges = []
            return self._format_kg_facts(kg_edges, filter_noise=False)
        except Exception as e:
            print(f"[Ablation] KG retrieval failed: {e}")
            return ""

    def _build_prompt(self, question, query_type, kg_facts, v_context, require_citations=True):
        """构建自适应提示词"""
        citation_instruction = (
            "请在回答中标注信息来源：文献证据处标注[D1][D2]，图谱证据处标注[G1][G2]。"
            if require_citations else ""
        )

        system_instruction = (
            "你是一位专业地质学家。你必须综合以下两类信息回答问题：\n"
            "1. 知识图谱事实 — 结构化提取的地层属性，为权威数据，优先采信\n"
            "2. 参考文献 — 地质志描述文本，用于补充细节和上下文\n"
            "重要：如果图谱事实与文献描述或你的训练知识存在冲突，以图谱事实为准。"
        )

        if query_type == 'REASONING':
            prompt = f"""{system_instruction}

## 知识图谱推理路径（结构化关系链，优先采信）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（辅助证据）
{v_context if v_context else '暂无相关文献'}

## 问题
{question}

请先分析图谱中的推理链条，再结合文献证据给出综合判断。{citation_instruction}"""

        elif query_type == 'COMPARATIVE':
            prompt = f"""{system_instruction}

## 知识图谱事实（各地层结构化属性，用于分维度对比）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（详细描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question}

请从岩性、年代、分布等维度进行系统对比。图谱提供对比框架，文献补充具体描述。{citation_instruction}"""

        else:
            prompt = f"""{system_instruction}

## 知识图谱事实（结构化地层属性，权威数据）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（补充描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question}

请以图谱中的结构化事实为主要依据回答。文献内容用于补充细节，如果文献描述的是局部剖面特征，请与图谱中的整体属性加以区分。{citation_instruction}"""

        return prompt

    # ================================================================
    # 运行全部消融实验
    # ================================================================

    def run_all_ablations(self, test_set, use_llm_judge=True):
        """运行完整方法 + 3种消融变体"""
        evaluator = Evaluator(model=self.model)

        variants = {
            "dual_path":       self._runner_full,
            "abl_no_classify": self._runner_no_classify,
            "abl_no_filter":   self._runner_no_filter,
            "abl_no_citation": self._runner_no_citation,
        }

        all_results = {}
        comparison_table = {}

        for var_key, runner in variants.items():
            name = self.ABLATION_NAMES[var_key]
            print(f"\n{'='*60}")
            print(f"[Ablation] Running: {name}")
            print(f"{'='*60}")

            results = evaluator.run_full_evaluation(
                test_set, runner, use_llm_judge=use_llm_judge
            )
            all_results[var_key] = results

            for metric, value in results["overall_scores"].items():
                if metric not in comparison_table:
                    comparison_table[metric] = {}
                comparison_table[metric][var_key] = value

        # 确定最优
        overall_scores = {k: v["overall_scores"]["overall_score"]
                         for k, v in all_results.items()}
        winner = max(overall_scores, key=overall_scores.get)

        final_results = {
            "ablations": all_results,
            "comparison": comparison_table,
            "winner": winner,
            "winner_name": self.ABLATION_NAMES[winner],
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model": self.model
        }

        # 保存
        os.makedirs("./evaluation/results", exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = f"./evaluation/results/ablation_{timestamp}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n[Ablation] Results saved to: {result_path}")
        return final_results, result_path

    def print_summary(self, results):
        """打印消融对比摘要"""
        comp = results.get("comparison", {})
        variants = ["dual_path", "abl_no_classify", "abl_no_filter", "abl_no_citation"]

        key_metrics = [
            "overall_score", "qa_correctness", "qa_faithfulness",
            "qa_completeness", "citation_precision"
        ]
        metric_labels = {
            "overall_score": "综合得分", "qa_correctness": "正确性",
            "qa_faithfulness": "忠实度", "qa_completeness": "完整性",
            "citation_precision": "引用精确率"
        }

        print(f"\n{'='*90}")
        print(f"{'指标':<14}", end="")
        for v in variants:
            print(f"{self.ABLATION_NAMES[v]:>22}", end="")
        print()

        print("-" * 102)

        for metric in key_metrics:
            if metric not in comp:
                continue
            print(f"{metric_labels.get(metric, metric):<14}", end="")
            scores = []
            for v in variants:
                val = comp[metric].get(v, 0)
                scores.append(val)
                print(f"{val:>22.4f}", end="")

            # 标注最优和最差
            best = max(scores)
            worst = min(scores)
            for i, s in enumerate(scores):
                if s == best and metric != 'avg_latency_ms':
                    marker = " ← 最优"
                elif s == worst and metric != 'avg_latency_ms' and best != worst:
                    marker = " ↓"
                else:
                    marker = ""
            if best != worst:
                print()
            else:
                print()

        print(f"\n最优方法: {results.get('winner_name', '?')}")


def main():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--test-set", default="test_sets/test_10q.json")
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    print(f"[Ablation] Model: {args.model}")
    print(f"[Ablation] Test set: {args.test_set}")

    with open(args.test_set, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"[Ablation] Questions: {len(test_set['questions'])}")

    runner = AblationRunner(model=args.model)
    results, path = runner.run_all_ablations(
        test_set, use_llm_judge=not args.no_llm_judge
    )
    runner.print_summary(results)


if __name__ == '__main__':
    main()
