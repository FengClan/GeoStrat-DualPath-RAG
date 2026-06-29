# evaluation/baselines.py — 多基线对比评估
"""
四种检索策略的系统对比:
  A. Direct LLM (无RAG) — 纯LLM回答
  B. Vector RAG only — 仅向量检索 + LLM
  C. Graph RAG only — 仅图谱检索 + LLM
  D. Dual-path Adaptive Fusion (Proposed) — 双路检索 + 自适应融合

每种基线被封装为 SystemRunner callable，接受 question dict 返回系统输出。
"""

import time
import json
import os
import sys

# 确保能导入项目服务
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class BaselineRunner:
    """管理多基线对比评估的运行器"""

    BASELINE_NAMES = {
        "direct_llm": "纯LLM (无RAG)",
        "vector_rag": "向量RAG",
        "graph_rag": "图谱RAG",
        "dual_path": "双路自适应融合 (Proposed)"
    }

    def __init__(self, model="qwen3.5-plus", kb_id="all", results_dir=None):
        self.model = model
        self.kb_id = kb_id
        self.results_dir = results_dir or "./evaluation/results"

    # ================================================================
    # Baseline A: Direct LLM (no retrieval)
    # ================================================================

    def _runner_direct_llm(self, question_dict):
        """纯LLM回答，无任何检索"""
        from services.model_router import get_model_response

        t0 = time.time()
        prompt = f"""你是一位专业地质学家。请回答以下问题：

问题：{question_dict['question']}

请给出专业、准确的回答。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        return {
            "answer": answer,
            "retrieved_docs": [],
            "citations": [],
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Helper: 直接调用 ChromaDB 检索，返回 Document 列表
    # ================================================================

    def _retrieve_docs(self, query, kb_id, top_k=5):
        """绕过 retrieve_from_vectordb（返回元组），直接获取Document对象"""
        from rag_core import vectordb
        try:
            if kb_id and kb_id != "all":
                return vectordb.similarity_search(query, k=top_k, filter={"kb_id": kb_id})
            else:
                return vectordb.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[Baseline] Vector search failed: {e}")
            return []

    # ================================================================
    # Baseline B: Vector RAG only
    # ================================================================

    # ================================================================
    # Helper: 将KG原始边列表格式化为可读的结构化事实
    # ================================================================

    def _format_kg_facts(self, kg_edges, filter_noise=False):
        """将 retrieve_graph_context 返回的边列表格式化为分组事实"""
        if not kg_edges:
            return ""
        if isinstance(kg_edges, str):
            return kg_edges

        # kg_edges is list of "A --[REL]--> B" strings
        from collections import defaultdict
        entity_facts = defaultdict(lambda: defaultdict(list))

        # 无意义的通用节点（对具体问题帮助不大）
        GENERIC_TARGETS = {'福建省', '地层小区', '中国', '福建省区域', '福建省地质'}
        # 优先展示的关系类型
        PRIORITY_RELS = {'COMPOSED_OF', 'DATED_AS', 'BELONGS_TO', 'SAME_AS',
                        'CONTAINS_FOSSIL', 'CONTACTS', 'HAS_THICKNESS'}

        for edge_text in kg_edges:
            import re
            m = re.match(r'(.+?)\s*--\[(.+?)\]-+>\s*(.+)', edge_text)
            if m:
                src, rel, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                # 过滤无意义分布节点
                if filter_noise and rel == 'DISTRIBUTED_IN' and tgt in GENERIC_TARGETS:
                    continue
                entity_facts[src][rel].append(tgt)

        lines = []
        for entity, rels in entity_facts.items():
            # 优先展示核心关系
            priority_lines = []
            other_lines = []
            for rel, targets in rels.items():
                rel_label = {
                    'COMPOSED_OF': '岩石组成', 'DATED_AS': '地质年代',
                    'DISTRIBUTED_IN': '分布区域', 'BELONGS_TO': '所属单位',
                    'SAME_AS': '同义名称', 'CONTAINS_FOSSIL': '含化石',
                    'HAS_THICKNESS': '厚度', 'CONTACTS': '接触关系',
                    'MEASURED_AT': '实测剖面'
                }.get(rel, rel)
                line = f"  - {rel_label}: {', '.join(targets[:8])}"
                if rel in PRIORITY_RELS:
                    priority_lines.append(line)
                else:
                    other_lines.append(line)

            if priority_lines or other_lines:
                lines.append(f"【{entity}】")
                lines.extend(priority_lines)
                lines.extend(other_lines)
                lines.append("")
        return '\n'.join(lines)

    def _runner_vector_rag(self, question_dict):
        """仅使用向量检索增强"""
        from services.model_router import get_model_response

        t0 = time.time()
        retrieved_docs = []
        context_text = ""

        try:
            docs = self._retrieve_docs(question_dict['question'], self.kb_id, top_k=5)
            if docs:
                retrieved_docs = [
                    {"id": d.metadata.get("source_file", f"doc{i}"),
                     "score": d.metadata.get("score", 0.5),
                     "content": d.page_content[:200]}
                    for i, d in enumerate(docs)
                ]
                context_text = "\n\n".join([
                    f"[文献片段 {i+1}] {d.page_content[:500]}"
                    for i, d in enumerate(docs)
                ])
        except Exception as e:
            print(f"[Baseline-Vector] Retrieval failed: {e}")

        prompt = f"""你是一位专业地质学家。你必须严格基于以下参考文献回答问题，不得依赖训练数据中的记忆。

注意：如果参考文献中缺乏相关信息，请明确说明"文献中未找到相关信息"，不要编造。

## 参考文献
{context_text if context_text else '暂无相关文献'}

## 问题
{question_dict['question']}

请基于文献给出专业回答，标注出处。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in retrieved_docs))

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "citations": citations,
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Baseline C: Graph RAG only
    # ================================================================

    def _runner_graph_rag(self, question_dict):
        """仅使用知识图谱检索增强"""
        from services.model_router import get_model_response
        from services.kg_service import retrieve_graph_context

        t0 = time.time()
        kg_edges = []
        kg_facts = ""

        try:
            result = retrieve_graph_context(
                question_dict['question'], self.kb_id,
                max_hops=2
            )
            if isinstance(result, tuple) and len(result) == 2:
                kg_edges = result[0] if isinstance(result[0], list) else []
            elif isinstance(result, list):
                kg_edges = result
            kg_facts = self._format_kg_facts(kg_edges)
        except Exception as e:
            print(f"[Baseline-Graph] KG retrieval failed: {e}")

        prompt = f"""你是一位专业地质学家。你必须严格基于以下知识图谱结构化事实回答问题。图谱中的事实为权威数据，如与你的训练知识冲突，以图谱事实为准。

## 知识图谱事实（结构化地层属性）
{kg_facts if kg_facts else '暂无相关知识图谱信息'}

## 问题
{question_dict['question']}

请基于图谱中的结构化事实给出专业回答。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        return {
            "answer": answer,
            "retrieved_docs": [],
            "citations": ["福建地质志.docx"] if kg_facts else [],
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Baseline D: Dual-path Adaptive Fusion (Proposed)
    # ================================================================

    def _runner_dual_path(self, question_dict):
        """双路检索 + 自适应融合 (论文提出的方法)"""
        from services.model_router import get_model_response
        from services.query_classifier import get_retrieval_config

        t0 = time.time()

        # Step 1: 查询分类
        query_type = "FACTUAL"
        v_ratio, g_ratio, max_hops = 0.6, 0.4, 2
        try:
            config = get_retrieval_config(question_dict['question'])
            query_type = config.get("query_type", "FACTUAL")
            v_ratio = config.get("vector_ratio", 0.6)
            g_ratio = config.get("graph_ratio", 0.4)
            max_hops = config.get("max_hops", 2)
        except Exception as e:
            print(f"[Baseline-Dual] Query classification failed: {e}")

        # Step 2: 并行检索
        vector_docs = []
        kg_facts = ""

        # 向量检索
        try:
            docs = self._retrieve_docs(question_dict['question'], self.kb_id, top_k=5)
            if docs:
                vector_docs = [
                    {"id": d.metadata.get("source_file", f"doc{i}"),
                     "score": d.metadata.get("score", 0.5),
                     "content": d.page_content[:200]}
                    for i, d in enumerate(docs)
                ]
        except Exception as e:
            print(f"[Baseline-Dual] Vector retrieval failed: {e}")

        # 图谱检索
        try:
            from services.kg_service import retrieve_graph_context
            kg_result = retrieve_graph_context(
                question_dict['question'], self.kb_id,
                max_hops=max_hops
            )
            if isinstance(kg_result, tuple) and len(kg_result) == 2:
                kg_edges = kg_result[0] if isinstance(kg_result[0], list) else []
            elif isinstance(kg_result, list):
                kg_edges = kg_result
            else:
                kg_edges = []
            kg_facts = self._format_kg_facts(kg_edges)
        except Exception as e:
            print(f"[Baseline-Dual] KG retrieval failed: {e}")

        # Step 3: 融合构建Prompt
        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.get('content', '')[:400]}"
            for i, d in enumerate(vector_docs[:3])
        ])

        # 查询类型自适应 Prompt 模板
        # 核心原则：图谱事实为结构化权威数据，文献为补充描述
        system_instruction = (
            "你是一位专业地质学家。你必须综合以下两类信息回答问题：\n"
            "1. 知识图谱事实 — 结构化提取的地层属性，为权威数据，优先采信\n"
            "2. 参考文献 — 地质志描述文本，用于补充细节和上下文\n"
            "重要：如果图谱事实与文献描述或你的训练知识存在冲突，以图谱事实为准。"
        )

        if query_type == 'REASONING':
            # 图谱主导：先展示推理路径，再补充文献
            prompt = f"""{system_instruction}

## 知识图谱推理路径（结构化关系链，优先采信）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（辅助证据）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

请先分析图谱中的推理链条，再结合文献证据给出综合判断。在图谱证据处标注[G]、文献证据处标注[D]。"""

        elif query_type == 'COMPARATIVE':
            # 均衡模式：图谱提供结构化对比维度，文献提供细节
            prompt = f"""{system_instruction}

## 知识图谱事实（各地层结构化属性，用于分维度对比）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（详细描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

请从岩性、年代、分布等维度进行系统对比。图谱提供对比框架，文献补充具体描述。"""

        else:
            # FACTUAL / SPATIAL：图谱为主、文献为辅
            prompt = f"""{system_instruction}

## 知识图谱事实（结构化地层属性，权威数据）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（补充描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

请以图谱中的结构化事实为主要依据回答。文献内容用于补充细节，如果文献描述的是局部剖面特征，请与图谱中的整体属性加以区分。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        # 收集所有引用
        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts:
            citations.append("福建地质志.docx")

        return {
            "answer": answer,
            "retrieved_docs": vector_docs,
            "citations": citations,
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": query_type,
            "latency_ms": latency
        }

    # ================================================================
    # 运行所有基线对比
    # ================================================================

    def get_runner(self, baseline_name):
        """根据基线名称返回对应的runner函数"""
        runners = {
            "direct_llm": self._runner_direct_llm,
            "vector_rag": self._runner_vector_rag,
            "graph_rag": self._runner_graph_rag,
            "dual_path": self._runner_dual_path,
        }
        return runners.get(baseline_name, self._runner_direct_llm)

    def run_all_baselines(self, test_set, use_llm_judge=True):
        """
        运行所有基线对比评估。
        Returns: {
            "baselines": {name: {overall_scores, ...}},
            "comparison": {metric: {baseline: score}},
            "winner": str
        }
        """
        from .evaluator import Evaluator
        evaluator = Evaluator(model=self.model, results_dir=self.results_dir)

        all_results = {}
        comparison_table = {}

        for baseline_key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
            name = self.BASELINE_NAMES[baseline_key]
            print(f"\n{'='*60}")
            print(f"[Baseline] Running: {name}")
            print(f"{'='*60}")

            runner = self.get_runner(baseline_key)
            results = evaluator.run_full_evaluation(
                test_set, runner, use_llm_judge=use_llm_judge
            )

            all_results[baseline_key] = results

            # 构建对比表
            for metric, value in results["overall_scores"].items():
                if metric not in comparison_table:
                    comparison_table[metric] = {}
                comparison_table[metric][baseline_key] = value

        # 确定最优方法 (基于overall_score)
        overall_scores = {k: v["overall_scores"]["overall_score"]
                         for k, v in all_results.items()}
        winner = max(overall_scores, key=overall_scores.get)

        final_results = {
            "baselines": all_results,
            "comparison": comparison_table,
            "winner": winner,
            "winner_name": self.BASELINE_NAMES[winner],
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model": self.model
        }

        # 保存结果
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(self.results_dir, f"baseline_comparison_{timestamp}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n[Baseline] Results saved to: {result_path}")
        return final_results

    def print_comparison_summary(self, results):
        """打印对比摘要"""
        comp = results.get("comparison", {})
        key_metrics = [
            "overall_score", "qa_correctness", "qa_faithfulness",
            "retrieval_recall@5", "retrieval_mrr",
            "citation_precision", "citation_recall",
            "avg_latency_ms"
        ]

        print(f"\n{'='*80}")
        print(f"{'指标':<25}", end="")
        for key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
            print(f"{self.BASELINE_NAMES[key]:>20}", end="")
        print()

        print("-" * 105)

        for metric in key_metrics:
            if metric not in comp:
                continue
            print(f"{metric:<25}", end="")
            for key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
                val = comp[metric].get(key, 0)
                if "latency" in metric:
                    print(f"{val:>20.1f}", end="")
                else:
                    print(f"{val:>20.4f}", end="")
            print()

        print(f"{'='*80}")
        print(f"Best overall: {results.get('winner_name', 'Unknown')}")
        print(f"{'='*80}")
