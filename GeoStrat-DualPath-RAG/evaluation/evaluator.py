# evaluation/evaluator.py — 多维度地质系统评估引擎
"""
评估维度:
1. 抽取精度 (Extraction): precision/recall/F1 for NER, RE
2. 检索质量 (Retrieval): recall@K, MRR, hit@K
3. QA质量 (QA Quality): LLM-as-judge correctness, faithfulness, citation metrics
4. 对比一致性 (Contrast): dimension agreement, conclusion correctness
"""

import json
import math
import time
import re
from collections import defaultdict
from difflib import SequenceMatcher


class Evaluator:
    def __init__(self, model="qwen3.5-plus", results_dir=None):
        self.model = model
        self.results_dir = results_dir or "./evaluation/results"
        import os
        os.makedirs(self.results_dir, exist_ok=True)

    # ================================================================
    # 1. 抽取精度评估
    # ================================================================

    def evaluate_extraction(self, predicted, ground_truth):
        """
        评估NER/RE抽取精度。
        Args:
            predicted: [{entity, entity_type, relation, target, ...}]
            ground_truth: [{entity, entity_type, relation, target, ...}]
        Returns: {precision, recall, f1, details}
        """
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "details": "no ground truth"}

        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "details": "no predictions"}

        # 实体评估
        pred_entities = set()
        gt_entities = set()
        for p in predicted:
            entity = p.get('entity', '') or p.get('head', '') or p.get('subject', '')
            if entity:
                pred_entities.add(entity.strip())
        for g in ground_truth:
            entity = g.get('entity', '') or g.get('head', '') or g.get('subject', '')
            if entity:
                gt_entities.add(entity.strip())

        # 宽松匹配：子串匹配也算中
        entity_tp = 0
        for pe in pred_entities:
            for ge in gt_entities:
                if pe in ge or ge in pe:
                    entity_tp += 1
                    break

        entity_precision = entity_tp / len(pred_entities) if pred_entities else 1.0
        entity_recall = entity_tp / len(gt_entities) if gt_entities else 1.0
        entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0

        # 三元组评估
        pred_triples = set()
        gt_triples = set()
        for p in predicted:
            h = p.get('entity', '') or p.get('head', '') or p.get('subject', '')
            r = p.get('relation', '') or p.get('predicate', '')
            t = p.get('target', '') or p.get('object', '')
            if h and r and t:
                pred_triples.add((h.strip(), r.strip(), t.strip()))
        for g in ground_truth:
            h = g.get('entity', '') or g.get('head', '') or g.get('subject', '')
            r = g.get('relation', '') or g.get('predicate', '')
            t = g.get('target', '') or g.get('object', '')
            if h and r and t:
                gt_triples.add((h.strip(), r.strip(), t.strip()))

        triple_tp = 0
        for pt in pred_triples:
            for gt in gt_triples:
                ph, pr, pp = pt
                gh, gr, gp = gt
                if ((ph in gh or gh in ph) and
                    (pr in gr or gr in pr) and
                    (pp in gp or gp in pp)):
                    triple_tp += 1
                    break

        triple_precision = triple_tp / len(pred_triples) if pred_triples else 1.0
        triple_recall = triple_tp / len(gt_triples) if gt_triples else 1.0
        triple_f1 = 2 * triple_precision * triple_recall / (triple_precision + triple_recall) if (triple_precision + triple_recall) > 0 else 0.0

        return {
            "entity": {"precision": round(entity_precision, 4), "recall": round(entity_recall, 4),
                       "f1": round(entity_f1, 4)},
            "triple": {"precision": round(triple_precision, 4), "recall": round(triple_recall, 4),
                       "f1": round(triple_f1, 4)},
            "pred_count": len(pred_triples),
            "gt_count": len(gt_triples),
            "entity_pred_count": len(pred_entities),
            "entity_gt_count": len(gt_entities)
        }

    # ================================================================
    # 2. 检索质量评估
    # ================================================================

    def evaluate_retrieval(self, retrieved_docs, relevant_ids, k_values=[1, 3, 5, 10]):
        """
        评估检索质量。
        Args:
            retrieved_docs: 排序后的检索结果 [{id, score, content, ...}]
            relevant_ids: 标注的相关文档ID集合
            k_values: 评估的截断位置
        Returns: {recall@{k}, MRR, hit@{k}, ...}
        """
        if not relevant_ids:
            return {"recall@5": 0.0, "mrr": 0.0, "hit@5": 0.0, "details": "no relevance judgments"}

        if not retrieved_docs:
            return {"recall@5": 0.0, "mrr": 0.0, "hit@5": 0.0, "details": "no retrieved documents"}

        retrieved_id_list = [d.get('id', d.get('source', '')) for d in retrieved_docs]
        relevant_set = set(relevant_ids)

        results = {}

        # Recall@K
        for k in k_values:
            retrieved_at_k = set(retrieved_id_list[:k])
            relevant_retrieved = retrieved_at_k & relevant_set
            results[f"recall@{k}"] = round(len(relevant_retrieved) / len(relevant_set), 4)

        # MRR (Mean Reciprocal Rank)
        for rank, doc_id in enumerate(retrieved_id_list, 1):
            if doc_id in relevant_set:
                results["mrr"] = round(1.0 / rank, 4)
                results["first_relevant_rank"] = rank
                break
        else:
            results["mrr"] = 0.0
            results["first_relevant_rank"] = None

        # Hit@K (二值: 是否至少命中一个相关文档)
        for k in k_values:
            retrieved_at_k = set(retrieved_id_list[:k])
            results[f"hit@{k}"] = 1.0 if (retrieved_at_k & relevant_set) else 0.0

        # NDCG@K (使用相关性二值化)
        for k in k_values:
            dcg = 0.0
            idcg = 0.0
            for i in range(min(k, len(retrieved_id_list))):
                rel = 1.0 if retrieved_id_list[i] in relevant_set else 0.0
                dcg += rel / math.log2(i + 2)
            for i in range(min(k, len(relevant_set))):
                idcg += 1.0 / math.log2(i + 2)
            results[f"ndcg@{k}"] = round(dcg / idcg, 4) if idcg > 0 else 0.0

        return results

    # ================================================================
    # 3. QA质量评估 (LLM-as-Judge)
    # ================================================================

    def evaluate_qa(self, question, predicted_answer, ground_truth, key_points=None,
                    supporting_sources=None, predicted_citations=None, use_llm=True):
        """
        综合评估QA质量。

        Returns:
            {
                "correctness": 0-1 semantic correctness score,
                "key_point_coverage": fraction of key points covered,
                "faithfulness": 0-1 faithfulness to sources,
                "citation_precision": 0-1,
                "citation_recall": 0-1,
                "llm_judge_score": 0-1,
                "llm_judge_reason": str
            }
        """
        results = {}

        # 3a. Key Point Coverage (基于关键词匹配)
        if key_points:
            covered = 0
            for kp in key_points:
                if kp in predicted_answer:
                    covered += 1
                else:
                    # 尝试部分匹配
                    for word in kp:
                        if word in predicted_answer:
                            covered += 0.5
                            break
            results["key_point_coverage"] = round(min(covered / len(key_points), 1.0), 4)
        else:
            results["key_point_coverage"] = 1.0

        # 3b. Citation Metrics
        if predicted_citations is not None and supporting_sources is not None:
            pred_sources = set(predicted_citations) if predicted_citations else set()
            gt_sources = set(supporting_sources) if supporting_sources else set()

            if pred_sources:
                tp = len(pred_sources & gt_sources)
                results["citation_precision"] = round(tp / len(pred_sources), 4)
            else:
                # 无预测引用但存在Ground Truth来源 → 精确率为0
                results["citation_precision"] = 0.0 if gt_sources else 1.0

            if gt_sources:
                tp = len(pred_sources & gt_sources)
                results["citation_recall"] = round(tp / len(gt_sources), 4)
            else:
                results["citation_recall"] = 1.0
        else:
            results["citation_precision"] = 1.0
            results["citation_recall"] = 1.0

        # 3c. LLM-as-Judge (语义正确性 + 忠实度)
        if use_llm:
            llm_results = self._llm_judge_eval(question, predicted_answer, ground_truth)
            results.update(llm_results)
        else:
            # 快速规则评估
            results["correctness"] = self._rule_based_correctness(predicted_answer, ground_truth)
            results["faithfulness"] = 1.0
            results["llm_judge_score"] = results["correctness"]
            results["llm_judge_reason"] = "Rule-based evaluation"

        return results

    def _llm_judge_eval(self, question, predicted_answer, ground_truth):
        """使用LLM作为裁判评估答案质量"""
        from services.model_router import get_model_response

        prompt = f"""你是一位严格的地质学评估专家。请评估以下问答的质量。

## 问题
{question}

## 标准答案 (Ground Truth)
{ground_truth}

## 系统回答 (Predicted)
{predicted_answer}

## 评估维度 (每项0-10分)

1. **Correctness (正确性)**: 系统回答是否在事实上正确？与标准答案的一致性如何？(0=完全错误, 10=完全正确)
2. **Completeness (完整性)**: 系统回答是否涵盖了标准答案中的关键信息？(0=严重遗漏, 10=完整覆盖)
3. **Faithfulness (忠实性)**: 系统回答中是否有与已知事实相矛盾的内容（幻觉）？(0=严重幻觉, 10=完全忠实)
4. **Coherence (连贯性)**: 系统回答的表达是否清晰连贯、逻辑合理？(0=混乱, 10=清晰)
5. **Domain Accuracy (领域准确性)**: 地质术语使用是否准确？地层关系描述是否正确？(0=术语错误, 10=完全准确)

## 输出格式 (严格JSON)
```json
{{
    "correctness": 0-10,
    "completeness": 0-10,
    "faithfulness": 0-10,
    "coherence": 0-10,
    "domain_accuracy": 0-10,
    "overall": 0-10,
    "reason": "简短的评估理由（1-2句话）"
}}
```

仅返回JSON。"""
        try:
            raw = get_model_response(prompt, self.model)
            cleaned = raw.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            scores = json.loads(cleaned)

            return {
                "correctness": float(scores.get("correctness", 5)) / 10.0,
                "completeness": float(scores.get("completeness", 5)) / 10.0,
                "faithfulness": float(scores.get("faithfulness", 5)) / 10.0,
                "coherence": float(scores.get("coherence", 5)) / 10.0,
                "domain_accuracy": float(scores.get("domain_accuracy", 5)) / 10.0,
                "llm_judge_score": float(scores.get("overall", 5)) / 10.0,
                "llm_judge_reason": scores.get("reason", "")
            }
        except Exception as e:
            print(f"[Evaluator] LLM judge failed: {e}")
            return {
                "correctness": 0.5, "completeness": 0.5,
                "faithfulness": 0.5, "coherence": 0.5,
                "domain_accuracy": 0.5, "llm_judge_score": 0.5,
                "llm_judge_reason": f"Evaluation failed: {e}"
            }

    def _rule_based_correctness(self, predicted, ground_truth):
        """基于规则的快速正确性评估 (字符串相似度 + 关键词匹配)"""
        # 计算文本相似度
        sim = SequenceMatcher(None, predicted, ground_truth).ratio()

        # 提取中文词汇进行关键词覆盖率计算
        cn_pattern = re.compile(r'[一-鿿]{2,}')
        pred_words = set(cn_pattern.findall(predicted))
        gt_words = set(cn_pattern.findall(ground_truth))

        if gt_words:
            coverage = len(pred_words & gt_words) / len(gt_words)
        else:
            coverage = 1.0

        return round(0.3 * sim + 0.7 * coverage, 4)

    # ================================================================
    # 4. 地层对比一致性评估
    # ================================================================

    def evaluate_contrast(self, predicted_dimensions, expected_dimensions):
        """
        评估地层对比的一致性。

        Args:
            predicted_dimensions: [{"label": str, "col_a": {...}, "col_b": {...}}]
            expected_dimensions: [{"label": str, "col_a": {...}, "col_b": {...}}]

        Returns: {"dimension_agreement": 0-1, "conclusion_correctness": 0-1, ...}
        """
        if not expected_dimensions:
            return {"dimension_agreement": 0.0, "details": "no ground truth"}

        if not predicted_dimensions:
            return {"dimension_agreement": 0.0, "details": "no predictions"}

        # 维度名称匹配
        pred_labels = {d.get("label", ""): d for d in predicted_dimensions}
        exp_labels = {d.get("label", ""): d for d in expected_dimensions}

        matched = 0
        dimension_scores = []
        for exp_label, exp_dim in exp_labels.items():
            # 查找匹配的预测维度
            best_match = None
            for pred_label, pred_dim in pred_labels.items():
                if exp_label in pred_label or pred_label in exp_label:
                    best_match = pred_dim
                    break

            if best_match:
                score = self._compare_dimension(best_match, exp_dim)
                dimension_scores.append(score)
                matched += 1
            else:
                dimension_scores.append(0.0)

        agreement = round(sum(dimension_scores) / len(dimension_scores), 4) if dimension_scores else 0.0

        return {
            "dimension_agreement": agreement,
            "dimensions_matched": f"{matched}/{len(exp_labels)}",
            "per_dimension_scores": dimension_scores,
            "dimension_names": list(exp_labels.keys())
        }

    def _compare_dimension(self, pred_dim, exp_dim):
        """比较单个维度的内容相似度"""
        # 提取文本特征
        def extract_texts(dim, key):
            side = dim.get(key, {})
            if isinstance(side, str):
                return [side]
            texts = []
            for field in ['tags_common', 'tags_diff', 'description', 'conclusion']:
                val = side.get(field, '')
                if isinstance(val, list):
                    texts.extend(val)
                elif val:
                    texts.append(val)
            return texts

        pred_texts = extract_texts(pred_dim, 'col_a') + extract_texts(pred_dim, 'col_b')
        exp_texts = extract_texts(exp_dim, 'col_a') + extract_texts(exp_dim, 'col_b')

        pred_str = ' '.join(pred_texts)
        exp_str = ' '.join(exp_texts)

        return round(SequenceMatcher(None, pred_str, exp_str).ratio(), 4)

    # ================================================================
    # 综合评估运行器
    # ================================================================

    def run_full_evaluation(self, test_set, system_runner, use_llm_judge=True):
        """
        运行完整评估流程。

        Args:
            test_set: dict with "questions" list
            system_runner: callable(question_dict) -> {
                "answer": str,
                "retrieved_docs": [{id, score, content}],
                "citations": [str],
                "extracted_triples": [{entity, relation, target}],
                "contrast_dimensions": [...],
                "query_type": str,
                "latency_ms": float
            }
            use_llm_judge: bool

        Returns: {
            "overall_scores": {...},
            "per_question_results": [...],
            "by_difficulty": {...},
            "by_query_type": {...},
            "timing": {...}
        }
        """
        per_question_results = []
        aggregated = {
            "qa_correctness": [], "qa_faithfulness": [], "qa_completeness": [],
            "citation_precision": [], "citation_recall": [],
            "retrieval_recall5": [], "retrieval_mrr": [],
            "extraction_entity_f1": [], "extraction_triple_f1": [],
            "contrast_agreement": [],
            "latency_ms": []
        }

        total = len(test_set["questions"])
        start_time = time.time()

        for i, q in enumerate(test_set["questions"]):
            print(f"\r[Evaluator] Question {i + 1}/{total}: {q['id']}", end="")

            try:
                # 运行系统
                sys_output = system_runner(q)

                # QA评估
                qa_scores = self.evaluate_qa(
                    question=q["question"],
                    predicted_answer=sys_output.get("answer", ""),
                    ground_truth=q.get("expected_answer", ""),
                    key_points=q.get("key_points", []),
                    supporting_sources=q.get("supporting_sources", []),
                    predicted_citations=sys_output.get("citations", []),
                    use_llm=use_llm_judge
                )

                # 检索评估
                ret_scores = self.evaluate_retrieval(
                    retrieved_docs=sys_output.get("retrieved_docs", []),
                    relevant_ids=q.get("supporting_sources", [])
                )

                # 抽取评估
                ext_scores = self.evaluate_extraction(
                    predicted=sys_output.get("extracted_triples", []),
                    ground_truth=q.get("expected_kg_path", [])
                )

                # 对比评估
                contrast_scores = self.evaluate_contrast(
                    predicted_dimensions=sys_output.get("contrast_dimensions", []),
                    expected_dimensions=q.get("expected_dimensions", [])
                )

                q_result = {
                    "id": q["id"],
                    "question": q["question"][:80],
                    "query_type": q.get("query_type", "FACTUAL"),
                    "difficulty": q.get("difficulty", "medium"),
                    "qa_metrics": qa_scores,
                    "retrieval_metrics": ret_scores,
                    "extraction_metrics": ext_scores,
                    "contrast_metrics": contrast_scores,
                    "latency_ms": sys_output.get("latency_ms", 0)
                }
                per_question_results.append(q_result)

                # 聚合
                aggregated["qa_correctness"].append(qa_scores.get("correctness", 0.5))
                aggregated["qa_faithfulness"].append(qa_scores.get("faithfulness", 0.5))
                aggregated["qa_completeness"].append(qa_scores.get("completeness", 0.5))
                aggregated["citation_precision"].append(qa_scores.get("citation_precision", 1.0))
                aggregated["citation_recall"].append(qa_scores.get("citation_recall", 1.0))
                aggregated["retrieval_recall5"].append(ret_scores.get("recall@5", 1.0))
                aggregated["retrieval_mrr"].append(ret_scores.get("mrr", 1.0))
                aggregated["extraction_entity_f1"].append(ext_scores.get("entity", {}).get("f1", 1.0))
                aggregated["extraction_triple_f1"].append(ext_scores.get("triple", {}).get("f1", 1.0))
                aggregated["contrast_agreement"].append(contrast_scores.get("dimension_agreement", 1.0))
                aggregated["latency_ms"].append(sys_output.get("latency_ms", 0))

            except Exception as e:
                print(f"\n[Evaluator] Error on question {q['id']}: {e}")
                per_question_results.append({
                    "id": q["id"],
                    "error": str(e)
                })

        total_time = time.time() - start_time
        print()  # newline after progress

        # 计算总体分数
        def safe_mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        overall_scores = {
            "qa_correctness": safe_mean(aggregated["qa_correctness"]),
            "qa_faithfulness": safe_mean(aggregated["qa_faithfulness"]),
            "qa_completeness": safe_mean(aggregated["qa_completeness"]),
            "citation_precision": safe_mean(aggregated["citation_precision"]),
            "citation_recall": safe_mean(aggregated["citation_recall"]),
            "retrieval_recall@5": safe_mean(aggregated["retrieval_recall5"]),
            "retrieval_mrr": safe_mean(aggregated["retrieval_mrr"]),
            "extraction_entity_f1": safe_mean(aggregated["extraction_entity_f1"]),
            "extraction_triple_f1": safe_mean(aggregated["extraction_triple_f1"]),
            "contrast_agreement": safe_mean(aggregated["contrast_agreement"]),
            "avg_latency_ms": safe_mean(aggregated["latency_ms"]),
            # 综合总分 (加权求和, 权重合计=1.0)
            # 注意: extraction/contrast 暂不参与评分 (baselines未实现三元组抽取)
            "overall_score": round(
                safe_mean(aggregated["qa_correctness"]) * 0.35 +
                safe_mean(aggregated["qa_faithfulness"]) * 0.20 +
                safe_mean(aggregated["qa_completeness"]) * 0.10 +
                safe_mean(aggregated["citation_precision"]) * 0.10 +
                safe_mean(aggregated["citation_recall"]) * 0.05 +
                safe_mean(aggregated["retrieval_recall5"]) * 0.15 +
                safe_mean(aggregated["retrieval_mrr"]) * 0.05,
            4)
        }

        # 按难度分组
        by_difficulty = self._group_by_key(per_question_results, "difficulty")

        # 按问题类型分组
        by_query_type = self._group_by_key(per_question_results, "query_type")

        full_results = {
            "overall_scores": overall_scores,
            "per_question_results": per_question_results,
            "by_difficulty": by_difficulty,
            "by_query_type": by_query_type,
            "timing": {
                "total_seconds": round(total_time, 2),
                "total_questions": total,
                "avg_per_question_ms": round(total_time * 1000 / total, 1) if total > 0 else 0
            },
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "evaluator_model": self.model
        }

        return full_results

    def _group_by_key(self, results, key):
        """按指定键分组计算平均分数"""
        groups = defaultdict(list)
        for r in results:
            if "qa_metrics" not in r:
                continue
            k = r.get(key, "unknown")
            score = (r.get("qa_metrics", {}).get("correctness", 0.5) +
                     r.get("qa_metrics", {}).get("faithfulness", 0.5)) / 2
            groups[k].append(score)

        return {k: {"avg_score": round(sum(v) / len(v), 4), "count": len(v)}
                for k, v in groups.items()}


def compute_reliability(citation_density, source_coverage, attribution_density, graph_completeness):
    """计算综合可靠性分数 (用于实时QA)"""
    return round(
        0.25 * citation_density +
        0.25 * source_coverage +
        0.25 * attribution_density +
        0.25 * graph_completeness,
        4
    )
