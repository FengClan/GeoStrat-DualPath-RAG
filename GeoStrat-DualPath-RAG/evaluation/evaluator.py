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
import os
import time
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
# 别名：3.11 起它就是内置 TimeoutError，3.10 及更早是 concurrent.futures 自己的类，
# 显式引入才能两个版本都捕获到（本项目跑在 3.10）。
from concurrent.futures import TimeoutError as FuturesTimeoutError
from difflib import SequenceMatcher


# 连续的中文/字母数字片段（长度 >= 2），用于 key_point 的部分匹配。
# 长度下限 2 是为了避免单字命中即得分（单字几乎在任何中文回答里都会出现）。
_KP_UNIT_RE = re.compile(r'[一-鿿A-Za-z0-9]{2,}')


def _kp_units(text):
    """把要点拆成可用于部分匹配的词元（连续 >=2 字符的中英文数字片段）。"""
    return _KP_UNIT_RE.findall(text or "")


class Evaluator:
    def __init__(self, model="qwen3.5-plus", results_dir=None, judge_model=None):
        """
        Args:
            model: 生成模型名（记录用，实际生成由 baseline runner 负责）
            results_dir: 结果输出目录
            judge_model: LLM 裁判模型名。默认与生成模型相同（存在 self-preference 偏差，
                         仅用于内部对照）；正式实验应传入独立裁判，如 "deepseek-flash"。
        """
        self.model = model
        self.judge_model = judge_model or model
        self.results_dir = results_dir or "./evaluation/results"
        import os
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def is_independent_judge(self):
        """裁判是否独立于生成模型（回应审稿意见 R1#4 / R2#7）。"""
        return self.judge_model != self.model

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
            # 注意：返回结构必须与下方成功路径一致（嵌套 entity/triple）。
            # 此前这里返回扁平 dict（键为 f1），调用方按 ext_scores["triple"]["f1"]
            # 取值必然落空并落到默认值，导致所有基线的抽取 F1 恒为伪造的 1.0。
            return {"entity": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "triple": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "pred_count": 0, "gt_count": 0,
                    "entity_pred_count": 0, "entity_gt_count": 0,
                    "details": "no ground truth"}

        if not predicted:
            return {"entity": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "triple": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "pred_count": 0, "gt_count": len(ground_truth),
                    "entity_pred_count": 0, "entity_gt_count": 0,
                    "details": "no predictions"}

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

    def evaluate_evidence(self, surfaced_spans, gold_spans, applicable=True):
        """
        片段级证据召回 —— 替代退化的文档级 recall@5/MRR 与 citation 指标（R2#5）。

        背景（2026-09-14 实测）：测试集每题的 supporting_sources 都是
        ["福建地质志.docx"]（单文档），向量库 chunk 的 source_file 也是同一篇，
        于是"召回"退化成"有没有返回任何文档"——vector_rag / dual_path 恒为 1.0，
        而用 evidence_span 做真值实测，真实的片段级召回只有约 47%。
        旧指标把 1.0 报成了检索完美，等于**掩盖了一半以上的检索失败**。

        本方法的真值来自测试集的 `gold_evidence_spans`：只有**逐字存在于语料**的
        evidence_span（或对比题按分号拆分后逐字的子句）才能当检索真值——
        专家改写的概括在语料里根本没有对应文字，拿它当真值会把"真值不存在"
        记成"检索失败"。故此类题返回 None（N/A），不记 0。

        Args:
            surfaced_spans: 该系统实际展示给生成器的文本片段（top-k chunk 全文）
            gold_spans: 该题的逐字证据片段（可能多段，对比题即两侧各一段）
            applicable: 该基线是否存在文本片段通道；graph_rag/direct_llm 传 False
        Returns: {evidence_recall@5, n_gold, n_found, ...}；不适用时为 None
        """
        if not applicable:
            return {"evidence_recall@5": None, "applicable": False, "n_gold": 0,
                    "details": "N/A — 该基线无文本片段通道（R2#6）"}

        gold = [str(g) for g in (gold_spans or []) if g and str(g).strip()]
        if not gold:
            return {"evidence_recall@5": None, "applicable": False, "n_gold": 0,
                    "details": "N/A — 该题 evidence_span 为专家改写，语料中无逐字对应，"
                               "不可作检索真值"}

        texts = [re.sub(r"\s+", "", s) for s in (surfaced_spans or [])]
        found = 0
        for g in gold:
            ng = re.sub(r"\s+", "", g)
            if ng and any(ng in t for t in texts):
                found += 1
        return {
            "evidence_recall@5": round(found / len(gold), 4),
            "applicable": True,
            "n_gold": len(gold),
            "n_found": found,
            "details": f"{found}/{len(gold)} 段逐字证据被展示的片段覆盖",
        }

    def evaluate_retrieval(self, retrieved_docs, relevant_ids, k_values=[1, 3, 5, 10],
                           applicable=True):
        """
        【仅作诊断保留，勿用于论文结论】文档级检索指标。

        它按 supporting_sources（= ["福建地质志.docx"]，单文档）判定，实测
        vector_rag / dual_path 恒为 1.0，不区分检索质量（R2#5）。真正反映检索
        质量的是 evaluate_evidence() 的片段级证据召回，主表请用它。
        此处保留仅为与历史结果对照。

        评估检索质量。
        Args:
            retrieved_docs: 排序后的检索结果 [{id, score, content, ...}]
            relevant_ids: 标注的相关文档ID集合
            k_values: 评估的截断位置
            applicable: 该基线是否存在"文本片段检索"这一环。图谱检索（graph_rag）
                和纯 LLM（direct_llm）没有文本片段召回，对其报 recall@5=0 是
                **把"不适用"记成了"得了零分"**——审稿意见 R2#6 明确要求改报 N/A。
                传 False 时各项返回 None（JSON 里即 null），不参与任何聚合。
        Returns: {recall@{k}, MRR, hit@{k}, ...}；不适用的项值为 None
        """
        if not applicable:
            return {"recall@5": None, "mrr": None, "hit@5": None,
                    "applicable": False,
                    "details": "N/A — 该基线无文本片段检索通道，召回类指标不适用（R2#6）"}

        if not relevant_ids:
            return {"recall@5": 0.0, "mrr": 0.0, "hit@5": 0.0, "applicable": True,
                    "details": "no relevance judgments"}

        if not retrieved_docs:
            return {"recall@5": 0.0, "mrr": 0.0, "hit@5": 0.0, "applicable": True,
                    "details": "no retrieved documents"}

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
                    # 尝试部分匹配：按「词元」而非「单字」匹配。
                    # 修复前此处为 `for word in kp`，迭代的是字符串的每个字符，
                    # 命中任意一个汉字即给 0.5 分，导致 58% 的题覆盖度恰好=0.5。
                    for word in _kp_units(kp):
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
        # 裁判偶发返回空串/非 JSON（实测约 3%：30 题对照里每臂各 1 次），
        # 原实现直接落 0.5 —— 而 correctness 在 overall_score 里占 0.4375，
        # 一次静默 0.5 就把该题总分压低，且**结果文件里看不出是哪几题**。
        # 现在解析失败重试一次，仍失败则标记 judge_failed 供事后统计剔除。
        last_err = None
        for attempt in range(2):
            try:
                # 裁判固定 temperature=0，保证可复现（回应 R2#8/#9 对随机波动的质疑）
                raw = get_model_response(prompt, self.judge_model, None, {"temperature": 0.0})
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
                    "llm_judge_reason": scores.get("reason", ""),
                    "judge_failed": False
                }
            except Exception as e:
                last_err = e
                if attempt == 0:
                    print(f"[Evaluator] 裁判返回无法解析，重试一次: {e}")

        print(f"[Evaluator] LLM judge failed（已重试）: {last_err}")
        return {
            "correctness": 0.5, "completeness": 0.5,
            "faithfulness": 0.5, "coherence": 0.5,
            "domain_accuracy": 0.5, "llm_judge_score": 0.5,
            "llm_judge_reason": f"Evaluation failed: {last_err}",
            # 显式标记：论文里应报告该比例，并说明这些题按 0.5 中性值计入
            "judge_failed": True
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

    def run_full_evaluation(self, test_set, system_runner, use_llm_judge=True,
                            checkpoint_path=None, workers=1):
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
            workers: 并发题数。1=串行（历史行为）。
                ⚠️ 并发不改变每题的生成/评判语义（同一 prompt、同一参数、同一裁判），
                   只把"等 API 返回"的墙钟时间叠起来；API 调用次数与总 token 不变。
                   进度落盘与聚合已加锁，断点续跑语义与串行一致。

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
            "evidence_recall": [],
            "extraction_entity_f1": [], "extraction_triple_f1": [],
            "contrast_agreement": [],
            "latency_ms": []
        }

        total = len(test_set["questions"])

        # --- 断点续跑：加载已完成进度 ---
        done_ids = set()
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                # 只恢复"成功"的题（含 qa_metrics）；失败题丢弃，重启时重跑
                restored = [r for r in ckpt.get("per_question_results", [])
                            if "qa_metrics" in r]
                if restored:
                    per_question_results = restored
                    aggregated = ckpt.get("aggregated", aggregated)
                    # 旧 checkpoint 里可能没有新增的聚合键（如 evidence_recall）：
                    # 补空列表，否则后续 append 会 KeyError 而不是给出可读的报错。
                    for k in ("qa_correctness", "qa_faithfulness", "qa_completeness",
                              "citation_precision", "citation_recall",
                              "retrieval_recall5", "retrieval_mrr", "evidence_recall",
                              "extraction_entity_f1", "extraction_triple_f1",
                              "contrast_agreement", "latency_ms"):
                        aggregated.setdefault(k, [])
                    done_ids = {r["id"] for r in restored}
                print(f"\n[Evaluator] 断点续跑：已完成 {len(done_ids)}/{total} 题，"
                      f"继续剩余 {total - len(done_ids)} 题")
            except Exception as e:
                print(f"\n[Evaluator] 读取 checkpoint 失败，从头开始: {e}")

        def _save_checkpoint():
            if not checkpoint_path:
                return
            tmp = checkpoint_path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({"per_question_results": per_question_results,
                           "aggregated": aggregated},
                          f, ensure_ascii=False)
            os.replace(tmp, checkpoint_path)

        start_time = time.time()

        pending = [q for q in test_set["questions"] if q["id"] not in done_ids]
        print(f"\n[Evaluator] 待评测 {len(pending)}/{total} 题，并发 workers={workers}")

        lock = threading.Lock()
        stop = threading.Event()        # 欠费等永久性错误：置位后不再发起新调用
        progress = [0]

        def _process(q):
            """单题：跑系统 + 四项评估。只读共享状态，返回 (q_result, err)。

            刻意不在此函数内触碰 per_question_results / aggregated / checkpoint，
            统一交给 _merge() 在锁内处理——否则并发下聚合会错乱。
            """
            if stop.is_set():
                return None, None       # 已停止：不再发起任何 API 调用
            try:
                sys_output = system_runner(q)

                qa_scores = self.evaluate_qa(
                    question=q["question"],
                    predicted_answer=sys_output.get("answer", ""),
                    ground_truth=q.get("expected_answer", ""),
                    key_points=q.get("key_points", []),
                    supporting_sources=q.get("supporting_sources", []),
                    predicted_citations=sys_output.get("citations", []),
                    use_llm=use_llm_judge
                )
                ret_scores = self.evaluate_retrieval(
                    retrieved_docs=sys_output.get("retrieved_docs", []),
                    relevant_ids=q.get("supporting_sources", []),
                    # 基线自己声明有没有文本片段检索通道（graph_rag / direct_llm 没有）
                    applicable=sys_output.get("retrieval_applicable", True)
                )
                # 片段级证据召回（R2#5 的正解）：真值是该题逐字存在的 evidence_span
                ev_scores = self.evaluate_evidence(
                    surfaced_spans=sys_output.get("surfaced_spans", []),
                    gold_spans=q.get("gold_evidence_spans", []),
                    applicable=sys_output.get("evidence_applicable", False)
                )
                ext_scores = self.evaluate_extraction(
                    predicted=sys_output.get("extracted_triples", []),
                    ground_truth=q.get("expected_kg_path", [])
                )
                contrast_scores = self.evaluate_contrast(
                    predicted_dimensions=sys_output.get("contrast_dimensions", []),
                    expected_dimensions=q.get("expected_dimensions", [])
                )
                return {
                    "id": q["id"],
                    "question": q["question"][:80],
                    "query_type": q.get("query_type", "FACTUAL"),
                    "difficulty": q.get("difficulty", "medium"),
                    # 保存原始回答：换裁判/复核时可直接重评，无需重新生成（省一半成本）
                    "answer": sys_output.get("answer", ""),
                    "expected_answer": q.get("expected_answer", ""),
                    "qa_metrics": qa_scores,
                    "retrieval_metrics": ret_scores,
                    "evidence_metrics": ev_scores,
                    "extraction_metrics": ext_scores,
                    "contrast_metrics": contrast_scores,
                    "latency_ms": sys_output.get("latency_ms", 0)
                }, None
            except Exception as e:
                return {"id": q["id"], "error": str(e)}, str(e)

        skipped = [0]

        def _merge(q_result, err):
            """锁内：落结果 + 聚合 + 落盘。串行/并发走同一条路径。"""
            with lock:
                if q_result is None:
                    # _process 因 stop 置位而跳过该题（欠费停机）：
                    # 不能把 None 塞进结果列表，否则后续 safe_mean / 分组全崩。
                    skipped[0] += 1
                    return
                per_question_results.append(q_result)
                if err is None:
                    qa = q_result["qa_metrics"]
                    ret = q_result["retrieval_metrics"]
                    ext = q_result["extraction_metrics"]
                    con = q_result["contrast_metrics"]
                    aggregated["qa_correctness"].append(qa.get("correctness", 0.5))
                    aggregated["qa_faithfulness"].append(qa.get("faithfulness", 0.5))
                    aggregated["qa_completeness"].append(qa.get("completeness", 0.5))
                    aggregated["citation_precision"].append(qa.get("citation_precision", 1.0))
                    aggregated["citation_recall"].append(qa.get("citation_recall", 1.0))
                    # None = 该基线不适用（R2#6）：不塞进聚合列表，否则会被当成 0 分平均
                    if ret.get("recall@5") is not None:
                        aggregated["retrieval_recall5"].append(ret["recall@5"])
                    if ret.get("mrr") is not None:
                        aggregated["retrieval_mrr"].append(ret["mrr"])
                    # 片段级证据召回：同样只在"有逐字证据真值 + 该基线有文本通道"时计入
                    ev = q_result.get("evidence_metrics", {})
                    if ev.get("evidence_recall@5") is not None:
                        aggregated["evidence_recall"].append(ev["evidence_recall@5"])
                    # 默认值 0.0（而非 1.0）：取不到就当"未测到"，不能凭空给满分。
                    # 配合 evaluate_extraction() 早退分支改为嵌套结构，此项不再伪造 1.0。
                    aggregated["extraction_entity_f1"].append(ext.get("entity", {}).get("f1", 0.0))
                    aggregated["extraction_triple_f1"].append(ext.get("triple", {}).get("f1", 0.0))
                    aggregated["contrast_agreement"].append(con.get("dimension_agreement", 0.0))
                    aggregated["latency_ms"].append(q_result.get("latency_ms", 0))
                else:
                    print(f"\n[Evaluator] Error on question {q_result['id']}: {err}")

                progress[0] += 1
                print(f"\r[Evaluator] {progress[0]}/{len(pending)} 完成", end="")

                _save_checkpoint()  # 每题结束后立即落盘（串行/并发共用，锁内调用）

                # 欠费/账号异常等永久性错误：立即停止本基线，不再空转刷完剩余题。
                # "Arrearage"/"overdue-payment"/"good standing" 是百炼(DashScope)的文案；
                # "Insufficient Balance" 是 DeepSeek 的（HTTP 402）——两家都要认，
                # 否则裁判侧余额耗尽时匹配不上，脚本会带着 402 空转刷完剩下的题。
                if err and any(k in err for k in ("Arrearage", "overdue-payment",
                                                   "good standing",
                                                   "Insufficient Balance")):
                    if not stop.is_set():
                        print("\n[Evaluator] 检测到账号欠费(Arrearage)，立即停止本基线。"
                              "已保存进度，充值后重跑同一命令即可续跑。")
                    stop.set()

        # --- 并发执行（可被 Ctrl+C 立即中断）---
        # 刻意不用 `with ThreadPoolExecutor(...)`：它的 __exit__ 走 shutdown(wait=True)
        # 且 cancel_futures 默认为 False，会把**已提交但未开跑**的题全部跑完才返回
        # （实测：40 题/4 并发下发 SIGINT，进程要等 5.0s 满工作量跑完才退出）。
        # 全量评测是一次性提交 1000+ 个 future，那等于"Ctrl+C 之后还要跑几小时"。
        # 现在改为显式 shutdown(wait=False, cancel_futures=True)：未开跑的题直接丢弃，
        # 已在跑的最多 workers 道会自行收尾（它们未被记录，重跑时自动补上）。
        interrupted = [False]
        ex = ThreadPoolExecutor(max_workers=workers) if (workers > 1 and len(pending) > 1) else None
        futs = []

        # _merge 的幂等包装：中断抢救时要重新遍历一遍 futures，
        # 没有这层去重，同一道题会被并入两次（结果表重复、聚合被灌水）。
        merged_ids = set()

        def _merge_once(q_result, err):
            if q_result is not None:
                qid = q_result.get("id")
                if qid in merged_ids:
                    return
                merged_ids.add(qid)
            _merge(q_result, err)

        try:
            if ex is not None:
                futs = [ex.submit(_process, q) for q in pending]
                for f in futs:
                    # 用 1s 轮询而非直接 f.result()：主线程阻塞在等待里时**不执行字节码**，
                    # 收不到 Ctrl+C——实测中断要等到下一题完成才生效（t=1s 发的信号
                    # 直到 t=10s 才抛出）。轮询让主线程每秒回到解释器一次，中断立即响应。
                    while True:
                        try:
                            outcome = f.result(timeout=1.0)   # _process 返回 (q_result, err)
                            break
                        except FuturesTimeoutError:
                            continue
                    _merge_once(*outcome)
            else:
                for q in pending:
                    _merge_once(*_process(q))
        except KeyboardInterrupt:
            interrupted[0] = True
            stop.set()          # 双保险：万一还有排队任务被拉起，_process 入口即返回
            print("\n[Evaluator] 收到中断（Ctrl+C）：立即停止，不再发起新题。")
            # 并发下主线程按提交顺序 merge，会出现"后面几道已经跑完、前面那道还没完"
            # 的局面——这些已完成但尚未并入的题，API 钱已经花掉了，中断时抢救回来，
            # 否则它们会被丢弃、重跑时再花一次。
            if ex is not None and futs:
                before = len(per_question_results)
                for f in futs:
                    if f.done() and not f.cancelled():
                        try:
                            _merge_once(*f.result(timeout=0))
                        except Exception:
                            pass
                saved = len(per_question_results) - before
                if saved:
                    print(f"[Evaluator] 已抢救回 {saved} 道跑完但尚未并入的题。")
            n_running = sum(1 for f in futs if not f.done()) if futs else 0
            print(f"[Evaluator] {n_running} 道仍在收尾的题本次不计入，"
                  f"重跑同一命令时会自动补上。")
        finally:
            if ex is not None:
                # 正常结束时所有 future 都已完成，cancel_futures 对空队列是空操作；
                # 中断时它负责丢掉未开跑的题。两条路径都不阻塞。
                ex.shutdown(wait=False, cancel_futures=True)

        if interrupted[0]:
            _save_checkpoint()
            n_done = sum(1 for r in per_question_results if "qa_metrics" in r)
            print(f"[Evaluator] 进度已落盘：{n_done} 题完成。"
                  f"恢复：重跑同一条命令（同 --test-set/--total/--seed/--results-dir）")

        if skipped[0]:
            print(f"\n[Evaluator] 因停机跳过 {skipped[0]} 题（未计入结果，"
                  f"重跑同一命令会自动补上）")

        total_time = time.time() - start_time
        print()  # newline after progress

        # 计算总体分数
        def safe_mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        def safe_mean_na(vals):
            """无数据时返回 None（JSON null）而不是 0.0 —— 用于"不适用"的指标。

            0.0 和"不适用"在读者眼里是两回事：前者是"考了零分"，后者是"没考这门"。
            审稿意见 R2#6 要的就是这个区分。
            """
            return round(sum(vals) / len(vals), 4) if vals else None

        # --- 各分项覆盖情况（用于诚实标注"哪些指标实际未测到"）---
        n_total = len(per_question_results)
        n_ext_gt = sum(1 for r in per_question_results
                       if r.get("extraction_metrics", {}).get("gt_count"))
        n_contrast_gt = sum(1 for r in per_question_results
                            if r.get("contrast_metrics", {}).get("dimension_names"))
        n_judge_failed = sum(1 for r in per_question_results
                             if r.get("qa_metrics", {}).get("judge_failed"))
        # 生成/评判中途抛异常的题（_process 的 except 分支只记 {"id","error"}）。
        # 这些题**没有** qa_metrics，不进任何聚合，但会被计入上面的 total_questions。
        n_errors = sum(1 for r in per_question_results if "error" in r)

        overall_scores = {
            "qa_correctness": safe_mean(aggregated["qa_correctness"]),
            "qa_faithfulness": safe_mean(aggregated["qa_faithfulness"]),
            "qa_completeness": safe_mean(aggregated["qa_completeness"]),
            "citation_precision": safe_mean(aggregated["citation_precision"]),
            "citation_recall": safe_mean(aggregated["citation_recall"]),
            # ↓ 仅作透明性报告，【不参与综合得分】。理由见下方 overall_score 注释。
            # 图谱检索/纯 LLM 无文本召回通道 → null（N/A），不是 0（R2#6）。
            "retrieval_recall@5": safe_mean_na(aggregated["retrieval_recall5"]),
            "retrieval_mrr": safe_mean_na(aggregated["retrieval_mrr"]),
            "extraction_entity_f1": safe_mean(aggregated["extraction_entity_f1"]),
            "extraction_triple_f1": safe_mean(aggregated["extraction_triple_f1"]),
            "contrast_agreement": safe_mean(aggregated["contrast_agreement"]),
            "avg_latency_ms": safe_mean(aggregated["latency_ms"]),
            # 指标覆盖率：对应分项实际有多少题具备 ground truth（0 表示该项未真正评测）
            "coverage": {
                "total_questions": n_total,
                # 真正被计入聚合的题数 = total_questions - errors。全量几千题、且
                # config.py 里 max_retries=0 的前提下，一次瞬时 429/超时就会留下一题；
                # 不单独报这个数，1000 题里少算了 20 题没人看得出来。这些题没有
                # qa_metrics，重跑同一条命令时会自动补上（断点续跑只恢复有 qa_metrics 的）。
                "errors": n_errors,
                "evaluated_questions": n_total - n_errors,
                "extraction_with_gt": n_ext_gt,
                "contrast_with_gt": n_contrast_gt,
                # 片段级证据召回真正算到的题数（其余为专家改写，真值不存在 → N/A）。
                # 论文里报该指标时必须一并给出这个数，否则 47% 的覆盖率会被误读成全量结论。
                "evidence_recall_evaluated": len(aggregated["evidence_recall"]),
                # 裁判两次都没返回可解析 JSON 的题数（这些题按 0.5 中性值计入总分）。
                # 该比例应在论文中如实报告——否则裁判稳定性无从核查（R2#7）。
                "judge_failed": n_judge_failed,
            },
            # ================================================================
            # 综合总分
            # ================================================================
            # 【2026-09-14 二次修复】再把 citation_precision / citation_recall
            # （合计权重 0.1875）移出综合总分。
            #
            # 原因：这两项也由构造决定必然满分，与已移除的 recall@5/MRR 同源。
            # baselines.py 里 retrieved_docs 的 "id" 字段装的是 source_file
            # （`d.metadata.get("source_file")`），graph_rag 更是直接硬写
            # ["福建地质志.docx"]；而测试集 ground truth 就是同一个文档名。
            # 于是任何"返回了东西"的系统一律 precision=recall=1.0：
            # 实测 30 题里 vector_rag / graph_rag / dual_path 三者**精确等于 1.0000**。
            # 这 0.1875 是白送给所有 RAG 系统的常数，其效果是**压缩**系统间差距
            # （实测双路对向量RAG的差距 0.0348 被压到 0.0429 才是真实比例）。
            #
            # 正解不是删指标，而是换真值：测试集带 evidence_span，但实测仅 55% 逐字
            # 存在于语料（专家改写的不可作检索真值），故新的片段级指标
            # evidence_recall@5 作为**报告项**另行给出，不塞进综合分——
            # 因为它的覆盖不均（FACTUAL 66%、COMPARATIVE 仅 1.8%），
            # 计入加权会让不同类型题的得分不可比。对应审稿意见 R2#5。
            #
            # 现在综合分只由三个裁判打分的 QA 维度构成，四项基线权重相同、口径一致。
            # extraction / contrast 不参与（baselines 未实现三元组抽取与维度对比输出）。
            "overall_score": round(
                safe_mean(aggregated["qa_correctness"]) * 0.5385 +   # 0.4375/0.8125
                safe_mean(aggregated["qa_faithfulness"]) * 0.3077 +  # 0.2500/0.8125
                safe_mean(aggregated["qa_completeness"]) * 0.1538,   # 0.1250/0.8125
            4),
            # 片段级证据召回：覆盖不均（见上），故只报告不加权。None = 无适用题
            "evidence_recall@5": safe_mean_na(aggregated["evidence_recall"]),
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
                "total_seconds": round(total_time, 2),          # 墙钟（并发时已压缩）
                "total_questions": total,
                "workers": workers,
                # 并发下 total_time/total 是"均摊墙钟"而非单题延迟，两者要分开报，
                # 否则会让人误以为模型变快了。单题延迟看 avg_latency_ms。
                "avg_per_question_ms": round(total_time * 1000 / total, 1) if total > 0 else 0,
                "sum_model_seconds": round(sum(aggregated["latency_ms"]) / 1000, 2),
                "speedup_vs_serial": (
                    round(sum(aggregated["latency_ms"]) / 1000 / total_time, 2)
                    if total_time > 0 and workers > 1 else 1.0),
            },
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "evaluator_model": self.judge_model,
            "generator_model": self.model,
            "independent_judge": self.is_independent_judge,
            # True = 本次是被 Ctrl+C 中断的，结果不完整（只覆盖已完成的题）。
            # 调用方据此停止后续基线，不要把它当成一次完整评测去汇总/比较。
            "interrupted": interrupted[0],
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
