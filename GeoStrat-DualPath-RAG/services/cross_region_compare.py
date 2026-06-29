# services/cross_region_compare.py — 跨区域地层对比 & 同义异名处理
import time
import re
import json
from difflib import SequenceMatcher

from flask import Blueprint, request, jsonify

from .model_router import get_model_response
from .kg_service import load_graph_for_kb
from .strat_service import collect_evidence
from .ontology import ENTITY_TYPES, HIERARCHY_ORDER

cross_region_bp = Blueprint('cross_region', __name__)
DEFAULT_MODEL = "qwen3.5-plus"


# ============================================================
# 同义异名检测
# ============================================================


def detect_synonyms(entity_name, kb_id="all", region_hint=None):
    """
    检测地层的同义异名（同一地层在不同区域的名称）。
    策略：
    1. 从 KG 中查询 SAME_AS 关系
    2. 模糊名称匹配 (命名模式相似度)
    3. 语义相似度 (调用 LLM 判断)
    """
    candidates = []

    # 策略1: KG SAME_AS 边
    try:
        G = load_graph_for_kb(kb_id)
        matched = None
        for node in G.nodes():
            if str(node) == entity_name or entity_name in str(node):
                matched = str(node)
                break
        if matched and G.has_node(matched):
            # 查找同义关系
            for nb in (set(G.successors(matched)) | set(G.predecessors(matched))):
                edge = G.get_edge_data(matched, nb) or G.get_edge_data(nb, matched) or {}
                if edge.get('relation', '') == 'SAME_AS':
                    candidates.append({
                        "name": nb,
                        "source": "KG:SAME_AS",
                        "confidence": 0.9,
                        "method": "knowledge_graph"
                    })
    except Exception as e:
        print(f"[Synonym] KG lookup failed: {e}")

    # 策略2: 模糊名称匹配
    base_name = entity_name.rstrip('群组段')  # 去掉后缀
    try:
        G = load_graph_for_kb(kb_id)
        for node in G.nodes():
            node_str = str(node)
            if node_str == entity_name or node_str in [c['name'] for c in candidates]:
                continue
            # 计算名称相似度
            sim = SequenceMatcher(None, entity_name, node_str).ratio()
            if sim > 0.6 and (base_name in node_str or node_str.rstrip('群组段') in entity_name):
                candidates.append({
                    "name": node_str,
                    "source": "fuzzy_match",
                    "confidence": round(sim * 0.8, 2),
                    "method": "fuzzy_match"
                })
    except Exception:
        pass

    return candidates


def detect_by_llm(entity_name, candidates, model=DEFAULT_MODEL):
    """
    使用 LLM 判断候选同义异名是否确实指向同一地层。
    """
    if not candidates:
        return []

    prompt = f"""你是地质命名专家。请判断以下候选名称是否是 "{entity_name}" 的同义异名（同一地层在不同地区的不同名称）。

当前地层: {entity_name}
候选同义异名:
{json.dumps([c['name'] for c in candidates], ensure_ascii=False)}

判断标准:
- 「同物异名」: 同一套地层在不同区域有不同的正式名称 (如"龙山组"在另一区域被称为"黄连组")
- 「异物同名」: 名称相同但实际是不同地层 → 不是同义
- 只包含同一层级（如群对应群，组对应组）

返回JSON数组，每个候选给出判断：
[{{"name": "候选名", "is_synonym": true/false, "confidence": 0.8, "reason": "简短理由"}}]

仅返回JSON。"""
    try:
        raw = get_model_response(prompt, model)
        cleaned = raw.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        return json.loads(cleaned)
    except Exception as e:
        print(f"[Synonym] LLM detection failed: {e}")
        return []


# ============================================================
# 跨区域对比
# ============================================================


def cross_region_compare(entity_name, region_a, region_b, kb_id="all", model=DEFAULT_MODEL):
    """
    跨区域地层对比：分析同一地层在两个区域的特征差异
    """
    # 1. 收集两区域的证据
    evidence_a = collect_evidence(f"{entity_name} {region_a}", kb_id)
    evidence_b = collect_evidence(f"{entity_name} {region_b}", kb_id)

    # 2. 同义异名检测
    synonyms_in_region = detect_synonyms(entity_name, kb_id, region_a)

    # 3. 构建跨区域对比提示词
    ev_a_text = "\n".join([f"[{e['source_type']}] {e['evidence_text']}" for e in evidence_a[:8]])
    ev_b_text = "\n".join([f"[{e['source_type']}] {e['evidence_text']}" for e in evidence_b[:8]])

    prompt = f"""你是跨区域地层对比专家。请对比分析 "{entity_name}" 在 {region_a} 和 {region_b} 两个区域的异同。

## 区域 {region_a} 证据
{ev_a_text if ev_a_text else '暂无证据'}

## 区域 {region_b} 证据
{ev_b_text if ev_b_text else '暂无证据'}

## 可能的同义异名
{json.dumps(synonyms_in_region, ensure_ascii=False) if synonyms_in_region else '未发现'}

请从以下维度进行跨区域对比：
1. **岩性差异**: 岩石组合是否一致？厚度变化趋势？
2. **化石组合**: 所含有化石是否相同？能否用于区域对比？
3. **命名一致性**: 是否存在同义异名？命名是否需要统一？
4. **接触关系**: 上下地层的接触关系在两地是否相同？

返回格式 (严格JSON):
```json
{{
    "entity": "{entity_name}",
    "regions": ["{region_a}", "{region_b}"],
    "synonyms_detected": [...],
    "comparison": {{
        "lithology": {{"{region_a}": "...", "{region_b}": "...", "agreement": "一致|部分一致|差异显著"}},
        "thickness": {{"{region_a}": "...", "{region_b}": "...", "trend": "稳定|减薄|增厚|不确定"}},
        "fossils": {{"{region_a}": "...", "{region_b}": "...", "correlation": "可对比|不可对比|不确定"}}
    }},
    "conclusion": "跨区域对比结论...",
    "naming_issues": ["命名不一致问题..."],
    "confidence": 0.85
}}
```

仅返回JSON。"""
    raw = get_model_response(prompt, model)
    cleaned = raw.strip()
    if cleaned.startswith("```json"): cleaned = cleaned[7:]
    if cleaned.startswith("```"): cleaned = cleaned[3:]
    if cleaned.endswith("```"): cleaned = cleaned[:-3]

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {"error": "JSON parse failed", "raw_response": cleaned}

    result["evidence_counts"] = {
        f"region_{region_a}": len(evidence_a),
        f"region_{region_b}": len(evidence_b)
    }

    return result


# ============================================================
# API Routes
# ============================================================


@cross_region_bp.route('/strat/cross-region', methods=['POST'])
def cross_region_endpoint():
    """跨区域地层对比"""
    data = request.get_json(silent=True) or {}
    entity_name = data.get('entity', '')
    region_a = data.get('region_a', '')
    region_b = data.get('region_b', '')
    kb_id = data.get('kb_id', 'all')
    model = data.get('model', DEFAULT_MODEL)

    if not entity_name or not region_a or not region_b:
        return jsonify({"error": "entity, region_a, region_b 均不能为空"}), 400

    try:
        result = cross_region_compare(entity_name, region_a, region_b, kb_id, model)
        return jsonify({"status": "success", "comparison": result})
    except Exception as e:
        return jsonify({"error": f"跨区域对比失败: {str(e)}"}), 500


@cross_region_bp.route('/strat/synonyms', methods=['POST'])
def detect_synonyms_endpoint():
    """同义异名检测"""
    data = request.get_json(silent=True) or {}
    entity_name = data.get('entity', '')
    kb_id = data.get('kb_id', 'all')
    use_llm = data.get('use_llm', False)

    if not entity_name:
        return jsonify({"error": "实体名称不能为空"}), 400

    candidates = detect_synonyms(entity_name, kb_id)
    results = []

    if use_llm and candidates:
        verified = detect_by_llm(entity_name, candidates)
        results = verified
    else:
        results = candidates

    return jsonify({
        "entity": entity_name,
        "candidates": results,
        "total": len(results)
    })
