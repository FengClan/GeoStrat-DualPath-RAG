# services/strat_service.py — 多源证据融合地层对比
import time
import json
import re
import logging

from flask import Blueprint, request, jsonify

from .model_router import get_model_response
from .kg_service import load_graph_for_kb, retrieve_graph_context
from .query_classifier import get_retrieval_config
from .logger import sanitize

logger = logging.getLogger(__name__)

strat_bp = Blueprint('strat', __name__)
DEFAULT_MODEL = "qwen3.5-plus"


# ============================================================
# 多源证据收集
# ============================================================


def collect_evidence(entity_name, kb_id="all"):
    """
    从多个数据源收集地层相关证据。
    数据源: 1) NetworkX 知识图谱  2) ChromaDB 向量库  3) 规则/启发式
    返回: [{source_type, source_name, evidence_text, confidence, metadata}, ...]
    """
    evidence_list = []

    # 源1: 知识图谱 — 实体为中心的1-hop子图
    try:
        G = load_graph_for_kb(kb_id)
        if G.number_of_nodes() > 0:
            matched = None
            for node in G.nodes():
                if str(node) == entity_name or entity_name in str(node):
                    matched = str(node)
                    break
            if matched and G.has_node(matched):
                for neighbor in (set(G.successors(matched)) | set(G.predecessors(matched)) if G.is_directed()
                                 else set(G.neighbors(matched))):
                    edge_data = G.get_edge_data(matched, neighbor) or G.get_edge_data(neighbor, matched) or {}
                    rel = edge_data.get('relation', '关联')
                    ev = f"KG: {matched} --[{rel}]--> {neighbor}"
                    evidence_list.append({
                        "source_type": "knowledge_graph",
                        "source_name": f"KG:{kb_id}",
                        "evidence_text": ev,
                        "confidence": 0.9,
                        "metadata": {"relation": rel, "target": neighbor}
                    })
    except Exception as e:
        logger.warning("[Evidence] KG source failed for %s: %s", sanitize(str(entity_name)), sanitize(str(e)))

    # 源2: ChromaDB 向量检索 — 文献片段
    try:
        from rag_core import retrieve_from_vectordb
        docs = retrieve_from_vectordb(entity_name, kb_id, top_k=5)
        if docs and len(docs) > 0:
            for doc in docs[:5]:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                evidence_list.append({
                    "source_type": "vector_db",
                    "source_name": doc.metadata.get("source_file", "文献") if hasattr(doc, 'metadata') else "文献",
                    "evidence_text": content[:300],
                    "confidence": 0.7,
                    "metadata": {"semantic_type": doc.metadata.get("semantic_type", "") if hasattr(doc, 'metadata') else ""}
                })
    except Exception as e:
        logger.warning("[Evidence] Vector source failed for %s: %s", sanitize(str(entity_name)), sanitize(str(e)))

    # 源3: 启发式规则 — 名称模式匹配
    heuristic_evidence = _heuristic_evidence(entity_name)
    if heuristic_evidence:
        evidence_list.extend(heuristic_evidence)

    return evidence_list


def _heuristic_evidence(entity_name):
    """基于名称模式的启发式证据"""
    evidence = []

    # 检测地层等级
    if entity_name.endswith("群"):
        evidence.append({
            "source_type": "heuristic",
            "source_name": "命名规则",
            "evidence_text": f"{entity_name} 是最高级别岩石地层单位（群），通常包含多个组",
            "confidence": 0.85,
            "metadata": {"level": "GROUP"}
        })
    elif entity_name.endswith("组"):
        evidence.append({
            "source_type": "heuristic",
            "source_name": "命名规则",
            "evidence_text": f"{entity_name} 是基本岩石地层单位（组）",
            "confidence": 0.85,
            "metadata": {"level": "FORMATION"}
        })
    elif entity_name.endswith("段"):
        evidence.append({
            "source_type": "heuristic",
            "source_name": "命名规则",
            "evidence_text": f"{entity_name} 是组的细分单位（段）",
            "confidence": 0.85,
            "metadata": {"level": "MEMBER"}
        })

    return evidence


# ============================================================
# 多阶段证据分析流水线
# ============================================================


def build_multistage_prompt(entity_a, feat_a, evidence_a, entity_b, feat_b, evidence_b):
    """构建多阶段地层对比分析提示词"""

    # 证据源汇总
    def format_evidence(ev_list, max_per_source=3):
        lines = []
        for i, ev in enumerate(ev_list[:max_per_source * 4], 1):
            lines.append(f"  E{i}. [{ev['source_type']}] {ev['source_name']}: {ev['evidence_text']}")
        return "\n".join(lines) if lines else "  暂无证据"

    ev_a_formatted = format_evidence(evidence_a)
    ev_b_formatted = format_evidence(evidence_b)

    prompt = f"""你是一位经验丰富的地质学家。请基于以下多源证据对 "{entity_a}" 和 "{entity_b}" 进行四阶段系统对比分析。

## 阶段1: 证据收集 (Evidence Collection)
### {entity_a} 的证据
{ev_a_formatted}

### {entity_b} 的证据
{ev_b_formatted}

## 阶段2: 单源分析 (Per-Source Analysis)
分析每个证据源揭示的地层特征。

### {entity_a} 的特征数据
{json.dumps(feat_a, ensure_ascii=False, indent=2)}

### {entity_b} 的特征数据
{json.dumps(feat_b, ensure_ascii=False, indent=2)}

## 阶段3: 跨源综合 (Cross-Source Synthesis)
综合所有证据源，识别一致性和矛盾之处。

## 阶段4: 最终对比 (Final Comparison)
从以下维度进行系统对比:
1. **岩性组合**: 主要岩石类型、组合特征
2. **颜色特征**: 代表性颜色、风化色
3. **沉积环境/构造背景**: 沉积相、构造单元
4. **古生物/化石**: 化石组合、年代指示
5. **厚度分布**: 厚度范围、空间变化

## 输出格式 (严格JSON)
```json
{{
    "evidence_summary": {{
        "entity_a": {{ "total_sources": {len(evidence_a)}, "source_types": [] }},
        "entity_b": {{ "total_sources": {len(evidence_b)}, "source_types": [] }}
    }},
    "dimensions": [
        {{
            "label": "维度名称",
            "col_a": {{ "name": "{entity_a}", "tags_common": [], "tags_diff": [], "source_refs": [] }},
            "col_b": {{ "name": "{entity_b}", "tags_common": [], "tags_diff": [], "source_refs": [] }}
        }}
    ],
    "cross_source_analysis": {{
        "consistent_findings": ["多个来源一致支持的特征"],
        "contradictions": ["不同来源存在矛盾的发现"],
        "source_reliability": {{"knowledge_graph": 0.9, "vector_db": 0.7, "heuristic": 0.85}}
    }},
    "conclusion": "总结性结论文本...",
    "confidence": 0.85
}}
```

请仅返回JSON，不要包含额外解释。"""
    return prompt


# ============================================================
# API Routes
# ============================================================


@strat_bp.route('/strat/analyze', methods=['POST'])
def strat_analyze():
    """多源证据融合地层对比分析"""
    logger.info("--- Stratigraphic Analysis with Multi-Source Evidence ---")
    try:
        data = request.get_json(silent=True) or {}
        name_a = data.get('entityA_name', '地层A')
        feat_a = data.get('entityA_data', {})
        name_b = data.get('entityB_name', '地层B')
        feat_b = data.get('entityB_data', {})
        model_name = data.get('model', DEFAULT_MODEL)
        kb_id = data.get('kb_id', 'all')

        # Stage 1: Collect multi-source evidence
        logger.info("[Strat] Collecting evidence for '%s' and '%s'", sanitize(str(name_a)), sanitize(str(name_b)))
        evidence_a = collect_evidence(name_a, kb_id)
        evidence_b = collect_evidence(name_b, kb_id)

        # Stage 2-4: Build prompt & call LLM
        prompt = build_multistage_prompt(name_a, feat_a, evidence_a, name_b, feat_b, evidence_b)
        response_text = get_model_response(prompt, model_name)

        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        result_data = json.loads(cleaned)

        # Attach raw evidence for citation display
        result_data["raw_evidence"] = {
            "entity_a": [{"source": e["source_name"], "type": e["source_type"],
                          "text": e["evidence_text"][:200]} for e in evidence_a[:10]],
            "entity_b": [{"source": e["source_name"], "type": e["source_type"],
                          "text": e["evidence_text"][:200]} for e in evidence_b[:10]]
        }

        return jsonify(result_data)

    except json.JSONDecodeError as e:
        logger.error("[ERROR] LLM returned invalid JSON: %s", sanitize(str(e)))
        return jsonify({
            "dimensions": [],
            "conclusion": f"AI 分析结果格式解析失败",
            "error": str(e)
        }), 200
    except Exception as e:
        logger.error("[ERROR] Strat Analysis Failed: %s", sanitize(str(e)))
        return jsonify({"error": str(e)}), 500


@strat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    """地层对比场景下的智能问答"""
    try:
        data = request.get_json(silent=True) or {}
        user_input = data.get('question', '')
        context = data.get('context', '')
        model_name = data.get('model', DEFAULT_MODEL)
        kb_id = data.get('kb_id', 'all')

        if not user_input:
            return jsonify({"error": "Question is required"}), 400

        # 收集证据作为上下文
        prompt_parts = ["你是一个地质学专家助手。用户正在进行地层对比分析。"]
        if context:
            prompt_parts.append(f"\n当前对比上下文:\n{context}")

        # 尝试从问题中提取地层名并收集相关证据
        evidence_context = []
        formation_pattern = r'\S{1,6}(?:群|组|段)'
        formations = re.findall(formation_pattern, user_input)
        for fm in formations[:2]:
            ev_list = collect_evidence(fm, kb_id)
            for ev in ev_list[:3]:
                evidence_context.append(f"[{ev['source_type']}] {ev['evidence_text']}")

        if evidence_context:
            prompt_parts.append(f"\n相关证据:\n" + "\n".join(evidence_context))

        prompt_parts.append(f"\n\n问题：{user_input}\n请给出专业的回答。")
        prompt = "\n".join(prompt_parts)

        answer = get_model_response(prompt, model_name)

        return jsonify({
            "answer": answer,
            "evidence_used": len(evidence_context),
            "status": "success"
        })
    except Exception as e:
        logger.error("Chat Error: %s", sanitize(str(e)))
        return jsonify({"error": str(e)}), 500


@strat_bp.route('/strat/evidence', methods=['POST'])
def get_evidence():
    """获取指定实体的多源证据列表（供前端证据面板使用）"""
    data = request.get_json(silent=True) or {}
    entity_name = data.get('entity', '')
    kb_id = data.get('kb_id', 'all')

    if not entity_name:
        return jsonify({"error": "实体名称不能为空"}), 400

    evidence_list = collect_evidence(entity_name, kb_id)

    return jsonify({
        "entity": entity_name,
        "total_sources": len(evidence_list),
        "evidence": evidence_list,
        "by_source": {
            "knowledge_graph": len([e for e in evidence_list if e["source_type"] == "knowledge_graph"]),
            "vector_db": len([e for e in evidence_list if e["source_type"] == "vector_db"]),
            "heuristic": len([e for e in evidence_list if e["source_type"] == "heuristic"]),
        }
    })
