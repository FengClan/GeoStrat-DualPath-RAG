# services/kg_extraction.py — 本体约束的知识图谱三元组抽取
import re
import json
import time
import logging
from flask import Blueprint, request, jsonify

from .model_router import get_model_response
from .ontology import (
    ENTITY_TYPES, RELATION_TYPES, HIERARCHY_ORDER,
    validate_triple, validate_triples_batch, get_entity_type_suggestions
)
from .logger import sanitize

logger = logging.getLogger(__name__)

kg_extraction_bp = Blueprint('kg_extraction', __name__)

DEFAULT_MODEL = "qwen3.5-plus"

# ============================================================
# Ontology-Aware Prompt Builder
# ============================================================


def build_ontology_prompt():
    """构建本体约束描述文本，嵌入 LLM 提示"""
    lines = ["## 实体类型定义 (Entity Types)"]
    for etype, edef in ENTITY_TYPES.items():
        examples = "、".join(edef.get("examples", [])[:3])
        lines.append(f"- **{etype}** ({edef['label']}): {edef['description']}。如: {examples}")

    lines.append("\n## 关系类型定义 (Relation Types)")
    for rname, rdef in RELATION_TYPES.items():
        domain = "、".join(rdef.get("domain", []))
        rng = "、".join(rdef.get("range", []))
        lines.append(f"- **{rname}** ({rdef['label']}): {rdef['description']}。Subject类型: [{domain}] → Object类型: [{rng}]")

    return "\n".join(lines)


def build_extraction_prompt(text, source_name=""):
    """构建三元组抽取的完整提示词"""
    ontology_schema = build_ontology_prompt()

    prompt = f"""你是一个地质领域知识图谱构建专家。请从以下地质文献文本中抽取知识图谱三元组。

{ontology_schema}

## 抽取要求
1. 识别文本中的地质实体（群、组、段、岩石类型、地质年代、地理位置等），并为其标注正确的实体类型
2. 识别实体之间的关系，关系类型必须从上表中选择
3. 每个三元组必须包含证据原文（原文中支撑该三元组的句子片段）
4. 给出每个三元组的置信度 (0.0 ~ 1.0)

## 输出格式 (严格JSON数组)
```json
[
  {{
    "entity1": "实体1名称",
    "entity1_type": "ENTITY_TYPE",
    "relation": "RELATION_TYPE",
    "entity2": "实体2名称",
    "entity2_type": "ENTITY_TYPE",
    "evidence": "原文证据片段",
    "confidence": 0.9
  }}
]
```

## 待抽取文本
{text}

请严格按照上述JSON格式输出，不要添加额外的解释。"""
    return prompt


# ============================================================
# Response Parser
# ============================================================


def parse_extraction_response(raw_response):
    """解析 LLM 返回的 JSON 数组为三元组列表"""
    # 尝试直接解析 JSON
    try:
        data = json.loads(raw_response)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "triples" in data:
            return data["triples"]
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 数组
    arr_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw_response)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def normalize_triple(triple):
    """标准化三元组字段名"""
    return {
        "entity1": str(triple.get("entity1", triple.get("head", ""))).strip(),
        "entity1_type": str(triple.get("entity1_type", triple.get("head_type", ""))).strip().upper(),
        "relation": str(triple.get("relation", triple.get("rel", ""))).strip().upper(),
        "entity2": str(triple.get("entity2", triple.get("tail", ""))).strip(),
        "entity2_type": str(triple.get("entity2_type", triple.get("tail_type", ""))).strip().upper(),
        "evidence": str(triple.get("evidence", triple.get("source", triple.get("span", "")))).strip(),
        "confidence": float(triple.get("confidence", 0.5))
    }


# ============================================================
# Extraction API Route
# ============================================================


@kg_extraction_bp.route('/kg/extract', methods=['POST'])
def extract_triples():
    """
    从文本中抽取知识图谱三元组（本体约束）
    请求体: { "text": "...", "source_name": "...", "model": "..." }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    source_name = data.get("source_name", "")
    model = data.get("model", DEFAULT_MODEL)

    if not text or not text.strip():
        return jsonify({"error": "输入文本不能为空"}), 400

    logger.info("[KG Extraction] 开始抽取，文本长度: %d 字符", len(text))

    try:
        prompt = build_extraction_prompt(text, source_name)
        raw_response = get_model_response(prompt, model)
        raw_triples = parse_extraction_response(raw_response)

        triples = [normalize_triple(t) for t in raw_triples]

        # 本体验证
        validated_triples = []
        for t in triples:
            valid, err, warns = validate_triple(
                t["entity1"], t["entity1_type"],
                t["relation"],
                t["entity2"], t["entity2_type"]
            )
            t["valid"] = valid
            if err:
                t["error"] = err
            if warns:
                t["warnings"] = warns
            validated_triples.append(t)

        valid_count = sum(1 for t in validated_triples if t["valid"])
        invalid_count = len(validated_triples) - valid_count

        stats = {
            "total": len(validated_triples),
            "valid": valid_count,
            "invalid": invalid_count,
            "accuracy": round(valid_count / max(len(validated_triples), 1), 3)
        }

        logger.info("[KG Extraction] 完成: %s", stats)

        return jsonify({
            "status": "success",
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            "triples": validated_triples,
            "stats": stats,
            "raw_response": raw_response if data.get("debug") else None
        })

    except Exception as e:
        logger.error("[KG Extraction] failed: %s", sanitize(str(e)))
        return jsonify({"error": f"三元组抽取失败: {str(e)}"}), 500


@kg_extraction_bp.route('/kg/validate', methods=['POST'])
def validate_triples_endpoint():
    """验证前端传入的三元组列表"""
    data = request.get_json(silent=True) or {}
    triples_input = data.get("triples", [])

    if not triples_input:
        return jsonify({"error": "三元组列表不能为空"}), 400

    normalized = [normalize_triple(t) for t in triples_input]
    result = validate_triples_batch(
        [(t["entity1"], t["entity1_type"], t["relation"], t["entity2"], t["entity2_type"],
          t.get("evidence"), t.get("confidence")) for t in normalized]
    )

    return jsonify({
        "status": "success",
        "validation": result
    })


@kg_extraction_bp.route('/kg/ontology', methods=['GET'])
def get_ontology():
    """获取完整本体定义，供前端使用"""
    return jsonify({
        "entity_types": {k: {"label": v["label"], "description": v["description"], "examples": v["examples"]}
                         for k, v in ENTITY_TYPES.items()},
        "relation_types": RELATION_TYPES,
        "hierarchy": {k: v for k, v in sorted(HIERARCHY_ORDER.items(), key=lambda x: x[1])}
    })


@kg_extraction_bp.route('/kg/entity-suggest', methods=['POST'])
def suggest_entity_type():
    """对输入的实体名称推荐可能的类型"""
    data = request.get_json(silent=True) or {}
    entity_name = data.get("entity_name", "")
    if not entity_name:
        return jsonify({"error": "实体名称不能为空"}), 400

    suggestions = get_entity_type_suggestions(entity_name)
    return jsonify({
        "entity_name": entity_name,
        "suggested_types": suggestions
    })
