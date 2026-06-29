# services/query_classifier.py — 查询类型分类: FACTUAL / COMPARATIVE / REASONING / SPATIAL
import time

from flask import Blueprint, request, jsonify

query_classifier_bp = Blueprint('query_classifier', __name__)

# 查询类型与检索策略映射
QUERY_TYPES = {
    "FACTUAL": {
        "label": "事实型",
        "description": "查询具体地质事实（如某地层岩性、厚度、年代）",
        "vector_ratio": 0.8,
        "graph_ratio": 0.2,
        "max_hops": 1,
        "keywords": ["是什么", "什么岩性", "多厚", "厚度", "年代", "属于", "哪里", "位于",
                      "岩性是什么", "什么组", "哪个群", "化石", "定年"]
    },
    "COMPARATIVE": {
        "label": "对比型",
        "description": "比较两个或多个地层单位（如栖霞组vs茅口组）",
        "vector_ratio": 0.5,
        "graph_ratio": 0.5,
        "max_hops": 2,
        "keywords": ["对比", "比较", "差异", "区别", "异同", "vs", "和", "与", "之间",
                      "哪个更", "有何不同", "异名", "同义"]
    },
    "REASONING": {
        "label": "推理型",
        "description": "需要多跳推理的复杂地质问题（如构造演化、沉积相分析）",
        "vector_ratio": 0.3,
        "graph_ratio": 0.7,
        "max_hops": 3,
        "keywords": ["为什么", "原因", "演化", "形成", "成因", "分析", "推断", "推测",
                      "机制", "如何形成", "构造", "沉积环境"]
    },
    "SPATIAL": {
        "label": "空间型",
        "description": "涉及空间分布、地理位置的问题",
        "vector_ratio": 0.6,
        "graph_ratio": 0.4,
        "max_hops": 2,
        "keywords": ["分布", "出露", "在哪里", "位置", "区域", "空间", "地理", "剖面",
                      "覆盖", "范围", "延伸", "走向", "倾向"]
    }
}


def classify_by_keywords(question):
    """基于关键词规则快速分类（无需 LLM 调用）"""
    scores = {}
    for qtype, qdef in QUERY_TYPES.items():
        score = 0
        for kw in qdef["keywords"]:
            if kw in question:
                score += 1
        scores[qtype] = score

    if sum(scores.values()) == 0:
        return "FACTUAL"

    best = max(scores, key=scores.get)
    return best


def classify_by_llm(question, model="qwen3.5-plus"):
    """使用 LLM 进行查询类型分类"""
    from .model_router import get_model_response

    prompt = f"""你是一个地质查询分类器。请将用户问题分类为以下类型之一：FACTUAL, COMPARATIVE, REASONING, SPATIAL。

类型定义：
- FACTUAL (事实型): 查询具体地质事实，如某地层岩性、厚度、年代、归属
- COMPARATIVE (对比型): 比较两个或多个地层单位的异同
- REASONING (推理型): 需要多跳推理的复杂地质问题，如构造演化、成因分析
- SPATIAL (空间型): 涉及空间分布、地理位置的问题

只返回类型名称（一个单词），不要输出任何其他内容。

用户问题：{question}
"""
    try:
        result = get_model_response(prompt, model).strip().upper()
        if result in QUERY_TYPES:
            return result
        return "FACTUAL"
    except Exception:
        return classify_by_keywords(question)


def get_retrieval_config(question, use_llm=False, model=None):
    """
    根据查询类型返回最优检索配置。
    返回: {"query_type": str, "config": dict, "method": "keyword"|"llm"}
    """
    if use_llm:
        query_type = classify_by_llm(question, model or "qwen3.5-plus")
        method = "llm"
    else:
        query_type = classify_by_keywords(question)
        method = "keyword"

    config = {
        "query_type": query_type,
        "query_type_label": QUERY_TYPES[query_type]["label"],
        "vector_ratio": QUERY_TYPES[query_type]["vector_ratio"],
        "graph_ratio": QUERY_TYPES[query_type]["graph_ratio"],
        "max_hops": QUERY_TYPES[query_type]["max_hops"],
        "method": method
    }
    return config


# ============================================================
# API
# ============================================================


@query_classifier_bp.route('/query/classify', methods=['POST'])
def classify_endpoint():
    """查询类型分类接口"""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    use_llm = data.get("use_llm", False)
    model = data.get("model", "QwQ-32B")

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    config = get_retrieval_config(question, use_llm=use_llm, model=model)

    return jsonify({
        "status": "success",
        "question": question,
        "classification": config
    })
