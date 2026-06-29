# services/rag_service.py — 双路检索增强: 自适应融合 + 细粒度引用 + 真实评分
import re
import time
import hashlib
import logging

from flask import Blueprint, request, jsonify

from .model_router import get_model_response
from .kg_service import retrieve_graph_context, load_graph_for_kb
from .query_classifier import get_retrieval_config
from .logger import sanitize
from rag_core import retrieve_from_vectordb

logger = logging.getLogger(__name__)

rag_bp = Blueprint('rag', __name__)

# ============================================================
# 自适应上下文构建
# ============================================================


def build_adaptive_prompt(question, vector_docs, graph_contexts, query_config):
    """根据查询类型融合向量检索和图谱检索的上下文，构建加权提示"""
    vec_weight = query_config.get("vector_ratio", 0.6)
    graph_weight = query_config.get("graph_ratio", 0.4)
    qtype = query_config.get("query_type", "FACTUAL")
    qtype_label = query_config.get("query_type_label", "事实型")

    parts = []
    parts.append(f"【问题类型】{qtype} — {qtype_label}\n")

    # 图谱上下文（多跳推理路径）
    if graph_contexts:
        graph_section = "【知识图谱推理路径】\n"
        for i, ctx in enumerate(graph_contexts, 1):
            graph_section += f"  G{i}. {ctx}\n"
        parts.append(graph_section)

    # 向量检索上下文（文档片段）
    if vector_docs:
        vec_section = "【地质文献参考片段】\n"
        for i, doc in enumerate(vector_docs, 1):
            source = doc.metadata.get("source_file", "未知")
            s_type = doc.metadata.get("semantic_type", "")
            type_tag = f"[{s_type}]" if s_type else ""
            vec_section += f"  D{i}. {type_tag}(来源: {source}):\n     {doc.page_content[:500]}\n"
        parts.append(vec_section)

    context = "\n".join(parts)

    role_guide = {
        "FACTUAL": "请基于参考资料中的事实信息直接回答。如参考资料不足，请明确指出。",
        "COMPARATIVE": "请从多个维度（岩性、厚度、年代、分布、化石等）进行系统对比。引用具体来源。",
        "REASONING": "请基于图谱推理路径和文献片段进行多步推理分析，展示推理链条。",
        "SPATIAL": "请关注空间分布信息，结合图谱中的 DISTRIBUTED_IN 关系和文献描述。"
    }

    prompt = f"""{context}

{role_guide.get(qtype, role_guide["FACTUAL"])}

【提示词权重】向量检索权重={vec_weight}, 图谱权重={graph_weight}

【用户问题】{question}

请回答上述问题，并在回答中以 [D1][G2] 等标记注明信息来源。"""
    return prompt


# ============================================================
# 细粒度引用解析
# ============================================================


def parse_citations_in_answer(answer, vector_docs, graph_contexts):
    """
    解析 LLM 回复中的引用标记（如 [D1], [G2]），
    生成 answer_fragments 列表，建立答案片段→来源的映射。
    """
    # 按句子分割答案
    sentences = re.split(r'(?<=[。！？\n])', answer)
    fragments = []

    for sent in sentences:
        if not sent.strip():
            continue
        # 提取引用标记
        cited = []
        for m in re.finditer(r'\[(D|G)(\d+)\]', sent):
            prefix = m.group(1)
            idx = int(m.group(2)) - 1
            if prefix == "D" and idx < len(vector_docs):
                doc = vector_docs[idx]
                cited.append({
                    "source_type": "document",
                    "doc_name": doc.metadata.get("source_file", "未知"),
                    "excerpt": doc.page_content[:200],
                    "index": idx
                })
            elif prefix == "G" and idx < len(graph_contexts):
                cited.append({
                    "source_type": "graph",
                    "path": graph_contexts[idx],
                    "index": idx
                })

        fragments.append({
            "text": sent.strip(),
            "citations": cited,
            "has_citation": len(cited) > 0
        })

    return fragments


# ============================================================
# 真实可信度评分
# ============================================================


def compute_reliability_scores(answer, fragments, vector_docs, graph_contexts, query_config):
    """
    计算真实可信度评分指标 (替代 Math.random 模拟)。
    评分维度：
    - citation_density: 有引用支撑的答案比例
    - source_coverage: 使用了多少不同的来源
    - graph_completeness: 图谱推理路径是否完整
    - confidence: 加权综合可信度
    """
    n_fragments = max(len(fragments), 1)
    n_cited = sum(1 for f in fragments if f["has_citation"])

    # 引用密度
    citation_density = round(n_cited / n_fragments, 4)

    # 来源覆盖度
    cited_sources = set()
    for f in fragments:
        for c in f.get("citations", []):
            cited_sources.add(c.get("doc_name", c.get("path", "unknown")))
    source_coverage = round(len(cited_sources) / max(n_fragments, 1), 4)

    # 答案置信度 (基于引用标记密度)
    cited_markers = len(re.findall(r'\[[DG]\d+\]', answer))
    total_chars = max(len(answer), 1)
    attribution_density = min(cited_markers / max(total_chars / 100, 1), 1.0)

    # 图谱完备度
    if graph_contexts:
        graph_completeness = min(len(graph_contexts) / max(query_config.get("max_hops", 2), 1), 1.0)
    else:
        graph_completeness = 0.5 if vector_docs else 0.0

    # 综合评分
    overall = round(
        citation_density * 0.35 +
        source_coverage * 0.20 +
        attribution_density * 0.25 +
        graph_completeness * 0.20,
        4
    )

    return {
        "citation_density": citation_density,
        "source_coverage": source_coverage,
        "attribution_density": round(attribution_density, 4),
        "graph_completeness": round(graph_completeness, 4),
        "overall_reliability": overall
    }


# ============================================================
# Main GeoQA Route
# ============================================================


@rag_bp.route('/geoqa', methods=['POST'])
def geo_qa():
    data = request.get_json(silent=True) or {}
    question = data.get('question', '')
    model_name = data.get('model', 'qwen3.5-plus')
    use_rag = data.get('use_rag', True)
    kb_id = data.get('kb_type', data.get('kb_id', 'all'))
    rag_strategy = data.get('rag_strategy', 'adaptive')
    use_llm_classify = data.get('use_llm_classify', False)

    if not question.strip():
        return jsonify({"error": "问题不能为空"}), 400

    logger.info("[QA] Question: %s...", sanitize(question[:100]))
    logger.info("[QA] Strategy: %s, Model: %s, KB: %s", sanitize(rag_strategy), sanitize(model_name), sanitize(str(kb_id)))

    # 步骤1: 查询分类
    query_config = get_retrieval_config(question, use_llm=use_llm_classify, model=model_name)
    logger.info("[QA] Query type: %s (%s), Vec=%s, Graph=%s",
                query_config['query_type_label'], query_config['query_type'],
                query_config['vector_ratio'], query_config['graph_ratio'])

    # 步骤2: 检索（向量 + 图谱）
    vector_docs = []
    graph_contexts = []
    citations_base = []

    if use_rag:
        # 向量检索
        if rag_strategy != 'graph':
            try:
                vec_text, vec_citations = retrieve_from_vectordb(question, kb_id, top_k=8)
                # 需要获取原始 Document 列表用于引用解析
                vector_docs = _retrieve_docs(question, kb_id, top_k=8)
                citations_base.extend(vec_citations)
            except Exception as e:
                logger.warning("[QA] Vector retrieval failed: %s", sanitize(str(e)))

        # 图谱检索
        if rag_strategy != 'naive':
            try:
                graph_contexts, graph_citations = retrieve_graph_context(
                    question, kb_id,
                    max_hops=query_config.get("max_hops", 2)
                )
                citations_base.extend(graph_citations)
            except Exception as e:
                logger.warning("[QA] Graph retrieval failed: %s", sanitize(str(e)))

    # 步骤3: 构建自适应 Prompt
    if use_rag:
        final_prompt = build_adaptive_prompt(question, vector_docs, graph_contexts, query_config)
        final_prompt += "\n\n请在你的回答中使用 [D1][D2] 标记引用文档片段，用 [G1][G2] 标记引用图谱推理路径。"
    else:
        final_prompt = question

    # 步骤4: 生成回答
    answer = get_model_response(final_prompt, model_name)

    # 步骤5: 解析细粒度引用
    fragments = parse_citations_in_answer(answer, vector_docs, graph_contexts)

    # 步骤6: 计算真实可信度评分
    metrics = compute_reliability_scores(answer, fragments, vector_docs, graph_contexts, query_config)

    # 步骤7: 构建返回结果
    result = {
        "answer": answer,
        "fragments": fragments,
        "citations": citations_base[:10],
        "query_type": query_config,
        "retrieval_stats": {
            "vector_docs_count": len(vector_docs),
            "graph_paths_count": len(graph_contexts),
            "strategy": rag_strategy
        },
        "metrics": metrics,
        "status": "success"
    }

    logger.info("[QA] Done: reliability=%s, fragments=%d, citations=%d",
                metrics['overall_reliability'], len(fragments), len(citations_base[:10]))

    return jsonify(result)


# ============================================================
# Helper: 获取原始 Document 对象列表
# ============================================================


def _retrieve_docs(query, kb_id, top_k=8):
    """返回 LangChain Document 对象列表（用于细粒度引用）"""
    from rag_core import vectordb
    filter_dict = {}
    if kb_id and kb_id != "all":
        filter_dict["kb_id"] = kb_id
    try:
        if filter_dict:
            docs = vectordb.similarity_search(query, k=top_k, filter=filter_dict)
        else:
            docs = vectordb.similarity_search(query, k=top_k)
        return docs
    except Exception:
        return []


# ============================================================
# Reasoning Path Visualization Endpoint
# ============================================================


@rag_bp.route('/geoqa/reasoning-path', methods=['POST'])
def reasoning_path():
    """返回问题对应的图谱推理路径，用于前端可视化"""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    kb_id = data.get("kb_id", "all")
    max_hops = data.get("max_hops", 3)

    if not question.strip():
        return jsonify({"error": "问题不能为空"}), 400

    G = load_graph_for_kb(kb_id)
    if G.number_of_nodes() == 0:
        return jsonify({"status": "success", "nodes": [], "edges": [], "message": "图谱为空"})

    # 找到匹配的实体节点
    matched = []
    for node in G.nodes():
        node_str = str(node)
        for token in question:
            if token in node_str:
                matched.append(node_str)
                break

    if not matched:
        return jsonify({"status": "success", "nodes": [], "edges": [], "message": "未匹配到实体"})

    # BFS 多跳遍历构建子图
    visited = set(matched)
    layer = set(matched)
    nodes_data = []
    edges_data = []

    for hop in range(max_hops):
        next_layer = set()
        for node in layer:
            if G.is_directed():
                neighbors = set(G.successors(node)) | set(G.predecessors(node))
            else:
                neighbors = set(G.neighbors(node))

            for nb in neighbors:
                edge_data = G.get_edge_data(node, nb)
                if edge_data is None:
                    edge_data = G.get_edge_data(nb, node)

                if nb not in visited:
                    nodes_data.append({
                        "name": str(nb),
                        "hop": hop + 1,
                        "attributes": dict(G.nodes.get(nb, {}))
                    })

                edges_data.append({
                    "source": str(node),
                    "target": str(nb),
                    "relation": edge_data.get("relation", "关联") if edge_data else "关联",
                    "hop": hop
                })

                if nb not in visited:
                    next_layer.add(str(nb))
                    visited.add(str(nb))

        layer = next_layer

    # 添加起始节点
    for m in matched:
        nodes_data.append({
            "name": str(m),
            "hop": 0,
            "is_start": True,
            "attributes": dict(G.nodes.get(m, {}))
        })

    return jsonify({
        "status": "success",
        "matched_entities": matched,
        "nodes": nodes_data,
        "edges": edges_data,
        "total_nodes": len(nodes_data),
        "total_edges": len(edges_data)
    })
