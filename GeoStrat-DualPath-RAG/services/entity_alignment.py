# services/entity_alignment.py — 多源实体对齐: 精确→模糊→语义→上下文
import re
import json
import time
import logging
from difflib import SequenceMatcher
from collections import Counter

import networkx as nx
from flask import Blueprint, request, jsonify

from .ontology import ENTITY_TYPES, HIERARCHY_ORDER
from .kg_service import load_graph_for_kb

entity_alignment_bp = Blueprint('entity_alignment', __name__)

log = logging.getLogger(__name__)

# 尝试加载 sentence-transformers 用于语义匹配
_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("[INFO] Entity alignment: loaded sentence-transformers model.")
        except Exception as e:
            print(f"[WARN] Cannot load sentence-transformers for entity alignment: {e}")
            _semantic_model = False
    return _semantic_model if _semantic_model is not False else None


# ============================================================
# String Similarity
# ============================================================


def normalize_entity_name(name):
    """标准化实体名称用于比较"""
    if not name:
        return ""
    name = str(name).strip()
    # 去除括号中的补充说明用于比较
    # e.g. "栖霞组（福建）" -> "栖霞组"
    name = re.sub(r'[（(][^)）]*[)）]', '', name).strip()
    return name


def exact_match(name1, name2):
    """精确匹配：标准化后完全相同"""
    return normalize_entity_name(name1) == normalize_entity_name(name2)


def fuzzy_similarity(name1, name2):
    """模糊匹配：使用 SequenceMatcher 计算字符串相似度"""
    n1 = normalize_entity_name(name1)
    n2 = normalize_entity_name(name2)
    if not n1 or not n2:
        return 0.0
    return SequenceMatcher(None, n1, n2).ratio()


def edit_distance_similarity(name1, name2):
    """编辑距离相似度"""
    n1 = normalize_entity_name(name1)
    n2 = normalize_entity_name(name2)
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0

    # Levenshtein distance
    m, n = len(n1), len(n2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if n1[i - 1] == n2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    max_len = max(m, n)
    return 1.0 - (dp[m][n] / max_len)


def semantic_similarity(name1, name2, model=None):
    """语义相似度：使用 embedding 计算余弦相似度"""
    if model is None:
        model = _get_semantic_model()
    if model is None or model is False:
        return None  # 模型不可用

    try:
        embeddings = model.encode([name1, name2], convert_to_numpy=True)
        from numpy import dot
        from numpy.linalg import norm
        cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(cos_sim)
    except Exception:
        return None


# ============================================================
# Context Similarity (Graph Neighborhood)
# ============================================================


def context_similarity(entity1, entity2, graph):
    """基于图结构的上下文相似度：比较邻居节点集合的 Jaccard 相似度"""
    if graph is None or graph.number_of_nodes() == 0:
        return None

    # 找到图中的对应节点
    node1 = _find_node(graph, entity1)
    node2 = _find_node(graph, entity2)

    if node1 is None or node2 is None:
        return None

    # 获取邻居集合
    if graph.is_directed():
        neighbors1 = set(graph.successors(node1)) | set(graph.predecessors(node1))
        neighbors2 = set(graph.successors(node2)) | set(graph.predecessors(node2))
    else:
        neighbors1 = set(graph.neighbors(node1))
        neighbors2 = set(graph.neighbors(node2))

    if not neighbors1 and not neighbors2:
        return None

    # Jaccard similarity
    intersection = neighbors1 & neighbors2
    union = neighbors1 | neighbors2
    if not union:
        return None
    return len(intersection) / len(union)


def _find_node(graph, entity_name):
    """在图实体中查找匹配节点（支持部分匹配）"""
    n = normalize_entity_name(entity_name)
    for node in graph.nodes():
        if normalize_entity_name(str(node)) == n:
            return node
    # 尝试部分匹配
    for node in graph.nodes():
        if n in normalize_entity_name(str(node)) or normalize_entity_name(str(node)) in n:
            return node
    return None


# ============================================================
# Multi-Stage Entity Alignment
# ============================================================


def align_entities(source_entities, target_entities, graph=None,
                   exact_threshold=0.95, fuzzy_threshold=0.80,
                   semantic_threshold=0.75, context_threshold=0.3):
    """
    多阶段实体对齐算法。
    source_entities: [{"name": "...", "type": "..."}, ...]
    target_entities: [{"name": "...", "type": "..."}, ...]
    返回: {matches: [...], ambiguous: [...], unmatched: [...], stats: {...}}
    """
    matches = []
    ambiguous = []
    matched_target_indices = set()

    semantic_model = _get_semantic_model()

    for i, src in enumerate(source_entities):
        src_name = src.get("name", "")
        src_type = src.get("type", "")

        best_match = None
        best_score = 0.0
        best_stage = ""
        all_candidates = []

        for j, tgt in enumerate(target_entities):
            if j in matched_target_indices:
                continue

            tgt_name = tgt.get("name", "")
            tgt_type = tgt.get("type", "")
            tgt_score = 0.0

            # Stage 0: 类型过滤 — 仅匹配相同类型的实体
            if src_type and tgt_type and src_type != tgt_type:
                continue

            # Stage 1: 精确匹配
            if exact_match(src_name, tgt_name):
                tgt_score = 1.0
                best_stage = "exact"

            # Stage 2: 模糊匹配
            if best_stage != "exact":
                fuzzy_score = fuzzy_similarity(src_name, tgt_name)
                if fuzzy_score >= exact_threshold:
                    tgt_score = max(tgt_score, fuzzy_score)
                    best_stage = "fuzzy" if best_stage != "exact" else "exact"

            # Stage 3: 编辑距离 (仅在模糊未达阈值时)
            if tgt_score < fuzzy_threshold:
                edit_sim = edit_distance_similarity(src_name, tgt_name)
                tgt_score = max(tgt_score, edit_sim * 0.9)  # 稍低权重

            # Stage 4: 语义相似度
            if tgt_score < semantic_threshold and semantic_model:
                sem_sim = semantic_similarity(src_name, tgt_name, semantic_model)
                if sem_sim is not None:
                    tgt_score = max(tgt_score, sem_sim * 0.85)

            # Stage 5: 上下文相似度
            if graph and tgt_score < fuzzy_threshold:
                ctx_sim = context_similarity(src_name, tgt_name, graph)
                if ctx_sim is not None:
                    tgt_score = max(tgt_score, ctx_sim * 0.7)

            if tgt_score > 0:
                all_candidates.append({
                    "target_index": j,
                    "target_name": tgt_name,
                    "target_type": tgt_type,
                    "score": round(tgt_score, 4),
                    "stage": best_stage if tgt_score == best_score else "combined"
                })

            if tgt_score > best_score:
                best_score = tgt_score
                best_match = {
                    "source_name": src_name,
                    "source_type": src_type,
                    "target_name": tgt_name,
                    "target_index": j,
                    "target_type": tgt_type,
                    "score": round(tgt_score, 4),
                    "stage": best_stage
                }

        if best_match and best_score >= fuzzy_threshold:
            # 检查是否歧义（存在其他高相似度候选）
            close_candidates = [c for c in all_candidates
                                if c["target_index"] != best_match["target_index"]
                                and c["score"] >= best_score * 0.85]

            if close_candidates:
                best_match["ambiguous"] = True
                best_match["close_candidates"] = close_candidates
                ambiguous.append(best_match)
            else:
                best_match["ambiguous"] = False
                matches.append(best_match)

            matched_target_indices.add(best_match["target_index"])
        else:
            # 未匹配 — 记录最佳候选（如果有的话）
            entry = {
                "source_name": src_name,
                "source_type": src_type,
                "best_candidate": all_candidates[0] if all_candidates else None,
                "top_candidates": sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:5]
            }
            ambiguous.append(entry)

    # 统计
    unmatched_source = [s for i, s in enumerate(source_entities)
                        if not any(m.get("source_name") == s.get("name") for m in matches)]

    stats = {
        "source_total": len(source_entities),
        "target_total": len(target_entities),
        "matched": len(matches),
        "ambiguous": len(ambiguous),
        "unmatched_source": len(unmatched_source),
        "match_rate": round(len(matches) / max(len(source_entities), 1), 3)
    }

    return {
        "matches": matches,
        "ambiguous": ambiguous,
        "unmatched_source": unmatched_source,
        "stats": stats
    }


# ============================================================
# API Routes
# ============================================================


@entity_alignment_bp.route('/kg/align', methods=['POST'])
def align_entities_endpoint():
    """
    实体对齐接口。
    请求体:
    {
        "source_entities": [{"name": "栖霞组", "type": "FORMATION"}, ...],
        "target_entities": [{"name": "栖霞组（福建）", "type": "FORMATION"}, ...],
        "kb_id": "default_kb"  // 可选，用于上下文对齐
    }
    """
    data = request.get_json(silent=True) or {}
    source_entities = data.get("source_entities", [])
    target_entities = data.get("target_entities", [])
    kb_id = data.get("kb_id")

    if not source_entities or not target_entities:
        return jsonify({"error": "source_entities 和 target_entities 均不能为空"}), 400

    # 加载图谱用于上下文匹配
    graph = None
    if kb_id:
        try:
            graph = load_graph_for_kb(kb_id)
        except Exception:
            pass

    try:
        result = align_entities(source_entities, target_entities, graph)
        return jsonify({
            "status": "success",
            "alignment": result
        })
    except Exception as e:
        log.exception("Entity alignment failed")
        return jsonify({"error": f"实体对齐失败: {str(e)}"}), 500


@entity_alignment_bp.route('/kg/align/confirm', methods=['POST'])
def confirm_alignment():
    """
    确认或拒绝对齐匹配。
    请求体:
    {
        "confirmed": [
            {"source_name": "栖霞组", "target_name": "栖霞组（福建）", "action": "accept"},
            {"source_name": "茅口组", "target_name": "茅口组", "action": "reject"}
        ],
        "kb_id": "default_kb"
    }
    """
    data = request.get_json(silent=True) or {}
    confirmed = data.get("confirmed", [])
    kb_id = data.get("kb_id", "default_kb")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    accepted = []
    rejected = []

    for item in confirmed:
        action = item.get("action", "accept")
        entry = {
            "source_name": item.get("source_name"),
            "target_name": item.get("target_name"),
            "action": action,
            "timestamp": timestamp
        }
        if action == "accept":
            accepted.append(entry)
        else:
            rejected.append(entry)

    # 记录对齐日志
    log_entry = {
        "kb_id": kb_id,
        "timestamp": timestamp,
        "accepted": accepted,
        "rejected": rejected
    }

    print(f"[{timestamp}] [ALIGNMENT] {kb_id}: accepted={len(accepted)}, rejected={len(rejected)}")

    return jsonify({
        "status": "success",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected)
    })
