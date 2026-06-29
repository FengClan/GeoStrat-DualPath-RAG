# services/kg_service.py — 知识图谱: 加载/保存/上传图谱 + 知识库文件上传 + 知识融合
import os, time, json, uuid
import logging

import networkx as nx
import pandas as pd
from flask import Blueprint, request, jsonify

from .config import UPLOAD_FOLDER, GRAPH_STORAGE_DIR
from .logger import sanitize
from rag_core import process_and_store_file

logger = logging.getLogger(__name__)

kg_bp = Blueprint('kg', __name__)

# ============================================================
# 知识融合 (Knowledge Fusion)
# ============================================================


def fuse_knowledge_graphs(kb_id, alignment_confirmed=None):
    """
    融合知识库下所有源图谱。
    1. 加载所有 .graphml 子图
    2. 执行实体对齐
    3. 合并节点和边
    4. 冲突检测与消解
    :param kb_id: 知识库ID
    :param alignment_confirmed: 用户确认的对齐映射 {src_name: target_name}
    :return: (fused_graph, fusion_stats)
    """
    if alignment_confirmed is None:
        alignment_confirmed = {}

    graph_dir = os.path.join(GRAPH_STORAGE_DIR, kb_id)
    if not os.path.exists(graph_dir):
        return nx.DiGraph(), {"error": f"知识库 {kb_id} 没有源图谱目录"}

    sub_graphs = []
    for filename in os.listdir(graph_dir):
        if filename.endswith(".graphml"):
            path = os.path.join(graph_dir, filename)
            try:
                g = nx.read_graphml(path)
                source_name = filename.replace(".graphml", "")
                sub_graphs.append((source_name, g))
            except Exception as e:
                print(f"[WARN] 读取子图 {filename} 失败: {e}")

    if not sub_graphs:
        return nx.DiGraph(), {"error": f"知识库 {kb_id} 没有有效的图谱文件"}

    # 合并所有图
    fused = nx.DiGraph()
    edge_count = 0
    node_sources = {}   # node -> [source_graph_names]
    edge_sources = {}   # (u, v) -> [source_graph_names]

    for src_name, g in sub_graphs:
        for node in g.nodes():
            node_str = str(node)
            if node_str not in fused:
                fused.add_node(node_str)
                node_sources[node_str] = [src_name]
            else:
                node_sources[node_str].append(src_name)

            # 复制节点属性
            for attr_key, attr_val in g.nodes[node].items():
                fused.nodes[node_str][attr_key] = attr_val
                fused.nodes[node_str][f"source_{src_name}"] = True

        for u, v, data in g.edges(data=True):
            u_str, v_str = str(u), str(v)

            # 应用用户确认的对齐映射
            if u_str in alignment_confirmed:
                u_str = alignment_confirmed[u_str]
            if v_str in alignment_confirmed:
                v_str = alignment_confirmed[v_str]

            if not fused.has_node(u_str):
                fused.add_node(u_str)
                node_sources[u_str] = [src_name]
            if not fused.has_node(v_str):
                fused.add_node(v_str)
                node_sources[v_str] = [src_name]

            edge_key = (u_str, v_str)
            if fused.has_edge(u_str, v_str):
                # 冲突：同一边已存在 → 保留并记录多源
                existing_relation = fused.edges[u_str, v_str].get('relation', '')
                new_relation = data.get('relation', '')
                if new_relation and new_relation != existing_relation:
                    fused.edges[u_str, v_str]['alt_relation'] = existing_relation
                    fused.edges[u_str, v_str]['relation_conflict'] = True
                fused.edges[u_str, v_str]['source_count'] = fused.edges[u_str, v_str].get('source_count', 1) + 1
                edge_sources[edge_key].append(src_name)
            else:
                fused.add_edge(u_str, v_str, **dict(data))
                fused.edges[u_str, v_str]['source_count'] = 1
                fused.edges[u_str, v_str]['source_graphs'] = [src_name]
                edge_sources[edge_key] = [src_name]
                edge_count += 1

            # 记录证据来源
            fused.edges[u_str, v_str][f"from_{src_name}"] = True
            if 'source_graphs' not in fused.edges[u_str, v_str]:
                fused.edges[u_str, v_str]['source_graphs'] = []
            if src_name not in fused.edges[u_str, v_str]['source_graphs']:
                fused.edges[u_str, v_str]['source_graphs'].append(src_name)

    # 保存融合图
    fused_path = os.path.join(GRAPH_STORAGE_DIR, f"{kb_id}_fused.graphml")
    nx.write_graphml(fused, fused_path)

    # 统计
    rel_conflicts = sum(1 for _, _, d in fused.edges(data=True) if d.get('relation_conflict'))

    stats = {
        "total_nodes": fused.number_of_nodes(),
        "total_edges": fused.number_of_edges(),
        "source_graphs": len(sub_graphs),
        "source_names": [s for s, _ in sub_graphs],
        "relation_conflicts": rel_conflicts,
        "multi_source_edges": sum(1 for _, _, d in fused.edges(data=True) if d.get('source_count', 1) > 1),
        "single_source_nodes": sum(1 for n, sources in node_sources.items() if len(sources) == 1),
        "multi_source_nodes": sum(1 for n, sources in node_sources.items() if len(sources) > 1),
        "fused_graph_path": fused_path
    }

    return fused, stats


# ============================================================
# Graph Storage Helpers
# ============================================================


def load_graph_for_kb(kb_id):
    """从本地读取当前知识库的图谱"""
    if kb_id == 'all':
        merged_G = nx.DiGraph()
        if not os.path.exists(GRAPH_STORAGE_DIR):
            return merged_G
        for filename in os.listdir(GRAPH_STORAGE_DIR):
            if filename.endswith(".graphml"):
                path = os.path.join(GRAPH_STORAGE_DIR, filename)
                try:
                    sub_g = nx.read_graphml(path)
                    merged_G = nx.compose(merged_G, sub_g)
                except Exception as e:
                    logger.warning("读取子图 %s 失败: %s", sanitize(str(filename)), sanitize(str(e)))
        return merged_G
    else:
        graph_path = os.path.join(GRAPH_STORAGE_DIR, f"{kb_id}.graphml")
        if os.path.exists(graph_path):
            return nx.read_graphml(graph_path)
        return nx.DiGraph()


def save_graph_for_kb(kb_id, graph):
    """将图谱保存到本地硬盘"""
    graph_path = os.path.join(GRAPH_STORAGE_DIR, f"{kb_id}.graphml")
    nx.write_graphml(graph, graph_path)


def save_graph_to_kb_subdir(kb_id, source_name, graph):
    """将源图谱保存到知识库子目录"""
    subdir = os.path.join(GRAPH_STORAGE_DIR, kb_id)
    os.makedirs(subdir, exist_ok=True)
    graph_path = os.path.join(subdir, f"{source_name}.graphml")
    nx.write_graphml(graph, graph_path)
    return graph_path


def load_all_sub_graphs(kb_id):
    """加载知识库下所有源图谱"""
    subdir = os.path.join(GRAPH_STORAGE_DIR, kb_id)
    graphs = []
    if os.path.exists(subdir):
        for filename in os.listdir(subdir):
            if filename.endswith(".graphml"):
                path = os.path.join(subdir, filename)
                try:
                    g = nx.read_graphml(path)
                    source_name = filename.replace(".graphml", "")
                    graphs.append({"source_name": source_name, "graph": g, "path": path})
                except Exception as e:
                    logger.warning("读取子图 %s 失败: %s", sanitize(str(filename)), sanitize(str(e)))
    return graphs


def retrieve_graph_context(question, kb_id, max_hops=2, max_edges=50):
    """GraphRAG 核心算法：实体链接与子图多跳检索"""
    G = load_graph_for_kb(kb_id)
    if G.number_of_nodes() == 0:
        return [], []

    matched_nodes = set()
    for node in G.nodes():
        if str(node) in question:
            matched_nodes.add(node)

    if not matched_nodes:
        return [], []

    subgraph_nodes = set(matched_nodes)
    current_layer = set(matched_nodes)

    for _ in range(max_hops):
        next_layer = set()
        for node in current_layer:
            if G.is_directed():
                neighbors = set(G.successors(node)) | set(G.predecessors(node))
            else:
                neighbors = set(G.neighbors(node))
            next_layer.update(neighbors)
        subgraph_nodes.update(next_layer)
        current_layer = next_layer

    subgraph = G.subgraph(subgraph_nodes)
    context_texts = []
    citations = []
    edge_count = 0

    for u, v, data in subgraph.edges(data=True):
        if edge_count >= max_edges:
            break
        rel = data.get('relation', '关联')
        text_desc = f"{u} --[{rel}]--> {v}"
        context_texts.append(text_desc)
        if edge_count < 10:
            citations.append({
                "docName": "知识图谱路径",
                "excerpt": text_desc
            })
        edge_count += 1

    return context_texts, citations


@kg_bp.route('/kb/upload', methods=['POST'])
def upload_knowledge_file():
    logger.info("--- New KB Upload Request ---")
    if 'file' not in request.files:
        return jsonify({"error": "没有找到文件"}), 400

    file = request.files['file']
    kb_id = request.form.get('kb_id', 'default_kb')

    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    try:
        original_filename = file.filename
        ext = os.path.splitext(original_filename)[1]
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        file.save(file_path)
        logger.info("正在将文件 %s 存入知识库 [%s]", sanitize(str(original_filename)), sanitize(str(kb_id)))

        chunks_count = process_and_store_file(file_path, original_filename, kb_id)

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "message": f"文件 {original_filename} 处理成功！",
            "chunks_processed": chunks_count,
            "status": "success"
        })
    except Exception as e:
        logger.error("知识库处理失败: %s", sanitize(str(e)))
        return jsonify({"error": f"知识库处理失败: {str(e)}"}), 500


@kg_bp.route('/kb/upload_graph', methods=['POST'])
def upload_graph():
    """处理前端传来的三元组文件 (CSV, Excel, JSON)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    kb_id = request.form.get('kb_id')

    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename.lower()

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif filename.endswith('.json'):
            data = json.load(file)
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "不支持的文件格式"}), 400

        col_mapping = {
            '实体1': 'head', '主体': 'head', 'source': 'head',
            '关系': 'relation', 'rel': 'relation', 'edge': 'relation',
            '实体2': 'tail', '客体': 'tail', 'target': 'tail'
        }
        df.rename(columns=col_mapping, inplace=True)

        if not {'head', 'relation', 'tail'}.issubset(df.columns):
            return jsonify({"error": "表格必须包含 'head', 'relation', 'tail' 这三列"}), 400

        G = load_graph_for_kb(kb_id)
        triplets_added = 0

        for index, row in df.iterrows():
            head = str(row['head']).strip()
            rel = str(row['relation']).strip()
            tail = str(row['tail']).strip()
            if head and rel and tail and head != 'nan' and tail != 'nan':
                G.add_edge(head, tail, relation=rel)
                triplets_added += 1

        save_graph_for_kb(kb_id, G)
        logger.info("[GraphRAG] 成功导入图谱，本次新增 %d 个三元组，知识库图谱总节点数: %d", triplets_added, G.number_of_nodes())

        return jsonify({
            "status": "success",
            "triplets_processed": triplets_added
        }), 200

    except Exception as e:
        logger.error("图谱解析失败: %s", sanitize(str(e)))
        return jsonify({"error": str(e)}), 500


@kg_bp.route('/kb/fuse', methods=['POST'])
def fuse_graphs():
    """融合知识库下所有源图谱，执行实体对齐与冲突消解"""
    data = request.get_json(silent=True) or {}
    kb_id = data.get("kb_id", "default_kb")
    confirmed_mappings = data.get("confirmed_mappings", {})

    logger.info("[Fusion] 开始融合知识库 [%s]", sanitize(str(kb_id)))

    try:
        fused_graph, stats = fuse_knowledge_graphs(kb_id, confirmed_mappings)

        if "error" in stats:
            return jsonify({
                "status": "error",
                "message": stats["error"]
            }), 404

        # 统计节点类型分布
        type_dist = {}
        for node in fused_graph.nodes():
            ntype = fused_graph.nodes[node].get('type', 'UNKNOWN')
            type_dist[ntype] = type_dist.get(ntype, 0) + 1

        # 统计关系类型分布
        rel_dist = {}
        for _, _, data in fused_graph.edges(data=True):
            rel = data.get('relation', 'UNKNOWN')
            rel_dist[rel] = rel_dist.get(rel, 0) + 1

        stats["node_type_distribution"] = type_dist
        stats["relation_type_distribution"] = rel_dist

        logger.info("[Fusion] 完成: %s 节点, %s 边", stats['total_nodes'], stats['total_edges'])

        return jsonify({
            "status": "success",
            "kb_id": kb_id,
            "stats": stats
        })

    except Exception as e:
        logger.error("知识融合失败: %s", sanitize(str(e)))
        return jsonify({"error": f"知识融合失败: {str(e)}"}), 500


@kg_bp.route('/kb/stats', methods=['GET'])
def get_kg_stats():
    """获取知识库的知识图谱统计信息"""
    kb_id = request.args.get("kb_id", "default_kb")
    G = load_graph_for_kb(kb_id)

    if G.number_of_nodes() == 0:
        return jsonify({
            "status": "success",
            "kb_id": kb_id,
            "total_nodes": 0,
            "total_edges": 0,
            "message": "图谱为空"
        })

    type_dist = {}
    for node in G.nodes():
        ntype = G.nodes[node].get('type', 'UNKNOWN')
        type_dist[ntype] = type_dist.get(ntype, 0) + 1

    rel_dist = {}
    for _, _, data in G.edges(data=True):
        rel = data.get('relation', 'UNKNOWN')
        rel_dist[rel] = rel_dist.get(rel, 0) + 1

    # 度中心性 Top 10
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    return jsonify({
        "status": "success",
        "kb_id": kb_id,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_type_distribution": type_dist,
        "relation_type_distribution": rel_dist,
        "top_nodes_by_degree": [{"name": n, "degree": d} for n, d in top_nodes]
    })
