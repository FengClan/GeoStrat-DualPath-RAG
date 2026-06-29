# services/pipeline_orchestrator.py — 全流水线编排器
"""
完整流水线: 文档上传 → KG抽取 → 实体对齐 → 知识融合 → QA → 地层对比 → 评估
每个阶段有状态跟踪: pending → running → completed → failed
"""

import time
import json
import os
import traceback
from datetime import datetime
from flask import Blueprint, request, jsonify

pipeline_bp = Blueprint('pipeline', __name__)

# 流水线阶段定义
PIPELINE_STAGES = [
    {"key": "document_ingestion", "label": "文档入库", "pillar": 1, "icon": "upload"},
    {"key": "kg_extraction",      "label": "知识抽取", "pillar": 1, "icon": "extract"},
    {"key": "entity_alignment",   "label": "实体对齐", "pillar": 1, "icon": "align"},
    {"key": "knowledge_fusion",   "label": "知识融合", "pillar": 1, "icon": "fusion"},
    {"key": "qa_setup",           "label": "问答就绪", "pillar": 2, "icon": "qa"},
    {"key": "contrast_setup",     "label": "对比就绪", "pillar": 3, "icon": "contrast"},
    {"key": "evaluation",         "label": "系统评估", "pillar": 4, "icon": "eval"},
]

# 全局流水线状态 (内存存储，生产环境应使用数据库)
_pipeline_state = {
    "run_id": None,
    "status": "idle",  # idle | running | completed | failed
    "current_stage": None,
    "stages": {},
    "started_at": None,
    "completed_at": None,
    "results_summary": {},
    "errors": []
}


def reset_pipeline():
    """重置流水线状态"""
    global _pipeline_state
    _pipeline_state = {
        "run_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "running",
        "current_stage": None,
        "stages": {s["key"]: {"status": "pending", "started_at": None, "completed_at": None,
                              "duration_ms": 0, "output": None, "error": None}
                    for s in PIPELINE_STAGES},
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "results_summary": {},
        "errors": []
    }


def update_stage(stage_key, status, output=None, error=None):
    """更新流水线阶段状态"""
    now = datetime.now()
    stage = _pipeline_state["stages"].get(stage_key, {})
    stage["status"] = status

    if status == "running":
        stage["started_at"] = now.isoformat()
        _pipeline_state["current_stage"] = stage_key
    elif status in ("completed", "failed"):
        stage["completed_at"] = now.isoformat()
        if stage.get("started_at"):
            started = datetime.fromisoformat(stage["started_at"])
            stage["duration_ms"] = round((now - started).total_seconds() * 1000)

    if output is not None:
        stage["output"] = output
    if error is not None:
        stage["error"] = str(error)
        if status != "completed":
            _pipeline_state["errors"].append({"stage": stage_key, "error": str(error)})


def run_full_pipeline(kb_id="default", model="qwen3.5-plus", documents=None):
    """
    运行完整流水线。
    每个阶段独立执行，失败不阻断后续阶段（最佳努力）。
    """
    reset_pipeline()
    results = {}

    # ===== Stage 1: 文档入库 =====
    update_stage("document_ingestion", "running")
    try:
        doc_count = 0
        if documents:
            for doc in documents:
                # 使用 RAG 服务处理文档
                try:
                    from rag_core import process_and_store_file
                    process_and_store_file(doc.get("content", ""), kb_id,
                                          source_name=doc.get("title", "unknown"))
                    doc_count += 1
                except Exception as e:
                    print(f"[Pipeline] Document ingestion failed for {doc.get('title')}: {e}")
        update_stage("document_ingestion", "completed",
                     output={"documents_processed": doc_count})
        results["documents_ingested"] = doc_count
    except Exception as e:
        update_stage("document_ingestion", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 2: KG抽取 =====
    update_stage("kg_extraction", "running")
    try:
        from services.kg_extraction import build_extraction_prompt, parse_extraction_response
        from services.model_router import get_model_response
        from services.kg_service import save_graph_to_kb_subdir
        import networkx as nx

        all_triples = []
        if documents:
            for doc in documents:
                content = doc.get("content", "")[:4000]
                if not content.strip():
                    continue
                prompt = build_extraction_prompt(content)
                raw = get_model_response(prompt, model)
                triples = parse_extraction_response(raw)
                all_triples.extend(triples)

        if all_triples:
            # 构建抽取图谱
            G = nx.DiGraph()
            for t in all_triples:
                head = t.get("entity", "") or t.get("head", "") or t.get("subject", "")
                rel = t.get("relation", "") or t.get("predicate", "")
                tail = t.get("target", "") or t.get("object", "")
                if head and rel and tail:
                    G.add_edge(head, tail, relation=rel,
                              confidence=t.get("confidence", 0.5),
                              source="pipeline_extraction")
            save_graph_to_kb_subdir(kb_id, "extraction", G)

        update_stage("kg_extraction", "completed",
                     output={"triples_extracted": len(all_triples)})
        results["triples_extracted"] = len(all_triples)
    except Exception as e:
        update_stage("kg_extraction", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 3: 实体对齐 =====
    update_stage("entity_alignment", "running")
    try:
        from services.entity_alignment import align_entities

        alignment_result = align_entities(kb_id, model=model)
        confirmed = alignment_result.get("confirmed", [])
        ambiguous = alignment_result.get("ambiguous", [])

        update_stage("entity_alignment", "completed", output={
            "confirmed_matches": len(confirmed),
            "ambiguous_matches": len(ambiguous),
            "total_entities": alignment_result.get("total_entities", 0)
        })
        results["alignments"] = {
            "confirmed": len(confirmed),
            "ambiguous": len(ambiguous)
        }
    except Exception as e:
        update_stage("entity_alignment", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 4: 知识融合 =====
    update_stage("knowledge_fusion", "running")
    try:
        from services.kg_service import fuse_knowledge_graphs, load_graph_for_kb

        fused = fuse_knowledge_graphs(kb_id, alignment_confirmed=True)
        G = load_graph_for_kb(kb_id)

        update_stage("knowledge_fusion", "completed", output={
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "fused": fused.get("fused", False)
        })
        results["kg_stats"] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        }
    except Exception as e:
        update_stage("knowledge_fusion", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 5: QA就绪检查 =====
    update_stage("qa_setup", "running")
    try:
        from services.query_classifier import classify_by_keywords

        # 测试查询分类
        test_queries = ["南园组的岩性是什么？", "请对比南园组和长林组"]
        qa_ready = True
        qa_details = []
        for q in test_queries:
            qtype = classify_by_keywords(q)
            qa_details.append({"query": q, "classified_as": qtype})

        update_stage("qa_setup", "completed", output={
            "query_classifier_ready": qa_ready,
            "test_classifications": qa_details
        })
        results["qa_ready"] = True
    except Exception as e:
        update_stage("qa_setup", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 6: 对比就绪检查 =====
    update_stage("contrast_setup", "running")
    try:
        from services.kg_service import load_graph_for_kb

        G = load_graph_for_kb(kb_id)
        # 统计可对比的地层实体
        formation_nodes = [str(n) for n in G.nodes()
                          if str(n).endswith("组") or str(n).endswith("群")]

        update_stage("contrast_setup", "completed", output={
            "formations_available": len(formation_nodes),
            "sample_formations": formation_nodes[:5]
        })
        results["contrast_ready"] = True
        results["formations_count"] = len(formation_nodes)
    except Exception as e:
        update_stage("contrast_setup", "failed", error=e)
        traceback.print_exc()

    # ===== Stage 7: 评估 =====
    update_stage("evaluation", "running")
    try:
        from evaluation.test_set_builder import TestSetBuilder
        from evaluation.evaluator import Evaluator

        builder = TestSetBuilder()
        test_set = builder.build_from_builtin()
        evaluator = Evaluator(model=model)

        # 使用简化的快速评估（无LLM裁判以加速）
        def quick_runner(q):
            from services.model_router import get_model_response
            t0 = time.time()
            answer = get_model_response(f"你是地质专家。请回答：{q['question']}", model)
            latency = (time.time() - t0) * 1000
            return {"answer": answer, "retrieved_docs": [], "citations": [],
                    "extracted_triples": [], "contrast_dimensions": [],
                    "query_type": q.get("query_type", "FACTUAL"), "latency_ms": latency}

        eval_results = evaluator.run_full_evaluation(
            test_set, quick_runner, use_llm_judge=False
        )

        update_stage("evaluation", "completed", output={
            "overall_score": eval_results["overall_scores"]["overall_score"],
            "qa_correctness": eval_results["overall_scores"]["qa_correctness"],
            "questions_evaluated": eval_results["timing"]["total_questions"]
        })
        results["evaluation"] = eval_results["overall_scores"]
    except Exception as e:
        update_stage("evaluation", "failed", error=e)
        traceback.print_exc()

    # 流水线完成
    _pipeline_state["status"] = "completed" if not _pipeline_state["errors"] else "completed_with_errors"
    _pipeline_state["completed_at"] = datetime.now().isoformat()
    _pipeline_state["results_summary"] = results

    return _pipeline_state


# ================================================================
# API Routes
# ================================================================


@pipeline_bp.route('/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """获取当前流水线状态"""
    return jsonify(_pipeline_state)


@pipeline_bp.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """启动完整流水线"""
    global _pipeline_state

    if _pipeline_state.get("status") == "running":
        return jsonify({"error": "流水线正在运行中，请等待完成", "status": "running"}), 409

    data = request.get_json(silent=True) or {}
    kb_id = data.get("kb_id", "default")
    model = data.get("model", "QwQ-32B")
    documents = data.get("documents", None)

    try:
        # 同步运行流水线（对于长时间任务应考虑使用后台线程）
        result = run_full_pipeline(kb_id, model, documents)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "failed"}), 500


@pipeline_bp.route('/pipeline/reset', methods=['POST'])
def reset_pipeline_endpoint():
    """重置流水线"""
    reset_pipeline()
    _pipeline_state["status"] = "idle"
    return jsonify({"status": "reset", "pipeline": _pipeline_state})


@pipeline_bp.route('/pipeline/stages', methods=['GET'])
def get_stages():
    """获取阶段定义和当前状态"""
    return jsonify({
        "stages": PIPELINE_STAGES,
        "current_state": {k: {
            "status": v["status"],
            "duration_ms": v.get("duration_ms", 0),
            "output_summary": str(v.get("output", ""))[:200] if v.get("output") else None
        } for k, v in _pipeline_state.get("stages", {}).items()},
        "pipeline_status": _pipeline_state.get("status", "idle")
    })
