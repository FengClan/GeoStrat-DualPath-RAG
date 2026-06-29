# services/eval_service.py — 评估服务API (供前端EvaluationDashboard调用)
import os
import json
import time
import glob

from flask import Blueprint, request, jsonify

eval_bp = Blueprint('eval', __name__)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'results')
TEST_SETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'test_sets')


@eval_bp.route('/eval/test-set', methods=['GET'])
def get_test_set():
    """获取当前测试集"""
    try:
        from evaluation.test_set_builder import TestSetBuilder
        builder = TestSetBuilder(output_dir=TEST_SETS_DIR)
        test_set = builder.build_from_builtin()
        stats = builder.get_statistics(test_set)
        return jsonify({
            "test_set": test_set,
            "statistics": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route('/eval/run', methods=['POST'])
def run_evaluation():
    """触发评估运行"""
    data = request.get_json(silent=True) or {}
    baseline = data.get('baseline', 'dual_path')
    use_llm_judge = data.get('use_llm_judge', False)
    model = data.get('model', 'qwen3.5-plus')

    try:
        from evaluation.test_set_builder import TestSetBuilder
        from evaluation.evaluator import Evaluator
        from evaluation.baselines import BaselineRunner

        builder = TestSetBuilder(output_dir=TEST_SETS_DIR)
        test_set = builder.build_from_builtin()

        runner = BaselineRunner(model=model, results_dir=RESULTS_DIR)
        system_runner = runner.get_runner(baseline)

        evaluator = Evaluator(model=model, results_dir=RESULTS_DIR)
        results = evaluator.run_full_evaluation(
            test_set, system_runner,
            use_llm_judge=use_llm_judge
        )

        # 保存
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(RESULTS_DIR, f"eval_{baseline}_{timestamp}.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return jsonify({
            "status": "completed",
            "results": results,
            "saved_to": result_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route('/eval/baselines', methods=['POST'])
def run_baselines():
    """运行4路基线对比"""
    data = request.get_json(silent=True) or {}
    model = data.get('model', 'qwen3.5-plus')
    use_llm_judge = data.get('use_llm_judge', False)

    try:
        from evaluation.test_set_builder import TestSetBuilder
        from evaluation.baselines import BaselineRunner

        builder = TestSetBuilder(output_dir=TEST_SETS_DIR)
        test_set = builder.build_from_builtin()

        runner = BaselineRunner(model=model, results_dir=RESULTS_DIR)
        results = runner.run_all_baselines(test_set, use_llm_judge=use_llm_judge)

        return jsonify({
            "status": "completed",
            "baselines": {k: {
                "overall_scores": v["overall_scores"],
                "timing": v["timing"]
            } for k, v in results["baselines"].items()},
            "comparison": results["comparison"],
            "winner": results["winner"],
            "winner_name": results["winner_name"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route('/eval/results', methods=['GET'])
def list_results():
    """列出所有历史评估结果"""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")),
                      key=os.path.getmtime, reverse=True)
        results_list = []
        for f in files[:20]:
            stat = os.stat(f)
            results_list.append({
                "filename": os.path.basename(f),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
        return jsonify({"results": results_list, "count": len(results_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route('/eval/results/<filename>', methods=['GET'])
def get_result_detail(filename):
    """获取单个评估结果的详细内容"""
    try:
        filepath = os.path.join(RESULTS_DIR, os.path.basename(filename))
        if not os.path.exists(filepath):
            return jsonify({"error": "Result file not found"}), 404

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route('/eval/stats', methods=['GET'])
def get_eval_stats():
    """获取评估统计摘要 (供Dashboard使用)"""
    try:
        # 尝试加载最新的评估结果
        os.makedirs(RESULTS_DIR, exist_ok=True)
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")),
                      key=os.path.getmtime, reverse=True)

        if not files:
            # 无历史结果，返回占位
            return jsonify({
                "has_results": False,
                "message": "暂无评估结果，请运行评估"
            })

        latest = files[0]
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 判断结果类型
        if "comparison" in data:
            # 基线对比结果
            return jsonify({
                "has_results": True,
                "type": "baseline_comparison",
                "winner": data.get("winner_name", "Unknown"),
                "comparison": data["comparison"],
                "evaluated_at": data.get("evaluated_at", ""),
                "model": data.get("model", "")
            })
        else:
            # 单次评估结果
            return jsonify({
                "has_results": True,
                "type": "single_evaluation",
                "overall_scores": data.get("overall_scores", {}),
                "by_difficulty": data.get("by_difficulty", {}),
                "by_query_type": data.get("by_query_type", {}),
                "timing": data.get("timing", {}),
                "evaluated_at": data.get("evaluated_at", ""),
                "per_question_results": data.get("per_question_results", [])
            })
    except Exception as e:
        return jsonify({"error": str(e), "has_results": False}), 500
