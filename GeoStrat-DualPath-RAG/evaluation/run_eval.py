#!/usr/bin/env python3
# evaluation/run_eval.py — 评估框架CLI入口
"""
用法:
  python evaluation/run_eval.py                    # 运行完整评估 (内置测试集)
  python evaluation/run_eval.py --baselines        # 运行4路基线对比
  python evaluation/run_eval.py --output results/  # 指定输出目录
  python evaluation/run_eval.py --no-llm-judge    # 跳过LLM裁判评估
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.test_set_builder import TestSetBuilder
from evaluation.evaluator import Evaluator
from evaluation.baselines import BaselineRunner


def run_single_evaluation(args):
    """运行单一系统评估"""
    test_set = _load_test_set(args)

    stats = builder.get_statistics(test_set)
    print(f"[RUN] Test set: {stats['total_questions']} questions")
    print(f"       By type: {stats['by_type']}")
    print(f"       By difficulty: {stats['by_difficulty']}")

    print(f"\n[RUN] Running evaluation (LLM judge: {not args.no_llm_judge})...")
    evaluator = Evaluator(model=args.model or "qwen3.5-plus",
                          results_dir=args.output or "./evaluation/results")

    # 使用双路自适应方法作为默认系统
    baseline_runner = BaselineRunner(model=args.model or "qwen3.5-plus")
    system_runner = baseline_runner._runner_dual_path

    results = evaluator.run_full_evaluation(
        test_set, system_runner,
        use_llm_judge=not args.no_llm_judge
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in results["overall_scores"].items():
        print(f"  {metric:<30}: {score}")

    print(f"\n  By difficulty:")
    for diff, info in results.get("by_difficulty", {}).items():
        print(f"    {diff}: avg={info['avg_score']:.4f} (n={info['count']})")

    print(f"\n  Timing: {results['timing']['total_seconds']}s total, "
          f"{results['timing']['avg_per_question_ms']}ms per question")

    # 保存结果
    import time
    result_path = os.path.join(
        evaluator.results_dir,
        f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[RUN] Results saved to: {result_path}")

    return results


def _load_test_set(args):
    """根据参数加载测试集"""
    if args.test_set:
        path = args.test_set
        print(f"[RUN] Loading test set from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("[RUN] Building built-in test set...")
        builder = TestSetBuilder(output_dir=args.output or "./evaluation/test_sets")
        return builder.build_from_builtin()

def run_baseline_comparison(args):
    """运行多基线对比评估"""
    test_set = _load_test_set(args)

    runner = BaselineRunner(
        model=args.model or "qwen3.5-plus",
        results_dir=args.output or "./evaluation/results"
    )

    results = runner.run_all_baselines(test_set, use_llm_judge=not args.no_llm_judge)
    runner.print_comparison_summary(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="地质文本空间化平台 - 评估框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluation/run_eval.py                     # 默认：内置测试集 + 双路融合方法
  python evaluation/run_eval.py --baselines         # 4路基线对比
  python evaluation/run_eval.py --model deepseek-R1-32B  # 指定模型
  python evaluation/run_eval.py --no-llm-judge      # 快速评估 (无LLM裁判)
  python evaluation/run_eval.py --questions 5       # 限制问题数量
        """
    )
    parser.add_argument("--baselines", action="store_true",
                        help="运行4路基线对比 (direct/vector/graph/dual)")
    parser.add_argument("--model", type=str, default="qwen3.5-plus",
                        help="LLM模型名称 (默认: qwen3.5-plus)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: ./evaluation/results)")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="跳过LLM裁判评估 (加速但降低准确性)")
    parser.add_argument("--questions", type=int, default=0,
                        help="限制评估问题数量 (0=全部)")
    parser.add_argument("--test-set", type=str, default=None,
                        help="测试集JSON文件路径 (默认: 内置12题)")

    args = parser.parse_args()

    print(f"[RUN] Model: {args.model}")
    print(f"[RUN] LLM Judge: {'disabled' if args.no_llm_judge else 'enabled'}")

    if args.baselines:
        run_baseline_comparison(args)
    else:
        run_single_evaluation(args)


if __name__ == '__main__':
    main()
