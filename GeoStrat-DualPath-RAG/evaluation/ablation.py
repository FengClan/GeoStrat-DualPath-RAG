# evaluation/ablation.py — 消融实验
"""
消融变体与完整方法 (Dual-path) 的对比:

  D (完整方法):          自适应分类 + 细粒度引用
  D − 自适应分类:        跳过分类器，全部问题按 FACTUAL 模板 + max_hops=2
  D − 细粒度引用:        删掉要求标注 [G]/[D] 的指令

⚠️ 2026-09-19 重构 —— 本文件此前的问题：
  它自带一套 _runner_full / _runner_no_* / _build_prompt，是从 baselines.py
  **复制出来后没有跟着同步**的旧实现：旧 system 指令（"图谱为权威数据，优先
  采信，冲突以图谱为准"）、没有 _graph_is_blind 门控、没有图谱盲区 prompt 分支。
  后果是消融表里的 "D. 完整方法" 与主表里的 "双路" 根本不是同一个系统，两者
  之差不可解释，也无法与主表并排报告。

  现在所有臂一律委托 BaselineRunner._dual_path_impl，**只差一个开关**：
      dual_path             → _dual_path_impl()
      dual_path_no_classify → _dual_path_impl(no_classify=True)
      dual_path_no_citation → _dual_path_impl(no_citation=True)
  因此 "D. 完整方法" 与主表 "双路" 逐条 prompt 完全一致，结果可直接复用主表
  那一份（run_multirun.py --reuse-r1），不必重跑。

  原先还有一个 abl_no_filter 臂，已删除：filter_noise 参数全代码库从未传过
  True，该臂与完整方法实际完全相同（空跑）；且论文表 4 中本来就没有这一项。

用法（多轮，正式报告用这个）:
  python evaluation/run_multirun.py --mode ablation --rounds 3 --total 300 \\
      --test-set test_sets/test_set_honest_test.json \\
      --reuse-r1 evaluation/results_full_20260914
用法（单轮小集，冒烟用）:
  python evaluation/ablation.py --model qwen3.5-plus --test-set test_sets/test_10q.json
"""

import time, json, os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from evaluation.baselines import BaselineRunner
from evaluation.evaluator import Evaluator


class AblationRunner(BaselineRunner):
    """消融实验运行器。

    不自带任何 runner 实现 —— 消融臂就注册在 BaselineRunner.get_runner 里，
    由它统一派发到 _dual_path_impl。这样在结构上就不存在"两份实现跑偏"的可能。
    """

    ABLATION_NAMES = {
        "dual_path":             "D. 完整方法 (Proposed)",
        "dual_path_no_classify": "D − 自适应分类",
        "dual_path_no_citation": "D − 细粒度引用",
    }

    ARMS = list(ABLATION_NAMES)

    # ================================================================
    # 运行全部消融实验（单轮；多轮请用 run_multirun.py --mode ablation）
    # ================================================================

    def run_all_ablations(self, test_set, use_llm_judge=True):
        """运行完整方法 + 各消融变体"""
        evaluator = Evaluator(model=self.model)

        all_results = {}
        comparison_table = {}

        for var_key in self.ARMS:
            name = self.ABLATION_NAMES[var_key]
            print(f"\n{'='*60}")
            print(f"[Ablation] Running: {name}")
            print(f"{'='*60}")

            results = evaluator.run_full_evaluation(
                test_set, self.get_runner(var_key), use_llm_judge=use_llm_judge
            )
            all_results[var_key] = results

            for metric, value in results["overall_scores"].items():
                comparison_table.setdefault(metric, {})[var_key] = value

        # 确定最优
        overall_scores = {k: v["overall_scores"]["overall_score"]
                          for k, v in all_results.items()}
        winner = max(overall_scores, key=overall_scores.get)

        final_results = {
            "ablations": all_results,
            "comparison": comparison_table,
            "winner": winner,
            "winner_name": self.ABLATION_NAMES[winner],
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model": self.model,
        }

        os.makedirs("./evaluation/results", exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = f"./evaluation/results/ablation_{timestamp}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n[Ablation] Results saved to: {result_path}")
        return final_results, result_path

    def print_summary(self, results):
        """打印消融对比摘要"""
        comp = results.get("comparison", {})

        key_metrics = [
            "overall_score", "qa_correctness", "qa_faithfulness",
            "qa_completeness", "citation_precision",
        ]
        metric_labels = {
            "overall_score": "综合得分", "qa_correctness": "正确性",
            "qa_faithfulness": "忠实度", "qa_completeness": "完整性",
            "citation_precision": "引用精确率",
        }

        print(f"\n{'='*90}")
        print(f"{'指标':<14}", end="")
        for v in self.ARMS:
            print(f"{self.ABLATION_NAMES[v]:>22}", end="")
        print()
        print("-" * 102)

        for metric in key_metrics:
            if metric not in comp:
                continue
            print(f"{metric_labels.get(metric, metric):<14}", end="")
            for v in self.ARMS:
                print(f"{comp[metric].get(v, 0):>22.4f}", end="")
            print()

        # 最直白的读数：每个变体相对完整方法掉了多少
        full = comp.get("overall_score", {}).get("dual_path")
        if full:
            print(f"\n相对 D. 完整方法 (overall_score={full:.4f})：")
            for v in self.ARMS[1:]:
                val = comp.get("overall_score", {}).get(v)
                if val is not None:
                    print(f"  {self.ABLATION_NAMES[v]:<24}{val:.4f}  "
                          f"Δ = {val - full:+.4f} ({(val - full) / full * 100:+.1f}%)")

        print(f"\n最优方法: {results.get('winner_name', '?')}")


def main():
    parser = argparse.ArgumentParser(description="消融实验（单轮）")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--test-set", default="test_sets/test_10q.json")
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    # Windows 控制台默认 GBK，臂名里的 "−"(U+2212) 会直接抛 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"[Ablation] Model: {args.model}")
    print(f"[Ablation] Test set: {args.test_set}")

    with open(args.test_set, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"[Ablation] Questions: {len(test_set['questions'])}")

    runner = AblationRunner(model=args.model)
    results, _ = runner.run_all_ablations(
        test_set, use_llm_judge=not args.no_llm_judge
    )
    runner.print_summary(results)


if __name__ == '__main__':
    main()
