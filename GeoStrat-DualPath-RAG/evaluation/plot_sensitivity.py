#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 run_sensitivity.py 的输出画成敏感性曲线图（R2#4）。

审稿人要求：对 hop=1,2,3 分别运行评估并**生成敏感性曲线图**。

⚠️ 两点必须在论文里写清楚：
1. 本分析跑在 **dev 集** 上（默认 150 题），**不是 test 集** —— 遵守 R2#3 的划分纪律：
   调参/诊断只看 dev，test 集在最终报告前不碰。
2. 曲线只作**分析报告**，**不得据此回头改主实验的 hop 设定** —— 否则就成了"用测试集调参"，
   正好坐实 R2#3 的指控。若 hop=2 不是最优，如实写出，并说明该设定来自领域经验。

用法:
    python evaluation/plot_sensitivity.py                      # 自动取最新的 sensitivity_*.json
    python evaluation/plot_sensitivity.py --input <文件> --out <目录>
"""
import sys, os, json, glob, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ARM_COLOR = {"graph_rag": "#6C5CE7", "dual_path": "#E17055"}
ARM_MARK = {"graph_rag": "s", "dual_path": "o"}

# 中英两套文案：论文中英双版本要保持一致，图也要各出一份
TXT = {
    "zh": {
        "arm": {"graph_rag": "图谱 RAG", "dual_path": "双路融合（提出方法）"},
        "xlabel": "图谱检索跳数 max_hops",
        "ylabel_a": "综合得分 overall_score",
        "ylabel_b": "平均延迟（秒）",
        "title_a": "(a) 综合得分随跳数的变化",
        "title_b": "(b) 平均延迟随跳数的变化",
        "suptitle": "图谱检索跳数的敏感性分析（开发集 {n} 题，未参与调参）",
        "stem": "fig_hop_sensitivity",
        "font": ["SimHei", "Microsoft YaHei", "STXihei"],
    },
    "en": {
        "arm": {"graph_rag": "Graph RAG", "dual_path": "Dual-path (proposed)"},
        "xlabel": "Graph hop count (max_hops)",
        "ylabel_a": "Composite score",
        "ylabel_b": "Average latency (s)",
        "title_a": "(a) Composite score vs. hop count",
        "title_b": "(b) Average latency vs. hop count",
        "suptitle": "Sensitivity of graph hop count "
                    "({n}-item development set, not used for tuning)",
        "stem": "fig_hop_sensitivity_en",
        "font": ["DejaVu Sans", "Arial"],
    },
}


def latest_input(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "sensitivity_*.json")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser(description="R2#4 跳数敏感性曲线图")
    ap.add_argument("--input", default=None, help="sensitivity_*.json；缺省取最新的")
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--out", default=None, help="输出目录（缺省为仓库根的 figures/）")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"],
                    help="图内文案语言；中英论文各出一份")
    args = ap.parse_args()
    T = TXT[args.lang]
    ARM_LABEL = T["arm"]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    inp = args.input or latest_input(args.results_dir)
    if not inp or not os.path.exists(inp):
        print(f"找不到 sensitivity_*.json（在 {args.results_dir} 下）。先跑 run_sensitivity.py。")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": T["font"],
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
        "font.size": 10,
    })

    d = json.load(open(inp, encoding="utf-8"))
    res = d["results"]
    hops = sorted({int(h) for per_h in res.values() for h in per_h})
    n_q = sum(d.get("sample_composition", {}).values()) or "?"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#FAFBFC")

    # ---- (a) overall vs hops ----
    ax = axes[0]
    ax.set_facecolor("#FAFBFC")
    for bk, per_h in res.items():
        ys = [per_h[str(h)]["overall_score"] if str(h) in per_h
              else per_h[h]["overall_score"] for h in hops]
        ax.plot(hops, ys, marker=ARM_MARK.get(bk, "o"), markersize=9,
                linewidth=2.4, color=ARM_COLOR.get(bk, "#333"),
                label=ARM_LABEL.get(bk, bk), zorder=3)
        for h, y in zip(hops, ys):
            ax.annotate(f"{y:.4f}", (h, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9,
                        color=ARM_COLOR.get(bk, "#333"), fontweight="bold")
    # 头部留白：数值标签挂在点上方 11pt，不预留会顶到子图标题上
    _lo = min(per_h[str(h)]["overall_score"] if str(h) in per_h else per_h[h]["overall_score"]
              for per_h in res.values() for h in hops)
    _hi = max(per_h[str(h)]["overall_score"] if str(h) in per_h else per_h[h]["overall_score"]
              for per_h in res.values() for h in hops)
    _m = max((_hi - _lo) * 0.25, 0.02)
    ax.set_ylim(_lo - _m, _hi + _m)
    ax.set_xlabel(T["xlabel"], fontsize=11)
    ax.set_ylabel(T["ylabel_a"], fontsize=11)
    ax.set_title(T["title_a"], fontsize=13, fontweight="bold", loc="left")
    ax.set_xticks(hops)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, edgecolor="#E0E4E8", facecolor="white")

    # ---- (b) latency vs hops ----
    ax = axes[1]
    ax.set_facecolor("#FAFBFC")
    for bk, per_h in res.items():
        ys = [(per_h[str(h)] if str(h) in per_h else per_h[h])["avg_latency_ms"] / 1000.0
              for h in hops]
        ax.plot(hops, ys, marker=ARM_MARK.get(bk, "o"), markersize=9,
                linewidth=2.4, color=ARM_COLOR.get(bk, "#333"),
                label=ARM_LABEL.get(bk, bk), zorder=3)
        for h, y in zip(hops, ys):
            ax.annotate(f"{y:.1f}", (h, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9,
                        color=ARM_COLOR.get(bk, "#333"), fontweight="bold")
    ax.set_xlabel(T["xlabel"], fontsize=11)
    ax.set_ylabel(T["ylabel_b"], fontsize=11)
    ax.set_title(T["title_b"], fontsize=13, fontweight="bold", loc="left")
    ax.set_xticks(hops)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, edgecolor="#E0E4E8", facecolor="white")

    fig.suptitle(T["suptitle"].format(n=n_q), fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg"):
        f = os.path.join(out_dir, f"{T['stem']}.{ext}")
        fig.savefig(f, facecolor=fig.get_facecolor())
        print(f"已保存: {f}")

    # ---- 控制台判读 ----
    print(f"\n输入: {inp}")
    print(f"{'臂':<22}" + "".join(f"{'hop=' + str(h):>12}" for h in hops))
    for bk, per_h in res.items():
        row = f"{ARM_LABEL.get(bk, bk):<22}"
        for h in hops:
            g = per_h.get(str(h)) or per_h.get(h)
            row += f"{g['overall_score']:>12.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
