#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文实验图生成：fig3 主结果 / fig4 分题型热力图 / fig5 消融（可选出 fig2 示意图）。

用法:
    python evaluation/plot_paper_figures.py                # 出 fig3/4/5
    python evaluation/plot_paper_figures.py --with-fig2    # 连示意图一起出
    python evaluation/plot_paper_figures.py --out <目录>

数据来源（都在本目录 results/ 下，自动取最新的一份，不写死时间戳）:
    multirun_baseline_*.json   主实验 4 臂 × 3 轮
    multirun_ablation_*.json   消融 3 臂 × 3 轮
    以及与它们同名的子目录里的逐题明细 *_r1/r2/r3.json（分题型统计要用）

输出: 默认 ../../../figures/（与 kg、跳数敏感性等论文图放一起），png + pdf + svg。

── 与仓库根目录旧脚本 generate_figures.py 的差别（旧脚本保留作对照，勿删）──
  1. 旧脚本把数据源写死成 6 月的 baseline_comparison_20260604_021540.json 与
     ablation_20260603_180747.json（`python服务/evaluation/results/` 下，注意那是**另一个**
     results 目录，不是本目录）；本脚本按前缀取最新，且只用本目录 results/。
  2. 旧数据是单次运行，本数据是 3 轮。故柱状图加 run-to-run 误差棒（±1 SD），
     直接回应 R2#8/R2#9 要求的 mean±SD。
  3. 题型样本数 n 从数据算出，不再硬编码（旧图写的是 145/62/32/61，现为 119/101/46/34）。
  4. 消融臂名对齐当前实现：dual_path / _no_classify / _no_citation。
     （旧脚本把 abl_no_filter 当作满分的 "Full Method"，那套命名已不存在。）
  5. 雷达图给每条臂独立的 marker。旧图四条臂都用 'o-'，只靠颜色区分；
     而 Dual-path(#E7298A) 与 Graph RAG(#7570B3) 在红绿色盲模拟下 ΔE 仅 7.6，
     仅凭颜色不够分（配色本身不动，以保持与已刊出图一致）。

── 口径说明（有意为之，勿当 bug 改）──
  逐题明细里 judge_failed 的题 `correctness` 记的是 0.5（评估器的中性兜底值）。
  本脚本**保留**这些 0.5，因为 summary 里的均值和主表就是这么算的——
  排除它们会让 fig4 与 fig3b 对不上。全量 58 题次属此情况。
"""
import sys, os, json, glob, argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "figures"))

# ── 四臂（顺序即图例顺序，固定不变）─────────────────────────────
BASELINE_ORDER = ["direct_llm", "vector_rag", "graph_rag", "dual_path"]
BASELINE_LABEL = {
    "direct_llm": "A. Pure LLM",
    "vector_rag": "B. Vector RAG",
    "graph_rag":  "C. Graph RAG",
    "dual_path":  "D. Dual-path (Ours)",
}
C_BASELINE = {
    "direct_llm": "#D95F02",   # 橙
    "vector_rag": "#1B9E77",   # 青
    "graph_rag":  "#7570B3",   # 紫
    "dual_path":  "#E7298A",   # 品红（提出方法）
}
# 次要编码：旧图四条臂同为 'o-'，红绿色盲下 D 与 C 难分，故各给一个 marker
BASELINE_MARKER = {
    "direct_llm": "o", "vector_rag": "s", "graph_rag": "^", "dual_path": "D",
}

ABLATION_ORDER = ["dual_path", "dual_path_no_classify", "dual_path_no_citation"]
ABLATION_LABEL = {
    "dual_path":             "Full\nMethod",
    "dual_path_no_classify": "w/o Adaptive\nClassification",
    "dual_path_no_citation": "w/o Fine-grained\nCitation",
}
C_ABLATION = ["#2878B5", "#E57373", "#67A9CF"]

QTYPES = ["FACTUAL", "COMPARATIVE", "REASONING", "SPATIAL"]
# ⚠️ 这里是**逐题明细** qa_metrics 里的键名，没有 qa_ 前缀。
# 汇总 summary 里才叫 qa_correctness/qá_faithfulness…… 两者不可混用：
# 写成 "qa_correctness" 去取逐题的 qa_metrics 会全部拿到默认值 0，
# 雷达图会缩成原点（本轮已踩过）。
RADAR_DIMS = ["correctness", "faithfulness", "completeness",
              "coherence", "domain_accuracy"]
RADAR_LABELS = ["Correctness", "Faithfulness", "Completeness",
                "Coherence", "Domain Accuracy"]


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════
def latest(prefix):
    """取 results/ 下最新的 <prefix>*.json（排除 .log）。"""
    hits = sorted(glob.glob(os.path.join(RESULTS_DIR, prefix + "*.json")))
    hits = [h for h in hits if os.path.isfile(h) and not h.endswith(".log")]
    if not hits:
        raise FileNotFoundError(f"{RESULTS_DIR} 下找不到 {prefix}*.json")
    return hits[-1]


def _round_files(json_path, arm_key):
    """与汇总 json 同名的子目录里的逐题明细 <arm_key>_r*.json。"""
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return sorted(glob.glob(os.path.join(RESULTS_DIR, stem, arm_key + "_r*.json")))


def load_baseline(path):
    """→ (overall, per_type, radar)

    overall = {arm: {metric: (mean, sd)}}
    per_type = {qt: {arm: {'correctness':m, 'faithfulness':m, 'count':n}}}
    radar    = {arm: ([5 个维度均值], [5 个维度轮间 SD])}
    """
    d = json.load(open(path, encoding="utf-8"))
    prefix = os.path.basename(path).split("_")[0] + "__"   # 'multirun' -> 用不到
    # 汇总 json 里的键形如 baseline__direct_llm；文件名前缀不固定，按键名找
    def key(arm):
        for k in d["summary"]:
            if k.endswith("__" + arm):
                return k
        raise KeyError(f"{path} 里没有臂 {arm}")

    overall = {}
    for arm in BASELINE_ORDER:
        overall[arm] = {m: (v["mean"], v["sd"]) for m, v in d["summary"][key(arm)].items()}

    # ── 分题型：从逐题明细按轮聚合（by_query_type 只存 overall_score，没有分维度的）──
    acc = defaultdict(lambda: defaultdict(lambda: {"correctness": [], "faithfulness": []}))
    radar_rounds = defaultdict(lambda: defaultdict(list))   # arm -> dim -> [每轮均值]
    n_rounds = 0
    for arm in BASELINE_ORDER:
        rfs = _round_files(path, "baseline__" + arm)
        n_rounds = max(n_rounds, len(rfs))
        for rf in rfs:
            recs = json.load(open(rf, encoding="utf-8"))["per_question_results"]
            for r in recs:
                qa = r["qa_metrics"]
                acc[r["query_type"]][arm]["correctness"].append(qa.get("correctness", 0))
                acc[r["query_type"]][arm]["faithfulness"].append(qa.get("faithfulness", 0))
            for dim in RADAR_DIMS:
                radar_rounds[arm][dim].append(
                    float(np.mean([r["qa_metrics"].get(dim, 0) for r in recs])))

    per_type = {}
    for qt in QTYPES:
        per_type[qt] = {}
        for arm in BASELINE_ORDER:
            c = acc[qt][arm]["correctness"]
            f = acc[qt][arm]["faithfulness"]
            per_type[qt][arm] = {
                "correctness": float(np.mean(c)) if c else 0.0,
                "faithfulness": float(np.mean(f)) if f else 0.0,
                # 报**每轮**的题数（题集轮间固定）。这里池化了 3 轮，
                # 直接报 len(c)=357 会被读成"该题型有 357 道题"，是错的。
                "count": len(c) // n_rounds if n_rounds else len(c),
            }

    radar = {}
    for arm in BASELINE_ORDER:
        means = [float(np.mean(radar_rounds[arm][dim])) for dim in RADAR_DIMS]
        sds = [float(np.std(radar_rounds[arm][dim], ddof=1)) for dim in RADAR_DIMS]
        radar[arm] = (means, sds)

    return overall, per_type, radar


def load_ablation(path):
    """→ {arm: {metric: (mean, sd)}}"""
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for arm in ABLATION_ORDER:
        k = next((x for x in d["summary"] if x.endswith("__" + arm)), None)
        if k is None:
            raise KeyError(f"{path} 里没有消融臂 {arm}")
        out[arm] = {m: (v["mean"], v["sd"]) for m, v in d["summary"][k].items()}
    return out


# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": 8, "legend.frameon": True,
        "legend.edgecolor": "#cccccc", "legend.framealpha": 0.9,
        "legend.borderpad": 0.4,
    })
    return plt


def make_savefig(plt, outdir):
    os.makedirs(outdir, exist_ok=True)

    def savefig(fig, name):
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(os.path.join(outdir, name + ext), facecolor="white")
        plt.close(fig)
        print(f"  已保存: {os.path.join(outdir, name)}.png/.pdf/.svg")

    return savefig


def spine_off(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


# ── FIG 3：五维雷达 + 综合得分柱状（带误差棒）──────────────────
def plot_fig3(plt, savefig, overall, radar):
    angles = np.linspace(0, 2 * np.pi, len(RADAR_DIMS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(13, 5.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.2], wspace=0.35,
                          left=0.05, right=0.95)

    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    ax_r.set_facecolor("white")
    for arm in BASELINE_ORDER:
        vals, _ = radar[arm]
        v = vals + vals[:1]
        ax_r.fill(angles, v, alpha=0.06, color=C_BASELINE[arm])
        ax_r.plot(angles, v, marker=BASELINE_MARKER[arm], linewidth=1.8,
                  markersize=5.0, color=C_BASELINE[arm], label=BASELINE_LABEL[arm])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(RADAR_LABELS, fontsize=9)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_r.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
    ax_r.set_rlabel_position(30)
    ax_r.grid(True, alpha=0.3, linewidth=0.4)
    ax_r.set_title("(a) Five-Dimensional Quality Profile",
                   fontsize=10, fontweight="bold", pad=18)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.40, 1.08),
                fontsize=7.5, framealpha=0.85)

    # (b) 综合得分：mean ± SD（3 轮）
    ax_b = fig.add_subplot(gs[0, 1])
    means = [overall[a]["overall_score"][0] for a in BASELINE_ORDER]
    sds = [overall[a]["overall_score"][1] for a in BASELINE_ORDER]
    x = np.arange(len(BASELINE_ORDER))
    ax_b.bar(x, means, yerr=sds, capsize=4,
             color=[C_BASELINE[a] for a in BASELINE_ORDER],
             edgecolor="white", linewidth=0.5, width=0.55,
             error_kw=dict(elinewidth=1.0, ecolor="#37474F", capthick=1.0), zorder=3)

    hi = max(m + s for m, s in zip(means, sds))
    ax_b.set_ylim(0, min(1.0, hi * 1.42))
    for i, (m, s) in enumerate(zip(means, sds)):
        ax_b.text(i, m + s + hi * 0.035, f"{m:.4f}", ha="center",
                  fontsize=9, fontweight="bold", color="#1A237E")
    gain = (means[3] - means[1]) / means[1] * 100
    ax_b.annotate(f"+{gain:.1f}%\nvs Vector RAG",
                  xy=(3, means[3] + sds[3]), xytext=(3.45, means[3] + hi * 0.18),
                  fontsize=7.5, color="#1B5E20", fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Pure\nLLM", "Vector\nRAG", "Graph\nRAG", "Dual-path\n(Ours)"],
                         fontsize=8.5)
    ax_b.set_ylabel("Composite Score", fontsize=10)
    ax_b.set_yticks(np.arange(0, ax_b.get_ylim()[1], 0.2))
    spine_off(ax_b)
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)
    ax_b.set_title("(b) Composite Score (mean ± SD, 3 runs)",
                   fontsize=10, fontweight="bold")

    savefig(fig, "fig3_main_results")


# ── FIG 4：分题型 Correctness / Faithfulness 双热力图 ───────────
def plot_fig4(plt, savefig, per_type):
    qlabels = [f"{qt}\n(n={per_type[qt][BASELINE_ORDER[0]]['count']})" for qt in QTYPES]
    blabels = ["Pure LLM", "Vector RAG", "Graph RAG", "Dual-path\n(Ours)"]

    mat_c = np.array([[per_type[qt][a]["correctness"] for qt in QTYPES]
                      for a in BASELINE_ORDER])
    mat_f = np.array([[per_type[qt][a]["faithfulness"] for qt in QTYPES]
                      for a in BASELINE_ORDER])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    vmin = min(mat_c.min(), mat_f.min()) - 0.05
    vmax = max(mat_c.max(), mat_f.max()) + 0.02

    def draw(ax, mat, title):
        im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels(qlabels, fontsize=9)
        ax.set_yticks(range(4)); ax.set_yticklabels(blabels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if v > (vmin + vmax) / 2 else "#333333")
        # 与 Pure LLM 的差（极性：正绿负红）。
        # ⚠️ 报**绝对差**不报百分比：新数据下 Pure LLM 的 correctness 只有 ~0.1，
        # 拿它当分母会算出 "+581%" 这种数字——看着像夸大宣传，审稿人一定会问。
        # 0-1 分制上绝对差本身就有量纲，不需要再转百分比。
        for j in range(4):
            d = mat[3, j] - mat[0, j]
            ax.text(j, 4.05, f"{d:+.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color="#2E7D32" if d > 0 else "#C62828")
        ax.text(-0.78, 4.05, r"$\Delta$ abs:", fontsize=7.5,
                color="#546E7A", va="bottom")
        return im

    draw(ax1, mat_c, "(a) Correctness by Query Type")
    im2 = draw(ax2, mat_f, "(b) Faithfulness by Query Type")

    # 色标用 add_axes 手动摆位，与 tight_layout 不兼容（会告警且可能摆错），
    # 故这里不用 tight_layout，直接给 subplots 留边。
    fig.subplots_adjust(left=0.09, right=0.88, top=0.88, bottom=0.16, wspace=0.32)
    cbar_ax = fig.add_axes([0.91, 0.20, 0.013, 0.60])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Score", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    savefig(fig, "fig4_per_type_heatmap")


# ── FIG 5：消融（点图 + 分指标柱状，均带误差棒）────────────────
def plot_fig5(plt, savefig, ab):
    """⚠️ 这里**不用**原图那种"瀑布链"，原因有二，都是实质性的：

    1. 三个消融臂是**各自独立地从 Full 上摘掉一个组件**，不是逐级累加。
       "w/o Fine-grained Citation" 不是"Full − 分类器 − 引用标记"。
       画成瀑布链（full → no_classify → no_citation）等于宣称了一个并不存在的累加关系，
       而且会让第三根柱子相对**前一根**向上——读者看到上升箭头会读成"去掉引用反而更好"，
       实际它仍低于 Full。R2#19 挑的正是这类误读。
    2. 瀑布链是"柱长"编码，柱长必须从 0 起；三臂差距只有 0.012，
       从 0 起就看不见、从 0.6 起就是截断纵轴夸大差异，两头都不对。

    改成**点图 + 误差棒**：点没有"长度"可言，不受零基线约束，
    且误差棒直接把"差异是否落在轮间噪声内"摆在图上——这正是 R2#8/#9 要的。
    三臂的 Δ 一律相对 Full Method 标注。
    """
    scores = [ab[a]["overall_score"][0] for a in ABLATION_ORDER]
    sds = [ab[a]["overall_score"][1] for a in ABLATION_ORDER]
    full = scores[0]

    fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={"width_ratios": [1.25, 1]},
                                     facecolor="white")

    # ── (a) 点图：每个臂的得分 ± SD，Δ 相对 Full ──
    ys = np.arange(len(ABLATION_ORDER))
    for i, (a, s, sd) in enumerate(zip(ABLATION_ORDER, scores, sds)):
        ax_p.errorbar(s, i, xerr=sd, fmt="o", markersize=10,
                      color=C_ABLATION[i], markeredgecolor="white",
                      markeredgewidth=1.2, capsize=5,
                      elinewidth=1.6, ecolor=C_ABLATION[i], zorder=4+i)
        ax_p.text(s, i + 0.30, f"{s:.4f}", ha="center", fontsize=10,
                  fontweight="bold", color="#1A237E", zorder=6)
        if i > 0:
            d = s - full
            # 落在轮间 SD 之内就如实标明，别让读者以为测出了差异
            within = abs(d) < max(sd, sds[0])
            ax_p.text(min(s, full) - 0.004, i - 0.26,
                      f"Δ {d:+.4f}" + ("  (within noise)" if within else ""),
                      ha="right", fontsize=8.5, fontweight="bold",
                      color="#C62828" if d < 0 else "#1565C0", zorder=6)

    ax_p.axvline(full, color="#546E7A", linestyle="--", linewidth=1.0,
                 alpha=0.7, zorder=1)
    ax_p.text(full, len(ABLATION_ORDER) - 0.42, "  Full Method",
              fontsize=7.5, color="#546E7A", style="italic", va="center")

    ax_p.set_yticks(ys)
    ax_p.set_yticklabels([ABLATION_LABEL[a] for a in ABLATION_ORDER], fontsize=9)
    ax_p.set_ylim(-0.55, len(ABLATION_ORDER) - 0.35)
    lo = min(s - sd for s, sd in zip(scores, sds))
    hi = max(s + sd for s, sd in zip(scores, sds))
    span = max(hi - lo, 0.006)
    ax_p.set_xlim(lo - span * 1.8, hi + span * 0.9)
    ax_p.set_xlabel("Composite Score (mean ± SD, 3 runs)", fontsize=10)
    spine_off(ax_p, spines=("top", "right", "left"))
    ax_p.grid(axis="x", alpha=0.25, linewidth=0.4, zorder=0)
    ax_p.set_title("(a) Ablation vs Full Method", fontsize=10, fontweight="bold")

    # (b) 分指标：Correctness / Faithfulness / Completeness
    mk = ["qa_correctness", "qa_faithfulness", "qa_completeness"]
    mlabel = ["Correctness", "Faithfulness", "Completeness"]
    x = np.arange(3)
    w = 0.24
    for i, arm in enumerate(ABLATION_ORDER):
        vals = [ab[arm][m][0] for m in mk]
        errs = [ab[arm][m][1] for m in mk]
        ax_d.bar(x + i * w, vals, w, yerr=errs, capsize=2.5,
                 label=ABLATION_LABEL[arm].replace("\n", " "),
                 color=C_ABLATION[i], edgecolor="white", linewidth=0.5,
                 error_kw=dict(elinewidth=0.8, ecolor="#546E7A", capthick=0.8),
                 zorder=3)

    ax_d.set_ylim(0, min(1.0, max(ab[a][m][0] + ab[a][m][1]
                                  for a in ABLATION_ORDER for m in mk) * 1.18))
    for j in range(3):
        full = ab["dual_path"][mk[j]][0]
        for i, arm in enumerate(ABLATION_ORDER[1:], start=1):
            v = ab[arm][mk[j]][0]
            ax_d.text(x[j] + i * w, v + ab[arm][mk[j]][1] + 0.012,
                      f"{(v - full) / full * 100:+.1f}%", ha="center",
                      fontsize=7, fontweight="bold",
                      color="#C62828" if v < full else "#1565C0")
    ax_d.set_xticks(x + w)
    ax_d.set_xticklabels(mlabel, fontsize=9)
    ax_d.set_ylabel("Score", fontsize=10)
    spine_off(ax_d)
    ax_d.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax_d.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)
    ax_d.set_title("(b) Per-Metric Breakdown", fontsize=10, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig5_ablation")


# ── FIG 2：双路检索示意图（无数据依赖；两处数字改为现读，免得再过期）──
def plot_fig2(plt, savefig):
    from matplotlib.patches import FancyBboxPatch

    kg_nodes = kg_edges = None
    try:
        import networkx as nx
        g = nx.read_graphml(os.path.join(HERE, "..", "kb_storage", "graphs",
                                         "experiment_kb.graphml"))
        kg_nodes, kg_edges = g.number_of_nodes(), g.number_of_edges()
    except Exception as e:                                    # noqa: BLE001
        print(f"  [warn] 读不到图谱规模，图 2 用占位符: {e}")

    hops = {}
    try:
        from services.query_classifier import QUERY_TYPES
        hops = {k: v.get("max_hops") for k, v in QUERY_TYPES.items()}
    except Exception as e:                                    # noqa: BLE001
        print(f"  [warn] 读不到 hop 配置: {e}")

    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.5); ax.axis("off")
    fig.patch.set_facecolor("white")

    C_INPUT, C_CLASS = "#FFF3E0", "#E8F5E9"
    C_VECTOR, C_GRAPH = "#E3F2FD", "#FCE4EC"
    C_FUSION, C_OUTPUT = "#FFF8E1", "#E8EAF6"
    EDGE, EDGE_V, EDGE_G, EDGE_F = "#546E7A", "#1565C0", "#C62828", "#F9A825"

    def box(cx, cy, w, h, text, fill=C_INPUT, edge=EDGE, fs=7.5, bold=False, za=3):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.06", facecolor=fill,
                                    edgecolor=edge, linewidth=1.0, zorder=za))
        kw = {"ha": "center", "va": "center", "fontsize": fs, "zorder": za + 1}
        if bold:
            kw["fontweight"] = "bold"
        ax.text(cx, cy, text, **kw)

    def arrow(x1, y1, x2, y2, color=EDGE, lw=1.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    ax.text(6, 8.2, "Fig. 2  Dual-Path Parallel Retrieval and Adaptive Fusion",
            ha="center", fontsize=13, fontweight="bold", color="#1A237E")

    box(6, 7.45, 5.0, 0.55, "User Natural-Language Question",
        fill=C_INPUT, edge=EDGE, bold=True, fs=8.5)
    arrow(6, 7.18, 6, 6.95)
    box(6, 6.7, 5.8, 0.55,
        "Step 1 — Query Type Classifier  |  Keyword + LLM Cascade  →  "
        "FACTUAL / COMPARATIVE / REASONING / SPATIAL",
        fill=C_CLASS, edge=EDGE, fs=7)
    arrow(3, 6.43, 3, 6.0); arrow(9, 6.43, 9, 6.0)
    ax.text(6, 6.2, "Parallel Retrieval", ha="center", fontsize=7.5,
            color="#78909C", style="italic")

    # 说明文字放进箱体内部当末行。原图把它放在箱体下方居中，
    # 而两条下行箭头正好从 x=3 / x=9 垂直穿过同一位置，字被箭头压掉。
    BOX_CY, BOX_H = 5.22, 1.78
    box(3, BOX_CY, 4.8, BOX_H,
        "Step 2a — Vector Channel\n"
        "text2vec-base-chinese Embedding (768-d)\n"
        "FAISS IndexFlatIP  ·  Cosine Similarity\n"
        "→ Top-k Document Chunks + Source Metadata\n"
        "→ [D1]…[Dn] Citation Anchors\n\n"
        "Document Corpus: stratigraphic chunks",
        fill=C_VECTOR, edge=EDGE_V, fs=6.8)
    box(9, BOX_CY, 4.8, BOX_H,
        "Step 2b — Graph Channel\n"
        "Entity Sub-string Matching in KG\n"
        "BFS Multi-hop Subgraph Expansion (1-3 hop)\n"
        "→ Subgraph Triples + Entity Metadata\n"
        "→ [G1]…[Gn] Citation Anchors\n\n"
        + ("Knowledge Graph: " + (f"{kg_nodes:,} nodes, {kg_edges:,} edges"
                                  if kg_nodes else "—")),
        fill=C_GRAPH, edge=EDGE_G, fs=6.8)

    # hop 映射单独占一行；走线用"先下、再横、再下"，
    # 原图是两条斜线直插，正好穿过这行文字。
    HOP_Y = BOX_CY - BOX_H / 2 - 0.28
    if hops:
        ax.text(6, HOP_Y,
                "  |  ".join(f"{k} → {v}-hop" for k, v in hops.items()),
                ha="center", fontsize=6.2, color="#90A4AE")

    RUN_Y = HOP_Y - 0.26
    arrow(3, BOX_CY - BOX_H / 2, 3, RUN_Y)          # 左臂下行
    arrow(9, BOX_CY - BOX_H / 2, 9, RUN_Y)          # 右臂下行
    ax.plot([3, 6], [RUN_Y, RUN_Y], color=EDGE, lw=1.0, zorder=2)   # 左横
    ax.plot([6, 9], [RUN_Y, RUN_Y], color=EDGE, lw=1.0, zorder=2)   # 右横
    arrow(6, RUN_Y, 6, 3.57)                        # 汇入融合框
    box(6, 3.15, 7.0, 0.85,
        "Step 3 — Adaptive Fusion Generation\n"
        "RRF Fusion (k=60)  ·  Type-adaptive Prompt Templates  ·  LLM Generation\n"
        "Output: Answer with [Dn] / [Gn] Fine-grained Citation Markers",
        fill=C_FUSION, edge=EDGE_F, fs=7, bold=True)
    arrow(6, 2.72, 6, 2.1)
    box(6, 1.65, 6.5, 0.85,
        "Step 4 — Fine-grained Citation Tracing\n"
        "Sentence-level parsing → Regex [Dn]/[Gn] extraction\n"
        "→ Map to source document sentences / KG triples",
        fill=C_OUTPUT, edge=EDGE, fs=7)
    arrow(6, 1.22, 6, 0.65)
    box(6, 0.4, 5.5, 0.45,
        "Output: Cited Answer + Reliability Assessment (0–1 score)",
        fill=C_INPUT, edge=EDGE, bold=True, fs=8)

    savefig(fig, "fig2_dual_retrieval")


# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="论文实验图（fig3/4/5，可选出 fig2）")
    ap.add_argument("--baseline", default=None, help="主实验汇总 json；缺省取最新")
    ap.add_argument("--ablation", default=None, help="消融汇总 json；缺省取最新")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--with-fig2", action="store_true", help="连示意图一起出")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    plt = setup_matplotlib()
    savefig = make_savefig(plt, args.out)

    bp = args.baseline or latest("multirun_baseline_")
    ap_ = args.ablation or latest("multirun_ablation_")
    print(f"主实验数据: {os.path.basename(bp)}")
    print(f"消融数据  : {os.path.basename(ap_)}")
    print(f"输出目录  : {args.out}\n")

    overall, per_type, radar = load_baseline(bp)
    ab = load_ablation(ap_)

    print("--- Fig. 3 主结果（雷达 + 综合得分）---")
    plot_fig3(plt, savefig, overall, radar)
    print("--- Fig. 4 分题型热力图 ---")
    plot_fig4(plt, savefig, per_type)
    print("--- Fig. 5 消融 ---")
    plot_fig5(plt, savefig, ab)
    if args.with_fig2:
        print("--- Fig. 2 双路检索示意图 ---")
        plot_fig2(plt, savefig)

    # 控制台复核：与论文正文将要引用的数字对齐
    print(f"\n{'综合得分（mean ± SD, 3 轮）':<26}")
    for a in BASELINE_ORDER:
        m, s = overall[a]["overall_score"]
        print(f"  {BASELINE_LABEL[a]:<24} {m:.4f} ± {s:.4f}")
    print(f"\n{'消融':<26}")
    for a in ABLATION_ORDER:
        m, s = ab[a]["overall_score"]
        print(f"  {ABLATION_LABEL[a].replace(chr(10),' '):<24} {m:.4f} ± {s:.4f}")
    print(f"\n{'分题型 n':<26}")
    print("  " + "  ".join(f"{qt}={per_type[qt]['dual_path']['count']}" for qt in QTYPES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
