# 评估结果封盘归档（2026-09-14）

本目录归档 **2026-09-14 之前**的全部评估结果，已封盘，不再追加。

## 为什么封盘

这批结果全部是 **qwen3.5-plus 同模型裁判**（生成答案与打分用同一个模型）产出的，
存在 self-preference 偏差，**不能作为论文最终数字**（对应审稿意见 R1#4 / R2#7）。

2026-09-11 与导师讨论后决定：**生成模型仍用 qwen3.5-plus，裁判模型改为 DeepSeek-V4.1-Flash**。
裁判模型一变，旧分数与新分数不可混用，故整体封盘。

## ⚠️ 特别注意：progress_*.json 是断点续跑 checkpoint

这 4 个 `progress_*.json` 是各家基线的断点续跑进度文件：**已一并迁移，不要拿回 `results/`**。
否则新评估会从旧 checkpoint 续跑，导致同一次结果里混进 qwen 裁判的旧分数。

如需归档数据的内部相对排序参考，可读本目录下的
`baseline_direct_llm_20260910_095153.json`、`baseline_vector_rag_20260910_095153.json`、
`baseline_graph_rag_20260910_095153.json`。
（dual_path 仅跑到 213/300，见 `progress_dual_path.json`）

## 这批数据的已知局限（除裁判问题外）

- 答案文本未落盘，只存了分数 —— 无法在不重新生成的情况下换裁判重评。
  新版 `evaluator.py` 已修复（`q_result` 增加 `answer` 字段）。
- 抽样为 1005 题中按比例分层取 300 题（SPATIAL 33 全取），
  而导师会议要求测试集扩到 4000–5000 条，故题集基数也将被替换。
- 跑的过程中曾发现 graph_rag 因 graphml 文件头混入 `·` 字符被读成空图，
  已修复（`services/kg_service.py` 的 `_read_graphml`），旧 checkpoint 中
  `progress_graph_rag.json` 的早期部分可能包含空图结果。
