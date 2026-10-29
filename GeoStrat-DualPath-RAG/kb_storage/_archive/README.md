# 归档

这里放的是**已被取代、但尚未删除**的中间产物，避免与当前交付物混淆。

## `_gen_partial_20250825.json`
- 原路径：`kb_storage/_gen_partial.json`（2026-08-25 生成，582 题）
- 来源：早期 `scripts/generate_1000q.py` 的逐批落盘中间文件
- 被取代原因：题 ID 是顺序号（`FJ-I-0001`），生成器一改，已有题的 ID 就会平移，
  无法跨版本对齐。现行 `scripts/generate_honest_set.py` 改用**内容哈希 ID**
  （`FJ-I-<sha1(题干)[:8]>`），与生成顺序无关。
- 另：该版本 `source` 字段一律写 `independent_handwritten`（1005 题全是），
  无法区分程序化题与手写题，已在现行生成器中修正。
- 全仓库已无任何脚本引用本文件。
