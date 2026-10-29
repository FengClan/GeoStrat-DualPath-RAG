#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把诚实测试集切分为 dev（开发/调参）+ test（留出/只报一次）——回应审稿意见 R2#3。

审稿人 R2#3 的原话是要证明：测试问题没有被直接或间接用于选择提示词、融合权重、
检索参数或图跳数。本仓库此前**没有任何 dev/test 划分**，所有题都是"测试集"——
这意味着今后任何一次调参（包括我们自己）都会当场违反 R2#3。

切分规则：
  - 按 query_type 分层按比例分配（最大余数法），保证四类在 dev/test 中占比一致；
  - **以题目内容哈希 id 为划分依据**（`FJ-I-<sha1>`），不用行号/序号——
    行号会随生成器版本漂移（本仓库历史上已经踩过一次，见 kb_storage/_archive/README.md），
    内容哈希则同一个问题永远落在同一侧；
  - 固定 seed，脚本可重复执行得到完全相同的结果。

产出：
  test_sets/test_set_honest_dev.json    —— 调参用；prompt/权重/hop 只准在这里试
  test_sets/test_set_honest_test.json   —— 留出集；主实验只报一次数
  test_sets/test_set_honest_split.json  —— 划分清单（记录哪些 id 在哪一侧 + 方法）

用法:
    python scripts/split_dev_test.py                 # 默认 dev 150 题
    python scripts/split_dev_test.py --dev-size 200 --seed 42
"""
import sys, os, json, re, random, argparse
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SETS = os.path.join(PKG_DIR, "test_sets")
QUERY_TYPES = ["FACTUAL", "REASONING", "COMPARATIVE", "SPATIAL"]


def allocate(n, sizes):
    """最大余数法：把 n 按 sizes 比例分配到各桶，保证总和恰为 n。"""
    total = sum(sizes)
    if total == 0:
        return [0] * len(sizes)
    raw = [n * s / total for s in sizes]
    floors = [int(r) for r in raw]
    rem = n - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    alloc = floors[:]
    for i in order[:rem]:
        alloc[i] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser(description="切分 dev / test（回应 R2#3）")
    ap.add_argument("--test-set",
                    default=os.path.join(TEST_SETS, "test_set_honest.json"))
    ap.add_argument("--dev-size", type=int, default=150,
                    help="dev 集题数（默认 150，约占 1307 的 11%%）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="test_set_honest")
    args = ap.parse_args()

    data = json.load(open(args.test_set, encoding="utf-8"))
    questions = data["questions"]
    total = len(questions)

    if args.dev_size >= total:
        raise SystemExit(f"dev-size({args.dev_size}) 必须小于总题数({total})")

    # 分层：按 query_type 分组，组内按 id 排序后抽样（排序 → 结果与输入顺序无关）
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.get("query_type", "FACTUAL")].append(q)
    for t in by_type:
        by_type[t].sort(key=lambda q: str(q["id"]))

    types = [t for t in QUERY_TYPES if by_type.get(t)]
    types += [t for t in sorted(by_type) if t not in types]      # 兜底：未预期的新类型
    alloc = allocate(args.dev_size, [len(by_type[t]) for t in types])

    rng = random.Random(args.seed)
    dev_ids = set()
    for t, n in zip(types, alloc):
        pool = by_type[t]
        n = min(n, len(pool))
        for q in rng.sample(pool, n):
            dev_ids.add(q["id"])

    dev = [q for q in questions if q["id"] in dev_ids]
    test = [q for q in questions if q["id"] not in dev_ids]

    # ---- 不变量校验：切分必须无损、无重叠 ----
    assert len(dev) + len(test) == total, "切分后总数对不上"
    assert not (set(q["id"] for q in dev) & set(q["id"] for q in test)), "dev/test 有重叠"
    assert len(set(q["id"] for q in dev)) == len(dev), "dev 内有重复 id"

    def write(name, qs):
        path = os.path.join(TEST_SETS, name)
        out = dict(data)
        out["questions"] = qs
        out["split"] = {"role": "dev" if name.endswith("_dev.json") else "test",
                        "seed": args.seed, "source": os.path.basename(args.test_set)}
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return path

    dev_path = write(f"{args.prefix}_dev.json", dev)
    test_path = write(f"{args.prefix}_test.json", test)

    manifest = {
        "method": "按 query_type 分层 + 最大余数分配；以内容哈希 id 为划分依据（非行号），"
                  "组内按 id 排序后 rng.sample，seed 固定 → 可重复",
        "seed": args.seed,
        "source": os.path.basename(args.test_set),
        "source_total": total,
        "dev_size": len(dev),
        "test_size": len(test),
        "dev_by_type": dict(Counter(q.get("query_type") for q in dev)),
        "test_by_type": dict(Counter(q.get("query_type") for q in test)),
        "dev_ids": sorted(dev_ids),
        "note": "调参（prompt/融合权重/hop）只准用 dev；test 只报一次（R2#3）。"
                "两侧 ID 互斥且并集等于全集，已在脚本内断言。",
    }
    mpath = os.path.join(TEST_SETS, f"{args.prefix}_split.json")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"源: {os.path.basename(args.test_set)}  共 {total} 题  seed={args.seed}")
    print(f"\n{'题型':<14}{'dev':>8}{'test':>8}{'合计':>8}")
    print("-" * 40)
    for t in types:
        d = manifest["dev_by_type"].get(t, 0)
        s = manifest["test_by_type"].get(t, 0)
        print(f"{t:<14}{d:>8}{s:>8}{d + s:>8}")
    print("-" * 40)
    print(f"{'合计':<14}{len(dev):>8}{len(test):>8}{total:>8}")
    print(f"\n已写出:\n  {dev_path}\n  {test_path}\n  {mpath}")
    print("\n★ 今后调参只在 dev 上做；test 集在最终报告前不要碰（R2#3）。")


if __name__ == "__main__":
    main()
