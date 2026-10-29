#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2 方案执行脚本：从《福建地质志.docx》用 LLM 抽取知识图谱三元组（带 evidence/confidence）

- 数据源 : E:\\地质文本空间化平台\\福建地质志.docx（文字版，约 41 万字 / 3915 段）
- 抽取   : 复用 services/kg_extraction.py 的 prompt + services/model_router.py 的 qwen3.5-plus
- 本体   : services/ontology.py（已按 Excel 21 列表头精简，删 CONTACT/THICKNESS 等）
- 产出   : kb_storage/graphs/experiment_kb_llm.graphml（节点带 entity_type；边带 relation/confidence/evidence）
           kb_storage/experiment_triples_llm.json（每条带 evidence/confidence）
- 支持   : 断点续跑（kb_storage/extraction_progress.jsonl）、--limit 做 dry-run

用法:
  python scripts/extract_fujian_kg.py --limit 5          # dry-run：只抽前 5 块
  python scripts/extract_fujian_kg.py --start 3 --limit 5 # 跳过前 3 块（目录等）再抽 5 块
  python scripts/extract_fujian_kg.py                    # 全量
"""
import sys, os, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor

# Windows 终端中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 包根 = GeoStrat-DualPath-RAG
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

# 先加载真实 .env（python服务/.env），确保 BAILIAN_API_KEY 进入环境变量后再 import config
_PY_DIR = os.path.dirname(PKG_DIR)  # python服务/
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PY_DIR, ".env"))
except ImportError:
    pass

import docx
import networkx as nx

from services.kg_extraction import (
    build_extraction_prompt, parse_extraction_response, normalize_triple
)
from services.model_router import get_model_response
from services.ontology import validate_triple

DEFAULT_MODEL = "qwen3.5-plus"
DOCX_PATH = r"E:\地质文本空间化平台\福建地质志.docx"
OUT_GRAPH = os.path.join(PKG_DIR, "kb_storage", "graphs", "experiment_kb_llm.graphml")
OUT_JSON = os.path.join(PKG_DIR, "kb_storage", "experiment_triples_llm.json")
PROGRESS = os.path.join(PKG_DIR, "kb_storage", "extraction_progress.jsonl")


def load_paragraphs(path):
    """读 docx，返回非空段落文本列表"""
    d = docx.Document(path)
    return [p.text.strip() for p in d.paragraphs if p.text.strip()]


def chunk_paragraphs(paras, max_chars=2500):
    """把段落聚合成不超过 max_chars 字的块，边界对齐段落"""
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if len(p) > max_chars:  # 超长段落硬切
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
            continue
        if cur_len + len(p) + 1 > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def extract_one(chunk, model, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            prompt = build_extraction_prompt(chunk, source_name="福建地质志")
            raw = get_model_response(prompt, model, client_data={"max_tokens": 8192})
            return [normalize_triple(t) for t in parse_extraction_response(raw)]
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise last_err


def validate(triples):
    for t in triples:
        valid, err, warns = validate_triple(
            t["entity1"], t["entity1_type"], t["relation"], t["entity2"], t["entity2_type"]
        )
        t["valid"] = valid
        if err:
            t["error"] = err
        if warns:
            t["warnings"] = warns
    return triples


def load_progress(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    return done


_lock = threading.Lock()


def _recover_done(progress_path):
    """从进度文件恢复已完成块的三元组"""
    all_triples = []
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_triples.extend(json.loads(line)["triples"])
                except Exception:
                    pass
    return all_triples


def run(chunks, model, progress_path, resume, workers=1):
    done = load_progress(progress_path) if resume else set()
    all_triples = _recover_done(progress_path) if resume else []
    pending = [i for i in range(len(chunks)) if i not in done]

    def work(i):
        chunk = chunks[i]
        t0 = time.time()
        try:
            triples = validate(extract_one(chunk, model))
            err = None
        except Exception as e:
            triples, err = [], str(e)
        with _lock:
            preview = chunk.replace("\n", " ")[:36]
            if err is None:
                # 成功才写进度文件；失败块不标记完成，续跑会自动重试
                with open(progress_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps({"idx": i, "chunk": chunk, "triples": triples}, ensure_ascii=False) + "\n")
                    pf.flush()
                valid = sum(1 for t in triples if t.get("valid"))
                print(f"[块 {i}/{len(chunks)}] {len(chunk)}字 | {preview}… → {len(triples)} 条(有效 {valid})  {time.time()-t0:.1f}s")
            else:
                print(f"[块 {i}/{len(chunks)}] {len(chunk)}字 | {preview}… → 失败(续跑重试) {err[:60]}")
        return triples

    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for triples in ex.map(work, pending):
                all_triples.extend(triples)
    else:
        for i in pending:
            all_triples.extend(work(i))
    return all_triples


def dedup_and_build(all_triples):
    """去重（保留置信度更高者）+ 建 DiGraph"""
    G = nx.DiGraph()
    seen = {}
    final = []
    type_conflict = 0
    for t in all_triples:
        if not t.get("valid"):
            continue
        h, r, ta = t["entity1"], t["relation"], t["entity2"]
        htype, ttype = t["entity1_type"], t["entity2_type"]
        try:
            conf = float(t.get("confidence") or 0.5)
        except (TypeError, ValueError):
            conf = 0.5
        key = (h, r, ta)
        if key in seen and seen[key][0] >= conf:
            continue
        seen[key] = (conf, t)
        for nm, nt in [(h, htype), (ta, ttype)]:
            if G.has_node(nm):
                if G.nodes[nm].get("entity_type") != nt:
                    type_conflict += 1
            else:
                G.add_node(nm, entity_type=nt)
        G.add_edge(h, ta, relation=r, confidence=conf, evidence=t.get("evidence", ""))
        final.append({
            "head": h, "head_type": htype,
            "relation": r,
            "tail": ta, "tail_type": ttype,
            "evidence": t.get("evidence", ""),
            "confidence": conf
        })
    return G, final, type_conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default=DOCX_PATH)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--start", type=int, default=0, help="跳过前 N 块（目录等）")
    ap.add_argument("--limit", type=int, default=0, help="只抽 N 块（从 --start 算起；结果写正式进度，可被全量续跑）")
    ap.add_argument("--dry", action="store_true", help="dry-run：写临时进度文件、不续跑，纯验证 pipeline 用")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=1, help="并发线程数（不增加 API 成本，只缩短墙钟时间）")
    args = ap.parse_args()

    paras = load_paragraphs(args.docx)
    chunks = chunk_paragraphs(paras, args.max_chars)
    total = len(chunks)
    chunks = chunks[args.start:]
    if args.limit:
        chunks = chunks[:args.limit]

    dry = args.dry
    progress_path = PROGRESS.replace(".jsonl", "_dry.jsonl") if dry else PROGRESS
    resume = (not dry) and (not args.no_resume)

    print(f"段落 {len(paras)} 段 → 全量 {total} 块（max {args.max_chars} 字/块）；"
          f"本次从第 {args.start} 块起抽 {len(chunks)} 块；dry={dry}；model={args.model}；workers={args.workers}")

    all_triples = run(chunks, args.model, progress_path, resume, workers=args.workers)
    G, final, conflicts = dedup_and_build(all_triples)

    os.makedirs(os.path.dirname(OUT_GRAPH), exist_ok=True)
    nx.write_graphml(G, OUT_GRAPH)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    from collections import Counter
    rel_dist = Counter(t["relation"] for t in final)
    etype_dist = Counter(n for _, n in G.nodes(data="entity_type"))
    print("\n=== 抽取完成（本批） ===")
    print(f"节点 {G.number_of_nodes()} / 边 {G.number_of_edges()} / 三元组 {len(final)}")
    print(f"节点类型冲突 {conflicts}")
    print("关系分布:", dict(rel_dist.most_common()))
    print("实体类型分布:", dict(etype_dist.most_common()))
    print(f"图: {OUT_GRAPH}")
    print(f"三元组: {OUT_JSON}")


if __name__ == "__main__":
    main()
