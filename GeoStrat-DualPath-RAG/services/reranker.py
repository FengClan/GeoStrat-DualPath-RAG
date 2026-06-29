# services/reranker.py — 混合检索 + Cross-Encoder 重排序
import re
import time
import math
from collections import defaultdict

from flask import Blueprint, request, jsonify

reranker_bp = Blueprint('reranker', __name__)

# 尝试加载 Cross-Encoder
_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
            print("[INFO] Reranker: loaded BAAI/bge-reranker-large.")
        except Exception as e:
            print(f"[WARN] Cannot load reranker model: {e}")
            _reranker_model = False
    return _reranker_model if _reranker_model is not False else None


# ============================================================
# BM25 稀疏检索
# ============================================================


class BM25Retriever:
    """轻量级 BM25 检索器，无需外部索引库"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.avgdl = 0.0

    def index(self, documents):
        """documents: [{"id": str, "text": str}, ...]"""
        self.documents = documents
        self.doc_freqs.clear()
        for doc in documents:
            terms = set(self._tokenize(doc["text"]))
            for t in terms:
                self.doc_freqs[t] += 1
        total_len = sum(len(self._tokenize(d["text"])) for d in documents)
        self.avgdl = total_len / max(len(documents), 1)

    def _tokenize(self, text):
        """中文+英文混合分词"""
        tokens = []
        # 中文字符单独作为 token
        for ch in text:
            if '一' <= ch <= '鿿':
                tokens.append(ch)
        # 英文/数字词
        for word in re.findall(r'[a-zA-Z0-9]+', text):
            tokens.append(word.lower())
        return tokens

    def search(self, query, top_k=10):
        """BM25 评分检索"""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        N = len(self.documents)
        scores = []

        for i, doc in enumerate(self.documents):
            doc_terms = self._tokenize(doc["text"])
            doc_len = len(doc_terms)
            term_freqs = defaultdict(int)
            for t in doc_terms:
                term_freqs[t] += 1

            score = 0.0
            for qt in query_terms:
                tf = term_freqs[qt]
                if tf == 0:
                    continue
                df = self.doc_freqs.get(qt, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1.0))
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[i] for i, _ in scores[:top_k]]


# ============================================================
# 混合检索 + 重排序
# ============================================================


def hybrid_retrieve(query, vector_docs, bm25_index, top_k=20):
    """
    混合检索：向量检索 + BM25 稀疏检索 → 候选并集 → Cross-Encoder 重排序
    vector_docs: 向量检索返回的文档列表 [Document, ...]
    bm25_index: BM25Retriever 实例
    """
    # Dense candidates
    dense_ids = set()
    dense_map = {}
    for i, doc in enumerate(vector_docs):
        doc_id = doc.metadata.get('chunk_index', f"d_{i}")
        dense_ids.add(doc_id)
        dense_map[doc_id] = {"doc": doc, "dense_rank": i + 1}

    # Sparse candidates
    bm25_results = bm25_index.search(query, top_k=top_k)
    sparse_map = {}
    for i, item in enumerate(bm25_results):
        doc_id = item.get("id", f"s_{i}")
        sparse_map[doc_id] = {"doc": item, "sparse_rank": i + 1}

    # Candidate union
    all_ids = set(dense_ids) | set(sparse_map.keys())

    # RRF (Reciprocal Rank Fusion)
    rrf_scores = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in dense_map:
            score += 1.0 / (60 + dense_map[doc_id]["dense_rank"])
        if doc_id in sparse_map:
            score += 1.0 / (60 + sparse_map[doc_id]["sparse_rank"])
        rrf_scores[doc_id] = score

    # Top candidates
    sorted_ids = sorted(all_ids, key=lambda x: rrf_scores.get(x, 0), reverse=True)[:top_k]

    # Cross-Encoder 重排序
    reranker = _get_reranker()
    if reranker:
        pairs = []
        for doc_id in sorted_ids:
            doc_text = ""
            if doc_id in dense_map:
                doc_text = dense_map[doc_id]["doc"].page_content
            elif doc_id in sparse_map:
                doc_text = sparse_map[doc_id]["doc"].get("text", "")
            pairs.append([query, doc_text[:400]])

        scores = reranker.predict(pairs)
        # 按重排序分数重新排列
        scored = list(zip(sorted_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_ids = [sid for sid, _ in scored[:top_k]]

    # 构建最终结果
    final_docs = []
    for doc_id in sorted_ids:
        if doc_id in dense_map:
            final_docs.append(dense_map[doc_id]["doc"])
        elif doc_id in sparse_map:
            item = sparse_map[doc_id]["doc"]
            from langchain.schema import Document
            final_docs.append(Document(
                page_content=item.get("text", ""),
                metadata=item.get("metadata", {})
            ))

    return final_docs


def build_bm25_from_vectordb(kb_id="all"):
    """从 ChromaDB 集合构建 BM25 索引"""
    from rag_core import vectordb
    try:
        collection = vectordb._collection
        result = collection.get(include=["documents", "metadatas"])
        docs = []
        if result and result.get("ids"):
            for i, doc_id in enumerate(result["ids"]):
                doc_text = result["documents"][i] if result.get("documents") else ""
                meta = result["metadatas"][i] if result.get("metadatas") else {}
                if kb_id == "all" or meta.get("kb_id") == kb_id:
                    docs.append({"id": doc_id, "text": doc_text, "metadata": meta})
        bm25 = BM25Retriever()
        bm25.index(docs)
        return bm25
    except Exception as e:
        print(f"[WARN] BM25 index build failed: {e}")
        return None


# ============================================================
# API
# ============================================================


@reranker_bp.route('/retrieve/hybrid', methods=['POST'])
def hybrid_retrieve_endpoint():
    """混合检索接口"""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    kb_id = data.get("kb_id", "all")
    top_k = data.get("top_k", 10)

    if not query:
        return jsonify({"error": "查询字符串不能为空"}), 400

    try:
        from rag_core import retrieve_from_vectordb
        vector_docs = retrieve_from_vectordb(query, kb_id, top_k=top_k * 2)

        bm25 = build_bm25_from_vectordb(kb_id)

        if bm25 and vector_docs:
            results = hybrid_retrieve(query, vector_docs, bm25, top_k=top_k)
        elif vector_docs:
            results = vector_docs[:top_k]
        else:
            results = []

        formatted = []
        for doc in results:
            formatted.append({
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source_file", "unknown"),
                "semantic_type": doc.metadata.get("semantic_type", "paragraph"),
                "score": round(float(doc.metadata.get("score", 0)), 4)
            })

        return jsonify({
            "status": "success",
            "query": query,
            "count": len(formatted),
            "results": formatted
        })
    except Exception as e:
        return jsonify({"error": f"混合检索失败: {str(e)}"}), 500
