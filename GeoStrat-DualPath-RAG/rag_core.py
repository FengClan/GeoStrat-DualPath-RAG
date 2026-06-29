# 文件名：rag_core.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import re
import uuid
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---- Embedding 配置 ----
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese")

print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    print("[INFO] Embedding model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Embedding model load failed: {e}")
    print("[INFO] Trying local path fallback: ./text2vec-base-chinese ...")
    try:
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text2vec-base-chinese")
        embedding_model = SentenceTransformer(local_path)
        print("[INFO] Embedding model loaded from local path.")
    except Exception as e2:
        print(f"[ERROR] Local path fallback also failed: {e2}")
        raise RuntimeError("Embedding model load failed, system cannot start.")

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)


# ============================================================
# FAISSVectorStore — 使用 FAISS 替代 ChromaDB
# ============================================================

class FAISSVectorStore:
    """基于FAISS的向量存储，接口兼容langchain Chroma"""

    def __init__(self, persist_dir, embedding_model):
        import faiss
        import pickle
        self._model = embedding_model
        self._dir = persist_dir
        # 使用 serialize_index/deserialize_index 避免 FAISS C++ fopen 不支持中文路径
        self._index_path = os.path.join(persist_dir, 'faiss.index')
        self._meta_path = os.path.join(persist_dir, 'faiss_meta.pkl')
        dim = embedding_model.get_sentence_embedding_dimension()
        self._dim = dim if isinstance(dim, int) else 768
        self._documents = []
        self._index = None

        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            try:
                import numpy as np
                with open(self._index_path, 'rb') as f:
                    self._index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
                with open(self._meta_path, 'rb') as f:
                    self._documents = pickle.load(f)
                print(f"[INFO] FAISS loaded: {self._index.ntotal} docs")
            except Exception as e:
                print(f"[WARN] FAISS load failed, starting fresh: {e}")
                self._index = faiss.IndexFlatIP(self._dim)
                self._documents = []
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._documents = []

    def _save(self):
        import pickle, faiss
        data = faiss.serialize_index(self._index)
        with open(self._index_path, 'wb') as f:
            f.write(data.tobytes() if hasattr(data, 'tobytes') else data)
        with open(self._meta_path, 'wb') as f:
            pickle.dump(self._documents, f)

    def _normalize(self, vecs):
        import numpy as np
        vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def add_documents(self, documents):
        if not documents:
            return []
        texts = [d.page_content for d in documents]
        embeddings = self._model.encode(texts)
        embeddings = self._normalize(embeddings).astype('float32')
        start = len(self._documents)
        self._index.add(embeddings)
        self._documents.extend(documents)
        self._save()
        return list(range(start, len(self._documents)))

    def similarity_search(self, query, k=5, filter=None):
        import numpy as np
        if self._index.ntotal == 0:
            return []
        q_emb = self._model.encode([query])
        q_emb = self._normalize(q_emb).astype('float32')
        scores, indices = self._index.search(q_emb, k)

        docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            meta = dict(doc.metadata) if doc.metadata else {}
            meta['score'] = float(score)

            if filter:
                skip = False
                for kf, vf in filter.items():
                    if meta.get(kf) != vf:
                        skip = True
                        break
                if skip:
                    continue

            docs.append(Document(page_content=doc.page_content, metadata=meta))
        return docs[:k]


vectordb = FAISSVectorStore(PERSIST_DIRECTORY, embedding_model)


# ============================================================
# Stratigraphic Chunker — 岩石地层自适应文本分块
# ============================================================


class StratigraphicChunker:
    """根据地质文献结构特征进行智能分块，保留语义单元边界"""

    # 地质章节标题模式
    SECTION_PATTERNS = [
        re.compile(r'^第[一二三四五六七八九十\d]+章\s*[.\s]'),
        re.compile(r'^第[一二三四五六七八九十\d]+节\s*[.\s]'),
        re.compile(r'^\d+[.、]\s*'),
        re.compile(r'^[一二三四五六七八九十]+[、.]\s*'),
    ]

    # 地层剖面/钻孔描述起始标记
    BOREHOLE_PATTERN = re.compile(
        r'^(钻孔|钻孔编号|剖面|实测剖面|柱状图|ZK\d+|CK\d+|BH\d+|B\d+)',
        re.IGNORECASE
    )

    # 表格检测
    TABLE_PATTERN = re.compile(r'^(表\d+|Table\s*\d+|附表\d+)')

    # 地层单位描述段落标记
    FORMATION_PATTERN = re.compile(
        r'^(\S*(群|组|段|岩|系|统|阶))\s*[:：]?\s*(以|位于|分布|出露|发育|厚度|岩性|主要|属于)'
    )

    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
        )

    def _detect_semantic_type(self, text):
        """检测文本片段的语义类型"""
        text_stripped = text.strip()
        if any(p.match(text_stripped) for p in self.SECTION_PATTERNS):
            return "section_header"
        if self.BOREHOLE_PATTERN.match(text_stripped):
            return "borehole_entry"
        if self.TABLE_PATTERN.match(text_stripped):
            return "table_caption"
        if self.FORMATION_PATTERN.match(text_stripped):
            return "formation_description"
        if len(text_stripped) > 200 and any(kw in text_stripped for kw in ['岩性', '厚度', '化石', '接触']):
            return "stratigraphic_column"
        return "paragraph"

    def split_text(self, text, source_name="", kb_id="default_kb"):
        """
        按地质语义边界分块。
        返回 LangChain Document 列表，每块带有 semantic_type / unit_name / page 等元数据
        """
        documents = []

        # Step 1: 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        # Step 2: 合并过短段落，保持语义完整
        merged = []
        buf = ""
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(buf) + len(p) < self.chunk_size:
                buf = (buf + "\n\n" + p).strip() if buf else p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        # Step 3: 对过长段落使用 fallback 分块
        for i, para in enumerate(merged):
            if len(para) <= self.chunk_size:
                chunk_text = para
                chunks = [chunk_text]
            else:
                fallback_chunks = self._fallback.split_text(para)
                chunks = fallback_chunks

            for j, chunk_text in enumerate(chunks):
                semantic_type = self._detect_semantic_type(chunk_text)
                # 尝试提取地层单位名
                unit_name = self._extract_unit_name(chunk_text)

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": source_name,
                        "kb_id": kb_id,
                        "semantic_type": semantic_type,
                        "unit_name": unit_name or "",
                        "chunk_index": len(documents),
                        "para_index": i,
                        "char_count": len(chunk_text)
                    }
                )
                documents.append(doc)

        return documents

    def _extract_unit_name(self, text):
        """从文本片段中提取可能的地层单位名"""
        patterns = [
            r'(\S{1,6}(?:群|组|段|岩))(?:[:：，,\s]|$)',
            r'(?:称为?|命名?为|称作)\s*(\S{1,6}(?:群|组|段))',
        ]
        for ptn in patterns:
            m = re.search(ptn, text)
            if m:
                return m.group(1)
        return None


# ============================================================
# 领域 Embedding 别名映射
# ============================================================

EMBEDDING_MODELS = {
    "text2vec": "shibing624/text2vec-base-chinese",
    "bge-large": "BAAI/bge-large-zh-v1.5",
    "bge-base": "BAAI/bge-base-zh-v1.5",
    "m3e": "moka-ai/m3e-base",
    "stella": "infgrad/stella-base-zh-v3-1792d",
}

_strat_chunker = None


def get_stratigraphic_chunker():
    global _strat_chunker
    if _strat_chunker is None:
        _strat_chunker = StratigraphicChunker()
    return _strat_chunker


def change_embedding_model(model_key="bge-large"):
    """切换 Embedding 模型（运行时调用）"""
    global embeddings, vectordb
    model_name = EMBEDDING_MODELS.get(model_key, model_key)
    print(f"[INFO] Switching to embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return model_name


def process_and_store_file(file_path, filename, kb_id):
    """解析文件、使用 StratigraphicChunker 分块并存入向量数据库"""
    try:
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
        elif filename.lower().endswith(('.txt', '.md')):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {filename}")

        docs = loader.load()

        # 合并所有文档内容用于语义分块
        full_text = "\n\n".join([d.page_content for d in docs])
        chunker = get_stratigraphic_chunker()
        chunks = chunker.split_text(full_text, source_name=filename, kb_id=kb_id)

        if chunks:
            vectordb.add_documents(chunks)

        return len(chunks)
    except Exception as e:
        print(f"[RAG ERROR] 文件处理失败 {filename}: {str(e)}")
        raise e


def retrieve_from_vectordb(query, kb_id="all", top_k=3):
    """
    根据问题从向量数据库中检索，支持按 kb_id 过滤，并返回引用的格式化数据
    返回: (context_text, citations_list)
    """
    try:
        # 构造过滤条件
        filter_dict = {}
        if kb_id and kb_id != "all":
            filter_dict["kb_id"] = kb_id

        # 检索相似文档块
        if filter_dict:
            docs = vectordb.similarity_search(query, k=top_k, filter=filter_dict)
        else:
            docs = vectordb.similarity_search(query, k=top_k)

        context_text = ""
        citations = []

        for i, doc in enumerate(docs):
            doc_name = doc.metadata.get('source_file', '未知文档')
            # 清理换行符，让片段更连贯
            content = doc.page_content.replace('\n', ' ')

            # 1. 拼接给大模型看的 Context
            context_text += f"【参考片段 {i + 1}】(来源: {doc_name}):\n{content}\n\n"

            # 2. 构造给前端溯源气泡展示的 Citation
            excerpt = content[:150] + "..." if len(content) > 150 else content
            citations.append({
                "docName": doc_name,
                "excerpt": excerpt
            })

        return context_text.strip(), citations
    except Exception as e:
        print(f"[RAG ERROR] 检索失败: {str(e)}")
        return "", []