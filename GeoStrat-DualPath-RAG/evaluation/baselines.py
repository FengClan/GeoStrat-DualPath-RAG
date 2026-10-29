# evaluation/baselines.py — 多基线对比评估
"""
四种检索策略的系统对比:
  A. Direct LLM (无RAG) — 纯LLM回答
  B. Vector RAG only — 仅向量检索 + LLM
  C. Graph RAG only — 仅图谱检索 + LLM
  D. Dual-path Adaptive Fusion (Proposed) — 双路检索 + 自适应融合

每种基线被封装为 SystemRunner callable，接受 question dict 返回系统输出。
"""

import time
import json
import os
import sys
import threading

# 确保能导入项目服务
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 并发评测时保护向量库查询（见 _retrieve_docs 注释）
_VEC_LOCK = threading.Lock()


# —— 图谱本体覆盖判断：图谱只对「它确实编码的关系」权威 ——
# D2 阶段已删除 THICKNESS/HAS_THICKNESS、CONTACTS 关系，且本体无「命名/创名/沉积环境」关系。
# 当问题只问这些「图谱盲区」属性时，关闭图谱通道、退化为文献通道（真正的自适应退化）。
_GRAPH_BLIND_ATTRS = (
    '厚度', '命名', '创名', '创建', '接触', '不整合', '整合',
    '沉积环境', '沉积相', '原岩', '古水流', '含矿',
)
_GRAPH_COVER_ATTRS = (
    '岩性', '岩石', '年代', '时代', '分布', '展布', '出露', '位于', '见于',
    '化石', '同义', '组成', '层位', '含煤', '序列', '上下', '归属',
)


def _graph_is_blind(question):
    """判断问题所问属性是否在图谱本体之外（图谱盲区），若是则双路法应退化为文献通道。"""
    blind = any(k in question for k in _GRAPH_BLIND_ATTRS)
    cover = any(k in question for k in _GRAPH_COVER_ATTRS)
    return blind and not cover


class BaselineRunner:
    """管理多基线对比评估的运行器"""

    BASELINE_NAMES = {
        "direct_llm": "纯LLM (无RAG)",
        "vector_rag": "向量RAG",
        "graph_rag": "图谱RAG",
        "dual_path": "双路自适应融合 (Proposed)",
        # 对照组（2026-09-16 加）：与 dual_path 逐行相同，**只改一个变量**——
        # 图谱启用时也把文献上下文给满（top-5/500），而非 dual_path 的 top-3/400。
        # 目的：把「图谱带来的增益」与「少给 40% 文献带来的代价」分开。否则
        # "双路 vs 向量"比的是「减配文献 + 图谱」vs「满配文献」，说不清功劳归谁。
        "dual_path_fullctx": "双路+满配文献 (对照)",
        # 消融（2026-09-16 加）：删掉图谱盲区门控，图谱永不关闭。论文把"自适应退化"
        # 写成设计却从未消融过它，这是补上那个缺失的消融。
        "dual_path_nogate": "D − 自适应退化 (图谱永不关)",
        # 消融（2026-09-19 归位）：以下两臂与 dual_path 共用 _dual_path_impl，
        # 只差一个开关。此前它们住在 ablation.py、是一份复制后未同步的旧实现
        # （旧 system 指令、无门控、无盲区分支），跑出来的"D. 完整方法"与主表
        # "双路"不是同一个东西，差值不可解释。现统一到唯一实现体。
        "dual_path_no_classify": "D − 自适应分类 (固定FACTUAL模板)",
        "dual_path_no_citation": "D − 细粒度引用 (无[G]/[D]标注)",
    }

    def __init__(self, model="qwen3.5-plus", kb_id="all", results_dir=None,
                 hops_override=None):
        self.model = model
        self.kb_id = kb_id
        self.results_dir = results_dir or "./evaluation/results"
        # 跳数敏感性分析用（R2#4）：设了就强制覆盖分类器给出的 max_hops，
        # 用来跑 hop=1/2/3 的对照。None = 正常行为（按查询类型自适应）。
        # ⚠️ 敏感性实验只作分析报告，**不得据此挑"最好的 hop"写进主实验**——
        #    那等于用测试集调参，正好违反 R2#3。
        self.hops_override = hops_override

    # ================================================================
    # Baseline A: Direct LLM (no retrieval)
    # ================================================================

    def _runner_direct_llm(self, question_dict):
        """纯LLM回答，无任何检索"""
        from services.model_router import get_model_response

        t0 = time.time()
        prompt = f"""你是一位专业地质学家。请回答以下问题：

问题：{question_dict['question']}

请给出专业、准确的回答。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        return {
            "answer": answer,
            "retrieved_docs": [],
            # 纯 LLM 没有任何检索通道 → 召回类指标不适用，报 N/A 而非 0（R2#6）
            "retrieval_applicable": False,
            # 无检索 → 无展示片段，片段级证据召回同样记 N/A
            "surfaced_spans": [],
            "evidence_applicable": False,
            "citations": [],
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Helper: 直接调用 ChromaDB 检索，返回 Document 列表
    # ================================================================

    def _retrieve_docs(self, query, kb_id, top_k=5):
        """绕过 retrieve_from_vectordb（返回元组），直接获取Document对象

        并发评测（run_full_evaluation(workers>1)）时用 _VEC_LOCK 串行化：
        similarity_search 内部要调嵌入模型做 query 编码，底层模型句柄不是
        线程安全的；而检索本身只占单题耗时极小一部分（生成要几十秒），
        串行化它不会抵消并发收益。
        """
        from rag_core import vectordb
        try:
            with _VEC_LOCK:
                if kb_id and kb_id != "all":
                    return vectordb.similarity_search(query, k=top_k, filter={"kb_id": kb_id})
                else:
                    return vectordb.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[Baseline] Vector search failed: {e}")
            return []

    # ================================================================
    # Baseline B: Vector RAG only
    # ================================================================

    # ================================================================
    # Helper: 将KG原始边列表格式化为可读的结构化事实
    # ================================================================

    def _format_kg_facts(self, kg_edges, filter_noise=False):
        """将 retrieve_graph_context 返回的边列表格式化为分组事实"""
        if not kg_edges:
            return ""
        if isinstance(kg_edges, str):
            return kg_edges

        # kg_edges is list of "A --[REL]--> B" strings
        from collections import defaultdict
        entity_facts = defaultdict(lambda: defaultdict(list))

        # 无意义的通用节点（对具体问题帮助不大）
        GENERIC_TARGETS = {'福建省', '地层小区', '中国', '福建省区域', '福建省地质'}
        # 优先展示的关系类型
        PRIORITY_RELS = {'COMPOSED_OF', 'DATED_AS', 'BELONGS_TO', 'SAME_AS',
                        'CONTAINS_FOSSIL', 'CONTACTS', 'HAS_THICKNESS'}

        for edge_text in kg_edges:
            import re
            m = re.match(r'(.+?)\s*--\[(.+?)\]-+>\s*(.+)', edge_text)
            if m:
                src, rel, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                # 过滤无意义分布节点
                if filter_noise and rel == 'DISTRIBUTED_IN' and tgt in GENERIC_TARGETS:
                    continue
                entity_facts[src][rel].append(tgt)

        lines = []
        for entity, rels in entity_facts.items():
            # 优先展示核心关系
            priority_lines = []
            other_lines = []
            for rel, targets in rels.items():
                rel_label = {
                    'COMPOSED_OF': '岩石组成', 'DATED_AS': '地质年代',
                    'DISTRIBUTED_IN': '分布区域', 'BELONGS_TO': '所属单位',
                    'SAME_AS': '同义名称', 'CONTAINS_FOSSIL': '含化石',
                    'HAS_THICKNESS': '厚度', 'CONTACTS': '接触关系',
                    'MEASURED_AT': '实测剖面'
                }.get(rel, rel)
                line = f"  - {rel_label}: {', '.join(targets[:8])}"
                if rel in PRIORITY_RELS:
                    priority_lines.append(line)
                else:
                    other_lines.append(line)

            if priority_lines or other_lines:
                lines.append(f"【{entity}】")
                lines.extend(priority_lines)
                lines.extend(other_lines)
                lines.append("")
        return '\n'.join(lines)

    def _runner_vector_rag(self, question_dict):
        """仅使用向量检索增强"""
        from services.model_router import get_model_response

        t0 = time.time()
        retrieved_docs = []
        context_text = ""
        # docs 必须在 try 之外先绑定：检索抛异常时它不会被赋值，而下面返回值里
        # 要引用它（surfaced_spans）——否则检索失败会从"退化为无上下文"变成 NameError。
        docs = []

        try:
            docs = self._retrieve_docs(question_dict['question'], self.kb_id, top_k=5)
            if docs:
                retrieved_docs = [
                    {"id": d.metadata.get("source_file", f"doc{i}"),
                     "source_file": d.metadata.get("source_file", ""),
                     "chunk_index": d.metadata.get("chunk_index"),
                     # 存全文而非前 200 字：片段级证据召回要在完整 chunk 里找
                     # evidence_span，截断会让真命中被判成未命中。
                     "score": d.metadata.get("score", 0.5),
                     "content": d.page_content}
                    for i, d in enumerate(docs)
                ]
                context_text = "\n\n".join([
                    f"[文献片段 {i+1}] {d.page_content[:500]}"
                    for i, d in enumerate(docs)
                ])
        except Exception as e:
            print(f"[Baseline-Vector] Retrieval failed: {e}")

        prompt = f"""你是一位专业地质学家。你必须严格基于以下参考文献回答问题，不得依赖训练数据中的记忆。

注意：如果参考文献中缺乏相关信息，请明确说明"文献中未找到相关信息"，不要编造。

## 参考文献
{context_text if context_text else '暂无相关文献'}

## 问题
{question_dict['question']}

请基于文献给出专业回答，标注出处。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        citations = list(set(d.get("id", "") for d in retrieved_docs))

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "citations": citations,
            # 被召回片段的完整正文——片段级证据召回的真值比对用这个。
            # 判定单位是"片段"，故用整段而非 prompt 中截断后的文本：
            # prompt 里的 500 字截断属生成侧设置，不该影响"是否召回了证据"的判定。
            "surfaced_spans": [d.page_content for d in docs],
            "evidence_applicable": True,
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Baseline C: Graph RAG only
    # ================================================================

    def _runner_graph_rag(self, question_dict):
        """仅使用知识图谱检索增强"""
        from services.model_router import get_model_response
        from services.kg_service import retrieve_graph_context

        t0 = time.time()
        kg_edges = []
        kg_facts = ""

        try:
            result = retrieve_graph_context(
                question_dict['question'], self.kb_id,
                max_hops=self.hops_override or 2
            )
            if isinstance(result, tuple) and len(result) == 2:
                kg_edges = result[0] if isinstance(result[0], list) else []
            elif isinstance(result, list):
                kg_edges = result
            kg_facts = self._format_kg_facts(kg_edges)
        except Exception as e:
            print(f"[Baseline-Graph] KG retrieval failed: {e}")

        prompt = f"""你是一位专业地质学家。你必须严格基于以下知识图谱结构化事实回答问题。图谱中的事实为权威数据，如与你的训练知识冲突，以图谱事实为准。

## 知识图谱事实（结构化地层属性）
{kg_facts if kg_facts else '暂无相关知识图谱信息'}

## 问题
{question_dict['question']}

请基于图谱中的结构化事实给出专业回答。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        return {
            "answer": answer,
            "retrieved_docs": [],
            # 图谱通道返回的是三元组、不是文本片段，无 recall@K/MRR 可言
            # → 报 N/A 而非 0（R2#6）
            "retrieval_applicable": False,
            # 图谱边其实带 evidence（实测 2894/2894 条都有），但
            # retrieve_graph_context 返回的是 "A --[REL]--> B" 纯字符串，
            # evidence 从未传给生成器 → 无可比对的"展示片段"，片段级指标记 N/A。
            # 这是**行为缺口**（非度量问题），单独作为改进项处理，不与本次度量修正混做。
            "surfaced_spans": [],
            "evidence_applicable": False,
            "citations": ["福建地质志.docx"] if kg_facts else [],
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": "FACTUAL",
            "latency_ms": latency
        }

    # ================================================================
    # Baseline D: Dual-path Adaptive Fusion (Proposed)
    # ================================================================

    def _runner_dual_path(self, question_dict):
        """双路检索 + 自适应融合 (论文提出的方法)"""
        return self._dual_path_impl(question_dict, full_context=False)

    def _runner_dual_path_fullctx(self, question_dict):
        """对照组：与 _runner_dual_path 逐行相同，仅「图谱启用时给多少文献」这一个变量不同。

        动机：_runner_dual_path 在图谱启用时主动把文献从 top-5/500 砍到 top-3/400
        （原注释称"避免无关片段稀释图谱信号"），于是"双路 vs 向量"实际比的是
        「减配 40% 文献 + 图谱」vs「满配文献」，分不清增益来自图谱还是来自那次减法。
        本臂把文献补满、其余一切不变，两者之差即那次减配的净代价。

        用**函数参数**而非实例属性传开关：--workers>1 是多线程，实例属性会串
        （一个线程还原成 False 时另一个线程仍在跑），故这里不碰 self。
        """
        return self._dual_path_impl(question_dict, full_context=True)

    def _runner_dual_path_nogate(self, question_dict):
        """消融「D − 自适应退化」：删掉 _graph_is_blind 门控，图谱**永不关闭**。

        动机：门控是论文 3.3.2 节写明的设计（"问的属性不在图谱本体里 → 退化为纯文献"），
        但论文从未消融过它。实测该门控在留出集上关掉了 28.9%（334/1157）的问题，
        且这些题**全部**能从图谱检索到非空事实——返回的正是对比所需的层级、岩性组合、
        年代与地层分区等事实。门控把"所问属性缺失"当成了"图谱对这个实体无用"。

        本臂只改这一个变量：honor_gate=False。其余（prompt 模板、文献取 top-3/400、
        检索参数）一律与 _runner_dual_path 相同。若本臂更差，则门控是必要设计；
        若更好，则门控在过度触发——两种结果都能写进论文。
        """
        return self._dual_path_impl(question_dict, full_context=False, honor_gate=False)

    def _runner_dual_path_no_classify(self, question_dict):
        """消融「D − 自适应分类」：跳过查询分类器，全部问题按 FACTUAL 处理。

        分类器在这条路径上实际只控制两件事：prompt 模板（三分支）与 max_hops。
        本臂把两者钉死在默认值，其余（检索、文献取法、引用指令、门控）一字不动。

        说明：论文摘要/4.7 节曾把分类器描述为"差异化向量与图谱权重"，但代码里
        config 的 vector_ratio/graph_ratio 从未被读取。因此本臂量的**不是**权重
        差异，而是"按题型换模板 + 按题型调跳数"的净效果 —— 论文相应表述须改。
        """
        return self._dual_path_impl(question_dict, no_classify=True)

    def _runner_dual_path_no_citation(self, question_dict):
        """消融「D − 细粒度引用」：删掉要求标注 [G]/[D] 的指令，其余一字不动。"""
        return self._dual_path_impl(question_dict, no_citation=True)

    def _dual_path_impl(self, question_dict, full_context=False, honor_gate=True,
                        no_classify=False, no_citation=False):
        """_runner_dual_path 的唯一实现体 —— 主表与消融全部走这一条路径。

        full_context：图谱启用时是否把文献补满到 top-5/500（默认否，即 top-3/400）。
        honor_gate：是否启用 _graph_is_blind 门控（默认是）。False = 图谱永不关闭。
        no_classify：跳过查询分类器，全部问题按 FACTUAL 模板 + max_hops=2 处理
                     （消融「D − 自适应分类」）。
        no_citation：删掉要求标注 [G]/[D] 的指令（消融「D − 细粒度引用」）。

        ⚠️ 每个开关只动它自己那一个变量，其余一律不变 —— 否则消融差值不是
        单一变量的贡献，跟主表也没法并排比。
        """
        from services.model_router import get_model_response
        from services.query_classifier import get_retrieval_config

        t0 = time.time()

        # Step 1: 查询分类（no_classify=True 时整段跳过）
        # 注意：分类器在这条代码路径上**只影响两件事**——prompt 模板
        # （REASONING/COMPARATIVE/FACTUAL 三分支）和 max_hops。config 里的
        # vector_ratio/graph_ratio 从未被本文件任何一处读取（是历史上的死变量），
        # 故不取出，免得读代码的人以为权重真的参与了融合。
        query_type = "FACTUAL"
        max_hops = 2
        if not no_classify:
            try:
                config = get_retrieval_config(question_dict['question'])
                query_type = config.get("query_type", "FACTUAL")
                max_hops = config.get("max_hops", 2)
            except Exception as e:
                print(f"[Baseline-Dual] Query classification failed: {e}")

        if self.hops_override:      # R2#4 跳数敏感性：强制覆盖分类器给的值
            max_hops = self.hops_override

        # Step 2: 并行检索
        vector_docs = []
        vector_raw = []  # 完整 page_content，用于构建 context（与 vector_rag 对齐）
        kg_facts = ""

        # 向量检索
        try:
            docs = self._retrieve_docs(question_dict['question'], self.kb_id, top_k=5)
            if docs:
                vector_raw = docs
                vector_docs = [
                    {"id": d.metadata.get("source_file", f"doc{i}"),
                     "source_file": d.metadata.get("source_file", ""),
                     "chunk_index": d.metadata.get("chunk_index"),
                     "score": d.metadata.get("score", 0.5),
                     # 存全文而非前 200 字：片段级证据召回要在完整 chunk 里找
                     # evidence_span，截断会让真命中被判成未命中。
                     "content": d.page_content}
                    for i, d in enumerate(docs)
                ]
        except Exception as e:
            print(f"[Baseline-Dual] Vector retrieval failed: {e}")

        # 图谱检索
        try:
            from services.kg_service import retrieve_graph_context
            kg_result = retrieve_graph_context(
                question_dict['question'], self.kb_id,
                max_hops=max_hops
            )
            if isinstance(kg_result, tuple) and len(kg_result) == 2:
                kg_edges = kg_result[0] if isinstance(kg_result[0], list) else []
            elif isinstance(kg_result, list):
                kg_edges = kg_result
            else:
                kg_edges = []
            kg_facts = self._format_kg_facts(kg_edges)
        except Exception as e:
            print(f"[Baseline-Dual] KG retrieval failed: {e}")

        # 图谱覆盖门控：问题只问图谱本体缺失的属性（厚度/命名/接触/沉积环境/原岩）时，
        # 关闭图谱通道、退化为文献通道（自适应退化，而非把无关图谱事实硬套进去）。
        # honor_gate=False（消融「D − 自适应退化」）时恒为 False：图谱永不关闭，
        # 后续 prompt 与文献取法也随之走"图谱启用"那一条路，符合"只删门控"这一个改动。
        graph_blind = _graph_is_blind(question_dict['question']) if honor_gate else False
        if graph_blind:
            kg_facts = ""

        # Step 3: 融合构建Prompt
        # 向量上下文大小随「图谱是否覆盖所问属性」自适应：
        # - 图谱盲区问题（厚度/命名/沉积环境等）：图谱已关闭，答案只能来自文献 → 取全量 top-5/500（与 vector_rag 对齐）
        # - 图谱覆盖问题（岩性/年代/分布/推理）：图谱主导，文献仅辅助 → 取 top-3/400，避免无关片段稀释图谱信号
        if graph_blind or full_context:
            # 盲区问题：图谱已关，答案只能来自文献 → 给满，与 vector_rag 对齐。
            # full_context=True（对照组）：图谱虽启用，也把文献给满——这正是与
            # dual_path 唯一不同的那个变量，用来分离"图谱的增益"与"少给 40% 文献的代价"。
            v_docs, v_chars = vector_raw[:5], 500
        else:
            v_docs, v_chars = vector_raw[:3], 400
        v_context = "\n\n".join([
            f"[文献 {i+1}] {d.page_content[:v_chars]}"
            for i, d in enumerate(v_docs)
        ])

        # 查询类型自适应 Prompt 模板
        # 核心原则：图谱只对「它确实编码的属性」权威（岩性/年代/分布/化石/层位/同义名）；
        #           厚度/命名/沉积环境等图谱本体缺失的属性，必须以文献为准，避免硬套图谱。
        # 消融「D − 细粒度引用」（no_citation=True）：删掉所有要求标注 [G]/[D] 的
        # 指令，其余一字不动。COMPARATIVE/FACTUAL 两个模板本来就没有引用句，
        # 受影响的只有公共的 system_instruction 和另外两个模板的收尾句。
        # ⚠️ 换行必须跟着这一行走：no_citation=False 时拼出来的是
        # "…以文献为准。\n- 在答案中标注依据来源：…。"，与重构前**逐字相同**。
        # 若把 \n 写在本串尾部，模板里 {system_instruction} 后面本来就有一个
        # 换行，整段会多出一个空行 —— 主表 prompt 就被悄悄改掉了。
        _cite_bullet = ("" if no_citation else
                        "\n- 在答案中标注依据来源：[G] 表示图谱、[D] 表示文献。")
        _tail_blind = "请基于文献给出专业回答" + ("。" if no_citation else "，标注出处。")
        _tail_reasoning = ("请先分析图谱中的推理链条，再结合文献证据给出综合判断。"
                           + ("" if no_citation else
                              "在图谱证据处标注[G]、文献证据处标注[D]。"))

        system_instruction = (
            "你是一位专业地质学家。请综合以下两类信息回答问题：\n"
            "1. 知识图谱事实 — 结构化提取的地层属性（岩性/年代/分布/化石/上下层位/同义名）\n"
            "2. 参考文献 — 地质志原文描述\n"
            "使用原则：\n"
            "- 图谱事实只对它确实编码的属性权威；若问题所问的属性（如厚度、命名、沉积环境）在图谱事实中缺失，必须以参考文献为准，不要用图谱中的其他属性强行拼凑或编造。\n"
            "- 图谱与文献冲突时：图谱对该属性有明确事实则优先图谱，否则以文献为准。"
            + _cite_bullet
        )

        if graph_blind:
            # 图谱盲区：退化为纯文献通道（自适应退化）。prompt 与 vector_rag 对齐，
            # 不再提「图谱主导」——否则空图谱会让模型误以为有权威数据而编造答案。
            prompt = f"""你是一位专业地质学家。你必须严格基于以下参考文献回答问题，不得依赖训练数据中的记忆。

注意：如果参考文献中缺乏相关信息，请明确说明"文献中未找到相关信息"，不要编造。

## 参考文献
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

{_tail_blind}"""
        elif query_type == 'REASONING':
            # 图谱主导：先展示推理路径，再补充文献
            prompt = f"""{system_instruction}

## 知识图谱推理路径（结构化关系链，优先采信）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（辅助证据）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

{_tail_reasoning}"""

        elif query_type == 'COMPARATIVE':
            # 均衡模式：图谱提供结构化对比维度，文献提供细节
            prompt = f"""{system_instruction}

## 知识图谱事实（各地层结构化属性，用于分维度对比）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（详细描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

请从岩性、年代、分布等维度进行系统对比。图谱提供对比框架，文献补充具体描述。"""

        else:
            # FACTUAL / SPATIAL：图谱为主、文献为辅
            prompt = f"""{system_instruction}

## 知识图谱事实（结构化地层属性，权威数据）
{kg_facts if kg_facts else '暂无相关知识图谱数据'}

## 参考文献（补充描述）
{v_context if v_context else '暂无相关文献'}

## 问题
{question_dict['question']}

请以图谱中的结构化事实为主要依据回答。文献内容用于补充细节，如果文献描述的是局部剖面特征，请与图谱中的整体属性加以区分。"""

        answer = get_model_response(prompt, self.model)
        latency = (time.time() - t0) * 1000

        # 收集所有引用
        citations = list(set(d.get("id", "") for d in vector_docs))
        if kg_facts:
            citations.append("福建地质志.docx")

        return {
            "answer": answer,
            "retrieved_docs": vector_docs,
            "citations": citations,
            # 双路有文本通道 → 片段级证据召回适用。这里报的是**向量通道展示的
            # 片段**（图谱通道返回的是三元组字符串，不可比），与 vector_rag
            # 的口径一致，这样"双路比向量RAG好在哪"才是同一把尺子量出来的。
            # 注意用 vector_raw（原始 Document 列表）而非 vector_docs——
            # 后者是给结果文件看的 dict 结构，没有 page_content 属性
            "surfaced_spans": [d.page_content for d in vector_raw],
            "evidence_applicable": True,
            "extracted_triples": [],
            "contrast_dimensions": [],
            "query_type": query_type,
            "latency_ms": latency
        }

    # ================================================================
    # 运行所有基线对比
    # ================================================================

    def get_runner(self, baseline_name):
        """根据基线名称返回对应的runner函数"""
        runners = {
            "direct_llm": self._runner_direct_llm,
            "vector_rag": self._runner_vector_rag,
            "graph_rag": self._runner_graph_rag,
            "dual_path": self._runner_dual_path,
            "dual_path_fullctx": self._runner_dual_path_fullctx,
            "dual_path_nogate": self._runner_dual_path_nogate,
            "dual_path_no_classify": self._runner_dual_path_no_classify,
            "dual_path_no_citation": self._runner_dual_path_no_citation,
        }
        return runners.get(baseline_name, self._runner_direct_llm)

    def run_all_baselines(self, test_set, use_llm_judge=True):
        """
        运行所有基线对比评估。
        Returns: {
            "baselines": {name: {overall_scores, ...}},
            "comparison": {metric: {baseline: score}},
            "winner": str
        }
        """
        from .evaluator import Evaluator
        evaluator = Evaluator(model=self.model, results_dir=self.results_dir)

        all_results = {}
        comparison_table = {}

        for baseline_key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
            name = self.BASELINE_NAMES[baseline_key]
            print(f"\n{'='*60}")
            print(f"[Baseline] Running: {name}")
            print(f"{'='*60}")

            runner = self.get_runner(baseline_key)
            results = evaluator.run_full_evaluation(
                test_set, runner, use_llm_judge=use_llm_judge
            )

            all_results[baseline_key] = results

            # 构建对比表
            for metric, value in results["overall_scores"].items():
                if metric not in comparison_table:
                    comparison_table[metric] = {}
                comparison_table[metric][baseline_key] = value

        # 确定最优方法 (基于overall_score)
        overall_scores = {k: v["overall_scores"]["overall_score"]
                         for k, v in all_results.items()}
        winner = max(overall_scores, key=overall_scores.get)

        final_results = {
            "baselines": all_results,
            "comparison": comparison_table,
            "winner": winner,
            "winner_name": self.BASELINE_NAMES[winner],
            "evaluated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model": self.model
        }

        # 保存结果
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(self.results_dir, f"baseline_comparison_{timestamp}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n[Baseline] Results saved to: {result_path}")
        return final_results

    def print_comparison_summary(self, results):
        """打印对比摘要"""
        comp = results.get("comparison", {})
        key_metrics = [
            "overall_score", "qa_correctness", "qa_faithfulness",
            "retrieval_recall@5", "retrieval_mrr",
            "citation_precision", "citation_recall",
            "avg_latency_ms"
        ]

        print(f"\n{'='*80}")
        print(f"{'指标':<25}", end="")
        for key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
            print(f"{self.BASELINE_NAMES[key]:>20}", end="")
        print()

        print("-" * 105)

        for metric in key_metrics:
            if metric not in comp:
                continue
            print(f"{metric:<25}", end="")
            for key in ["direct_llm", "vector_rag", "graph_rag", "dual_path"]:
                val = comp[metric].get(key, 0)
                if "latency" in metric:
                    print(f"{val:>20.1f}", end="")
                else:
                    print(f"{val:>20.4f}", end="")
            print()

        print(f"{'='*80}")
        print(f"Best overall: {results.get('winner_name', 'Unknown')}")
        print(f"{'='*80}")
