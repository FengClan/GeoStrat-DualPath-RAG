"""
扩展测试集：从12题→35-40题
基于KG数据和福建地质志，生成覆盖4种查询类型的多样化测试题
"""
import json, os, sys, re, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_router import get_model_response

# ═══════════════════════════════════════════════════════════
# Step 1: 从KG中获取可用的地层名和关系
# ═══════════════════════════════════════════════════════════

def load_kg_info(graphml_path):
    """加载KG信息，返回地层名列表和关系统计"""
    import networkx as nx
    G = nx.read_graphml(graphml_path)

    formations = []
    groups = []
    members = []
    for node, attrs in G.nodes(data=True):
        etype = attrs.get('entity_type', '')
        if etype == 'FORMATION':
            formations.append(node)
        elif etype == 'GROUP':
            groups.append(node)
        elif etype == 'MEMBER':
            members.append(node)

    # 统计有COMPOSED_OF关系的地层
    formations_with_rocks = []
    for fm in formations:
        rocks = [t for h, t, d in G.out_edges(fm, data=True) if d.get('relation') == 'COMPOSED_OF']
        if rocks:
            formations_with_rocks.append((fm, rocks[:5]))

    # 有DATED_AS的地层
    formations_with_time = []
    for fm in formations:
        times = [t for h, t, d in G.out_edges(fm, data=True) if d.get('relation') == 'DATED_AS']
        if times:
            formations_with_time.append((fm, times[0]))

    # 有SAME_AS的
    formations_with_synonym = []
    for fm in formations:
        syns = [t for h, t, d in G.out_edges(fm, data=True) if d.get('relation') == 'SAME_AS']
        if syns:
            formations_with_synonym.append((fm, syns))

    return {
        'formations': formations[:80],
        'groups': groups[:20],
        'formations_with_rocks': formations_with_rocks[:40],
        'formations_with_time': formations_with_time[:40],
        'formations_with_synonym': formations_with_synonym[:10],
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges()
    }


# ═══════════════════════════════════════════════════════════
# Step 2: LLM生成题目
# ═══════════════════════════════════════════════════════════

QUESTION_GEN_PROMPT = """你是一位地质学教育和评估专家。请为岩石地层学问答系统生成测试题。

## 可用地层信息
{kg_summary}

## 查询类型: {query_type} ({type_label})
## 难度: {difficulty}
## 需要生成: {count}题

## 题目要求
- 每题必须包含：id (格式: FJ-0XX), question (完整中文问题), expected_answer (100-300字标准答案), key_points (3-5个关键知识点), supporting_sources (引用来源列表), expected_kg_path (预期KG推理路径的triple列表), expected_entities (预期涉及的实体名称)
- 查询类型描述:
  - FACTUAL: 地层基本属性查询（岩性/年代/厚度/命名/分布等单一事实）
  - COMPARATIVE: 两个（或多个）地层之间的对比分析
  - REASONING: 需要多步推理的地层问题（推断构造环境、判断地层归属、同义异名判断等）
  - SPATIAL: 关注地层空间分布、走向、厚度变化等地理空间问题
- 难度: easy(单实体直接查询), medium(涉及多实体或简单推理), hard(多步推理或跨区域对比)
- 参照《福建省区域地质志》的知识体系，使用专业地质术语

## 输出格式（严格JSON数组）
```json
[{{
  "id": "FJ-0XX",
  "question": "...",
  "expected_answer": "...",
  "query_type": "{query_type}",
  "difficulty": "{difficulty}",
  "key_points": ["...", "..."],
  "supporting_sources": ["福建地质志"],
  "expected_kg_path": [{{"entity": "...", "relation": "...", "target": "..."}}],
  "expected_entities": ["...", "..."],
  "category": "...",
  "kb_id": "experiment"
}}]
```
仅返回JSON数组。"""


def generate_questions(kg_info, query_type, difficulty, count=2, model="qwen3.5-plus"):
    """使用LLM生成指定类型和难度的题目"""
    # 构建KG摘要
    sample_formations = random.sample(kg_info['formations'], min(15, len(kg_info['formations'])))
    sample_with_rocks = random.sample(kg_info['formations_with_rocks'], min(8, len(kg_info['formations_with_rocks'])))
    sample_with_time = random.sample(kg_info['formations_with_time'], min(5, len(kg_info['formations_with_time'])))

    kg_summary = f"""
可用地层名称（部分）: {', '.join(sample_formations[:15])}
群级单位: {', '.join(kg_info['groups'][:10])}
含岩性信息的地层: {', '.join(f'{f}({",".join(r[:3])})' for f, r in sample_with_rocks)}
含年代信息的地层: {', '.join(f'{f}({t})' for f, t in sample_with_time)}
"""

    type_labels = {
        'FACTUAL': '事实查询型',
        'COMPARATIVE': '对比分析型',
        'REASONING': '推理判断型',
        'SPATIAL': '空间分布型'
    }

    prompt = QUESTION_GEN_PROMPT.format(
        kg_summary=kg_summary,
        query_type=query_type,
        type_label=type_labels[query_type],
        difficulty=difficulty,
        count=count
    )

    try:
        raw = get_model_response(prompt, model)
        cleaned = raw.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        questions = json.loads(cleaned)
        if isinstance(questions, dict):
            questions = [questions]
        return questions
    except Exception as e:
        print(f"  LLM生成失败: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# Step 3: 去重与验证
# ═══════════════════════════════════════════════════════════

def validate_question(q):
    """验证题目必要字段是否完整"""
    required = ['id', 'question', 'expected_answer', 'query_type', 'difficulty']
    for r in required:
        if r not in q or not q[r]:
            return False, f"Missing {r}"
    if len(q.get('key_points', [])) < 2:
        return False, "Too few key_points"
    if len(q.get('question', '')) < 10:
        return False, "Question too short"
    if len(q.get('expected_answer', '')) < 30:
        return False, "Answer too short"
    return True, "OK"


def deduplicate(questions, existing_ids):
    """去重：移除id重复和question高度相似的题目"""
    from difflib import SequenceMatcher
    seen_ids = set(existing_ids)
    unique = []
    for q in questions:
        if q['id'] in seen_ids:
            continue
        # 检查与已有题目的相似度
        is_dup = False
        for uq in unique:
            sim = SequenceMatcher(None, q['question'], uq['question']).ratio()
            if sim > 0.7:
                is_dup = True
                break
        if not is_dup:
            seen_ids.add(q['id'])
            unique.append(q)
    return unique


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    # 加载KG
    graphml_path = os.path.join(os.path.dirname(__file__), '..', 'kb_storage', 'graphs', 'experiment_kb.graphml')
    if not os.path.exists(graphml_path):
        print("KG not found! Run convert_excel_to_kg.py first.")
        return

    kg_info = load_kg_info(graphml_path)
    print(f"KG loaded: {kg_info['total_nodes']} nodes, {kg_info['total_edges']} edges")
    print(f"  Formations: {len(kg_info['formations'])}, Groups: {len(kg_info['groups'])}")

    # 加载已有测试题
    existing_path = os.path.join(os.path.dirname(__file__), '..', 'test_sets', 'test_set_20260512_150301.json')
    existing_ids = []
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing_ids = [q['id'] for q in existing.get('questions', [])]
        print(f"Existing test questions: {len(existing_ids)} ({existing_ids})")

    # 生成计划：每类型每难度2-3题
    plan = [
        ('FACTUAL', 'easy', 3),
        ('FACTUAL', 'medium', 3),
        ('FACTUAL', 'hard', 2),
        ('COMPARATIVE', 'medium', 3),
        ('COMPARATIVE', 'hard', 2),
        ('REASONING', 'medium', 2),
        ('REASONING', 'hard', 3),
        ('SPATIAL', 'easy', 2),
        ('SPATIAL', 'medium', 2),
        ('SPATIAL', 'hard', 2),
    ]

    all_new = []
    for qtype, diff, count in plan:
        print(f"\nGenerating {count} {qtype}/{diff}...")
        questions = generate_questions(kg_info, qtype, diff, count, model="qwen3.5-plus")
        valid_count = 0
        for q in questions:
            ok, msg = validate_question(q)
            if ok:
                all_new.append(q)
                valid_count += 1
            else:
                print(f"  Invalid: {q.get('id','?')} - {msg}")
        print(f"  Generated: {len(questions)}, Valid: {valid_count}")
        time.sleep(1.5)

    # 去重
    unique = deduplicate(all_new, existing_ids)
    print(f"\nAfter dedup: {len(unique)} unique questions (from {len(all_new)} generated)")

    # 合并并保存
    if os.path.exists(existing_path):
        merged = existing['questions'] + unique
    else:
        merged = unique

    # 重新分配ID
    for i, q in enumerate(merged):
        if not q.get('id') or q['id'] in [f'FJ-{j:03d}' for j in range(100)]:
            q['id'] = f'FJ-{i+1:03d}'

    # 添加必要字段默认值
    for q in merged:
        q.setdefault('kb_id', 'experiment')
        q.setdefault('category', q.get('query_type', 'FACTUAL').lower())
        q.setdefault('supporting_sources', ['福建地质志'])
        q.setdefault('difficulty', 'medium')

    test_set = {
        'version': '2.0',
        'created_at': time.strftime('%Y-%m-%d'),
        'description': f'扩展岩石地层问答测试集 ({len(merged)}题)',
        'questions': merged
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'test_sets', 'test_set_expanded.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    type_dist = Counter(q['query_type'] for q in merged)
    diff_dist = Counter(q['difficulty'] for q in merged)
    print(f'\n=== Test Set Saved ===')
    print(f'Total: {len(merged)} questions')
    print(f'By type: {dict(type_dist)}')
    print(f'By difficulty: {dict(diff_dist)}')
    print(f'Saved to: {out_path}')

if __name__ == '__main__':
    main()
