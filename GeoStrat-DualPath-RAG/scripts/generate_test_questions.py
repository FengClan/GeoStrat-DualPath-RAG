"""
从KG数据生成200+测试题（无需LLM）
"""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx


def build_questions(G):
    formations = []; groups = []
    for n, a in G.nodes(data=True):
        t = a.get('entity_type', '')
        if t == 'FORMATION': formations.append(n)
        elif t == 'GROUP': groups.append(n)

    fm_info = {}
    for fm in formations:
        info = {'name': fm, 'rocks': [], 'times': [], 'locs': [], 'synonyms': [], 'fossils': []}
        for _, t, d in G.out_edges(fm, data=True):
            rel = d.get('relation', '')
            if rel == 'COMPOSED_OF': info['rocks'].append(t)
            elif rel == 'DATED_AS': info['times'].append(t)
            elif rel == 'DISTRIBUTED_IN': info['locs'].append(t)
            elif rel == 'SAME_AS': info['synonyms'].append(t)
            elif rel == 'CONTAINS_FOSSIL': info['fossils'].append(t)
        if info['rocks'] or info['times']:
            fm_info[fm] = info

    print(f'Formations with data: {len(fm_info)}')
    questions = []; qid = 20

    # ═══════════════════════════════════════
    # 1. FACTUAL - 岩性 (easy) — 全部有岩石数据的地层
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if len(info['rocks']) < 2: continue
        rock_list = '、'.join(info['rocks'][:5])
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'FACTUAL', 'difficulty': 'easy',
            'question': f'{fm}的主要岩性特征是什么？',
            'expected_answer': f'{fm}的岩石类型主要包括{rock_list}。',
            'key_points': [f'{fm}的岩性组成', f'主要岩石：{rock_list[:50]}'],
            'supporting_sources': ['福建地质志'], 'category': 'lithology', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'COMPOSED_OF', 'target': r} for r in info['rocks'][:3]],
            'expected_entities': [fm] + info['rocks'][:3]
        }); qid += 1

    # ═══════════════════════════════════════
    # 2. FACTUAL - 年代 (easy) — 全部有年代数据的地层
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if not info['times']: continue
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'FACTUAL', 'difficulty': 'easy',
            'question': f'{fm}形成于什么地质年代？',
            'expected_answer': f'{fm}的地质年代为{info["times"][0]}。',
            'key_points': [f'{fm}的地质年代', info['times'][0]],
            'supporting_sources': ['福建地质志'], 'category': 'geochronology', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'DATED_AS', 'target': info['times'][0]}],
            'expected_entities': [fm, info['times'][0]]
        }); qid += 1

    # ═══════════════════════════════════════
    # 3. FACTUAL - 分布 (medium) — 有>=2个分布地的地层
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if len(info['locs']) < 2: continue
        loc_list = '、'.join(info['locs'][:4])
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'FACTUAL', 'difficulty': 'medium',
            'question': f'{fm}主要分布在哪些地区？',
            'expected_answer': f'{fm}主要分布于{loc_list}等地。',
            'key_points': [f'{fm}的分布范围', f'分布区：{loc_list[:60]}'],
            'supporting_sources': ['福建地质志'], 'category': 'spatial_distribution', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'DISTRIBUTED_IN', 'target': l} for l in info['locs'][:3]],
            'expected_entities': [fm] + info['locs'][:3]
        }); qid += 1

    # ═══════════════════════════════════════
    # 4. FACTUAL - 化石 (medium)
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if not info['fossils']: continue
        fos_list = '、'.join(info['fossils'][:4])
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'FACTUAL', 'difficulty': 'medium',
            'question': f'{fm}含有哪些古生物化石？',
            'expected_answer': f'{fm}含有的化石包括{fos_list}。',
            'key_points': [f'{fm}的化石组合', fos_list],
            'supporting_sources': ['福建地质志'], 'category': 'paleontology', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'CONTAINS_FOSSIL', 'target': f} for f in info['fossils'][:3]],
            'expected_entities': [fm] + info['fossils'][:3]
        }); qid += 1

    # ═══════════════════════════════════════
    # 5. COMPARATIVE - 岩性对比 (random pairs)
    # ═══════════════════════════════════════
    rich = [(fm, i) for fm, i in fm_info.items() if len(i['rocks']) >= 2]
    random.seed(42)
    pairs = set()
    for _ in range(min(40, len(rich) * (len(rich) - 1) // 2)):
        a = random.choice(rich); b = random.choice(rich)
        if a[0] == b[0]: continue
        key = tuple(sorted([a[0], b[0]]))
        if key in pairs: continue
        pairs.add(key)
        common = set(a[1]['rocks']) & set(b[1]['rocks'])
        diff_a = set(a[1]['rocks']) - set(b[1]['rocks'])
        diff_b = set(b[1]['rocks']) - set(a[1]['rocks'])
        comp = f'{a[0]}的岩石类型包括{"、".join(a[1]["rocks"][:3])}；{b[0]}的岩石类型包括{"、".join(b[1]["rocks"][:3])}。'
        if common: comp += f'两者均含有{"、".join(list(common)[:2])}。'
        kpts = [f'{a[0]}岩性：{"、".join(a[1]["rocks"][:3])}', f'{b[0]}岩性：{"、".join(b[1]["rocks"][:3])}']
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'COMPARATIVE', 'difficulty': 'medium',
            'question': f'请对比{a[0]}与{b[0]}的岩性特征差异。',
            'expected_answer': comp, 'key_points': kpts,
            'supporting_sources': ['福建地质志'], 'category': 'stratigraphic_comparison', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': a[0], 'relation': 'COMPOSED_OF', 'target': r} for r in a[1]['rocks'][:2]] +
                              [{'entity': b[0], 'relation': 'COMPOSED_OF', 'target': r} for r in b[1]['rocks'][:2]],
            'expected_entities': [a[0], b[0]] + a[1]['rocks'][:2] + b[1]['rocks'][:2]
        }); qid += 1

    # ═══════════════════════════════════════
    # 6. COMPARATIVE - 年代对比 (hard)
    # ═══════════════════════════════════════
    timed = [(fm, i) for fm, i in fm_info.items() if i['times']]
    seen_t = set()
    for _ in range(min(20, len(timed) * (len(timed) - 1) // 2)):
        a = random.choice(timed); b = random.choice(timed)
        if a[0] == b[0]: continue
        key = tuple(sorted([a[0], b[0]]))
        if key in seen_t: continue
        seen_t.add(key)
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'COMPARATIVE', 'difficulty': 'hard',
            'question': f'{a[0]}（{a[1]["times"][0]}）与{b[0]}（{b[1]["times"][0]}）在地质年代上有何差异？',
            'expected_answer': f'{a[0]}的地质年代为{a[1]["times"][0]}，{b[0]}为{b[1]["times"][0]}。两者分属不同地质时期。',
            'key_points': [f'{a[0]}年代：{a[1]["times"][0]}', f'{b[0]}年代：{b[1]["times"][0]}'],
            'supporting_sources': ['福建地质志'], 'category': 'geochronology_comparison', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': a[0], 'relation': 'DATED_AS', 'target': a[1]['times'][0]},
                               {'entity': b[0], 'relation': 'DATED_AS', 'target': b[1]['times'][0]}],
            'expected_entities': [a[0], b[0], a[1]['times'][0], b[1]['times'][0]]
        }); qid += 1

    # ═══════════════════════════════════════
    # 7. REASONING - 同义异名 (hard)
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if not info['synonyms']: continue
        syn = info['synonyms'][0]
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'REASONING', 'difficulty': 'hard',
            'question': f'根据区域地层对比资料，{fm}与{syn}是否为同一地层单位？请说明判断依据。',
            'expected_answer': f'{fm}与{syn}为同物异名，属于同一地层单位。依据包括二者在岩性组合和地层层位上具有一致性，已在地层清理工作中被认定为同义名称。',
            'key_points': [f'{fm}与{syn}为同物异名', '岩性组合及层位一致性为判定依据'],
            'supporting_sources': ['福建地质志'], 'category': 'synonym_detection', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'SAME_AS', 'target': syn}],
            'expected_entities': [fm, syn]
        }); qid += 1

    # ═══════════════════════════════════════
    # 8. REASONING - 构造环境推断 (hard)
    # ═══════════════════════════════════════
    volcanic = [(fm, i) for fm, i in fm_info.items()
                if any(r in ''.join(i['rocks']) for r in ['玄武岩','安山岩','流纹岩','凝灰岩','火山','辉长岩','闪长岩'])]
    for fm, info in random.sample(volcanic, min(20, len(volcanic))):
        rock_str = '、'.join(info['rocks'][:3])
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'REASONING', 'difficulty': 'hard',
            'question': f'根据{fm}的岩石组合（{rock_str}），推测其形成的构造环境。',
            'expected_answer': f'{fm}的岩石组合（{rock_str}）指示其可能形成于活动大陆边缘或岛弧环境，与板块俯冲过程有关。岩浆系列特征反映了陆壳与洋壳的相互作用。',
            'key_points': [f'岩石组合：{rock_str}', '构造环境：活动大陆边缘或岛弧', '与板块俯冲有关'],
            'supporting_sources': ['福建地质志'], 'category': 'tectonic_reasoning', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'COMPOSED_OF', 'target': r} for r in info['rocks'][:3]],
            'expected_entities': [fm] + info['rocks'][:3]
        }); qid += 1

    # ═══════════════════════════════════════
    # 9. REASONING - 地层归属判断 (hard)
    # ═══════════════════════════════════════
    has_loc = [(fm, i) for fm, i in fm_info.items() if len(i['locs']) >= 2]
    for fm, info in random.sample(has_loc, min(15, len(has_loc))):
        loc_str = '、'.join(info['locs'][:3])
        time_str = info['times'][0] if info['times'] else '?'
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'REASONING', 'difficulty': 'hard',
            'question': f'在{loc_str}等地出露的一套以{info["rocks"][:2]}为主的地层，年代推测为{time_str}，请判断该地层可能属于哪个地层单位。',
            'expected_answer': f'综合分布范围（{loc_str}）、岩石组合及年代特征，该地层单位应为{fm}。其以{info["rocks"][:2]}为主要岩性、分布于上述区域的特征与{fm}的定义吻合。',
            'key_points': [f'分布范围：{loc_str}', f'岩石组合：{info["rocks"][:2]}', f'年代：{time_str}', f'结论：{fm}'],
            'supporting_sources': ['福建地质志'], 'category': 'stratigraphic_identification', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'DISTRIBUTED_IN', 'target': l} for l in info['locs'][:2]] +
                              [{'entity': fm, 'relation': 'COMPOSED_OF', 'target': r} for r in info['rocks'][:2]],
            'expected_entities': [fm] + info['locs'][:2] + info['rocks'][:2]
        }); qid += 1

    # ═══════════════════════════════════════
    # 10. SPATIAL - 空间分布特征
    # ═══════════════════════════════════════
    for fm, info in fm_info.items():
        if len(info['locs']) < 2: continue
        locs_str = '、'.join(info['locs'][:4])
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'SPATIAL', 'difficulty': 'medium',
            'question': f'{fm}在福建省的空间展布有何特征？',
            'expected_answer': f'{fm}在福建省内主要分布于{locs_str}等地，受区域构造控制，呈带状或面状展布。',
            'key_points': [f'分布区域：{locs_str}', '受区域构造控制'],
            'supporting_sources': ['福建地质志'], 'category': 'spatial_distribution', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'DISTRIBUTED_IN', 'target': l} for l in info['locs'][:3]],
            'expected_entities': [fm] + info['locs'][:3]
        }); qid += 1

    # ═══════════════════════════════════════
    # 11. SPATIAL - 厚度变化 (hard)
    # ═══════════════════════════════════════
    for fm, info in random.sample(list(fm_info.items()), min(15, len(fm_info))):
        loc_str = info['locs'][0] if info['locs'] else '福建省'
        questions.append({
            'id': f'FJ-{qid:03d}', 'query_type': 'SPATIAL', 'difficulty': 'hard',
            'question': f'{fm}在{loc_str}一带的厚度变化规律是什么？',
            'expected_answer': f'{fm}在{loc_str}一带的厚度受沉积环境和后期构造改造的双重影响。区域上，该单位通常呈现由沉积中心向边缘减薄或受断层切割导致厚度突变的特征。',
            'key_points': [f'{fm}厚度受沉积环境控制', '后期构造改造影响厚度分布', '由中心向边缘减薄或断裂切割突变'],
            'supporting_sources': ['福建地质志'], 'category': 'thickness_analysis', 'kb_id': 'experiment',
            'expected_kg_path': [{'entity': fm, 'relation': 'DISTRIBUTED_IN', 'target': loc_str}],
            'expected_entities': [fm, loc_str]
        }); qid += 1

    return questions


def main():
    G = nx.read_graphml(os.path.join(os.path.dirname(__file__), '..', 'kb_storage', 'graphs', 'experiment_kb.graphml'))
    print(f'KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    random.seed(42)
    new_qs = build_questions(G)
    print(f'\nGenerated {len(new_qs)} new questions')

    # 加载已有12题
    existing_path = os.path.join(os.path.dirname(__file__), '..', 'test_sets', 'test_set_20260512_150301.json')
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)

    all_qs = existing['questions'] + new_qs
    for i, q in enumerate(all_qs):
        q['id'] = f'FJ-{i+1:03d}'
        q.setdefault('kb_id', 'experiment')
        q.setdefault('supporting_sources', ['福建地质志'])
        q.setdefault('key_points', [])
        q.setdefault('expected_kg_path', [])
        q.setdefault('expected_entities', [])

    from collections import Counter
    test_set = {
        'version': '2.0', 'created_at': '2026-06-01',
        'description': f'岩石地层问答测试集 ({len(all_qs)}题)',
        'questions': all_qs
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'test_sets', 'test_set_expanded.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f'Total: {len(all_qs)} questions')
    bt = Counter(q['query_type'] for q in all_qs)
    bd = Counter(q['difficulty'] for q in all_qs)
    print(f'By type: {dict(bt)}')
    print(f'By difficulty: {dict(bd)}')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
