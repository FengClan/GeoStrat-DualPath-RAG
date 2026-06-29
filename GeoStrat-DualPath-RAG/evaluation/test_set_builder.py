# evaluation/test_set_builder.py — 测试集构建器：解析地质志生成标注QA对
import os
import json
import re
import hashlib
from datetime import datetime


class TestSetBuilder:
    """
    构建地质领域标注测试集。
    支持：内置样本 + 从文本语料自动生成 + LLM辅助生成。
    """

    # 福建/浙江地层领域的内置高质量测试样本
    BUILTIN_SAMPLES = [
        # ========== FACTUAL 类型 ==========
        {
            "id": "FJ-001",
            "question": "南园组的岩性特征是什么？",
            "query_type": "FACTUAL",
            "expected_answer": "南园组主要岩性为流纹岩、凝灰岩、英安质凝灰熔岩，夹有少量砂岩和页岩。颜色以紫红色、灰紫色为主。属于晚侏罗世火山喷发产物。",
            "key_points": ["流纹岩", "凝灰岩", "英安质凝灰熔岩", "晚侏罗世", "紫红色"],
            "supporting_sources": ["福建省区域地质志", "福建省岩石地层"],
            "expected_entities": ["南园组", "流纹岩", "凝灰岩", "英安质凝灰熔岩"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "流纹岩"},
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "凝灰岩"},
                {"entity": "南园组", "relation": "DATED_AS", "target": "晚侏罗世"}
            ],
            "difficulty": "easy",
            "category": "lithology",
            "kb_id": "all"
        },
        {
            "id": "FJ-002",
            "question": "长林组的地质年代是什么？",
            "query_type": "FACTUAL",
            "expected_answer": "长林组属于晚侏罗世，是闽东地区晚中生代火山岩系的重要组成部分。",
            "key_points": ["晚侏罗世", "晚中生代"],
            "supporting_sources": ["福建省区域地质志"],
            "expected_entities": ["长林组", "晚侏罗世"],
            "expected_kg_path": [
                {"entity": "长林组", "relation": "DATED_AS", "target": "晚侏罗世"}
            ],
            "difficulty": "easy",
            "category": "geochronology",
            "kb_id": "all"
        },
        {
            "id": "FJ-003",
            "question": "福建省出露的最古老地层是什么？",
            "query_type": "FACTUAL",
            "expected_answer": "福建省出露的最古老地层为古元古代的麻源群，主要分布在闽北武夷山地区，由片麻岩、片岩、变粒岩等中深变质岩组成。",
            "key_points": ["麻源群", "古元古代", "闽北武夷山", "片麻岩", "片岩", "变粒岩"],
            "supporting_sources": ["福建省区域地质志"],
            "expected_entities": ["麻源群", "古元古代"],
            "expected_kg_path": [
                {"entity": "麻源群", "relation": "DATED_AS", "target": "古元古代"},
                {"entity": "麻源群", "relation": "COMPOSED_OF", "target": "片麻岩"}
            ],
            "difficulty": "medium",
            "category": "stratigraphy",
            "kb_id": "all"
        },

        # ========== COMPARATIVE 类型 ==========
        {
            "id": "FJ-004",
            "question": "请对比南园组和长林组的岩性差异。",
            "query_type": "COMPARATIVE",
            "expected_answer": "南园组以中酸性火山岩为主，包括流纹岩、凝灰岩、英安质凝灰熔岩，颜色偏紫红色；长林组以陆相碎屑沉积为主，包括砂岩、粉砂岩、页岩，夹火山碎屑岩。两者反映了不同的沉积环境：南园组为大规模火山喷发产物，长林组为火山活动间歇期的正常沉积。",
            "key_points": ["南园组中酸性火山岩", "长林组陆相碎屑沉积", "沉积环境差异", "火山喷发与间歇期"],
            "supporting_sources": ["福建省区域地质志", "福建省岩石地层"],
            "expected_entities": ["南园组", "长林组", "流纹岩", "砂岩"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "流纹岩"},
                {"entity": "长林组", "relation": "COMPOSED_OF", "target": "砂岩"}
            ],
            "difficulty": "medium",
            "category": "stratigraphic_comparison",
            "kb_id": "all"
        },
        {
            "id": "FJ-005",
            "question": "闽西南和闽东地区的晚中生代地层有何异同？",
            "query_type": "COMPARATIVE",
            "expected_answer": "闽西南地区晚中生代以陆相红色碎屑岩建造为主（如沙县组、赤水组），火山岩不发育；闽东地区晚中生代则以大规模的火山岩建造为主（如南园组、小溪组），火山活动强烈。两区差异反映了东南沿海火山岩带与内陆盆地的不同构造环境。",
            "key_points": ["闽西南红色碎屑岩", "闽东火山岩建造", "构造环境差异", "沙县组", "南园组"],
            "supporting_sources": ["福建省区域地质志", "中国区域地质概论"],
            "expected_entities": ["沙县组", "赤水组", "南园组", "小溪组"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "DISTRIBUTED_IN", "target": "闽东"},
                {"entity": "沙县组", "relation": "DISTRIBUTED_IN", "target": "闽西南"}
            ],
            "difficulty": "hard",
            "category": "cross_region_comparison",
            "kb_id": "all"
        },

        # ========== REASONING 类型 ==========
        {
            "id": "FJ-006",
            "question": "根据南园组的岩性组合特征，推断其形成的构造环境。",
            "query_type": "REASONING",
            "expected_answer": "南园组以大规模中酸性火山岩（流纹岩、英安岩、凝灰岩）为主，属于钙碱性系列，结合其形成于晚侏罗世，反映其形成于活动大陆边缘的火山弧环境，与古太平洋板块向欧亚板块的俯冲作用有关。该构造环境与浙闽沿海中生代火山岩带整体背景一致。",
            "key_points": ["钙碱性系列", "活动大陆边缘", "火山弧", "板块俯冲", "晚侏罗世"],
            "supporting_sources": ["福建省区域地质志", "中国区域地质概论"],
            "expected_entities": ["南园组", "古太平洋板块", "欧亚板块"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "流纹岩"},
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "英安岩"},
                {"entity": "南园组", "relation": "DATED_AS", "target": "晚侏罗世"}
            ],
            "difficulty": "hard",
            "category": "tectonic_reasoning",
            "kb_id": "all"
        },
        {
            "id": "FJ-007",
            "question": "如果在闽东某钻孔中发现厚层流纹岩和凝灰岩互层，判断其可能属于哪个地层单位？请给出推理过程。",
            "query_type": "REASONING",
            "expected_answer": "最可能属于南园组。推理过程：1) 流纹岩+凝灰岩互层是南园组的标志性岩性组合；2) 闽东地区晚中生代火山岩以南园组分布最广；3) 厚层特征说明是火山喷发主体而非边缘相；4) 需排除小溪组（层位更高、厚度较薄）。建议结合化石或同位素年龄进一步确认。",
            "key_points": ["南园组岩性标志", "闽东地区分布", "排除小溪组", "层位判断"],
            "supporting_sources": ["福建省岩石地层"],
            "expected_entities": ["南园组", "流纹岩", "凝灰岩", "小溪组"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "流纹岩"},
                {"entity": "南园组", "relation": "COMPOSED_OF", "target": "凝灰岩"},
                {"entity": "南园组", "relation": "DISTRIBUTED_IN", "target": "闽东"}
            ],
            "difficulty": "hard",
            "category": "stratigraphic_identification",
            "kb_id": "all"
        },

        # ========== SPATIAL 类型 ==========
        {
            "id": "FJ-008",
            "question": "南园组在福建省的空间分布情况如何？",
            "query_type": "SPATIAL",
            "expected_answer": "南园组广泛分布于闽东地区，从北部的福鼎、福安到南部的漳州、平和均有出露。主要分布区域包括：周宁梨坪（厚度约1552m）、福安甲峰（厚度约302m）、寿宁武曲（厚度约224m）。整体呈NE-NNE向带状展布，受区域性断裂构造控制。",
            "key_points": ["闽东广泛分布", "福鼎到漳州", "周宁梨坪1552m", "NE-NNE向带状"],
            "supporting_sources": ["福建省区域地质志", "福建省地质图说明书"],
            "expected_entities": ["南园组", "周宁梨坪", "福安甲峰", "寿宁武曲"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "DISTRIBUTED_IN", "target": "周宁梨坪"},
                {"entity": "南园组", "relation": "DISTRIBUTED_IN", "target": "福安甲峰"},
                {"entity": "南园组", "relation": "HAS_THICKNESS", "target": "1552m"}
            ],
            "difficulty": "medium",
            "category": "spatial_distribution",
            "kb_id": "all"
        },
        {
            "id": "FJ-009",
            "question": "福建省地层的厚度变化有何规律？",
            "query_type": "SPATIAL",
            "expected_answer": "福建省地层厚度变化总体呈现以下规律：1) 闽东火山岩系厚度巨大（南园组可达1000-3000m），由沿海向内陆减薄；2) 闽西南沉积岩系厚度较稳定（一般数百米至千米）；3) 闽北变质岩基底厚度变化大，受后期构造改造明显。厚度变化主要受古地理格局和同沉积构造控制。",
            "key_points": ["闽东厚度大1000-3000m", "沿海向内陆减薄", "闽西南较稳定", "受古地理和构造控制"],
            "supporting_sources": ["福建省区域地质志"],
            "expected_entities": ["南园组", "闽东", "闽西南", "闽北"],
            "expected_kg_path": [
                {"entity": "南园组", "relation": "HAS_THICKNESS", "target": "1000-3000m"},
                {"entity": "南园组", "relation": "DISTRIBUTED_IN", "target": "闽东"}
            ],
            "difficulty": "medium",
            "category": "thickness_analysis",
            "kb_id": "all"
        },

        # ========== 跨区域对比 ==========
        {
            "id": "ZJ-001",
            "question": "浙江的磨石山群与福建的南园组是否可以对比？",
            "query_type": "COMPARATIVE",
            "expected_answer": "浙江磨石山群与福建南园组均为晚侏罗世火山岩系，岩性均以流纹岩、凝灰岩为主，层位大致相当，可以进行区域对比。但磨石山群内部包含更多沉积岩夹层（如茶湾组、九里坪组），而南园组以火山岩为主。两者在火山喷发强度和沉积夹层比例上存在差异，反映了火山活动在不同区域的差异性。",
            "key_points": ["同为晚侏罗世", "火山岩系可对比", "磨石山群沉积夹层更多", "反映火山活动区域差异"],
            "supporting_sources": ["浙江省区域地质志", "福建省区域地质志", "中国东南部中生代火山地质"],
            "expected_entities": ["磨石山群", "南园组", "茶湾组", "九里坪组"],
            "expected_kg_path": [
                {"entity": "磨石山群", "relation": "DATED_AS", "target": "晚侏罗世"},
                {"entity": "南园组", "relation": "DATED_AS", "target": "晚侏罗世"},
                {"entity": "磨石山群", "relation": "COMPOSED_OF", "target": "流纹岩"}
            ],
            "difficulty": "hard",
            "category": "cross_province_comparison",
            "kb_id": "all"
        },

        # ========== 简单题补充 ==========
        {
            "id": "FJ-010",
            "question": "什么是岩石地层单位的【组】？",
            "query_type": "FACTUAL",
            "expected_answer": "组（Formation）是岩石地层划分的基本单位。组具有岩性、岩相和变质程度的一致性，可以由单一岩石类型构成，也可以由多种岩石类型的组合构成。组的厚度从数米到数千米不等，是进行区域地质填图的基本单位。",
            "key_points": ["基本岩石地层单位", "岩性岩相一致", "可单一或多种岩石", "区域填图基本单位"],
            "supporting_sources": ["中国地层指南"],
            "expected_entities": ["组"],
            "expected_kg_path": [],
            "difficulty": "easy",
            "category": "stratigraphic_knowledge",
            "kb_id": "all"
        },
        {
            "id": "FJ-011",
            "question": "坂头组的地质时代和主要岩性是什么？",
            "query_type": "FACTUAL",
            "expected_answer": "坂头组属于早白垩世，主要岩性为紫红色砂砾岩、砂岩、粉砂岩夹凝灰岩，属于陆相盆地沉积，含有叶肢介、介形虫等化石。",
            "key_points": ["早白垩世", "紫红色砂砾岩", "陆相盆地沉积", "含化石"],
            "supporting_sources": ["福建省区域地质志"],
            "expected_entities": ["坂头组", "早白垩世"],
            "expected_kg_path": [
                {"entity": "坂头组", "relation": "DATED_AS", "target": "早白垩世"}
            ],
            "difficulty": "medium",
            "category": "lithology",
            "kb_id": "all"
        },

        # ========== 同义异名 ==========
        {
            "id": "FJ-012",
            "question": "闽西南地区的【梨山组】与闽北的【焦坑组】是否为同一地层？",
            "query_type": "REASONING",
            "expected_answer": "梨山组和焦坑组不是同一地层，但需要进行仔细的区域对比。两者均属于早侏罗世陆相含煤沉积，但梨山组分布在闽西南（永安-龙岩一带），焦坑组分布在闽北（邵武-建阳一带）。两者在含煤性和岩性组合上有相似之处，但由于分布在不同的沉积盆地中，目前作为不同的岩石地层单位处理。是否存在同义异名关系需要进一步的化石对比和盆地分析来确定。",
            "key_points": ["不同沉积盆地", "均为早侏罗世含煤沉积", "目前作为不同单位", "需化石对比确认"],
            "supporting_sources": ["福建省岩石地层", "福建省区域地质志"],
            "expected_entities": ["梨山组", "焦坑组", "早侏罗世"],
            "expected_kg_path": [
                {"entity": "梨山组", "relation": "DATED_AS", "target": "早侏罗世"},
                {"entity": "焦坑组", "relation": "DATED_AS", "target": "早侏罗世"}
            ],
            "difficulty": "hard",
            "category": "synonym_detection",
            "kb_id": "all"
        }
    ]

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'test_sets')
        os.makedirs(self.output_dir, exist_ok=True)

    def build_from_builtin(self, test_set_name="fujian_stratigraphy", description=None):
        """从内置样本构建测试集"""
        test_set = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "description": description or "福建/浙江岩石地层领域标注测试集（内置样本）",
            "questions": self.BUILTIN_SAMPLES
        }
        return test_set

    def build_from_documents(self, documents, test_set_name, use_llm=False, model=None):
        """
        从地质文档自动生成测试QA对。

        Args:
            documents: list of {title, content} dicts
            test_set_name: 测试集名称
            use_llm: 是否使用LLM辅助生成
            model: LLM模型名称
        """
        questions = []

        if use_llm and documents:
            questions = self._auto_generate_questions(documents, model)

        # 合并内置样本作为基础
        test_set = self.build_from_builtin()
        if questions:
            test_set["questions"].extend(questions)
            test_set["description"] += f" + {len(questions)}个自动生成问题"

        return test_set

    def _auto_generate_questions(self, documents, model=None):
        """使用LLM从文档中自动生成问答对"""
        from services.model_router import get_model_response

        questions = []
        for doc in documents[:3]:  # 限制处理前3篇文档
            content = doc.get('content', '')[:3000]
            title = doc.get('title', '地质文献')

            prompt = f"""你是一位地质学测试题设计专家。请从以下地质文献中提取或生成3-5个高质量的问答对，用于评估地质知识问答系统。

文献标题: {title}
文献内容:
{content}

要求:
1. 覆盖FACTUAL（事实型）、COMPARATIVE（对比型）、REASONING（推理型）、SPATIAL（空间型）四种类型
2. 每个问题包含清晰的预期答案
3. 标注难度 (easy/medium/hard)
4. 返回JSON数组

只返回JSON数组，格式如下:
[{{"question": "...", "query_type": "FACTUAL", "expected_answer": "...", "key_points": [...], "difficulty": "medium", "category": "lithology"}}]"""

            try:
                raw = get_model_response(prompt, model or "qwen3.5-plus")
                cleaned = raw.strip()
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                if cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                generated = json.loads(cleaned)

                for i, q in enumerate(generated):
                    q["id"] = f"AUTO-{hashlib.md5((title + str(i)).encode()).hexdigest()[:8]}"
                    q["supporting_sources"] = [title]
                    q["expected_kg_path"] = []
                    q["expected_entities"] = []
                    q["kb_id"] = "all"
                    questions.append(q)
            except Exception as e:
                print(f"[TestSetBuilder] LLM generation failed for {title}: {e}")

        return questions

    def load_test_set(self, test_set_path):
        """加载已有测试集"""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_test_set(self, test_set, filename=None):
        """保存测试集为JSON文件"""
        if filename is None:
            filename = f"test_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        print(f"[TestSetBuilder] Test set saved: {filepath} ({len(test_set['questions'])} questions)")
        return filepath

    def get_statistics(self, test_set):
        """获取测试集统计信息"""
        stats = {
            "total_questions": len(test_set["questions"]),
            "by_type": {},
            "by_difficulty": {},
            "by_category": {}
        }
        for q in test_set["questions"]:
            stats["by_type"][q.get("query_type", "UNKNOWN")] = \
                stats["by_type"].get(q.get("query_type", "UNKNOWN"), 0) + 1
            stats["by_difficulty"][q.get("difficulty", "medium")] = \
                stats["by_difficulty"].get(q.get("difficulty", "medium"), 0) + 1
            stats["by_category"][q.get("category", "general")] = \
                stats["by_category"].get(q.get("category", "general"), 0) + 1
        return stats


if __name__ == '__main__':
    builder = TestSetBuilder()
    test_set = builder.build_from_builtin()
    path = builder.save_test_set(test_set)
    stats = builder.get_statistics(test_set)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
