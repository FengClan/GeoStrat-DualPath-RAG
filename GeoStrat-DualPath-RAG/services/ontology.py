# services/ontology.py — 岩石地层领域本体定义与校验
"""地质领域本体：实体类型、关系类型及其约束"""

# ============================================================
# 实体类型
# ============================================================
ENTITY_TYPES = {
    "GROUP": {
        "label": "群",
        "description": "岩石地层最高等级单位，由多个组组成",
        "examples": ["龙山群", "板溪群", "五台群"],
        "pattern": r".+群$"
    },
    "FORMATION": {
        "label": "组",
        "description": "岩石地层基本单位，具有岩性、岩相和变质程度的一致性",
        "examples": ["栖霞组", "茅口组", "龙潭组"],
        "pattern": r".+组$"
    },
    "MEMBER": {
        "label": "段",
        "description": "组的细分单位",
        "examples": ["瘤状灰岩段", "硅质岩段"],
        "pattern": r".+段$"
    },
    "LITHOLOGY": {
        "label": "岩石类型",
        "description": "岩石的类型描述",
        "examples": ["灰岩", "砂岩", "页岩", "白云岩", "花岗岩"],
        "pattern": None
    },
    "GEOLOGICAL_TIME": {
        "label": "地质年代",
        "description": "地层的地质年代归属",
        "examples": ["二叠纪", "寒武纪", "奥陶纪", "志留纪", "中生代"],
        "pattern": None
    },
    "LOCATION": {
        "label": "地理位置",
        "description": "地层出露或分布的地理位置",
        "examples": ["福建龙岩", "浙江江山", "武夷山脉"],
        "pattern": None
    },
    "SECTION": {
        "label": "剖面",
        "description": "地层测量剖面，包含多个层位",
        "examples": ["龙岩剖面", "江山剖面", "黄泥塘剖面"],
        "pattern": r".+剖面$"
    },
    "FOSSIL": {
        "label": "化石",
        "description": "地层中含有的古生物化石",
        "examples": ["蜓类", "珊瑚", "腕足类", "菊石"],
        "pattern": None
    },
    "THICKNESS": {
        "label": "厚度",
        "description": "地层厚度数值",
        "examples": [">100m", "50-80米"],
        "pattern": None
    },
    "CONTACT_RELATIONSHIP": {
        "label": "接触关系",
        "description": "地层之间的接触关系描述",
        "examples": ["整合接触", "平行不整合", "角度不整合", "断层接触"],
        "pattern": None
    }
}

# ============================================================
# 关系类型及其 domain/range 约束
# ============================================================
RELATION_TYPES = {
    "BELONGS_TO": {
        "label": "属于",
        "description": "下级单位属于上级单位",
        "domain": ["FORMATION", "MEMBER", "SECTION"],  # subject must be one of these
        "range": ["GROUP", "FORMATION"],               # object must be one of these
        "inverse": "HAS_SUBUNIT"
    },
    "HAS_SUBUNIT": {
        "label": "包含",
        "description": "上级单位包含下级单位",
        "domain": ["GROUP", "FORMATION"],
        "range": ["FORMATION", "MEMBER"],
        "inverse": "BELONGS_TO"
    },
    "COMPOSED_OF": {
        "label": "岩性组成",
        "description": "地层单位由何种岩石组成",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["LITHOLOGY"]
    },
    "DATED_AS": {
        "label": "定年为",
        "description": "地层单位的地质年代归属",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["GEOLOGICAL_TIME"]
    },
    "DISTRIBUTED_IN": {
        "label": "分布于",
        "description": "地层单位的地理分布",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["LOCATION"]
    },
    "CONTACTS": {
        "label": "接触关系",
        "description": "地层单位之间的接触关系",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["GROUP", "FORMATION", "MEMBER", "CONTACT_RELATIONSHIP"]
    },
    "HAS_THICKNESS": {
        "label": "厚度",
        "description": "地层单位的厚度值",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["THICKNESS"]
    },
    "CONTAINS_FOSSIL": {
        "label": "含化石",
        "description": "地层单位含有的化石类型",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["FOSSIL"]
    },
    "MEASURED_AT": {
        "label": "测量于",
        "description": "地层单位在哪个剖面被测量",
        "domain": ["FORMATION", "MEMBER"],
        "range": ["SECTION"]
    },
    "SAME_AS": {
        "label": "同义",
        "description": "同一地层单位在不同区域的异名",
        "domain": ["GROUP", "FORMATION", "MEMBER"],
        "range": ["GROUP", "FORMATION", "MEMBER"]
    }
}

# ============================================================
# 地层等级顺序 (用于层级一致性检查)
# ============================================================
HIERARCHY_ORDER = {"GROUP": 1, "FORMATION": 2, "MEMBER": 3}

# ============================================================
# 验证函数
# ============================================================


def get_hierarchy_level(entity_type):
    """返回实体类型的地层等级"""
    return HIERARCHY_ORDER.get(entity_type, None)


def validate_entity(entity_name, entity_type):
    """验证单个实体：检查类型是否存在及命名模式"""
    if entity_type not in ENTITY_TYPES:
        return False, f"未知实体类型: {entity_type}，已知类型: {list(ENTITY_TYPES.keys())}"

    if not entity_name or not entity_name.strip():
        return False, f"{entity_type} 实体名称不能为空"

    # 可选：检查命名模式
    pattern = ENTITY_TYPES[entity_type].get("pattern")
    if pattern:
        import re
        if not re.match(pattern, entity_name.strip()):
            return False, f"{entity_type} 实体 '{entity_name}' 不匹配命名模式 {pattern}"

    return True, None


def validate_triple(entity1, entity1_type, relation, entity2, entity2_type):
    """
    验证三元组是否符合本体约束。
    返回 (valid: bool, error: str|None, warnings: list[str])
    """
    warnings = []
    errors = []

    # 1. 验证实体
    for ent, etype in [(entity1, entity1_type), (entity2, entity2_type)]:
        valid, err = validate_entity(ent, etype)
        if not valid:
            errors.append(err)

    # 2. 验证关系类型
    if relation not in RELATION_TYPES:
        errors.append(f"未知关系类型: {relation}，已知类型: {list(RELATION_TYPES.keys())}")

    if errors:
        return False, "; ".join(errors), warnings

    # 3. 验证 domain/range 约束
    rel_def = RELATION_TYPES.get(relation)
    if rel_def:
        domain = rel_def.get("domain", [])
        rng = rel_def.get("range", [])

        if entity1_type not in domain:
            errors.append(
                f"关系 {relation} 的 subject 类型应为 {domain}，但给出了 {entity1_type}"
            )

        if entity2_type not in rng:
            errors.append(
                f"关系 {relation} 的 object 类型应为 {rng}，但给出了 {entity2_type}"
            )

    # 4. 层级一致性检查 (仅对层级关系)
    if relation in ("BELONGS_TO", "HAS_SUBUNIT") and not errors:
        level1 = get_hierarchy_level(entity1_type)
        level2 = get_hierarchy_level(entity2_type)
        if level1 is not None and level2 is not None:
            if relation == "BELONGS_TO" and not (level1 == level2 + 1 or level1 > level2):
                warnings.append(
                    f"BELONGS_TO 通常表示下级→上级，但 {entity1_type} 层级不高于 {entity2_type}"
                )
            if relation == "HAS_SUBUNIT" and not (level2 == level1 + 1 or level2 > level1):
                warnings.append(
                    f"HAS_SUBUNIT 通常表示上级→下级，但 {entity2_type} 层级不低于 {entity1_type}"
                )

    # 5. SAME_AS 不能连接不同类型的实体
    if relation == "SAME_AS" and entity1_type != entity2_type:
        errors.append(f"SAME_AS 关系要求两端实体类型相同，这里是 {entity1_type} ↔ {entity2_type}")

    return len(errors) == 0, "; ".join(errors) if errors else None, warnings


def validate_triples_batch(triples):
    """
    批量验证三元组。
    triples: [(e1, e1_type, rel, e2, e2_type, evidence, confidence), ...]
    返回: {valid: [...], invalid: [...], warnings: [...]}
    """
    result = {"valid": [], "invalid": [], "warnings": []}
    for i, triple in enumerate(triples):
        e1, t1, rel, e2, t2 = triple[:5]
        evidence = triple[5] if len(triple) > 5 else None
        confidence = triple[6] if len(triple) > 6 else None

        valid, err, warns = validate_triple(e1, t1, rel, e2, t2)
        entry = {
            "index": i,
            "triple": (e1, t1, rel, e2, t2),
            "evidence": evidence,
            "confidence": confidence
        }
        if valid:
            if warns:
                entry["warnings"] = warns
                result["warnings"].append(entry)
            result["valid"].append(entry)
        else:
            entry["error"] = err
            if warns:
                entry["warnings"] = warns
            result["invalid"].append(entry)

    return result


def get_entity_type_suggestions(entity_name):
    """根据命名模式推荐实体类型候选"""
    import re
    suggestions = []
    for etype, edef in ENTITY_TYPES.items():
        pattern = edef.get("pattern")
        if pattern and re.match(pattern, entity_name):
            suggestions.append(etype)
    return suggestions if suggestions else list(ENTITY_TYPES.keys())


def get_allowed_relations(entity_type):
    """返回以该类型为 subject 的所有允许关系"""
    allowed = []
    for rname, rdef in RELATION_TYPES.items():
        if entity_type in rdef.get("domain", []):
            allowed.append({
                "relation": rname,
                "label": rdef["label"],
                "target_types": rdef.get("range", [])
            })
    return allowed
