#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事实卡抽取 v2：在 v1 的 7 个维度上扩展，并修复 v1 的三处结构性缺陷。

v1 的三个问题（本版修复）：
  1. 关键词优先级只在「同一段内」生效。v1 的 _find() 是双重循环里直接 return，
     实际语义是「第一个含任意关键词的段胜出」，而不是注释宣称的「按 kws 优先级」。
     维度一多、关键词一宽，高优先级关键词就形同虚设。
     → v2 收集全部候选后按 (关键词优先级, 段序) 排序取最优。
  2. 丢弃溯源信息。v1 只返回句子，无法回溯到原文段落。
     → v2 同时返回全局段落号，写入 evidence_span / src_para，
       使每道题的 gold 都能拿回 docx 原文核对（这是回应审稿意见 R2#1
       「gold 与系统检索来源同源」的最直接证据）。
  3. 无跨维度去重。同一句话会被多个维度重复取走——实测 thickness_num 有
     49/62 与 thickness 字段完全相同，overlying 有 35/58 与 contact 相同。
     → v2 按维度顺序「占用」句子，已被占用的句子不再被后续维度取用。

维度顺序刻意保持 v1 的 7 个维度在前且关键词不变，保证旧字段取值与 v1 一致；
新增维度只在剩余未被占用的句子里挑。

原则不变：宁缺毋滥。抽不准就留空，不出该维度题，绝不凑错误 gold 答案。

- 输入 : kb_storage/formation_descriptions.json（含 start_idx/end_idx）
         + 原始 docx（用于把非空段序号换算回全局段落号）
- 输出 : kb_storage/formation_facts_v2.json
"""
import sys, os, json, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC = os.path.join(PKG_DIR, "kb_storage", "formation_descriptions.json")
OUT = os.path.join(PKG_DIR, "kb_storage", "formation_facts_v2.json")
DOCX = r"E:\地质文本空间化平台\福建地质志.docx"

# ------------------------------------------------------------------
# 维度定义
# ------------------------------------------------------------------
# 每项: (字段名, 关键词列表[按优先级], 负向正则列表, 最短长度, 取值要求)
# 取值要求 need: "" = 不限 | "digit" = 须含数字 | "meter" = 须含米制厚度值
# 旧 7 维关键词与 v1 完全一致（保证可比），新增维度用收紧后的术语关键词。

OLD_DIMS = [
    ("lithology",    ["为一套", "指一套", "岩石组合为", "岩性为", "岩性由", "由下粗上细", "为一套海相"], [], 6, ""),
    ("distribution", ["主要分布于", "仅分布于", "分布于", "广布于", "主要出露于", "出露于", "见于", "分布在"], [], 6, ""),
    # ⚠️ 本维关键词**不再与 v1 保持一致**（其余旧维度仍保持）。
    # v1 的表里有裸的「组合」「组合为」，实测会让 51 条里 38 条命中「岩石组合」
    # 「岩性组合」这类岩性描述——问「X 含有哪些古生物化石？」而 gold 是
    # 「二者层位、时代、岩性组合均相若，可以对比」，是**错误的 gold**。
    # 改为只保留真正指向化石的术语；去掉裸的「产」（会撞「产生」），
    # 保留「双壳」等词根以覆盖「双壳类」的变体。
    ("fossil",       ["化石组合", "动物群", "植物群", "植物组合", "生物组合",
                      "化石有", "含化石", "产化石", "采获",
                      "菊石", "腕足", "双壳", "笔石", "孢粉", "珊瑚",
                      "苔藓虫", "海百合", "腹足", "介形虫", "牙形石",
                      "放射虫", "疑源类", "几丁虫", "三叶虫", "头足",
                      "植物化石", "动物化石", "蜓"],
                       # 排除章节标题（「二、生物组合与地层时代讨论」这类）
                       [r"^[一二三四五六七八九十]+、", r"^第.+节",
                        r"^（[一二三四五六七八九十]+）"], 6, ""),
    ("thickness",    ["总厚度", "总厚", "厚度大于", "厚大于", "厚度约", "厚度"], [], 6, ""),
    ("naming",       ["所创", "创建", "创名", "命名地点", "一名源于", "创名地点"], [], 6, ""),
    ("synonym",      ["曾称", "原称", "原名", "旧称", "即称", "沿用"], [], 6, ""),
    ("contact",      ["不整合", "整合接触", "整合于", "上覆地层", "下伏地层", "平行不整合", "整合覆于", "不整合覆盖"], [], 6, ""),
]

# 新增维度。关键词一律用地质术语，不用单字（v1 的候选里「带」「段」这类单字
# 会大量误命中——实测「带」有 56% 命中的是「带状分布」这类展布句）。
NEW_DIMS = [
    ("section",        ["正层型", "层型剖面", "建组剖面", "命名剖面", "参考剖面", "实测剖面", "剖面位于", "剖面在"],
                       [r"创名", r"^图", r"^表"], 15, ""),
    ("metamorph",      ["变质程度", "变质相", "绿片岩相", "角闪岩相", "麻粒岩相", "高绿片岩相", "低角闪岩相", "区域变质", "变质作用"],
                       [], 15, ""),
    # 负向规则：实测纯「含矿性/矿化」关键词会抽到「第八节 …地层含矿性简述」这类
    # 章节标题，以及「对…含矿层位时代的深入研究，研究程度不断提高」这类研究史评述。
    ("mineralization", ["含矿性", "含矿", "矿化", "赋存", "含煤", "煤层"],
                       [r"^第.+节", r"研究程度", r"研究史", r"简述\s*$",
                        r"倾向性意见", r"成因",
                        # 海进-海退旋回叙述：讲的是区域海平面变化史，
                        # 不是本单位含矿性（罗峰溪组实测被这条句抽走过）
                        r"海进"], 15, ""),
    # 负向规则：实测会抽到「二者在岩石组合、沉积环境、形成时代上基本相似」
    # 「由于沉积环境不同，各地沉积物差别较大」这类**提到**沉积环境但**没给出**
    # 是什么环境的句子，对「X 形成于什么沉积环境」这个问法是无效 gold。
    ("sed_env",        ["沉积环境", "沉积相", "滨浅海", "浅海", "滨海", "半深海", "深海", "三角洲", "碳酸盐台地", "潮坪", "泻湖", "陆源碎屑", "河流相", "湖泊相", "海相", "陆相"],
                       [r"基本相似", r"差别较大", r"变化较大", r"不同而异", r"可以对比",
                        r"相似[,，。]",
                        # 「沉积环境的差异/不同/变化」——只说明环境有变化，没说是哪种
                        r"沉积环境(?:的|等)?(?:不同|变化|差异)",
                        r"变化颇大", r"的探讨", r"图解",
                        # 「主要出露于…滨海山麓」是分布，不是沉积环境；
                        # 「皆为海相动物化石」只列化石，没说环境
                        r"主要出露于", r"皆为海相动物化石"], 15, ""),
    # 厚度数值：必须出现真正的米制数值。此前只查「含任意数字」，结果
    # 「上覆地层 楼子坝组（Pt33al）灰黄色中—厚层变质中细粒（长石）石英杂砂岩夹」
    # 这类**剖面表头行**靠单位代号里的 33 蒙混过关（实测 53 条命中里约 15 条是它）。
    # 另排除以上/下伏地层开头的剖面分栏行——那是接触关系，不是本组厚度。
    ("thickness_num",  ["厚度为", "厚约", "厚达", "最厚", "厚度大于", "总厚度为", "厚"],
                       [r"^上覆地层", r"^下伏地层",
                        r"层厚\d"], 15, "meter_term"),
    # 负向规则：排除图题（「图6-20 …柱状对比图」）、「对比只能是示意性的」这类
    # 讨论对比方法本身的句子，以及岩性/地化对比（「微量元素含量与玄武岩对比」）。
    # need="unit" 进一步要求句中出现地层单位名，剔除「相当于第3层上部」这类
    # 只是「层位相当」而非地层对比的句子。
    ("correlation",    ["可与", "相当于", "对比"],
                       [r"^图", r"柱状对比图", r"不可比", r"示意性",
                        r"原岩", r"微量元素", r"组合带", r"皆详见",
                        # 以下为复核实测的误命中：化石列表、植物群、剖面对比、
                        # 沉积相区对比、「各地变化不大」的空泛对比
                        r"微体化石", r"植物群", r"出露不全", r"优势相",
                        r"相区", r"变化不大"], 20, "unit"),
    ("overlying",      ["上覆地层", "上覆为", "上覆于", "上覆"],
                       [], 15, ""),
    ("underlying",     ["下伏地层", "下伏为", "下伏于", "下伏"],
                       [], 15, ""),
    # 关键词收紧了：去掉裸的「自下而上」。实测它会抽到
    # 「沉积环境自下而上由浅海相…向陆相转化」（讲沉积相变，不是分段）、
    # 「自下而上构成一个完整的沉积-火山喷发旋回」（讲旋回，不是分段）。
    # 保留的写法都自带「划分」义。
    ("subdivision",    ["自下而上划分为", "自下而上分为", "自下而上包含",
                        "可分为", "划分为", "进一步划分", "分为"],
                       [r"所创", r"创名", r"阶段", r"时段",
                        r"成分", r"组分", r"水分",          # 「脉体成分为花岗质…」里的「分为」
                        r"蜓带", r"生物带", r"组合带", r"类群"], 15, ""),
]

# ------------------------------------------------------------------
# 串味判据（只对「描述本单位自身属性」的维度生效）
# ------------------------------------------------------------------
# 实测：单位章节里的沿革/区域对比段落会成段引用邻近单位，抽出来的句子讲的是
# **别的组**。例：
#   东山组  抽到「长乐组（Qh2-3c） 厚20.75m」      ← 长乐组的厚度
#   黄连组  抽到「西溪组杂砂岩的结构及成分成熟度…显示快速、动荡的沉积环境」
#   桃子坑组 抽到「福建省石炭纪地层…划分为下统林地组、上统经畲组和老虎洞组」
# 这些句子如果当 gold，问题问的是甲组、答案说的是乙组，是**错误的 gold**，
# 比抽不到严重得多。
#
# 判据：句中出现了别的单位名，且既没有本单位的名字、也没有「本组/该组/本岩组」
# 这类回指 → 判为串味，丢弃。
#
# ⚠️ 下面四个维度**不适用**：它们的答案本来就是「点别的组名」——
#    overlying/underlying 答的必然是上/下伏的那个组；
#    contact 答的是与哪个组呈什么接触；
#    correlation 答的就是「可与哪些组对比」。
# 对它们套用本判据会把正确答案一并杀掉（实测 overlying 38→9、underlying 47→12）。
SELF_ONLY_DIMS = {
    "lithology", "distribution", "fossil", "thickness", "naming", "synonym",
    "section", "metamorph", "mineralization", "sed_env", "thickness_num",
    "subdivision",
}

# 少数维度用更严的判据：句子必须**出现本单位名或「本组/该组」回指**才收。
# 只用于 subdivision——它是实测精度最差的一维（23 条里 8 条答非所问，如
# 「安仁组」抽到「福建晚石炭世古地理面貌可分为三个古地理区」，
# 时代都不是同一个）。因为这批错句里没有别的单位名，串味判据拦不住。
# 代价是每条单位少 0.15 个事实位，换来的是不再有「问甲答乙」的 gold。
STRICT_SELF_DIMS = {"subdivision"}

_SELF_ANAPHOR_RE = re.compile(r'本(?:岩|地|群|组|段|单)?(?:组|岩组|群|段)|该(?:岩)?(?:组|群|段)')
# 原书写法「黄潭（岩）组」「龙北溪（岩）组」——括号里的「岩」使子串匹配不上
# 不括号的正式名「黄潭组」，串味判据会漏判（实测龙北溪组因此抽到讲万全群的句子）。
_UNIT_PAREN_RE = re.compile(r'[（(]岩[）)]')

# 地层单位名（用于 correlation：地层对比的答案里必须点到具体单位名，
# 否则会收进「与玄武岩对比」「相当于第 3 层」这类非地层对比句）
_UNIT_TOKEN_RE = re.compile(r'[一-龥]{1,8}(?:组|群|段)')

AGE = {
    "天井坪岩组": "古元古代", "麻源岩群": "古元古代", "大金山岩组": "古元古代",
    "南山岩组": "古元古代", "迪口岩组": "古元古代", "桃溪岩组": "古元古代",
    "东岩组": "青白口纪—南华纪", "大岭组": "青白口纪—南华纪", "龙北溪组": "青白口纪—南华纪",
    "黄潭组": "青白口纪—南华纪", "下峰组": "青白口纪—南华纪",
    "楼前组": "震旦纪", "西溪组": "震旦纪", "楼子坝组": "早震旦世",
    "南岩组": "晚震旦世", "黄连组": "晚震旦世",
    "林田组": "早—晚寒武世", "东坑口组": "末寒武世", "魏坊组": "早奥陶世", "罗峰溪组": "中—晚奥陶世",
    "天瓦崠组": "晚泥盆世", "桃子坑组": "晚泥盆世",
    "林地组": "早石炭世", "经畲组": "中石炭世", "老虎洞组": "中石炭世",
    "船山组": "早二叠世", "栖霞组": "早二叠世", "泉上组": "中二叠世",
    "文笔山组": "中二叠世", "童子岩组": "中二叠世", "翠屏山组": "晚二叠世",
    "罗坑组": "晚二叠世", "长兴组": "晚二叠世",
    "溪口组": "早三叠世", "安仁组": "中三叠世",
    "大坑村组": "晚三叠世", "文宾山组": "晚三叠世", "焦坑组": "晚三叠世",
    "象牙群": "早侏罗世", "下村组": "早侏罗世", "藩坑组": "早侏罗世",
    "梨山组": "早侏罗世", "漳平组": "中侏罗世",
    "长林组": "晚侏罗世", "南园组": "晚侏罗世—早白垩世",
    "小溪组": "早白垩世", "下渡组": "早白垩世", "坂头组": "早白垩世", "吉山组": "早白垩世",
    "石帽山群": "早白垩世晚期", "黄坑组": "早白垩世晚期", "寨下组": "早白垩世晚期",
    "石牛山组": "晚白垩世", "赤石群": "晚白垩世", "均口组": "晚白垩世",
    "沙县组": "晚白垩世", "崇安组": "晚白垩世",
    "佛昙组": "新近纪", "天宝组": "早更新世", "同安组": "中更新世",
    "龙海组": "晚更新世", "东山组": "早全新世", "长乐组": "中—晚全新世",
}

# ------------------------------------------------------------------
# 段落过滤
# ------------------------------------------------------------------
# 实测：3293 个非空段里有 1550 段（47%）是**实测剖面柱状描述**——由表格/柱状图
# 展平而来的编号行，形如「4.灰色厚—巨厚层状黄铁矿化绢云千枚岩 28.0m」。
# 它们不是叙述句，若不过滤会被当成 gold 抽出来（例如含矿性维度会抽出
# 「黄铁矿化绢云千枚岩 28.0m」这种柱状条目，问出来毫无意义）。
#
# v1 的 SKIP_RE 本意就是过滤它们，但 `$` 的位置使整个交替分支必须以行尾结束，
# 于是只匹配「整行恰好是一个编号」或「整行恰好是一个厚度」，实测仅过滤掉 49 段。
SKIP_RES = [
    # 裸厚度行 / 裸序号行 / 裸段名行
    re.compile(r'^\s*(?:'
               r'[＞>]?\d+(?:\.\d+)?\s*m'
               r'|[0-9]{1,3}[.．、]'
               r'|(?:上段|下段|中段|第一段|第二段|第三段|第四段|掩盖|未见顶|未见底)'
               r')\s*$'),
    # 柱状描述行：以「序号.」开头、以厚度值结尾
    re.compile(r'^\d{1,3}[.．、].*?[＞>]?\d+(?:\.\d+)?\s*m\s*$'),
    # 以「序号.」开头的任何行。上面两条只覆盖「整行是序号」和「序号开头+厚度结尾」，
    # 实测还有 331 行是**厚度值被拆到下一行**的柱状描述，例如
    #   6.浅灰色厚层黑云斜长变粒岩与长石石英岩互层，顶部为深灰色厚层角闪斜长变粒岩
    # 它们同样不是叙述句。全部剔除：实测总事实位只从 715 降到 710，
    # 丢掉的那 5 个恰好就是这类噪声行。
    re.compile(r'^\d{1,3}[.．、]'),
]


def is_noise(line):
    return any(r.match(line) for r in SKIP_RES)

_DIGIT_RE = re.compile(r'\d')
# 米制厚度值：数字紧跟 m/M/米 才算（允许中间有空格）。
_METER_RE = re.compile(r'\d+(?:\.\d+)?\s*(?:m|M|米)')
# 厚度术语。只有裸的「厚」不算——实测柱状行残片
# 「厚10-40cm，三者常呈韵律产出 89.9m」就是靠这个裸「厚」加上末尾的 89.9m
# 混进来的（它讲的是某个单层的韵律厚度，不是桃溪岩组的总厚度）。
_THICK_TERM_RE = re.compile(r'厚度|总厚|厚约|厚达|最厚')


def _clean(s):
    return re.sub(r'\s+', ' ', s).strip()


def _clip(s, n=200):
    s = _clean(s)
    return s if len(s) <= n else s[:n] + "…"


def _norm(s):
    """归一化，用于句子级去重（去空白 + 统一标点）。"""
    return re.sub(r'[\s。；，、．,;]', '', s or "")


def _sent_around(seg, pos):
    """取 pos 所在句子（前后句号/分号界定）。"""
    start = max(seg.rfind('。', 0, pos), seg.rfind('；', 0, pos), seg.rfind('．', 0, pos))
    start = start + 1 if start >= 0 else 0
    end = seg.find('。', pos)
    end2 = seg.find('；', pos)
    if end == -1 or (end2 != -1 and end2 < end):
        end = end2
    if end == -1:
        end = len(seg)
    return seg[start:end]


def _candidates(narrative, kws):
    """收集全部候选，返回 [(kw优先级, 段序, 句子)]，按 (kw优先级, 段序) 升序。

    这是修复 v1 缺陷 1 的关键：v1 是「第一段里有任意关键词就返回」，
    本版把关键词优先级提到段序之前，保证高优先级关键词真的优先。
    """
    out = []
    for si, seg in enumerate(narrative):
        for ki, kw in enumerate(kws):
            pos = seg.find(kw)
            if pos >= 0:
                out.append((ki, si, _sent_around(seg, pos)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _pick(narrative, kws, used, neg=None, minlen=6, maxlen=200, need=""):
    """按优先级挑一句未被占用、未被负向规则排除、长度合规的句子。

    Args:
        need: "" 不限 | "digit" 须含数字 | "meter" 须含米制厚度值
              | "unit" 须含地层单位名（××组/××群/××段）
        neg:  负向正则列表，命中任一即淘汰该候选

    Returns: (句子, 段序, 命中的关键词)  未命中返回 ("", -1, "")
    """
    for ki, si, sent in _candidates(narrative, kws):
        s = _clip(sent, maxlen)
        if len(s) < minlen:
            continue
        if need == "digit" and not _DIGIT_RE.search(s):
            continue
        if need == "meter" and not _METER_RE.search(s):
            continue
        if need == "meter_term" and not (_METER_RE.search(s) and _THICK_TERM_RE.search(s)):
            continue
        if need == "unit" and not _UNIT_TOKEN_RE.search(s):
            continue
        if neg and any(re.search(p, s) for p in neg):
            continue
        key = _norm(s)
        if key in used:
            continue
        return s, si, kws[ki]
    return "", -1, ""


def extract_unit(u, para_indices, all_names=()):
    """抽取单个单位的事实卡。

    Args:
        u: formation_descriptions.json 的一条（含 start_idx/end_idx/text）
        para_indices: 非空段序号 -> 全局段落号 的映射列表，
                      即 para_indices[j] 是 text 第 j 行的全局段落号
        all_names: 全部单位名，供串味判据判断句中出现了哪些「别的组」
    """
    segs = [l.strip() for l in u["text"].split("\n") if l.strip()]
    # narrative 记录 (非空段序号, 文本)，以便回溯全局段落号
    narrative, narr_idx = [], []
    for j, s in enumerate(segs):
        if not is_noise(s):
            narrative.append(s)
            narr_idx.append(j)

    name = u["name"]
    name_n = _UNIT_PAREN_RE.sub("", name)                      # 黄潭（岩）组 -> 黄潭组
    others = [_UNIT_PAREN_RE.sub("", n) for n in all_names if n != name]

    facts = {"name": name, "symbol": u["symbol"], "age": AGE.get(name, "")}
    evidence = {}
    used = set()
    stats = {"picked": 0, "cross_unit": 0, "strict_self": 0,
             "blocked_dup": 0, "blocked_short": 0, "blocked_neg": 0}

    for field, kws, neg, minlen, need in (OLD_DIMS + NEW_DIMS):
        s, si, kw = _pick(narrative, kws, used, neg, minlen, 200, need)
        if s and field in SELF_ONLY_DIMS:
            s_n = _UNIT_PAREN_RE.sub("", s)
            has_self = (name_n in s_n) or bool(_SELF_ANAPHOR_RE.search(s_n))
            # 严判据：见 STRICT_SELF_DIMS 上方注释
            if field in STRICT_SELF_DIMS and not has_self:
                stats["strict_self"] += 1
                s = ""
            # 串味判据：见 SELF_ONLY_DIMS 上方注释
            elif any(o in s_n for o in others) and not has_self:
                stats["cross_unit"] += 1
                s = ""
        if s:
            facts[field] = s
            used.add(_norm(s))
            gidx = narr_idx[si]
            evidence[field] = {
                "src_para": para_indices[gidx] if gidx < len(para_indices) else None,
                "kw": kw,
            }
            stats["picked"] += 1
        else:
            facts[field] = ""

    facts["evidence"] = evidence
    return facts, stats


def load_docx_paras():
    from docx import Document
    return [p.text.strip() for p in Document(DOCX).paragraphs]


def main():
    descs = json.load(open(DESC, encoding="utf-8"))
    paras = load_docx_paras()

    all_names = [u["name"] for u in descs]
    all_facts, totals = [], {"picked": 0, "cross_unit": 0, "strict_self": 0}
    for u in descs:
        # text 是由「非空段」join 而成，故第 j 行 = paras[start:end] 里第 j 个非空段
        seg_indices = [i for i in range(u["start_idx"], u["end_idx"])
                       if i < len(paras) and paras[i]]
        f, st = extract_unit(u, seg_indices, all_names)
        # 溯源自检：抽出的句子必须真的出现在原文那一句里
        bad = 0
        for field, ev in f.get("evidence", {}).items():
            gi = ev.get("src_para")
            if gi is None or gi >= len(paras):
                bad += 1
                continue
            probe = f[field].rstrip("…")[:12]
            if probe and probe not in paras[gi] and _norm(probe) not in _norm(paras[gi]):
                bad += 1
        f["_trace_ok"] = (bad == 0)
        if bad:
            print(f"  [WARN] {u['name']}: {bad} 个字段溯源校验失败")
        all_facts.append(f)
        totals["picked"] += st["picked"]
        totals["cross_unit"] += st["cross_unit"]
        totals["strict_self"] += st["strict_self"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_facts, f, ensure_ascii=False, indent=2)

    print(f"\n{'维度':<16}{'v2 命中':>10}{'v1 命中':>10}{'净新增':>10}")
    print("-" * 48)
    try:
        v1 = {x["name"]: x for x in json.load(open(
            os.path.join(PKG_DIR, "kb_storage", "formation_facts.json"), encoding="utf-8"))}
    except Exception:
        v1 = {}
    n_new = 0
    for field, *_ in (OLD_DIMS + NEW_DIMS):
        c2 = sum(1 for x in all_facts if x.get(field))
        c1 = sum(1 for x in v1.values() if x.get(field)) if v1 else 0
        if field not in [d[0] for d in OLD_DIMS]:
            n_new += c2
            tag = "  ← 新增"
        else:
            tag = ""
        print(f"{field:<16}{c2:>10}{c1:>10}{c2 - c1:>10}{tag}")

    n_ok = sum(1 for x in all_facts if x.get("_trace_ok"))
    n_fields = sum(1 for x in all_facts for f, *_ in (OLD_DIMS + NEW_DIMS) if x.get(f))
    print(f"\n单位数: {len(all_facts)}")
    print(f"总事实位: {n_fields}")
    print(f"新增维度贡献事实位: {n_new}")
    print(f"串味句拦截: {totals['cross_unit']} 条（句中说别组、无本单位指代）")
    print(f"严判据拦截: {totals['strict_self']} 条（subdivision 未提及本单位）")
    print(f"溯源校验通过的单位: {n_ok}/{len(all_facts)}")
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    main()
