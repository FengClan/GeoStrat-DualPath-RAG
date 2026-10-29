# services/config.py — 所有API密钥、模型配置、URL (从.env加载)
# 部署注意: 服务器仅部署 qwen3.5-plus 模型，其他模型相关代码已注释
import os
import logging

_logger = logging.getLogger(__name__)

# 尝试从 .env 文件加载，未安装 python-dotenv 则跳过
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

# --- 路径配置 ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'kb_uploads')
GRAPH_STORAGE_DIR = "./kb_storage/graphs"

# ============================================================
# 除 qwen3.5-plus 外的模型 API 凭证均改为非致命警告
# 服务器仅使用 qwen3.5-plus（百炼 DashScope），不需要星火/AI4X/Gemini 等密钥
# ============================================================

# --- Google Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "请在这里填入您的Google_API_KEY")

# --- 星火图像理解 (注释) ---
# SPARK_IMAGE_APPID = os.getenv("SPARK_IMAGE_APPID")
# SPARK_IMAGE_API_SECRET = os.getenv("SPARK_IMAGE_API_SECRET")
# SPARK_IMAGE_API_KEY = os.getenv("SPARK_IMAGE_API_KEY")
# if not all([SPARK_IMAGE_APPID, SPARK_IMAGE_API_SECRET, SPARK_IMAGE_API_KEY]):
#     raise RuntimeError("星火图像理解 API 凭证未设置，请检查 .env 文件")
# SPARK_IMAGE_URL = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"
# SPARK_IMAGE_DOMAIN = "imagev3"

# --- 星火文本生成 (注释) ---
# SPARKAI_URL_V35 = 'wss://spark-api.xf-yun.com/v3.5/chat'
# SPARKAI_APP_ID_V35 = os.getenv("SPARK_APP_ID_V35")
# SPARKAI_API_SECRET_V35 = os.getenv("SPARK_API_SECRET_V35")
# SPARKAI_API_KEY_V35 = os.getenv("SPARK_API_KEY_V35")
# if not all([SPARKAI_APP_ID_V35, SPARKAI_API_SECRET_V35, SPARKAI_API_KEY_V35]):
#     raise RuntimeError("星火 V3.5 API 凭证未设置，请检查 .env 文件")
# SPARKAI_DOMAIN_V35 = 'generalv3.5'

# SPARKAI_URL_V40 = 'wss://spark-api.xf-yun.com/v4.0/chat'
# SPARKAI_APP_ID_V40 = os.getenv("SPARK_APP_ID_V40")
# SPARKAI_API_SECRET_V40 = os.getenv("SPARK_API_SECRET_V40")
# SPARKAI_API_KEY_V40 = os.getenv("SPARK_API_KEY_V40")
# if not all([SPARKAI_APP_ID_V40, SPARKAI_API_SECRET_V40, SPARKAI_API_KEY_V40]):
#     raise RuntimeError("星火 V4.0 API 凭证未设置，请检查 .env 文件")
# SPARKAI_DOMAIN_V40 = '4.0Ultra'

# SPARKAI_URL_V31 = 'wss://spark-api.xf-yun.com/v3.1/chat'
# SPARKAI_APP_ID_V31 = os.getenv("SPARK_APP_ID_V31")
# SPARKAI_API_SECRET_V31 = os.getenv("SPARK_API_SECRET_V31")
# SPARKAI_API_KEY_V31 = os.getenv("SPARK_API_KEY_V31")
# if not all([SPARKAI_APP_ID_V31, SPARKAI_API_SECRET_V31, SPARKAI_API_KEY_V31]):
#     raise RuntimeError("星火 V3.1 API 凭证未设置，请检查 .env 文件")
# SPARKAI_DOMAIN_V31 = 'generalv3'

# --- 百炼 (阿里云 DashScope) ---
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
if not BAILIAN_API_KEY:
    BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not BAILIAN_API_KEY:
    print("[CONFIG] 警告: 百炼 API Key 未设置，qwen3.5-plus 不可用")
    bailian_client = None
else:
    from openai import OpenAI
    # timeout / max_retries 必须显式设置：
    # 不设时 SDK 默认 timeout=600s、max_retries=2，一次丢包/卡住的调用会让该 worker
    # 干等十几分钟（2026-09-14 实测：单题卡了 11 分钟才返回，整个基线的墙钟时间被
    # 一题拖垮；并发评测时这等于浪费一个 worker 名额）。
    # max_retries=0：重试交给 model_router 自己的指数退避（1s/2s/4s 共 4 次），
    # 否则两层重试叠加成最多 12 次请求，欠费等致命错误也会被白等好几轮。
    bailian_client = OpenAI(
        api_key=BAILIAN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # 300s 而非更短：实测开启思考时单题生成 60~95s、completion 可达 2000+ token，
        # 卡太紧会把"正常但较慢"的生成掐断，重跑一次比多等一会更贵。
        timeout=300.0,
        max_retries=0,
    )

# --- DeepSeek (独立裁判模型, OpenAI 兼容) ---
# 生成仍用 qwen3.5-plus；裁判改用 DeepSeek，避免同模型 self-preference 偏差 (R1#4 / R2#7)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_JUDGE_MODEL = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-flash")
if not DEEPSEEK_API_KEY:
    print("[CONFIG] 警告: DeepSeek API Key 未设置，deepseek-flash 不可用（裁判将无法独立）")
    deepseek_client = None
else:
    from openai import OpenAI
    # 同 bailian_client：显式 timeout + 关闭 SDK 自带重试（走 model_router 的统一退避）。
    # 裁判要产出的 JSON 很短，但思考模式实测单题仍烧 1000~3200 completion token，故同样给 300s。
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=300.0,
        max_retries=0,
    )

# --- 模型列表 ---
SUPPORTED_MODELS = {
    # "Spark Max": "Spark Max",                         # 注释：服务器未部署
    # "Spark Pro": "Spark Pro",                         # 注释：服务器未部署
    # "Spark4.0 Ultra": "Spark4.0 Ultra",               # 注释：服务器未部署
    # "deepseek-r1:7b": "deepseek-r1:7b",              # 注释：服务器未部署
    # "deepseek-r1:8b": "deepseek-r1:8b",              # 注释：服务器未部署
    # "llama3.1:8b": "llama3.1:8b",                    # 注释：服务器未部署
    # "gemma3:4b": "gemma3:4b",                        # 注释：服务器未部署
    # "Qwen3-30B-A3B": "Qwen3-30B-A3B",               # 注释：服务器未部署
    # "deepseek-R1-32B": "deepseek-R1-32B",            # 注释：服务器未部署
    "qwen3.5-plus": "qwen3.5-plus",
    "deepseek-flash": "deepseek-flash",                 # 独立裁判模型
    # "QwQ-32B": "QwQ-32B",                              # 注释：已切换到 qwen3.5-plus
    # "Gemini Pro": "gemini-pro"                        # 注释：服务器未部署
}

MULTIMODAL_MODELS = {
    # "llava": "llava",                                 # 注释：服务器未部署
    # "bakllava": "bakllava",                           # 注释：服务器未部署
    # "moondream": "moondream",                         # 注释：服务器未部署
    # "Gemini Pro Vision": "gemini-pro-vision",         # 注释：服务器未部署
    # "Spark-Image": "Spark-Image"                      # 注释：服务器未部署
}

# --- 局业务网 AI4X 模型服务 ---
AI4X_QWQ_TOKEN = os.getenv("AI4X_QWQ_TOKEN")
# AI4X_DEEPSEEK_TOKEN = os.getenv("AI4X_DEEPSEEK_TOKEN")  # 注释：服务器未部署

# AI4X_CONFIG = {                                       # 注释：已切换到 qwen3.5-plus
    # "QwQ-32B": {
    #     "url": "https://ai4x.cgs.cn/api/model/services/69705c9eb8331b9cc94e6365/app/v1",
    #     "token": AI4X_QWQ_TOKEN,
    #     "api_model_name": "QwQ-32B"
    # },
    # # "deepseek-R1-32B": {                              # 注释：服务器未部署
    # #     "url": "https://ai4x.cgs.cn/api/model/services/67bb1b8aacc3b46c46bff251/app/v1",
    # #     "token": AI4X_DEEPSEEK_TOKEN,
    # #     "api_model_name": "deepseek-32b"
    # # }
# }
AI4X_CONFIG = {}  # 已停用 QwQ-32B，改为百炼 qwen3.5-plus

# 检查 API Token
if not BAILIAN_API_KEY:
    raise RuntimeError("百炼 API Key 未设置。请检查 .env 文件（BAILIAN_API_KEY）")
# if not AI4X_QWQ_TOKEN:                               # 注释：已停用 QwQ-32B
#     print("[CONFIG] 警告: QwQ-32B Token 未设置，该模型不可用")

# --- 抽取任务 Prompt 模板 ---
TASK_PROMPT = {
    "wordrec": "对上述句子进行中文分词，如输入\"张三毕业于中国地质大学（武汉），现就职于微软，任开发岗位。\"后应按照张三/毕业于/中国地质大学（武汉）/，/现就职于/微软/，/任/开发/岗位/。的格式输出。",
    "ner": "基于以上句子请进行实体抽取，务必输出全部实体及标签。返回格式为实体，标签,每组之间换行。例如，输入：张三是中国人。应返回：张三，人物  中国，国家",
    "ree": "基于以上句子请进行关系抽取，要求严格按照三元组的格式返回，如（实体，关系，实体）",
    "theme": "基于以上句子请进行主题抽取，严格按照如以下的格式输出：主题1: [关键词1, 关键词2, 关键词3]  主题2: [关键词1, 关键词2, 关键词3]",
    "pe": "基于以上句子进行属性抽取，严格按照（实体，属性，属性值）的形式输出，只给出结果，不要有多余的回答。例如：(糖尿病,症状,多饮)、(糖尿病,治疗方法,定期血糖检测)",
    "ste": "基于以上句子进行时空抽取，严格按照 TIME:XXX LOC:XXX 的形式输出，只给出结果，不要有多余的回答。"
}
