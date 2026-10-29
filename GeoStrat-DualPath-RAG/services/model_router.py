# services/model_router.py — 多模型路由: get_model_response()
# 部署注意: 服务器仅部署 qwen3.5-plus（百炼 DashScope），其他模型相关代码均已注释
import os, time, re, json, logging, threading
import requests

from .logger import sanitize

logger = logging.getLogger(__name__)

# ============================================================
# 以下非 qwen3.5-plus 模型的 SDK 导入均已注释（服务器未部署）
# ============================================================

# try:
#     import google.generativeai as genai
# except ImportError:
#     genai = None

# try:
#     from PIL import Image
# except ImportError:
#     Image = None

# try:
#     from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
#     from sparkai.core.messages import ChatMessage
# except ImportError:
#     ChatSparkLLM = None
#     ChunkPrintHandler = None
#     ChatMessage = None

# from io import BytesIO
# import base64, ssl, hmac, hashlib, threading, datetime
# from urllib.parse import urlparse, urlencode
# from wsgiref.handlers import format_date_time
# from time import mktime
# import websocket

from .config import SUPPORTED_MODELS, MULTIMODAL_MODELS, AI4X_CONFIG

# ============================================================
# 以下为非 qwen3.5-plus 模型的辅助函数和类，均已注释
# ============================================================

# _is_google_configured = False
# if genai is not None and GOOGLE_API_KEY and GOOGLE_API_KEY != "请在这里填入您的Google_API_KEY":
#     try:
#         genai.configure(api_key=GOOGLE_API_KEY)
#         _is_google_configured = True
#         logger.info("Google Generative AI configured successfully.")
#     except Exception as e:
#         logger.error("Failed to configure Google Generative AI: %s", sanitize(str(e)))

# def gen_params_spark_image(appid, domain, question_list): ...

# class SparkImageWebSocketClient: ...


# 欠费/额度耗尽等永久性错误：不重试，直接抛出（避免空转刷题）
_FATAL_ERR_KEYS = (
    "Arrearage", "overdue-payment", "good standing",       # 百炼欠费
    "Insufficient Balance", "insufficient_quota",           # DeepSeek / OpenAI 额度耗尽
)


# ============================================================
# 思考模式开关（qwen3.5-plus 提速 / 降本）
# ============================================================
# qwen3.5-plus 在 DashScope 兼容接口上**默认开思考**，而 _call_openai_compatible
# 最终只取 message.content——思维链整段被丢弃，时间和 token 都白花。
#
# 玩具问题实测（短答案，2026-09-14）：
#   带思考  41.0s / completion 2076 tok / 隐藏思维链 6553 字 → 答案 33 字
#   关思考   2.4s / completion   24 tok / 思维链 0 字        → 答案 32 字
#
# 但**真实题目上关思考会掉分**。30 题 dual_path 对照（evaluation/compare_thinking.py，
# seed=42，裁判 deepseek-flash，臂内并发 6）：
#   overall_score   0.7333 → 0.6902  (-0.0431)  ← 超过 ±0.02 持平阈值
#   qa_correctness  0.6600 → 0.5900  (-0.0700)
#   输出 token       117666 → 15333  (降到 1/7.7)
#   墙钟              507.3s → 132.1s (快 3.8 倍)
#   按题型：FACTUAL +0.008(n=19) / COMPARATIVE +0.017(n=3) 几乎不变，
#           但 REASONING **-0.193**(n=7) —— 推理题确实靠思考。
#
# 结论：**不要全局关思考**。全量评测靠 --workers 并发提速（语义完全不变），
# 不要靠这个开关。保留此开关是为了：(a) 复现上面的对照数据供 R2#18 报效率/成本；
# (b) 万一将来模型侧思考行为变化，有个现成的实验变量。
# ⚠️ 关思考跑出的结果与 results_archive_20260914/ 不可直接对比，论文中须注明配置。
_THINKING_ENV = "QWEN_ENABLE_THINKING"
_THINKING_MODELS = ("qwen3.5-plus",)   # 仅该模型默认开思考；deepseek-flash 不适用


# ============================================================
# Token 用量累计（R2#18 要的效率/成本数据；也是思考开关降本幅度的量尺）
# ============================================================
# 只统计、不改返回值——既有调用点（baselines / evaluator / Flask 蓝图）一行都不必动。
# 代价是每次调用多一次加锁计数，相对几十秒的网络等待可忽略。
_USAGE_LOCK = threading.Lock()
_USAGE = {}          # model -> {calls, prompt_tokens, completion_tokens, total_tokens}


def _record_usage(model, usage):
    if usage is None:
        return
    with _USAGE_LOCK:
        u = _USAGE.setdefault(model, {"calls": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "total_tokens": 0})
        u["calls"] += 1
        u["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        u["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        u["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


def usage_stats(reset=False):
    """各模型累计 token 用量快照。reset=True 则先清零再返回（便于分轮统计）。"""
    with _USAGE_LOCK:
        snap = {k: dict(v) for k, v in _USAGE.items()}
        if reset:
            _USAGE.clear()
    return snap


def _thinking_extra_body(model, client_data):
    """需要在请求里显式关闭思考时返回 extra_body，否则返回 None。

    优先级：client_data 显式指定 > 环境变量 QWEN_ENABLE_THINKING > 默认开启。
    只有默认开思考的模型才需要显式传参关闭。
    """
    if model not in _THINKING_MODELS:
        return None
    if client_data and "enable_thinking" in client_data:
        want = bool(client_data["enable_thinking"])
    else:
        want = os.environ.get(_THINKING_ENV, "1").strip().lower() not in (
            "0", "false", "no", "off")
    return {"enable_thinking": False} if not want else None


def _call_openai_compatible(client, model, prompt, client_data):
    """OpenAI 兼容接口调用 + 瞬时故障指数退避重试 (1s/2s/4s，共 4 次)。"""
    last_err = None
    extra = _thinking_extra_body(model, client_data)
    for attempt in range(4):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=client_data.get("temperature", 0.3),
                top_p=client_data.get("top_p", 0.95),
                max_tokens=client_data.get("max_tokens", 4096)
            )
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)
            _record_usage(model, getattr(response, "usage", None))
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if any(k in str(e) for k in _FATAL_ERR_KEYS):
                raise
            if attempt < 3:
                wait = 2 ** attempt
                logger.warning("[WARN] %s 调用失败(第%d次)，%.1fs 后重试: %s",
                               sanitize(str(model)), attempt + 1, wait, sanitize(str(e)))
                time.sleep(wait)
            else:
                raise
    raise last_err


def get_model_response(prompt, model_name, image_base64=None, client_data=None):
    """
    封装调用不同模型的逻辑。
    当前部署仅启用 qwen3.5-plus（百炼 DashScope），其余模型代码均已注释。
    :return: 模型返回的清理后的文本
    """
    if client_data is None:
        client_data = {}

    logger.info("---> Sending request to AI model: [%s]...", sanitize(str(model_name)))
    start_time = time.time()
    cleaned_response = ""

    try:
        # ============================================================
        # Gemini Pro / Gemini Pro Vision — 注释（服务器未部署）
        # ============================================================
        # if model_name == "Gemini Pro": ...
        # elif model_name == "Gemini Pro Vision": ...

        # ============================================================
        # Spark Max / Spark Pro / Spark4.0 Ultra — 注释（服务器未部署）
        # ============================================================
        # elif model_name == "Spark Max": ...
        # elif model_name == "Spark Pro": ...
        # elif model_name == "Spark4.0 Ultra": ...

        # ============================================================
        # Spark-Image — 注释（服务器未部署）
        # ============================================================
        # elif model_name == "Spark-Image": ...

        # ============================================================
        # QwQ-32B (局业务网 AI4X) — 注释（已切换到百炼 qwen3.5-plus）
        # ============================================================
        # if model_name == "QwQ-32B":
        #             if image_base64:
        #         raise ValueError("QwQ-32B 是文本模型，不支持图片输入。")
        #     cfg = AI4X_CONFIG["QwQ-32B"]
        #     headers = {"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"}
        #     payload = {
        #         "model_name": cfg["api_model_name"],
        #         "prompt": prompt,
        #         "parameters": {
        #             "temperature": client_data.get("temperature", 0.7),
        #             "top_p": client_data.get("top_p", 0.95),
        #             "max_tokens": client_data.get("max_tokens", 4096)
        #         }
        #     }
        #     response = requests.post(cfg["url"], headers=headers, json=payload, timeout=120)
        #     response.raise_for_status()
        #     res_json = response.json()
        #     if res_json.get("code") == 0 and "data" in res_json and "text" in res_json["data"]:
        #         cleaned_response = res_json["data"]["text"].strip()
        #             else:
        #         raise Exception(f"局业务网 QwQ-32B 接口请求失败: {res_json.get('message', '未知错误')}")

        # ============================================================
        # deepseek-R1-32B — 注释（服务器未部署）
        # ============================================================
        # elif model_name == "deepseek-R1-32B": ...

        # ============================================================
        # qwen3.5-plus (百炼 DashScope)
        # ============================================================
        if model_name == "qwen3.5-plus":
            from .config import bailian_client
            if bailian_client is None:
                raise RuntimeError("百炼 API Key 未配置，无法使用 qwen3.5-plus")
            cleaned_response = _call_openai_compatible(
                bailian_client, "qwen3.5-plus", prompt, client_data)

        # ============================================================
        # deepseek-flash (独立裁判模型, OpenAI 兼容)
        # 生成仍用 qwen3.5-plus；裁判用 DeepSeek 以避开同模型 self-preference 偏差
        # ============================================================
        elif model_name == "deepseek-flash":
            from .config import deepseek_client
            if deepseek_client is None:
                raise RuntimeError("DeepSeek API Key 未配置，无法使用 deepseek-flash")
            cleaned_response = _call_openai_compatible(
                deepseek_client, "deepseek-flash", prompt, client_data)

        # ============================================================
        # Ollama 本地模型 — 注释（服务器未部署）
        # ============================================================
        # else:
        #     actual_ollama_model = SUPPORTED_MODELS.get(model_name) or MULTIMODAL_MODELS.get(model_name)
        #     ...

        else:
            raise ValueError(f"模型 '{model_name}' 未注册。当前支持: qwen3.5-plus, deepseek-flash")

    except Exception as e:
        logger.error("[ERROR] Model call failed for [%s]: %s", sanitize(str(model_name)), sanitize(str(e)))
        raise e

    duration = time.time() - start_time
    logger.info("<--- Received response from [%s]. Duration: %.2fs", sanitize(str(model_name)), duration)
    return cleaned_response
