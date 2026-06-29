# services/model_router.py — 多模型路由: get_model_response()
# 部署注意: 服务器仅部署 QwQ-32B，其他模型相关代码均已注释
import time, re, json, logging
import requests

from .logger import sanitize

logger = logging.getLogger(__name__)

# ============================================================
# 以下非 QwQ-32B 模型的 SDK 导入均已注释（服务器未部署）
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
# 以下为非 QwQ-32B 模型的辅助函数和类，均已注释
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


def get_model_response(prompt, model_name, image_base64=None, client_data=None):
    """
    封装调用不同模型的逻辑。
    当前部署仅启用 QwQ-32B（局业务网 AI4X），其余模型代码均已注释。
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
        # QwQ-32B (局业务网 AI4X) — 唯一启用的模型
        # ============================================================
        if model_name == "QwQ-32B":
            if image_base64:
                raise ValueError("QwQ-32B 是文本模型，不支持图片输入。")
            cfg = AI4X_CONFIG["QwQ-32B"]
            headers = {"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"}
            payload = {
                "model_name": cfg["api_model_name"],
                "prompt": prompt,
                "parameters": {
                    "temperature": client_data.get("temperature", 0.7),
                    "top_p": client_data.get("top_p", 0.95),
                    "max_tokens": client_data.get("max_tokens", 4096)
                }
            }
            response = requests.post(cfg["url"], headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            if res_json.get("code") == 0 and "data" in res_json and "text" in res_json["data"]:
                cleaned_response = res_json["data"]["text"].strip()
            else:
                raise Exception(f"局业务网 QwQ-32B 接口请求失败: {res_json.get('message', '未知错误')}")

        # ============================================================
        # deepseek-R1-32B — 注释（服务器未部署）
        # ============================================================
        # elif model_name == "deepseek-R1-32B": ...

        # ============================================================
        # qwen3.5-plus (百炼 DashScope)
        # ============================================================
        elif model_name == "qwen3.5-plus":
            from .config import bailian_client
            if bailian_client is None:
                raise RuntimeError("百炼 API Key 未配置，无法使用 qwen3.5-plus")
            response = bailian_client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=client_data.get("temperature", 0.3),
                top_p=client_data.get("top_p", 0.95),
                max_tokens=client_data.get("max_tokens", 4096)
            )
            cleaned_response = response.choices[0].message.content.strip()

        # ============================================================
        # Ollama 本地模型 — 注释（服务器未部署）
        # ============================================================
        # else:
        #     actual_ollama_model = SUPPORTED_MODELS.get(model_name) or MULTIMODAL_MODELS.get(model_name)
        #     ...

        else:
            raise ValueError(f"模型 '{model_name}' 未注册。当前支持: QwQ-32B, qwen3.5-plus")

    except Exception as e:
        logger.error("[ERROR] Model call failed for [%s]: %s", sanitize(str(model_name)), sanitize(str(e)))
        raise e

    duration = time.time() - start_time
    logger.info("<--- Received response from [%s]. Duration: %.2fs", sanitize(str(model_name)), duration)
    return cleaned_response
