# services/litho_service.py — 岩石地层信息: 文本/图片/表格抽取
import os, time, json, base64
import logging
from io import BytesIO

import pandas as pd
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from .config import SUPPORTED_MODELS, MULTIMODAL_MODELS
from .model_router import get_model_response
from .logger import sanitize

logger = logging.getLogger(__name__)

litho_bp = Blueprint('litho', __name__)


def build_prompt(config, question, context=None, is_chunk=False):
    """构建文本抽取的提示词"""
    preamble = ""
    if is_chunk:
        preamble = "重要提示：..."
    base_prompt = (
        f"{config}\n，请学习这种方法，生成表格，务必保证表格的表头严格上述方法中的表头保持一致。"
        f"表头与表头之间务必采用|分割，千万别用逗号。数据与数据之间务必采用|分割，千万别用逗号。"
        f"先确定好表头后再抽取数据，数据务必保证仅根据以下文本抽取，不包括之前提到的方法中的数据：{question}\n "
        f"返回结果中一定要有'表头：''数据行：'这几个字，一定要注明表头：数据行："
        f"（如表头：表头1|表头2    数据行：数据1|数据2|  数据行：数据3|数据4|），每行数据前都要有数据行：这几个字。"
        f"在表头和每行数据前面。不要回复除结果外多余的话。数据严格从文本中复制，不要捏造。"
        f"有几组就返回几组数据，确保数据严谨可靠。允许某数据值为空，如果某数据值为空，以'/'代替。"
    )
    if context:
        return f"### 相关背景知识：\n{context}\n\n### 任务要求：\n{base_prompt}"
    return base_prompt


def build_image_prompt(config):
    """构建图件/图片表格抽取的提示词"""
    return (
        f"你是一位专业的区域地质调查和矿产勘查地质学家/表格数据分析助手。请仔细分析这张图片（可能包含表格或地质图件），"
        f"并严格按照以下 JSON 配置的定义和要求，从图片内容中抽取出关键信息，并以包含'表头'和'数据行'的表格形式返回。\n"
        f"配置说明：\n{config}\n\n"
        f"请确保返回结果严格按照Markdown表格格式，并且只包含抽取后的数据，不要包含任何额外的解释或说明文字。"
    )


def build_table_prompt(config, table_string):
    """根据配置和表格内容构建提示词"""
    return (
        f"你是一个表格数据分析助手。请根据以下配置，从提供的表格内容中抽取信息并按要求格式化输出。\n"
        f"配置：{config}\n\n"
        f"表格内容(Markdown格式):\n{table_string}\n\n"
        f"请确保返回结果严格按照Markdown表格格式，并且只包含抽取后的数据，不要包含任何额外的解释或说明文字。"
    )


def save_to_file(data, filename):
    """保存日志文件"""
    try:
        folder_path = os.path.join(os.getcwd(), "lithoextractlog")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))
    except Exception as e:
        logger.error("Failed to save log file %s: %s", sanitize(filename), sanitize(str(e)))


@litho_bp.route('/litho', methods=['POST'])
def process_request():
    logger.info("--- New Litho Request ---")
    client_data = request.get_json()
    logger.info("Received Raw JSON: %s...", sanitize(str(client_data)[:500]))

    question = client_data.get('text', '')
    selected_models = client_data.get('model', [])
    config = client_data.get("config", "")
    use_rag = client_data.get("useRAG", False)

    if not isinstance(selected_models, list):
        selected_models = [selected_models]

    valid_models = [model for model in selected_models if model in SUPPORTED_MODELS]
    if not valid_models:
        return jsonify({"results": [{"model": ", ".join(selected_models),
                                      "error": "后端未找到任何有效的文本模型。"}]})

    results = []
    final_prompt = build_prompt(config, question)

    for model in valid_models:
        try:
            response_text = get_model_response(final_prompt, model, client_data=client_data)
            results.append({"model": model, "result": response_text})
        except Exception as e:
            results.append({"model": model, "error": str(e)})

    logger.info("Finished processing.")
    save_to_file({"results": results}, f"litho_result_{int(time.time())}.json")
    return jsonify({"results": results})


@litho_bp.route('/litho-image', methods=['POST'])
def process_image_request():
    logger.info("--- New Litho Image Request ---")
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        selected_models_str = request.form.get('models', '[]')
        selected_models = json.loads(selected_models_str)
        config = request.form.get("config", "")
        client_data = request.form.to_dict()
    except Exception as e:
        return jsonify({"error": f"Failed to parse parameters: {e}"}), 400

    if not isinstance(selected_models, list) or not selected_models:
        return jsonify({"error": "Models parameter must be a non-empty list"}), 400

    try:
        buffered = BytesIO()
        file.save(buffered)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"Image encoding failed: {e}"}), 500

    final_prompt = build_image_prompt(config)
    results = []

    valid_multimodal_models = [model for model in selected_models if model in MULTIMODAL_MODELS]
    if not valid_multimodal_models:
        return jsonify({"results": [{"model": ", ".join(selected_models),
                                      "error": "后端未找到任何有效的多模态模型。"}]})

    for model_name in valid_multimodal_models:
        try:
            response_text = get_model_response(final_prompt, model_name, image_base64=img_base64,
                                               client_data=client_data)
            results.append({"model": model_name, "result": response_text})
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})

    save_to_file({"results": results}, f"image_result_{int(time.time())}.json")
    return jsonify({"results": results})


@litho_bp.route('/table-extract', methods=['POST'])
def process_table_request():
    logger.info("--- New Table Extract Request ---")
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' part"}), 400
    file = request.files['file']
    filename = file.filename
    if filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        selected_models_str = request.form.get('models', '[]')
        selected_models = json.loads(selected_models_str)
        config = request.form.get("config", "")
        client_data = request.form.to_dict()
    except Exception as e:
        return jsonify({"error": f"Failed to parse parameters: {e}"}), 400

    if not isinstance(selected_models, list) or not selected_models:
        return jsonify({"error": "Models parameter must be a non-empty list"}), 400

    results = []

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_base64 = None
        try:
            buffered = BytesIO()
            file.save(buffered)
            buffered.seek(0)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            return jsonify({"error": f"Image encoding failed: {e}"}), 500

        final_prompt = build_image_prompt(config)
        valid_models_for_image = [model for model in selected_models if model in MULTIMODAL_MODELS]
        if not valid_models_for_image:
            return jsonify({"results": [{"model": ", ".join(selected_models),
                                          "error": "未选择任何支持处理图片的多模态模型。"}]})
        for model_name in valid_models_for_image:
            try:
                response_text = get_model_response(final_prompt, model_name, image_base64=img_base64,
                                                   client_data=client_data)
                results.append({"model": model_name, "result": response_text})
            except Exception as e:
                results.append({"model": model_name, "error": str(e)})

    elif filename.lower().endswith(('.xlsx', '.csv')):
        table_string = ""
        try:
            file_content = file.read()
            file.seek(0)
            if filename.lower().endswith('.xlsx'):
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            else:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(BytesIO(file_content), encoding='gbk')
            table_string = df.to_markdown(index=False)
        except Exception as e:
            return jsonify({"results": [{"model": ", ".join(selected_models),
                                          "error": f"Reading/converting table file failed: {str(e)}"}]}), 500

        final_prompt = build_table_prompt(config, table_string)
        valid_models_for_text = [model for model in selected_models if model in SUPPORTED_MODELS]
        if not valid_models_for_text:
            return jsonify({"results": [{"model": ", ".join(selected_models),
                                          "error": "未选择任何支持处理文本表格的模型。"}]})
        for model_name in valid_models_for_text:
            try:
                response_text = get_model_response(final_prompt, model_name, client_data=client_data)
                results.append({"model": model_name, "result": response_text})
            except Exception as e:
                results.append({"model": model_name, "error": str(e)})
    else:
        return jsonify({"results": [{"model": ", ".join(selected_models),
                                      "error": f"Unsupported file type: {filename}"}]}), 400

    save_to_file({"results": results}, f"table_extract_result_{int(time.time())}.json")
    return jsonify({"results": results})
