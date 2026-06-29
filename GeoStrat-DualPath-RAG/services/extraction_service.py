# services/extraction_service.py — 地质信息抽取: NER/RE/STE/PE/theme/wordrec/triple
import time, json
from flask import Blueprint, request, jsonify

from .config import TASK_PROMPT
from .model_router import get_model_response

extraction_bp = Blueprint('extraction', __name__)

DEFAULT_MODEL = "qwen3.5-plus"


def execute_task(task_type, data):
    ques = data.get('text', '')
    try:
        prompt = f'{ques}, {TASK_PROMPT[task_type]}'
        generated_text = get_model_response(prompt, DEFAULT_MODEL, client_data=data)
        return jsonify({"result": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@extraction_bp.route('/wordrec', methods=['POST'])
def wordrec(): return execute_task("wordrec", request.json)


@extraction_bp.route('/ner', methods=['POST'])
def ner(): return execute_task("ner", request.json)


@extraction_bp.route('/ree', methods=['POST'])
def ree(): return execute_task("ree", request.json)


@extraction_bp.route('/theme', methods=['POST'])
def theme(): return execute_task("theme", request.json)


@extraction_bp.route('/pe', methods=['POST'])
def pe(): return execute_task("pe", request.json)


@extraction_bp.route('/ste', methods=['POST'])
def ste(): return execute_task("ste", request.json)


@extraction_bp.route('/extract', methods=['POST'])
def extract():
    data = request.json
    input_text = data.get('text', '')
    selected_features = data.get('features', [])
    try:
        prompt = f"基于以下句子进行三元组抽取（抽取的结果不要有序号）...：{', '.join(selected_features)}。\n输入文本：{input_text}"
        generated_text = get_model_response(prompt, DEFAULT_MODEL, client_data=data)
        return jsonify({"result": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
