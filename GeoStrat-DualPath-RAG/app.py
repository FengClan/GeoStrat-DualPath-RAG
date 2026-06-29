# app.py — Flask 应用工厂，注册所有服务蓝图
import os
import logging
from flask import Flask
from flask_cors import CORS

from services.config import UPLOAD_FOLDER, GRAPH_STORAGE_DIR
from services.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 确保必要的目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(GRAPH_STORAGE_DIR, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # 注册服务蓝图 (所有API路由前缀 /api)
    from services.extraction_service import extraction_bp
    app.register_blueprint(extraction_bp, url_prefix='/api')

    from services.litho_service import litho_bp
    app.register_blueprint(litho_bp, url_prefix='/api')

    from services.kg_service import kg_bp
    app.register_blueprint(kg_bp, url_prefix='/api')

    from services.rag_service import rag_bp
    app.register_blueprint(rag_bp, url_prefix='/api')

    from services.strat_service import strat_bp
    app.register_blueprint(strat_bp, url_prefix='/api')

    from services.kg_extraction import kg_extraction_bp
    app.register_blueprint(kg_extraction_bp, url_prefix='/api')

    from services.entity_alignment import entity_alignment_bp
    app.register_blueprint(entity_alignment_bp, url_prefix='/api')

    from services.reranker import reranker_bp
    app.register_blueprint(reranker_bp, url_prefix='/api')

    from services.query_classifier import query_classifier_bp
    app.register_blueprint(query_classifier_bp, url_prefix='/api')

    from services.cross_region_compare import cross_region_bp
    app.register_blueprint(cross_region_bp, url_prefix='/api')

    from services.eval_service import eval_bp
    app.register_blueprint(eval_bp, url_prefix='/api')

    from services.pipeline_orchestrator import pipeline_bp
    app.register_blueprint(pipeline_bp, url_prefix='/api')

    return app


app = create_app()

if __name__ == '__main__':
    print("[INFO] Starting Geo-Platform Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
