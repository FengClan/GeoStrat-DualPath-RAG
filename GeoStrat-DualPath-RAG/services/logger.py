# services/logger.py — 统一日志模块（含日志伪造防护）
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

_initialized = False


def sanitize(s):
    """净化用户输入中的控制字符，防止日志伪造"""
    if s is None:
        return None
    if not isinstance(s, str):
        return str(s)
    return s.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')


def setup_logging(name="geology_platform", log_file="logs/python_service.log",
                  console_level=logging.INFO, file_level=logging.DEBUG):
    """初始化日志系统（幂等，多次调用无副作用）"""
    global _initialized
    if _initialized:
        return logging.getLogger(name)
    _initialized = True

    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    return logging.getLogger(name)
