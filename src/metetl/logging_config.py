"""Модуль настройки логгера."""

import logging
import logging.config
import json
import os

CONFIG_FILE = "logging_config.json"

# Настройки по умолчанию
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        },
        "simple": {
            "format": "%(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "metetl": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}

def get_logger() -> logging.Logger:
    """Инициализирует логгер, используя JSON-конфиг или настройки по умолчанию."""

    os.makedirs("logs", exist_ok=True)

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            print(f"Ошибка загрузки {CONFIG_FILE}: {e}. Используем дефолтные настройки.")
            logging.config.dictConfig(DEFAULT_CONFIG)
    else:
        # Если файла нет — создаем его и применяем дефолтную конфигурацию
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            logging.config.dictConfig(DEFAULT_CONFIG)
        except Exception as e:
            print(f"Не удалось создать {CONFIG_FILE}: {e}")
            logging.config.dictConfig(DEFAULT_CONFIG)

    return logging.getLogger("metetl")

logger = get_logger()
