"""Модуль настройки логгера."""

import logging
import os


def get_logger() -> logging.Logger:
    """Создает и настраивает логгер для проекта.

    В файл пишутся логи уровня DEBUG, в консоль - INFO.
    """
    logger = logging.getLogger("metetl")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)

    # Файловый хэндлер (DEBUG)
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    fmt_file = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(fmt_file)

    # Консольный хэндлер (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    fmt_console = logging.Formatter("%(message)s")
    console_handler.setFormatter(fmt_console)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = get_logger()