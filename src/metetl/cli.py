"""Модуль обработки консольного интерфейса."""

import argparse
import asyncio

from metetl.analysis.aggregations import run_analysis
from metetl.analysis.data_to_download import prepare_metadata
from metetl.images.processing import AsyncImageProcessor
from metetl.logging_config import logger


def main() -> None:
    """Инициализация CLI и маппинг команд."""
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="Пайплайн загрузки и обработки данных Метрополитен-музея"
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # 1. prepare
    prep_parser = subparsers.add_parser("prepare", help="Подготовка json файла с метаданными")
    prep_parser.add_argument("--csv", required=True, help="Путь к CSV файлу")
    prep_parser.add_argument("--output", required=True, help="Путь к выходному JSON")

    # 2. process
    proc_parser = subparsers.add_parser("process", help="Запуск пайплайна обработки")
    proc_parser.add_argument("--input", required=True, help="Путь к JSON с метаданными")
    proc_parser.add_argument("--output", required=True, help="Директория для сохранения")
    proc_parser.add_argument("--num", required=True, type=int, help="Кол-во изображений")

    # 3. analyze
    ana_parser = subparsers.add_parser("analyze", help="Анализ датасета")
    ana_parser.add_argument("--csv", required=True, help="Путь к CSV файлу")
    ana_parser.add_argument("--output-dir", required=True, help="Директория для графиков")

    args = parser.parse_args()

    if args.command == "prepare":
        logger.info("--- Старт подготовки метаданных ---")
        prepare_metadata(args.csv, args.output)
        logger.info("--- Подготовка завершена ---")

    elif args.command == "process":
        logger.info("--- Старт асинхронной обработки изображений ---")
        app = AsyncImageProcessor(save_dir=args.output)
        try:
            asyncio.run(app.run_pipeline(args.input, args.num))
        except KeyboardInterrupt:
            logger.info("Выполнение прервано пользователем.")
        logger.info("--- Обработка завершена ---")

    elif args.command == "analyze":
        logger.info("--- Старт анализа датасета ---")
        run_analysis(args.csv, args.output_dir)
        logger.info("--- Анализ завершен ---")

    else:
        parser.print_help()