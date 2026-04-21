"""Точка входа в приложение для выполнения Лабораторной работы №4."""

import argparse
import asyncio
import sys

from processor import AsyncImageProcessor


def main() -> None:
    """
    Основной сценарий:
    1. Парсинг аргументов командной строки.
    2. Инициализация асинхронного процессора.
    3. Запуск пайплайна (скачивание -> параллельная обработка -> асинхронное сохранение).
    """
    parser = argparse.ArgumentParser(description="Асинхронная обработка произведений искусства.")
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=3,
        help="Количество изображений для скачивания и обработки (по умолчанию 3)"
    )
    args = parser.parse_args()

    app = AsyncImageProcessor(save_dir='paintings_lab4')

    print("--- Запуск асинхронного пайплайна обработки ---")

    try:
        # Запускаем event loop
        asyncio.run(app.run_pipeline(args.count))
        print("--- Работа успешно завершена ---")
    except KeyboardInterrupt:
        print("\n[INFO] Выполнение прервано пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"[CRITICAL] Произошла непредвиденная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Защита точки входа важна для Windows, чтобы ProcessPoolExecutor не вошел в бесконечный цикл
    main()
