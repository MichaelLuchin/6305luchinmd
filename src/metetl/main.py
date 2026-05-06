"""Точка входа приложения."""

from metetl.cli import main as cli_main
from metetl.logging_config import logger


def main() -> None:
    """Основной запуск программы через CLI."""
    try:
        cli_main()
    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    main()