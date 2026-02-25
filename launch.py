"""Модуль для последовательного запуска скриптов обработки данных."""

import subprocess
import sys


def run_script(script_name: str) -> None:
    """
    Запускает Python-скрипт и проверяет код его завершения.

    Args:
        script_name (str): Имя файла скрипта для запуска.
    """
    print(f"--- Запуск {script_name} ---")

    result = subprocess.run([sys.executable, script_name], check=False)

    if result.returncode == 0:
        print(f"--- {script_name} завершен успешно ---\n")
    else:
        print(f"!!! Ошибка при выполнении {script_name} !!!")
        sys.exit(1)


if __name__ == "__main__":
    run_script('api.py')
    run_script('processing.py')

    print("Вся цепочка действий выполнена!")
