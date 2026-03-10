"""Точка входа в приложение для выполнения Лабораторной работы №2."""

import sys

from processor import ImageProcessor


def main() -> None:
    """
    Основной сценарий:
    1. Инициализация процессора с указанием директории.
    2. Скачивание случайного произведения.
    3. Запуск автоматической обработки и сохранения.
    """
    # Создаем экземпляр управляющего класса
    # Путь пройдет через дескриптор ValidatedPath для проверки
    app = ImageProcessor(save_dir='lab2_results')

    print("--- Запуск процесса обработки ---")

    try:
        # 1. Скачиваем (метод задекорирован @timer_decorator)
        artwork = app.download_artwork()

        if artwork:
            # Вывод информации через перегруженный метод __str__
            print(f"[INFO] Объект готов: {artwork}")

            # 2. Обрабатываем (метод задекорирован @timer_decorator)
            app.process_and_save_all(artwork)

            print("--- Работа успешно завершена ---")
        else:
            print("[ERROR] Не удалось получить данные о картине.")
            sys.exit(1)

    except Exception as e:
        print(f"[CRITICAL] Произошла непредвиденная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
