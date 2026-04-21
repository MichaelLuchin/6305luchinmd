"""Модуль вспомогательных инструментов: декораторы и дескрипторы."""

import functools
import time
from typing import Any, Callable


def timer_decorator(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения функции.

    Args:
        func: Функция, которую нужно обернуть.

    Returns:
        Обернутая функция.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Вспомогательная функция-обертка.

        Args:
            args: Позиционные аргументы.
            kwargs: Именованные аргументы.

        Returns:
            Результат выполнения функции.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"[LOG] Функция '{func.__name__}' выполнена за {duration:.4f} сек.")
        return result
    return wrapper

def async_timer_decorator(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения асинхронной функции."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"[LOG] Асинхронный пайплайн '{func.__name__}' выполнен за {duration:.4f} сек.")
        return result
    return wrapper


class ValidatedPath:
    """Дескриптор для проверки строковых путей.

    Гарантирует, что путь для сохранения является строкой.
    """

    def __set_name__(self: 'ValidatedPath', owner: Any, name: str) -> None:
        """Устанавливает имя приватного атрибута.

        Args:
            owner: Класс-владелец дескриптора.
            name: Имя атрибута.
        """
        self._private_name = '_' + name

    def __get__(self: 'ValidatedPath', instance: Any, owner: Any) -> Any:
        """Возвращает значение атрибута.

        Args:
            instance: Экземпляр класса.
            owner: Класс-владелец.

        Returns:
            Значение атрибута или сам дескриптор.
        """
        if instance is None:
            return self

        return getattr(instance, self._private_name)

    def __set__(self: 'ValidatedPath', instance: Any, value: Any) -> None:
        """Проверяет и устанавливает значение пути.

        Args:
            instance: Экземпляр класса.
            value: Устанавливаемое значение.

        Raises:
            TypeError: Если значение не строка.
            ValueError: Если строка пустая.
        """
        if not isinstance(value, str):
            raise TypeError(f"Путь должен быть строкой (str), получено: {type(value).__name__}")
        if not value.strip():
            raise ValueError("Путь не может быть пустой строкой.")

        setattr(instance, self._private_name, value)
