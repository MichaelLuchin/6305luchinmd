"""Модуль вспомогательных инструментов: декораторы и дескрипторы."""

import functools
import time
from typing import Any, Callable


def timer_decorator(func: Callable) -> Callable:
    """
    Декоратор для замера времени выполнения функции.
    Выводит в консоль имя функции и затраченное время.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"[LOG] Функция '{func.__name__}' выполнена за {duration:.4f} сек.")
        return result
    return wrapper


class ValidatedPath:
    """
    Дескриптор для проверки строковых путей.
    Гарантирует, что путь для сохранения является строкой.
    """
    def __set_name__(self, owner: Any, name: str) -> None:
        self._private_name = '_' + name

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return getattr(instance, self._private_name)

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"Путь должен быть строкой (str), получено: {type(value).__name__}"
            )
        if not value.strip():
            raise ValueError("Путь не может быть пустой строкой.")
        setattr(instance, self._private_name, value)
