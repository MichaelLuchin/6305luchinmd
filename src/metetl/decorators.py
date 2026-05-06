"""Модуль вспомогательных инструментов: декораторы и дескрипторы."""

import functools
import time
from typing import Any, Callable

from metetl.logging_config import logger


def timer_decorator(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения функции."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.debug(f"[LOG] Функция '{func.__name__}' выполнена за {duration:.4f} сек.")
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
        logger.debug(f"[LOG] Асинхронный пайплайн '{func.__name__}' выполнен за {duration:.4f} сек.")
        return result
    return wrapper


class ValidatedPath:
    """Дескриптор для проверки строковых путей."""

    def __set_name__(self: 'ValidatedPath', owner: Any, name: str) -> None:
        self._private_name = '_' + name

    def __get__(self: 'ValidatedPath', instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return getattr(instance, self._private_name)

    def __set__(self: 'ValidatedPath', instance: Any, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Путь должен быть строкой, получено: {type(value).__name__}")
        if not value.strip():
            raise ValueError("Путь не может быть пустой строкой.")

        setattr(instance, self._private_name, value)