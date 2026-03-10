"""Модуль моделей данных и иерархии классов изображений."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(slots=True)
class ArtworkMetadata:
    """Инкапсуляция метаданных картины."""
    object_id: int
    title: str
    department: str
    raw_data: dict


class Artwork(ABC):
    """Абстрактный базовый класс для работы с изображениями."""

    def __init__(self, image: np.ndarray, metadata: ArtworkMetadata) -> None:
        self._image = image
        self._metadata = metadata

    @property
    def image(self) -> np.ndarray:
        """Доступ к массиву изображения."""
        return self._image

    @property
    def metadata(self) -> ArtworkMetadata:
        """Доступ к метаданным."""
        return self._metadata

    def __str__(self) -> str:
        """Преобразование объекта в строку."""
        return f"Artwork(ID: {self._metadata.object_id}, Title: '{self._metadata.title}')"

    def __add__(self, other: 'Artwork') -> 'Artwork':
        """Перегрузка сложения (наложение изображений 50/50)."""
        if not isinstance(other, Artwork):
            raise TypeError("Складывать можно только объекты Artwork.")

        img1 = self._image
        img2 = other.image

        # Приведение к одному размеру, если они разные
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Смешивание
        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        return self.__class__(blended, self._metadata)

    @staticmethod
    def get_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
        """Генерация ядра Гаусса (из твоей Лаб №1)."""
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def apply_gamma(self, gamma: float = 1.0) -> np.ndarray:
        """Ручная гамма-коррекция (из твоей Лаб №1)."""
        inv_gamma = 1.0 / gamma
        # Нормировка -> Степень -> Возврат к 0-255
        corrected = np.power(self._image / 255.0, inv_gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def extract_edges(self) -> np.ndarray:
        """Выделение границ оператором Собеля."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return self.apply_convolution(sobel_x)

    def smooth(self, size: int = 15, sigma: float = 3.0) -> np.ndarray:
        """Сглаживание фильтром Гаусса."""
        kernel = self.get_gaussian_kernel(size, sigma)
        return self.apply_convolution(kernel)

    @abstractmethod
    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Абстрактный метод свертки (полиморфизм)."""
        pass

    @abstractmethod
    def equalize_histogram(self) -> np.ndarray:
        """Абстрактный метод выравнивания гистограммы (полиморфизм)."""
        pass


class BlackAndWhiteArtwork(Artwork):
    """Класс для полутоновых (ЧБ) изображений."""

    def __init__(self, image: np.ndarray, metadata: ArtworkMetadata) -> None:
        if len(image.shape) != 2:
            raise ValueError("ЧБ изображение должно быть 2D массивом.")
        super().__init__(image, metadata)

    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Ручная свертка для одного канала (твой алгоритм из Лаб №1)."""
        img_h, img_w = self._image.shape
        k_h, k_w = kernel.shape
        pad = k_h // 2
        padded = np.pad(self._image, pad, mode='constant')
        output = np.zeros_like(self._image, dtype=np.float32)

        for i in range(img_h):
            for j in range(img_w):
                region = padded[i:i + k_h, j:j + k_w]
                output[i, j] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)

    def equalize_histogram(self) -> np.ndarray:
        """Ручное выравнивание гистограммы ЧБ (твой алгоритм через CDF)."""
        hist, _ = np.histogram(self._image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[self._image]


class ColorArtwork(Artwork):
    """Класс для цветных (BGR) изображений."""

    def __init__(self, image: np.ndarray, metadata: ArtworkMetadata) -> None:
        if len(image.shape) != 3:
            raise ValueError("Цветное изображение должно быть 3D массивом.")
        super().__init__(image, metadata)

    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Поканальная ручная свертка для BGR."""
        channels = cv2.split(self._image)
        processed_channels = []
        # Используем логику свертки для каждого канала
        for ch in channels:
            # Создаем временный объект ЧБ для вызова его реализации свертки
            temp_bw = BlackAndWhiteArtwork(ch, self._metadata)
            processed_channels.append(temp_bw.apply_convolution(kernel))
        return cv2.merge(processed_channels)

    def equalize_histogram(self) -> np.ndarray:
        """Выравнивание через LAB (твой алгоритм из Лаб №1)."""
        lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Выравниваем L-канал используя логику из ЧБ
        temp_l = BlackAndWhiteArtwork(l, self._metadata)
        l_eq = temp_l.equalize_histogram()

        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    def to_grayscale(self) -> BlackAndWhiteArtwork:
        """Ручной перевод в ЧБ (твоя формула 0.299R + 0.587G + 0.114B)."""
        # В OpenCV порядок BGR
        b, g, r = cv2.split(self._image)
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        return BlackAndWhiteArtwork(gray, self._metadata)
