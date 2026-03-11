"""Модуль моделей данных и иерархии классов изображений."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2

import numpy as np


@dataclass(slots=True)
class ArtworkMetadata:
    """Инкапсуляция метаданных картины.

    Attributes:
        object_id: Уникальный идентификатор объекта.
        title: Название произведения.
        department: Отдел музея.
        raw_data: Исходные данные в формате словаря.
    """
    object_id: int
    title: str
    department: str
    raw_data: dict[str, Any]


class Artwork(ABC):
    """Абстрактный базовый класс для работы с изображениями.

    Attributes:
        image: Массив пикселей изображения.
        metadata: Метаданные произведения.
    """

    def __init__(self: 'Artwork', image: np.ndarray, metadata: ArtworkMetadata) -> None:
        """Инициализация объекта Artwork.

        Args:
            image: Массив numpy с данными изображения.
            metadata: Объект метаданных.
        """
        self._image = image
        self._metadata = metadata

    @property
    def image(self: 'Artwork') -> np.ndarray:
        """Доступ к массиву изображения.

        Returns:
            Массив numpy с данными изображения.
        """
        return self._image

    @property
    def metadata(self: 'Artwork') -> ArtworkMetadata:
        """Доступ к метаданным.

        Returns:
            Объект ArtworkMetadata.
        """
        return self._metadata

    def __str__(self: 'Artwork') -> str:
        """Преобразование объекта в строку.

        Returns:
            Строковое представление объекта.
        """
        return f"Artwork(ID: {self._metadata.object_id}, Title: '{self._metadata.title}')"

    def __add__(self: 'Artwork', other: 'Artwork') -> 'Artwork':
        """Перегрузка сложения (наложение изображений 50/50).

        Args:
            other: Другой объект Artwork для наложения.

        Returns:
            Новый объект Artwork того же класса.

        Raises:
            TypeError: Если второй объект не является экземпляром Artwork.
        """
        if not isinstance(other, Artwork):
            raise TypeError("Складывать можно только объекты Artwork.")

        img1 = self._image
        img2 = other.image

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if (h1, w1) != (h2, w2):
            row_indices = (np.arange(h1) * (h2 / h1)).astype(np.intp)
            col_indices = (np.arange(w1) * (w2 / w1)).astype(np.intp)

            img2 = img2[row_indices[:, np.newaxis], col_indices]

        blended = (img1.astype(np.float32) * 0.5 + img2.astype(np.float32) * 0.5)

        blended = blended.astype(np.uint8)
        return self.__class__(blended, self._metadata)

    @staticmethod
    def get_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
        """Генерация ядра Гаусса.

        Args:
            size: Размер ядра.
            sigma: Коэффициент отклонения.

        Returns:
            Массив numpy, представляющий ядро Гаусса.
        """
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def apply_gamma(self: 'Artwork', gamma: float = 1.0) -> np.ndarray:
        """Ручная гамма-коррекция.

        Args:
            gamma: Значение гаммы.

        Returns:
            Обработанный массив изображения.
        """
        inv_gamma = 1.0 / gamma
        corrected = np.power(self._image / 255.0, inv_gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def extract_edges(self: 'Artwork') -> np.ndarray:
        """Выделение границ оператором Собеля.

        Returns:
            Изображение с выделенными границами.
        """
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return self.apply_convolution(sobel_x)

    def smooth(self: 'Artwork', size: int = 15, sigma: float = 3.0) -> np.ndarray:
        """Сглаживание фильтром Гаусса.

        Args:
            size: Размер ядра.
            sigma: Отклонение.


        Returns:
            Сглаженное изображение.
        """
        kernel = self.get_gaussian_kernel(size, sigma)
        return self.apply_convolution(kernel)

    @abstractmethod
    def apply_convolution(self: 'Artwork', kernel: np.ndarray) -> np.ndarray:
        """Абстрактный метод свертки.

        Args:
            kernel: Ядро свертки.
        """
        pass

    @abstractmethod
    def equalize_histogram(self: 'Artwork') -> np.ndarray:
        """Абстрактный метод выравнивания гистограммы."""
        pass


class BlackAndWhiteArtwork(Artwork):
    """Класс для полутоновых (ЧБ) изображений."""

    def __init__(
        self: 'BlackAndWhiteArtwork',
        image: np.ndarray,
        metadata: ArtworkMetadata,
    ) -> None:
        """Инициализация ЧБ изображения.

        Args:
            image: 2D массив.
            metadata: Метаданные.

        Raises:
            ValueError: Если передан не 2D массив.
        """
        if len(image.shape) != 2:
            raise ValueError("ЧБ изображение должно быть 2D массивом.")

        super().__init__(image, metadata)

    def apply_convolution(self: 'BlackAndWhiteArtwork', kernel: np.ndarray) -> np.ndarray:
        """Ручная свертка для одного канала.

        Args:
            kernel: Ядро свертки.

        Returns:
            Результат свертки.
        """
        img_h, img_w = self._image.shape
        k_h, k_w = kernel.shape
        pad = k_h // 2
        padded = np.pad(self._image, pad, mode='constant')
        output = np.zeros_like(self._image, dtype=np.float32)

        for row_idx in range(img_h):
            for col_idx in range(img_w):
                region = padded[row_idx:row_idx + k_h, col_idx:col_idx + k_w]
                output[row_idx, col_idx] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)

    def equalize_histogram(self: 'BlackAndWhiteArtwork') -> np.ndarray:
        """Ручное выравнивание гистограммы ЧБ (алгоритм через CDF).

        Returns:
            Изображение с улучшенным контрастом.
        """
        hist, _ = np.histogram(self._image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[self._image]


class ColorArtwork(Artwork):
    """Класс для цветных (BGR) изображений."""

    def __init__(self: 'ColorArtwork',
                 image: np.ndarray, metadata: ArtworkMetadata) -> None:
        """Инициализация цветного изображения.

        Args:
            image: 3D массив.
            metadata: Метаданные.

        Raises:
            ValueError: Если передан не 3D массив.
        """
        if len(image.shape) != 3:
            raise ValueError("Цветное изображение должно быть 3D массивом.")

        super().__init__(image, metadata)

    def apply_convolution(self: 'ColorArtwork', kernel: np.ndarray) -> np.ndarray:
        """Поканальная ручная свертка для BGR.

        Args:
            kernel: Ядро свертки.

        Returns:
            Обработанное цветное изображение.
        """
        channels = cv2.split(self._image)
        processed_channels = []
        for ch in channels:
            temp_bw = BlackAndWhiteArtwork(ch, self._metadata)
            processed_channels.append(temp_bw.apply_convolution(kernel))

        return cv2.merge(processed_channels)

    def equalize_histogram(self: 'ColorArtwork') -> np.ndarray:
        """Выравнивание через LAB (используя яркостный канал L).

        Returns:
            Изображение с выровненной яркостью.
        """
        lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        temp_l = BlackAndWhiteArtwork(l_chan, self._metadata)
        l_eq = temp_l.equalize_histogram()

        return cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    def to_grayscale(self: 'ColorArtwork') -> BlackAndWhiteArtwork:
        """Ручной перевод в ЧБ (0.299R + 0.587G + 0.114B).


        Returns:
            Новый объект BlackAndWhiteArtwork.
        """

        blue_ch, green_ch, red_ch = cv2.split(self._image)
        # Коэффициенты для перевода в оттенки серого
        gray = (0.299 * red_ch + 0.587 * green_ch + 0.114 * blue_ch).astype(np.uint8)
        return BlackAndWhiteArtwork(gray, self._metadata)
