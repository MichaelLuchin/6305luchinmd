"""Модуль моделей данных и иерархии классов изображений."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2

import numpy as np


@dataclass(slots=True)
class ArtworkMetadata:
    object_id: int
    title: str
    department: str
    raw_data: dict[str, Any]


class Artwork(ABC):
    def __init__(self: 'Artwork', image: np.ndarray, metadata: ArtworkMetadata) -> None:
        self._image = image
        self._metadata = metadata

    @property
    def image(self: 'Artwork') -> np.ndarray:
        return self._image

    @property
    def metadata(self: 'Artwork') -> ArtworkMetadata:
        return self._metadata

    def __str__(self: 'Artwork') -> str:
        return f"Artwork(ID: {self._metadata.object_id}, Title: '{self._metadata.title}')"

    def __add__(self: 'Artwork', other: 'Artwork') -> 'Artwork':
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
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def apply_gamma(self: 'Artwork', gamma: float = 1.0) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        corrected = np.power(self._image / 255.0, inv_gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    @abstractmethod
    def extract_edges(self: 'Artwork') -> np.ndarray:
        pass

    def smooth(self: 'Artwork', size: int = 15, sigma: float = 3.0) -> np.ndarray:
        kernel = self.get_gaussian_kernel(size, sigma)
        return self.apply_convolution(kernel)

    @abstractmethod
    def apply_convolution(self: 'Artwork', kernel: np.ndarray, clip_output: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def equalize_histogram(self: 'Artwork') -> np.ndarray:
        pass


class BlackAndWhiteArtwork(Artwork):
    def __init__(self: 'BlackAndWhiteArtwork', image: np.ndarray, metadata: ArtworkMetadata) -> None:
        if len(image.shape) != 2:
            raise ValueError("ЧБ изображение должно быть 2D массивом.")
        super().__init__(image, metadata)

    def apply_convolution(self: 'BlackAndWhiteArtwork', kernel: np.ndarray, clip_output: bool = True) -> np.ndarray:
        img_h, img_w = self._image.shape
        k_h, k_w = kernel.shape
        pad = k_h // 2
        padded = np.pad(self._image, pad, mode='constant')
        output = np.zeros_like(self._image, dtype=np.float32)

        for row in range(img_h):
            for col in range(img_w):
                region = padded[row:row + k_h, col:col + k_w]
                output[row, col] = np.sum(region * kernel)

        if clip_output:
            return np.clip(output, 0, 255).astype(np.uint8)
        return output

    def equalize_histogram(self: 'BlackAndWhiteArtwork') -> np.ndarray:
        hist, _ = np.histogram(self._image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[self._image]

    def extract_edges(self: 'BlackAndWhiteArtwork') -> np.ndarray:
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = self.apply_convolution(gx, clip_output=False).astype(np.float32)
        grad_y = self.apply_convolution(gy, clip_output=False).astype(np.float32)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255
        return magnitude.astype(np.uint8)


class ColorArtwork(Artwork):
    def __init__(self: 'ColorArtwork', image: np.ndarray, metadata: ArtworkMetadata) -> None:
        if len(image.shape) != 3:
            raise ValueError("Цветное изображение должно быть 3D массивом.")
        super().__init__(image, metadata)

    def apply_convolution(self: 'ColorArtwork', kernel: np.ndarray, clip_output: bool = True) -> np.ndarray:
        channels = cv2.split(self._image)
        processed_channels = []
        for ch in channels:
            temp_bw = BlackAndWhiteArtwork(ch, self._metadata)
            processed_channels.append(temp_bw.apply_convolution(kernel, clip_output=clip_output))
        return cv2.merge(processed_channels)

    def equalize_histogram(self: 'ColorArtwork') -> np.ndarray:
        lab = cv2.cvtColor(self._image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        temp_l = BlackAndWhiteArtwork(l_chan, self._metadata)
        l_eq = temp_l.equalize_histogram()

        lab_res = np.stack([l_eq, a_chan, b_chan], axis=2)
        return cv2.cvtColor(lab_res, cv2.COLOR_LAB2BGR)

    def to_grayscale(self: 'ColorArtwork') -> BlackAndWhiteArtwork:
        img_rgb = self._image[..., ::-1]
        gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return BlackAndWhiteArtwork(gray, self._metadata)

    def extract_edges(self: 'ColorArtwork') -> np.ndarray:
        temp_b = BlackAndWhiteArtwork(self._image[:, :, 0], self._metadata)
        temp_g = BlackAndWhiteArtwork(self._image[:, :, 1], self._metadata)
        temp_r = BlackAndWhiteArtwork(self._image[:, :, 2], self._metadata)

        chan_b = temp_b.extract_edges()
        chan_g = temp_g.extract_edges()
        chan_r = temp_r.extract_edges()

        return np.stack([chan_b, chan_g, chan_r], axis=2).astype(np.uint8)