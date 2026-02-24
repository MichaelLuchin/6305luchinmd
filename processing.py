"""Модуль для ручной и автоматической обработки изображений."""

import os
import time

import cv2

import numpy as np


def manual_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """
    Приведение цветного изображения к полутоновому.
    Формула: 0.299*R + 0.587*G + 0.114*B.

    Args:
        img_rgb: Исходное цветное изображение (RGB массива).

    Returns:
        Полутоновое изображение.
    """
    return np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def manual_conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Свёртка с использованием двумерной маски.

    Args:
        image: Исходное полутоновое изображение.
        kernel: Ядро свёртки (маска).

    Returns:
        Изображение после применения свёртки.
    """
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    # Создаем отступы вокруг изображения
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for row in range(i_h):
        for col in range(i_w):
            region = padded[row:row+k_h, col:col+k_w]
            output[row, col] = np.clip(np.sum(region * kernel), 0, 255)

    return output.astype(np.uint8)


def main() -> None:
    """
    Основная функция обработки.

    Считывает изображение, применяет самописные и встроенные фильтры,
    замеряет время выполнения и сохраняет результаты.
    """
    img_path = os.path.join('paintings', 'original.jpg')
    img = cv2.imread(img_path)
    if img is None:
        return

    # Используем cv2 для конвертации
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- GRAYSCALE ---
    t0 = time.time()
    gray_manual = manual_grayscale(img_rgb)
    print(f"Manual Gray: {time.time() - t0:.4f}s")  # Подсчёт времени

    t0 = time.time()
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #
    print(f"OpenCV Gray: {time.time() - t0:.4f}s")

    # --- GAUSSIAN BLUR (Сглаживание) ---
    gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    blur_manual = manual_conv(gray_manual, gauss_kernel)

    # --- SOBEL (Выделение границ) ---
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edges_manual = manual_conv(blur_manual, sobel_x)

    # Сохранение результатов
    cv2.imwrite(os.path.join('paintings', 'gray_manual.jpg'), gray_manual)
    cv2.imwrite(os.path.join('paintings', 'edges_manual.jpg'), edges_manual)
    print("Результаты обработки сохранены.")


if __name__ == "__main__":
    main()
