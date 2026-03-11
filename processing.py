"""Модуль для ручной и автоматической обработки изображений."""

import os
import time

import cv2

import numpy as np


def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Сглаживание (размытие) для подавления шумов.

    Args:
        size: Размер ядра (нечетное число).
        sigma: Степень размытия.

    Returns:
        Нормированное ядро Гаусса.
    """
    # одномерный массив расстояний от центра
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    # Создаём двумерные координатные сетки xx и yy
    xx, yy = np.meshgrid(ax, ax)
    # Вычисляем значение функции Гаусса
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    # Нормируем ядро
    kernel /= kernel.sum()
    return kernel


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


def manual_conv(
    image: np.ndarray,
    kernel: np.ndarray,
    clip_output: bool = True,
) -> np.ndarray:
    """
    Свёртка с использованием двумерной маски.

    Args:
        image: Исходное полутоновое изображение.
        kernel: Ядро свёртки (маска).
        clip_output: Флаг обрезки от 0 до 255.

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
            region = padded[row:row + k_h, col:col + k_w]
            output[row, col] = np.sum(region * kernel)

    if clip_output:
        return np.clip(output, 0, 255).astype(np.uint8)

    return output


def manual_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Гамма-коррекция изображения.

    Args:
        image: Исходное изображение.
        gamma: Коэффициент гаммы.

    Returns:
        Скорректированное изображение.
    """
    inv_gamma = 1.0 / gamma
    corrected = np.power(image / 255.0, inv_gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def cv2_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Гамма-коррекция изображения при помощи cv2.

    Args:
        image: Исходное изображение.
        gamma: Коэффициент гаммы.

    Returns:
        Скорректированное изображение.
    """
    inv_gamma = 1.0 / gamma
    corrected = cv2.pow(image / 255.0, inv_gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def manual_histogram_equalization(channel: np.ndarray) -> np.ndarray:
    """
    Выравнивание гистограммы для одного канала.

    Args:
        channel: Канал изображения (например, L из LAB).

    Returns:
        Канал с выровненной гистограммой.
    """
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[channel]


def process_lab_equalization(img_bgr: np.ndarray) -> np.ndarray:
    """
    Выравнивание гистограммы цветного изображения через LAB.

    Args:
        img_bgr: Исходное BGR изображение.

    Returns:
        Результат выравнивания.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_equalized = manual_histogram_equalization(l_channel)

    lab_res = np.stack([l_equalized, a_channel, b_channel], axis = 2)
    return cv2.cvtColor(lab_res, cv2.COLOR_LAB2BGR)


def sobel_manual(image: np.ndarray) -> np.ndarray:
    """
    Оператор Собеля для цветного изображения.

    Args:
        image: Исходное изображение (BGR или RGB).

    Returns:
        Магнитуда градиента для каждого канала.
    """
    image = image.astype(np.float32)

    chan_b = sobel_chanel(image[:, :, 0])
    chan_g = sobel_chanel(image[:, :, 1])
    chan_r = sobel_chanel(image[:, :, 2])

    return np.stack([chan_b, chan_g, chan_r], axis=2).astype(np.uint8)


def sobel_chanel(channel: np.ndarray) -> np.ndarray:
    """
    Оператор Собеля для одного канала.

    Args:
        channel: Одиночный цветовой канал.

    Returns:
        Магнитуда градиента (uint8).
    """
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    grad_x = manual_conv(channel, gx, clip_output=False).astype(np.float32)
    grad_y = manual_conv(channel, gy, clip_output=False).astype(np.float32)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255

    return magnitude.astype(np.uint8)


def sobel_cv2(image: np.ndarray) -> np.ndarray:
    """
    Реализация оператора Собеля через OpenCV.

    Args:
        image: Исходное цветное изображение (BGR).

    Returns:
        Магнитуда градиента.
    """
    img_float = image.astype(np.float32)

    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    for i in range(3):
        max_val = magnitude[:, :, i].max()
        if max_val > 0:
            magnitude[:, :, i] = (magnitude[:, :, i] / max_val) * 255

    return magnitude.astype(np.uint8)


def main() -> None:
    """
    Основная функция обработки изображений.
    """
    img_path = os.path.join('paintings', 'original.jpg')
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print("Ошибка: Изображение не найдено. Сначала запустите api.py")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    os.makedirs('paintings', exist_ok=True)

    # --- 1. GRAYSCALE ---
    t0 = time.time()
    gray_manual = manual_grayscale(img_rgb)
    print(f"Manual Gray: {time.time() - t0:.4f}s")

    t0 = time.time()
    gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"OpenCV Gray: {time.time() - t0:.4f}s")

    # --- 2. GAUSSIAN BLUR ---
    t0 = time.time()
    g_kernel = gaussian_kernel(21, 5.0)
    blur_manual = manual_conv(gray_manual, g_kernel)
    print(f"Manual Blur: {time.time() - t0:.4f}s")

    t0 = time.time()
    blur_cv = cv2.GaussianBlur(gray_cv, (21, 21), 5.0)
    print(f"OpenCV Blur: {time.time() - t0:.4f}s")

    # --- 3. SOBEL ---
    edges_manual = sobel_manual(img_bgr)

    t0 = time.time()
    edges_cv = sobel_cv2(img_bgr)
    print(f"OpenCV Sobel: {time.time() - t0:.4f}s")

    # --- 4. GAMMA CORRECTION ---
    t0 = time.time()
    gamma_manual = manual_gamma_correction(img_bgr, 2.2)
    print(f"Manual Gamma: {time.time() - t0:.4f}s")

    t0 = time.time()
    gamma_cv = cv2_gamma_correction(img_bgr, 2.2)
    print(f"OpenCV Gamma: {time.time() - t0:.4f}s")

    # --- 5. HISTOGRAM EQUALIZATION ---
    t0 = time.time()
    hist_manual = process_lab_equalization(img_bgr)
    print(f"Manual Hist Equalization: {time.time() - t0:.4f}s")

    t0 = time.time()
    lab_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab_cv)
    l_eq = cv2.equalizeHist(l_chan)
    hist_cv = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    print(f"OpenCV Hist Equalization: {time.time() - t0:.4f}s")

    # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
    cv2.imwrite(os.path.join('paintings', '1_gray_manual.jpg'), gray_manual)
    cv2.imwrite(os.path.join('paintings', '1_gray_opencv.jpg'), gray_cv)
    cv2.imwrite(os.path.join('paintings', '2_blur_manual.jpg'), blur_manual)
    cv2.imwrite(os.path.join('paintings', '2_blur_opencv.jpg'), blur_cv)
    cv2.imwrite(os.path.join('paintings', '3_edges_manual.jpg'), edges_manual)
    cv2.imwrite(os.path.join('paintings', '3_edges_opencv.jpg'), edges_cv)
    cv2.imwrite(os.path.join('paintings', '4_gamma_manual.jpg'), gamma_manual)
    cv2.imwrite(os.path.join('paintings', '4_gamma_opencv.jpg'), gamma_cv)
    cv2.imwrite(os.path.join('paintings', '5_hist_manual.jpg'), hist_manual)
    cv2.imwrite(os.path.join('paintings', '5_hist_opencv.jpg'), hist_cv)

    print("\nОбработка завершена.")


if __name__ == "__main__":
    main()
