"""Модуль менеджера обработки изображений."""

import csv
import json
import os
import random

import cv2

from models import ArtworkMetadata, ColorArtwork

import numpy as np

import requests

from utils import ValidatedPath, timer_decorator


class ImageProcessor:
    """Класс для управления процессом загрузки и обработки произведений искусства.

    Инкапсулирует логику взаимодействия с API и файловой системой.

    Attributes:
        save_dir: Путь к директории для сохранения результатов.
    """

    # Использование дескриптора для валидации пути сохранения
    save_dir = ValidatedPath()

    def __init__(self: 'ImageProcessor', save_dir: str = 'paintings') -> None:
        """Инициализация процессора.

        Args:
            save_dir: Имя папки для сохранения файлов.
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @timer_decorator
    def download_artwork(self: 'ImageProcessor') -> ColorArtwork | None:
        """Выбирает случайный объект из CSV и загружает данные через API.

        Returns:
            Экземпляр ColorArtwork или None в случае ошибки.
        """
        print("[LOG] Поиск случайного произведения в MetObjects.csv...")

        paintings = []
        try:
            with open('MetObjects.csv', mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Classification') == 'Paintings':
                        paintings.append(row)
        except FileNotFoundError:
            print("[ERROR] Файл MetObjects.csv не найден!")
            return None

        if not paintings:
            print("[ERROR] Список картин пуст.")
            return None

        target = random.choice(paintings)
        object_id = target['Object ID']
        base_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
        api_url = f"{base_url}{object_id}"

        try:
            response = requests.get(api_url, timeout=15).json()
            img_url = response.get('primaryImageSmall')

            if not img_url:
                print(f"[WARN] У объекта {object_id} нет фото. Пробую еще раз...")
                return self.download_artwork()

            # Создание объекта метаданных (Dataclass)
            metadata = ArtworkMetadata(
                object_id=int(response.get("objectID", 0)),
                title=response.get("title", "Untitled"),
                department=response.get("department", "Unknown"),
                raw_data=response,
            )

            # Загрузка самого изображения
            img_data = requests.get(img_url, timeout=15).content
            img_array = np.frombuffer(img_data, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            print(f"[LOG] Загружено: {metadata.title}")
            return ColorArtwork(img_bgr, metadata)

        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке: {e}")
            return None

    @timer_decorator
    def process_and_save_all(self: 'ImageProcessor', artwork: ColorArtwork) -> None:
        """Выполняет цикл обработки и сохраняет результаты на диск.

        Args:
            artwork: Объект цветного изображения для обработки.
        """
        print(f"[LOG] Начинаю обработку объекта: {artwork.metadata.title}")

        # 1. Сохранение оригинальных метаданных в JSON
        meta_path = os.path.join(self.save_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(artwork.metadata.raw_data, f, indent=4, ensure_ascii=False)

        # 2. Сохранение оригинала (из свойства .image)
        cv2.imwrite(os.path.join(self.save_dir, '0_original.jpg'), artwork.image)

        # 3. Гамма-коррекция (на цветном)
        gamma_img = artwork.apply_gamma(2.2)
        cv2.imwrite(os.path.join(self.save_dir, '1_gamma_corrected.jpg'), gamma_img)

        # 4. Выравнивание гистограммы (цветное через LAB)
        hist_img = artwork.equalize_histogram()
        cv2.imwrite(os.path.join(self.save_dir, '2_histogram_equalized.jpg'), hist_img)

        # 5. Преобразование в ЧБ (создание нового типа объекта)
        bw_artwork = artwork.to_grayscale()
        cv2.imwrite(os.path.join(self.save_dir, '3_grayscale.jpg'), bw_artwork.image)

        # 6. Сглаживание Гауссом (на ЧБ объекте)
        blurred = bw_artwork.smooth(size=15, sigma=3.0)
        cv2.imwrite(os.path.join(self.save_dir, '4_gaussian_blur.jpg'), blurred)

        # 7. Выделение границ Собелем (на ЧБ объекте)
        edges = bw_artwork.extract_edges()
        cv2.imwrite(os.path.join(self.save_dir, '5_sobel_edges.jpg'), edges)

        # 8. Демонстрация перегрузки оператора +
        hist_obj = ColorArtwork(hist_img, artwork.metadata)
        blended_artwork = artwork + hist_obj
        cv2.imwrite(
            os.path.join(self.save_dir, '6_blended_result.jpg'),
            blended_artwork.image,
        )

        print(f"[LOG] Все файлы сохранены в директорию: {self.save_dir}")
