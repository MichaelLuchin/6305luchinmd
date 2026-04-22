"""Модуль менеджера асинхронной обработки изображений."""

import asyncio
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, AsyncGenerator

import aiofiles
import aiohttp
import cv2
import numpy as np

from models import ArtworkMetadata, ColorArtwork
from utils import ValidatedPath, async_timer_decorator


def process_artwork_task(index: int, artwork: ColorArtwork) -> tuple[int, int, dict[str, bytes], dict]:
    """
    CPU-bound задача для выполнения в отдельном процессе пула.

    Осуществляет все матричные преобразования и сразу кодирует результат
    в байты (jpg), чтобы минимизировать нагрузку на главный асинхронный поток.
    """
    pid = os.getpid()
    obj_id = artwork.metadata.object_id
    print(f"[LOG] Convolution for image {index} (ID {obj_id}) started (PID {pid})")

    results_arrays = {}
    results_arrays['0_original.jpg'] = artwork.image

    gamma_img = artwork.apply_gamma(2.2)
    results_arrays['1_gamma_corrected.jpg'] = gamma_img

    hist_img = artwork.equalize_histogram()
    results_arrays['2_histogram_equalized.jpg'] = hist_img

    bw_artwork = artwork.to_grayscale()
    results_arrays['3_grayscale.jpg'] = bw_artwork.image

    blurred = bw_artwork.smooth(size=15, sigma=3.0)
    results_arrays['4_gaussian_blur.jpg'] = blurred

    edges = bw_artwork.extract_edges()
    results_arrays['5_sobel_edges.jpg'] = edges

    hist_obj = ColorArtwork(hist_img, artwork.metadata)
    blended = artwork + hist_obj
    results_arrays['6_blended_result.jpg'] = blended.image

    results_bytes = {}
    for filename, img_array in results_arrays.items():
        success, encoded = cv2.imencode('.jpg', img_array)
        if success:
            results_bytes[filename] = encoded.tobytes()

    print(f"[LOG] Convolution for image {index} (ID {obj_id}) finished (PID {pid})")
    return index, obj_id, results_bytes, artwork.metadata.raw_data


class AsyncImageProcessor:
    """
    Класс асинхронного генераторного пайплайна.

    Реализует паттерн Producer-Processor-Consumer.
    """
    save_dir = ValidatedPath()

    def __init__(self, save_dir: str = 'processed_artworks') -> None:
        """Инициализация процессора.

        Args:
            save_dir: Имя папки для сохранения файлов.
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _get_urls_to_download(self, count: int) -> list[tuple[int, int]]:
        """Парсит CSV и фиксирует порядковые номера."""
        paintings = []
        try:
            with open('MetObjects.csv', mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Classification') == 'Paintings':
                        paintings.append(row)
        except FileNotFoundError:
            print("[ERROR] Файл MetObjects.csv не найден!")
            return []

        if not paintings:
            return []

        targets = random.sample(paintings, min(count, len(paintings)))
        return [(i + 1, int(t['Object ID'])) for i, t in enumerate(targets)]

    async def _fetch_single(self, index: int, object_id: int, session: aiohttp.ClientSession, max_retries: int = 3) -> \
    tuple[int, ColorArtwork] | None:
        """Асинхронная загрузка одного изображения с повторными попытками при ошибках."""
        print(f"[LOG] Downloading image {index} started (ID {object_id})")
        base_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"

        for attempt in range(max_retries):
            try:
                async with session.get(base_url, timeout=15) as resp:
                    data = await resp.json()

                img_url = data.get('primaryImageSmall')
                if not img_url:
                    print(f"[WARN] У объекта {object_id} нет фото. Пропускаем.")
                    return None

                metadata = ArtworkMetadata(
                    object_id=object_id,
                    title=data.get("title", "Untitled"),
                    department=data.get("department", "Unknown"),
                    raw_data=data,
                )

                async with session.get(img_url, timeout=15) as img_resp:
                    img_bytes = await img_resp.read()

                img_array = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    print(f"[WARN] Не удалось декодировать изображение {object_id}")
                    return None

                print(f"[LOG] Downloading image {index} finished (ID {object_id})")
                return index, ColorArtwork(img_bgr, metadata)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"[WARN] Ошибка загрузки {object_id} (попытка {attempt + 1}/{max_retries}): {e}. Повтор через {wait_time} сек.")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[ERROR] Ошибка загрузки {object_id} после {max_retries} попыток: {e}")
                    return None
            except Exception as e:
                print(f"[ERROR] Неожиданная ошибка загрузки {object_id}: {e}")
                return None

        return None

    async def _check_done(self, futures: set, block: bool = False) -> tuple[set, set]:
        """Вспомогательный метод для ожидания завершения futures."""
        if not futures:
            return set(), set()

        if block:
            done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
        else:
            done, pending = await asyncio.wait(futures, timeout=0)
            if not done:
                await asyncio.sleep(0)

        return done, pending

    async def download_generator(
        self, object_ids: list[tuple[int, int]], session: aiohttp.ClientSession
    ) -> AsyncGenerator[tuple[int, ColorArtwork], None]:
        """Генератор №1: Асинхронно скачивает файлы и отдает их по мере готовности."""
        pending = {asyncio.create_task(self._fetch_single(idx, obj_id, session)) for idx, obj_id in object_ids}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                if result is not None:
                    yield result

    async def process_generator(
            self, download_stream: AsyncGenerator[tuple[int, ColorArtwork], None], executor: ProcessPoolExecutor
    ) -> AsyncGenerator[tuple[int, int, dict[str, bytes], dict], None]:
        """Генератор №2: Принимает скачанные объекты и параллельно распределяет их по ядрам."""
        loop = asyncio.get_running_loop()
        pending_futures = set()

        async for index, artwork in download_stream:
            future = loop.run_in_executor(executor, process_artwork_task, index, artwork)
            pending_futures.add(future)

            done, pending_futures = await self._check_done(pending_futures, block=False)
            for f in done:
                yield f.result()

        while pending_futures:
            done, pending_futures = await self._check_done(pending_futures, block=True)
            for f in done:
                yield f.result()

    async def save_consumer(self, process_stream: AsyncGenerator[tuple[int, int, dict[str, bytes], dict], None]) -> None:
        """Потребитель №3: Асинхронно сохраняет результаты на диск."""
        async for index, obj_id, results_bytes, raw_metadata in process_stream:
            print(f"[LOG] Saving for image {index} started")

            meta_path = os.path.join(self.save_dir, f"{index}_{obj_id}_metadata.json")
            async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(raw_metadata, indent=4, ensure_ascii=False))

            for filename, img_bytes in results_bytes.items():
                filepath = os.path.join(self.save_dir, f"{index}_{obj_id}_{filename}")
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(img_bytes)

            print(f"[LOG] Saving for image {index} finished")

    @async_timer_decorator
    async def run_pipeline(self, count: int) -> None:
        """Инициализатор пайплайна."""
        object_ids = self._get_urls_to_download(count)
        if not object_ids:
            print("[ERROR] Не удалось получить цели для скачивания.")
            return

        print(f"[INFO] Запланировано в пайплайн: {len(object_ids)} объектов.")

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            with ProcessPoolExecutor() as executor:
                download_stream = self.download_generator(object_ids, session)

                process_stream = self.process_generator(download_stream, executor)

                await self.save_consumer(process_stream)