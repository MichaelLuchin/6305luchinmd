"""Модуль менеджера асинхронной обработки изображений."""

import asyncio
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncGenerator

import aiofiles

import aiohttp

import cv2

import numpy as np

from metetl.decorators import ValidatedPath, async_timer_decorator
from metetl.images.models import ArtworkMetadata, ColorArtwork
from metetl.logging_config import logger


def process_artwork_task(index: int, artwork: ColorArtwork) -> tuple[int, int, dict[str, bytes], dict]:
    """CPU-bound задача для выполнения в отдельном процессе пула."""
    pid = os.getpid()
    obj_id = artwork.metadata.object_id
    logger.debug(f"[PROCESS] PID {pid} | Обработка {index} началась (ID {obj_id})")

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

    logger.debug(f"[PROCESS] PID {pid} | Обработка {index} завершена (ID {obj_id})")
    return index, obj_id, results_bytes, artwork.metadata.raw_data


class AsyncImageProcessor:
    """Класс асинхронного генераторного пайплайна."""
    save_dir = ValidatedPath()

    def __init__(self, save_dir: str = 'processed_artworks') -> None:
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.main_pid = os.getpid()
        logger.debug(f"[MAIN] PID главного процесса: {self.main_pid}")

    def _get_urls_to_download(self, input_json: str, count: int) -> list[tuple[int, int]]:
        """Парсит JSON с подготовленными метаданными и выбирает цели."""
        try:
            with open(input_json, mode='r', encoding='utf-8') as f:
                paintings = json.load(f)
        except FileNotFoundError:
            logger.error(f"[ERROR] Файл {input_json} не найден!")
            return []

        if not paintings:
            return []

        targets = random.sample(paintings, min(count, len(paintings)))
        return [(i + 1, int(t['Object ID'])) for i, t in enumerate(targets)]

    async def _fetch_single(self, index: int, object_id: int, session: aiohttp.ClientSession, max_retries: int = 3) -> tuple[int, ColorArtwork] | None:
        pid = os.getpid()
        logger.debug(f"[ASYNC] PID {pid} | Скачивание {index} началось (ID {object_id})")
        base_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"

        for attempt in range(max_retries):
            try:
                async with session.get(base_url, timeout=15) as resp:
                    data = await resp.json()

                img_url = data.get('primaryImageSmall')
                if not img_url:
                    logger.debug(f"[ASYNC] PID {pid} | У объекта {object_id} нет фото. Пропускаем.")
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
                    logger.error(f"[ASYNC] PID {pid} | Ошибка декодирования {object_id}")
                    return None

                logger.debug(f"[ASYNC] PID {pid} | Скачивание {index} завершено (ID {object_id})")
                return index, ColorArtwork(img_bgr, metadata)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"[ASYNC] Попытка {attempt + 1}/{max_retries} для {object_id} через {wait_time} с.")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[ASYNC] Ошибка загрузки {object_id}: {e}")
                    return None
            except Exception as e:
                logger.error(f"[ASYNC] Неожиданная ошибка загрузки {object_id}: {e}")
                return None

        return None

    async def _check_done(self, futures: set, block: bool = False) -> tuple[set, set]:
        if not futures:
            return set(), set()
        if block:
            done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
        else:
            done, pending = await asyncio.wait(futures, timeout=0)
            if not done:
                await asyncio.sleep(0)
        return done, pending

    async def download_generator(self, object_ids: list[tuple[int, int]], session: aiohttp.ClientSession) -> AsyncGenerator[tuple[int, ColorArtwork], None]:
        pid = os.getpid()
        logger.debug(f"[ASYNC] PID {pid} | Создано {len(object_ids)} задач")

        pending = {asyncio.create_task(self._fetch_single(idx, obj_id, session)) for idx, obj_id in object_ids}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                if result is not None:
                    yield result

    async def process_generator(self, download_stream: AsyncGenerator[tuple[int, ColorArtwork], None], executor: ProcessPoolExecutor) -> AsyncGenerator[tuple[int, int, dict[str, bytes], dict], None]:
        pid = os.getpid()
        loop = asyncio.get_running_loop()
        pending_futures = set()

        async for index, artwork in download_stream:
            logger.debug(f"[ASYNC] PID {pid} | Получено изображение {index}, отправка в процесс")
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
        pid = os.getpid()
        async for index, obj_id, results_bytes, raw_metadata in process_stream:
            logger.debug(f"[ASYNC] PID {pid} | Сохранение {index} началось")

            meta_path = os.path.join(self.save_dir, f"{index}_{obj_id}_metadata.json")
            async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(raw_metadata, indent=4, ensure_ascii=False))

            for filename, img_bytes in results_bytes.items():
                filepath = os.path.join(self.save_dir, f"{index}_{obj_id}_{filename}")
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(img_bytes)

            logger.debug(f"[ASYNC] PID {pid} | Сохранение {index} завершено")

    @async_timer_decorator
    async def run_pipeline(self, input_json: str, count: int) -> None:
        object_ids = self._get_urls_to_download(input_json, count)
        if not object_ids:
            logger.error("[ERROR] Не удалось получить цели для скачивания.")
            return

        logger.debug(f"[ASYNC] Запланировано объектов: {len(object_ids)}")

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            with ProcessPoolExecutor() as executor:
                download_stream = self.download_generator(object_ids, session)
                process_stream = self.process_generator(download_stream, executor)
                await self.save_consumer(process_stream)