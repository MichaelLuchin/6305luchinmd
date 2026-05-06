"""Чтение CSV, фильтрация и генерация данных."""

import csv
import json
import os
from typing import Generator

import pandas as pd

from metetl.logging_config import logger


def prepare_metadata(csv_path: str, output_path: str) -> None:
    """Парсит CSV и собирает JSON файл с метаданными (Лабораторная №1)."""
    paintings = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    paintings.append(row)
    except FileNotFoundError:
        logger.error(f"[ERROR] Файл {csv_path} не найден!")
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(paintings, f, indent=4, ensure_ascii=False)

    logger.debug(f"Успешно обработано {len(paintings)} записей. Сохранено в {output_path}")


def read_chunks(file_path: str, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
    """Чтение только необходимых столбцов для экономии памяти."""
    cols = ['Culture', 'AccessionYear', 'Object Begin Date']
    for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            usecols=cols,
            low_memory=False,
    ):
        yield chunk


def clean_and_calc_chunks(chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
    """Очистка и расчет возраста внутри чанка."""
    for chunk in chunks:
        chunk['AccessionYearClean'] = (
            chunk['AccessionYear'].astype(str).str.extract(r'(\d{4})')[0].astype(float)
        )
        chunk['Object Begin Date'] = pd.to_numeric(
            chunk['Object Begin Date'],
            errors='coerce',
        )
        chunk = chunk.dropna(subset=['Culture', 'AccessionYearClean', 'Object Begin Date'])
        chunk['Age'] = chunk['AccessionYearClean'] - chunk['Object Begin Date']

        yield chunk[chunk['Age'] >= 0].copy()