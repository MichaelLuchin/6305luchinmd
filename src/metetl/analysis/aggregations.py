"""Агрегация, вычисление статистик и построение графиков."""

import os
from typing import Generator

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from metetl.analysis.data_to_download import clean_and_calc_chunks, read_chunks
from metetl.logging_config import logger


def local_aggregate_chunks(chunks: Generator[pd.DataFrame, None, None]) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Сбор сырых сумм для каждой культуры внутри чанка."""
    for chunk in chunks:
        chunk['Age_sq'] = chunk['Age'] ** 2

        stats_chunk = chunk.groupby('Culture').agg(
            count=('Age', 'count'),
            sum=('Age', 'sum'),
            sum_sq=('Age_sq', 'sum'),
            min_date=('Object Begin Date', 'min'),
            max_date=('Object Begin Date', 'max'),
        )

        timeline_chunk = chunk.groupby(['Culture', 'AccessionYearClean']).agg(
            count=('Age', 'count'),
            sum=('Age', 'sum'),
        )

        yield stats_chunk, timeline_chunk


def collect_global_stats(local_agg_gen: Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Агрегирует локальные статистики в глобальные датафреймы."""
    df_stats = None
    df_timeline = None

    for stats_chunk, timeline_chunk in local_agg_gen:
        if df_stats is None:
            df_stats = stats_chunk
            df_timeline = timeline_chunk
        else:
            df_stats_new = df_stats.add(stats_chunk, fill_value=0)
            all_indices = df_stats.index.union(stats_chunk.index)

            s_min = df_stats['min_date'].reindex(all_indices, fill_value=np.inf)
            c_min = stats_chunk['min_date'].reindex(all_indices, fill_value=np.inf)
            df_stats_new['min_date'] = np.minimum(s_min, c_min)

            s_max = df_stats['max_date'].reindex(all_indices, fill_value=-np.inf)
            c_max = stats_chunk['max_date'].reindex(all_indices, fill_value=-np.inf)
            df_stats_new['max_date'] = np.maximum(s_max, c_max)

            df_stats = df_stats_new
            df_timeline = df_timeline.add(timeline_chunk, fill_value=0)

    return df_stats, df_timeline


def run_analysis(csv_path: str, output_dir: str) -> None:
    """Основная функция для анализа датасета."""
    logger.debug(f"Начало анализа датасета {csv_path}")
    raw_chunks = read_chunks(csv_path, chunk_size=10000)
    processed_chunks = clean_and_calc_chunks(raw_chunks)
    local_gen = local_aggregate_chunks(processed_chunks)

    df_stats, df_timeline = collect_global_stats(local_gen)

    df_top = df_stats.nlargest(10, 'count').copy()
    df_top['mean'] = df_top['sum'] / df_top['count']

    variance = (df_top['sum_sq'] / df_top['count']) - (df_top['mean'] ** 2)
    df_top['std'] = np.sqrt(variance.clip(lower=0))
    df_top['ci_95'] = 1.96 * (df_top['std'] / np.sqrt(df_top['count']))
    df_top['scatter_95'] = 1.96 * df_top['std']

    df_stats['history_len'] = df_stats['max_date'] - df_stats['min_date']
    max_hist_culture = df_stats['history_len'].idxmax()

    timeline = df_timeline.xs(max_hist_culture, level='Culture').copy()
    timeline['mean_age'] = timeline['sum'] / timeline['count']
    timeline = timeline.sort_index()
    timeline['rolling_mean'] = timeline['mean_age'].rolling(window=5, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    indices = np.arange(len(df_top))
    axes[0].errorbar(
        indices, df_top['mean'], yerr=df_top['scatter_95'], fmt='none',
        ecolor='lightblue', elinewidth=10, label='95% Scatter Interval', alpha=0.5,
    )
    axes[0].bar(
        indices, df_top['mean'], yerr=df_top['ci_95'], capsize=5,
        ecolor='black', color='steelblue', label='Mean & 95% CI',
    )
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(df_top.index, rotation=45, ha='right')
    axes[0].set_title("Топ-10 культур по возрасту объектов")
    axes[0].set_ylabel("Возраст (лет)")
    axes[0].legend()

    axes[1].plot(
        timeline.index, timeline['mean_age'],
        marker='.', alpha=0.3, label='Среднее за год',
    )
    axes[1].plot(
        timeline.index, timeline['rolling_mean'],
        color='red', linewidth=2, label='Скользящее среднее (5л)',
    )
    axes[1].set_title(f"Изменение возраста при поступлении\n({max_hist_culture})")
    axes[1].set_xlabel("Год поступления")
    axes[1].set_ylabel("Средний возраст (лет)")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "analysis_plot.png")
    plt.savefig(out_path)
    logger.debug(f"График успешно сохранен по пути: {out_path}")