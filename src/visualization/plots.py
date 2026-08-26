"""
시각화 유틸리티.

EDA에서 사용하는 시각화 함수 모음.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_missing_distribution(counts: pd.Series, *,
                               figsize: tuple[int, int] = (10, 5)) -> None:
    """
    결측률 구간별 컬럼 개수 시각화.

    Parameters
    ----------
    counts
        결측률 구간별 컬럼 개수.
    """

    plt.figure(figsize=figsize)

    counts.plot.bar()

    plt.title("Missing Value Distribution")

    plt.xlabel("Missing Ratio")

    plt.ylabel("Number of Features")

    plt.tight_layout()

    plt.show()


def plot_target_distribution(value_counts: pd.Series, *, 
                             figsize: tuple[int, int] = (6, 5)):
    """
    타겟 개수 시각화.

    Parameters
    ----------
    value_counts
        타겟 분류별 개수.
    """

    plt.figure(figsize=figsize)

    value_counts.plot.bar()

    plt.title("Missing Value Distribution")

    plt.xticks(rotation=0)
    
    plt.yticks(rotation=0)

    plt.xlabel("Missing Ratio")

    plt.ylabel("Number of Features")

    plt.tight_layout()

    plt.show()