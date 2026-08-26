"""
분석을 위한 데이터 정보를 간단하게 호출할 수 있는 유틸리티.

다양한 데이터셋을 적용하기 위해 재사용이 가능한 메서드 제공.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd


@dataclass
class DataProfiler:
    """
    데이터 프레임을 수정하지 않고 정보만 확인.

    Attributes
    ----------
    df
        원본 데이터 프레임
    """

    df: pd.DataFrame
    
    @cached_property
    def df_shape(self) -> tuple[int, int]:
        return self.df.shape

    @cached_property
    def missing_mask(self) -> pd.DataFrame:
        return self.df.isna()

    @cached_property
    def column_missing_count(self) -> pd.Series:
        return self.missing_mask.sum()

    @cached_property
    def row_missing_count(self) -> pd.Series:
        return self.missing_mask.sum(axis=1)

    @cached_property
    def column_missing_ratio(self) -> pd.Series:
        return (self.missing_mask.mean() * 100).round(2)

    @cached_property
    def row_missing_ratio(self) -> pd.Series:
        return (self.missing_mask.mean(axis=1) * 100).round(2)

    @cached_property
    def total_missing_ratio(self) -> float:
        return (self.column_missing_count.sum() / self.df.size * 100).round(2)

    @cached_property
    def duplicate_rows(self) -> int:
        return int(self.df.duplicated().sum())

    @cached_property
    def memory_mb(self) -> float:
        return (self.df.memory_usage(deep=True).sum() / 1024**2).round(2)
    
    @cached_property
    def unique_count(self) -> pd.Series:
        return self.df.nunique()

    def dataset_summary(self) -> dict:
        """
        데이터의 정보를 요약하여 반환.

        Returns
        -------
        dict
            데이터 요약 정보
        """

        rows, columns = self.df_shape

        return {
            "rows": rows,
            "columns": columns,
            "memory (mb)": self.memory_mb,
            "duplicate rows": self.duplicate_rows,
            "total missing cells": int(self.column_missing_count.sum()),
            "total missing ratio (%)": float(self.total_missing_ratio),
        }

    def features_summary(self) -> pd.DataFrame:
        """
        컬럼별 데이터 타입과 결측치 정보를 반환.

        Returns
        -------
        pandas.DataFrame
            피처 정보를 담은 데이터 프레임
        """

        summary = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "non_null_count": self.df.count(),
                "missing count per columns": self.column_missing_count,
                "missing ratio per columns (%)": self.column_missing_ratio,
                "constant": self.unique_count == 1,
                "unique": self.unique_count,
            }
        )

        return summary
    
    def target_summary(self, target_column: str) -> dict:
        """
        타겟 컬럼의 정보를 반환.

        Parameters
        ----------
        target_column : str
            타겟 컬럼명

        Returns
        -------
        dict
            타겟 컬럼 정보
        """

        try:
            target_series = self.df[target_column]
        except KeyError as e:
            raise KeyError(
                f"Column '{target_column}' does not exist."
            ) from e

        target_series = self.df[target_column]
        target_value_counts = target_series.value_counts()

        return {
            "dtype": target_series.dtype,
            "non_null_count": target_series.count(),
            "missing count": int(target_series.isna().sum()),
            "missing ratio (%)": float((target_series.isna().mean() * 100).round(2)),
            "unique": int(self.unique_count[target_column]),
            "value_counts": target_value_counts.to_dict(),
            "value_counts_ratio (%)": (target_value_counts / self.df_shape[0] * 100).round(2).to_dict(),
        } 

    def missing_columns(self) -> pd.DataFrame:
        """
        컬럼별 결측치 정보 반환.

        Returns
        -------
        pandas.DataFrame
        """

        result = pd.DataFrame(
            {
                "missing count per columns": self.column_missing_count,
                "missing ratio per columns (%)": self.column_missing_ratio,                
            }
        )

        return (
            result
            .sort_values(
                "missing ratio per columns (%)",
                ascending=False,
            )
        )
    
    def missing_rows(self) -> pd.DataFrame:
        """
        행별 결측치 정보 반환.

        Returns
        -------
        pandas.DataFrame
        """

        result = pd.DataFrame(
            {
                "missing count per rows": self.row_missing_count,
                "missing ratio per rows (%)": self.row_missing_ratio,                
            }
        )

        return (
            result
            .sort_values(
                "missing ratio per rows (%)",
                ascending=False,
            )
        )