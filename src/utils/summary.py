"""
데이터셋 요약 유틸
"""

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> None:
    """
    데이터셋 요약 정보 출력

    Args:
        df:
            입력 데이터 프레임
    """

    rows, cols = df.shape

    memory = df.memory_usage(deep=True).sum() / 1024**2

    duplicate_rows = df.duplicated().sum()

    missing_cells = df.isna().sum().sum()

    total_cells = rows * cols

    missing_ratio = missing_cells / total_cells * 100

    print("=" * 60)
    print("데이터셋 요약")
    print("=" * 60)

    print(f"Rows             : {rows:,}")
    print(f"Columns          : {cols:,}")
    print(f"Memory Usage     : {memory:.2f} MB")
    print(f"Duplicate Rows   : {duplicate_rows:,}")
    print(f"Missing Cells    : {missing_cells:,}")
    print(f"Missing Ratio    : {missing_ratio:.2f}%")

    print("=" * 60)

    print("\nData Types")

    dtype_counts = df.dtypes.value_counts()

    for dtype, count in dtype_counts.items():
        print(f"{str(dtype):15} : {count}")

    print("=" * 60)