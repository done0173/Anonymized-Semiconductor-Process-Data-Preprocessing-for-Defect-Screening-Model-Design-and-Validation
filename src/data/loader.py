"""
데이터 로드 모듈

데이터셋을 로드하는 기능 제공

Project : SECOM Yield Prediction
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# 프로젝트 데이터셋 경로 설정
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"


# ---------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------

def load_dataset(filename: str) -> pd.DataFrame:
    """
    원본 데이터셋이 위치한 경로에서 데이터셋을 로드.

    Parameters
    ----------
    filename : str
        데이터셋 파일명

    Returns
    -------
    pd.DataFrame
        로드한 데이터프레임
    
    Raises:
        FileNotFoundError:
            지정한 파일명이 존재하지 않을 경우 에러 발생
    """

    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)