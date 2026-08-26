"""
모델 생성 모듈

프로젝트에 사용할 분류 모델을 관리.

Project : SECOM Yield Prediction
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models() -> dict:
    """
    사용할 모델 반환.

    Returns
    -------
    dict
        모델 이름과 객체
    """

    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                random_state=42,
                max_iter=5000,
            ),
            "needs_scaling": True,
        },

        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=42,
            ),
            "needs_scaling": False,
        },

        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
            "needs_scaling": False,
        },
    }

    return models