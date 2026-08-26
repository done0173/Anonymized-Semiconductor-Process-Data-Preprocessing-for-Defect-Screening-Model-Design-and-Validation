import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

def calculate_metrics(
    y_true,
    model=None,
    X_test=None,
    y_pred=None,
    y_proba=None,
    threshold=0.5,
) -> dict:
    """
    분류 모델의 성능 지표를 계산하는 통합 모듈 (PE 직무용 특화 지표 반영).
    표준적인 테스트셋 평가와 OOF/임계치 스위핑 평가를 모두 단 하나의 함수로 처리합니다.

    Parameters
    ----------
    y_true : array-like
        실제 정답 레이블
    model : Pipeline or Estimator, optional
        학습이 완료된 모델 객체 (제공 시 내부에서 predict/predict_proba 수행)
    X_test : DataFrame or array-like, optional
        평가용 피처 데이터셋
    y_pred : array-like, optional
        직접 제공하는 예측 클래스 값
    y_proba : array-like, optional
        직접 제공하는 예측 확률 값 (제공 시 지정된 threshold가 적용되어 y_pred 계산)
    threshold : float, default 0.5
        불량 판정 임계값 (y_proba 연산 또는 model의 확률 출력 시 적용)

    Returns
    -------
    dict
        Accuracy, Precision, Recall, F1, F2-Score, Confusion Matrix,
        실제 Overkill(과검출)/Escape(미검출) 칩 개수, ROC-AUC, PR-AUC가 통합된 결과 딕셔너리
    """
    # 1. 입력 데이터 다형성(Polymorphism) 처리
    if model is not None and X_test is not None:
        # 모델 객체가 직접 인입된 경우
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = None
    elif y_proba is not None:
        # 이미 누적된 OOF 확률값만 주입된 경우 (임계치 스위핑 포함)
        y_pred = (y_proba >= threshold).astype(int)
    elif y_pred is not None:
        # 단순 예측 레이블만 주입된 경우
        pass
    else:
        raise ValueError(
            "반드시 (model, X_test) 세트, 혹은 (y_pred), 혹은 (y_proba) 중 하나는 제공되어야 합니다."
        )

    # 2. 극불균형 데이터셋에 특화된 안전한 혼동 행렬(Confusion Matrix) 매핑 계산
    # (모델이 0으로만 대답하는 대폭망 상황에서도 scikit-learn 차원 왜곡 에러를 완벽 차단함)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))  # Overkill: 정상인데 불량으로 잘못 판정
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))  # Escape: 불량인데 정상으로 유출됨
    
    cm = confusion_matrix(y_true, y_pred)

    # 3. 통합 성능 지표 딕셔너리 생성
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        
        # PE 핵심 지표: 불량 검출력(Recall)에 가중치(beta=2)를 준 F2-Score
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        
        "confusion_matrix": cm,
        
        # PE 핵심 지표: 실제 수량 단위의 양산 비용 정량화
        "overkill_count": int(fp),
        "escape_count": int(fn),
    }

    # 4. 확률값이 확보된 경우에만 임계값 독립 지표 산출
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics
