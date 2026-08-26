import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from preprocessing.pipeline import build_pipeline
from modeling.metrics import calculate_metrics

FEATURE_SELECTION_STEPS = ["missing", "variance", "correlation", "clustering"]

def get_feature_breakdown(pipeline):
    """
    최종 파이프라인에서 출력된 피처들을 분석하여
    [전체 피처 수, 원본 피처 수, 결측 인디케이터 피처 수]를 반환합니다.
    """
    feature_names = None
    # 역순으로 가장 마지막에 실행된 유효한 피처 선택 단계를 찾습니다.
    for step_name in reversed(FEATURE_SELECTION_STEPS):
        if step_name in pipeline.named_steps:
            selector = pipeline.named_steps[step_name]
            if hasattr(selector, "get_feature_names_out"):
                feature_names = selector.get_feature_names_out()
                break
                
    if feature_names is None:
        return None, None, None

    # 결측 인디케이터 피처 판별 (이름에 'missing', 'indicator', 'nan' 등이 포함된 경우)
    # 파이프라인에서 생성하는 인디케이터 명명 규칙에 맞춰 매칭 조건을 조절할 수 있습니다.
    indicator_features = [
        f for f in feature_names 
        if "missing" in f.lower() or "indicator" in f.lower() or "nan" in f.lower()
    ]
    original_features = [f for f in feature_names if f not in indicator_features]
    
    return len(feature_names), len(original_features), len(indicator_features)


def run_experiment(
    *,
    X_train,
    y_train,
    models,
    experiment_name,
    methods,
    parameter_name,
    pipeline_kwargs=None,
    classification_thresholds=None,
):
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 임계값 스위핑 여부 판별
    thresholds = classification_thresholds if classification_thresholds is not None else [0.5]

    for method_name, method in methods.items():
        for model_name, model_info in models.items():
            
            oof_probabilities = np.zeros(len(X_train))
            
            # 폴드별 피처 수 수집을 위한 리스트
            total_features_list = []
            original_features_list = []
            indicator_features_list = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                pipeline = build_pipeline(
                    model=model_info,
                    **pipeline_kwargs,
                    **{parameter_name: method},
                )
                
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                pipeline.fit(X_tr, y_tr)
                
                # 예측 확률 추출
                if hasattr(pipeline, "predict_proba"):
                    positive_index = list(pipeline.classes_).index(1)
                    oof_probabilities[val_idx] = pipeline.predict_proba(X_val)[:, positive_index]
                else:
                    raise ValueError(f"모델 {model_name}은 확률 예측(predict_proba)을 지원해야 OOF 검증이 가능합니다.")
                
                # ⭐ 피처 세부 구성 추출
                total_rem, orig_rem, ind_added = get_feature_breakdown(pipeline)
                if total_rem is not None:
                    total_features_list.append(total_rem)
                    original_features_list.append(orig_rem)
                    indicator_features_list.append(ind_added)

            # --- [교차 검증 완료 후 통계량 산출] ---
            # 평균값(Mean) 산출
            avg_total = np.mean(total_features_list) if total_features_list else None
            avg_original = np.mean(original_features_list) if original_features_list else None
            avg_indicator = np.mean(indicator_features_list) if indicator_features_list else None
            
            # 산포 제어 검증을 위한 표준편차(Std) 산출
            std_total = np.std(total_features_list) if total_features_list else None
            std_original = np.std(original_features_list) if original_features_list else None
            std_indicator = np.std(indicator_features_list) if indicator_features_list else None

            # --- [OOF 루프 평가 및 메트릭 기록] ---
            for threshold in thresholds:
                metrics = calculate_metrics(
                    y_true=y_train, 
                    y_proba=oof_probabilities, 
                    threshold=threshold
                )
                
                # 데이터프레임에 평균 및 표준편차 지표 추가
                metrics["remaining_features"] = avg_total
                metrics["remaining_features_std"] = std_total
                
                metrics["remaining_original_features"] = avg_original
                metrics["remaining_original_features_std"] = std_original
                
                metrics["added_indicator_features"] = avg_indicator
                metrics["added_indicator_features_std"] = std_indicator
                
                metrics["experiment"] = experiment_name
                metrics["method"] = method_name
                metrics["model"] = model_name
                
                if classification_thresholds is not None:
                    metrics["classification_threshold"] = threshold

                results.append(metrics)

    index_columns = ["experiment", "method", "model"]
    if classification_thresholds is not None:
        index_columns.insert(2, "classification_threshold")

    return pd.DataFrame(results).set_index(index_columns)

def run_cv_threshold_experiment(tuned_pipelines, X_train, y_train, cv):
    thresholds = np.arange(0.05, 0.95, 0.05)
    all_results = []
    
    for model_name, pipeline in tuned_pipelines.items():
        # 아웃 오브 폴드(OOF) 예측 확률 추출로 데이터 누설 원천 차단
        y_oof_proba = cross_val_predict(
            pipeline, 
            X_train, 
            y_train, 
            cv=cv, 
            method="predict_proba", 
            n_jobs=-1
        )[:, 1]
        
        # 임계값과 독립적인 모델 고유의 변별력 지표 산출
        roc_auc = roc_auc_score(y_train, y_oof_proba)
        pr_auc = average_precision_score(y_train, y_oof_proba)
        
        for th in thresholds:
            y_pred = (y_oof_proba >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
            
            # 테스트 엔지니어링 관점의 주요 평가 지표 계산
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
            
            all_results.append({
                "model": model_name,
                "threshold": round(th, 2),
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "f2": round(f2, 6),
                "overkill_count": fp,  # 오검출 (양산 수율 저하 수량)
                "escape_count": fn,    # 미검출 (품질 유출 사고 수량)
                "roc_auc": round(roc_auc, 6),
                "pr_auc": round(pr_auc, 6)
            })
            
    # 전체 결과를 단일 데이터프레임으로 통합하여 반환
    return pd.DataFrame(all_results)