from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from preprocessing.feature_selection import (
    CoefficientOfVariationThreshold,
    CorrelationFilter,
    HierarchicalFeatureClusterer,
)
from preprocessing.missing_rate_threshold import MissingRateFilterWithIndicators

def get_imputer(strategy: str):

    if strategy == "mean":
        return SimpleImputer(
            strategy="mean",
        ).set_output(transform="pandas")

    if strategy == "median":
        return SimpleImputer(
            strategy="median",
        ).set_output(transform="pandas")

    raise ValueError(f"Unknown imputer strategy: {strategy}")

def get_scaler(name: str):

    if name == "standard":
        return StandardScaler()

    if name == "robust":
        return RobustScaler()

    if name == "minmax":
        return MinMaxScaler()

    raise ValueError(f"Unknown scaler: {name}")

def build_pipeline(
        model,
        imputer="mean",
        scaler=None,
        sampler=None,
        variance_threshold=None,
        correlation_threshold=None,
        cluster_distance_threshold=None,
        missing_threshold=None,
    ):

    steps = []

    if missing_threshold is not None:
        steps.append(
            (
                "missing",
                MissingRateFilterWithIndicators(
                    threshold=missing_threshold,
                ),
            )
        )

    if variance_threshold is not None:
        steps.append(
            (
                "variance",
                CoefficientOfVariationThreshold(
                    threshold=variance_threshold,
                ),
            )
        )

    steps.append(
        (
        "imputer",
        get_imputer(imputer),
        )
    )

    if correlation_threshold is not None:
        steps.append(
            (
                "correlation",
                CorrelationFilter(
                    threshold=correlation_threshold,
                ),
            )
        )

    if cluster_distance_threshold is not None:
        steps.append(
            (
                "clustering",
                HierarchicalFeatureClusterer(
                    distance_threshold=cluster_distance_threshold,
                ),
            )
        )

    if scaler is not None and model.get("needs_scaling", False):
        steps.append(
            (
            "scaler",
            get_scaler(scaler),
            )
        )

    if sampler is not None:
        steps.append(
            (
            "sampler",
            sampler,
            )
        )

    estimator = model.get("estimator")

    if estimator is None:
        raise ValueError("Model dictionary must contain an 'estimator' key.")

    steps.append(
        (
        "model",
        estimator,
        )
    )

    return ImbPipeline(steps)
