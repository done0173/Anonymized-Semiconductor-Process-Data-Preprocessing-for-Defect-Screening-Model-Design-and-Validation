import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


class CoefficientOfVariationThreshold(BaseEstimator, TransformerMixin):
    """
    변동계수(CV = 표준편차 / |평균|)가 threshold 미만인 피처를 제거한다.

    평균이 0에 가까운 피처는 CV가 불안정해질 수 있다. 이 경우 변동이
    존재하면 CV를 무한대로 간주하여 유지하고, 상수 피처만 제거한다.
    """

    def __init__(self, threshold=0.0, epsilon=1e-12):
        self.threshold = threshold
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = self._to_dataframe(X)

        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")

        self.feature_names_in_ = X.columns.to_numpy()

        means = X.mean(axis=0, skipna=True)
        stds = X.std(axis=0, ddof=0, skipna=True)
        abs_means = means.abs()

        self.coefficient_of_variation_ = stds.divide(
            abs_means.where(abs_means > self.epsilon)
        )

        near_zero_mean = abs_means <= self.epsilon
        has_variation = stds > self.epsilon
        self.coefficient_of_variation_.loc[
            near_zero_mean & has_variation
        ] = np.inf
        self.coefficient_of_variation_.loc[
            near_zero_mean & ~has_variation
        ] = 0.0

        cv_support = (
            self.coefficient_of_variation_ >= self.threshold
        )
        indicator_support = pd.Series(
            [
                str(column).endswith("__is_missing")
                for column in self.feature_names_in_
            ],
            index=self.feature_names_in_,
        )

        # 결측 indicator는 결측 자체의 신호를 보존하기 위해 항상 유지한다.
        self.support_ = (cv_support | indicator_support).to_numpy()
        self.feature_names_out_ = self.feature_names_in_[self.support_]

        if not self.support_.any():
            raise ValueError(
                "No feature meets the coefficient-of-variation threshold."
            )

        return self

    def transform(self, X):
        X = self._to_dataframe(X)
        return X.loc[:, self.feature_names_out_]

    def get_support(self):
        return self.support_

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            columns = getattr(self, "feature_names_in_", X.columns)
            return X.reindex(columns=columns)

        return pd.DataFrame(
            X,
            columns=getattr(self, "feature_names_in_", None),
        )


class CorrelationFilter(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        self.feature_names_in_ = (
            X.columns.to_numpy()
        )

        correlation_matrix = X.corr().abs()

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )

        self.features_to_drop_ = [
            column
            for column in upper_triangle.columns
            if not str(column).endswith("__is_missing") and (
                upper_triangle[column]
                > self.threshold
            ).any()
        ]

        self.feature_names_out_ = [
            column
            for column in self.feature_names_in_
            if column not in self.features_to_drop_
        ]

        return self

    def transform(self, X):

        X = pd.DataFrame(
            X,
            columns=self.feature_names_in_,
        )

        return X.drop(
            columns=self.features_to_drop_,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return np.asarray(
            self.feature_names_out_,
            dtype=object,
        )


class HierarchicalFeatureClusterer(BaseEstimator, TransformerMixin):
    """
    절대 상관계수 기반 계층적 군집화로 유사 피처를 묶고,
    각 군집에서 다른 멤버와의 평균 유사도가 가장 높은 대표 피처만 남긴다.
    """

    def __init__(self, distance_threshold=0.2, linkage_method="average"):
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method

    def fit(self, X, y=None):
        X = self._to_dataframe(X)

        if not 0 <= self.distance_threshold <= 1:
            raise ValueError("distance_threshold must be between 0 and 1.")

        self.feature_names_in_ = X.columns.to_numpy()
        self.indicator_columns_ = [
            column
            for column in self.feature_names_in_
            if str(column).endswith("__is_missing")
        ]
        feature_columns = [
            column
            for column in self.feature_names_in_
            if column not in self.indicator_columns_
        ]

        if len(feature_columns) <= 1:
            self.cluster_labels_ = pd.Series(
                1,
                index=feature_columns,
                dtype=int,
            )
            self.representative_columns_ = feature_columns
        else:
            similarity = X.loc[:, feature_columns].corr().abs().fillna(0.0)
            np.fill_diagonal(similarity.values, 1.0)
            distance = 1.0 - similarity

            linkage_matrix = linkage(
                squareform(distance.to_numpy(), checks=False),
                method=self.linkage_method,
            )
            labels = fcluster(
                linkage_matrix,
                t=self.distance_threshold,
                criterion="distance",
            )
            self.cluster_labels_ = pd.Series(labels, index=feature_columns)
            self.representative_columns_ = self._select_representatives(
                similarity,
            )

        self.feature_names_out_ = np.asarray(
            self.representative_columns_ + self.indicator_columns_,
            dtype=object,
        )
        return self

    def transform(self, X):
        X = self._to_dataframe(X)
        return X.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _select_representatives(self, similarity):
        representatives = []

        for label in np.unique(self.cluster_labels_):
            members = self.cluster_labels_.index[
                self.cluster_labels_ == label
            ]
            centrality = similarity.loc[members, members].mean(axis=1)
            representatives.append(centrality.idxmax())

        return representatives

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            columns = getattr(self, "feature_names_in_", X.columns)
            return X.reindex(columns=columns)

        return pd.DataFrame(
            X,
            columns=getattr(self, "feature_names_in_", None),
        )
