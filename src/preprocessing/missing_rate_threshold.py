import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class MissingRateFilterWithIndicators(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._to_dataframe(X)

        self.feature_names_in_ = X.columns.to_numpy()
        missing_rate = X.isna().mean()

        self.kept_columns_ = missing_rate[
            missing_rate <= self.threshold
        ].index.to_list()

        self.indicator_columns_ = missing_rate[
            missing_rate > 0
        ].index.to_list()

        self.feature_names_out_ = np.asarray(
            self.kept_columns_
            + [f"{col}__is_missing" for col in self.indicator_columns_],
            dtype=object,
        )
        return self

    def transform(self, X):
        X = self._to_dataframe(X)

        result = X.loc[:, self.kept_columns_].copy()

        indicators = X.loc[:, self.indicator_columns_].isna().astype("int8")
        indicators.columns = [
            f"{col}__is_missing" for col in self.indicator_columns_
        ]

        return pd.concat([result, indicators], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reindex(columns=getattr(self, "feature_names_in_", X.columns))

        columns = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=columns)
