import numpy as np
import pandas as pd
import itertools
from collections import Counter

from paratus.baseModel import BaseModel


class CategoricalCombinations(BaseModel):
    def __init__(self, categorical_features, min_combinations=2, max_combination=2, prefix="comb"):
        self._features = categorical_features
        self._combinations = list(itertools.chain.from_iterable([
            list(itertools.combinations(categorical_features, x))
            for x in np.arange(min_combinations, max_combination+1
                               )]))
        self._prefix = prefix
        self._comb_value_dict = {}

    def fit(self, X):
        assert(len(X.shape) == 2)
        for combination in self._combinations:
            self._comb_value_dict[combination] = dict(
                [(x, i) for i, x in enumerate(sorted(X[list(combination)].drop_duplicates().itertuples(index=False, name=None)))])

    def transform(self, X):
        assert(len(X.shape) == 2)
        res = X.copy()
        for combination in self._combinations:
            value_dict = self._comb_value_dict[combination]
            feature = self._get_column_name(combination)
            res[feature] = [value_dict[row]
                            if row in value_dict else 0 for row in X[list(combination)].itertuples(index=False, name=None)]
            res[feature] = res[feature].astype('category')
        return res

    def get_new_column_names(self):
        res = []
        for combination in self._combinations:
            res.append(self._get_column_name(combination))
        return res

    def _get_column_name(self, combination):
        return "{}_{}".format(
            self._prefix, '_'.join(map(str, list(combination))))

    def inverse_transform(self, X):
        raise Exception("Not implemented")


class FrequencyEncoding(BaseModel):
    def __init__(self, categorical_features, prefix="freq"):
        self._features = categorical_features
        self._prefix = prefix
        self._feature_value_counts = {}

    def fit(self, X):
        for feature in self._features:
            length = len(X)
            counts = dict(Counter(X[feature]))
            for k in counts:
                counts[k] /= length
            self._feature_value_counts[feature] = counts
            self._feature_value_counts[feature][np.nan] = np.sum(
                pd.isnull(X[feature]))/float(length)

    def transform(self, X):
        res = X.copy()
        for f in self._features:
            value_counts = self._feature_value_counts[f]
            feature = self._get_column_name(f)
            res[feature] = [self._get_value(value_counts, v) for v in X[f]]
            res[feature] = res[feature].astype('category')
        return res

    def _get_value(self, value_counts, v):
        if v in value_counts:
            return value_counts[v]
        elif pd.isnull(v):
            return value_counts[np.nan]
        return 0.0

    def get_new_column_names(self):
        res = []
        for f in self._features:
            res.append(self._get_column_name(f))
        return res

    def _get_column_name(self, feature):
        return "{}_{}".format(self._prefix, '_'.join(map(str, feature)))

    def inverse_transform(self, X):
        raise Exception("Not implemented")
