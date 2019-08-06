import numpy as np
import pandas as pd
import itertools

from paratus.baseModel import BaseModel


class CategoricalCombinations(BaseModel):
    def __init__(self, categorical_features, min_combinations=2, max_combination=2, prefix="comb_"):
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
