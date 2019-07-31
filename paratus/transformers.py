import numpy as np
import pandas as pd

from paratus.baseModel import BaseModel


class LabelSetTransformer:
    def __init__(self):
        self._values = None
        self._value_dict = None
        self._inv_value_dict = None

    def fit(self, X):
        values = set()
        for row in X:
            values = values.union(set(row))
        values = list(values)
        self._value_dict = dict([(v, i) for (i, v) in enumerate(values)])
        self._inv_value_dict = dict(enumerate(values))
        self._values = values

    def transform(self, X):
        res = np.zeros((len(X), len(self._values)))
        for i, row in enumerate(X):
            for v in row:
                if (v in self._value_dict):
                    res[i][self._value_dict[v]] = 1
        return res

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, y):
        assert len(y.shape) == 2
        res = []
        for row in y:
            entry = []
            for v in np.nonzero(row)[0]:
                if v in self._inv_value_dict:
                    entry.append(self._inv_value_dict[v])
            res.append(entry)
        return res

    def names(self):
        return self._values


class SequenceTransformer:
    def __init__(self, length):
        self.length = length
        self._values = None
        self._value_dict = None

    def fit(self, X):
        values = set()
        for row in X:
            values = values.union(set(row))
        values = list(values)
        self._value_dict = dict([(v, i) for (i, v) in enumerate(values)])
        self._inv_value_dict = dict(enumerate(values))
        self._values = values

    def transform(self, X):
        res = np.zeros((len(X),  self.length, len(self._values)))
        for i, row in enumerate(X):
            offset = max(0, self.length - len(row))
            for j, v in enumerate(row[-self.length:]):
                if (v in self._value_dict):
                    res[i][j + offset][self._value_dict[v]] = 1
        return res

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, y):
        assert len(y.shape) == 3
        res = []
        for row in y:
            entry = []
            for element in row:
                for v in np.nonzero(element)[0]:
                    if v in self._inv_value_dict:
                        entry.append(self._inv_value_dict[v])
            res.append(entry)
        return res


class FeatureScaler(BaseModel):
    def __init__(self, lower=0, higher=1):
        self._lower = lower
        self._higher = higher
        self._mins = None
        self._maxs = None

    def fit(self, X):
        assert(len(X.shape) == 2)
        self._mins = np.min(X, axis=0)
        self._maxs = np.max(X, axis=0)

    def transform(self, X):
        assert(len(X.shape) == 2)
        diffs = self._maxs - self._mins
        diffs[diffs == 0] = 1
        return (X - self._mins)/diffs

    def inverse_transform(self, X):
        assert(len(X.shape) == 2)
        return X*(self._maxs - self._mins) + self._mins


class MultiOneHotEncoder(BaseModel):
    def __init__(self, features_to_encode):
        self._features_to_encode = features_to_encode
        self._value_int_dict = dict()
        self._int_value_dict = dict()

    def fit(self, X):
        for feature in self._features_to_encode:
            unique_values = X[feature].unique()
            self._int_value_dict[feature] = dict(
                enumerate(unique_values))
            self._value_int_dict[feature] = dict(
                [(y, x) for (x, y) in enumerate(unique_values)])

    def transform(self, X):
        keep = [c for c in X.columns if c not in self._features_to_encode]
        df = X[keep].copy()
        for feature in self._features_to_encode:
            d = self._value_int_dict[feature]
            for v in d:
                indices = pd.isna(X[feature]) if pd.isna(
                    v) else X[feature] == v
                df['{}_{}'.format(feature, d[v])] = indices.astype(np.uint8)
        return df

    def inverse_transform(self, X):
        raise Exception("Not implemented")


class MultiEncoder(BaseModel):
    def __init__(self, features_to_encode, min_frequency=1):
        self._features_to_encode = features_to_encode
        self._min_frequency = min_frequency
        self._low_frequency_encoding_value = -1
        self._value_int_dict = {}

    def fit(self, X):
        for feature in self._features_to_encode:
            unique_values = X[feature].unique()
            counts = X[feature].value_counts()
            values_to_encode = set(
                counts[counts >= self._min_frequency].index.values.tolist())
            self._value_int_dict[feature] = {}
            for x, y in enumerate(sorted(values_to_encode)):
                self._value_int_dict[feature][y] = x + 1
            for v in unique_values:
                if v not in values_to_encode and pd.notnull(v):
                    self._value_int_dict[feature][v] = self._low_frequency_encoding_value

    def transform(self, X):
        keep = [c for c in X.columns if c not in self._features_to_encode]
        df = X[keep].copy()
        for feature in self._features_to_encode:
            d = self._value_int_dict[feature]
            df[feature] = [d[v] if v in d else 0 for v in X[feature]]
            df[feature] = df[feature].astype('category')
        return df

    def inverse_transform(self, X):
        raise Exception("Not implemented")
