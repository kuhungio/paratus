from abc import abstractmethod, ABC


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X if y is None else y)

    @abstractmethod
    def inverse_transform(self, X):
        pass
