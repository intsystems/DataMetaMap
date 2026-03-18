import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor


class Validators:

    @staticmethod
    def is_positive(number: int) -> int:
        assert number > 0, "Number is non-positive"
        return number

    @staticmethod
    def non_negative(number: int) -> int:
        assert number >= 0, "Number is negative"
        return number

    @staticmethod
    def all_elements_positive(arr: list[int]) -> list[int]:
        assert all(map(lambda x: x > 0, arr)), \
            "List contains non-positive elements"
        return arr

    @staticmethod
    def non_empty(arr: list[int]) -> list[int]:
        assert len(arr) > 0, "List is empty"
        return arr


class DataUtils:

    @staticmethod
    def get_preprocessing_pipeline() -> BaseEstimator:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )),
        ]).set_output(transform="pandas")

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]).set_output(transform="pandas")

        pipeline = Pipeline([
            ("transformers", make_column_transformer(
                (cat_pipeline, make_column_selector(
                    dtype_include=("object", "category")
                )),
                (num_pipeline, make_column_selector(
                    dtype_include=np.number
                )),
            )),
        ]).set_output(transform="pandas")
        return pipeline

    @staticmethod
    def sample_random_subset(
        a: int | NDArray[np.generic],
    ) -> NDArray[np.generic]:
        if isinstance(a, int):
            a = np.arange(a)
        if len(a) == 1:
            return a
        subset_idx = np.random.uniform(size=len(a)) < 0.5
        if np.sum(subset_idx) == 0:
            return a
        return a[subset_idx]

    @staticmethod
    def index_tensor_using_lists(
        tensor: Tensor,
        rows_idx: NDArray[np.generic],
        col_idx: NDArray[np.generic],
    ) -> Tensor:
        return tensor[rows_idx].T[col_idx].T


class InvalidDataTypeException(Exception):
    pass


class InconsistentTypesException(Exception):
    pass
