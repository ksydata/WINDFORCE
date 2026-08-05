"""공통 전처리 계약만 정의하는 템플릿 메서드 모듈."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from .utils.spatial_utils import KPX_GROUPS, PreprocessingError
from .utils.time_utils import select_group


RATED_CAPACITY_KW = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}


def _load_data(
    data: pd.DataFrame | str | Path,
    encoding: str = "utf-8-sig",
    **read_kwargs: Any,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, **read_kwargs)
        if path.suffix.lower() in {".csv", ".gz"}:
            return pd.read_csv(path, encoding=encoding, **read_kwargs)
        raise ValueError(f"지원하지 않는 입력 확장자입니다: {path.suffix}")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise PreprocessingError(f"입력 파일을 읽지 못했습니다: {path}") from exc


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


@dataclass(frozen=True)
class GroupSpec:
    name: str
    rated_capacity_kw: float


class BasePreprocessor(ABC):
    """소스별 전처리기의 입력 계약과 실행 순서를 고정한다."""

    source: ClassVar[str]
    COLUMN_MAP: ClassVar[dict[str, str]]
    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("source", "COLUMN_MAP"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__}은 클래스 속성 '{name}'을 정의해야 합니다")

    def __init__(self, group: str | None = None):
        if group is not None and group not in KPX_GROUPS:
            raise ValueError(f"group은 {KPX_GROUPS} 중 하나여야 합니다: {group}")
        self.group = group
        self.scale_param: dict[str, tuple[float, float]] = {}

    def preprocess(self, data: pd.DataFrame | str | Path, **read_kwargs: Any) -> pd.DataFrame:
        """로드 → 검증 → 소스별 변환 훅을 고정된 순서로 실행한다."""
        df = _load_data(data, **read_kwargs)
        self.validateData(df, required_columns=self.REQUIRED_COLUMNS)

        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric):
            df = df.copy()
            df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)

        df = self.transformColumnName(df)
        df = self.transformTimeAxis(df)
        df = self.checkPhysicalLimit(df)
        df = getattr(self, "interpolateMissing", _identity)(df)
        df = self.calculatePhysicalFeature(df)
        df = getattr(self, "checkDerivedFlag", _identity)(df)
        df = getattr(self, "transformCyclicFeature", _identity)(df)
        return select_group(df, self.group)

    def validateData(
        self,
        df: pd.DataFrame,
        required_columns: Iterable[str] = (),
        allow_empty: bool = False,
    ) -> bool:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("입력은 pandas.DataFrame이어야 합니다")
        if df.empty and not allow_empty:
            raise PreprocessingError("입력 DataFrame이 비어 있습니다")
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise KeyError(f"필수 컬럼이 없습니다: {missing}")
        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            raise PreprocessingError(f"중복 컬럼명이 있습니다: {duplicated}")
        return True

    def computeScaleParam(
        self, df: pd.DataFrame, feature_cols: Iterable[str]
    ) -> dict[str, tuple[float, float]]:
        params = {}
        for col in feature_cols:
            if col not in df.columns:
                raise KeyError(f"스케일링 컬럼이 없습니다: {col}")
            values = pd.to_numeric(df[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std())
            params[col] = (mean, std if np.isfinite(std) and std > 1e-8 else 1.0)
        self.scale_param = params
        return params

    def transformInverseScale(
        self,
        values: np.ndarray | pd.Series | float,
        column: str,
        scale_param: Mapping[str, tuple[float, float]] | None = None,
    ) -> np.ndarray | pd.Series | float:
        params = scale_param if scale_param is not None else self.scale_param
        if not params or column not in params:
            raise KeyError(
                f"역변환 파라미터가 없습니다: {column}. computeScaleParam을 먼저 호출하세요"
            )
        mean, std = params[column]
        return values * std + mean

    @abstractmethod
    def transformColumnName(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 컬럼명을 표준 이름으로 통일한다."""

    @abstractmethod
    def transformTimeAxis(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 컬럼을 datetime으로 바꾸고 소스별 시간 메타를 만든다."""

    @abstractmethod
    def checkPhysicalLimit(self, df: pd.DataFrame) -> pd.DataFrame:
        """원시값의 물리한계를 검사해 플래그를 붙인다."""

    @abstractmethod
    def calculatePhysicalFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """소스별 물리 파생변수를 계산한다."""


FeatureEngineer = BasePreprocessor


def __getattr__(name: str) -> Any:
    if name in ("PreprocessorFactory", "FeatureEngineerFactory", "create_preprocessor"):
        from . import PreprocessorFactory as factory_module
        return getattr(factory_module, name)
    raise AttributeError(f"module {__name__!r}에 {name!r} 속성이 없습니다")


__all__ = [
    "BasePreprocessor", "FeatureEngineer", "GroupSpec", "PreprocessingError",
    "KPX_GROUPS", "RATED_CAPACITY_KW",
]
