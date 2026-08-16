"""공통 전처리 계약만 정의하는 템플릿 메서드 모듈."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from ..UTILS.spatial_utils import KPX_GROUPS, PreprocessingError
from ..UTILS.time_utils import select_group


RATED_CAPACITY_KW = {
    "kpx_group_1": 21_600.0,
    # VESTAS V126 3.6MW × 6기 = 21.6MW = 21,600kW
    "kpx_group_2": 21_600.0,
    # VESTAS V126 3.6MW × 6기 = 21.6MW = 21,600kW
    "kpx_group_3": 21_000.0,
    # UNISON U136 4.2MW × 5기 = 21.0MW = 21,000kW
}


def _load_data(
    data: pd.DataFrame | str | Path,
    encoding: str = "utf-8-sig",
    **read_kwargs: Any,
) -> pd.DataFrame:
    """입력을 DataFrame으로 읽는 내부 헬퍼 함수.

    Args:
        data:       DataFrame이면 그대로 복사하고,
                    문자열/Path면 확장자에 따라 CSV 또는 Excel로 읽는다.
        encoding:   CSV 읽기에 쓰는 인코딩 (기본 UTF-8 with BOM).
        read_kwargs: pd.read_csv 또는 pd.read_excel에 추가로 넘길 인자.

    Returns:
        로딩된 DataFrame 복사본.

    Raises:
        FileNotFoundError: 파일이 없으면.
        PreprocessingError: 읽기에 실패하면.
        ValueError:        지원하지 않는 확장자면.
    """
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
    """KPX 한 그룹의 모델링 기준을 담는 읽기 전용 데이터 클래스.

    Attributes:
        name:              그룹 이름 (kpx_group_1 / 2 / 3).
        rated_capacity_kw: 그룹 설비용량(kW).
    """

    name: str
    rated_capacity_kw: float


class BasePreprocessor(ABC):
    """소스별 전처리기의 입력 계약과 실행 순서를 고정하는 추상 클래스.

    Attributes:
        source:           데이터 소스 식별자 ("gfs" / "ldaps" / "scada"). 자식이 반드시 정의.
        COLUMN_MAP:       원본 컨럼명 -> 표준 컨럼명 매핑. 자식이 반드시 정의.
        REQUIRED_COLUMNS: 전처리 시작 전에 반드시 있어야 하는 컨럼 목록.
        group:            None이면 전체 그룹, 값이 있으면 해당 KPX 그룹만 반환.
        scale_param:      computeScaleParam으로 채워지는 표준화 통계 저장소.
    """

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
        """로드 → 검증 → 소스별 변환 훅을 고정된 순서로 실행한다.

        Args:
            data:       DataFrame 또는 CSV/Excel 파일 경로.
            read_kwargs: pd.read_csv 등에 추가로 넘길 인자.

        Returns:
            8단계를 거친 모델 입력용 DataFrame.

        Logic:
            transformColumnName -> transformTimeAxis -> checkPhysicalLimit
            -> interpolateMissing -> calculatePhysicalFeature
            -> checkDerivedFlag -> transformCyclicFeature -> select_group 순서로 호출.
        """
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
        """입력 DataFrame이 계약을 만족하는지 검증한다.

        Args:
            df:               검증할 DataFrame.
            required_columns: 반드시 있어야 하는 컨럼 목록.
            allow_empty:      True면 빈 DataFrame도 허용한다.

        Returns:
            모든 검증을 통과하면 True.

        Raises:
            TypeError:         입력이 DataFrame이 아닐 때.
            PreprocessingError: 빈 DataFrame이거나 중복 컨럼명이 있을 때.
            KeyError:          필수 컨럼이 끝일 때.
        """
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
        """표준화에 쓸 평균·표준편차를 계산하고 self.scale_param에 저장한다.

        Args:
            df:           **학습 구간만** 잘라낸 표. 검증·평가 구간이 섯이면 누수다.
            feature_cols: 표준화할 컨럼 목록.

        Returns:
            {컨럼: (평균, 표준편차)} 딕셔너리. 상수 컨럼는 std = 1.0으로 대체.
        """
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
        """표준화된 값을 원래 단위로 역변환한다.

        Args:
            values:      표준화된 값 (배열, Series, 또는 스칼라).
            column:      역변환할 컨럼명.
            scale_param: 사용할 통계 딕셔너리. None이면 self.scale_param을 쓴다.

        Returns:
            values × std + mean로 복원된 원래 단위 값.

        Raises:
            KeyError: computeScaleParam을 먼저 호출하지 않았으면.
        """
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


def __getattr__(name: str) -> Any:
    if name in ("PreprocessorFactory", "FeatureEngineerFactory", "create_preprocessor"):
        from . import PreprocessorFactory as factory_module
        return getattr(factory_module, name)
    raise AttributeError(f"module {__name__!r}에 {name!r} 속성이 없습니다")


__all__ = [
    "BasePreprocessor", "FeatureEngineer", "GroupSpec", "PreprocessingError",
    "KPX_GROUPS", "RATED_CAPACITY_KW",
]
