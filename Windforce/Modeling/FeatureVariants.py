"""baseline_6에서 쓰는 버전별 피처 선택·압축 전략 모음.

모든 변형(variant)은 동일한 fit/transform 계약을 따르며, 반드시 **학습 구간에만**
fit해야 한다. 이렇게 해야 상관도·PCA 주성분·트리 중요도 같은 통계량이
검증 구간 정보를 미리 들여다보는 data leakage 없이 계산된다.

    AllFeaturesVariant       숫자형 컬럼 전체 (중앙값 결측치 대체만 수행)
      -> CorrelationFilteredVariant  상관도 0.95 초과 쌍 중 하나를 그리디하게 제거
    PCAVariant                표준화 후 분산 보존 비율 기준 주성분으로 압축
    TreeImportanceVariant     ExtraTrees 중요도 상위 비율만 선택
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureVariant(ABC):
    """baseline_6이 사용하는 모든 피처 변형의 공통 인터페이스 (추상 클래스)

    Attributes:
        - name: 변형 식별자 문자열. 하위 클래스가 오버라이드해 로그·결과표에서 구분에 쓴다.
        - random_state: 확률적 요소(PCA/ExtraTrees 등)의 재현성을 보장하는 시드.
        - input_columns_: fit() 시점에 관측된 입력 숫자형 컬럼 목록.
        - output_columns_: transform() 결과로 나오는 최종 컬럼 목록 (변형별로 의미가 다름).
    """

    name = "base"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.input_columns_: list[str] = []
        self.output_columns_: list[str] = []
        # 두 목록 모두 fit() 전에는 빈 리스트. sklearn 관례를 따라 접미사 "_"로
        # "fit 이후에 채워지는 학습된 속성"임을 표시한다

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FeatureVariant":
        """학습 구간 데이터로 이 변형에 필요한 통계량(결측치 대체값, 상관도, 주성분 등)을 계산한다."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """fit()에서 학습한 통계량을 이용해 입력을 변형된 피처 표로 바꾼다."""
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """fit 후 바로 transform까지 한 번에 수행하는 편의 메서드 (학습 구간 전용으로만 쓸 것)"""
        return self.fit(X, y).transform(X)
        # 메서드 체이닝: fit()이 self를 반환하므로 바로 이어서 transform() 호출 가능

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """숫자형 컬럼만 골라 무한대(inf) 값을 결측치로 바꾸는 내부 공통 전처리 헬퍼

        Args:
            - X: 원본 피처 DataFrame (문자열·datetime 등 비숫자형 컬럼이 섞여 있을 수 있음)

        Returns:
            - 숫자형 컬럼만 남기고 ±inf를 NaN으로 치환한 DataFrame 복사본

        Raises:
            - ValueError: 숫자형 컬럼이 하나도 없으면
        """
        numeric = X.select_dtypes(include=[np.number]).copy()
        # copy(): 원본 X를 건드리지 않도록 방어적 복사
        if numeric.empty:
            raise ValueError("피처 변형에 사용할 숫자형 컬럼이 없습니다.")
            # 전처리 단계에서 컬럼 선택을 잘못했을 때 원인을 바로 알 수 있도록 조기에 에러
        return numeric.replace([np.inf, -np.inf], np.nan)
        # 나눗셈 등에서 나온 ±inf를 NaN으로 바꿔야 이후 SimpleImputer/PCA가 정상 동작함


class AllFeaturesVariant(FeatureVariant):
    """숫자형 컬럼 전체를 그대로 쓰되, 학습 구간 중앙값으로 결측치만 채우는 변형"""

    name = "all"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AllFeaturesVariant":
        del y
        # 이 변형은 타깃 정보를 쓰지 않으므로 명시적으로 버림 (인터페이스 통일을 위해 인자만 받음)
        numeric = self._numeric_frame(X)
        self.input_columns_ = numeric.columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median").fit(numeric)
        # 반드시 학습 구간(X)에만 fit해서 검증 구간의 중앙값이 섞여 들어가지 않게 한다
        self.output_columns_ = self.input_columns_.copy()
        # 컬럼을 하나도 걸러내지 않으므로 입력·출력 컬럼 목록이 동일
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self._numeric_frame(X).reindex(columns=self.input_columns_)
        # reindex: 학습 때 봤던 컬럼 순서·집합으로 강제 정렬 (평가 데이터에 컬럼이 빠지면 NaN으로 채워짐)
        frame = pd.DataFrame(
            self.imputer_.transform(values),
            columns=self.input_columns_,
            index=X.index,
        )
        # imputer.transform은 numpy 배열을 반환하므로 원래 컬럼명·인덱스를 다시 붙여 DataFrame으로 복원
        return frame[self.output_columns_].copy()
        # output_columns_ 순서로 재정렬해 반환 (AllFeaturesVariant는 input과 동일하지만
        # 하위 클래스(CorrelationFilteredVariant)가 이 transform을 그대로 상속해 쓰므로
        # output_columns_ 기준으로 슬라이싱하는 구조를 여기서 통일해둔다)


class CorrelationFilteredVariant(AllFeaturesVariant):
    """상관도 0.95를 넘는 쌍 중 하나씩을 그리디하게 제거하는 변형 (AllFeaturesVariant의 transform을 재사용)"""

    name = "corr_filtered"

    def __init__(self, threshold: float = 0.95, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.threshold = threshold
        # 상관계수 절대값이 이 값을 초과하면 "중복 정보"로 간주해 제거 대상에 포함

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CorrelationFilteredVariant":
        del y
        # 상관도 필터링은 피처끼리의 관계만 보므로 타깃(y)은 쓰지 않는다
        numeric = self._numeric_frame(X)
        self.input_columns_ = numeric.columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median").fit(numeric)
        # 상관계수 계산 전에 결측치를 먼저 채워야 corr()가 NaN 없이 정상 동작한다
        filled = pd.DataFrame(
            self.imputer_.transform(numeric), columns=self.input_columns_, index=numeric.index
        )
        corr = filled.corr().abs()
        # 부호가 반대여도(강한 음의 상관) 중복 정보이므로 절대값으로 판단
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # 상삼각행렬만 남겨 (i,j)와 (j,i) 쌍을 중복 검사하지 않도록 한다 (대각선도 k=1로 제외)
        drop = [column for column in upper.columns if (upper[column] > self.threshold).any()]
        # 각 컬럼이 자신보다 먼저 오는 어떤 컬럼과도 threshold를 넘게 상관되면 제거 후보에 넣는 그리디 방식
        self.output_columns_ = [c for c in self.input_columns_ if c not in drop]
        # 원래 컬럼 순서를 유지한 채 제거 대상만 걸러내 최종 출력 컬럼 목록을 만든다
        if not self.output_columns_:
            raise ValueError("상관도 필터 후 남은 피처가 없습니다.")
            # threshold를 너무 낮게 잡으면 전부 제거될 수 있으므로 조기에 명확히 에러
        return self
    # transform()은 AllFeaturesVariant에서 그대로 상속받아 쓴다
    # (input_columns_로 결측치 대체 후 output_columns_로 슬라이싱하는 로직이 동일하기 때문)


class PCAVariant(FeatureVariant):
    """표준화 후 지정한 분산 보존 비율만큼의 주성분으로 압축하는 변형"""

    name = "pca"

    def __init__(self, variance: float = 0.95, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.variance = variance
        # PCA(n_components=variance)로 전달되어 "이 비율 이상의 분산을 설명하는 최소 성분 수"를 자동 결정

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PCAVariant":
        del y
        # PCA는 비지도 방식이라 타깃(y)을 쓰지 않는다
        numeric = self._numeric_frame(X)
        self.input_columns_ = numeric.columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median").fit(numeric)
        filled = self.imputer_.transform(numeric)
        self.scaler_ = StandardScaler().fit(filled)
        # PCA는 스케일에 민감하므로(분산이 큰 피처가 주성분을 지배) 반드시 표준화를 먼저 한다
        scaled = self.scaler_.transform(filled)
        self.pca_ = PCA(n_components=self.variance, svd_solver="full", random_state=self.random_state)
        # svd_solver="full": n_components에 분산비율(float)을 넣을 때 필요한 정확한 SVD 방식
        self.pca_.fit(scaled)
        self.output_columns_ = [f"pca_{i + 1:03d}" for i in range(self.pca_.n_components_)]
        # 압축 후에는 원래 컬럼명이 의미가 없어지므로 pca_001, pca_002... 형태로 새로 이름 붙임
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric_frame(X).reindex(columns=self.input_columns_)
        # 학습 때와 동일한 컬럼 순서·집합으로 맞춰야 StandardScaler·PCA 변환이 어긋나지 않는다
        filled = self.imputer_.transform(numeric)
        values = self.pca_.transform(self.scaler_.transform(filled))
        # 결측치 대체 -> 표준화 -> PCA 투영 순서를 fit() 때와 정확히 동일하게 재현
        return pd.DataFrame(values, columns=self.output_columns_, index=X.index)
        # PCA 출력(numpy 배열)에 pca_XXX 컬럼명과 원본 인덱스를 다시 붙여 DataFrame으로 복원


class TreeImportanceVariant(AllFeaturesVariant):
    """ExtraTrees 회귀 모델의 피처 중요도 상위 비율만 남기는 변형

    SHAP을 런타임 의존성으로 요구하지 않기 위해 ExtraTrees를 대신 사용한다.
    선택된 컬럼은 여전히 해석 가능하며, `importance_` 속성으로 전체 중요도 순위를
    확인할 수 있다. 필요하면 이후 이 트리 모델에 SHAP을 추가로 적용할 수도 있다.
    """

    name = "tree_importance"

    def __init__(self, keep_ratio: float = 0.35, min_features: int = 8, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.keep_ratio = keep_ratio
        # 전체 피처 중 상위 몇 %를 남길지 결정하는 비율
        self.min_features = min_features
        # keep_ratio가 너무 작아도 최소 이 개수만큼은 보장 (피처가 과도하게 줄어드는 것을 방지)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TreeImportanceVariant":
        # 다른 변형과 달리 이 변형은 타깃(y)을 실제로 사용한다 (지도학습 기반 중요도 산출이므로 del y 없음)
        numeric = self._numeric_frame(X)
        self.input_columns_ = numeric.columns.tolist()
        self.imputer_ = SimpleImputer(strategy="median").fit(numeric)
        filled = self.imputer_.transform(numeric)
        self.selector_model_ = ExtraTreesRegressor(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            max_features=1.0,
        )
        # n_jobs=-1: 가용 코어를 모두 사용해 학습 속도를 높임
        # max_features=1.0: 매 분할마다 전체 피처를 다 고려해 중요도 추정이 특정 피처 부분집합에 치우치지 않게 함
        self.selector_model_.fit(filled, np.asarray(y))
        self.importance_ = pd.Series(
            self.selector_model_.feature_importances_, index=self.input_columns_
        ).sort_values(ascending=False)
        # 컬럼명을 인덱스로 붙여 어떤 피처가 중요한지 바로 조회할 수 있는 형태로 정리
        keep = max(self.min_features, int(np.ceil(len(self.input_columns_) * self.keep_ratio)))
        # keep_ratio 기준 개수와 min_features 하한 중 더 큰 값을 채택
        keep = min(keep, len(self.input_columns_))
        # 전체 피처 수를 넘지 않도록 상한도 함께 적용
        self.output_columns_ = self.importance_.head(keep).index.tolist()
        # 중요도 내림차순 정렬된 importance_에서 상위 keep개의 컬럼명만 추출
        return self
    # transform()은 AllFeaturesVariant에서 상속 — output_columns_로 슬라이싱하는 로직 재사용


def make_feature_variant(name: str, **kwargs) -> FeatureVariant:
    """baseline_6 피처 변형을 이름으로 생성하는 팩토리 함수

    Args:
        - name: "all" / "corr_filtered" / "pca" / "tree_importance" (대소문자·공백 무시)
        - kwargs: 각 변형의 __init__에 그대로 전달할 하이퍼파라미터

    Returns:
        - 요청한 이름에 해당하는 FeatureVariant 하위 클래스 인스턴스

    Raises:
        - ValueError: 등록되지 않은 이름이면
    """

    variants = {
        "all": AllFeaturesVariant,
        "corr_filtered": CorrelationFilteredVariant,
        "pca": PCAVariant,
        "tree_importance": TreeImportanceVariant,
    }
    key = name.strip().lower()
    # 앞뒤 공백 제거 + 소문자 정규화로 "PCA", " pca " 같은 입력도 허용
    if key not in variants:
        raise ValueError(f"지원하지 않는 피처 버전: {name}. 사용 가능: {sorted(variants)}")
        # 어떤 이름이 왜 실패했는지, 사용 가능한 값이 무엇인지 에러 메시지에 함께 담아 디버깅을 돕는다
    return variants[key](**kwargs)


__all__ = [
    "FeatureVariant",
    "AllFeaturesVariant",
    "CorrelationFilteredVariant",
    "PCAVariant",
    "TreeImportanceVariant",
    "make_feature_variant",
]