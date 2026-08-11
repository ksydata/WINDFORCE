# Step 8. LDAPS·GFS·SCADA 공통 전처리 계약 (추상 클래스)
# 03_Preprocessing.ipynb의 WindVectorTransformer·TimeFeatureEngineer를 승격하고
# LSTM 입력(3차원 시퀀스) 생성까지 담당하도록 확장한 모듈

"""소스와 무관한 전처리 계약과 물리 계산, LSTM 입력 생성을 담당하는 모듈.

이 모듈은 **무엇을 어떤 순서로 하는가**만 정하고, **어떻게 하는가**는 소스별 자식 클래스에 맡긴다.
``transform``은 순서가 고정된 템플릿 메서드이고, 그 안에서 호출되는 단계들이 추상/훅 메서드다.

전처리의 최종 목표는 **LSTM이 바로 받아쓸 수 있는 3차원 배열**이다. 그래서 파이프라인은
격자 -> 그룹 -> 시간정렬 표 -> (samples, timesteps, features) 순으로 좁혀 들어간다.

    원본 CSV
      -> transform()            8단계 파이프라인 (소스별)
      -> transformToGroupFeature() 격자를 KPX 그룹으로 (LDAPS·GFS)
      -> transformForecastDiff()   시계열 변화량·램프
      -> transformToModelInput()   숫자형만 남긴 2차원 표
      -> transformToSequence()     (samples, timesteps, features) 3차원 배열

파이프라인 8단계는 03_Preprocessing 노트북의 Step 순서를 그대로 따른다.

| 순서 | 메서드 | 종류 | 노트북 Step |
|---|---|---|---|
| 1 | transformColumnName | 추상 | Step 2. 컬럼 형식 통일 |
| 2 | transformTimeAxis | 추상 | Step 2. 시간 형식 통일 |
| 3 | checkPhysicalLimit | 추상 | Step 4. 물리한계 플래그 |
| 4 | interpolateMissing | 훅 | Step 5·7. 이상치 NaN 변환 + 단기 보간 |
| 5 | calculatePhysicalFeature | 추상 | Step 8. 파생변수 생성 |
| 6 | checkDerivedFlag | 훅 | Step 8. 파생값 기반 플래그 |
| 7 | transformCyclicFeature | 훅 | Step 10-3. 주기성 인코딩 |
| 8 | selectGroup | 구체 | 그룹별 모델 입력 분리 |

훅(hook)은 기본 구현이 ``return df``인 구체 메서드다. 그 단계가 필요 없는 소스는 그냥 두면 되고,
필요한 소스만 오버라이드한다. 반대로 추상 메서드는 소스마다 반드시 달라서 기본값을 줄 수 없는 것들이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from .utils.spatial_utils import compute_group_weight


KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
# 그룹 이름 순서를 고정해 표·행렬 인덱스에서 계속 재사용한다
RATED_CAPACITY_KW = {
    "kpx_group_1": 21_600.0,
    # VESTAS V126 3.6MW x 6기 = 21.6MW = 21,600kW
    "kpx_group_2": 21_600.0,
    # VESTAS V126 3.6MW x 6기 = 21.6MW = 21,600kW
    "kpx_group_3": 21_000.0,
    # UNISON U136 4.2MW x 5기 = 21.0MW = 21,000kW
}

R_D = 287.05
# 건조공기 기체상수 J/(kg·K)
EPSILON = 0.62197
# 건조공기/수증기 기체상수 비(Rd/Rv). 수증기압 식의 분모에 들어간다
ONE_MINUS_EPS = 0.37803
# 1 - EPSILON. 0.622/0.378로 반올림하면 밀도 4번째 자리가 흔들려 상수로 못 박아둔다
RHO_STD = 1.225
# 표준 대기 공기밀도 kg/m³. 계산 결과가 현실적인지 비교하는 기준
MIN_SAMPLES_PER_HOUR = 6
# 10분값 6개가 1시간을 구성 (60분 / 10분 = 6)
MAX_INTERP_STEPS = 2
# 연속 결측 2칸(= 20분) 이하만 보간한다. 그보다 길면 정비·고장이라 손대지 않는다
EARTH_RADIUS_KM = 6371.0
# 하버사인 거리 계산에 쓰는 지구 평균 반지름(km)
IDW_K = 4
# 터빈마다 가까운 격자 4개만 사용한다. 전 격자를 다 쓰면 먼 격자가 값을 희석시킨다
IDW_POWER = 2.0
# 역거리 가중의 지수. 1/거리^2 (거리가 2배 멀어지면 영향력은 1/4)
MIN_DISTANCE_KM = 1e-6
# 터빈이 격자점과 정확히 겹칠 때 0으로 나누는 것을 방지하는 하한값
SEASON_MAP = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
# (월 % 12) // 3 결과를 계절 이름으로. 12·1·2월이 winter가 된다
SEQ_LEN = 24
# LSTM 기본 입력 길이. 예보 한 발행분이 24개 대상 시각을 덮으므로 하루치를 기본으로 둔다


@dataclass(frozen = True)
class GroupSpec:
    """KPX 한 그룹의 모델링 기준을 담는 데이터 클래스

    Attributes:
        - name: 그룹 이름 (kpx_group_1 / 2 / 3)
        - rated_capacity_kw: 그룹 설비용량(kW). 1시간 최대 발전량(kWh)과 같은 값이다
    """

    name: str
    rated_capacity_kw: float


class FeatureEngineer(ABC):
    """소스와 무관한 전처리 계약 및 물리 계산을 담당하는 추상 클래스

    Attributes:
        - source: 데이터 소스 식별자. 자식 클래스가 반드시 정의한다
        - COLUMN_MAP: 원본 컬럼명 -> 표준 컬럼명 매핑. 자식 클래스가 반드시 정의한다
        - group: None이면 전체 그룹, 값이 있으면 해당 KPX 그룹만 반환
        - group_spec: 그룹별 정격용량 기준값
    """

    source: ClassVar[str]
    COLUMN_MAP: ClassVar[dict[str, str]]
    # 값을 주지 않고 선언만 해둔다. 자식이 정의하지 않으면 __init_subclass__에서 걸린다

    def __init_subclass__(cls, **kwargs) -> None:
        """자식 클래스가 필수 클래스 속성을 정의했는지 확인하는 메서드

        Logic:
            - @abstractmethod는 메서드만 강제할 수 있고 클래스 속성은 강제하지 못한다
            - source·COLUMN_MAP 누락은 import 시점에 잡아야 15만 행 돌린 뒤에 안 터진다
        """
        super().__init_subclass__(**kwargs)
        for name in ("source", "COLUMN_MAP"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__}은 클래스 속성 '{name}'을 정의해야 합니다")

    def __init__(self, group: str | None = None):
        if group is not None and group not in KPX_GROUPS:
            raise ValueError(f"group은 {KPX_GROUPS} 중 하나여야 합니다: {group}")
        self.group = group
        self.group_spec = GroupSpec(group, RATED_CAPACITY_KW[group]) if group else None
        # 그룹별 모델은 group만 바꿔 같은 계약으로 생성한다

    # ------------------------------------------------------------------
    # 템플릿 메서드 (구체) — 순서를 고정하는 것이 이 클래스의 존재 이유다
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """원천 DataFrame을 모델 입력용 DataFrame으로 변환하는 템플릿 메서드

        Args:
            - df: 소스 원본 표. 필요한 컬럼은 자식 클래스의 COLUMN_MAP이 정의한다

        Logic:
            - 순서를 바꾸면 결과가 틀린다. 예를 들어 품질 플래그(3)를 파생변수(5)보다 뒤에 두면
              정격 초과 극단값이 섞인 채로 공기밀도·풍력에너지가 계산된다
            - 그룹 분리(8)는 반드시 마지막이다. 중간에 자르면 터빈 간 중앙값 비교가 깨진다
        """
        df = df.copy()
        # 원본을 변형하지 않는다. 노트북에서 셀을 여러 번 실행해도 결과가 같다

        df = self.transformColumnName(df)
        df = self.transformTimeAxis(df)
        df = self.checkPhysicalLimit(df)
        df = self.interpolateMissing(df)
        df = self.calculatePhysicalFeature(df)
        df = self.checkDerivedFlag(df)
        df = self.transformCyclicFeature(df)
        return self.selectGroup(df)
        # 8단계를 모두 거친 모델 입력용 DataFrame 반환

    # ------------------------------------------------------------------
    # 추상 메서드 — 소스마다 반드시 다르게 구현해야 하는 단계
    # ------------------------------------------------------------------

    @abstractmethod
    def transformColumnName(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 컬럼명을 표준 snake 이름으로 통일하는 메서드"""

    @abstractmethod
    def transformTimeAxis(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 컬럼을 datetime으로 바꾸고 소스별 시간 구조를 만드는 메서드"""

    @abstractmethod
    def checkPhysicalLimit(self, df: pd.DataFrame) -> pd.DataFrame:
        """원시값의 물리한계를 검사해 플래그 컬럼을 붙이는 메서드"""

    @abstractmethod
    def calculatePhysicalFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """소스별 물리 파생변수를 계산하는 메서드"""

    # ------------------------------------------------------------------
    # 훅 메서드 (구체) — 필요한 소스만 오버라이드한다
    # ------------------------------------------------------------------

    def interpolateMissing(self, df: pd.DataFrame) -> pd.DataFrame:
        """못 믿을 값을 NaN으로 바꾸고 단기 결측을 보간하는 메서드

        Logic:
            - LDAPS·GFS는 수치예보 모델 산출물이라 결측이 없다. 기본 구현은 아무것도 하지 않는다
        """
        return df
        # 손대지 않은 DataFrame 반환 (기본 훅)

    def checkDerivedFlag(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생변수를 근거로 하는 품질 플래그를 붙이는 메서드

        Logic:
            - 풍속 급변처럼 원시 U·V만으로는 못 재고 ws를 만든 뒤에야 계산되는 플래그가 여기 온다
        """
        return df
        # 손대지 않은 DataFrame 반환 (기본 훅)

    def transformCyclicFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """풍향·시각처럼 순환하는 값을 sin/cos으로 인코딩하는 메서드"""
        return df
        # 손대지 않은 DataFrame 반환 (기본 훅)

    # ------------------------------------------------------------------
    # 공통 물리 계산 (구체) — 소스와 무관한 바람·밀도 계산
    # ------------------------------------------------------------------

    def computeWindSpeed(self, u: pd.Series, v: pd.Series) -> pd.Series:
        """U·V 바람 성분으로 풍속을 계산하는 메서드

        Args:
            - u: 동서 성분(m/s). 동쪽으로 부는 바람이 +
            - v: 남북 성분(m/s). 북쪽으로 부는 바람이 +

        Logic:
            - ws = √(u² + v²)
        """
        return np.sqrt(u**2 + v**2)
        # 풍속(m/s) 시리즈 반환. 반대로 풍속·풍향에서 U·V로도 되돌릴 수 있다

    def computeWindDirection(self, u: pd.Series, v: pd.Series) -> pd.Series:
        """U·V 바람 성분으로 기상학적 풍향을 계산하는 메서드

        Args:
            - u: 동서 성분(m/s)
            - v: 남북 성분(m/s)

        Logic:
            - wd = (degrees( arctan2(-u, -v) ) + 360) % 360
            - 기상학적 풍향은 바람이 '불어오는' 방향이라 부호를 뒤집는다 (북 0, 동 90, 남 180, 서 270)
            - atan(v/u)를 쓰면 부호를 하나만 보므로 동북풍과 서남풍을 구분하지 못하고
              u = 0에서 0으로 나누는 문제가 생긴다
        """
        degree = np.degrees( np.arctan2(-u, -v) )
        return (degree + 360) % 360
        # 0~360도 범위의 기상학적 풍향 시리즈 반환

    def computeUVComponent(self, ws: pd.Series, wd: pd.Series) -> tuple[pd.Series, pd.Series]:
        """풍속과 기상학적 풍향을 U·V 성분으로 되돌리는 메서드

        Args:
            - ws: 풍속(m/s)
            - wd: 기상학적 풍향(0~360도)

        Logic:
            - u = -ws x sin(풍향), v = -ws x cos(풍향)
            - 각도를 직접 평균하면 0도와 359도의 평균이 180도가 되므로 성분으로 바꿔서 평균한다
        """
        radian = np.deg2rad(wd)
        return -ws * np.sin(radian), -ws * np.cos(radian)
        # (U 성분, V 성분) 튜플 반환

    def computeAngleDifference(self, wd_a: pd.Series, wd_b: pd.Series) -> pd.Series:
        """두 풍향의 차이를 -180~180도로 감아서 계산하는 메서드

        Args:
            - wd_a, wd_b: 기상학적 풍향(0~360도)

        Logic:
            - diff = (a - b + 180) % 360 - 180
            - 350도와 10도의 차이는 340도가 아니라 -20도다
        """
        return (wd_a - wd_b + 180) % 360 - 180
        # -180~180도 범위의 각도 차 시리즈 반환

    def computeWindShear(
        self, ws_low: pd.Series, ws_high: pd.Series, height_low: float, height_high: float
    ) -> pd.Series:
        """두 고도의 풍속으로 멱법칙 전단지수(alpha)를 계산하는 메서드

        Args:
            - ws_low: 낮은 고도 풍속(m/s)
            - ws_high: 높은 고도 풍속(m/s)
            - height_low, height_high: 각 풍속의 고도(m)

        Logic:
            - 멱법칙 V2/V1 = (h2/h1)^alpha  ->  alpha = ln(V2/V1) / ln(h2/h1)
            - alpha를 알면 10m 예보풍속을 허브 높이(117m)로 외삽할 수 있다.
              대회 평가 기간에 SCADA가 없으므로 이 외삽이 사실상 유일한 허브풍속 단서다
            - 무풍 구간에서 log(0)이 -inf가 되므로 분자·분모 모두 하한을 둔다
        """
        ratio = ws_high.clip(lower = 1e-3) / ws_low.clip(lower = 1e-3)
        # 0으로 나누는 경우와 log(0)을 동시에 막기 위해 1e-3 m/s를 하한으로 둔다
        return np.log(ratio) / np.log(height_high / height_low)
        # 전단지수 alpha 시리즈 반환 (개활지 0.14 안팎, 복잡지형은 더 크다)

    def calculateAirDensity(
        self, df: pd.DataFrame, t_col: str, q_col: str, p_col: str, prefix: str = ""
    ) -> pd.DataFrame:
        """기온·비습·기압으로 습윤공기 밀도를 계산하는 메서드

        Args:
            - t_col: 기온 컬럼 (K 단위)
            - q_col: 비습 컬럼 (kg/kg 단위)
            - p_col: 지표면 기압 컬럼 (Pa 단위)
            - prefix: 결과 컬럼 접두사. LDAPS와 GFS를 한 표에 합칠 때 이름 충돌을 막는다

        Logic:
            - 수증기압      e   = (q x P) / (EPSILON + (1 - EPSILON) x q)
            - 습윤공기 밀도 rho = (P - 0.378 x e) / (R_d x T)
            - 단위를 먼저 확정해야 한다. 기온이 섭씨거나 기압이 hPa면 결과가 통째로 틀린다
        """
        required = {t_col, q_col, p_col}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"공기밀도 계산에 필요한 컬럼이 없습니다: {sorted(missing)}")

        df = df.copy()
        df[f"{prefix}e_pa"] = (df[q_col] * df[p_col]) / (EPSILON + ONE_MINUS_EPS * df[q_col])
        # 비습 -> 수증기압(Pa). 중간값도 남겨야 밀도가 이상할 때 어디서 틀렸는지 추적할 수 있다
        df[f"{prefix}air_density"] = (
            (df[p_col] - 0.378 * df[f"{prefix}e_pa"]) / (R_D * df[t_col])
        )
        return df
        # e_pa(Pa)와 air_density(kg/m³) 두 컬럼이 추가된 DataFrame 반환

    def calculateWindPowerDensity(self, air_density: pd.Series, ws: pd.Series) -> pd.Series:
        """공기밀도와 풍속으로 풍력에너지 밀도를 계산하는 메서드

        Args:
            - air_density: 공기밀도(kg/m³)
            - ws: 풍속(m/s)

        Logic:
            - 풍력에너지 밀도 = 0.5 x rho x V³ (W/m²)
            - 발전량이 여기 정비례한다고 강제하지 않고 입력변수로만 쓴다
        """
        return 0.5 * air_density * ws**3
        # 단위면적당 바람이 가진 힘(W/m²) 시리즈 반환

    # ------------------------------------------------------------------
    # 공통 인코딩·보간 (구체)
    # ------------------------------------------------------------------

    def transformWindDirection(
        self, df: pd.DataFrame, wind_direction_cols: list[str]
    ) -> pd.DataFrame:
        """풍향 컬럼에 주기성 인코딩(sin/cos)을 붙이는 메서드

        Args:
            - df: 풍향 컬럼을 가진 DataFrame
            - wind_direction_cols: 인코딩할 풍향 컬럼명 목록 (도 단위)
        """
        df = df.copy()
        for col in wind_direction_cols:
            if col not in df.columns:
                continue
                # 소스마다 있는 풍향이 달라 없는 컬럼은 조용히 건너뛴다
            radian = np.deg2rad(df[col])
            df[f"{col}_sin"] = np.sin(radian)
            df[f"{col}_cos"] = np.cos(radian)
            # 머신러닝의 0도-360도 연속성 인지를 위한 주기성 인코딩 (Sin, Cos)
            # 원본 0~360도 스케일은 두 값을 가장 먼 값으로 취급해 수치 왜곡을 일으킨다
        return df
        # 풍향마다 _sin, _cos 두 컬럼이 추가된 DataFrame 반환

    def transformCyclicTime(
        self, df: pd.DataFrame, time_col: str = "forecast_kst_dtm"
    ) -> pd.DataFrame:
        """시각을 일·연 주기 피처와 계절 라벨로 변환하는 메서드

        Args:
            - df: 시간 컬럼을 가진 DataFrame
            - time_col: 기준 시간 컬럼명

        Logic:
            - hour_sin/cos = sin·cos(2π x 시 / 24)         -> 23시와 0시를 붙인다
            - doy_sin/cos  = sin·cos(2π x 연중일 / 365.25) -> 12월 31일과 1월 1일을 붙인다
            - 365가 아니라 365.25로 나누는 것은 윤년 누적 오차를 없애기 위함이다
        """
        if time_col not in df.columns:
            raise KeyError(f"시간 컬럼이 없습니다: {time_col}")
        df = df.copy()
        dt = pd.to_datetime(df[time_col])

        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
        df["month"] = dt.dt.month
        df["season"] = (dt.dt.month % 12 // 3).map(SEASON_MAP)
        # 발전량은 월별 패턴이 반복되므로 계절 라벨도 같이 남긴다
        return df
        # 시간 순환 피처 6개가 추가된 DataFrame 반환

    def interpolateShortGap(
        self,
        df: pd.DataFrame,
        col: str,
        key: str = "turbine",
        max_steps: int = MAX_INTERP_STEPS,
    ) -> tuple[pd.Series, pd.Series]:
        """연속 결측이 짧은 구멍만 선형보간하는 메서드

        Args:
            - df: 보간 대상 컬럼과 경계 컬럼을 가진 DataFrame (key·시간순 정렬 상태여야 한다)
            - col: 보간할 컬럼명
            - key: 구간이 넘어가면 안 되는 경계 컬럼명 (터빈이 바뀌면 구간도 끊긴다)
            - max_steps: 이 칸수 이하의 연속 결측만 채운다

        Logic:
            - 결측 구간 **전체 길이**를 먼저 재고, 길이가 max_steps 이하인 구간만 채운다
            - pandas의 interpolate(limit = n)만 쓰면 긴 구멍의 앞 n칸이 채워진다.
              정비·고장으로 3일 비어 있는 구간의 앞 20분이 메워지면 그 시간은 실제 저발전으로 학습된다
            - limit_area = "inside"라 양쪽에 값이 있을 때만 채운다 (표 끝의 결측은 안 건드린다)
        """
        na_mask = df[col].isna()
        same_run = na_mask.eq( na_mask.shift() ) & df[key].eq( df[key].shift() )
        run_len = df.groupby( (~same_run).cumsum() )[col].transform("size")
        short_gap = na_mask & (run_len <= max_steps)
        # 구간 길이로 먼저 걸러야 긴 구멍의 앞부분이 새어나가지 않는다

        interpolated = df.groupby(key, observed = True)[col].transform(
            lambda s: s.interpolate(method = "linear", limit_area = "inside")
        )
        filled = df[col].where(~short_gap, interpolated)
        return filled, (short_gap & filled.notna())
        # (보간을 마친 값, 실제로 채운 자리 표시) 튜플 반환

    # ------------------------------------------------------------------
    # 공간 가중 (구체) — 격자 예보를 KPX 그룹으로 좁히는 공통 절차
    # ------------------------------------------------------------------

    def computeHaversineDistance(
        self,
        lat1: float | np.ndarray,
        lon1: float | np.ndarray,
        lat2: float | np.ndarray,
        lon2: float | np.ndarray,
    ) -> np.ndarray:
        """위경도 두 지점 사이의 지구 표면 거리를 계산하는 메서드

        Args:
            - lat1, lon1: 기준 지점의 위도·경도 (도 단위, scalar 또는 shape = (n, ))
            - lat2, lon2: 대상 지점의 위도·경도 (도 단위, scalar 또는 shape = (n, ))

        Logic:
            - a = sin²(Δφ/2) + cos(φ1) x cos(φ2) x sin²(Δλ/2)
            - d = 2 x R x arcsin(√a),  R = 6371 km
            - 태백 산지는 위도 37도라 위경도 1도의 실제 거리가 다르다. 유클리드 거리를 쓰면 안 된다
        """
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = p2 - p1
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin( np.sqrt(a) )
        # 두 지점 사이의 대권 거리(km) 반환

    def computeGroupWeight(
        self,
        turbine_meta: pd.DataFrame,
        grid_coords: pd.DataFrame,
        k: int = IDW_K,
        power: float = IDW_POWER,
    ) -> np.ndarray:
        """그룹 x 격자 역거리 가중치 행렬을 계산하는 메서드

        Args:
            - turbine_meta: 터빈 1기 = 1행. `group`, `lat`, `lon`, `cap_kw` 컬럼 필수
            - grid_coords: 격자 1개 = 1행. `latitude`, `longitude` 컬럼 필수 (index = grid_id)
            - k: 터빈마다 사용할 이웃 격자 개수
            - power: 역거리 가중의 지수

        Logic:
            - 터빈별 IDW  w_ij = (1 / d_ij^power) / Σ(1 / d_ij^power), 가까운 k개 격자만
            - 그룹별 집계 W_gj = Σ_i∈g (cap_i x w_ij) / Σ_i∈g cap_i
            - 격자를 통째로 평균하면 세 그룹에 똑같은 예보값이 들어가 거리·고도·바람장 차이가 사라진다
        """
        return compute_group_weight(turbine_meta, grid_coords, k = k, power = power)
        # 공통 유틸의 그룹 키 정규화·영행렬 가드를 재사용해 두 구현의 동작 차이를 막는다

    def transformToGroupFeature(
        self,
        df: pd.DataFrame,
        features: list[str],
        weight_group: np.ndarray,
        grid_index: pd.Index,
        time_col: str = "forecast_kst_dtm",
    ) -> pd.DataFrame:
        """격자별 값을 그룹별 공간 가중평균으로 바꾸는 메서드

        Args:
            - df: 한 행 = 한 격자 x 한 시각인 예보 표
            - features: 가중평균할 컬럼 목록 (격자별로 이미 계산을 마친 파생변수)
            - weight_group: computeGroupWeight의 결과. shape = (3그룹, 격자수)
            - grid_index: 가중치 행렬의 열 순서를 정의하는 grid_id 인덱스
            - time_col: 예보 대상 시각 컬럼명

        Logic:
            - (시각 x 격자) @ (격자 x 3그룹) 행렬곱 한 번으로 전체 기간을 처리한다
            - 풍향처럼 각도 값은 여기에 넣지 않는다. U·V를 가중평균한 뒤 방향을 복원해야 한다
        """
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"공간 가중평균에 필요한 컬럼이 없습니다: {missing}")

        times = np.sort(df[time_col].unique())
        idw_values = {}
        for feature in features:
            matrix = (
                df.pivot(index = time_col, columns = "grid_id", values = feature)
                .reindex(columns = grid_index)
            )
            idw_values[feature] = matrix.to_numpy() @ weight_group.T
            # (시각 x 격자) @ (격자 x 3그룹) = (시각 x 그룹)

        target_groups = [self.group] if self.group else list(KPX_GROUPS)
        group_frames = []
        for j, group in enumerate(KPX_GROUPS):
            if group not in target_groups:
                continue
            part = pd.DataFrame({feature: idw_values[feature][:, j] for feature in features})
            part.insert(0, "group", group)
            part.insert(0, time_col, times)
            group_frames.append(part)

        return pd.concat(group_frames, ignore_index = True)
        # 한 행 = 한 그룹 x 한 시각인 표 반환

    def transformGroupTimeFeature(
        self,
        group_df: pd.DataFrame,
        grid_df: pd.DataFrame,
        time_meta_cols: list[str],
        time_col: str = "forecast_kst_dtm",
    ) -> pd.DataFrame:
        """격자 표의 시간 메타를 그룹 표에 다시 붙이고 순환 인코딩하는 메서드

        Args:
            - group_df: transformToGroupFeature로 그룹 단위가 된 표
            - grid_df: 시간 메타를 갖고 있는 격자 단위 원본 표
            - time_meta_cols: 되살릴 시간 메타 컬럼 목록 (lead_hour, target_hour 등)
            - time_col: 예보 대상 시각 컬럼명

        Logic:
            - 공간 가중평균은 지정한 피처만 통과시키므로 시간 메타와 순환 인코딩이 통째로 날아간다.
              시각당 하나뿐인 값이라 groupby(시각).first()로 되살린 뒤 다시 인코딩한다
            - 순환 인코딩을 격자 단위에서 미리 해봤자 같은 값이 16번 반복될 뿐이라
              그룹으로 좁힌 다음에 하는 편이 낫다
        """
        cols = [c for c in time_meta_cols if c in grid_df.columns]
        if cols:
            time_meta = grid_df.groupby(time_col)[cols].first().reset_index()
            group_df = group_df.merge(time_meta, on = time_col, how = "left")
            # 격자마다 같은 값이므로 시각 단위로 하나만 꺼내온다
        return self.transformCyclicTime(group_df, time_col = time_col)
        # 시간 메타와 순환 피처가 복원된 그룹 단위 DataFrame 반환

    # ------------------------------------------------------------------
    # 시계열 파생 (구체) — 예보 소스 공통
    # ------------------------------------------------------------------

    def transformForecastDiff(
        self,
        df: pd.DataFrame,
        ws_col: str,
        u_col: str,
        v_col: str,
        wd_col: str,
        key: str = "group",
        prefix: str = "",
    ) -> pd.DataFrame:
        """예보 시계열의 변화량·램프 피처를 만드는 메서드

        Args:
            - df: 그룹별 시간순으로 **정렬을 마친** 예보 표 (정렬은 호출부 책임이다)
            - ws_col, u_col, v_col, wd_col: 풍속·U·V·풍향 컬럼명 (소스마다 다르다)
            - key: 변화량 경계 컬럼. 그룹이 바뀌면 시계열도 끊긴다
            - prefix: 생성 컬럼 접두사. LDAPS와 GFS를 한 표에 합쳐도 안 부딪히게 한다

        Logic:
            - ws_diff_1h/3h : 램프(급격한 증감) 구간 탐지용 변화량
            - ws_next3h_max/min : 앞으로 3시간 안에 닥칠 최대·최소 풍속 (미래 예보라 누수가 아니다)
            - wd_rotation : 풍향 회전 속도. -180~180으로 감은 뒤 절댓값
            - 반드시 key별로 계산한다. 안 그러면 그룹 경계에서 엉뚱한 차이가 생긴다
        """
        df = df.copy()
        by_key = df.groupby(key, observed = True)
        name = f"{prefix}_" if prefix else ""

        df[f"{name}ws_diff_1h"] = by_key[ws_col].diff(1)
        df[f"{name}ws_diff_3h"] = by_key[ws_col].diff(3)
        df[f"{name}u_diff_1h"] = by_key[u_col].diff(1)
        df[f"{name}v_diff_1h"] = by_key[v_col].diff(1)
        df[f"{name}ws_ramp_rate"] = df[f"{name}ws_diff_1h"].abs()
        # 풍속 램프율(변화 속도). 부호를 지운 크기만 본다

        df[f"{name}ws_next3h_max"] = by_key[ws_col].transform(
            lambda s: s.shift(-1).rolling(3, min_periods = 1).max()
        )
        df[f"{name}ws_next3h_min"] = by_key[ws_col].transform(
            lambda s: s.shift(-1).rolling(3, min_periods = 1).min()
        )
        # 예보는 미래 값을 미리 아는 자료라 앞을 보는 것이 데이터 누수가 아니다

        df[f"{name}wd_rotation"] = self.computeAngleDifference(
            df[wd_col], by_key[wd_col].shift(1)
        ).abs()
        return df
        # 변화량·램프 피처 8종이 추가된 DataFrame 반환

    # ------------------------------------------------------------------
    # LSTM 입력 생성 (구체) — 전처리의 최종 목표
    # ------------------------------------------------------------------

    def selectGroup(self, df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
        """설정된 KPX 그룹만 선택하는 메서드

        Args:
            - df: 그룹 컬럼을 포함한 DataFrame
            - group_col: 그룹 식별자 컬럼명

        Logic:
            - 그룹별 모델은 다른 그룹의 발전량·SCADA 파생값을 보지 않아야 하므로
              변환이 다 끝난 마지막 단계에서 그룹을 제한한다
        """
        if self.group is None or group_col not in df.columns:
            return df.copy()
        return df.loc[df[group_col].eq(self.group)].copy()
        # 설정된 그룹의 행만 남긴 DataFrame 반환

    def transformToModelInput(
        self,
        df: pd.DataFrame,
        time_col: str = "forecast_kst_dtm",
        key: str = "group",
        drop_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """LSTM이 받아들일 수 있는 숫자형 2차원 표로 정리하는 메서드

        Args:
            - df: 파생변수 생성을 마친 그룹 x 시각 표
            - time_col: 예보 대상 시각 컬럼명
            - key: 그룹 식별자 컬럼명
            - drop_cols: 피처에서 빼고 싶은 컬럼 목록 (예: 라벨 누수 위험이 있는 SCADA 컬럼)

        Logic:
            - bool 플래그는 0/1 int8로 바꾼다. LSTM은 True/False를 그대로 못 먹는다
            - season 같은 문자열은 원핫으로 편다
            - datetime과 식별자는 피처에서 빼되 컬럼 자체는 남긴다 (시퀀스를 자를 때 필요하다)
            - 정렬은 (그룹, 시각) 순으로 고정한다. 시퀀스가 시간 역순으로 잘리는 사고를 막는다
        """
        df = df.copy()
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
        sort_keys = [c for c in (key, time_col) if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys, ignore_index = True)

        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype("int8")
                # 품질 플래그도 모델이 참고할 수 있는 정보라 버리지 않고 0/1로 바꾼다

        if "season" in df.columns:
            season = pd.Categorical(df["season"], categories = list(SEASON_MAP.values()))
            season_dummy = pd.get_dummies(season, prefix = "season", dtype = "int8")
            season_dummy.index = df.index
            df = pd.concat([df.drop(columns = ["season"]), season_dummy], axis = 1)
            # 계절은 순서가 없는 범주라 숫자 하나로 넣으면 winter < spring 같은 가짜 순서가 생긴다
            # 범주를 4개로 고정하는 것이 핵심이다. 관측된 값만 펴면 겨울만 있는 평가 구간에서
            # 컬럼이 1개만 생겨 학습 때와 입력 차원이 달라진다

        if drop_cols:
            df = df.drop(columns = [c for c in drop_cols if c in df.columns])
        return df
        # 숫자형 피처와 식별자(time_col·key)만 남은 DataFrame 반환

    def computeFeatureColumn(
        self,
        df: pd.DataFrame,
        time_col: str = "forecast_kst_dtm",
        key: str = "group",
        target_col: str | None = None,
    ) -> list[str]:
        """모델 입력으로 쓸 숫자형 피처 컬럼 목록을 뽑는 메서드

        Args:
            - df: transformToModelInput을 거친 표
            - time_col, key: 피처에서 제외할 식별자 컬럼명
            - target_col: 피처에서 제외할 정답 컬럼명

        Logic:
            - 숫자형이 아닌 컬럼(datetime·object)은 자동으로 빠진다
            - 정답을 피처에 남기면 그대로 누수라 이름으로 확실히 제거한다
        """
        exclude = {time_col, key, target_col} - {None}
        return [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        # 피처로 쓸 컬럼 이름 리스트 반환

    def computeScaleParam(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> dict[str, tuple[float, float]]:
        """표준화에 쓸 평균·표준편차를 계산하는 메서드

        Args:
            - df: **학습 구간만** 잘라낸 표. 검증·평가 구간이 섞이면 누수다
            - feature_cols: 표준화할 컬럼 목록

        Logic:
            - z = (x - mean) / std
            - 상수 컬럼은 std가 0이라 1.0으로 대체한다 (0으로 나누는 것을 방지)
        """
        param = {}
        for col in feature_cols:
            mean = float(df[col].mean())
            std = float(df[col].std())
            param[col] = (mean, std if std > 1e-8 else 1.0)
            # std가 사실상 0인 상수 컬럼은 나눗셈을 무효화해 값을 그대로 통과시킨다
        return param
        # {컬럼: (평균, 표준편차)} 딕셔너리 반환

    def transformScale(
        self, df: pd.DataFrame, scale_param: dict[str, tuple[float, float]]
    ) -> pd.DataFrame:
        """학습 구간 통계로 표준화하는 메서드

        Args:
            - df: 표준화할 표 (학습·검증·평가 어느 구간이든 가능)
            - scale_param: computeScaleParam이 학습 구간에서 만든 통계

        Logic:
            - 평가 구간도 반드시 **학습 구간 통계**로 변환한다.
              구간마다 다시 계산하면 미래 정보가 새어들어온다
        """
        df = df.copy()
        for col, (mean, std) in scale_param.items():
            if col in df.columns:
                df[col] = (df[col] - mean) / std
        return df
        # 표준화된 DataFrame 반환

    def transformToSequence(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None = None,
        seq_len: int = SEQ_LEN,
        key: str = "group",
        time_col: str = "forecast_kst_dtm",
        dropna: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
        """2차원 표를 LSTM 입력용 3차원 시퀀스로 자르는 메서드

        Args:
            - df: transformToModelInput·transformScale을 마친 (그룹 x 시각) 표
            - feature_cols: 입력 피처 컬럼 목록
            - target_col: 정답 컬럼명. None이면 X와 인덱스만 반환한다 (평가 데이터용)
            - seq_len: 한 시퀀스의 길이 (기본 24 = 하루)
            - key: 시퀀스가 넘어가면 안 되는 경계 컬럼명
            - time_col: 예보 대상 시각 컬럼명

        Logic:
            - X shape = (samples, seq_len, n_features), y shape = (samples, )
            - y는 **창의 마지막 시각** 값이다. 과거 seq_len시간을 보고 그 시점을 맞추는 구조
            - 그룹 경계를 넘는 창은 만들지 않는다. 그룹1 끝과 그룹2 시작이 이어지면 안 된다
            - dropna = True면 창 안에 결측이 하나라도 있는 표본을 버린다.
              LSTM은 NaN을 만나면 손실이 통째로 NaN이 되어 학습이 죽는다
        """
        if seq_len < 1:
            raise ValueError("seq_len은 1 이상이어야 합니다")

        x_parts, y_parts, index_parts = [], [], []
        groups = df[key].unique() if key in df.columns else [None]

        for group in groups:
            part = df if group is None else df.loc[df[key].eq(group)]
            part = part.sort_values(time_col) if time_col in part.columns else part
            if len(part) < seq_len:
                continue
                # 창 하나도 못 만드는 짧은 그룹은 건너뛴다 (그룹3의 초기 구간)

            values = part[feature_cols].to_numpy(dtype = "float32")
            window = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis = 0)
            window = window.transpose(0, 2, 1)
            # sliding_window_view는 (samples, features, seq_len)으로 내주므로 축을 바꾼다

            tail = part.iloc[seq_len - 1:]
            # 각 창의 마지막 시각. 정답과 인덱스는 여기서 가져온다

            x_parts.append(window)
            index_parts.append(tail[[c for c in (key, time_col) if c in part.columns]])
            if target_col is not None:
                y_parts.append(tail[target_col].to_numpy(dtype = "float32"))

        if not x_parts:
            raise ValueError(f"시퀀스를 만들 수 있는 구간이 없습니다 (seq_len = {seq_len})")

        x_seq = np.concatenate(x_parts, axis = 0)
        index_df = pd.concat(index_parts, ignore_index = True)
        y_seq = np.concatenate(y_parts, axis = 0) if target_col is not None else None

        if dropna:
            keep = ~np.isnan(x_seq).any(axis = (1, 2))
            if y_seq is not None:
                keep &= ~np.isnan(y_seq)
            x_seq = x_seq[keep]
            y_seq = y_seq[keep] if y_seq is not None else None
            index_df = index_df.loc[keep].reset_index(drop = True)
            # 창 하나에 결측이 하나라도 있으면 그 표본을 통째로 버린다

        return x_seq, y_seq, index_df
        # (X 3차원 배열, y 1차원 배열 또는 None, 창 마지막 시각 인덱스 표) 반환

    def clipPower(self, df: pd.DataFrame, power_col: str = "scada_power_kwh") -> pd.DataFrame:
        """그룹 정격용량 범위로 발전량을 제한하는 메서드

        Args:
            - df: 발전량 컬럼을 가진 DataFrame
            - power_col: 제한할 발전량 컬럼명 (kWh 단위)

        Logic:
            - 0 <= 발전량 <= 그룹 설비용량(kWh)
            - **전처리 단계에서 부르면 안 된다.** 정격을 넘는 실측은 센서 오류라는 증거이므로
              잘라서 감추지 말고 is_power_outlier 플래그로 남겨야 한다 (노트북 Step 4·12)
            - 이 메서드는 최종 예측값을 제출 범위로 맞출 때만 쓴다
        """
        if self.group_spec is None or power_col not in df.columns:
            return df.copy()
        df = df.copy()
        df[power_col] = df[power_col].clip(lower = 0, upper = self.group_spec.rated_capacity_kw)
        return df
        # 0~그룹 설비용량으로 잘린 발전량을 담은 DataFrame 반환


__all__ = [
    "FeatureEngineer", "GroupSpec",
    "KPX_GROUPS", "RATED_CAPACITY_KW",
    "R_D", "EPSILON", "ONE_MINUS_EPS", "RHO_STD",
    "MIN_SAMPLES_PER_HOUR", "MAX_INTERP_STEPS",
    "EARTH_RADIUS_KM", "IDW_K", "IDW_POWER",
    "SEASON_MAP", "SEQ_LEN",
]
