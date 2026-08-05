"""GFS 예보의 컬럼 통일과 다층 바람 파생변수 생성을 담당하는 모듈.

GFS를 쓰는 이유는 단 하나, **고도**다. LDAPS는 10m와 50m(최대·최소)만 주는데
GFS는 80m·100m 순간값을 준다. 허브 높이가 117m라 100m 바람이 가장 가깝고,
10m와 100m를 함께 쓰면 멱법칙 전단지수(alpha)를 직접 추정할 수 있다.
평가 기간에 SCADA가 없으므로 이 외삽이 사실상 유일한 허브풍속 단서다.

컬럼 이름에는 전부 ``gfs_`` 접두사를 붙인다. LDAPS도 10m U·V와 2m 기온을 갖고 있어
한 표에 합칠 때 이름이 부딪히기 때문이다.

    preprocess()           격자별 파생변수 + 품질 플래그 (템플릿은 부모가 정의)
      -> transformToGroupIDW()   9격자 -> KPX 그룹 (역거리 가중)
      -> transformForecastFeature() 시계열 변화량·램프
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from .FeatureEngineer import BasePreprocessor
from .utils.spatial_utils import (
    IDW_K,
    IDW_POWER,
    PreprocessingError,
    compute_group_weight,
    transform_to_group_feature,
)
from .utils.time_utils import (
    transform_cyclic_time,
    transform_forecast_diff,
    transform_group_time_feature,
)
from .utils.wind_utils import (
    calculate_air_density,
    calculate_wind_power_density,
    compute_wind_direction,
    compute_wind_shear,
    compute_wind_speed,
    transform_wind_direction,
)


HUB_HEIGHT_M = 117.0
# V126 허브 높이(m). 100m 바람이 가장 가깝고 여기까지 외삽해서 쓴다
RH_PHYS_MAX = 100.0
# 상대습도 물리한계(%). 수치예보 산출물이라 100%를 살짝 넘는 값이 나올 수 있다
GUST_RATIO_MAX = 5.0
# 돌풍이 평균풍속의 5배를 넘으면 비물리적이다. 값은 두고 플래그만 남긴다


class GFSFeatureEngineer(BasePreprocessor):
    """GFS 예보의 다층 바람 파생변수와 KPX 그룹 공간 가중을 담당하는 클래스

    Attributes:
        - 지상 2m   : 기온·이슬점·상대습도·비습 (gfs_t2m / gfs_dpt2m / gfs_rh2m / gfs_q2m)
        - 지상 10m  : 순간값 U·V — LDAPS와 비교할 기준 고도
        - 지상 80m  : 순간값 U·V — 허브 높이 근처
        - 지상 100m : 순간값 U·V — 허브 높이(117m)에 가장 가깝다
        - 행성경계층(PBL) : U·V와 환기율(VRATE)
        - 상층 850/700/500 hPa : 기온·U·V (상층 바람이 지상으로 내려오는 혼합 강도의 단서)
        - 지표면    : 기압·복사·강수·돌풍·운량
    """

    source: ClassVar[str] = "gfs"

    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "heightAboveGround_10_10u": "gfs_u10",
        # 지상 10m 동서(U) 바람 (m/s)
        "heightAboveGround_10_10v": "gfs_v10",
        # 지상 10m 남북(V) 바람 (m/s)
        "heightAboveGround_80_u": "gfs_u80",
        "heightAboveGround_80_v": "gfs_v80",
        # 지상 80m 바람. LDAPS에는 없는 고도다
        "heightAboveGround_100_100u": "gfs_u100",
        "heightAboveGround_100_100v": "gfs_v100",
        # 지상 100m 바람. 허브 높이 117m에 가장 가깝다
        "heightAboveGround_2_2t": "gfs_t2m",
        # 지상 2m 기온 (K)
        "heightAboveGround_2_2d": "gfs_dpt2m",
        # 지상 2m 이슬점온도 (K)
        "heightAboveGround_2_2r": "gfs_rh2m",
        # 지상 2m 상대습도 (%)
        "heightAboveGround_2_2sh": "gfs_q2m",
        # 지상 2m 비습 (kg/kg)
        "planetaryBoundaryLayer_0_u": "gfs_pbl_u",
        "planetaryBoundaryLayer_0_v": "gfs_pbl_v",
        # 행성경계층 바람. 지표 마찰이 미치는 층 전체의 대표 바람이다
        "planetaryBoundaryLayer_0_VRATE": "gfs_vrate",
        # 환기율(m²/s). 경계층 높이 x 평균풍속이라 혼합이 얼마나 활발한지를 나타낸다
        "surface_0_sp": "gfs_sp",
        # 지표면 기압 (Pa)
        "meanSea_0_prmsl": "gfs_prmsl",
        # 해면기압 (Pa)
        "surface_0_gust": "gfs_gust",
        # 돌풍 풍속 (m/s)
        "surface_0_dswrf": "gfs_dswrf",
        "surface_0_dlwrf": "gfs_dlwrf",
        # 하향 단파·장파 복사. 지표 가열 -> 대기 안정도와 연결된다
        "surface_0_prate": "gfs_prate",
        "surface_0_tp": "gfs_tp",
        # 강수강도와 총강수량
        "lowCloudLayer_0_lcc": "gfs_lcc",
        "middleCloudLayer_0_mcc": "gfs_mcc",
        "highCloudLayer_0_hcc": "gfs_hcc",
        "atmosphere_0_tcc": "gfs_tcc",
        # 하층·중층·상층·전운량 (%)
        "isobaricInhPa_850_t": "gfs_t850",
        "isobaricInhPa_850_u": "gfs_u850",
        "isobaricInhPa_850_v": "gfs_v850",
        "isobaricInhPa_850_r": "gfs_rh850",
        # 850hPa(약 1.5km) 상층. 지상과의 기온차가 대기 안정도를 말해준다
        "isobaricInhPa_700_t": "gfs_t700",
        "isobaricInhPa_700_u": "gfs_u700",
        "isobaricInhPa_700_v": "gfs_v700",
        "isobaricInhPa_500_gh": "gfs_gh500",
        "isobaricInhPa_500_t": "gfs_t500",
        "isobaricInhPa_500_u": "gfs_u500",
        "isobaricInhPa_500_v": "gfs_v500",
        # 500hPa(약 5.5km) 상층. 종관 규모 기압계의 위치를 나타낸다
    }

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
    )
    # 10m U·V와 시간·격자 구조만 필수로 한다. 고도별 자료가 빠진 축약 입력도 처리한다

    TIME_COLS: ClassVar[list[str]] = ["forecast_kst_dtm", "data_available_kst_dtm"]
    # 예보 대상 시각과 예보 사용 가능 시각. 둘의 차이가 선행시간(lead_hour)이다

    UV_LEVELS: ClassVar[dict[str, tuple[str, str]]] = {
        "10": ("gfs_u10", "gfs_v10"),
        "80": ("gfs_u80", "gfs_v80"),
        "100": ("gfs_u100", "gfs_v100"),
        "pbl": ("gfs_pbl_u", "gfs_pbl_v"),
        "850": ("gfs_u850", "gfs_v850"),
        "700": ("gfs_u700", "gfs_v700"),
        "500": ("gfs_u500", "gfs_v500"),
    }
    # Key: 생성될 변수의 고도·레벨 표시. gfs_ws_{key} / gfs_wd_{key} 로 펴진다
    # Value: (동서방향 U 성분 컬럼명, 남북방향 V 성분 컬럼명)
    # 레벨이 7개든 20개든 이 딕셔너리 한 줄만 추가하면 된다

    IDW_FEATURES: ClassVar[list[str]] = [
        "gfs_u10", "gfs_v10", "gfs_ws_10",
        "gfs_u80", "gfs_v80", "gfs_ws_80",
        "gfs_u100", "gfs_v100", "gfs_ws_100",
        # 고도별 바람. 허브 높이 외삽의 재료다
        "gfs_ws_hub", "gfs_shear_alpha", "gfs_air_density", "gfs_wind_power_density",
        # 전단·밀도·에너지
        "gfs_pbl_u", "gfs_pbl_v", "gfs_ws_pbl", "gfs_vrate",
        # 경계층
        "gfs_t2m", "gfs_dpt2m", "gfs_rh2m", "gfs_q2m", "gfs_t_dew_diff",
        # 기온·습도
        "gfs_sp", "gfs_prmsl", "gfs_gust", "gfs_gust_factor",
        # 기압·돌풍
        "gfs_dswrf", "gfs_dlwrf", "gfs_prate", "gfs_tp",
        # 복사·강수
        "gfs_lcc", "gfs_mcc", "gfs_hcc", "gfs_tcc",
        # 운량
        "gfs_u850", "gfs_v850", "gfs_ws_850", "gfs_t850", "gfs_rh850",
        "gfs_u500", "gfs_v500", "gfs_ws_500", "gfs_gh500",
        "gfs_lapse_850", "gfs_bulk_shear_850",
        # 상층·안정도
    ]
    # 격자별로 이미 계산을 마친 값만 넣는다. 풍향(각도)은 넣지 않고 U·V에서 복원한다

    def __init__(
        self,
        group: str | None = None,
        k: int = IDW_K,
        power: float = IDW_POWER,
    ):
        super().__init__(group = group)
        self.k = k
        self.power = power

    # ------------------------------------------------------------------
    # 추상 메서드 구현 — preprocess 템플릿이 순서대로 호출한다
    # ------------------------------------------------------------------

    def transformColumnName(self, df: pd.DataFrame) -> pd.DataFrame:
        """긴 원본 컬럼명을 gfs_ 접두사가 붙은 짧은 이름으로 바꾸는 메서드

        Args:
            - df: GFS 원본 표 (heightAboveGround_100_100u 같은 긴 컬럼명)

        Logic:
            - LDAPS도 10m U·V와 2m 기온을 갖고 있어 접두사 없이 합치면 이름이 부딪힌다
            - df.filter(like = "gfs_")로 GFS 컬럼만 통째로 뽑을 수 있게 된다
        """
        return df.rename(
            columns = {k: v for k, v in self.COLUMN_MAP.items() if k in df.columns}
        )
        # gfs_u10·gfs_t2m 같은 이름으로 통일된 DataFrame 반환

    def transformTimeAxis(self, df: pd.DataFrame) -> pd.DataFrame:
        """예보 시간 구조를 컬럼으로 명시 보존하는 메서드

        Args:
            - df: forecast_kst_dtm / data_available_kst_dtm 컬럼을 가진 표

        Logic:
            - forecast_issue_date = 예보 발행일 (13시에 쓸 수 있게 된 그 날)
            - lead_hour = (대상 시각 - 사용가능 시각) / 3600초
            - LDAPS와 같은 시간 규칙을 쓴다. 두 소스를 forecast_kst_dtm으로 붙일 것이기 때문이다
        """
        df = df.copy()
        for col in self.TIME_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors = "raise")

        if set(self.TIME_COLS).issubset(df.columns):
            df["forecast_issue_date"] = df["data_available_kst_dtm"].dt.normalize()
            df["gfs_lead_hour"] = (
                df["forecast_kst_dtm"] - df["data_available_kst_dtm"]
            ).dt.total_seconds() / 3600
            # 초 -> 시간 환산. LDAPS의 lead_hour와 값이 다를 수 있어 접두사를 붙인다
            df["target_hour"] = df["forecast_kst_dtm"].dt.hour
        return df
        # 예보 시간 메타 3개가 추가된 DataFrame 반환

    def checkPhysicalLimit(self, df: pd.DataFrame) -> pd.DataFrame:
        """GFS 원시값의 물리 일관성을 검사해 플래그를 붙이는 메서드

        Args:
            - df: gfs_t2m·gfs_dpt2m(K), gfs_rh2m(%), gfs_sp(Pa) 컬럼을 가진 표

        Logic:
            - is_gfs_temp_below_dew   : 기온 < 이슬점. 물리적으로 불가능한 과포화 상태
            - is_gfs_rh_over_100      : 상대습도 > 100%
            - is_gfs_pressure_invalid : 기압 <= 0. 단위 사고를 잡는 안전망
            - 수치예보 모델 산출물이라 값은 지우지 않고 플래그만 남긴다.
              통계적 이상치 제거를 쓰지 않는 이유도 LDAPS와 같다 (폭풍이 가장 중요한 사례다)
        """
        df = df.copy()
        if {"gfs_t2m", "gfs_dpt2m"}.issubset(df.columns):
            df["is_gfs_temp_below_dew"] = df["gfs_t2m"] < df["gfs_dpt2m"]
        if "gfs_rh2m" in df.columns:
            df["is_gfs_rh_over_100"] = df["gfs_rh2m"] > RH_PHYS_MAX
        if "gfs_sp" in df.columns:
            df["is_gfs_pressure_invalid"] = df["gfs_sp"] <= 0
        return df
        # 물리 일관성 플래그 3종이 추가된 DataFrame 반환. 원본 값은 그대로다

    def calculatePhysicalFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """격자별 다층 바람·전단·밀도 파생변수를 계산하는 메서드

        Args:
            - df: 컬럼명 통일을 마친 GFS 표 (한 행 = 한 격자 x 한 시각)

        Logic:
            - 고도별 풍속·풍향을 UV_LEVELS 딕셔너리로 한 번에 만든다
            - 전단지수 alpha = ln(V100/V10) / ln(100/10) -> 허브 높이 외삽에 쓴다
            - 허브 풍속 V_hub = V100 x (117/100)^alpha
            - 비선형 계산(밀도, 에너지)은 격자별로 먼저 끝낸다. 공간 평균을 먼저 하면 값이 틀어진다
            - 850hPa 기온과 지상 기온의 차(lapse)는 대기 안정도의 대리변수다.
              안정하면 바람이 층으로 갈리고 불안정하면 상층 운동량이 지상으로 내려온다
        """
        df = df.copy()

        for level, (u_col, v_col) in self.UV_LEVELS.items():
            if {u_col, v_col}.issubset(df.columns):
                # 컬럼 존재 여부를 방어적으로 확인
                df[f"gfs_ws_{level}"] = compute_wind_speed(df[u_col], df[v_col])
                df[f"gfs_wd_{level}"] = compute_wind_direction(df[u_col], df[v_col])

        if {"gfs_ws_10", "gfs_ws_100"}.issubset(df.columns):
            df["gfs_shear_alpha"] = compute_wind_shear(
                df["gfs_ws_10"], df["gfs_ws_100"], height_low = 10.0, height_high = 100.0
            )
            df["gfs_ws_hub"] = df["gfs_ws_100"] * (HUB_HEIGHT_M / 100.0) ** df["gfs_shear_alpha"]
            # 100m -> 허브 높이 117m 외삽. 대회 평가 기간에 SCADA가 없어 이 값이 허브풍속 대리변수다

        if {"gfs_t2m", "gfs_q2m", "gfs_sp"}.issubset(df.columns):
            df = calculate_air_density(
                df, t_col = "gfs_t2m", q_col = "gfs_q2m", p_col = "gfs_sp", prefix = "gfs_"
            )
            ws_for_power = "gfs_ws_hub" if "gfs_ws_hub" in df.columns else "gfs_ws_100"
            if ws_for_power in df.columns:
                df["gfs_wind_power_density"] = calculate_wind_power_density(
                    df["gfs_air_density"], df[ws_for_power]
                )
                # LDAPS는 10m 풍속으로 계산하지만 GFS는 허브 높이 풍속을 쓸 수 있다

        if {"gfs_gust", "gfs_ws_10"}.issubset(df.columns):
            df["gfs_gust_factor"] = df["gfs_gust"] / df["gfs_ws_10"].clip(lower = 1e-3)
            # 돌풍/평균 비율 = 난류 강도. 무풍에서 0으로 나누는 것을 방지하기 위해 하한을 둔다
            df["is_gfs_gust_extreme"] = df["gfs_gust_factor"] > GUST_RATIO_MAX

        if {"gfs_t2m", "gfs_dpt2m"}.issubset(df.columns):
            df["gfs_t_dew_diff"] = df["gfs_t2m"] - df["gfs_dpt2m"]
            # 기온 - 이슬점 차. 값이 클수록 건조하다
        if {"gfs_t2m", "gfs_t850"}.issubset(df.columns):
            df["gfs_lapse_850"] = df["gfs_t2m"] - df["gfs_t850"]
            # 지상 - 850hPa 기온차. 클수록 불안정해 상층 운동량이 지상까지 섞여 내려온다
        if {"gfs_u850", "gfs_u10", "gfs_v850", "gfs_v10"}.issubset(df.columns):
            df["gfs_bulk_shear_850"] = np.sqrt(
                (df["gfs_u850"] - df["gfs_u10"]) ** 2 + (df["gfs_v850"] - df["gfs_v10"]) ** 2
            )
            # 지상~850hPa 벡터 차의 크기. 저층 제트가 있으면 이 값이 커진다
        return df
        # 고도별 풍속·풍향과 전단·밀도·안정도 파생변수가 추가된 DataFrame 반환

    # ------------------------------------------------------------------
    # 훅 메서드 오버라이드 — GFS는 주기 인코딩만 쓴다
    # ------------------------------------------------------------------

    def transformCyclicFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """고도별 풍향과 예보 대상 시각을 주기성 피처로 변환하는 메서드

        Args:
            - df: 고도별 풍향(gfs_wd*)과 forecast_kst_dtm을 가진 표
        """
        wd_cols = [c for c in df.columns if c.startswith("gfs_wd_")]
        df = transform_wind_direction(df, wind_direction_cols = wd_cols)
        if "forecast_kst_dtm" in df.columns:
            df = transform_cyclic_time(df, time_col = "forecast_kst_dtm")
        return df
        # 고도별 풍향 sin/cos와 시간 순환 피처가 추가된 DataFrame 반환

    # ------------------------------------------------------------------
    # 격자 -> KPX 그룹 (구체)
    # ------------------------------------------------------------------

    def transformToGroupIDW(
        self, df: pd.DataFrame, turbine_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """9격자를 터빈 위치 기준 역거리 가중으로 그룹별 피처로 바꾸는 메서드

        Args:
            - df: preprocess를 마친 격자 단위 표
            - turbine_meta: 터빈 1기 = 1행. `group`, `lat`, `lon`, `cap_kw` 컬럼 필수

        Logic:
            - GFS 격자는 9개뿐이고 간격도 LDAPS보다 넓어 가중치가 더 평평해진다.
              그래도 그룹마다 다른 값이 나와야 공간 매핑이 의미가 있다
            - 풍향은 가중평균한 U·V에서 다시 복원한다. 각도를 직접 평균하면 안 된다
        """
        required = {"grid_id", "latitude", "longitude", "forecast_kst_dtm"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(f"GFS 공간 집계에 필요한 컬럼이 없습니다: {missing}")

        grid_coords = (
            df.groupby("grid_id", observed = True)[["latitude", "longitude"]]
            .first()
            .sort_index()
        )
        # 격자별 좌표. 기간 내내 고정이라 첫 행만 꺼내 쓴다

        grid_index = grid_coords.index
        weight_group = compute_group_weight(
            turbine_meta, grid_coords, k = min(self.k, len(grid_coords)), power = self.power
        )

        features = [f for f in self.IDW_FEATURES if f in df.columns]
        if not features:
            raise PreprocessingError("GFS 공간 집계에 사용할 파생 피처가 없습니다")
        group_df = transform_to_group_feature(
            df, features = features,
            weight_group = weight_group, grid_index = grid_index,
            time_col = "forecast_kst_dtm",
            target_groups = [self.group] if self.group else None,
        )

        for level, (u_col, v_col) in self.UV_LEVELS.items():
            if {u_col, v_col}.issubset(group_df.columns):
                name = f"gfs_wd_{level}"
                group_df[name] = compute_wind_direction(group_df[u_col], group_df[v_col])
                group_df = transform_wind_direction(group_df, wind_direction_cols = [name])
                # 각도는 가중평균하면 안 되므로 U·V를 평균한 뒤 방향을 복원한다

        if "gfs_ws_hub" in group_df.columns:
            group_df["gfs_ws_hub_2"] = group_df["gfs_ws_hub"] ** 2
            group_df["gfs_ws_hub_3"] = group_df["gfs_ws_hub"] ** 3
            # 풍력에너지가 풍속 세제곱에 비례하므로 비선형 항을 미리 만들어준다

        group_df = transform_group_time_feature(
            group_df, df,
            time_meta_cols = ["gfs_lead_hour", "target_hour", "forecast_issue_date"],
        )
        # 공간 가중평균은 IDW_FEATURES만 통과시키므로 시간 메타를 여기서 되살린다

        return group_df.sort_values(["group", "forecast_kst_dtm"], ignore_index = True)
        # 한 행 = 한 그룹 x 한 시각인 공간가중 예보표 반환

    def transformForecastFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """그룹별 예보 시계열의 변화량·램프 피처를 붙이는 메서드

        Args:
            - df: transformToGroupIDW를 마친 (그룹 x 시각) 표

        Logic:
            - 허브 높이 풍속(gfs_ws_hub)을 기준으로 변화량을 잰다.
              발전량과 직접 연결되는 고도라 10m 변화량보다 신호가 강하다
            - 100m 자료가 없는 축약 입력에서는 10m 컬럼으로 대체해 계산한다
        """
        df = df.sort_values(["group", "forecast_kst_dtm"], ignore_index = True)
        ws_col = "gfs_ws_hub" if "gfs_ws_hub" in df.columns else "gfs_ws_100"
        if ws_col not in df.columns:
            ws_col = "gfs_ws_10"
        u_col = "gfs_u100" if "gfs_u100" in df.columns else "gfs_u10"
        v_col = "gfs_v100" if "gfs_v100" in df.columns else "gfs_v10"
        wd_col = "gfs_wd_100" if "gfs_wd_100" in df.columns else "gfs_wd_10"
        return transform_forecast_diff(
            df, ws_col = ws_col, u_col = u_col, v_col = v_col, wd_col = wd_col,
            key = "group", prefix = "gfs",
        )
        # gfs_ws_diff_1h 등 변화량 피처 8종이 추가된 DataFrame 반환

    def transformGFS(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 호출부(BASELINE 노트북)를 위한 전처리 호환 메서드"""
        return self.preprocess(df)
        # preprocess 템플릿과 완전히 같은 결과를 반환한다


__all__ = [
    "GFSFeatureEngineer",
    "HUB_HEIGHT_M", "RH_PHYS_MAX", "GUST_RATIO_MAX",
]
