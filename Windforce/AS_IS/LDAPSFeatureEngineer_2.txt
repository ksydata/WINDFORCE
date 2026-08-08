# Step 2·4·8·10. LDAPS 예보의 형식 통일, 격자 파생변수, KPX 그룹 공간 가중
# 03_Preprocessing.ipynb의 LDAPSPreprocessor·LDAPSSpatialMapper·LDAPSGridStatsBuilder를
# FeatureEngineer 계약에 맞춰 하나로 승격한 모듈

"""LDAPS 예보의 컬럼 통일과 물리·공간·시간 파생변수 생성을 담당하는 모듈.

LDAPS는 1.5km 해상도 16격자를 매시간 제공한다. 예보 발행은 매일 13시 한 번이고
그 한 발행분이 다음날 24개 대상 시각을 덮는다 (선행시간 12~35시간).

파생 순서가 중요하다. 공기밀도·풍력에너지 밀도처럼 **비선형** 계산은 반드시 격자별로
먼저 끝내고 나중에 공간 가중평균해야 한다. 평균 기온·평균 기압으로 밀도를 한 번 계산하면
값이 미묘하게 틀리고, 평균 풍속을 세제곱하면 발전 가능 에너지를 과소평가한다(젠센 부등식).

    transform()            격자별 파생변수 + 품질 플래그
      -> transformToGroupIDW()   16격자 -> KPX 그룹 (역거리 가중)
      -> transformToGridStats()  16격자 -> 시각별 공간 통계 (비교용 대조군)
      -> transformForecastDiff() 시계열 변화량·램프
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ..FeatureEngineer import FeatureEngineer, IDW_K, IDW_POWER


WS_TIME_JUMP_MS = 15.0
# 같은 격자에서 1시간에 15 m/s 넘게 변하면 시간 급변으로 표시한다
WS_SPACE_JUMP_MS = 10.0
# 같은 시각인데 격자 간 풍속이 10 m/s 넘게 벌어지면 공간 급변으로 표시한다
RH_PHYS_MAX = 100.0
# 상대습도 물리한계(%). 모델 산출물이라 100%를 살짝 넘는 값이 실제로 존재한다
LDAPS_GRID_COUNT = 16
# LDAPS 격자 수. 약 1.5km 공간해상도로 발전단지 주변을 덮는다


class LDAPSFeatureEngineer(FeatureEngineer):
    """LDAPS 예보의 격자 파생변수와 KPX 그룹 공간 가중을 담당하는 클래스

    Attributes:
        - 지상 2m  : 기온·이슬점·비습·상대습도 (t2m / dpt2m / q2m / rh2m)
        - 지상 5m  : 경계층 바람 X·Y 성분 (blws_x / blws_y)
        - 지상 10m : 순간값 U·V (u10 / v10)
        - 지상 50m : 최댓값·최솟값 U·V (변동폭 계산 전용)
        - 지표면   : 기압·고도·강수·복사·육해 마스크
        - weight_group : 그룹 x 격자 역거리 가중치 행렬 (computeGroupWeight가 채운다)
        - grid_index   : 가중치 행렬의 열 순서를 정의하는 grid_id 인덱스
    """

    source: ClassVar[str] = "ldaps"

    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "heightAboveGround_10_10u": "u10",
        # 지상 10m 동서(U) 바람 (m/s)
        "heightAboveGround_10_10v": "v10",
        # 지상 10m 남북(V) 바람 (m/s)
        "heightAboveGround_50_50MUmax": "u50_max",
        "heightAboveGround_50_50MUmin": "u50_min",
        "heightAboveGround_50_50MVmax": "v50_max",
        "heightAboveGround_50_50MVmin": "v50_min",
        # 50m 바람은 max·min을 직접 합쳐 '최대 풍속 벡터'를 만들면 안 된다.
        # 두 최댓값이 같은 시점에 발생했다는 보장이 없어 변동폭으로만 쓴다
        "heightAboveGround_5_XBLWS": "blws_x",
        "heightAboveGround_5_YBLWS": "blws_y",
        "heightAboveGround_2_t": "t2m",
        # 지상 2m 기온 (K)
        "heightAboveGround_2_dpt": "dpt2m",
        # 지상 2m 이슬점온도 (K)
        "heightAboveGround_2_q": "q2m",
        # 지상 2m 비습 (kg/kg)
        "heightAboveGround_2_r": "rh2m",
        # 지상 2m 상대습도 (%)
        "surface_0_sp": "sp",
        # 지표면 기압 (Pa)
        "meanSea_0_prmsl": "prmsl",
        # 해면기압 (Pa)
        "etc_0_blh": "blh",
        # 경계층 높이 (m)
        "surface_0_NDNSW": "nd_sw",
        "surface_0_NDNLW": "nd_lw",
        "heightAboveGround_2_SWDIR": "sw_dir",
        "heightAboveGround_2_SWDIF": "sw_dif",
        "etc_0_hcc": "hcc",
        "etc_0_mcc": "mcc",
        "etc_0_lcc": "lcc",
        "etc_0_VLCDC": "vlcdc",
        "surface_0_avg_lsprate": "ls_prate",
        "surface_0_lssrate": "ls_srate",
        "surface_0_ncpcp": "ncpcp",
        "surface_0_snol": "snol",
        "surface_0_SNOM": "snom",
        "surface_0_lsm": "lsm",
        # 육지/해양 마스크. 우리 격자는 전부 육지(1)다
        "surface_0_h": "surface_h",
        # 지표면 고도 (m). 태백 산지라 900m 안팎이 정상
    }

    TIME_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]
    # 예보 대상 시각과 예보 사용 가능 시각. 둘의 차이가 선행시간(lead_hour)이다

    IDW_FEATURES = [
        "u10", "v10", "ws10", "air_density", "wind_power_density",
        # 바람·밀도·에너지
        "t2m", "dpt2m", "rh2m", "q2m", "t_dew_diff",
        # 기온·습도
        "sp", "prmsl", "blh", "surface_h", "blws",
        # 기압·경계층·고도
        "u50_range", "v50_range", "ws10_spatial_range",
        # 변동성
        "lcc", "mcc", "hcc", "vlcdc",
        # 운량
        "ncpcp", "ls_prate", "nd_sw", "nd_lw",
        # 강수·복사
    ]
    # 격자별로 이미 계산을 마친 값만 넣는다. 풍향(각도)은 넣지 않고 U·V에서 복원한다

    STAT_FEATURES = [
        "ws10", "u10", "v10", "air_density", "wind_power_density",
        "t2m", "q2m", "sp", "blh", "rh2m", "lcc", "ncpcp",
    ]
    # 16격자 단순 통계 버전에 쓸 컬럼. 공간가중(IDW) 버전과 A/B 비교하기 위한 대조군이다

    def __init__(self, group: str | None = None, k: int = IDW_K, power: float = IDW_POWER):
        super().__init__(group)
        self.k = k
        self.power = power
        self.weight_group: np.ndarray | None = None
        self.grid_index: pd.Index | None = None
        # 격자 좌표는 기간 내내 고정이므로 가중치는 한 번만 구하고 계속 재사용한다

    # ------------------------------------------------------------------
    # 추상 메서드 구현 — transform 템플릿이 순서대로 호출한다
    # ------------------------------------------------------------------

    def transformColumnName(self, df: pd.DataFrame) -> pd.DataFrame:
        """긴 원본 컬럼명을 snake 짧은 이름으로 바꾸는 메서드

        Args:
            - df: LDAPS 원본 표 (heightAboveGround_10_10u 같은 긴 컬럼명)
        """
        return df.rename(
            columns = {k: v for k, v in self.COLUMN_MAP.items() if k in df.columns}
        )
        # u10·t2m·sp 같은 짧은 이름으로 통일된 DataFrame 반환

    def transformTimeAxis(self, df: pd.DataFrame) -> pd.DataFrame:
        """예보 시간 구조를 컬럼으로 명시 보존하는 메서드

        Args:
            - df: forecast_kst_dtm / data_available_kst_dtm 컬럼을 가진 표

        Logic:
            - forecast_issue_date = 예보 발행일 (13시에 쓸 수 있게 된 그 날)
            - lead_hour = (대상 시각 - 사용가능 시각) / 3600초. 12~35시간이어야 정상
            - 달력 날짜의 00~23시로 임의 재배치하지 않는 것이 핵심이다.
              24개 대상 시각이 같은 발행분을 공유한다는 사실을 컬럼으로 붙여두면
              나중에 하루씩 밀리는 사고를 막을 수 있다
        """
        df = df.copy()
        for col in self.TIME_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        if set(self.TIME_COLS).issubset(df.columns):
            df["forecast_issue_date"] = df["data_available_kst_dtm"].dt.normalize()
            df["lead_hour"] = (
                df["forecast_kst_dtm"] - df["data_available_kst_dtm"]
            ).dt.total_seconds() / 3600
            # 초 -> 시간 환산
            df["target_hour"] = df["forecast_kst_dtm"].dt.hour
        return df
        # 예보 시간 메타 3개가 추가된 DataFrame 반환

    def checkPhysicalLimit(self, df: pd.DataFrame) -> pd.DataFrame:
        """LDAPS 원시값의 물리 일관성을 검사해 플래그를 붙이는 메서드

        Args:
            - df: t2m·dpt2m(K), rh2m(%) 컬럼을 가진 표

        Logic:
            - is_temp_below_dew : 기온 < 이슬점. 물리적으로 불가능한 과포화 상태
            - is_rh_over_100    : 상대습도 > 100%
            - ASOS·AWS는 센서 관측이라 고장을 잡는 게 목적이지만 LDAPS는 수치예보 모델 결과다.
              두 위반 모두 모델 산출물의 특성이므로 **값은 지우지 않고 플래그만 남긴다**
            - IQR·3시그마 같은 통계적 이상치 제거도 쓰지 않는다. 폭풍은 통계적 극단값이지만
              발전량 예측에서 가장 중요한 사례라 지우면 손해다
        """
        df = df.copy()
        if {"t2m", "dpt2m"}.issubset(df.columns):
            df["is_temp_below_dew"] = df["t2m"] < df["dpt2m"]
        if "rh2m" in df.columns:
            df["is_rh_over_100"] = df["rh2m"] > RH_PHYS_MAX
        return df
        # 물리 일관성 플래그 2종이 추가된 DataFrame 반환. 원본 값은 그대로다

    def calculatePhysicalFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """격자별 물리 파생변수를 계산하는 메서드

        Args:
            - df: 컬럼명 통일을 마친 LDAPS 표 (한 행 = 한 격자 x 한 시각)

        Logic:
            - 비선형 계산(공기밀도, 풍력에너지 밀도)을 격자별로 **먼저** 끝낸다.
              공간 가중평균을 먼저 하면 젠센 부등식 때문에 값이 틀어진다
            - 50m 바람은 변동폭(max - min)으로만 쓴다. 난류·돌풍의 대리변수다
        """
        df = df.copy()

        if {"u10", "v10"}.issubset(df.columns):
            df["ws10"] = self.computeWindSpeed(df["u10"], df["v10"])
            df["wd10"] = self.computeWindDirection(df["u10"], df["v10"])

        if {"t2m", "q2m", "sp"}.issubset(df.columns):
            df = self.calculateAirDensity(df, t_col = "t2m", q_col = "q2m", p_col = "sp")
            if "ws10" in df.columns:
                df["wind_power_density"] = self.calculateWindPowerDensity(
                    df["air_density"], df["ws10"]
                )
            # q2m이 없으면 밀도를 계산하지 않는다. 단위가 불명확한 입력을 조용히 변환하지 않는다

        if {"u50_max", "u50_min"}.issubset(df.columns):
            df["u50_range"] = df["u50_max"] - df["u50_min"]
        if {"v50_max", "v50_min"}.issubset(df.columns):
            df["v50_range"] = df["v50_max"] - df["v50_min"]
        if {"blws_x", "blws_y"}.issubset(df.columns):
            df["blws"] = self.computeWindSpeed(df["blws_x"], df["blws_y"])
            # 경계층 바람 크기. 지표 마찰의 영향을 받는 층이라 허브 높이 보정의 단서가 된다
        if {"t2m", "dpt2m"}.issubset(df.columns):
            df["t_dew_diff"] = df["t2m"] - df["dpt2m"]
            # 기온 - 이슬점 차. 값이 클수록 건조하다
        return df
        # 격자별 물리 파생변수가 추가된 DataFrame 반환

    def checkDerivedFlag(self, df: pd.DataFrame) -> pd.DataFrame:
        """풍속의 시간·공간 급변에 플래그를 붙이는 메서드

        Args:
            - df: calculatePhysicalFeature를 거쳐 ws10을 가진 표. grid_id 컬럼 필수

        Logic:
            - ws10_diff_1h       : 같은 격자에서 1시간 사이 풍속 변화량
            - ws10_spatial_range : 같은 시각 격자 간 풍속 최대 - 최소 (공간 불확실성)
            - ws10이 만들어진 뒤에야 계산할 수 있어 원시값 검사(checkPhysicalLimit)와 분리했다
            - 격자 간 풍속 편차는 16격자를 통째로 평균하면 사라지는 정보라 컬럼으로 남긴다
        """
        if not {"grid_id", "forecast_kst_dtm", "ws10"}.issubset(df.columns):
            return df
            # 격자 구조가 없으면(이미 그룹으로 합쳐진 표라면) 급변 검사를 건너뛴다

        df = df.sort_values(["grid_id", "forecast_kst_dtm"], ignore_index = True)
        # 격자별 시간순 정렬은 변화량 계산 전에 필수다

        df["ws10_diff_1h"] = df.groupby("grid_id", observed = True)["ws10"].diff()
        df["is_ws_time_jump"] = df["ws10_diff_1h"].abs() > WS_TIME_JUMP_MS

        ws_by_time = df.groupby("forecast_kst_dtm", observed = True)["ws10"]
        df["ws10_spatial_range"] = ws_by_time.transform("max") - ws_by_time.transform("min")
        df["is_ws_space_jump"] = df["ws10_spatial_range"] > WS_SPACE_JUMP_MS
        return df
        # 급변 플래그 2종과 변화량·편차 컬럼이 추가된 DataFrame 반환

    def transformCyclicFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """풍향과 예보 대상 시각을 주기성 피처로 변환하는 메서드

        Args:
            - df: wd10과 forecast_kst_dtm을 가진 표
        """
        df = self.transformWindDirection(df, wind_direction_cols = ["wd10"])
        if "forecast_kst_dtm" in df.columns:
            df = self.transformCyclicTime(df, time_col = "forecast_kst_dtm")
        return df
        # wd10_sin/cos와 시간 순환 피처가 추가된 DataFrame 반환

    # ------------------------------------------------------------------
    # 격자 -> KPX 그룹 (구체) — 16격자를 모델 입력 단위로 좁힌다
    # ------------------------------------------------------------------

    def transformToGroupIDW(
        self, df: pd.DataFrame, turbine_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """16격자를 터빈 위치 기준 역거리 가중으로 그룹별 피처로 바꾸는 메서드

        Args:
            - df: transform을 마친 격자 단위 표
            - turbine_meta: 터빈 1기 = 1행. `group`, `lat`, `lon`, `cap_kw` 컬럼 필수

        Logic:
            - 16격자를 통째로 평균하면 세 그룹에 **똑같은 예보값**이 들어가
              거리·고도·바람장 차이와 격자 간 공간 불확실성이 전부 사라진다
            - 풍향은 가중평균한 U·V에서 다시 복원한다. 각도를 직접 평균하면 안 된다
            - ws10_2·ws10_3은 가중평균 이후에 만든다. 그룹을 대표하는 단일 풍속의 제곱·세제곱이라
              격자별 평균(ldaps_mean_ws3)과는 의미가 다른 별개 피처다
        """
        grid_coords = df.groupby("grid_id")[["latitude", "longitude"]].first().sort_index()
        # 격자별 좌표. 기간 내내 고정이라 첫 행만 꺼내 쓴다

        self.weight_group = self.computeGroupWeight(
            turbine_meta, grid_coords, k = self.k, power = self.power
        )
        self.grid_index = grid_coords.index

        features = [f for f in self.IDW_FEATURES if f in df.columns]
        group_df = self.transformToGroupFeature(
            df, features = features,
            weight_group = self.weight_group, grid_index = self.grid_index,
            time_col = "forecast_kst_dtm",
        )

        if {"u10", "v10"}.issubset(group_df.columns):
            group_df["ws10_vector"] = self.computeWindSpeed(group_df["u10"], group_df["v10"])
            group_df["wd10"] = self.computeWindDirection(group_df["u10"], group_df["v10"])
            group_df = self.transformWindDirection(group_df, wind_direction_cols = ["wd10"])
            # scalar 평균(ws10)과 vector 평균(ws10_vector)을 둘 다 남긴다.
            # 둘의 비율이 그 시각 격자 간 풍향이 얼마나 일정했는지를 알려준다

        if "ws10" in group_df.columns:
            group_df["ws10_2"] = group_df["ws10"] ** 2
            group_df["ws10_3"] = group_df["ws10"] ** 3
            # 풍력에너지가 풍속 세제곱에 비례하므로 비선형 항을 미리 만들어준다

        group_df = self.transformGroupTimeFeature(
            group_df, df, time_meta_cols = ["lead_hour", "target_hour", "forecast_issue_date"]
        )
        # 공간 가중평균은 IDW_FEATURES만 통과시키므로 시간 메타를 여기서 되살린다

        return group_df.sort_values(
            ["group", "forecast_kst_dtm"], ignore_index = True
        )
        # 한 행 = 한 그룹 x 한 시각인 공간가중 예보표 반환

    def transformToGridStats(self, df: pd.DataFrame) -> pd.DataFrame:
        """시각별 16격자 공간 통계를 계산하는 메서드 (IDW 대조군)

        Args:
            - df: transform을 마친 격자 단위 표

        Logic:
            - 평균 하나로는 전 격자 8m/s인 경우와 절반 4m/s·절반 12m/s인 경우를 구분할 수 없다.
              표준편차·최소·최대·분위수까지 넣어 공간 분포 자체를 피처로 만든다
            - 세제곱 계열은 '격자별로 먼저 계산 -> 평균' 순서를 지킨다.
              평균 풍속을 세제곱한 값과 다르다 (젠센 부등식)
            - 이 표는 그룹 구분이 없다. 세 그룹에 같은 값이 붙는 대조군이라는 뜻이다
        """
        df = df.copy()
        df["ws10_2"] = df["ws10"] ** 2
        df["ws10_3"] = df["ws10"] ** 3
        df["rho_ws3"] = df["air_density"] * df["ws10"] ** 3
        # 격자별로 먼저 계산해야 한다

        features = [f for f in self.STAT_FEATURES if f in df.columns]
        grouped = df.groupby("forecast_kst_dtm")
        stats = {
            "mean": grouped[features].mean(),
            "std": grouped[features].std(),
            # 격자 간 표준편차 = 공간 변동성
            "min": grouped[features].min(),
            "max": grouped[features].max(),
        }
        grid_stats = pd.concat(stats, axis = 1)
        grid_stats.columns = [f"ldaps_{col}_{stat}" for stat, col in grid_stats.columns]
        # ldaps_ws10_mean 형태로 이름 정리

        quantile = grouped["ws10"].quantile([0.25, 0.5, 0.75]).unstack()
        quantile.columns = ["ldaps_ws10_q25", "ldaps_ws10_median", "ldaps_ws10_q75"]
        grid_stats = grid_stats.join(quantile)

        grid_stats["ldaps_mean_ws2"] = grouped["ws10_2"].mean()
        grid_stats["ldaps_mean_ws3"] = grouped["ws10_3"].mean()
        grid_stats["ldaps_mean_rho_ws3"] = grouped["rho_ws3"].mean()
        grid_stats["ldaps_ws10_range"] = (
            grid_stats["ldaps_ws10_max"] - grid_stats["ldaps_ws10_min"]
        )
        grid_stats["ldaps_wd10"] = self.computeWindDirection(
            grid_stats["ldaps_u10_mean"], grid_stats["ldaps_v10_mean"]
        )
        # 풍향은 격자 U·V 평균에서 복원한다
        return grid_stats.reset_index()
        # 한 행 = 한 시각인 공간 통계표 반환 (그룹 구분 없음)

    def transformForecastFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """그룹별 예보 시계열의 변화량·램프 피처를 붙이는 메서드

        Args:
            - df: transformToGroupIDW를 마친 (그룹 x 시각) 표

        Logic:
            - 공통 구현(transformForecastDiff)에 LDAPS 컬럼 이름만 넘긴다
            - 그룹별 시간순 정렬을 먼저 보장한다. 안 그러면 diff가 엉뚱한 행끼리 계산된다
        """
        df = df.sort_values(["group", "forecast_kst_dtm"], ignore_index = True)
        return self.transformForecastDiff(
            df, ws_col = "ws10", u_col = "u10", v_col = "v10", wd_col = "wd10",
            key = "group", prefix = "ldaps",
        )
        # ldaps_ws_diff_1h 등 변화량 피처 8종이 추가된 DataFrame 반환


__all__ = [
    "LDAPSFeatureEngineer",
    "WS_TIME_JUMP_MS", "WS_SPACE_JUMP_MS", "RH_PHYS_MAX", "LDAPS_GRID_COUNT",
]
