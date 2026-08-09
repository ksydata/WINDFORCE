"""SCADA 10분 실측의 품질 정제와 그룹별 시간 집계를 담당하는 모듈.

터빈 17기가 컬럼으로 옆으로 늘어선 wide 표(`vestas_wtg01_power_kw10m`, ...)는
``transformToLong``으로 먼저 녹인다. 한 행 = 한 터빈 x 한 시각인 세로 표라야
터빈마다 같은 코드를 17번 반복하지 않는다.

원본은 절대 지우지 않는다. ``power_raw``(원본) / ``power_clean``(정제) / ``power_filled``(보간)
세 단계를 모두 컬럼으로 남겨 언제든 되돌려 확인할 수 있게 한다.

**SCADA는 최종 모델 입력이 아니다.** 평가 기간에는 LDAPS·GFS만 제공되고 SCADA가 없다.
이 모듈의 산출물은 LDAPS 10m 풍속 -> 허브 높이 풍속 보정 모델의 **정답값**과
파워커브·이상 상태 분석에만 쓴다. 보정 모델을 주 모델 입력으로 쓸 때는 학습 구간에
Out-of-Fold 예측값을 만들어야 간접 누수를 막을 수 있다.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from .FeatureEngineer import BasePreprocessor
from ..utils.spatial_utils import KPX_GROUPS
from ..utils.time_utils import MAX_INTERP_STEPS, interpolate_short_gap
from ..utils.wind_utils import (
    RHO_STD,
    compute_uv_component,
    compute_wind_direction,
    compute_wind_speed,
    transform_wind_direction,
)


POWER_CAP_RATIO = 1.05
# 정격의 105%를 넘는 발전량은 비현실적이라 센서 오류로 본다
NEG_POWER_RATIO = 0.05
# 정격의 5%보다 큰 음수만 '큰 음수'(센서·단위 오류)로 본다. 작은 음수는 대기전력일 수 있다
FROZEN_STEPS = 18
# 같은 값이 10분 간격으로 18번 넘게 반복 = 3시간 고정. 센서 고착을 의심한다 (18 x 10분 = 180분)
CUT_IN_MS = 3.0
MIN_SAMPLES_PER_HOUR = 6
# 컷인 풍속(m/s). V126·U136 둘 다 3 m/s (INFO/V126_vs_U136_스펙비교.html)
HIGH_WS_MS = 5.0
# 이 정도 바람이면 발전하는 게 정상이다. 그런데 0이면 고장·정비·출력제한 후보다
WS_PHYS_MAX = 75.0
# 풍속 물리한계 상한(m/s). MD/05 문서의 기상청 물리한계표(AWS) 기준
DISAGREE_RATIO = 0.5
# 같은 그룹 다른 터빈의 중앙값과 정격 50% 이상 벌어지면 개별 장비 이상 후보다
MIN_SAMPLES_RELAXED = 5
# 10분값 5개(5/6)만 있어도 인정하는 완화 버전. 엄격 버전과 나란히 만들어 나중에 비교한다
LOW_WS_MASK = 1.0
# 풍속이 1 m/s보다 낮으면 풍향은 의미가 약하니 방향 통계에서 빼둔다
FULL_TURN_DEG = 360
# 풍향 한 바퀴(도). UNISON이 -180~180 표기라 0~360으로 맞출 때 쓴다

SWEPT_AREA_M2 = {
    "V126": 12469.0,
    # VESTAS V126 수풍면적(m²). 로터 지름 126m
    "U136": 14470.0,
    # UNISON U136 수풍면적(m²). 로터 지름 136m. 04 문서 추정치(14,527) 대신 실제 스펙값
}
CUT_OUT_MS = {
    "V126": 22.5,
    # VESTAS V126 컷아웃 풍속(m/s)
    "U136": 22.0,
    # UNISON U136 컷아웃 풍속(m/s). 이 이상 불면 터빈을 세우므로 발전량 0이 정상이다
}
META_COLS = ["turbine", "group", "model", "cap_kwh10m", "cut_out_ms", "swept_area_m2"]
# 검사·파생에 필요한 터빈 메타 컬럼. 터빈마다 정격·컷아웃·수풍면적이 다르다


class SCADAFeatureEngineer(BasePreprocessor):
    """SCADA 10분 실측의 품질 정제와 그룹별 시간 집계를 담당하는 클래스

    Attributes:
        - turbine_meta: 터빈 1기 = 1행인 메타 표 (turbine/group/model/capacity/coordinate)
        - max_interpolation_steps: 이 칸수 이하의 연속 결측만 보간한다 (기본 2칸 = 20분)
        - FLAG_COLS: 플래그 이름 목록. 순서가 QC 비트 코드의 자릿수를 결정한다
        - DROP_FLAGS: 값을 못 믿겠다고 판단해 NaN 처리할 플래그
    """

    source: ClassVar[str] = "scada"

    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "datetime": "kst_dtm",
        # 시각 컬럼 이름이 파일마다 달라 표준 이름으로 모은다
        "power": "power_raw",
        # 10분 발전량 (kWh). 원본이라는 뜻으로 _raw를 붙인다
        "ws": "ws_raw",
        # 나셀 풍속계 10분 평균 풍속 (m/s)
        "wd": "wd_raw",
        # 나셀 풍향 (도). VESTAS는 0~360, UNISON은 -180~180 표기다
    }

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ()
    # 원본이 wide/long 어느 형식이든 받으므로 템플릿의 사전 검사는 비워 둔다.
    # 세로 표 계약 검사는 rename 직후 transformColumnName이 수행한다 (아래 REQUIRED_COLS)

    FLAG_COLS: ClassVar[list[str]] = [
        "is_power_outlier", "is_power_neg_large", "is_power_neg_small", "is_frozen_sensor",
        "is_high_wind_zero_power", "is_low_wind_zero_power", "is_cut_out",
        "is_ws_out_of_range", "is_turbine_disagreement",
    ]
    # 순서를 바꾸면 이미 저장한 power_qc_flag 값의 해석이 통째로 달라진다. 추가는 뒤에만 한다

    DROP_FLAGS: ClassVar[list[str]] = [
        "is_power_outlier", "is_power_neg_large", "is_frozen_sensor",
    ]
    # 이 세 가지만 센서를 못 믿겠다는 근거가 되어 NaN 처리 대상이다

    REQUIRED_COLS: ClassVar[list[str]] = ["kst_dtm", "turbine", "power_raw", "ws_raw", "wd_raw"]
    # 세로 표가 반드시 가져야 하는 컬럼. 하나라도 없으면 뒤 단계가 전부 무의미해진다

    def __init__(
        self,
        group: str | None = None,
        turbine_meta: pd.DataFrame | None = None,
        max_interpolation_steps: int = MAX_INTERP_STEPS,
    ):
        super().__init__(group = group)

        if turbine_meta is None:
            raise ValueError(
                "SCADAFeatureEngineer는 turbine_meta가 필요합니다. "
                "터빈마다 정격(VESTAS 600 / UNISON 700 kWh)과 컷아웃이 달라 "
                "행별 물리한계 기준을 만들 수 없습니다"
            )
        if max_interpolation_steps < 0:
            raise ValueError("max_interpolation_steps는 0 이상이어야 합니다")

        self.turbine_meta = self.transformTurbineSpec(turbine_meta)
        self.max_interpolation_steps = max_interpolation_steps

        self._turbine_group = dict(
            zip(self.turbine_meta["turbine"], self.turbine_meta["group"])
        )
        self._turbine_cap_kwh10m = dict(
            zip(self.turbine_meta["turbine"], self.turbine_meta["cap_kwh10m"])
        )
        # 집계 단계에서 터빈 -> 그룹·정격을 반복 조회하므로 딕셔너리로 미리 펴둔다

    # ------------------------------------------------------------------
    # 입력 형식 변환 (구체) — preprocess 전에 한 번 부른다
    # ------------------------------------------------------------------

    def transformTurbineSpec(self, turbine_meta: pd.DataFrame) -> pd.DataFrame:
        """터빈 메타에 모델별 스펙과 10분 정격을 채우는 메서드

        Args:
            - turbine_meta: 최소한 `turbine`, `group`, `model`, `cap_kw`를 가진 표

        Logic:
            - 설비용량(kW) / 6 = 10분 동안 낼 수 있는 최대 발전량(kWh)
            - 수풍면적·컷아웃 풍속은 모델명으로 스펙 시트 값을 붙인다
        """
        meta = turbine_meta.copy()
        if "cap_kwh10m" not in meta.columns:
            if "cap_kw" not in meta.columns:
                raise KeyError("turbine_meta에 cap_kwh10m 또는 cap_kw가 있어야 합니다")
            meta["cap_kwh10m"] = meta["cap_kw"] / MIN_SAMPLES_PER_HOUR
            # kW(1시간 기준) -> kWh(10분 기준) 환산
        if "cut_out_ms" not in meta.columns:
            meta["cut_out_ms"] = meta["model"].map(CUT_OUT_MS)
        if "swept_area_m2" not in meta.columns:
            meta["swept_area_m2"] = meta["model"].map(SWEPT_AREA_M2)

        missing = [c for c in META_COLS if c not in meta.columns]
        if missing:
            raise KeyError(f"turbine_meta에 필요한 컬럼이 없습니다: {missing}")
        return meta
        # 스펙이 모두 채워진 터빈 메타 표 반환

    def transformToLong(
        self, df: pd.DataFrame, turbines: list[str], time_col: str = "kst_dtm"
    ) -> pd.DataFrame:
        """터빈이 컬럼으로 늘어선 wide 표를 세로 표로 바꾸는 메서드

        Args:
            - df: 원본 SCADA 표 (`vestas_wtg01_power_kw10m` 형태의 컬럼)
            - turbines: 이 파일에 들어있는 터빈 이름 목록
            - time_col: 시각 컬럼명

        Logic:
            - 완전한 10분 간격 시간축으로 먼저 reindex한다.
              누락 시각이 NaN 행으로 남아야 '없는 것'과 '결측'을 구분할 수 있다
            - float32로 낮춰 240만 행 표의 메모리를 절반으로 줄인다
        """
        work = df.copy()
        work[time_col] = pd.to_datetime(work[time_col])
        full_index = pd.date_range(work[time_col].min(), work[time_col].max(), freq = "10min")
        work = work.set_index(time_col).reindex(full_index)
        work.index.name = time_col
        work = work.reset_index()
        # 시간축 복원. 원본에 없던 시각도 NaN 행으로 자리를 잡는다

        frames = []
        for turbine in turbines:
            frames.append(pd.DataFrame({
                "kst_dtm": work[time_col],
                "turbine": turbine,
                "power_raw": work[f"{turbine}_power_kw10m"].astype("float32"),
                # 원본 발전량 (손대지 않은 값)
                "ws_raw": work[f"{turbine}_ws"].astype("float32"),
                "wd_raw": work[f"{turbine}_wd"].astype("float32"),
            }))
        return pd.concat(frames, ignore_index = True)
        # 한 행 = 한 터빈 x 한 시각인 세로 표 반환

    # ------------------------------------------------------------------
    # 추상 메서드 구현 — preprocess 템플릿이 순서대로 호출한다
    # ------------------------------------------------------------------

    def transformColumnName(self, df: pd.DataFrame) -> pd.DataFrame:
        """SCADA 컬럼명을 표준 _raw 이름으로 통일하는 메서드

        Args:
            - df: 한 행 = 한 터빈 x 한 시각인 세로 표

        Logic:
            - power/ws/wd 같은 짧은 이름과 이미 _raw가 붙은 이름을 둘 다 받아들인다
            - `_raw` 접미사로 가공 단계를 남겨두면 뒤에 _clean, _filled를 붙여도 충돌하지 않는다
        """
        df = df.rename(
            columns = {k: v for k, v in self.COLUMN_MAP.items() if k in df.columns}
        )
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"SCADA 세로 표에 필요한 컬럼이 없습니다: {missing}. "
                f"터빈이 컬럼으로 늘어선 wide 표라면 transformToLong으로 먼저 녹이세요"
            )
        return df
        # kst_dtm/turbine/power_raw/ws_raw/wd_raw를 갖춘 DataFrame 반환

    def transformTimeAxis(self, df: pd.DataFrame) -> pd.DataFrame:
        """시각을 datetime으로 바꾸고 시간 집계 키와 터빈 메타를 붙이는 메서드

        Args:
            - df: kst_dtm(10분 간격)과 turbine 컬럼을 가진 세로 표

        Logic:
            - hour_end = ceil("1h"). 공식 라벨(train_labels)의 kst_dtm은 **집계 구간의 종료 시각**이라
              01:00은 00:00:01~01:00:00 구간을 뜻한다. 즉 00:10~01:00의 10분값 6개가 01:00 칸에 들어간다
            - 여기서 터빈별 시간순 정렬을 끝내야 뒤의 shift·diff 연산이 터빈 경계를 넘지 않는다
        """
        df = df.copy()
        df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
        df["hour_end"] = df["kst_dtm"].dt.ceil("1h")

        attach = [c for c in META_COLS if c not in df.columns or c == "turbine"]
        df = df.merge(self.turbine_meta[attach], on = "turbine", how = "left")
        # 이미 있는 메타 컬럼은 다시 붙이지 않는다 (_x/_y 접미사 사고 방지)

        return df.sort_values(["turbine", "kst_dtm"], ignore_index = True)
        # 메타가 붙고 터빈별 시간순 정렬된 DataFrame 반환

    def checkPhysicalLimit(self, df: pd.DataFrame) -> pd.DataFrame:
        """SCADA 물리한계 플래그 9종과 QC 비트 코드를 붙이는 메서드

        Args:
            - df: transformTimeAxis로 메타를 붙이고 정렬을 마친 세로 표

        Logic:
            - 정격 105% 초과  : |발전량| > cap_kwh10m x 1.05 -> 비현실적인 값 (센서 오류)
            - 큰 음수         : 발전량 < -cap_kwh10m x 0.05 -> 센서·단위 오류 의심
            - 작은 음수       : -cap_kwh10m x 0.05 <= 발전량 < 0 -> 대기전력·보조전력일 수 있다
            - 컷아웃 이상     : 풍속 >= cut_out_ms -> 터빈을 세우는 게 정상이라 발전량 0이 맞다
            - 고풍속 무발전   : 5 m/s <= 풍속 < 컷아웃 인데 발전량 <= 0 -> 고장·정비·출력제한 후보
            - 순서가 있다. 터빈 불일치 검사는 is_power_outlier가 먼저 있어야 중앙값이 안 오염된다
        """
        df = df.copy()
        cap10 = df["cap_kwh10m"]
        # 터빈별 10분 정격 발전량(kWh). VESTAS 600, UNISON 700

        df["is_power_outlier"] = df["power_raw"].abs() > cap10 * POWER_CAP_RATIO
        df["is_power_neg_large"] = df["power_raw"] < -cap10 * NEG_POWER_RATIO
        df["is_power_neg_small"] = (df["power_raw"] < 0) & (~df["is_power_neg_large"])
        # 큰 음수를 먼저 확정한 뒤 나머지 음수를 작은 음수로 돌린다 (두 플래그가 겹치면 안 된다)

        df["is_ws_out_of_range"] = (df["ws_raw"] < 0) | (df["ws_raw"] > WS_PHYS_MAX)
        df["is_cut_out"] = df["ws_raw"] >= df["cut_out_ms"]
        df["is_low_wind_zero_power"] = (df["ws_raw"] < CUT_IN_MS) & (df["power_raw"] <= 0)
        # 바람도 없고 발전도 0. 정상 정지이므로 그대로 둔다
        df["is_high_wind_zero_power"] = (
            (df["ws_raw"] >= HIGH_WS_MS)
            & (df["ws_raw"] < df["cut_out_ms"])
            & (df["power_raw"] <= 0)
        )

        df = self.checkFrozenSensor(df)
        df = self.checkTurbineDisagreement(df)
        df["power_qc_flag"] = self.computeQcCode(df)
        return df
        # 플래그 9종 + power_qc_flag(int32)가 추가된 DataFrame 반환

    def checkFrozenSensor(self, df: pd.DataFrame) -> pd.DataFrame:
        """같은 값이 오래 반복되는 센서 고착을 탐지하는 메서드

        Args:
            - df: 터빈별 시간순 정렬을 마친 세로 표

        Logic:
            - 직전 행과 값이 같고 터빈도 같은 구간에 같은 번호를 매겨 구간 길이를 잰다
            - 길이 >= 18(= 3시간)이면 고착으로 본다
            - 0이 반복되는 건 무풍·정지 상태의 정상 동작이라 제외한다
        """
        df = df.copy()
        same_as_prev = (
            df["power_raw"].eq( df["power_raw"].shift() )
            & df["turbine"].eq( df["turbine"].shift() )
        )
        run_len = df.groupby( (~same_as_prev).cumsum() )["power_raw"].transform("size")
        # 값이 바뀔 때마다 번호가 올라가므로 '같은 값 구간'이 같은 번호를 갖는다

        df["is_frozen_sensor"] = (
            (run_len >= FROZEN_STEPS) & df["power_raw"].notna() & (df["power_raw"] != 0)
        )
        return df
        # is_frozen_sensor가 추가된 DataFrame 반환

    def checkTurbineDisagreement(self, df: pd.DataFrame) -> pd.DataFrame:
        """같은 그룹·같은 시각의 다른 터빈과 크게 어긋나는 값을 탐지하는 메서드

        Args:
            - df: is_power_outlier를 이미 가진 세로 표

        Logic:
            - |내 발전량 - 같은 그룹 같은 시각 중앙값| > cap_kwh10m x 0.5 이면 개별 장비 이상 후보
            - 중앙값을 낼 때 정격 초과 극단값을 먼저 빼야 기준 자체가 오염되지 않는다
        """
        df = df.copy()
        power_for_median = df["power_raw"].mask(df["is_power_outlier"])
        group_median = power_for_median.groupby(
            [df["kst_dtm"], df["group"]], observed = True
        ).transform("median")

        df["is_turbine_disagreement"] = (
            (power_for_median - group_median).abs() > df["cap_kwh10m"] * DISAGREE_RATIO
        )
        return df
        # is_turbine_disagreement가 추가된 DataFrame 반환

    def computeQcCode(self, df: pd.DataFrame) -> pd.Series:
        """플래그 9종의 상태를 정수 하나에 담는 메서드

        Args:
            - df: FLAG_COLS 9개 컬럼을 모두 가진 세로 표

        Logic:
            - i번째 플래그에 2^i를 배정해 더한다 (is_power_outlier = 1, is_power_neg_large = 2, ...)
            - 나중에 (power_qc_flag & 2**FLAG_COLS.index("is_cut_out")) > 0 으로 특정 상태만 골라본다
            - 컬럼 9개를 계속 들고 다니는 대신 int32 하나로 압축하는 것이 목적이다
        """
        missing = [c for c in self.FLAG_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"QC 코드 계산에 필요한 플래그가 없습니다: {missing}")

        qc_code = np.zeros(len(df), dtype = np.int32)
        for i, name in enumerate(self.FLAG_COLS):
            qc_code += df[name].to_numpy() * (1 << i)
            # True인 행에만 그 자리의 비트를 더한다

        return pd.Series(qc_code, index = df.index, name = "power_qc_flag")
        # 행마다 어떤 플래그가 걸렸는지를 담은 int32 시리즈 반환

    # ------------------------------------------------------------------
    # 훅 메서드 오버라이드 — SCADA는 결측 정제·보간이 핵심 단계다
    # ------------------------------------------------------------------

    def interpolateMissing(self, df: pd.DataFrame) -> pd.DataFrame:
        """못 믿을 값만 NaN으로 바꾸고 짧은 결측만 보간하는 메서드

        Args:
            - df: 플래그 9종을 가진 세로 표

        Logic:
            - NaN으로 바꾸는 건 DROP_FLAGS 3종(정격 초과 · 큰 음수 · 센서 고착)뿐이다
            - **발전량 0은 결측으로 바꾸지 않는다.** 작은 음수·고풍속 무발전·컷아웃도 그대로 둔다.
              전부 실제 운영 상태이고, 지우면 고장·정비를 실제 저발전으로 학습하게 된다
            - 풍향은 각도라서 선형보간하면 0도와 359도 사이에서 엉뚱한 값이 나온다.
              여기서는 안 건드리고 calculatePhysicalFeature에서 U·V로 바꾼 다음에 보간한다
        """
        df = df.copy()

        df["power_clean"] = df["power_raw"].mask( df[self.DROP_FLAGS].any(axis = 1) )
        df["ws_clean"] = df["ws_raw"].mask(df["is_ws_out_of_range"])
        df["wd_clean"] = df["wd_raw"] % FULL_TURN_DEG
        # UNISON의 -180~180 표기를 0~360으로 맞춘다 (VESTAS는 원래 0~360이라 값이 안 변한다)

        df["power_filled"], df["is_power_interpolated"] = interpolate_short_gap(
            df, col = "power_clean", key = "turbine", max_steps = self.max_interpolation_steps
        )
        df["ws_filled"], df["is_ws_interpolated"] = interpolate_short_gap(
            df, col = "ws_clean", key = "turbine", max_steps = self.max_interpolation_steps
        )

        n_valid = df.groupby(["turbine", "hour_end"], observed = True)["power_clean"].transform("count")
        df["n_valid_in_hour"] = n_valid
        df["is_partial_missing"] = (n_valid > 0) & (n_valid < MIN_SAMPLES_PER_HOUR)
        # 10분값이 1~5개만 남은 시간
        df["is_full_missing_hour"] = n_valid == 0
        # 그 시간이 통째로 비어있는 경우
        return df
        # _clean·_filled 컬럼과 시간 커버리지 정보가 추가된 DataFrame 반환

    def calculatePhysicalFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """풍향 U·V 변환과 이론 출력을 계산하는 메서드

        Args:
            - df: ws_filled(m/s)와 wd_clean(0~360도)을 가진 세로 표

        Logic:
            - u = -ws x sin(풍향), v = -ws x cos(풍향)
            - 각도를 직접 보간·평균하면 0도/359도 경계에서 엉뚱한 값이 나오므로
              성분으로 바꾼 뒤에 보간한다
            - 이론 출력 P = 0.5 x rho x A x V³ (W) -> 1000으로 나눠 kW
              실제 밀도는 LDAPS에서 오므로 여기서는 표준 밀도(1.225)로만 계산한다
            - 10분 에너지(kWh)에 6을 곱해 순간 출력(kW)으로 환산한다
        """
        df = df.copy()
        u, v = compute_uv_component(df["ws_filled"], df["wd_clean"])
        df["u_scada"] = u.astype("float32")
        df["v_scada"] = v.astype("float32")
        # 240만 행 표라 float32로 낮춰 메모리를 절반으로 줄인다

        df["u_scada"], df["is_u_interpolated"] = interpolate_short_gap(
            df, col = "u_scada", key = "turbine", max_steps = self.max_interpolation_steps
        )
        df["v_scada"], _ = interpolate_short_gap(
            df, col = "v_scada", key = "turbine", max_steps = self.max_interpolation_steps
        )
        df["dir_valid"] = df["ws_filled"] >= LOW_WS_MASK

        df["p_theory_std_kw"] = (
            0.5 * RHO_STD * df["swept_area_m2"] * df["ws_filled"] ** 3 / 1000
        ).astype("float32")
        # W -> kW 환산
        df["power_kw"] = (df["power_filled"] * MIN_SAMPLES_PER_HOUR).astype("float32")
        # 10분 에너지(kWh) x 6 = 순간 출력(kW)
        return df
        # U·V 성분과 이론 출력·순간 출력이 추가된 DataFrame 반환

    def transformCyclicFeature(self, df: pd.DataFrame) -> pd.DataFrame:
        """정제된 풍향을 sin/cos 주기성 피처로 변환하는 메서드"""
        return transform_wind_direction(df, wind_direction_cols = ["wd_clean"])
        # wd_clean_sin·wd_clean_cos가 추가된 DataFrame 반환

    # ------------------------------------------------------------------
    # 시간 집계 (구체) — 10분 -> 터빈 시간별 -> 그룹 시간별 2단계
    # ------------------------------------------------------------------

    def transformToTurbineHourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """정제된 10분 자료를 터빈 x 시간 단위로 집계하는 메서드

        Args:
            - df: preprocess를 마친 세로 표 (power_filled, ws_filled, u_scada, v_scada 포함)

        Logic:
            - power_sum_strict  : 10분값 6개가 다 있을 때만 인정 (min_count = 6)
            - power_sum_relaxed : 5/6만 있어도 인정하되 6/n으로 보정한다.
              보정을 안 하면 5개만 더한 값이 1시간 발전량으로 들어가 과소평가된다
            - 두 버전을 같이 만들어 나중에 검증 데이터에서 어느 쪽이 나은지 비교한다
        """
        need = ["power_filled", "ws_filled", "u_scada", "v_scada", "hour_end", "turbine"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"터빈 시간 집계에 필요한 컬럼이 없습니다: {missing}")

        grouped = df.groupby(["turbine", "hour_end"], observed = True)
        agg = grouped.agg(
            n_valid = ("power_filled", "count"),
            # 그 시간에 살아있는 10분값 개수 (0~6)
            power_sum_all = ("power_filled", "sum"),
            # 일단 단순 합계. min_count는 아래에서 직접 적용한다
            ws_mean = ("ws_filled", "mean"),
            ws_std = ("ws_filled", "std"),
            # 10분 사이 풍속 변동 = 난류 강도의 대리변수
            ws_max = ("ws_filled", "max"),
            u_mean = ("u_scada", "mean"),
            v_mean = ("v_scada", "mean"),
            is_high_wind_zero = ("is_high_wind_zero_power", "any"),
            is_cut_out_hour = ("is_cut_out", "any"),
            is_outlier_hour = ("is_power_outlier", "any"),
            is_interp_hour = ("is_power_interpolated", "any"),
        ).reset_index()

        ws_quantile = grouped["ws_filled"].quantile([0.25, 0.75]).unstack()
        ws_quantile.columns = ["ws_q25", "ws_q75"]
        agg = agg.merge(ws_quantile.reset_index(), on = ["turbine", "hour_end"], how = "left")

        agg["power_sum_strict"] = agg["power_sum_all"].where(
            agg["n_valid"] >= MIN_SAMPLES_PER_HOUR
        )
        agg["power_sum_relaxed"] = (
            agg["power_sum_all"] * MIN_SAMPLES_PER_HOUR / agg["n_valid"].replace(0, np.nan)
        ).where(agg["n_valid"] >= MIN_SAMPLES_RELAXED)
        # n_valid = 0인 시간에서 0으로 나누는 것을 방지하기 위해 먼저 NaN으로 바꾼다

        agg["group"] = agg["turbine"].map(self._turbine_group)
        agg["cap_kwh10m"] = agg["turbine"].map(self._turbine_cap_kwh10m)
        return agg
        # 한 행 = 한 터빈 x 한 시간인 집계표 반환

    def transformToGroupHourly(self, agg_turbine: pd.DataFrame) -> pd.DataFrame:
        """터빈 시간별 표를 KPX 그룹 시간별 표로 집계하는 메서드

        Args:
            - agg_turbine: transformToTurbineHourly의 결과표

        Logic:
            - 그룹 합계에 min_count = len(members)를 줘서 한 터빈이라도 비면 NaN이 되게 한다.
              이걸 빼면 터빈이 전부 죽어 있어도 합계가 0으로 나와 통신 장애가 저발전으로 둔갑한다
            - coverage_ratio  = Σ유효샘플 / (6 x 터빈수). 분모에 터빈수를 반드시 곱한다
            - avail_cap_ratio = 자료가 온전한 터빈의 설비용량 비율
            - 풍속은 scalar 평균과 vector 평균을 둘 다 보존한다.
              둘의 비율(resultant)이 그 시간 풍향이 얼마나 일정했는지를 알려준다
        """
        wide = {
            key: agg_turbine.pivot(index = "hour_end", columns = "turbine", values = key)
            for key in ["power_sum_strict", "power_sum_relaxed", "n_valid", "ws_mean", "u_mean", "v_mean"]
        }
        target_groups = [self.group] if self.group else list(KPX_GROUPS)
        # 그룹을 지정해 만든 전처리기면 그 그룹만 집계한다

        group_parts = []
        for group in target_groups:
            members = [
                t for t in wide["power_sum_strict"].columns if self._turbine_group.get(t) == group
            ]
            if not members:
                continue
                # 해당 기간에 그 그룹 터빈이 하나도 없으면 건너뛴다 (그룹3의 2022년)

            caps = np.array([self._turbine_cap_kwh10m[t] for t in members])
            cap_total = caps.sum() * MIN_SAMPLES_PER_HOUR
            # 그룹 1시간 설비용량(kWh). RATED_CAPACITY_KW와 같은 값이어야 한다

            power_strict = wide["power_sum_strict"][members].sum(axis = 1, min_count = len(members))
            power_relaxed = wide["power_sum_relaxed"][members].sum(axis = 1, min_count = len(members))

            coverage_ratio = wide["n_valid"][members].sum(axis = 1) / (
                MIN_SAMPLES_PER_HOUR * len(members)
            )
            avail_cap_ratio = (
                (wide["n_valid"][members] >= MIN_SAMPLES_PER_HOUR)
                .mul(caps, axis = 1).sum(axis = 1) / (caps.sum() + 1e-8)
            )
            # 0으로 나누는 경우를 방지하기 위해 작은 값 1e-8을 더함

            ws_scalar = wide["ws_mean"][members].mean(axis = 1)
            ws_std_across = wide["ws_mean"][members].std(axis = 1)
            # 터빈 간 풍속 편차. 후류 효과의 단서가 된다
            u_group = wide["u_mean"][members].mean(axis = 1)
            v_group = wide["v_mean"][members].mean(axis = 1)
            ws_vector = compute_wind_speed(u_group, v_group)
            wd_group = compute_wind_direction(u_group, v_group)
            # 풍향은 U·V를 평균한 뒤 복원한다. 각도를 직접 평균하면 안 된다

            resultant = (ws_vector / (ws_scalar + 1e-8)).clip(upper = 1.0)
            # 1이면 완전히 한 방향. 무풍 시간에 0으로 나누는 것을 방지하기 위해 1e-8을 더함
            spread = np.degrees( np.sqrt( -2 * np.log( resultant.clip(lower = 1e-6) ) ) )
            # 원형 표준편차(도). log(0)이 -inf가 되는 것을 막기 위해 하한을 둔다

            group_parts.append(pd.DataFrame({
                "hour_end": power_strict.index,
                "group": group,
                "scada_power_kwh": power_strict.to_numpy(),
                "scada_power_kwh_relaxed": power_relaxed.to_numpy(),
                "coverage_ratio": coverage_ratio.to_numpy(),
                "avail_cap_ratio": avail_cap_ratio.to_numpy(),
                "scada_ws_scalar": ws_scalar.to_numpy(),
                "scada_ws_vector": ws_vector.to_numpy(),
                "scada_ws_std": ws_std_across.to_numpy(),
                "scada_u": u_group.to_numpy(),
                "scada_v": v_group.to_numpy(),
                "scada_wd": wd_group.to_numpy(),
                "scada_dir_resultant": resultant.to_numpy(),
                "scada_dir_spread": spread.to_numpy(),
                "capacity_factor": (power_strict / cap_total).to_numpy(),
            }))

        if not group_parts:
            raise ValueError("집계할 그룹이 없습니다. turbine_meta의 group 값을 확인하세요")

        result = pd.concat(group_parts, ignore_index = True).dropna(subset = ["hour_end"])
        return result.sort_values(["group", "hour_end"], ignore_index = True)
        # 한 행 = 한 그룹 x 한 시간인 집계표 반환.
        # clipPower는 여기서 부르지 않는다. 정격 초과 실측은 센서 오류라는 증거라서
        # 잘라 감추지 않고 is_outlier_hour 플래그로 남긴다


__all__ = [
    "SCADAFeatureEngineer",
    "POWER_CAP_RATIO", "NEG_POWER_RATIO", "FROZEN_STEPS",
    "CUT_IN_MS", "HIGH_WS_MS", "WS_PHYS_MAX", "DISAGREE_RATIO",
    "MIN_SAMPLES_RELAXED", "LOW_WS_MASK", "FULL_TURN_DEG",
    "SWEPT_AREA_M2", "CUT_OUT_MS",
]
