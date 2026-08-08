"""시간축·범주·시계열 입력 변환 유틸리티."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .wind_utils import compute_angle_difference


MAX_INTERP_STEPS = 2
# 연속 결측 2칸(= 20분) 이하만 보간한다. 더 길면 정비·고장이라 손대지 않는다
SEASON_MAP = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
# (월 % 12) // 3 결과를 계절 이름으로. 12·1·2월이 winter가 된다
SEQ_LEN = 24
# LSTM 기본 입력 길이. 예보 한 발행분이 24개 대상 시각을 덮으므로 하루치를 기본으로 둔다


class TimeUtils:
    """시간축·범주·시계열 입력 변환을 모아둔 정적 메서드 전용 클래스.

    상태를 갖지 않는 순수 변환 절차라 인스턴스화하지 않고 staticmethod로만 호출한다.
    """

    @staticmethod
    def transform_cyclic_time(df: pd.DataFrame, time_col: str = "forecast_kst_dtm") -> pd.DataFrame:
        """시각을 일·연 주기 피처와 계절 라벨로 변환한다.

        Args:
            df:       시간 컬럼을 가진 DataFrame.
            time_col: 기준 시간 컬럼명.

        Returns:
            hour_sin/cos, doy_sin/cos, month, season 컬럼이 추가된 DataFrame.

        Logic:
            - hour_sin/cos = sin·cos(2π × 시 / 24)          → 23시와 0시를 붙인다
            - doy_sin/cos  = sin·cos(2π × 연중일 / 365.25)  → 12월 31일과 1월 1일을 붙인다
            - 365가 아닌 365.25로 나누는 것은 윤년 누적 오차를 없애기 위함이다
        """
        if time_col not in df.columns:
            raise KeyError(f"시간 컬럼이 없습니다: {time_col}")
        result = df.copy()
        dt = pd.to_datetime(result[time_col], errors="raise")
        result["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        result["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        result["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
        result["month"] = dt.dt.month
        result["season"] = (dt.dt.month % 12 // 3).map(SEASON_MAP)
        return result

    @staticmethod
    def transform_categorical(
        df: pd.DataFrame,
        columns: Iterable[str],
        categories: Mapping[str, Iterable[Any]] | None = None,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """범주형 컬럼을 원핫 인코딩으로 변환한다.

        Args:
            df:            변환할 DataFrame.
            columns:       원핫으로 펼칠 컬럼 이름 목록.
            categories:    {컬럼: 범주 순서} 딕셔너리. None이면 관측된 값만 사용한다.
                           학습·평가 구간이 다른 범주를 가질 수 있으므로 명시적으로 전달하는 것을 권장.
            drop_original: True면 원본 컬럼을 제거하고 원핫 컬럼만 남긴다.

        Returns:
            지정한 컬럼이 원핫 컬럼(int8)으로 대체된 DataFrame.
        """
        result = df.copy()
        categories = categories or {}
        for column in columns:
            if column not in result.columns:
                continue
            values = pd.Categorical(result[column], categories=categories.get(column))
            encoded = pd.get_dummies(values, prefix=column, dtype="int8")
            encoded.index = result.index
            if drop_original:
                result = result.drop(columns=[column])
            result = pd.concat([result, encoded], axis=1)
        return result

    @staticmethod
    def interpolate_short_gap(
        df: pd.DataFrame,
        col: str,
        key: str = "turbine",
        max_steps: int = MAX_INTERP_STEPS,
    ) -> tuple[pd.Series, pd.Series]:
        """연속 결측이 짧은 구간만 선형보간한다.

        Args:
            df:        보간 대상 컬럼과 경계 컬럼을 가진 DataFrame (key·시간순 정렬 상태여야 한다).
            col:       보간할 컬럼명.
            key:       구간이 넘어가면 안 되는 경계 컬럼명 (터빈이 바뀌면 구간도 끊긴다).
            max_steps: 이 칸수 이하의 연속 결측만 채운다.

        Returns:
            (보간을 마친 값 Series, 실제로 채워진 위치 bool Series) 튜플.

        Logic:
            - pandas interpolate(limit=n)만 쓰면 긴 구멍의 앞 n칸이 채워진다.
              정비·고장으로 3일 비어 있는 구간의 앞 20분이 메워지면 그 시간은 실제 저발전으로 학습된다.
            - 결측 구간 전체 길이를 먼저 재고, 길이가 max_steps 이하인 구간만 채운다.
            - limit_area="inside"라 양쪽에 값이 있을 때만 채운다 (표 끝의 결측은 안 건드린다).
        """
        if col not in df.columns or key not in df.columns:
            raise KeyError(f"보간에 필요한 컬럼이 없습니다: {col}, {key}")
        na_mask = df[col].isna()
        same_run = na_mask.eq(na_mask.shift()) & df[key].eq(df[key].shift())
        run_len = df.groupby((~same_run).cumsum())[col].transform("size")
        short_gap = na_mask & (run_len <= max_steps)
        interpolated = df.groupby(key, observed=True)[col].transform(
            lambda s: s.interpolate(method="linear", limit_area="inside")
        )
        filled = df[col].where(~short_gap, interpolated)
        return filled, short_gap & filled.notna()

    @staticmethod
    def transform_group_time_feature(
        group_df: pd.DataFrame,
        grid_df: pd.DataFrame,
        time_meta_cols: Iterable[str],
        time_col: str = "forecast_kst_dtm",
    ) -> pd.DataFrame:
        """격자 표의 시간 메타를 그룹 표에 다시 붙이고 순환 인코딩한다.

        Args:
            group_df:      공간 가중평균을 마친 그룹 단위 표.
            grid_df:       시간 메타(lead_hour, target_hour 등)를 가진 격자 단위 원본 표.
            time_meta_cols: 되살릴 시간 메타 컬럼 목록.
            time_col:      예보 대상 시각 컬럼명.

        Returns:
            시간 메타와 순환 피처(hour_sin/cos, doy_sin/cos 등)가 복원된 그룹 단위 DataFrame.

        Logic:
            - 공간 가중평균은 지정한 피처만 통과시키므로 시간 메타가 날아간다.
              시각당 하나뿐인 값이라 groupby(시각).first()로 되살린 뒤 다시 인코딩한다.
        """
        cols = [c for c in time_meta_cols if c in grid_df.columns]
        result = group_df
        if cols:
            time_meta = grid_df.groupby(time_col, observed=True)[cols].first().reset_index()
            result = result.merge(time_meta, on=time_col, how="left")
        return TimeUtils.transform_cyclic_time(result, time_col=time_col)

    @staticmethod
    def transform_forecast_diff(
        df: pd.DataFrame,
        ws_col: str,
        u_col: str,
        v_col: str,
        wd_col: str,
        key: str = "group",
        prefix: str = "",
    ) -> pd.DataFrame:
        """예보 시계열의 변화량·램프 파생 피처를 추가한다.

        Args:
            df:     그룹별 시간순으로 정렬을 마친 예보 표 (정렬은 호출부 책임이다).
            ws_col: 풍속 컬럼명.
            u_col:  동서 성분(U) 컬럼명.
            v_col:  남북 성분(V) 컬럼명.
            wd_col: 풍향 컬럼명.
            key:    변화량 경계 컬럼. 그룹이 바뀌면 시계열도 끊긴다.
            prefix: 생성 컬럼 접두사. LDAPS와 GFS를 한 표에 합쳐도 이름이 부딪히지 않게 한다.

        Returns:
            ws_diff_1h/3h, u_diff_1h, v_diff_1h, ws_ramp_rate,
            ws_next3h_max/min, wd_rotation 8종이 추가된 DataFrame.

        Logic:
            - ws_diff_1h/3h : 램프(급격한 증감) 구간 탐지용 변화량.
            - ws_next3h_max/min : 앞으로 3시간 안에 닥칠 최대·최소 풍속 (예보라 누수가 아니다).
            - wd_rotation : 풍향 회전 속도. -180~180으로 감은 뒤 절댓값.
        """
        required = [key, ws_col, u_col, v_col, wd_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"변화량 계산에 필요한 컬럼이 없습니다: {missing}")

        result = df.copy()
        by_key = result.groupby(key, observed=True)
        name = f"{prefix}_" if prefix else ""
        result[f"{name}ws_diff_1h"] = by_key[ws_col].diff(1)
        result[f"{name}ws_diff_3h"] = by_key[ws_col].diff(3)
        result[f"{name}u_diff_1h"] = by_key[u_col].diff(1)
        result[f"{name}v_diff_1h"] = by_key[v_col].diff(1)
        result[f"{name}ws_ramp_rate"] = result[f"{name}ws_diff_1h"].abs()
        result[f"{name}ws_next3h_max"] = by_key[ws_col].transform(
            lambda s: s.shift(-1).iloc[::-1].rolling(3, min_periods=1).max().iloc[::-1]
        )
        result[f"{name}ws_next3h_min"] = by_key[ws_col].transform(
            lambda s: s.shift(-1).iloc[::-1].rolling(3, min_periods=1).min().iloc[::-1]
        )
        result[f"{name}wd_rotation"] = compute_angle_difference(
            result[wd_col], by_key[wd_col].shift(1)
        ).abs()
        return result

    @staticmethod
    def select_group(df: pd.DataFrame, group: str | None, group_col: str = "group") -> pd.DataFrame:
        """설정된 KPX 그룹 행만 선택한다.

        Args:
            df:        그룹 컬럼을 포함한 DataFrame.
            group:     선택할 그룹 이름. None이면 전체 반환.
            group_col: 그룹 식별자 컬럼명.

        Returns:
            지정 그룹의 행만 남긴 DataFrame 복사본. group이 None이면 전체 복사본.
        """
        if group is None or group_col not in df.columns:
            return df.copy()
        return df.loc[df[group_col].eq(group)].copy()

    @staticmethod
    def transform_to_model_input(
        df: pd.DataFrame,
        time_col: str = "forecast_kst_dtm",
        key: str = "group",
        drop_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """LSTM이 받아들일 수 있는 숫자형 2차원 표로 정리한다.

        Args:
            df:        파생변수 생성을 마친 그룹 x 시각 표.
            time_col:  예보 대상 시각 컬럼명.
            key:       그룹 식별자 컬럼명.
            drop_cols: 피처에서 뺄 컬럼 목록 (라벨 누수 위험이 있는 컬럼 등).

        Returns:
            숫자형 피처와 식별자(time_col·key)만 남은 DataFrame.

        Logic:
            - bool 플래그 → 0/1 int8. LSTM은 True/False를 그대로 처리하지 못한다.
            - season 같은 문자열 → 원핫. 범주를 4개로 고정해 평가 구간에서도 컬럼이 같다.
            - datetime과 식별자는 피처에서 빼되 컬럼 자체는 남긴다 (시퀀스를 자를 때 필요).
            - 정렬은 (그룹, 시각) 순으로 고정한다. 시퀀스가 역순으로 잘리는 사고를 막는다.
        """
        result = df.copy()
        if time_col in result.columns:
            result[time_col] = pd.to_datetime(result[time_col], errors="raise")
        sort_keys = [c for c in (key, time_col) if c in result.columns]
        if sort_keys:
            result = result.sort_values(sort_keys, ignore_index=True)
        for col in result.columns:
            if result[col].dtype == bool:
                result[col] = result[col].astype("int8")
        if "season" in result.columns:
            result = TimeUtils.transform_categorical(
                result, ["season"], categories={"season": list(SEASON_MAP.values())}
            )
        if drop_cols:
            result = result.drop(columns=[c for c in drop_cols if c in result.columns])
        return result

    @staticmethod
    def compute_feature_columns(
        df: pd.DataFrame,
        time_col: str = "forecast_kst_dtm",
        key: str = "group",
        target_col: str | None = None,
    ) -> list[str]:
        """모델 입력으로 쓸 숫자형 피처 컬럼 목록을 반환한다.

        Args:
            df:         transform_to_model_input을 거친 표.
            time_col:   제외할 시각 식별자 컬럼명.
            key:        제외할 그룹 식별자 컬럼명.
            target_col: 제외할 정답 컬럼명. None이면 제외하지 않는다.

        Returns:
            숫자형이고 식별자·정답이 아닌 컬럼 이름 리스트.
        """
        exclude = {time_col, key, target_col} - {None}
        return [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

    @staticmethod
    def transform_scale(
        df: pd.DataFrame, scale_param: Mapping[str, tuple[float, float]]
    ) -> pd.DataFrame:
        """학습 구간 통계(평균·표준편차)로 표준화한다.

        Args:
            df:          표준화할 표 (학습·검증·평가 어느 구간이든 가능).
            scale_param: {컬럼: (평균, 표준편차)} 딕셔너리.
                         반드시 **학습 구간**에서 계산한 통계여야 한다.
                         구간마다 다시 계산하면 미래 정보가 새어 들어온다.

        Returns:
            표준화된 DataFrame 복사본.
        """
        result = df.copy()
        for col, (mean, std) in scale_param.items():
            if col in result.columns:
                result[col] = (result[col] - mean) / std
        return result

    @staticmethod
    def transform_to_sequence(
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None = None,
        seq_len: int = SEQ_LEN,
        key: str = "group",
        time_col: str = "forecast_kst_dtm",
        dropna: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
        """2차원 표를 LSTM 입력용 3차원 시퀀스로 자른다.

        Args:
            df:           transform_to_model_input·transform_scale을 마친 (그룹 x 시각) 표.
            feature_cols: 입력 피처 컬럼 목록.
            target_col:   정답 컬럼명. None이면 X와 인덱스만 반환한다 (평가 데이터용).
            seq_len:      한 시퀀스의 길이 (기본 24 = 하루).
            key:          시퀀스가 넘어가면 안 되는 경계 컬럼명 (그룹이 바뀌면 시퀀스도 끊긴다).
            time_col:     예보 대상 시각 컬럼명.
            dropna:       True면 창 안에 결측이 하나라도 있는 표본을 버린다.

        Returns:
            (X: shape=(samples, seq_len, n_features),
             y: shape=(samples,) 또는 None,
             index: 창 마지막 시각 인덱스 DataFrame) 튜플.

        Logic:
            - y는 창의 **마지막 시각** 값이다. 과거 seq_len시간을 보고 그 시점을 맞추는 구조.
            - 그룹 경계를 넘는 창은 만들지 않는다.
            - dropna=True면 LSTM이 NaN을 만나 손실이 통째로 NaN이 되는 사고를 방지한다.
        """
        if seq_len < 1:
            raise ValueError("seq_len은 1 이상이어야 합니다")
        missing = [c for c in feature_cols if c not in df.columns]
        if target_col is not None and target_col not in df.columns:
            missing.append(target_col)
        if missing:
            raise KeyError(f"시퀀스 입력 컬럼이 없습니다: {missing}")

        x_parts, y_parts, index_parts = [], [], []
        groups = df[key].drop_duplicates().tolist() if key in df.columns else [None]
        for group in groups:
            part = df if group is None else df.loc[df[key].eq(group)]
            part = part.sort_values(time_col) if time_col in part.columns else part
            if len(part) < seq_len:
                continue
            values = part[feature_cols].to_numpy(dtype="float32")
            window = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
            x_parts.append(window.transpose(0, 2, 1))
            tail = part.iloc[seq_len - 1:]
            index_parts.append(tail[[c for c in (key, time_col) if c in part.columns]])
            if target_col is not None:
                y_parts.append(tail[target_col].to_numpy(dtype="float32"))

        if not x_parts:
            raise ValueError(f"시퀀스를 만들 수 있는 구간이 없습니다 (seq_len = {seq_len})")
        x_seq = np.concatenate(x_parts, axis=0)
        index_df = pd.concat(index_parts, ignore_index=True)
        y_seq = np.concatenate(y_parts, axis=0) if target_col is not None else None
        if dropna:
            keep = ~np.isnan(x_seq).any(axis=(1, 2))
            if y_seq is not None:
                keep &= ~np.isnan(y_seq)
            x_seq = x_seq[keep]
            y_seq = y_seq[keep] if y_seq is not None else None
            index_df = index_df.loc[keep].reset_index(drop=True)
        return x_seq, y_seq, index_df


# 기존 호출부(Preprocessing 서브패키지 등)가 자유 함수로 import하던 이름을 그대로 유지한다.
transform_cyclic_time = TimeUtils.transform_cyclic_time
transform_categorical = TimeUtils.transform_categorical
interpolate_short_gap = TimeUtils.interpolate_short_gap
transform_group_time_feature = TimeUtils.transform_group_time_feature
transform_forecast_diff = TimeUtils.transform_forecast_diff
select_group = TimeUtils.select_group
transform_to_model_input = TimeUtils.transform_to_model_input
compute_feature_columns = TimeUtils.compute_feature_columns
transform_scale = TimeUtils.transform_scale
transform_to_sequence = TimeUtils.transform_to_sequence


__all__ = [
    "MAX_INTERP_STEPS", "SEASON_MAP", "SEQ_LEN", "TimeUtils",
    "compute_feature_columns",
    "interpolate_short_gap", "select_group", "transform_categorical",
    "transform_cyclic_time", "transform_forecast_diff", "transform_group_time_feature",
    "transform_scale", "transform_to_model_input", "transform_to_sequence",
]
