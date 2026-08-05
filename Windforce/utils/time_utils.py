"""시간축·범주·시계열 입력 변환 유틸리티."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .wind_utils import compute_angle_difference


MAX_INTERP_STEPS = 2
SEASON_MAP = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
SEQ_LEN = 24


def transform_cyclic_time(df: pd.DataFrame, time_col: str = "forecast_kst_dtm") -> pd.DataFrame:
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


def transform_categorical(
    df: pd.DataFrame,
    columns: Iterable[str],
    categories: Mapping[str, Iterable[Any]] | None = None,
    drop_original: bool = True,
) -> pd.DataFrame:
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


def interpolate_short_gap(
    df: pd.DataFrame,
    col: str,
    key: str = "turbine",
    max_steps: int = MAX_INTERP_STEPS,
) -> tuple[pd.Series, pd.Series]:
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


def transform_group_time_feature(
    group_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    time_meta_cols: Iterable[str],
    time_col: str = "forecast_kst_dtm",
) -> pd.DataFrame:
    cols = [c for c in time_meta_cols if c in grid_df.columns]
    result = group_df
    if cols:
        time_meta = grid_df.groupby(time_col, observed=True)[cols].first().reset_index()
        result = result.merge(time_meta, on=time_col, how="left")
    return transform_cyclic_time(result, time_col=time_col)


def transform_forecast_diff(
    df: pd.DataFrame,
    ws_col: str,
    u_col: str,
    v_col: str,
    wd_col: str,
    key: str = "group",
    prefix: str = "",
) -> pd.DataFrame:
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


def select_group(df: pd.DataFrame, group: str | None, group_col: str = "group") -> pd.DataFrame:
    if group is None or group_col not in df.columns:
        return df.copy()
    return df.loc[df[group_col].eq(group)].copy()


def transform_to_model_input(
    df: pd.DataFrame,
    time_col: str = "forecast_kst_dtm",
    key: str = "group",
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
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
        result = transform_categorical(
            result, ["season"], categories={"season": list(SEASON_MAP.values())}
        )
    if drop_cols:
        result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    return result


def compute_feature_columns(
    df: pd.DataFrame,
    time_col: str = "forecast_kst_dtm",
    key: str = "group",
    target_col: str | None = None,
) -> list[str]:
    exclude = {time_col, key, target_col} - {None}
    return [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def transform_scale(
    df: pd.DataFrame, scale_param: Mapping[str, tuple[float, float]]
) -> pd.DataFrame:
    result = df.copy()
    for col, (mean, std) in scale_param.items():
        if col in result.columns:
            result[col] = (result[col] - mean) / std
    return result


def transform_to_sequence(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    seq_len: int = SEQ_LEN,
    key: str = "group",
    time_col: str = "forecast_kst_dtm",
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
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


__all__ = [
    "MAX_INTERP_STEPS", "SEASON_MAP", "SEQ_LEN", "compute_feature_columns",
    "interpolate_short_gap", "select_group", "transform_categorical",
    "transform_cyclic_time", "transform_forecast_diff", "transform_group_time_feature",
    "transform_scale", "transform_to_model_input", "transform_to_sequence",
]
