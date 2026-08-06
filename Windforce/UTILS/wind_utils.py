"""바람·대기 물리 계산 유틸리티."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


R_D = 287.05
EPSILON = 0.62197
ONE_MINUS_EPS = 0.37803
RHO_STD = 1.225


def compute_wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.sqrt(u**2 + v**2)


def compute_wind_direction(u: pd.Series, v: pd.Series) -> pd.Series:
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def compute_uv_component(ws: pd.Series, wd: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(wd)
    return -ws * np.sin(radians), -ws * np.cos(radians)


def compute_angle_difference(wd_a: pd.Series, wd_b: pd.Series) -> pd.Series:
    return (wd_a - wd_b + 180) % 360 - 180


def compute_wind_shear(
    ws_low: pd.Series, ws_high: pd.Series, height_low: float, height_high: float
) -> pd.Series:
    ratio = ws_high.clip(lower=1e-3) / ws_low.clip(lower=1e-3)
    return np.log(ratio) / np.log(height_high / height_low)


def calculate_air_density(
    df: pd.DataFrame, t_col: str, q_col: str, p_col: str, prefix: str = ""
) -> pd.DataFrame:
    missing = sorted({t_col, q_col, p_col}.difference(df.columns))
    if missing:
        raise KeyError(f"공기밀도 계산에 필요한 컬럼이 없습니다: {missing}")

    result = df.copy()
    result[f"{prefix}e_pa"] = (
        result[q_col] * result[p_col] / (EPSILON + ONE_MINUS_EPS * result[q_col])
    )
    result[f"{prefix}air_density"] = (
        (result[p_col] - 0.378 * result[f"{prefix}e_pa"]) / (R_D * result[t_col])
    )
    return result


def calculate_wind_power_density(air_density: pd.Series, ws: pd.Series) -> pd.Series:
    return 0.5 * air_density * ws**3


def transform_wind_direction(
    df: pd.DataFrame, wind_direction_cols: Iterable[str]
) -> pd.DataFrame:
    result = df.copy()
    for col in wind_direction_cols:
        if col not in result.columns:
            continue
        radians = np.deg2rad(result[col])
        result[f"{col}_sin"] = np.sin(radians)
        result[f"{col}_cos"] = np.cos(radians)
    return result


__all__ = [
    "R_D", "EPSILON", "ONE_MINUS_EPS", "calculate_air_density",
    "calculate_wind_power_density", "compute_angle_difference",
    "compute_uv_component", "compute_wind_direction", "compute_wind_shear",
    "compute_wind_speed", "transform_wind_direction",
]
