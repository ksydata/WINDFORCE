"""바람·대기 물리 계산 유틸리티."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


R_D = 287.05
# 건조공기 기체상수 J/(kg·K)
EPSILON = 0.62197
# 건조공기/수증기 기체상수 비(Rd/Rv). 수증기압 식의 분모에 들어간다
ONE_MINUS_EPS = 0.37803
# 1 - EPSILON. 0.622/0.378로 반올림하면 밀도 4번째 자리가 흔들려 상수로 멃 박아둔다
RHO_STD = 1.225
# 표준 대기 공기밀도 kg/m³. 계산 결과가 현실적인지 비교하는 기준


def compute_wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    """U·V 바람 성분으로 풍속을 계산한다.

    Args:
        u: 동서 성분(m/s). 동쪽으로 부는 바람이 +.
        v: 남북 성분(m/s). 북쪽으로 부는 바람이 +.

    Returns:
        풍속(m/s) = sqrt(u² + v²) 시리즈.
    """
    return np.sqrt(u**2 + v**2)


def compute_wind_direction(u: pd.Series, v: pd.Series) -> pd.Series:
    """기상학적 풍향(0~360도)을 계산한다.

    Args:
        u: 동서 성분(m/s).
        v: 남북 성분(m/s).

    Returns:
        기상학적 풍향(도). 바람이 '불어오는' 방향이라 부호를 뒤집는다
        (북 0, 동 90, 남 180, 서 270). atan2를 써서 사새를 모두 구분한다.
    """
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def compute_uv_component(ws: pd.Series, wd: pd.Series) -> tuple[pd.Series, pd.Series]:
    """풍속과 기상학적 풍향을 U·V 성분으로 변환한다.

    Args:
        ws: 풍속(m/s).
        wd: 기상학적 풍향(0~360도).

    Returns:
        (U 성분, V 성분) 튜플. u = -ws×sin(풍향), v = -ws×cos(풍향).
        각도를 직접 평균하면 0도와 359도 사이에서 업당한 값이 나오므로
        성분으로 바꽈 뒤 평균한다.
    """
    radians = np.deg2rad(wd)
    return -ws * np.sin(radians), -ws * np.cos(radians)


def compute_angle_difference(wd_a: pd.Series, wd_b: pd.Series) -> pd.Series:
    """두 풍향의 차이를 -180~180도로 세서 계산한다.

    Args:
        wd_a, wd_b: 기상학적 풍향(0~360도).

    Returns:
        (wd_a - wd_b)를 -180~180도 범위로 감은 차 시리즈.
        350도와 10도의 차이는 340도가 아닌 -20도다.
    """
    return (wd_a - wd_b + 180) % 360 - 180


def compute_wind_shear(
    ws_low: pd.Series, ws_high: pd.Series, height_low: float, height_high: float
) -> pd.Series:
    """두 고도의 풍속으로 멱법칙 전단지수(alpha)를 계산한다.

    Args:
        ws_low:      낙은 고도 풍속(m/s).
        ws_high:     높은 고도 풍속(m/s).
        height_low:  낙은 고도(m).
        height_high: 높은 고도(m).

    Returns:
        전단지수 alpha 시리즈. alpha = ln(V2/V1) / ln(h2/h1).
        alpha를 알면 허브 높이로 풍속을 외삽할 수 있다.
        무풍 구간에서 log(0)이 -inf가 되지 않도록 비운 풍속에 1e-3 m/s 하한을 남긴다.
    """
    ratio = ws_high.clip(lower=1e-3) / ws_low.clip(lower=1e-3)
    return np.log(ratio) / np.log(height_high / height_low)


def calculate_air_density(
    df: pd.DataFrame, t_col: str, q_col: str, p_col: str, prefix: str = ""
) -> pd.DataFrame:
    """기온·비습·기압으로 습윤공기 밀도를 계산한다.

    Args:
        df:     계산에 필요한 컨럼을 가진 DataFrame.
        t_col:  기온 컨럼 (K 단위).
        q_col:  비습 컨럼 (kg/kg 단위).
        p_col:  지표면 기압 컨럼 (Pa 단위).
        prefix: 결과 컨럼 접두사. LDAPS와 GFS를 한 표에 합칠 때 이름 충돌을 막는다.

    Returns:
        ``{prefix}e_pa``(Pa)와 ``{prefix}air_density``(kg/m³) 컨럼이 추가된 DataFrame.

    Logic:
        - 수증기압 e = (q × P) / (EPSILON + ONE_MINUS_EPS × q)
        - 습윤공기 밀도 rho = (P - 0.378 × e) / (R_D × T)
        - 기온이 섭씨거나 기압이 hPa이면 결과가 통째로 틀린다.
    """
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
    """공기밀도와 풍속으로 풍력에너지 밀도를 계산한다.

    Args:
        air_density: 공기밀도(kg/m³).
        ws:          풍속(m/s).

    Returns:
        단위면적당 바람이 가진 힘(W/m²) = 0.5 × rho × V³ 시리즈.
        발전량이 여기 정비례한다고 강제하지 않고 입력변수로만 쓴다.
    """
    return 0.5 * air_density * ws**3


def transform_wind_direction(
    df: pd.DataFrame, wind_direction_cols: Iterable[str]
) -> pd.DataFrame:
    """풍향 컨럼에 주기성 인코딩(sin/cos)을 붙인다.

    Args:
        df:                 풍향 컨럼을 가진 DataFrame.
        wind_direction_cols: 인코딩할 풍향 컨럼명 목록 (도 단위).

    Returns:
        컨럼마다 ``{col}_sin``과 ``{col}_cos`` 컨럼이 추가된 DataFrame.

    Logic:
        0~360도 스케일은 두 값을 가장 먼 값으로 취급해 수치 왜곡을 일으키므로
        sin/cos로 풍향의 연속성을 보존한다.
    """
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
