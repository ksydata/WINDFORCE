"""격자 좌표·IDW 공간 집계 유틸리티."""

import numpy as np
import pandas as pd


KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
EARTH_RADIUS_KM = 6371.0
IDW_K = 4
IDW_POWER = 2.0
MIN_DISTANCE_KM = 1e-6


class PreprocessingError(ValueError):
    """전처리 입력 계약 또는 변환 과정에서 발생한 오류."""


def compute_haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_group_weight(
    turbine_meta: pd.DataFrame,
    grid_coords: pd.DataFrame,
    k: int = IDW_K,
    power: float = IDW_POWER,
) -> np.ndarray:
    missing = [c for c in ("group", "lat", "lon") if c not in turbine_meta.columns]
    if "cap_kw" not in turbine_meta.columns:
        if "capacity_mw" in turbine_meta.columns:
            turbine_meta = turbine_meta.copy()
            turbine_meta["cap_kw"] = turbine_meta["capacity_mw"] * 1000
        else:
            missing.append("cap_kw")
    if missing:
        raise KeyError(f"turbine_meta에 필요한 컬럼이 없습니다: {missing}")
    if not {"latitude", "longitude"}.issubset(grid_coords.columns):
        raise KeyError("grid_coords에는 latitude와 longitude가 필요합니다")
    if grid_coords.empty:
        raise PreprocessingError("격자 좌표가 비어 있습니다")

    meta = turbine_meta.reset_index(drop=True)
    grid_lat = grid_coords["latitude"].to_numpy(dtype=float)
    grid_lon = grid_coords["longitude"].to_numpy(dtype=float)
    distance = np.zeros((len(meta), len(grid_coords)))
    for i in range(len(meta)):
        distance[i] = compute_haversine_distance(
            float(meta.loc[i, "lat"]), float(meta.loc[i, "lon"]), grid_lat, grid_lon
        )

    neighbor_count = max(1, min(int(k), len(grid_coords)))
    weight_turbine = np.zeros_like(distance)
    for i in range(len(meta)):
        near = np.argsort(distance[i])[:neighbor_count]
        inverse = 1.0 / np.maximum(distance[i, near], MIN_DISTANCE_KM) ** power
        weight_turbine[i, near] = inverse / inverse.sum()

    weight_group = np.zeros((len(KPX_GROUPS), len(grid_coords)))
    for j, group in enumerate(KPX_GROUPS):
        selected = (meta["group"] == group).to_numpy()
        if not selected.any():
            continue
        caps = meta.loc[selected, "cap_kw"].to_numpy(dtype=float)
        weight_group[j] = (
            (caps[:, None] * weight_turbine[selected]).sum(axis=0) / (caps.sum() + 1e-8)
        )
    return weight_group


def transform_to_group_feature(
    df: pd.DataFrame,
    features: list[str],
    weight_group: np.ndarray,
    grid_index: pd.Index,
    time_col: str = "forecast_kst_dtm",
    target_groups: list[str] | None = None,
) -> pd.DataFrame:
    missing = [c for c in (time_col, "grid_id", *features) if c not in df.columns]
    if missing:
        raise KeyError(f"공간 가중평균에 필요한 컬럼이 없습니다: {missing}")
    if len(grid_index) != weight_group.shape[1]:
        raise ValueError("grid_index와 가중치 행렬의 격자 수가 다릅니다")

    times = np.sort(df[time_col].unique())
    idw_values = {}
    for feature in features:
        matrix = (
            df.pivot(index=time_col, columns="grid_id", values=feature)
            .reindex(columns=grid_index)
        )
        idw_values[feature] = matrix.to_numpy() @ weight_group.T

    target_groups = target_groups or list(KPX_GROUPS)
    group_frames = []
    for j, group in enumerate(KPX_GROUPS):
        if group not in target_groups:
            continue
        part = pd.DataFrame({feature: idw_values[feature][:, j] for feature in features})
        part.insert(0, "group", group)
        part.insert(0, time_col, times)
        group_frames.append(part)
    return pd.concat(group_frames, ignore_index=True)


__all__ = [
    "KPX_GROUPS", "EARTH_RADIUS_KM", "IDW_K", "IDW_POWER", "MIN_DISTANCE_KM",
    "PreprocessingError", "compute_group_weight", "compute_haversine_distance",
    "transform_to_group_feature",
]
