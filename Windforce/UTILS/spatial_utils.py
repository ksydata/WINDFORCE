"""격자 좌표·IDW 공간 집계 유틸리티."""

from __future__ import annotations

import numpy as np
import pandas as pd


KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
# KPX 그룹 이름 순서를 고정해 행렬 인덱스에서 계속 재사용한다
EARTH_RADIUS_KM = 6371.0
# 하버사인 거리 계산에 쓰는 지구 평균 반지름(km)
IDW_K = 4
# 터빈마다 가까운 격자 4개만 사용한다. 전 격자를 다 쓰면 먼 격자가 값을 희석시킨다
IDW_POWER = 2.0
# 역거리 가중의 지수. 1/거리^2 (거리가 2배 멀어지면 영향력은 1/4)
MIN_DISTANCE_KM = 1e-6
# 터빈이 격자점과 정확히 겹칠 때 0으로 나누는 것을 방지하는 하한값


class PreprocessingError(ValueError):
    """전처리 입력 계약 또는 변환 과정에서 발생한 오류."""


class SpatialUtils:
    """격자 좌표 변환과 IDW(역거리가중) 공간 집계를 모아둔 정적 메서드 전용 클래스.

    상태를 갖지 않는 순수 계산 절차라 인스턴스화하지 않고 staticmethod로만 호출한다.
    """

    @staticmethod
    def compute_haversine_distance(
        lat1: float | np.ndarray,
        lon1: float | np.ndarray,
        lat2: float | np.ndarray,
        lon2: float | np.ndarray,
    ) -> np.ndarray:
        """위경도 두 지점 사이의 지구 표면 거리를 계산한다 (하버사인 공식).

        Args:
            lat1, lon1: 기준 지점의 위도·경도 (도 단위, scalar 또는 shape=(n,))
            lat2, lon2: 대상 지점의 위도·경도 (도 단위, scalar 또는 shape=(n,))

        Returns:
            두 지점 사이의 대권 거리(km). 태백 산지는 위도 37도라 위경도 1도의
            실제 거리가 다르므로 유클리드 거리를 쓰면 안 된다.
        """
        p1, p2 = np.radians(lat1), np.radians(lat2)
        # 위도는 0~90도라 라디안으로 바꾸면 0~π/2. 경도는 -180~180도라 라디안으로 바꾸면 -π~π
        dphi = p2 - p1
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    @staticmethod
    def compute_group_weight(
        turbine_meta: pd.DataFrame,
        grid_coords: pd.DataFrame,
        k: int = IDW_K,
        power: float = IDW_POWER,
    ) -> np.ndarray:
        """그룹 x 격자 역거리(IDW) 가중치 행렬을 계산한다.

        Args:
            turbine_meta: 터빈 1기 = 1행. ``group``, ``lat``, ``lon``, ``cap_kw`` 컬럼 필수.
                          ``group``은 1/2/3 또는 ``kpx_group_1/2/3`` 형식을 허용한다.
                          ``cap_kw`` 대신 ``capacity_mw`` 가 있으면 자동으로 kW 환산한다.
            grid_coords:  격자 1개 = 1행. ``latitude``, ``longitude`` 컬럼 필수
                          (index = grid_id).
            k:    터빈마다 사용할 이웃 격자 개수.
            power: 역거리 가중의 지수. 1/d^power.

        Returns:
            shape = (3그룹, 격자수)인 가중치 행렬. 행 합계는 1이다.

        Logic:
            - 터빈별 IDW:  w_ij = 1/d_ij^power  / Σ(1/d_ij^power), 가까운 k개 격자만
            - 그룹별 집계: W_gj = Σ_i∈g (cap_i × w_ij) / Σ_i∈g cap_i
            - 격자를 통째로 평균하면 세 그룹에 똑같은 예보값이 들어가 거리·고도·바람장 차이가 사라진다.
        """
        meta = turbine_meta.reset_index(drop=True).copy()
        # 원본 DataFrame 변형을 막고 아래 행렬 인덱스와 맞게 행 번호를 다시 매긴다

        missing = [c for c in ("group", "lat", "lon") if c not in meta.columns]
        if "cap_kw" not in meta.columns:
            if "capacity_mw" in meta.columns:
                meta["cap_kw"] = meta["capacity_mw"] * 1000
                # MW 단위 설비용량을 IDW 그룹 가중에 쓰는 kW 단위로 환산한다
            else:
                missing.append("cap_kw")

        if missing:
            raise KeyError(f"turbine_meta에 필요한 컬럼이 없습니다: {missing}")
        if not {"latitude", "longitude"}.issubset(grid_coords.columns):
            raise KeyError("grid_coords에는 latitude와 longitude가 필요합니다")
        if grid_coords.empty:
            raise PreprocessingError("격자 좌표가 비어 있습니다")

        meta["group"] = meta["group"].astype(str)
        # 문자열 키는 보존하고 숫자형 그룹도 안전하게 판별할 수 있도록 표현을 통일한다
        numeric_group = pd.to_numeric(meta["group"], errors="coerce")
        # 정수·실수·숫자 문자열로 들어온 그룹 번호를 숫자로 변환한다
        integer_group = numeric_group.notna() & numeric_group.mod(1).eq(0)
        # 1.5 같은 비정수 값은 그룹 번호로 묵시 변환하지 않고 아래 unknown 검사에서 막는다
        meta.loc[integer_group, "group"] = (
            "kpx_group_" + numeric_group.loc[integer_group].astype(int).astype(str)
        )
        # 정수 그룹 번호를 KPX_GROUPS와 같은 문자열 키로 자동 승격한다

        unknown = sorted(set(meta["group"]) - set(KPX_GROUPS))
        if unknown:
            raise ValueError(f"turbine_meta의 group 값이 KPX_GROUPS에 없습니다: {unknown}")
            # 오타·지원하지 않는 그룹을 조용히 0 가중치로 만들지 않고 즉시 실패시킨다

        grid_lat = grid_coords["latitude"].to_numpy(dtype=float)
        grid_lon = grid_coords["longitude"].to_numpy(dtype=float)
        distance = np.zeros((len(meta), len(grid_coords)))
        for i in range(len(meta)):
            distance[i] = SpatialUtils.compute_haversine_distance(
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

        row_sums = weight_group.sum(axis=1)
        # 각 그룹 가중치 행은 격자 전체에 대해 합이 1이어야 한다
        active_rows = np.array([group in set(meta["group"]) for group in KPX_GROUPS])
        # 부분 그룹 호출도 유지하면서 실제 메타에 존재하는 그룹 행만 정규화 여부를 검사한다
        if (
            not np.isfinite(weight_group).all()
            or not active_rows.any()
            or not np.allclose(row_sums[active_rows], 1.0)
        ):
            raise PreprocessingError(
                f"IDW 가중치가 유효하지 않습니다. turbine_meta의 group 키와 cap_kw를 확인하세요: "
                f"행 합계={row_sums.tolist()}"
            )
            # 그룹 키 불일치·0 설비용량·비유한값이 결과 피처를 0 행렬로 만드는 것을 차단한다

        return weight_group

    @staticmethod
    def transform_to_group_feature(
        df: pd.DataFrame,
        features: list[str],
        weight_group: np.ndarray,
        grid_index: pd.Index,
        time_col: str = "forecast_kst_dtm",
        target_groups: list[str] | None = None,
    ) -> pd.DataFrame:
        """격자별 값을 그룹별 공간 가중평균으로 변환한다.

        Args:
            df:           한 행 = 한 격자 x 한 시각인 예보 표. ``grid_id`` 컬럼 필수.
            features:     가중평균할 컬럼 목록 (격자별로 계산을 마친 파생변수).
                          풍향처럼 각도 값은 여기에 넣지 않고 U·V를 평균한 뒤 방향을 복원한다.
            weight_group: compute_group_weight의 결과. shape = (3그룹, 격자수).
            grid_index:   가중치 행렬의 열 순서를 정의하는 grid_id 인덱스.
            time_col:     예보 대상 시각 컬럼명.
            target_groups: 반환할 그룹 목록. None이면 KPX_GROUPS 전체를 반환.

        Returns:
            한 행 = 한 그룹 x 한 시각인 표.
            (시각 × 격자) @ (격자 × 3그룹) 행렬곱 한 번으로 전체 기간을 처리한다.
        """
        missing = [c for c in features if c not in df.columns]
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


# 기존 호출부(Preprocessing 서브패키지 등)가 자유 함수로 import하던 이름을 그대로 유지한다.
# `from .spatial_utils import compute_haversine_distance` 형태의 기존 import가 깨지지 않도록
# 클래스의 staticmethod를 모듈 레벨 이름에 그대로 연결한다.
compute_haversine_distance = SpatialUtils.compute_haversine_distance
compute_group_weight = SpatialUtils.compute_group_weight
transform_to_group_feature = SpatialUtils.transform_to_group_feature


__all__ = [
    "KPX_GROUPS", "EARTH_RADIUS_KM", "IDW_K", "IDW_POWER", "MIN_DISTANCE_KM",
    "PreprocessingError", "SpatialUtils",
    "compute_group_weight", "compute_haversine_distance",
    "transform_to_group_feature",
]
