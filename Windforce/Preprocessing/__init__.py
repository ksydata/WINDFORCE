"""WINDFORCE의 공개 전처리 API

내부 파일 구조가 바뀌어도 `from Windforce import X` 한 줄은 안 깨지게 한다.

    FeatureEngineer (추상)          공통 물리계산 + 8단계 파이프라인 + LSTM 시퀀스 생성
    ├── LDAPSFeatureEngineer        16격자 예보. 파생 -> IDW 그룹 가중 -> 격자 통계
    ├── GFSFeatureEngineer          9격자 예보. 80m·100m 바람 -> 허브 높이 외삽
    └── SCADAFeatureEngineer        10분 실측. wide->long -> 품질 플래그 -> 시간 집계

    FeatureEngineerFactory          source·group 문자열로 위 셋을 생성

전처리의 최종 산출물은 LSTM이 바로 받는 (samples, timesteps, features) 3차원 배열이다.

    fe = FeatureEngineerFactory.create("ldaps")
    grid = fe.transform(ldaps_raw)
    group = fe.transformToGroupIDW(grid, turbine_meta)
    feat = fe.transformForecastFeature(group)
    model_input = fe.transformToModelInput(feat)
    x, y, index = fe.transformToSequence(model_input, feature_cols, target_col = "scada_power_kwh")
"""

from ..FeatureEngineer import (
    EARTH_RADIUS_KM,
    EPSILON,
    FeatureEngineer,
    GroupSpec,
    IDW_K,
    IDW_POWER,
    KPX_GROUPS,
    MAX_INTERP_STEPS,
    MIN_SAMPLES_PER_HOUR,
    ONE_MINUS_EPS,
    RATED_CAPACITY_KW,
    RHO_STD,
    R_D,
    SEASON_MAP,
    SEQ_LEN,
)
from .FeatureEngineerFactory import FeatureEngineerFactory
from .GFSFeatureEngineer import (
    GFS_GRID_COUNT,
    GFSFeatureEngineer,
    GUST_RATIO_MAX,
    HUB_HEIGHT_M,
    WIND_LEVELS,
)
from .LDAPSFeatureEngineer import (
    LDAPS_GRID_COUNT,
    LDAPSFeatureEngineer,
    RH_PHYS_MAX,
    WS_SPACE_JUMP_MS,
    WS_TIME_JUMP_MS,
)
from .SCADAFeatureEngineer import (
    CUT_IN_MS,
    CUT_OUT_MS,
    DISAGREE_RATIO,
    FROZEN_STEPS,
    HIGH_WS_MS,
    LOW_WS_MASK,
    MIN_SAMPLES_RELAXED,
    NEG_POWER_RATIO,
    POWER_CAP_RATIO,
    SCADAFeatureEngineer,
    SWEPT_AREA_M2,
    WS_PHYS_MAX,
)

__all__ = [
    # --- 전처리 계약 ---
    "FeatureEngineer",
    "FeatureEngineerFactory",
    "GroupSpec",
    # --- 소스별 전처리기 ---
    "LDAPSFeatureEngineer",
    "GFSFeatureEngineer",
    "SCADAFeatureEngineer",
    # --- 공통 상수 ---
    "KPX_GROUPS",
    "RATED_CAPACITY_KW",
    "R_D",
    "EPSILON",
    "ONE_MINUS_EPS",
    "RHO_STD",
    "MIN_SAMPLES_PER_HOUR",
    "MAX_INTERP_STEPS",
    "EARTH_RADIUS_KM",
    "IDW_K",
    "IDW_POWER",
    "SEASON_MAP",
    "SEQ_LEN",
    # --- LDAPS 상수 ---
    "LDAPS_GRID_COUNT",
    "WS_TIME_JUMP_MS",
    "WS_SPACE_JUMP_MS",
    "RH_PHYS_MAX",
    # --- GFS 상수 ---
    "GFS_GRID_COUNT",
    "HUB_HEIGHT_M",
    "WIND_LEVELS",
    "GUST_RATIO_MAX",
    # --- SCADA 상수 ---
    "POWER_CAP_RATIO",
    "NEG_POWER_RATIO",
    "FROZEN_STEPS",
    "CUT_IN_MS",
    "HIGH_WS_MS",
    "WS_PHYS_MAX",
    "DISAGREE_RATIO",
    "MIN_SAMPLES_RELAXED",
    "LOW_WS_MASK",
    "SWEPT_AREA_M2",
    "CUT_OUT_MS",
]