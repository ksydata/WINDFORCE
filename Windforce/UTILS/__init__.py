"""WINDFORCE 공통 유틸리티 패키지."""

from .spatial_utils import (
    KPX_GROUPS,
    EARTH_RADIUS_KM,
    IDW_K,
    IDW_POWER,
    MIN_DISTANCE_KM,
    PreprocessingError,
    SpatialUtils,
    compute_group_weight,
    compute_haversine_distance,
    transform_to_group_feature,
)
from .time_utils import (
    MAX_INTERP_STEPS,
    SEASON_MAP,
    SEQ_LEN,
    TimeUtils,
    compute_feature_columns,
    interpolate_short_gap,
    select_group,
    transform_categorical,
    transform_cyclic_time,
    transform_forecast_diff,
    transform_group_time_feature,
    transform_scale,
    transform_to_model_input,
    transform_to_sequence,
)
from .wind_utils import (
    R_D,
    EPSILON,
    ONE_MINUS_EPS,
    WindUtils,
    calculate_air_density,
    calculate_wind_power_density,
    compute_angle_difference,
    compute_uv_component,
    compute_wind_direction,
    compute_wind_shear,
    compute_wind_speed,
    transform_wind_direction,
)

__all__ = [
    # spatial
    "KPX_GROUPS", "EARTH_RADIUS_KM", "IDW_K", "IDW_POWER", "MIN_DISTANCE_KM",
    "PreprocessingError", "SpatialUtils", "compute_group_weight", "compute_haversine_distance",
    "transform_to_group_feature",
    # time
    "MAX_INTERP_STEPS", "SEASON_MAP", "SEQ_LEN", "TimeUtils", "compute_feature_columns",
    "interpolate_short_gap", "select_group", "transform_categorical",
    "transform_cyclic_time", "transform_forecast_diff", "transform_group_time_feature",
    "transform_scale", "transform_to_model_input", "transform_to_sequence",
    # wind
    "R_D", "EPSILON", "ONE_MINUS_EPS", "WindUtils", "calculate_air_density",
    "calculate_wind_power_density", "compute_angle_difference",
    "compute_uv_component", "compute_wind_direction", "compute_wind_shear",
    "compute_wind_speed", "transform_wind_direction",
]
