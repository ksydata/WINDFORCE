"""WINDFORCE의 공개 API

내부 파일 구조가 바뀌어도 `from Windforce import X` 한 줄은 안 깨지게 한다.

    BasePreprocessor (추상)         공통 물리계산 + 8단계 preprocess 템플릿 + LSTM 시퀀스 생성
    ├── LDAPSFeatureEngineer        16격자 예보. 파생 -> IDW 그룹 가중 -> 격자 통계
    ├── GFSFeatureEngineer          9격자 예보. 80m·100m 바람 -> 허브 높이 외삽
    └── SCADAFeatureEngineer        10분 실측. wide->long -> 품질 플래그 -> 시간 집계

    PreprocessorFactory             source·group 문자열로 위 셋을 생성

``ScoreLossFunction``은 torch가 있어야 동작한다. 전처리 모듈만 쓰는 환경에서 torch
미설치로 `import Windforce` 전체가 죽지 않도록, 실제로 접근하는 시점에만 불러온다
(모듈 수준 ``__getattr__``, PEP 562).
"""

from .WindforceDataLoader import WindforceDataLoader
from .FeatureEngineer import (
    BasePreprocessor,
    FeatureEngineer,
    GroupSpec,
    PreprocessingError,
    KPX_GROUPS,
    RATED_CAPACITY_KW as GROUP_RATED_CAPACITY_KW,
)
from .utils.time_utils import SEASON_MAP, SEQ_LEN
# FeatureEngineer의 RATED_CAPACITY_KW는 키가 "kpx_group_1" 문자열이라
# EvaluationMetrics의 동명 상수(키가 정수 1/2/3)와 이름이 겹친다. 섞어 쓰면
# KeyError가 나므로 이름을 분리해 둘 다 안전하게 export한다
from .LDAPSFeatureEngineer import LDAPSFeatureEngineer
from .GFSFeatureEngineer import GFSFeatureEngineer
from .SCADAFeatureEngineer import SCADAFeatureEngineer
from .PreprocessorFactory import PreprocessorFactory, FeatureEngineerFactory, create_preprocessor
from .EvaluationMetrics import EvaluationMetrics, RATED_CAPACITY_KW, TIME_STEP_HOURS
# 기존 BASELINE 노트북이 `from Windforce import RATED_CAPACITY_KW`로 쓰는 것은
# 정수 키 버전이라 이름을 그대로 유지한다 (하위 호환)


def __getattr__(name: str):
    """torch가 필요한 ScoreLossFunction만 실제 접근 시점에 불러오는 지연 로딩 훅

    Logic:
        - 여기서 무조건 import하면 전처리 모듈만 쓰려는 환경(torch 미설치)에서도
          `import Windforce` 자체가 ModuleNotFoundError로 죽는다
        - PEP 562 모듈 수준 __getattr__은 `from X import Y`의 hasattr 검사에도
          관여하므로, 서브모듈 폴백이 클래스 대신 동명 서브모듈을 바인딩하는
          충돌 없이 정확히 ScoreLossFunction 클래스를 반환한다
    """
    if name == "ScoreLossFunction":
        from .ScoreLossFunction import ScoreLossFunction as _ScoreLossFunction
        return _ScoreLossFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WindforceDataLoader",
    # --- 전처리 계약 ---
    "BasePreprocessor",
    "FeatureEngineer",
    "GroupSpec",
    "PreprocessingError",
    "PreprocessorFactory",
    "FeatureEngineerFactory",
    "create_preprocessor",
    # --- 소스별 전처리기 ---
    "LDAPSFeatureEngineer",
    "GFSFeatureEngineer",
    "SCADAFeatureEngineer",
    # --- 전처리 공통 상수 ---
    "KPX_GROUPS",
    "GROUP_RATED_CAPACITY_KW",
    "SEASON_MAP",
    "SEQ_LEN",
    # --- 평가·손실 (ScoreLossFunction만 torch 필요) ---
    "EvaluationMetrics",
    "RATED_CAPACITY_KW",
    "TIME_STEP_HOURS",
    "ScoreLossFunction",
]
