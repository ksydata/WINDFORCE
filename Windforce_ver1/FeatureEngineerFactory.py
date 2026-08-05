# Step 8. 데이터 소스·KPX 그룹별 전처리기 생성 (팩토리)

"""데이터 소스와 KPX 그룹에 맞는 전처리기를 생성하는 것을 담당하는 모듈.

호출부가 ``LDAPSFeatureEngineer``인지 ``SCADAFeatureEngineer``인지 알 필요가 없게 만든다.
그룹별 모델 3개를 돌릴 때 ``source``와 ``group`` 문자열만 바꿔 같은 코드를 재사용하고,
GFS 같은 새 소스가 붙어도 ``register`` 한 줄로 끝나 호출부는 손대지 않는다.

``create``는 §2.1의 고정 동사 표에 없는 동사다. 계산·변환이 아니라 **객체 생성**이라는
다른 종류의 동작이라 팩토리 표준 어휘를 그대로 쓴다.
"""

from __future__ import annotations

from typing import Literal

from .FeatureEngineer import FeatureEngineer, KPX_GROUPS
from .GFSFeatureEngineer import GFSFeatureEngineer
from .LDAPSFeatureEngineer import LDAPSFeatureEngineer
from .SCADAFeatureEngineer import SCADAFeatureEngineer


class FeatureEngineerFactory:
    """데이터 소스와 KPX 그룹에 맞는 전처리기를 생성하는 팩토리 클래스

    Attributes:
        - _REGISTRY: 소스 문자열 -> FeatureEngineer 자식 클래스 매핑
    """

    _REGISTRY: dict[str, type[FeatureEngineer]] = {
        "ldaps": LDAPSFeatureEngineer,
        # LDAPS 예보 16격자. 1.5km 해상도지만 고도는 10m·50m뿐이다
        "gfs": GFSFeatureEngineer,
        # GFS 예보 9격자. 해상도는 낮지만 80m·100m 바람이 허브 높이에 가깝다
        "scada": SCADAFeatureEngineer,
        # SCADA 10분 실측. turbine_meta를 kwargs로 반드시 넘겨야 한다
    }

    @classmethod
    def create(
        cls,
        source: Literal["ldaps", "gfs", "scada"] | str,
        group: str | None = None,
        **kwargs: object,
    ) -> FeatureEngineer:
        """소스·그룹별 구체 전처리기를 생성하는 메서드

        Args:
            - source: 등록된 소스 문자열 (`ldaps` / `gfs` / `scada`). 대소문자를 가리지 않는다
            - group: `kpx_group_1` / `2` / `3`. None이면 전체 그룹
            - kwargs: 구현체별 옵션 (예: `turbine_meta`, `max_interpolation_steps`)

        Logic:
            - 소스 이름이 틀렸을 때 등록된 목록을 같이 보여줘야 오타를 바로 고칠 수 있다
        """
        source_key = str(source).lower()
        if source_key not in cls._REGISTRY:
            raise ValueError(
                f"지원하지 않는 source입니다: {source}. 등록된 소스 = {sorted(cls._REGISTRY)}"
            )
        return cls._REGISTRY[source_key](group = group, **kwargs)
        # FeatureEngineer 계약을 만족하는 구체 전처리기 인스턴스 반환

    @classmethod
    def createGroupSet(
        cls, source: Literal["ldaps", "gfs", "scada"] | str, **kwargs: object
    ) -> dict[str, FeatureEngineer]:
        """KPX 그룹 3개의 전처리기를 한번에 반환하는 헬퍼 메서드

        Args:
            - source: 등록된 소스 문자열
            - kwargs: 모든 그룹에 공통으로 넘길 구현체 옵션

        Logic:
            - 그룹별 모델은 같은 입력 계약을 쓰고 group만 달라야 데이터 누수가 줄어든다
            - 그룹마다 create를 세 번 쓰다가 하나를 빠뜨리는 실수를 막는 용도다
        """
        return {group: cls.create(source, group = group, **kwargs) for group in KPX_GROUPS}
        # {그룹 이름: 전처리기} 딕셔너리 반환

    @classmethod
    def register(cls, source: str, engineer: type[FeatureEngineer]) -> None:
        """새 데이터 소스 전처리기를 팩토리에 등록하는 메서드

        Args:
            - source: 소스 식별 문자열 (예: `gfs`)
            - engineer: FeatureEngineer를 상속한 구체 클래스

        Logic:
            - GFS를 추가할 때 이 메서드만 부르면 되고 create 호출부는 손대지 않는다
        """
        if not isinstance(engineer, type) or not issubclass(engineer, FeatureEngineer):
            raise TypeError("engineer는 FeatureEngineer의 하위 클래스여야 합니다")
        cls._REGISTRY[source.lower()] = engineer
        # 반환값 없음. 클래스 레지스트리에 소스를 추가하는 것이 목적이다

    @classmethod
    def checkSource(cls) -> dict[str, bool]:
        """현재 등록된 소스 목록을 확인하는 메서드

        Logic:
            - 전처리를 돌리기 전에 필요한 소스가 등록돼 있는지 먼저 확인한다
        """
        return {source: True for source in sorted(cls._REGISTRY)}
        # {소스 이름: 등록 여부} 딕셔너리 반환


__all__ = ["FeatureEngineerFactory"]
