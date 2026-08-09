"""소스 이름으로 구체 전처리기를 생성하는 팩토리."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from .FeatureEngineer import BasePreprocessor
from .GFSFeatureEngineer import GFSFeatureEngineer
from .LDAPSFeatureEngineer import LDAPSFeatureEngineer
from .SCADAFeatureEngineer import SCADAFeatureEngineer
from ..utils.spatial_utils import KPX_GROUPS


class PreprocessorFactory:
    """소스 이름으로 구체 전처리기를 생성하는 팩토리 클래스.

    고정된 ``_REGISTRY``와 ``_ALIASES`` 딕셔너리로 소스 이름 또는 별칭을
    구체 클래스로 연결하기 때문에, 사용자는 클래스 이름을 기억하지 않아도 된다.

    Example::

        fe = PreprocessorFactory.create("ldaps", group="kpx_group_1")
        group_set = PreprocessorFactory.createGroupSet("gfs")
    """

    _REGISTRY: ClassVar[dict[str, type[BasePreprocessor]]] = {
        "gfs": GFSFeatureEngineer,
        "ldaps": LDAPSFeatureEngineer,
        "scada": SCADAFeatureEngineer,
    }
    _ALIASES: ClassVar[dict[str, str]] = {
        "gfsfeatureenginner": "gfs",
        "ldapsfeatureenginner": "ldaps",
        "gfsfeatureengineer": "gfs",
        "ldapsfeatureengineer": "ldaps",
        "scadafeatureengineer": "scada",
    }

    @classmethod
    def create(
        cls,
        source: Literal["gfs", "ldaps", "scada"] | str,
        group: str | None = None,
        **kwargs: Any,
    ) -> BasePreprocessor:
        """source 이름(또는 별칭)으로 구체 전처리기 인스턴스를 생성한다.

        Args:
            source: "gfs", "ldaps", "scada" 중 하나. 클래스 전체 이름 길이 문자열도 대소문자 무시하고 적용한다.
            group:  KPX 그룹 이름. None이면 전체 그룹을 처리하는 전처리기를 반환한다.
            kwargs: 각 전처리기 __init__에 추가로 넣어줄 인자.

        Returns:
            요청한 소스에 맞는 BasePreprocessor 하위 클래스 인스턴스.

        Raises:
            ValueError: 등록되지 않은 source 이름이면.
        """
        key = str(source).strip().lower()
        key = cls._ALIASES.get(key, key)
        try:
            preprocessor = cls._REGISTRY[key]
        except KeyError as exc:
            raise ValueError(
                f"지원하지 않는 전처리 방식입니다: {source}. "
                f"등록된 소스 = {sorted(cls._REGISTRY)}"
            ) from exc
        return preprocessor(group = group, **kwargs)

    @classmethod
    def createGroupSet(
        cls, source: Literal["gfs", "ldaps", "scada"] | str, **kwargs: Any
    ) -> dict[str, BasePreprocessor]:
        """세 그룹에 대한 전처리기를 한번에 생성한다.

        Args:
            source: "gfs", "ldaps", "scada" 중 하나.
            kwargs: create에 추가로 넣어줄 인자.

        Returns:
            {"kpx_group_1": ..., "kpx_group_2": ..., "kpx_group_3": ...} 딕셔너리.
        """
        return {group: cls.create(source, group=group, **kwargs) for group in KPX_GROUPS}


def create_preprocessor(name: str, **kwargs: Any) -> BasePreprocessor:
    """소스 이름으로 전처리기를 생성하는 모듈 수준 단쳐 함수.

    PreprocessorFactory.create의 단순 래퍼이다.
    
    Args:
        name:   소스 이름 ("gfs" / "ldaps" / "scada").
        kwargs: create에 추가로 넣어줄 인자.

    Returns:
        요청한 소스에 맞는 BasePreprocessor 하위 클래스 인스턴스.
    """
    return PreprocessorFactory.create(name, **kwargs)


FeatureEngineerFactory = PreprocessorFactory


__all__ = ["PreprocessorFactory", "FeatureEngineerFactory", "create_preprocessor"]
