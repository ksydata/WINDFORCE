"""소스 이름으로 구체 전처리기를 생성하는 팩토리."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from .FeatureEngineer import BasePreprocessor
from .GFSFeatureEngineer import GFSFeatureEngineer
from .LDAPSFeatureEngineer import LDAPSFeatureEngineer
from .SCADAFeatureEngineer import SCADAFeatureEngineer
from .utils.spatial_utils import KPX_GROUPS


class PreprocessorFactory:
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
        return {group: cls.create(source, group = group, **kwargs) for group in KPX_GROUPS}


def create_preprocessor(name: str, **kwargs: Any) -> BasePreprocessor:
    return PreprocessorFactory.create(name, **kwargs)


FeatureEngineerFactory = PreprocessorFactory


__all__ = ["PreprocessorFactory", "FeatureEngineerFactory", "create_preprocessor"]
