"""그룹별 기상 피처와 라벨을 결합하는 데이터셋 빌더."""

import pandas as pd


class WindforceDatasetBuilder:
    """LDAPS/GFS 그룹 집계 결과를 KPX 그룹별 학습 테이블로 변환한다.

    Attributes:
        - ldaps_group_df: LDAPSFeatureEngineer.transformToGroupIDW()의 결과
          (한 행 = 한 그룹 x 한 시각인 공간가중 예보표).
        - gfs_group_df: GFSFeatureEngineer.transformToGroupIDW()의 결과. ldaps와 동일한 구조.
        - labels_df: WindforceDataLoader["train_labels"]. kst_dtm 기준 그룹별 실제 발전량.
    """

    def __init__(self, ldaps_group_df: pd.DataFrame, gfs_group_df: pd.DataFrame,
                 labels_df: pd.DataFrame):
        self.ldaps_group_df = ldaps_group_df
        self.gfs_group_df = gfs_group_df
        self.labels_df = labels_df
        # 세 표 모두 build() 호출 시점에 그룹 필터링·병합에 쓰이므로 그대로 보관만 해둔다

    @staticmethod
    def featureCols(df: pd.DataFrame) -> list[str]:
        """build()가 반환한 표에서 타깃·시각 식별자를 제외한 순수 피처 컬럼만 골라내는 정적 메서드

        Args:
            - df: build(group)의 반환값 (또는 그와 같은 컬럼 구조를 가진 표)

        Logic:
            - kst_dtm / forecast_kst_dtm: 시각 식별자이므로 모델 입력 피처로 쓰면 안 된다
            - kpx_group_1/2/3: 현재 그룹의 타깃뿐 아니라 다른 그룹의 타깃까지 전부 제외해야
              그룹 간 정보 누수(다른 그룹 발전량을 피처로 쓰는 실수)를 막을 수 있다
        """
        exclude = {"kst_dtm", "forecast_kst_dtm"} | {f"kpx_group_{i}" for i in range(1, 4)}
        # 시각 식별자 2개와 그룹 타깃 3개를 합집합으로 묶어 제외 대상 집합을 만든다
        return [column for column in df.columns if column not in exclude]
        # 제외 대상에 없는 컬럼만 원래 순서 그대로 남겨 반환 (컬럼 순서를 흔들지 않아야
        # baseline_5의 Step7 스키마 정렬 로직이 학습·평가 간 동일한 순서를 기대할 수 있다)

    def build(self, group: int) -> pd.DataFrame:
        """특정 KPX 그룹의 LDAPS·GFS 피처와 실제 발전량을 병합해 학습용 표 하나로 만드는 메서드

        Args:
            - group: KPX 그룹 번호 (1/2/3 중 하나)

        Returns:
            - forecast_kst_dtm(시각) + 숫자형 피처(LDAPS·GFS) + kpx_group_{group}(타깃)을
              모두 포함하는, 시간순으로 정렬된 DataFrame

        Logic:
            - LDAPS·GFS는 forecast_kst_dtm 기준 inner join으로 합친다. 두 소스 모두
              같은 시각을 커버해야만 그 시각의 학습 샘플을 만들 수 있기 때문이다
              (한쪽에만 있는 시각은 피처가 절반만 있는 셈이라 버리는 게 안전하다)
            - 라벨은 kst_dtm과 forecast_kst_dtm을 매칭해 left join으로 붙인다.
              평가 기간처럼 라벨이 아예 없는 구간에서도 피처 자체는 살아있어야 하므로
              (Step7에서 재사용) 여기서는 일단 left join으로 붙여둔다
            - 다만 build()의 최종 반환에서는 dropna(subset=[group_key])로 라벨이 없는
              행을 제거한다 — 이 메서드 자체는 "학습 테이블"이 목적이라, 라벨 없는
              평가 구간 피처는 Step7에서 별도 로직으로 다시 만든다
        """
        group_key = f"kpx_group_{group}"
        # 이후 컬럼 필터링·병합·라벨 매칭에서 반복적으로 쓰이는 타깃 컬럼명을 한 번만 계산

        ldaps_group = self.ldaps_group_df[self.ldaps_group_df["group"] == group_key].drop(
            columns="group", errors="ignore"
        )
        # 이 그룹에 해당하는 행만 필터링한 뒤, group 컬럼은 이제 상수라 더 필요 없으므로 제거
        # errors="ignore": group 컬럼이 이미 없는 입력이 들어와도 에러 없이 통과
        gfs_group = self.gfs_group_df[self.gfs_group_df["group"] == group_key].drop(
            columns="group", errors="ignore"
        )
        # GFS도 동일한 필터링·컬럼 제거 로직을 적용

        merged = pd.merge(
            ldaps_group, gfs_group, on="forecast_kst_dtm", how="inner",
            suffixes=("_ldaps", "_gfs"),
        )
        # inner join: 두 소스에 공통으로 존재하는 시각만 남긴다
        # suffixes: 두 표에 같은 이름의 파생 컬럼(예: ws10_2)이 있으면 출처를 구분할 수 있게 접미사를 붙인다

        labels = self.labels_df[["kst_dtm", group_key]].copy()
        # 라벨 표에서 이 그룹의 타깃 컬럼만 필요하므로 미리 좁혀서 이후 병합 비용을 줄인다
        labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
        merged["forecast_kst_dtm"] = pd.to_datetime(merged["forecast_kst_dtm"])
        # 두 표의 시각 컬럼을 datetime으로 명시 변환해야 merge의 키 매칭이 정확하게 이뤄진다
        # (문자열로 남아있으면 포맷 차이로 매칭이 조용히 실패할 수 있다)

        result = pd.merge(
            merged, labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="left"
        ).drop(columns="kst_dtm", errors="ignore")
        # left join: 피처(merged) 쪽을 기준으로 라벨을 붙인다. 라벨이 없는 시각도 일단 살려둔 채
        # 붙이고, kst_dtm은 forecast_kst_dtm과 값이 같아 중복이므로 병합 후 바로 제거한다

        result = result.dropna(subset=[group_key]).sort_values("forecast_kst_dtm").reset_index(drop=True)
        # 실제 발전량(타깃)이 없는 행은 학습에 쓸 수 없으므로 여기서 제거
        # 이후 시계열 분할(train/test)이 시간순을 가정하므로 정렬 + 인덱스 재설정까지 함께 수행

        numeric = result.select_dtypes(include="number").columns.tolist()
        # 이 시점에 남아있을 수 있는 비숫자형 컬럼(예: 문자열 메타)을 걸러내기 위해 숫자형만 추출
        return result[numeric + ["forecast_kst_dtm"]].copy()
        # 숫자형 피처·타깃 전부 + 시각 식별자(forecast_kst_dtm)만 포함해 반환
        # forecast_kst_dtm은 featureCols()에서 제외 대상이라 모델 입력에는 안 쓰이지만,
        # 이후 시간순 분할·디버깅·병합 시 시각을 참조할 수 있도록 표에는 남겨둔다