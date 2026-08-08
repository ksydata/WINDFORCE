"""그룹별 기상 피처와 라벨을 결합하는 데이터셋 빌더."""

import pandas as pd


class WindforceDatasetBuilder:
    """LDAPS/GFS 그룹 집계 결과를 KPX 그룹별 학습 테이블로 변환한다."""

    def __init__(self, ldaps_group_df: pd.DataFrame, gfs_group_df: pd.DataFrame,
                 labels_df: pd.DataFrame):
        self.ldaps_group_df = ldaps_group_df
        self.gfs_group_df = gfs_group_df
        self.labels_df = labels_df

    @staticmethod
    def featureCols(df: pd.DataFrame) -> list[str]:
        exclude = {"kst_dtm", "forecast_kst_dtm"} | {f"kpx_group_{i}" for i in range(1, 4)}
        return [column for column in df.columns if column not in exclude]

    def build(self, group: int) -> pd.DataFrame:
        group_key = f"kpx_group_{group}"
        ldaps_group = self.ldaps_group_df[self.ldaps_group_df["group"] == group_key].drop(
            columns="group", errors="ignore"
        )
        gfs_group = self.gfs_group_df[self.gfs_group_df["group"] == group_key].drop(
            columns="group", errors="ignore"
        )
        merged = pd.merge(
            ldaps_group, gfs_group, on="forecast_kst_dtm", how="inner",
            suffixes=("_ldaps", "_gfs"),
        )
        labels = self.labels_df[["kst_dtm", group_key]].copy()
        labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
        merged["forecast_kst_dtm"] = pd.to_datetime(merged["forecast_kst_dtm"])
        result = pd.merge(
            merged, labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="left"
        ).drop(columns="kst_dtm", errors="ignore")
        result = result.dropna(subset=[group_key]).sort_values("forecast_kst_dtm").reset_index(drop=True)
        numeric = result.select_dtypes(include="number").columns.tolist()
        return result[numeric + ["forecast_kst_dtm"]].copy()
