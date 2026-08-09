"""모델 버전 x 피처 버전 조합을 전부 순회하는 실험 매트릭스 러너.

원본 GroupExperimentRunner(baseline_5)가 Persistence/SVR/LSTM 3개 모델을
그룹별로 한 번씩 돌렸다면, 이 클래스는 그룹 x 피처버전 x 모델버전의
3중 조합을 전부 순회해 baseline_6의 A/B 실험(어떤 피처 압축·어떤 LSTM
표현 전략이 더 나은가)을 한 번에 수행한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..EvaluationMetrics import RATED_CAPACITY_KW
from .FeatureVariants import make_feature_variant
from .LSTMPipelineVersions import make_lstm_pipeline


class VersionedExperimentRunner:
    """그룹 x 피처버전 x 모델버전 전체 조합을 학습·평가하는 클래스

    사용 순서:
        runner = VersionedExperimentRunner(datasetBuilder, metrics)
        result_df = runner.run_all()
        summary_df = runner.summary(result_df)
    """

    def __init__(self, dataset_builder, metrics, groups=(1, 2, 3), test_ratio=.2,
                 seq_len=24, model_versions=("v1", "v2", "v3"),
                 feature_versions=("all", "corr_filtered", "pca", "tree_importance"),
                 model_params=None, feature_params=None, random_state=42):
        """
        Args:
            - dataset_builder: WindforceDatasetBuilder처럼 build(group)을 호출하면
              해당 그룹의 피처+타깃 DataFrame을 반환하는 객체.
            - metrics: EvaluationMetrics 인스턴스. summarize(pred, actual, group)을 지원.
            - groups: 순회할 KPX 그룹 번호 목록.
            - test_ratio: 뒤쪽 몇 %를 검증(테스트)용으로 뗄지.
            - seq_len: LSTM 슬라이딩 윈도우 길이.
            - model_versions: 순회할 LSTM 버전 문자열 목록 ("v1"/"v2"/"v3").
            - feature_versions: 순회할 피처 변형 이름 목록.
            - model_params: 버전 이름과 무관하게 모든 LSTM 파이프라인에 공통 적용할 하이퍼파라미터.
            - feature_params: {피처버전이름: {해당 변형 전용 kwargs}} 형태의 딕셔너리.
            - random_state: 피처 변형·LSTM 초기화에 공통으로 쓰는 시드 (재현성 보장).
        """
        self.dataset_builder, self.metrics = dataset_builder, metrics
        self.groups, self.test_ratio, self.seq_len = list(groups), test_ratio, seq_len
        # groups를 list()로 감싸 튜플로 들어와도 이후 순회·재사용에 문제가 없도록 함
        self.model_versions, self.feature_versions = list(model_versions), list(feature_versions)
        self.model_params, self.feature_params = model_params or {}, feature_params or {}
        # None으로 들어오면 빈 딕셔너리로 대체해 이후 .get() 호출이 안전하게 동작하도록 함
        self.random_state, self.results = random_state, []
        # results: run_all() 실행 후 {group, model_version, feature_version, ...} 딕셔너리들이 누적됨

    @staticmethod
    def _feature_cols(df, target):
        """타깃 및 식별자 컬럼을 제외한 순수 피처 컬럼 목록을 반환하는 내부 헬퍼 메서드

        Args:
            - df: 그룹별 피처+타깃이 포함된 DataFrame
            - target: 제외할 타깃 컬럼명 (예: "kpx_group_1")

        Logic:
            - kst_dtm / forecast_kst_dtm: 시각 식별자 → 피처에 포함하면 미래 정보 누수 위험
            - kpx_group_*: 다른 그룹 타깃까지 전부 제외 (그룹 간 누수 방지)
            - GroupExperimentRunner의 _featureCols와 동일한 규칙이나, target 인자를
              명시적으로 한 번 더 걸러내 다른 그룹 타깃 컬럼명과 겹치는 경우까지 방어한다
        """
        exclude = {"kst_dtm", "forecast_kst_dtm"} | {f"kpx_group_{i}" for i in range(1, 4)}
        return [c for c in df.columns if c not in exclude and c != target]

    def _split(self, df, group):
        """시간순 분할만 담당하는 내부 헬퍼 메서드 (스케일링은 피처 변형 쪽에서 처리)

        Args:
            - df: 특정 그룹의 전체 피처+타깃 데이터프레임 (아직 분할 전)
            - group: KPX 그룹 번호

        Returns:
            - (X_train, X_test, y_train, y_test) 튜플. X는 DataFrame, y는 numpy 배열

        Logic:
            - 시계열 데이터이므로 랜덤 셔플 없이 앞쪽 (1-test_ratio)를 train,
              뒤쪽 test_ratio를 test로 나눈다 (미래 정보가 학습에 섞이는 leakage 방지)
            - 원본 GroupExperimentRunner와 달리 스케일링은 여기서 하지 않는다.
              각 피처 변형(FeatureVariant)이 자체적으로 fit_transform/transform을
              수행하므로, 스케일링 이전 단계인 "원본 피처 분할"까지만 책임진다
        """
        target = f"kpx_group_{group}"
        cols = self._feature_cols(df, target)
        n = int(len(df) * (1 - self.test_ratio))
        # 정수 인덱스로 자를 분할 지점. 소수점은 버림(int 변환)으로 처리
        return df[cols].iloc[:n].copy(), df[cols].iloc[n:].copy(), df[target].iloc[:n].to_numpy(), df[target].iloc[n:].to_numpy()
        # .copy(): 이후 변형기(imputer 등)가 원본 df를 의도치 않게 변경하지 않도록 방어

    def run_all(self) -> pd.DataFrame:
        """그룹 x 피처버전 x 모델버전 전체 조합을 학습·평가하고 결과를 DataFrame으로 반환하는 메서드

        Returns:
            - 조합별 {group, model_version, feature_version, n_features, best_epoch,
              nmae, ficr, score} 행을 담은 DataFrame

        Logic:
            - self.results를 매번 초기화한 뒤 다시 채운다 (재실행 시 중복 누적 방지)
            - 같은 (그룹, 피처버전) 조합은 모델 버전이 바뀌어도 피처 변형을 재사용하지 않고
              루프 구조상 그룹 -> 피처버전 -> 모델버전 순으로 진행하되, 피처 변형은
              (그룹, 피처버전) 조합마다 한 번만 fit해 모델버전 루프에서 재사용한다
        """
        self.results = []
        # 매 실행마다 결과 리스트를 새로 초기화 (누적 방지)

        for group in self.groups:
            df = self.dataset_builder.build(group)
            # 이 그룹 전용 피처+타깃 데이터프레임을 한 번만 생성해 아래 두 루프에서 재사용
            Xtr, Xte, ytr, yte = self._split(df, group)
            # 시간순 train/test 분할 (피처 변형·모델 버전과 무관하게 그룹당 한 번만 수행)

            for fv in self.feature_versions:
                variant = make_feature_variant(fv, **self.feature_params.get(fv, {}), random_state=self.random_state)
                # 이 피처버전 전용 kwargs를 self.feature_params에서 조회 (없으면 빈 딕셔너리로 기본값 사용)
                A = variant.fit_transform(Xtr, ytr)
                # 학습 구간에만 fit해 통계량(결측치 대체값·상관도·주성분·트리 중요도 등)을 계산
                B = variant.transform(Xte)
                # 검증 구간은 fit 없이 transform만 — 학습 통계량을 그대로 적용해 leakage 방지
                scaler = MinMaxScaler().fit(A)
                A, B = scaler.transform(A), scaler.transform(B)
                # 피처 변형 이후의 최종 표현(원본/상관필터/PCA/트리선택)에 맞춰 스케일링도
                # 반드시 여기서, 그리고 학습 구간(A) 기준으로만 fit한다

                for mv in self.model_versions:
                    params = dict(self.model_params)
                    params.setdefault("random_state", self.random_state)
                    # self.model_params를 매 반복마다 복사해 사용 (원본 딕셔너리가 오염되지 않도록)
                    # random_state가 model_params에 명시돼 있지 않을 때만 러너의 공통 시드를 기본값으로 채움
                    pipe = make_lstm_pipeline(mv, capacity_kw=RATED_CAPACITY_KW[group], seq_len=self.seq_len, **params)
                    # 그룹별 설비용량으로 이 (그룹, 피처버전, 모델버전) 조합 전용 파이프라인 생성
                    pipe.fit(A, ytr, X_val=B, y_val=yte)
                    pred = pipe.predict(B, yte)
                    actual = yte[self.seq_len:]
                    # SequenceDataset이 앞쪽 seq_len개 시점을 버리므로 정답도 동일하게 잘라 인덱스를 맞춤

                    if len(pred) != len(actual):
                        continue
                        # 조기종료 등으로 길이가 어긋나는 예외적인 경우, 이 조합만 건너뛰고 나머지는 계속 진행
                        # (한 조합의 실패가 전체 실험 매트릭스를 중단시키지 않도록 방어)

                    summary = self.metrics.summarize(pred, actual, group)
                    self.results.append({"group": group, "model_version": mv,
                                         "feature_version": fv, "n_features": A.shape[1],
                                         "best_epoch": pipe.best_epoch,
                                         **{k: v for k, v in summary.items() if k != "group"}})
                    # group 키는 이미 위에서 추가했으므로 summarize 반환값에서는 제외하고 병합
                    # n_features: 이 피처버전이 최종적으로 몇 개의 컬럼을 만들었는지 (PCA/트리선택 비교용)
                    # best_epoch: 조기종료가 어느 epoch에서 일어났는지 (버전별 수렴 속도 비교용)

        return pd.DataFrame(self.results)
        # 전체 결과를 DataFrame으로 변환해 반환 (groupby·pivot 등 후처리하기 편하도록)

    def summary(self, result_df=None) -> pd.DataFrame:
        """(모델버전, 피처버전) 조합별로 그룹 평균 성능을 집계하는 메서드

        Args:
            - result_df: run_all()의 반환값. None이면 self.results로부터 새로 만든다

        Returns:
            - (model_version, feature_version)별 평균 nmae/ficr/score와
              집계에 사용된 그룹 수·평균 피처 수를 담은 DataFrame,
              mean_score 내림차순으로 정렬됨 (가장 좋은 조합이 맨 위)
        """
        result_df = result_df if result_df is not None else pd.DataFrame(self.results)
        # 인자로 안 주면 self.results(run_all의 마지막 실행 결과)를 재사용
        if result_df.empty:
            return pd.DataFrame()
            # run_all()을 아직 안 돌렸거나 모든 조합이 스킵된 경우, 빈 표를 그대로 반환 (에러 대신 방어적 처리)

        out = result_df.groupby(["model_version", "feature_version"], as_index=False).agg(
            mean_nmae=("nmae", "mean"), mean_ficr=("ficr", "mean"), mean_score=("score", "mean"),
            groups=("group", "nunique"), mean_features=("n_features", "mean"),
        )
        # groups=nunique: 이 조합이 몇 개 그룹에서 성공적으로 집계됐는지 (일부 그룹만 스킵된 경우를 드러냄)
        # mean_features: 피처버전별로 실제 사용된 평균 컬럼 수 (PCA·트리선택 압축 효과 확인용)
        return out.sort_values("mean_score", ascending=False).reset_index(drop=True)
        # 총점(mean_score) 내림차순 정렬로 가장 좋은 (모델버전, 피처버전) 조합이 위로 오게 함


__all__ = ["VersionedExperimentRunner"]