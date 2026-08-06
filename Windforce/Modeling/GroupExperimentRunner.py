# GroupExperimentRunner: 그룹별(KPX_1/2/3) 전체 실험을 오케스트레이션하는 클래스

# 기존 runGroup 함수 + 실행부(for문, resultDf, finalScores 계산)를
# 하나의 상태 있는 객체로 통합한다.
# datasetBuilder, metrics, results를 self로 계속 들고 있으므로,
# 나중에 특정 그룹만 다시 실행하거나 중간 결과를 다시 꺼내보기 쉬워진다.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .BaselineModels import BaselineModels
from .LSTMPipeline import LSTMPipeline, SEQ_LEN
from ..EvaluationMetrics import EvaluationMetrics, RATED_CAPACITY_KW

GROUPS = [1, 2, 3]
# KPX 평가 그룹 번호 목록


def _featureCols(df: pd.DataFrame, target_col: str) -> list:
    """타깃 및 식별자 컬럼을 제외한 순수 피처 컬럼 목록을 반환하는 헬퍼 함수

    Args:
        - df: 그룹별 피처+타깃이 포함된 DataFrame
        - target_col: 제외할 타깃 컬럼명 (예: "kpx_group_1")

    Logic:
        - kst_dtm / forecast_kst_dtm : 시각 식별자 → 피처에 포함하면 미래 정보 누수 위험
        - kpx_group_*: 다른 그룹 타깃도 함께 존재하면 전부 제외 (그룹 간 누수 방지)
    """
    exclude = {"kst_dtm", "forecast_kst_dtm"} | {f"kpx_group_{i}" for i in range(1, 4)}
    return [c for c in df.columns if c not in exclude]


class GroupExperimentRunner:
    """그룹별(KPX_1/2/3) Persistence/SVR/LSTM 학습·평가를 총괄하는 클래스"""

    def __init__(self, datasetBuilder,
                 metrics: EvaluationMetrics, groups: list = GROUPS,
                 testRatio: float = 0.2, seqLen: int = SEQ_LEN):
        """
        Args:
            - datasetBuilder: Step4/6 연결부. build(group)을 호출하면
              해당 그룹 전용 피처+타깃 데이터프레임을 반환하는 객체.
            - metrics: Step5의 EvaluationMetrics 인스턴스.
              그룹 공용이며 내부 상태가 없으므로 그대로 재사용 가능.
            - groups: 순회할 KPX 그룹 번호 리스트 (기본값 [1,2,3]).
            - testRatio: 뒤쪽 몇 %를 검증(테스트)용으로 뗄지.
            - seqLen: LSTM 슬라이딩 윈도우 길이. LSTMPipeline 생성 시 그대로 넘김.
        """
        self.datasetBuilder = datasetBuilder
        self.metrics = metrics
        self.groups = groups
        self.testRatio = testRatio
        self.seqLen = seqLen

        self.results: list = []
        # runAll() 실행 후 이 리스트에 {group, model_name, nmae, ficr, score} 딕셔너리들이 누적됨
        # 처음엔 빈 리스트로 시작해서, "아직 한 번도 실행 안 했다"는 상태를 명확히 표현

    def _splitAndScale(self, groupDf: pd.DataFrame, targetCol: str):
        """시간순 분할 + train 구간 기준 스케일링을 담당하는 내부 헬퍼 메서드

        Args:
            - groupDf: 특정 그룹의 전체 피처+타깃 데이터프레임 (아직 분할 전)
            - targetCol: 타깃 컬럼명 (예: "kpx_group_1")

        Returns:
            - (X_train, X_test, y_train, y_test) 튜플

        Logic:
            - 시계열 데이터이므로 랜덤 셔플 split은 절대 금지한다
              (미래 정보가 학습에 섞여 들어가는 data leakage가 발생하기 때문).
            - MinMaxScaler는 반드시 train 구간에만 fit해서,
              test 구간의 통계(최솟값/최댓값)가 스케일링 파라미터에 섞이지 않도록 한다.
        """
        featCols = _featureCols(groupDf, targetCol)
        # kst_dtm, kpx_group_* 를 제외한 나머지를 입력 피처로 사용

        X_all = groupDf[featCols].values
        y_all = groupDf[targetCol].values
        # y_all의 단위는 kWh

        n = len(groupDf)
        split = int(n * (1 - self.testRatio))
        # 앞쪽 (1-testRatio)를 train, 뒤쪽 testRatio를 test로 시간순 분할

        xScaler = MinMaxScaler().fit(X_all[:split])
        # train 구간(X_all[:split])에만 fit
        X_scaled = xScaler.transform(X_all)
        # fit된 스케일러로 train/test 전체를 동일하게 변환

        return X_scaled[:split], X_scaled[split:], y_all[:split], y_all[split:]
        # X_train, X_test, y_train, y_test 순서로 반환

    def runGroup(self, group: int) -> list:
        """단일 그룹에 대해 Persistence/SVR/LSTM 세 모델을 학습·평가하는 메서드

        Args:
            - group: KPX 그룹 번호 (1/2/3)

        Returns:
            - [{"group":.., "model_name":.., "nmae":.., "ficr":.., "score":..}, ...]
              형태의 리스트 (Persistence 1개 + SVR 1개 + LSTM 1개 = 총 3개 dict)
        """
        groupDf = self.datasetBuilder.build(group)
        # Step4/6 연결부: 이 그룹 전용 피처+타깃 데이터프레임 생성
        targetCol = f"kpx_group_{group}"

        X_train, X_test, y_train, y_test = self._splitAndScale(groupDf, targetCol)
        # 내부 헬퍼로 시간순 분할 + 스케일링을 한 번에 처리

        groupResults = []
        # 이 그룹의 Persistence/SVR/LSTM 결과 3개를 담을 리스트

        # ---------------------------------------------------------------
        # (1) Persistence 베이스라인
        # ---------------------------------------------------------------
        persPred = BaselineModels.persistenceForecast(y_test)[1:]
        # np.roll로 한 칸 밀면 index 0은 wrap-around → 버리고 [1:]부터 사용
        groupResults.append({
            "group": group,
            "model_name": "Persistence",
            **{k: v for k, v in self.metrics.summarize(persPred, y_test[1:], group).items() if k != "group"},
            # group 키는 위에서 이미 추가했으므로 summarize 반환값에서 제외
        })

        # ---------------------------------------------------------------
        # (2) SVR 베이스라인
        # ---------------------------------------------------------------
        svr = BaselineModels.makeSVRModel()
        # 매 그룹마다 새 SVR 인스턴스 생성 (그룹별로 별도 학습해야 하므로)
        svr.fit(X_train, y_train)
        # 학습
        svrPred = svr.predict(X_test)
        # 예측
        groupResults.append({
            "group": group,
            "model_name": "SVR",
            **{k: v for k, v in self.metrics.summarize(svrPred, y_test, group).items() if k != "group"},
        })

        # ---------------------------------------------------------------
        # (3) LSTM (LSTMPipeline으로 학습·예측을 캡슐화)
        # ---------------------------------------------------------------
        print(f"[KPX_{group}] LSTM 학습 시작 (n_features={X_train.shape[1]})")

        lstmPipeline = LSTMPipeline(capacity_kw=RATED_CAPACITY_KW[group], seq_len=self.seqLen)
        # 이 그룹 전용 설비용량으로 LSTMPipeline 생성
        # (capacity_kw가 그룹마다 다르므로, 그룹 공용 인스턴스를 재사용하면 안 되고
        #  매 그룹마다 새로 만들어야 함)

        lstmPipeline.fit(X_train, y_train)
        # 학습 (내부적으로 self.model이 채워짐)

        lstmPred = lstmPipeline.predict(X_test, y_test)
        # 예측 (fit이 먼저 호출됐는지는 LSTMPipeline 내부에서 자체 검증함)

        yTestAligned = y_test[self.seqLen:]
        # SequenceDataset이 앞쪽 seqLen개 시점은 "정답을 만들 수 없어서" 버렸으므로
        # y_test도 동일하게 앞쪽 seqLen개를 잘라내야 예측값과 정답의 인덱스가 맞음

        groupResults.append({
            "group": group,
            "model_name": "LSTM",
            **{k: v for k, v in self.metrics.summarize(lstmPred, yTestAligned, group).items() if k != "group"},
        })

        return groupResults

    def runAll(self) -> pd.DataFrame:
        """전체 그룹(self.groups)을 순회하며 학습·평가하고, 결과를 DataFrame으로 반환하는 메서드

        Logic:
            - self.results를 매번 초기화한 뒤 다시 채운다.
              (runAll()을 여러 번 호출해도 이전 결과가 중복 누적되지 않도록 하기 위함)
        """
        self.results = []
        # 매 실행마다 결과 리스트를 새로 초기화

        for group in self.groups:
            print(f"===== KPX_{group} =====")
            self.results.extend(self.runGroup(group))
            # 그룹 하나당 3개(Persistence/SVR/LSTM) 결과 dict를 리스트에 이어붙임

        return pd.DataFrame(self.results)
        # 전체 결과를 DataFrame으로 변환해서 반환 (pivot 등 후처리하기 편하도록)

    def finalScores(self) -> dict:
        """모델별로 3개 그룹의 nmae/ficr을 평균 낸 뒤, 대회 정식 총점을 계산하는 메서드

        Returns:
            - {"Persistence": 0.xx, "SVR": 0.xx, "LSTM": 0.xx} 형태의 딕셔너리

        Raises:
            - RuntimeError: runAll()을 먼저 호출하지 않아 self.results가 비어있는 경우
        """
        if not self.results:
            # self.results가 빈 리스트([])면 아직 runAll()이 호출된 적 없다는 뜻
            raise RuntimeError("runAll()을 먼저 호출해 결과를 생성해야 합니다.")

        resultDf = pd.DataFrame(self.results)
        # 누적된 결과 리스트를 다시 DataFrame으로 변환

        scores = {}
        for modelName in resultDf["model_name"].unique():
            # "Persistence", "SVR", "LSTM" 각각에 대해 반복

            sub = resultDf[resultDf["model_name"] == modelName]
            # 이 모델에 해당하는 3개 그룹 행만 필터링

            groupNMAE = dict(zip(sub["group"], sub["nmae"]))
            groupFICR = dict(zip(sub["group"], sub["ficr"]))
            # {1: nmae_1, 2: nmae_2, 3: nmae_3} 형태의 딕셔너리로 변환
            # (evaluateTotalScore가 이 형태의 입력을 받도록 설계되어 있으므로)

            scores[modelName] = self.metrics.evaluateTotalScore(groupNMAE, groupFICR)
            # Step5의 evaluateTotalScore 재사용:
            # "3그룹 NMAE 평균 / FICR 평균을 먼저 낸 뒤 총점 공식에 대입"하는 순서를 그대로 따름

        return scores