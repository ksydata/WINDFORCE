"""baseline_6에서 쓰는 버전별(v1/v2/v3) LSTM 파이프라인 모음.

기존 :mod:`LSTMPipeline`(baseline_5, 단층 구조)은 그대로 두고 건드리지 않는다.
아래 변형들은 타깃 정규화(설비용량 대비 이용률)와 대회 지향 손실(SmoothL1 워밍업
+ ScoreLossFunction 혼합)은 그대로 재사용하면서, LSTM 출력을 시퀀스에서 어떻게
하나의 벡터로 뽑아내는지(표현 방식)만 v1/v2/v3로 다르게 한다.

    v1: 원본과 동일 — 마지막 시퀀스 위치의 hidden state만 사용
    v2: LayerNorm + Dropout을 얹어 과적합·내부 공변량 변화를 완화
    v3: Attention(softmax 가중합)으로 24개 시점 전부를 가중 반영
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Windforce.ScoreLossFunction import ScoreLossFunction
# 대회 총점(1-NMAE, FICR)을 미분 가능하게 근사한 손실함수. 부모 패키지에서 그대로 재사용


class _SequenceDataset(Dataset):
    """슬라이딩 윈도우 방식으로 (시퀀스, 타깃) 쌍을 만드는 PyTorch Dataset 클래스

    Args:
        - X: 정규화된 피처 배열, shape = (N, n_features)
        - y: 타깃 배열(설비용량 대비 이용률 0~1로 정규화된 값), shape = (N, )
        - seq_len: 슬라이딩 윈도우 길이 (과거 몇 시간을 볼지)

    Logic:
        - 원본 LSTMPipeline의 SequenceDataset과 동일한 규칙을 따른다:
          i번째 샘플 = X[i : i+seq_len]을 입력으로, y[i+seq_len]을 정답으로 사용
        - 클래스명을 "_SequenceDataset"으로 언더스코어를 붙여 모듈 내부 전용임을 표시한다
          (baseline_5의 공개 SequenceDataset과 이름이 겹치지 않도록)
    """

    def __init__(self, X, y, seq_len):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        # torch.as_tensor: 이미 텐서면 복사 없이 그대로 쓰고, 아니면 변환한다 (tensor()보다 가벼움)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.seq_len = int(seq_len)
        # 문자열 등 다른 타입으로 잘못 전달돼도 여기서 int로 강제 변환해 방어한다

    def __len__(self):
        # seq_len개를 소비하고 나서 예측 가능한 타임스텝 수
        # max(0, ...)로 감싸 len(X) <= seq_len인 극단적으로 짧은 입력에서도 음수 길이를 방지한다
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        # i번째 시퀀스 입력과, 그 바로 다음 시점의 정답을 튜플로 반환
        return self.X[i:i + self.seq_len], self.y[i + self.seq_len]


class _SequenceModel(nn.Module):
    """LSTM 기반 시계열 회귀 모델 (v1/v2/v3 표현 전략을 한 클래스에서 분기)

    Args:
        - n_features: 입력 피처 수 (X_train.shape[1])
        - hidden_size: LSTM 은닉 상태 차원 수
        - num_layers: 스택 LSTM 레이어 수
        - variant: "v1"(기본) / "v2"(정규화+드롭아웃) / "v3"(어텐션)
        - dropout: v2의 Dropout 확률이자, num_layers > 1일 때 LSTM 레이어 간 dropout으로도 재사용

    Logic:
        - v1: 원본과 동일하게 마지막 시퀀스 위치의 hidden state만 사용 (sequence-to-one)
        - v2: LayerNorm으로 표현을 정규화한 뒤 Dropout으로 과적합을 억제
        - v3: Linear(hidden_size -> 1)로 각 시점의 중요도 스칼라를 만들고
          softmax로 24개 시점에 대한 가중치를 정규화한 뒤 가중합(어텐션 컨텍스트 벡터) 사용
        - 발전량은 설비용량 대비 이용률(0~1)로 학습하므로 마지막에 sigmoid를 적용한다
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, variant="v1", dropout=0.2):
        super().__init__()
        self.variant = variant
        # forward()에서 어떤 표현 전략을 쓸지 분기하는 데 쓰는 식별자

        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        # batch_first=True: 입력 shape = (batch, seq_len, n_features)
        # PyTorch LSTM은 num_layers=1일 때 레이어 간 dropout을 지원하지 않으므로 조건부로 0.0 처리

        self.norm = nn.LayerNorm(hidden_size) if variant in {"v2", "v3"} else nn.Identity()
        # v1은 원본과 동일하게 정규화를 쓰지 않고, v2/v3만 LayerNorm 적용 (Identity는 그냥 통과)
        self.dropout = nn.Dropout(dropout) if variant == "v2" else nn.Identity()
        # dropout은 v2 전용 — v3는 어텐션 자체가 정규화 효과를 어느 정도 가지므로 별도 dropout을 안 씀
        self.attention = nn.Linear(hidden_size, 1) if variant == "v3" else None
        # 시점별 hidden state -> 스칼라 중요도 점수로 투영하는 어텐션 가중치 레이어 (v3 전용)
        self.fc = nn.Linear(hidden_size, 1)
        # 최종 표현 벡터를 스칼라 발전량(이용률) 예측값으로 변환하는 출력 레이어

    def forward(self, x):
        out, _ = self.lstm(x)
        # out shape = (batch, seq_len, hidden_size). LSTM의 셀 상태(_)는 여기서 쓰지 않는다

        if self.variant == "v3":
            weights = torch.softmax(self.attention(out).squeeze(-1), dim=1).unsqueeze(-1)
            # (batch, seq_len, hidden_size) -> (batch, seq_len, 1) -> squeeze -> (batch, seq_len)
            # softmax(dim=1): 시퀀스 축(24개 시점)을 확률분포로 정규화해 합이 1이 되게 함
            rep = (out * weights).sum(dim=1)
            # 시점별 hidden state를 가중치로 가중합 -> (batch, hidden_size) 컨텍스트 벡터
        else:
            rep = out[:, -1, :]
            # v1/v2: 마지막 시퀀스 위치의 hidden state만 추출 (원본 LSTMPipeline과 동일한 방식)

        rep = self.dropout(self.norm(rep))
        # v1은 둘 다 Identity라 값이 그대로 통과, v2/v3는 LayerNorm 적용, v2만 추가로 Dropout까지 적용
        return torch.sigmoid(self.fc(rep).squeeze(-1))
        # shape = (batch, ). sigmoid로 0~1 범위를 보장해 설비용량 대비 이용률로 해석 가능하게 함


class VersionedLSTMPipeline:
    """LSTM 표현 전략(v1/v2/v3)을 갈아끼울 수 있는 파이프라인 베이스 클래스

    사용 순서(원본 LSTMPipeline과 동일한 인터페이스를 따름):
        pipeline = LSTMPipelineV2(capacity_kw=21600.0)
        pipeline.fit(X_train, y_train, X_val=..., y_val=..., metrics=..., group=...)
        pred = pipeline.predict(X_test, y_test)

    Attributes:
        - version: 하위 클래스(V1/V2/V3)가 오버라이드하는 클래스 속성.
          _SequenceModel(variant=self.version)로 그대로 전달되어 표현 전략을 결정한다.
    """

    version = "base"
    # 서브클래스가 "v1"/"v2"/"v3"로 오버라이드하지 않으면 _SequenceModel이 인식 못 해 v1과 동일하게 동작

    def __init__(self, capacity_kw: float, seq_len=24, hidden_size=64, num_layers=2,
                 epochs=80, lr=1e-3, warmup_epochs=15, loss_k=40.0,
                 regression_weight=0.75, patience=12, batch_size=64,
                 dropout=0.2, random_state=42):
        """
        Args:
            - capacity_kw: 이 파이프라인이 다루는 그룹의 설비용량(kW).
              ScoreLossFunction 초기화와 predict() 역정규화에 그대로 쓰인다.
            - seq_len: LSTM 입력 시퀀스 길이 (과거 몇 시간을 볼지).
            - hidden_size, num_layers: LSTM 내부 구조 하이퍼파라미터.
            - epochs, lr: 학습 반복 횟수와 학습률.
            - warmup_epochs: 이 epoch 수까지는 SmoothL1 회귀 손실만 사용 (그래디언트 소실 방지).
            - loss_k: ScoreLossFunction의 sigmoid steepness.
            - regression_weight: 워밍업 이후 회귀 손실과 점수 손실의 혼합 비율.
            - patience: 검증 총점이 이 횟수만큼 개선되지 않으면 조기종료.
            - batch_size: DataLoader 배치 크기 (원본은 하드코딩 64였지만 여기선 인자로 노출).
            - dropout: v2/LSTM 레이어 간 dropout에 공용으로 쓰는 확률.
            - random_state: torch.manual_seed에 사용해 버전 간 비교 시 재현성을 보장.
        """
        self.capacity_kw = float(capacity_kw)
        # 문자열/정수로 들어와도 실수 연산에 안전하도록 float으로 명시 변환
        self.seq_len, self.hidden_size, self.num_layers = seq_len, hidden_size, num_layers
        self.epochs, self.lr, self.warmup_epochs = epochs, lr, warmup_epochs
        self.loss_k, self.regression_weight, self.patience = loss_k, regression_weight, patience
        self.batch_size, self.dropout, self.random_state = batch_size, dropout, random_state
        self.history, self.model, self.best_epoch = [], None, None
        # history: epoch별 loss·val_score 기록 / model: 학습 후 채워지는 nn.Module
        # best_epoch: 조기종료 판단 기준이 된 최고 검증 점수의 epoch 번호 (버전 간 비교용 메타데이터)

    def fit(self, X_train, y_train, X_val=None, y_val=None, metrics=None, group=None):
        """학습 데이터로 LSTM을 학습시키는 메서드

        Args:
            - X_train: 정규화된 학습용 피처 배열, shape = (N_train, n_features)
            - y_train: 학습용 실제 발전량(kWh) 배열, shape = (N_train, )
            - X_val, y_val: 검증용 피처·발전량. 주어지면 매 epoch 검증 점수로 조기종료 판단
            - metrics: EvaluationMetrics 인스턴스 또는 .score(y_true, y_pred[, capacity])를
              지원하는 객체. 없으면 검증 손실의 음수 절대오차로 대체 채점한다
            - group: VersionedExperimentRunner 호출부와 인터페이스를 맞추기 위해 받지만
              이 파이프라인 내부에서는 사용하지 않는다 (del group으로 명시적으로 버림)

        Returns:
            - self: 학습이 끝난 자기 자신을 반환해 메서드 체이닝을 지원한다
        """
        del group
        # 인터페이스 호환용으로만 받는 인자임을 명시 (사용하지 않는다는 의도를 드러냄)
        if len(X_train) <= self.seq_len:
            raise ValueError("학습 데이터 행 수는 seq_len보다 커야 합니다.")
            # seq_len보다 짧으면 슬라이딩 윈도우 샘플을 하나도 만들 수 없으므로 조기에 명확히 에러

        torch.manual_seed(self.random_state)
        # 버전(v1/v2/v3) 간 초기 가중치·배치 셔플 조건을 동일하게 맞춰 공정 비교가 가능하게 함

        y_norm = np.clip(np.asarray(y_train, dtype=np.float32) / self.capacity_kw, 0, 1)
        # kWh 원 단위 타깃을 설비용량 대비 이용률(0~1)로 정규화 (원본 LSTMPipeline과 동일한 관례)
        ds = _SequenceDataset(X_train, y_norm, self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        # shuffle=True: 배치 순서만 섞고, 각 시퀀스 내부의 시간 순서는 Dataset에서 이미 고정돼 안전함

        self.model = _SequenceModel(X_train.shape[1], self.hidden_size, self.num_layers,
                                    self.version, self.dropout)
        # self.version(클래스 속성)을 그대로 넘겨 v1/v2/v3 표현 전략을 결정
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Adam optimizer로 self.model의 모든 파라미터를 업데이트

        score_loss = ScoreLossFunction(capacity_kw=self.capacity_kw, k=self.loss_k)
        reg_loss = nn.SmoothL1Loss(beta=0.05)
        # score_loss 단독 사용 시 초기 큰 오차에서 sigmoid가 포화되므로, SmoothL1 회귀 신호를 함께 씀

        best_score, best_state, stale = -np.inf, None, 0
        # best_score: 지금까지 관측된 최고 검증 점수 (또는 최고 -loss)
        # best_state: 그 시점의 모델 가중치 스냅샷 (deepcopy로 저장해 이후 학습에 영향받지 않게 함)
        # stale: 개선이 없었던 연속 epoch 수 (patience와 비교해 조기종료 판단)
        self.history = []

        for epoch in range(self.epochs):
            self.model.train(); losses = []
            # 학습모드 전환 + 이번 epoch의 배치별 loss를 모을 리스트 초기화
            for xb, yb in loader:
                opt.zero_grad(); pred = self.model(xb)
                # 이전 스텝의 gradient 초기화 후 forward pass

                if epoch < self.warmup_epochs:
                    loss = reg_loss(pred, yb)
                    # 워밍업 구간: SmoothL1 회귀 손실만 사용해 학습 초반 그래디언트를 안정화
                else:
                    loss = self.regression_weight * reg_loss(pred, yb) + (1 - self.regression_weight) * score_loss(pred * self.capacity_kw, yb * self.capacity_kw)
                    # 워밍업 이후: 회귀 손실과 점수 손실을 가중합. score_loss는 kWh 단위로 역변환해 전달
                    # (ScoreLossFunction이 설비용량 기준 오차율로 계산하도록 대회 정의를 유지)

                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 역전파 후 그래디언트 노름을 1.0으로 클리핑해 LSTM 특유의 그래디언트 폭주를 방지
                # (원본 LSTMPipeline에는 없던 안전장치 — 버전 실험이라 여러 구조를 다뤄야 해서 추가)
                opt.step(); losses.append(float(loss.detach()))
                # 파라미터 업데이트 후, detach()로 그래프에서 분리한 loss 값을 기록용으로 저장

            val_score = np.nan
            # 검증 데이터가 없으면 이후 로직에서 np.isfinite(val_score)가 False가 되어 학습loss 기준으로 대체됨
            if X_val is not None and y_val is not None and len(X_val) > self.seq_len:
                pred = self.predict(X_val, y_val)
                yt = np.asarray(y_val)[self.seq_len:]
                # SequenceDataset이 앞쪽 seq_len개 시점을 버리므로 정답도 동일하게 잘라 인덱스를 맞춤
                val_score = _score(metrics, yt, pred, self.capacity_kw) if metrics is not None else -float(np.mean(np.abs(yt - pred)))
                # metrics가 있으면 대회 지표 기반 점수, 없으면 MAE의 음수(클수록 좋게 부호 통일)

            monitor = val_score if np.isfinite(val_score) else -float(np.mean(losses))
            # 검증 점수가 유효하지 않으면(검증셋 없음/너무 짧음) 학습 loss의 음수를 대신 모니터링
            self.history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_score": float(monitor)})

            if monitor > best_score:
                best_score, best_state, self.best_epoch, stale = monitor, copy.deepcopy(self.model.state_dict()), epoch + 1, 0
                # 개선되면 최고 기록 갱신 + 현재 가중치 스냅샷 저장 + 정체 카운터 리셋
            else:
                stale += 1
                # 개선이 없으면 정체 카운터 증가

            if stale >= self.patience:
                break
                # patience만큼 연속으로 개선이 없으면 조기종료 (원본 LSTMPipeline과 동일한 사상)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            # 학습 종료 후 마지막 epoch가 아니라 "최고 검증 점수" 시점의 가중치로 되돌림
        return self

    def predict(self, X_test, y_test=None):
        """학습된 모델로 예측값을 반환하는 메서드

        Args:
            - X_test: 정규화된 테스트용 피처 배열
            - y_test: 테스트용 실제 발전량 배열 (없으면 0으로 채운 더미를 사용 —
              슬라이딩 윈도우 shape만 맞추는 용도이며 예측 자체에는 영향 없음)

        Returns:
            - np.ndarray: 예측 발전량(kW) 배열. seq_len만큼 앞이 잘려서
              len(X_test) - seq_len 개의 값만 반환됨

        Raises:
            - RuntimeError: fit()을 먼저 호출하지 않고 predict()를 호출한 경우
        """
        if self.model is None:
            raise RuntimeError("먼저 fit()을 호출해야 합니다.")
            # self.model이 None이라는 것은 아직 학습된 적이 없다는 뜻이므로 조기에 명확히 에러

        dummy = np.zeros(len(X_test), dtype=np.float32) if y_test is None else np.asarray(y_test, dtype=np.float32)
        # y_test가 없으면 0으로 채운 더미 배열로 Dataset의 shape 요구사항만 맞춰준다
        ds = _SequenceDataset(X_test, dummy, self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        # shuffle=False: 예측 순서를 유지해야 y_test와 1:1로 정렬이 맞음

        self.model.eval(); values = []
        # 평가모드 전환 (Dropout·LayerNorm 등의 학습/평가 모드 차이를 반영) + 결과를 모을 리스트

        with torch.no_grad():
            # 추론이므로 gradient 계산을 꺼서 메모리·속도를 아낀다
            for xb, _ in loader:
                # 정답(yb)은 예측에 필요 없으므로 버림(_)
                values.extend((self.model(xb).cpu().numpy() * self.capacity_kw).tolist())
                # sigmoid 출력(이용률 0~1)에 설비용량을 곱해 kW 단위 발전량으로 역변환

        return np.asarray(values, dtype=np.float32)
        # 배치별 결과를 이어붙인 1차원 배열로 반환


def _score(metrics, y_true, y_pred, capacity):
    """metrics 객체의 인터페이스 차이를 흡수해 단일 스칼라 점수로 통일하는 헬퍼 함수

    Args:
        - metrics: .score(y_true, y_pred, capacity) 또는 .score(y_true, y_pred)를
          지원하는 객체. None이면 폴백 로직만 사용
        - y_true, y_pred: 비교할 실제·예측 발전량 배열
        - capacity: 설비용량(kW). metrics.score가 3-인자 시그니처일 때만 전달

    Logic:
        - VersionedExperimentRunner 등 외부에서 넘기는 metrics 객체가
          EvaluationMetrics처럼 capacity 인자를 요구할 수도, 요구하지 않을 수도 있어
          TypeError로 시그니처를 순차적으로 시도해본다 (덕 타이핑 방어 코드)
        - 그래도 실패하면 MAE의 음수를 반환해 "클수록 좋다"는 monitor 규약을 유지한다
    """
    if metrics is not None:
        try:
            return float(metrics.score(y_true, y_pred, capacity))
            # 3-인자 시그니처(EvaluationMetrics류)를 우선 시도
        except TypeError:
            try:
                return float(metrics.score(y_true, y_pred))
                # capacity를 받지 않는 2-인자 시그니처로 재시도
            except Exception:
                pass
                # 두 시그니처 모두 실패하면 아래 폴백으로 넘어감
    return -float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    # metrics가 없거나 호출에 실패한 경우의 최종 폴백: MAE에 음수를 취해 "클수록 좋음" 규약 유지


class LSTMPipelineV1(VersionedLSTMPipeline):
    """원본 LSTMPipeline과 동일한 표현 전략(마지막 hidden state만 사용)의 버전"""
    version = "v1"


class LSTMPipelineV2(VersionedLSTMPipeline):
    """LayerNorm + Dropout을 추가해 과적합·내부 공변량 변화를 완화한 버전"""
    version = "v2"


class LSTMPipelineV3(VersionedLSTMPipeline):
    """Attention 가중합으로 24개 시점 전부를 반영하는 버전"""
    version = "v3"


def make_lstm_pipeline(version, **kwargs):
    """버전 문자열로 LSTM 파이프라인 인스턴스를 생성하는 팩토리 함수

    Args:
        - version: "v1" / "v2" / "v3" (대소문자 무시)
        - kwargs: VersionedLSTMPipeline.__init__에 그대로 전달할 하이퍼파라미터

    Returns:
        - 요청한 버전의 VersionedLSTMPipeline 하위 클래스 인스턴스

    Raises:
        - ValueError: 등록되지 않은 버전 문자열이면
    """
    classes = {"v1": LSTMPipelineV1, "v2": LSTMPipelineV2, "v3": LSTMPipelineV3}
    key = str(version).lower()
    # 대소문자 혼용("V1" 등) 입력도 허용하기 위해 소문자로 정규화
    if key not in classes:
        raise ValueError(f"지원하지 않는 LSTM 버전: {version}")
        # 등록되지 않은 이름이면 어떤 값이 왜 실패했는지 바로 알 수 있게 에러 메시지에 원본 값을 포함
    return classes[key](**kwargs)
    # 매칭된 클래스에 하이퍼파라미터를 그대로 전달해 인스턴스 생성


__all__ = ["VersionedLSTMPipeline", "LSTMPipelineV1", "LSTMPipelineV2", "LSTMPipelineV3", "make_lstm_pipeline"]