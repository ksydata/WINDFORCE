# Step 6.2. LSTMPipeline: LSTM 모델의 생성·학습·예측을 하나로 묶어 관리하는 클래스
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys, os

# ScoreLossFunction을 Modeling 패키지 밖(부모 패키지)에서 가져온다
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from Windforce.ScoreLossFunction import ScoreLossFunction

# 기존 trainLSTM/predictLSTM 함수는 model 객체를 인자로 주고받는 구조였는데,
# 이렇게 하면 "이 model이 어느 그룹용으로 학습된 것인지" 호출부에서 매번
# 변수명으로 관리해야 해서 실수하기 쉽다.
# 이 클래스는 self.model에 학습된 가중치 상태를 직접 들고 있게 해서,
# "학습된 모델"이라는 개념 자체를 하나의 객체로 캡슐화한다.

SEQ_LEN = 24   # 최근 24시간 시퀀스로 다음 시점 예측


class SequenceDataset(Dataset):
    """슬라이딩 윈도우 방식으로 (시퀀스, 타깃) 쌍을 만드는 PyTorch Dataset 클래스

    Args:
        - X: 정규화된 피처 배열, shape = (N, n_features)
        - y: 타깃 배열(실제 발전량 kWh), shape = (N, )
        - seq_len: 슬라이딩 윈도우 길이 (과거 몇 시간을 볼지)

    Logic:
        - i번째 샘플 = X[i : i+seq_len] 을 입력으로, y[i+seq_len] 을 정답으로 사용한다
        - 앞쪽 seq_len개 시점은 정답을 만들 수 없어 버려진다
          → predict 후 y_test도 y_test[seq_len:] 로 잘라서 비교해야 한다
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        # 입력 피처를 float32 텐서로 변환
        self.y = torch.tensor(y, dtype=torch.float32)
        # 타깃(발전량)을 float32 텐서로 변환
        self.seq_len = seq_len
        # 슬라이딩 윈도우 길이

    def __len__(self) -> int:
        # seq_len개를 소비하고 나서 예측 가능한 타임스텝 수
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]
        # 인덱스 idx부터 seq_len 길이의 입력 시퀀스 슬라이싱
        y_target = self.y[idx + self.seq_len]
        # 시퀀스의 바로 다음 시점을 타깃으로 사용
        return x_seq, y_target


class _LSTMModel(nn.Module):
    """LSTM 기반 시계열 회귀 모델 (LSTMPipeline 내부 전용)

    Args:
        - n_features: 입력 피처 수 (X_train.shape[1])
        - hidden_size: LSTM 은닉 상태 차원 수
        - num_layers: 스택 LSTM 레이어 수

    Logic:
        - LSTM → 마지막 시퀀스 출력만 추출 → FC → 스칼라
        - 발전량은 음수가 될 수 없으므로 출력에 ReLU를 적용한다
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # batch_first=True: 입력 shape = (batch, seq_len, n_features)
        )
        self.fc = nn.Linear(hidden_size, 1)
        # LSTM 마지막 출력을 스칼라 발전량 예측값으로 선형 변환

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # out shape = (batch, seq_len, hidden_size)
        last_out = out[:, -1, :]
        # 마지막 시퀀스 위치의 hidden state만 추출 (N→1 예측)
        pred = self.fc(last_out).squeeze(-1)
        # shape = (batch, )
        return torch.relu(pred)
        # 발전량은 음수 불가 → ReLU로 하한을 0으로 보장


class LSTMPipeline:
    """LSTM 모델의 생성·학습·예측을 하나로 묶어 관리하는 클래스

    사용 순서:
        pipeline = LSTMPipeline(capacity_kw = 21600.0)
        pipeline.fit(X_train, y_train)          # 학습 (self.model이 채워짐)
        pred = pipeline.predict(X_test, y_test)  # 예측 (학습된 self.model 사용)
    """

    def __init__(self, capacity_kw: float, seq_len: int = SEQ_LEN,
                 hidden_size: int = 64, num_layers: int = 2,
                 epochs: int = 30, lr: float = 1e-3):
        """
        Args:
            - capacity_kw: 이 파이프라인이 다루는 그룹의 설비용량(kW).
              Step5의 ScoreLossFunction 초기화에 그대로 전달됨.
            - seq_len: LSTM 입력 시퀀스 길이 (과거 몇 시간을 볼지).
            - hidden_size, num_layers: LSTM 내부 구조 하이퍼파라미터.
            - epochs, lr: 학습 반복 횟수와 학습률.
        """
        self.capacity_kw = capacity_kw
        # Step5 ScoreLossFunction(capacity_kw=...) 호출 시 그대로 넘길 값
        self.seq_len = seq_len
        # SequenceDataset 생성 시 슬라이딩 윈도우 길이로 사용
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTMPipeline 생성 시 그대로 전달
        self.epochs = epochs
        self.lr = lr
        # 학습 루프(fit)에서 사용할 하이퍼파라미터

        self.model: nn.Module | None = None
        # 학습 전에는 None. fit()이 끝나야 학습된 LSTMPipeline 인스턴스가 여기 저장됨.
        # predict()에서 이 self.model이 None이면 아직 학습이 안 됐다는 뜻이므로 에러를 낸다.

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMPipeline":
        """학습 데이터로 LSTM을 학습시키는 메서드

        Args:
            - X_train: 정규화된 학습용 피처 배열, shape = (N_train, n_features)
            - y_train: 학습용 실제 발전량(kWh) 배열, shape = (N_train, )

        Returns:
            - self : 학습이 끝난 자기 자신을 반환한다.
              이렇게 하면 `pipeline = LSTMPipeline(...).fit(X, y)`처럼
              생성과 학습을 한 줄로 이어 쓸 수 있다 (메서드 체이닝).
        """
        trainDs = SequenceDataset(X_train, y_train, seq_len=self.seq_len)
        # 슬라이딩 윈도우 시퀀스 데이터셋 생성 (Step6에서 정의한 클래스 그대로 재사용)
        trainLoader = DataLoader(trainDs, batch_size=64, shuffle=True)
        # shuffle=True: 학습 시엔 배치 순서를 섞어 일반화 성능 향상
        # (단, 각 시퀀스 "내부"의 시간 순서는 SequenceDataset에서 이미 고정되어 있으므로 안전함)

        self.model = _LSTMModel(
            n_features=X_train.shape[1],
            # 입력 피처 개수는 X_train의 컬럼 수로 자동 결정 (하드코딩 안 함)
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        # 매 fit() 호출마다 새 모델을 만들어 self.model에 저장
        # (혹시 다시 fit()을 부르면 이전 학습 내용은 사라지고 새로 학습됨)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Adam optimizer로 self.model의 모든 파라미터를 업데이트하도록 설정
        criterion = ScoreLossFunction(capacity_kw=self.capacity_kw)
        # Step5에서 만든 손실함수를 그대로 재사용. capacity_kw는 이 그룹 전용 값.

        self.model.train()
        # dropout/batchnorm 등이 있다면 학습모드로 전환 (현재 구조엔 없지만 관례상 명시)

        for epoch in range(self.epochs):
            epochLoss = 0.0
            # 이번 epoch의 전체 loss 합을 누적할 변수

            for xb, yb in trainLoader:
                optimizer.zero_grad()
                # 이전 스텝에서 계산된 gradient가 남아있지 않도록 초기화

                pred = self.model(xb)
                # forward pass: 배치 입력 xb를 모델에 통과시켜 예측값 계산

                loss = criterion(pred, yb)
                # Step5 ScoreLossFunction으로 이번 배치의 손실 계산

                loss.backward()
                # 역전파: loss에 대한 모든 파라미터의 gradient 계산

                optimizer.step()
                # 계산된 gradient로 파라미터 업데이트 (경사하강법 한 스텝)

                epochLoss += loss.item() * xb.size(0)
                # loss.item()은 배치 평균값이므로, 배치 크기(xb.size(0))를 곱해
                # "배치 합"으로 복원해서 누적 (나중에 전체 데이터 개수로 다시 나눔)

            epochLoss /= len(trainLoader.dataset)
            # 전체 학습 데이터 개수로 나눠 "이번 epoch 전체 평균 loss"를 구함

            print(f"  epoch {epoch + 1}/{self.epochs} - loss(1-총점 근사) {epochLoss:.5f}")
            # 학습 진행 상황을 매 epoch마다 출력

        return self
        # 학습이 끝난 self(=self.model이 채워진 상태)를 그대로 반환

    @torch.no_grad()
    # 추론 시에는 gradient 계산이 불필요하므로 데코레이터로 꺼서 메모리/속도를 아낀다
    def predict(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """학습된 모델로 예측값을 반환하는 메서드

        Args:
            - X_test: 정규화된 테스트용 피처 배열
            - y_test: 테스트용 실제 발전량(kWh) 배열
              (SequenceDataset이 슬라이딩 윈도우 정답을 만드는 데 필요해서 함께 받음.
               예측 자체에는 y_test 값이 쓰이지 않고 shape만 맞추는 용도임)

        Returns:
            - np.ndarray: 예측 발전량(kW) 배열. seq_len만큼 앞이 잘려서
              len(X_test) - seq_len 개의 값만 반환됨에 유의.

        Raises:
            - RuntimeError: fit()을 먼저 호출하지 않고 predict()를 호출한 경우.
              self.model이 None인 상태로 그냥 진행하면 알아보기 어려운 에러가 나므로,
              여기서 미리 명확한 에러 메시지로 막아준다.
        """
        if self.model is None:
            raise RuntimeError("fit()을 먼저 호출해 모델을 학습시켜야 합니다.")
            # self.model이 아직 채워지지 않았다는 것은 fit()이 호출된 적 없다는 뜻

        testDs = SequenceDataset(X_test, y_test, seq_len=self.seq_len)
        # 테스트 데이터도 학습 때와 동일한 seq_len으로 슬라이딩 윈도우 구성
        testLoader = DataLoader(testDs, batch_size=64, shuffle=False)
        # shuffle=False: 예측 시엔 순서를 유지해야 y_test와 1:1로 정렬이 맞음

        self.model.eval()
        # 평가모드 전환 (dropout 등 비활성화 - 현재 구조엔 영향 없지만 관례상 명시)

        preds = []
        for xb, _ in testLoader:
            # 정답(yb)은 예측 시 필요 없으므로 버림(_)
            preds.append(self.model(xb).numpy())
            # 배치별 예측 결과를 numpy 배열로 변환해 리스트에 쌓음

        return np.concatenate(preds)
        # 배치별 결과를 이어붙여 하나의 1차원 배열로 반환
