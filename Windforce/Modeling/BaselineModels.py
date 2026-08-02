# Step 6.1. BaselineModels: 상태를 갖지 않는 베이스라인 모델 유틸 클래스

# Persistence와 SVR은 "학습된 가중치를 계속 들고 있어야 하는" 상태가 없으므로
# LSTMPipeline처럼 인스턴스 변수(self.model)를 가질 이유가 없다.
# 그래서 인스턴스화하지 않고 staticmethod로만 호출하는 네임스페이스 클래스로 만든다.

import numpy as np
from sklearn.svm import SVR

class BaselineModels:
    """Persistence/SVR처럼 상태를 갖지 않는 베이스라인 모델들을 모아두는 유틸 클래스

    Logic:
        - 인스턴스화하지 않고 static method로만 호출한다 (self가 필요 없으므로).
        - LSTM처럼 학습된 가중치를 들고 있어야 하는 모델은 여기 넣지 않고
          별도의 LSTMPipeline 클래스로 분리한다.
    """

    @staticmethod
    def persistenceForecast(series: np.ndarray) -> np.ndarray:
        """직전 시점 값을 그대로 이번 시점 예측값으로 사용하는 최소 성능 기준선 메서드

        Args:
            - series: 실제 발전량 시계열 배열, shape = (N, )

        Logic:
            - np.roll(series, 1) : 배열 전체를 한 칸씩 뒤로 밀어서
              idx번째 값이 idx-1번째 값이 되도록 만든다.
            - 즉 "한 시점 전 실제값을 이번 시점 예측값으로 그대로 쓴다"는 의미다.
            - 맨 앞 원소는 맨 뒤 값이 wrap-around로 들어가지만,
              평가 구간이 충분히 길면 그 영향은 무시할 수 있다.
        """
        return np.roll(series, 1)
        # 시계열을 한 칸 밀어서 반환 (베이스라인 예측값)

    @staticmethod
    def makeSVRModel() -> SVR:
        """SVR 베이스라인 모델을 생성하는 메서드

        Logics:
            - kernel = "rbf" : 풍속-발전량 관계처럼 비선형(S자형) 패턴을 포착하기 위해
              선형 커널이 아닌 RBF(가우시안) 커널을 사용한다.
            - C = 10.0 : 오차 허용 정도를 조절하는 정규화 파라미터 (초기 baseline 값).
            - epsilon = 0.01 : 오차를 무시하는 margin 폭 (이 안의 오차는 loss에 반영 안 함).
            - gamma = "scale" : 커널 폭을 데이터 분산에 맞춰 자동으로 정함.
            - 실제 데이터로 GridSearch 등을 통한 하이퍼파라미터 튜닝이 추가로 필요하다.
        """
        return SVR(kernel = "rbf", C = 10.0, epsilon = 0.01, gamma = "scale")
        # 새 SVR 인스턴스를 매번 생성해서 반환 (그룹마다 별도로 fit해야 하므로)