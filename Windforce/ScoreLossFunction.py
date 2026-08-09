# Step 5. 평가지표(1-NMAE, FICR) 계산 및 손실함수(ScoreLossFunction)

import torch
import torch.nn as nn

class ScoreLossFunction(nn.Module):
    """평가지표(NMAE, FICR)로 만든 대회 총점(1 - 총점)을 직접 최소화하는 손실함수

    Logics:
        - FICR의 정산단가 계단함수(4/3/0원)는 시그모이드 두 개로 완화하여 미분 가능하게 근사
        - 오차율 6%와 8% 구간별 정산단가 차이 반영을 위해, 오차율 5.9%에서 6.1%로 넘어가는 순간 
          단가가 1원 차이로 변화하는데 이 구간의 미분값이 0이 되지 않도록 시그모이드 함수의 steepness를 조절하는 k값을 설정 
    """

    def __init__(self, capacity_kw: float, k: float = 40.0,
                 NMAE_weight: float = 0.5, FICR_weight: float = 0.5):
        super().__init__()
        self.capacity_kw = capacity_kw
        # KPX 그룹별 설비용량(kW) 인스턴스에 저장
        self.k = k
        # 시그모이드 함수 steepness 조절 인스턴스에 저장
        self.NMAE_weight = NMAE_weight
        # NMAE 가중치 인스턴스에 저장
        self.FICR_weight = FICR_weight
        # FICR 가중치 인스턴스에 저장

    def forward(self, pred_kw: torch.Tensor, actual_kw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - pred_kw: 예측 발전량 배열 (kWh 단위), shape = (batch_size, )
            - actual_kw: 실제 발전량 배열 (kWh 단위), shape = (batch_size, )
            - batch_size: 배치 크기 (시간 단위)
        Logics:
            - scalar tensor (batch 전체에 대한 평균 loss) 반환
            - backward() 호출 시, NMAE와 FICR에 대한 미분값이 계산되어 gradient descent 수행 가능
        """
        errorRatio = torch.abs(pred_kw - actual_kw) / self.capacity_kw
        # |예측 발전량 - 실제 발전량| / KPX 그룹별 설비용량(kW), 오차율 배열을 텐서로 변환
        NMAE_term = errorRatio.mean()

        softPrice = 4.0 - 1.0 * torch.sigmoid(self.k * (errorRatio - 0.06)) \
                        - 3.0 * torch.sigmoid(self.k * (errorRatio - 0.08))
        # 오차율 6%와 8% 구간별 정산단가 차이 반영을 위해, 오차율 5.9%에서 6.1%로 넘어가는 순간 
        # 단가가 1원 차이로 변화하는데 이 구간의 미분값이 0이 되지 않도록
        # 시그모이드 함수의 steepness를 조절하는 k값을 설정 

        weight = actual_kw / (actual_kw.sum() + 1e-8)
        # 실제 발전량에 대한 가중치 계산 (실제 발전량이 큰 시간대에 더 큰 영향력 부여)
        # actual_kw를 정규화하여 가중치로 사용 및 0으로 나누는 경우를 방지하기 위해 작은 값 1e-8을 더함
        softFICR = (softPrice * weight).sum() / 4.0
        # FICR의 근사치 산정: 배치 내 가중평균 단가를 이론상 최대단가인 4원으로 나눈 값

        loss = self.NMAE_weight * NMAE_term + self.FICR_weight * (1 - softFICR)
        # 최종 손실함수 계산: NMAE와 FICR의 가중합
        # 손실함수이므로 FICR는 1 - softFICR로 변환하여 최소화하도록 함(작을수록 좋은 방향으로 일관성 확보)
        
        return loss
