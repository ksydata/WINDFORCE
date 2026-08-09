# Step 5. 평가지표(1-NMAE, FICR) 계산

import numpy as np

# kpx_info.xlsx의 그룹별 설비용량(MW)을 kW로 환산한 값
RATED_CAPACITY_KW = {
    1: 21_600.0,
    # KPX 그룹 1: 21.6MW = 21,600kW
    2: 21_600.0,
    # KPX 그룹 2: 21.6MW = 21,600kW
    3: 21_000.0,
    # KPX 그룹 3: 21.0MW = 21,000kW
}

TIME_STEP_HOURS = 1.0  
# train_labels가 kst_dtm 컬럼이 1시간 단위이므로 시간 해상도 1.0h로 설정
# 10분 단위 데이터에서 변환 작업 필요 시, 10분 단위 데이터의 시간 해상도는 1/6h = 0.1667h로 설정 가능

class EvaluationMetrics:
    """평가지표(1-NMAE, 정산금획득률(FICR)) 계산하는 클래스
    
    평가지표:
        - 1-NMAE : 그룹별 NMAE_g = mean(|예측-실제| / 설비용량) 을
                   3개 그룹 평균낸 뒤 1에서 뺀 값. 클수록(=오차가 작을수록) 좋음.
        - FICR   : 시간별 오차율(NMAE) 구간에 따라 정산단가가 달라짐
                   - 오차율 <= 6%  : 4원/kWh
                   - 6% < 오차율 <= 8% : 3원/kWh
                   - 오차율 > 8%  : 0원/kWh
                   FICR_g = (실제 획득 정산금) / (이론상 최대 정산금, 즉 모든 시간대가 6% 이하라고 가정했을 때의 정산금).
        - 총점    : 0.5*(1-NMAE) + 0.5*FICR  (NMAE, FICR 둘 다 "3개 그룹 평균값" 사용)
    """

    def __init__(self, rated_capacity_kw: dict = RATED_CAPACITY_KW,
                 time_step_hours: float = TIME_STEP_HOURS):
        self.rated_capacity_kw = rated_capacity_kw
        # KPX 그룹별 설비용량 딕셔너리 인스턴스에 저장
        self.time_step_hours = time_step_hours
        # 평가지표 계산에 필요한 시간 단위 인스턴스에 저장

    def evaluateNMAE(self, pred: np.ndarray, actual: np.ndarray, group: int) -> float:
        """단일 그룹의 NMAE(Normalized Mean Absolute Error)를 계산하는 메서드

        Args: 
            - pred: 예측 발전량 배열 (kWh 단위), shape = (시간, )
            - actual: 실제 발전량 배열 (kWh 단위), shape = (시간, )
            - group: KPX 그룹 번호 (1/2/3 중 하나)

        Logic:
            - NMAE = mean(|시간당 예측 발전량 - 시간당 실제 발전량| / KPX 그룹별 설비용량(kW))
            - 용량 대비 오차율로 정규화하고 그 비율들을 전체 시간에 대해 평균을 내어 NMAE를 계산
        """
        capacity = self.rated_capacity_kw[group]
        # KPX 그룹별 설비용량(kW) 가져오기
        return float(np.mean(np.abs(pred - actual) / capacity))
        # NMAE 계산: 시간별 오차율을 평균하여 단일 그룹의 NMAE 반환

    def evaluateFICR(self, pred: np.ndarray, actual: np.ndarray, group: int) -> float:
        """단일 그룹의 정산금획득률(FICR, Financial Incentive Capture Rate)을 계산하는 메서드

        Args:
            pred, actual, group: NMAE 계산과 동일한 입력값

        Logic:
            - 시간별 오차율(hourly NMAE)을 계산하고 정규화
            - 오차율 구간에 따라 시간별 NMAE 구간에 따라 단가(4원/3원/0원) 결정
            - 시간별 실제 발전량(kw)에 단가를 곱해 실제 획득 정산금 계산
            - FICR_g = 실제 획득 정산금 / 이론상 최대 정산금(모든 시간대 NMAE <= 6% = 단가 4원 가정)
        """
        capacity = self.rated_capacity_kw[group]
        # KPX 그룹별 설비용량(kW) 가져오기
        hourlyNMAE = np.abs( ( pred - actual ) / capacity )
        # 시간별 오차율 계산 (NMAE)

        unitPrice = np.where(
            hourlyNMAE <= 0.06, 4.0,
            # NMAE <= 0.06 : 3*1 + 1*1 = 4(원)
            np.where(hourlyNMAE <= 0.08, 3.0, 0.0)
            # 0.06 < NMAE <= 0.08 : 3*1 + 1*0 = 3(원)
            # 0.08 < NMAE : 3*0 + 1*0 = 0(원)        
        )
        # 계단함수 형태로 시간별 단가 결정(4원/3원/0원)

        generation_kwh = actual * self.time_step_hours
        # 실제 발전량(kWh) = 실측 발전량(kW) x 시간해상도(h)

        earningSettlement = np.sum(unitPrice * generation_kwh)
        maxSettlement = np.sum(4.0 * generation_kwh)
        # 이론상 최대 정산금 = 모든 시간대 NMAE <= 6% 가정 시 단가 4원 적용

        if maxSettlement <= 0:
            return 0.0
        return float(earningSettlement / maxSettlement)
        # FICR 계산: 실제 획득 정산금 / 이론상 최대 정산금 비율 계산   

    def evaluateTotalScore(self, groupNMAE: dict, groupFICR: dict) -> float:
        """대회 총점 계산하는 메서드
        
        Args: 
            - groupNMAE: {KPX그룹번호: NMAE 값} 딕셔너리
            - groupFICR: {그룹번호: FICR 값} 딕셔너리

        Logics:
            loss = 0.5*(1 - 3그룹 평균 NMAE) + 0.5*(3그룹 평균 FICR)
        """
        meanNMAE = float(np.mean(list(groupNMAE.values())))
        meanFICR = float(np.mean(list(groupFICR.values())))
        # FICR 계산 시 쓰이는 정산 단가 계단함수(불연속)를 
        # 딥러닝 모델 적합 시 미분을 위해(gradient = 0 구간) 시그모이드 2개로 근사하여 
        # 연속화한 softFICR를 사용하여 총점 계산하여야 함 
        totalScore = 0.5 * (1 - meanNMAE) + 0.5 * meanFICR
        # 총점 계산: 3개 그룹 평균 NMAE와 FICR을 이용하여 최종 점수 산출
        return totalScore

    def summarize(self, pred: np.ndarray, actual: np.ndarray, group: int) -> dict:
        """단일 그룹에 대한 nmae/ficr/score를 한번에 반환하는 헬퍼 메서드"""
        NMAE_group = self.evaluateNMAE(pred, actual, group)
        FICR_group = self.evaluateFICR(pred, actual, group)
        score_group = 0.5 * (1 - NMAE_group) + 0.5 * FICR_group
        return {"group": group, "nmae": NMAE_group, "ficr": FICR_group, "score": score_group}