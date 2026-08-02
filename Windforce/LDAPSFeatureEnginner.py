import pandas as pd
import numpy as np
from typing import List
from .FeatureEngineer import FeatureEngineer

class LDAPSFeatureEngineer(FeatureEngineer):
    """LDAPS 기상예보 데이터 전처리를 담당하는 클래스
    
    Attributes:
        - 지상 10m : 순간값
        - 지상 50m : 최댓값, 최솟값
        - 지상 5m 경계층 : 순간값                                                                                                                               
    """
    def __init__(self):
        super().__init__()
        # 부모 클래스 FeatureEngineer의 초기화 메서드 호출

    def transformLDAPS(self, df: pd.DataFrame) -> pd.DataFrame:
        """LDAPS 테이블 명세서 기반 전처리 메서드"""
        df = df.copy()
        
        # 1. 고도 및 성분별 바람 벡터 분해 및 결합 (풍속, 세제곱, 풍향, sin, cos 자동 생성)
        df["ldaps_10m_ws_raw"] = self.computeWindSpeed(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        df["ldaps_10m_wd_raw"] = self.computeWindDirection(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        
        # 지상 50m 최댓값 바람
        df["ldaps_50m_max_ws_raw"] = self.computeWindSpeed(df["heightAboveGround_50_50MUmax"], df["heightAboveGround_50_50MVmax"])
        df["ldaps_50m_max_wd_raw"] = self.computeWindDirection(df["heightAboveGround_50_50MUmax"], df["heightAboveGround_50_50MVmax"])
        
        # 지상 50m 최솟값 바람
        df["ldaps_50m_min_ws_raw"] = self.computeWindSpeed(df["heightAboveGround_50_50MUmin"], df["heightAboveGround_50_50MVmin"])
        df["ldaps_50m_min_wd_raw"] = self.computeWindDirection(df["heightAboveGround_50_50MUmin"], df["heightAboveGround_50_50MVmin"])
        
        # 지상 5m 경계층 바람 (X, Y 방향 성분 사용)
        df["ldaps_5m_bl_ws_raw"] = self.computeWindSpeed(df["heightAboveGround_5_XBLWS"], df["heightAboveGround_5_YBLWS"])
        df["ldaps_5m_bl_wd_raw"] = self.computeWindDirection(df["heightAboveGround_5_XBLWS"], df["heightAboveGround_5_YBLWS"])
        
        wd_cols = [
            "ldaps_10m_wd_raw", 
            "ldaps_50m_max_wd_raw", 
            "ldaps_50m_min_wd_raw", 
            "ldaps_5m_bl_wd_raw"]
        
        df = self.transformWindDirection(df, wind_direction_cols = wd_cols)
        # 부모 클래스 메서드로 풍향 주기성 인코딩 일괄 적용 (0도-360도 속성 부여 및 Sin/Cos 추출)
        
        return df