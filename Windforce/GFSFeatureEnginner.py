import pandas as pd
import numpy as np
from typing import List
from .FeatureEngineer import FeatureEngineer

class GFSFeatureEngineer(FeatureEngineer):
    """GFS 기상예보 데이터 전처리를 담당하는 클래스
    
    Attributes:
        - 지상 10m : 순간값
        - 지상 80m : 순간값
        - 지상 100m : 순간값
        - 행성경계층(PBL) 바람      
        - 850 hPa 상층 바람    
        - 700 hPa 상층 바람
        - 500 hPa 상층 바람                                                                                               
    """
    def __init__(self):
        super().__init__()
        # 부모 클래스 FeatureEngineer의 초기화 메서드 호출

    def transformGFS(self, df: pd.DataFrame) -> pd.DataFrame:
        """GFS 테이블 명세서 기반 전처리 메서드"""
        df = df.copy()
        
        # 고도 및 성분별 바람 벡터 분해 및 결합
        gfs_mapping = {
            # GFS 데이터 내 각 고도/기압 레벨별 U, V 성분 컬럼 명세 매핑 정의
            "gfs_10m": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            "gfs_80m": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
            "gfs_100m": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
            "gfs_pbl": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
            "gfs_850hPa": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
            "gfs_700hPa": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
            "gfs_500hPa": ("isobaricInhPa_500_u", "isobaricInhPa_500_v")
        }
        # Key: 새롭게 생성될 변수들의 접두사(Prefix)
        # Value: (동서방향 U 성분 컬럼명, 남북방향 V 성분 컬럼명)
        
        ws_cols = []
        wd_cols = []
        # 부모 클래스의 일괄 처리 메서드로 넘겨주기 위해 고도별 새 변수명을 담을 리스트        
        
        for prefix, (u_col, v_col) in gfs_mapping.items():
            # for 루프를 돌며 각 고도별 U, V 벡터 성분을 물리적 풍속(m/s)과 기상학적 풍향(0~360도)으로 복원            
            if u_col in df.columns and v_col in df.columns:
                ws_name = f"{prefix}_ws_raw" # 생성될 물리 절대 풍속 컬럼명
                wd_name = f"{prefix}_wd_raw" # 생성될 기상학적 각도 풍향 컬럼명
                
                df[ws_name] = self.computeWindSpeed(df[u_col], df[v_col])
                # u, v 화살표의 총 길이(절대 풍속)를 계산
                df[wd_name] = self.computeWindDirection(df[u_col], df[v_col])
                # 북쪽 0도 기준, 시계방향으로 불어오는 풍향 각도(0~360) 계산
                
                ws_cols.append(ws_name)
                wd_cols.append(wd_name)
                # 일괄 파생 변수 생성을 위한 리스트 업로드
        
        df = self.transformWindDirection(df, wind_direction_cols = wd_cols)
        # 부모 클래스 메서드로 풍향 주기성 인코딩 일괄 적용 (Sin/Cos 변환)

        return df