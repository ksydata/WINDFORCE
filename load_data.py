"""
데이터 압축 해제 및 로드 유틸리티
Jupyter Notebook에서 사용하세요.
"""

import zipfile
import os
import pandas as pd

def extract_data():
    """압축 파일들을 압축 해제합니다."""
    
    # 압축 파일 목록
    zip_files = ['train_data.zip', 'test_data.zip']
    
    for zip_file in zip_files:
        if os.path.exists(zip_file):
            print(f"'{zip_file}' 압축 해제 중...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
            print(f"✓ '{zip_file}' 압축 해제 완료!")
        else:
            print(f"⚠️  '{zip_file}' 파일을 찾을 수 없습니다.")
    
    print("\n모든 데이터 파일 압축 해제 완료!")

def load_train_data():
    """학습 데이터를 로드합니다."""
    
    # 압축 해제가 안 되어있으면 먼저 압축 해제
    if not os.path.exists('TRAIN/gfs_train.csv'):
        print("데이터 파일이 없습니다. 압축 해제를 시작합니다...\n")
        extract_data()
    
    print("\n학습 데이터 로드 중...")
    train_data = {
        'gfs': pd.read_csv('TRAIN/gfs_train.csv'),
        'ldaps': pd.read_csv('TRAIN/ldaps_train.csv'),
        'scada_unison': pd.read_csv('TRAIN/scada_unison_train.csv'),
        'scada_vestas': pd.read_csv('TRAIN/scada_vestas_train.csv'),
        'labels': pd.read_csv('TRAIN/train_labels.csv')
    }
    
    print("✓ 학습 데이터 로드 완료!")
    for name, df in train_data.items():
        print(f"  - {name}: {df.shape}")
    
    return train_data

def load_test_data():
    """테스트 데이터를 로드합니다."""
    
    # 압축 해제가 안 되어있으면 먼저 압축 해제
    if not os.path.exists('TEST/gfs_test.csv'):
        print("데이터 파일이 없습니다. 압축 해제를 시작합니다...\n")
        extract_data()
    
    print("\n테스트 데이터 로드 중...")
    test_data = {
        'gfs': pd.read_csv('TEST/gfs_test.csv'),
        'ldaps': pd.read_csv('TEST/ldaps_test.csv')
    }
    
    print("✓ 테스트 데이터 로드 완료!")
    for name, df in test_data.items():
        print(f"  - {name}: {df.shape}")
    
    return test_data

# Jupyter Notebook에서 사용 예시:
"""
# 데이터 압축 해제만 하기
from load_data import extract_data
extract_data()

# 또는 직접 로드 (자동으로 압축 해제됨)
from load_data import load_train_data, load_test_data

train_data = load_train_data()
test_data = load_test_data()

# 개별 데이터프레임 접근
gfs_train = train_data['gfs']
labels = train_data['labels']
"""
