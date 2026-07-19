# WINDFORCE

풍력 발전 예측 프로젝트

## 데이터 구조

데이터 파일들은 압축되어 저장되어 있습니다:
- `train_data.zip` (85MB) - 학습 데이터
- `test_data.zip` (22MB) - 테스트 데이터

## 사용 방법

### Jupyter Notebook에서 데이터 로드

```python
# 1. 자동으로 압축 해제 및 데이터 로드
from load_data import load_train_data, load_test_data

train_data = load_train_data()
test_data = load_test_data()

# 2. 개별 데이터프레임 접근
gfs_train = train_data['gfs']
ldaps_train = train_data['ldaps']
scada_unison = train_data['scada_unison']
scada_vestas = train_data['scada_vestas']
labels = train_data['labels']

gfs_test = test_data['gfs']
ldaps_test = test_data['ldaps']
```

### 수동 압축 해제

```python
import zipfile

# 학습 데이터 압축 해제
with zipfile.ZipFile('train_data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# 테스트 데이터 압축 해제
with zipfile.ZipFile('test_data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

## 파일 목록

### 학습 데이터 (TRAIN/)
- `gfs_train.csv` - GFS 기상 데이터
- `ldaps_train.csv` - LDAPS 기상 데이터
- `scada_unison_train.csv` - Unison SCADA 데이터
- `scada_vestas_train.csv` - Vestas SCADA 데이터
- `train_labels.csv` - 학습 레이블

### 테스트 데이터 (TEST/)
- `gfs_test.csv` - GFS 기상 데이터
- `ldaps_test.csv` - LDAPS 기상 데이터

### 문서 (MD/)
- `data_description.md` - 데이터 설명
- `Notices.md` - 공지사항

### 기타 (INFO/)
- `sample_submission.csv` - 제출 샘플
