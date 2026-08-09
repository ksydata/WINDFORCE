# WINDFORCE 프로젝트 구조 문서

## 1. 프로젝트 개요

**목표**: KPX 3개 풍력발전 그룹(Group 1·2·3)의 2025년 전체 기간 시간별 발전량을 예측한다.  
**평가지표**: `총점 = 0.5 × (1-NMAE) + 0.5 × FICR` — 높을수록 좋음  
**제출 형식**: 8,760행(2025년 시간별) × 4컬럼(`kst_dtm`, `kpx_group_1~3`) CSV

| KPX 그룹 | 터빈 제작사 | 터빈 수 | 설비용량 |
|---------|----------|-------|--------|
| Group 1 | VESTAS (V126) | 6기 | 21.6 MW (21,600 kWh/h) |
| Group 2 | VESTAS (V126) | 6기 | 21.6 MW (21,600 kWh/h) |
| Group 3 | UNISON (U136) | 5기 | 21.0 MW (21,000 kWh/h) |

---

## 2. 디렉토리 구조

```
WINDFORCE/
│
├── BASELINE/                    # 실험 노트북
│   ├── baseline_1.ipynb         # 탐색적 분석 (EDA) — 절차적 스타일
│   ├── baseline_2.ipynb         # OOP 버전 (Windforce 패키지 활용)
│   ├── baseline_3.ipynb         # 전체 파이프라인 완성본
│   ├── baseline_5.ipynb         # 손실함수·피처정렬 개선 적용
│   └── baseline5_howto.ipynb    # 피처중요도·SHAP 분석 + LightGBM ← 현행 주력 노트북
│
├── prep/                        # 전처리 완료본 (baseline5_howto Step 1이 생성)
│   ├── dataset_train_group{1,2,3}.csv.gz   # 피처 127개 + 타깃
│   ├── dataset_test_group{1,2,3}.csv.gz    # 평가 기간 피처 127개
│   └── selected_features.csv               # 그룹별 선별 피처 목록
│
├── TRAIN/                       # 학습 데이터 (2022-01-01 ~ 2025-01-01)
│   ├── ldaps_train.csv          # LDAPS 기상예보 (16격자 × 26,304시각)
│   ├── gfs_train.csv            # GFS 기상예보 (9격자 × 26,304시각)
│   ├── train_labels.csv         # KPX 그룹별 실제 발전량 (26,304행)
│   ├── scada_vestas_train.csv   # VESTAS 터빈 SCADA 10분 실측 (157,819행)
│   ├── scada_unison_train.csv   # UNISON 터빈 SCADA 10분 실측 (105,264행)
│   └── kpx_info.csv             # 터빈 메타 (그룹·좌표·허브높이·설비용량)
│
├── TEST/                        # 평가 데이터 (2025-01-01 ~ 2026-01-01)
│   ├── ldaps_test.csv           # LDAPS 기상예보 (16격자 × 8,760시각)
│   └── gfs_test.csv             # GFS 기상예보 (9격자 × 8,760시각)
│
├── INFO/
│   ├── domain.html              # 대회 도메인 참고자료
│   └── sample_submission.csv    # 제출 양식 (8,760행)
│
├── MD/                          # 프로젝트 문서
│   ├── coding_convention.md     # 코딩 컨벤션 (주석·네이밍·타입·모델링 규칙)
│   ├── baseline5_howto_next.md  # baseline5_howto 후속 개선 작업 지시서 (카드 A~G)
│   ├── improvement.md           # baseline_4 정리 및 정확도 개선 방안
│   ├── data_description.md      # 데이터 명세 (컬럼 설명, 기간, 단위)
│   ├── notes.md                 # 실험 노트
│   ├── Notices.md               # 대회 공지
│   └── project_structure.md    # ← 이 파일 (프로젝트 구조 전체 정리)
│
└── Windforce/                   # 핵심 Python 패키지
    ├── __init__.py
    ├── WindforceDataLoader.py
    ├── FeatureEngineer.py
    ├── EvaluationMetrics.py
    ├── ScoreLossFunction.py
    ├── Preprocessing/
    │   ├── __init__.py
    │   ├── FeatureEngineerFactory.py
    │   ├── LDAPSFeatureEngineer.py
    │   ├── GFSFeatureEngineer.py
    │   └── SCADAFeatureEngineer.py
    ├── Modeling/
    │   ├── __init__.py
    │   ├── BaselineModels.py
    │   ├── LSTMPipeline.py
    │   └── GroupExperimentRunner.py
    └── UTILS/
        ├── spatial_utils.py
        ├── time_utils.py
        └── wind_utils.py
```

---

## 3. Windforce 패키지 클래스 다이어그램

```
WindforceDataLoader (dataclass)
│  root / root_train / root_test / encoding
│  _paths (dict)  _cache (dict)
├─ check_paths()          파일 존재 여부 ✅/❌ 출력
├─ load(name)             단일 파일 로드 (캐시 지원)
├─ load_all()             전체 파일 일괄 로드
├─ summarize(name)        컬럼·날짜 범위 요약
└─ __getitem__(name)      loader["ldaps_train"] 딕셔너리 접근

FeatureEngineer (ABC)
│  source (ClassVar)  COLUMN_MAP (ClassVar)
│  group / group_spec
├─ transform(df)          ← 템플릿 메서드 (8단계 순서 고정)
│   ├─ transformColumnName(df)      [추상] 컬럼명 통일
│   ├─ transformTimeAxis(df)        [추상] datetime 변환·시간 메타
│   ├─ checkPhysicalLimit(df)       [추상] 물리한계 플래그
│   ├─ interpolateMissing(df)       [훅]  결측 보간 (기본: pass)
│   ├─ calculatePhysicalFeature(df) [추상] 파생변수 계산
│   ├─ checkDerivedFlag(df)         [훅]  파생 플래그 (기본: pass)
│   ├─ transformCyclicFeature(df)   [훅]  주기 인코딩 (기본: pass)
│   └─ selectGroup(df)              [구체] 그룹 필터
├─ transformToGroupIDW(df, turbine_meta)  16/9격자 → KPX그룹 IDW 집계
├─ transformToGridStats(df)              격자 공간통계 (IDW 대조군)
├─ transformForecastDiff(df)             시계열 변화량·램프
├─ transformToModelInput(df)             숫자형 2차원 표 정리
├─ transformToSequence(df, ...)          (samples, timesteps, features) 3차원 배열
├─ calculateAirDensity(df, ...)          공기밀도 (물리식)
└─ calculateWindPowerDensity(...)        풍력에너지밀도

LDAPSFeatureEngineer(FeatureEngineer)
│  source = "ldaps"   COLUMN_MAP (16m·5m·50m·지표 → snake 이름)
│  IDW_FEATURES (16격자 역거리가중에 쓸 컬럼 목록)
│  STAT_FEATURES (16격자 단순통계 대조군 컬럼 목록)
└─ 추상 메서드 전부 구현 + checkDerivedFlag·transformCyclicFeature 오버라이드

GFSFeatureEngineer(FeatureEngineer)
│  source = "gfs"     COLUMN_MAP (gfs_ 접두사 포함)
│  UV_LEVELS (7개 고도: 10·80·100m·PBL·850·700·500hPa)
│  IDW_FEATURES (9격자 역거리가중 컬럼 목록)
└─ 추상 메서드 전부 구현 + 허브높이 외삽(gfs_ws_hub)·전단지수(gfs_shear_alpha) 포함

SCADAFeatureEngineer(FeatureEngineer)
│  source = "scada"
└─ wide → long 변환 + 10분값 → 시간값 집계 + 품질 플래그

EvaluationMetrics
│  rated_capacity_kw (dict)   time_step_hours
├─ evaluateNMAE(pred, actual, group)   단일 그룹 NMAE
├─ evaluateFICR(pred, actual, group)   단일 그룹 FICR
├─ evaluateTotalScore(groupNMAE, groupFICR)  대회 총점
└─ summarize(pred, actual, group)      nmae/ficr/score 한번에

ScoreLossFunction(nn.Module)
│  capacity_kw  k  NMAE_weight  FICR_weight
└─ forward(pred_kw, actual_kw)   미분 가능한 대회 손실함수

BaselineModels (static only)
├─ persistenceForecast(series)  직전값 그대로 예측 (최소 성능 기준선)
└─ makeSVRModel()               RBF 커널 SVR 생성

SequenceDataset(Dataset)
│  X (tensor)  y (tensor)  seq_len
├─ __len__()
└─ __getitem__(idx)  → (x_seq[seq_len], y_target)

_LSTMModel(nn.Module)
│  lstm(n_features, hidden_size, num_layers)  fc(hidden → 1)
└─ forward(x) → ReLU(FC(LSTM_last_output))

LSTMPipeline
│  capacity_kw  seq_len  hidden_size  num_layers  epochs  lr
│  model (_LSTMModel | None)
├─ fit(X_train, y_train)     학습 (내부 _LSTMModel 생성 + Adam + ScoreLossFunction)
└─ predict(X_test, y_test)   예측 (fit 선행 필수)

WindforceDatasetBuilder  ← baseline_3.ipynb 내부 정의
│  ldaps_group_df  gfs_group_df  labels_df
├─ featureCols(df) [static]  타깃·식별자 제외 피처 목록
└─ build(group)              LDAPS + GFS + labels 결합

GroupExperimentRunner
│  datasetBuilder  metrics  groups  testRatio  seqLen
│  results (list)
├─ _splitAndScale(groupDf, targetCol)  시간순 분할 + MinMaxScaler
├─ runGroup(group)                     Persistence / SVR / LSTM 학습·평가
├─ runAll()                            전체 그룹 순회 → result_df
└─ finalScores()                       모델별 3그룹 평균 총점
```

---

## 4. 데이터 흐름 (Full Pipeline)

```
[TRAIN/ldaps_train.csv]   [TRAIN/gfs_train.csv]
         │                        │
  LDAPSFeatureEngineer      GFSFeatureEngineer
    .transform()              .transform()
         │                        │
   (16격자 파생변수)         (9격자 파생변수)
         │                        │
  .transformToGroupIDW()    .transformToGroupIDW()
    (turbine_meta)             (turbine_meta)
         │                        │
  ldaps_group_df            gfs_group_df
   (3그룹 × 시각)            (3그룹 × 시각)
         │                        │
         └───── WindforceDatasetBuilder.build(group) ────┐
                                                          │
                                               [TRAIN/train_labels.csv]
                                                          │
                                          피처+타깃 DataFrame (group별)
                                                          │
                                      GroupExperimentRunner.runGroup(group)
                                         ┌────────────────────────────┐
                                         │ Persistence (직전값)       │
                                         │ SVR (RBF 커널)             │
                                         │ LSTM (ScoreLossFunction)   │
                                         └────────────────────────────┘
                                                          │
                                          EvaluationMetrics.summarize()
                                           nmae / ficr / score per group
                                                          │
                                         GroupExperimentRunner.finalScores()
                                          최종 총점 (3그룹 평균)
                                                          │
                                       [TEST/ldaps_test.csv + gfs_test.csv]
                                         동일 파이프라인으로 평가기간 예측
                                                          │
                                         submission_baseline3.csv 저장
```

---

## 5. 평가지표 수식

$$
\text{NMAE}_g = \frac{1}{T} \sum_{t=1}^{T} \frac{|\hat{y}_{g,t} - y_{g,t}|}{\text{Cap}_g}
$$

$$
\text{Unit Price}_t = \begin{cases} 4 & \text{if } \text{hourly NMAE}_t \leq 0.06 \\ 3 & \text{if } 0.06 < \text{hourly NMAE}_t \leq 0.08 \\ 0 & \text{otherwise} \end{cases}
$$

$$
\text{FICR}_g = \frac{\sum_t \text{UnitPrice}_t \cdot y_{g,t}}{\sum_t 4 \cdot y_{g,t}}
$$

$$
\text{총점} = 0.5 \times (1 - \overline{\text{NMAE}}) + 0.5 \times \overline{\text{FICR}}
$$

여기서 $\overline{\text{NMAE}}$, $\overline{\text{FICR}}$ 은 KPX 3그룹 평균이다.

---

## 6. 주요 상수

| 상수 | 값 | 설명 |
|------|----|------|
| `RATED_CAPACITY_KW[1]` | 21,600 kW | KPX 그룹 1 설비용량 |
| `RATED_CAPACITY_KW[2]` | 21,600 kW | KPX 그룹 2 설비용량 |
| `RATED_CAPACITY_KW[3]` | 21,000 kW | KPX 그룹 3 설비용량 |
| `TIME_STEP_HOURS` | 1.0 | 발전량 시간 단위 (시간당 kWh) |
| `SEQ_LEN` | 24 | LSTM 입력 시퀀스 길이 (24시간) |
| `IDW_K` | 4 | IDW 가중에 사용할 격자 수 (최근접 4개) |
| `IDW_POWER` | 2.0 | IDW 지수 (거리 제곱에 반비례) |
| `HUB_HEIGHT_M` | 117.0 | VESTAS V126 허브 높이(m) |

---

## 7. 알려진 이슈 및 해결 내역

| 파일 | 이슈 | 해결 |
|------|------|------|
| `Windforce/__init__.py` | `from ..FeatureEngineer` — 잘못된 상대경로 | `from .FeatureEngineer` 로 수정 |
| `Windforce/__init__.py` | `LDAPSFeatureEnginner` (오타) + 경로 오류 | `Preprocessing.LDAPSFeatureEngineer` 로 수정 |
| `Windforce/__init__.py` | `GFSFeatureEnginner` (오타) + 경로 오류 | `Preprocessing.GFSFeatureEngineer` 로 수정 |
| `Windforce/Modeling/__init__.py` | 파일 없음 → `from Windforce.Modeling import ...` 실패 | 파일 생성 |
| `Windforce/Modeling/LSTMPipeline.py` | `SequenceDataset`, `_LSTMModel` 클래스 누락 | 두 클래스 추가 |
| `Windforce/Modeling/LSTMPipeline.py` | `self.model = LSTMPipeline(...)` — 자기 자신을 재귀 생성하는 버그 | `self.model = _LSTMModel(...)` 로 수정 |
| `Windforce/Modeling/LSTMPipeline.py` | `Dataset`, `ScoreLossFunction` import 누락 | `from torch.utils.data import Dataset`, `ScoreLossFunction` import 추가 |
| `Windforce/Modeling/GroupExperimentRunner.py` | `pd`, `MinMaxScaler`, `RATED_CAPACITY_KW`, `LSTMPipeline`, `BaselineModels`, `EvaluationMetrics` import 누락 | 상단에 전부 추가 |
| `Windforce/Modeling/GroupExperimentRunner.py` | `WindforceDatasetBuilder.featureCols()` — 외부 정의 클래스에 의존 | 독립적인 `_featureCols()` 유틸 함수로 대체 |
| `Windforce/Modeling/GroupExperimentRunner.py` | `WindforceDatasetBuilder` 타입 어노테이션 — 미정의 클래스 참조 | `datasetBuilder` 로 어노테이션 제거 (duck typing) |
| `baseline_3/4/5.ipynb` | **`turbine_meta["group"]`에 정수 1/2/3을 넘겨 `compute_group_weight()`의 문자열 비교가 전부 실패 → IDW 가중치 0 행렬 → LDAPS·GFS 그룹 피처가 통째로 0.0** | `baseline5_howto.ipynb`에서 `"kpx_group_" + N` 문자열 키로 수정. 패키지 차원 가드는 `MD/baseline5_howto_next.md` 카드 A 참조 |

---

## 8. 노트북별 역할 요약

| 노트북 | 목적 | 핵심 내용 |
|--------|------|----------|
| `baseline_1.ipynb` | EDA + 프로토타입 | 절차적 스타일. `FeatureEngineer`, `LDAPSFeatureEngineer`, `GFSFeatureEngineer` 인라인 정의. 파워 커브 시각화, 피어슨 상관계수 검증 |
| `baseline_2.ipynb` | OOP 전환 | Windforce 패키지 import. `WindforceDatasetBuilder` 골격 정의. `GroupExperimentRunner` 실험 실행 |
| `baseline_3.ipynb` | 완성 파이프라인 | 격자→IDW 그룹 집계, 터빈 메타 파싱, 완전한 `WindforceDatasetBuilder.build()`, 평가기간 예측 + 제출 파일 저장까지 end-to-end 동작 |
| `baseline_5.ipynb` | 손실함수·피처정렬 개선 | `ScoreLossFunction` 워밍업·steepness 완화(A), 학습·평가 피처 컬럼명 교집합 정렬(B) |
| `baseline5_howto.ipynb` | 피처 분석 기반 개선 | **이 노트북 사용 권장.** IDW group 키 버그 수정, `transformForecastFeature` 추가, LightGBM + SHAP 피처 선별. 제출 가능 모델 총점 0.4930 → **0.6592** |

---

## 9. 빠른 시작 가이드

```python
import sys
sys.path.insert(0, "/Users/ksydata/WINDFORCE")

from Windforce import (
    WindforceDataLoader,
    LDAPSFeatureEngineer, GFSFeatureEngineer,
    EvaluationMetrics, RATED_CAPACITY_KW, TIME_STEP_HOURS,
    ScoreLossFunction,
)
from Windforce.Modeling import LSTMPipeline, GroupExperimentRunner, BaselineModels

loader = WindforceDataLoader()
loader.check_paths()

ldaps_fe = LDAPSFeatureEngineer()
ldaps_grid  = ldaps_fe.transform(loader["ldaps_train"])
ldaps_group = ldaps_fe.transformToGroupIDW(ldaps_grid, turbine_meta)

# → baseline_3.ipynb 참조
```
