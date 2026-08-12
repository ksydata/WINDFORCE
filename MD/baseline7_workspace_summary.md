# WINDFORCE 워크스페이스 분석 — baseline_7 기준 현황 요약

> 작성일: 2026-08-12
> 목적: 신규 아이디어(VMD 분해, 리드타임 차등 모델링) 검증에 앞서 현재 파이프라인·베이스라인 히스토리·
> 액션플랜 방향성을 한 문서에 정리한다. `MD/baseline7_idea_evaluation.md`, 개정된
> `MD/baseline7_howto_next.md`가 이 문서의 사실관계를 전제로 한다.

---

## 1. 과업 정의

| 항목 | 내용 |
|---|---|
| 목표 | 2025년 전체(8,760시간) KPX 3개 그룹 시간별 발전량(kWh) 예측 |
| 운영 제약 | **day-ahead 일괄 예측.** 예보는 매일 09:00 초기화, 13:00부터 사용 가능. 다음날 01:00~그다음날 00:00 24시간을 한 번에 예측해야 하며, 예측 대상 구간의 실제 발전량·SCADA는 절대 쓸 수 없음 |
| 평가지표 | `총점 = 0.5×(1-NMAE) + 0.5×FICR` (3그룹 평균), 둘 다 클수록/작을수록 좋은 방향 통일 |
| KPX 그룹 | 그룹1·2: VESTAS V126, 21.6MW(21,600kWh/h) · 그룹3: UNISON U136, 21.0MW(21,000kWh/h) |
| 그룹3 특이사항 | 2022년 라벨 없음(2023~2024만 제공) → 다른 그룹보다 학습 데이터 1년 적음 |

### 지표 수식 (`Windforce/EvaluationMetrics.py`)

```
NMAE_g   = mean(|pred - actual| / capacity_g)
UnitPrice_t = 4원 (오차율 ≤6%) / 3원 (6~8%) / 0원 (>8%)
FICR_g   = Σ(UnitPrice_t × actual_t) / Σ(4 × actual_t)     ← 실제 발전량 가중, 계단함수
총점      = 0.5×(1 - mean(NMAE)) + 0.5×mean(FICR)
```

FICR은 **시간별 오차율의 계단형 분류**(≤6/6~8/>8%)에 좌우되므로, 평균 오차(NMAE)를 조금
줄이는 것보다 "6% 밴드 안에 드는 시간의 비율"을 늘리는 것이 총점에 직접적이다. 이 성질이
아래 3절의 현재 병목 진단과 4절의 두 신규 아이디어 평가 모두의 전제가 된다.

---

## 2. 데이터·전처리 구조

### 2-1. 입력 데이터 (`MD/data_description.md`)

| 파일 | 내용 | 규모 |
|---|---|---|
| `ldaps_train/test.csv` | 1.5km 해상도 16격자 기상예보 | 학습 420,864행 / 평가 140,160행 |
| `gfs_train/test.csv` | 0.25° 해상도 9격자 기상예보 (고도 7단) | 학습 236,736행 / 평가 78,840행 |
| `train_labels.csv` | 그룹별 실제 발전량(kWh), 1시간 단위 | 26,304행 |
| `scada_*_train.csv` | 터빈별 10분 단위 SCADA(파워커브 확인용, 현재 모델 입력에는 미사용) | VESTAS 157,819 / UNISON 105,264행 |

### 2-2. 예보 시간 구조 — **모든 카드 설계의 핵심 전제**

- 예보는 **매일 09:00 KST 한 번** 초기화되고 **13:00부터 사용 가능**하다고 간주한다.
- 한 번의 `data_available_kst_dtm`(전날 13:00)에 **다음날 01:00 ~ 그다음날 00:00까지 24개
  `forecast_kst_dtm`**이 대응된다.
- `lead_hour = (forecast_kst_dtm - data_available_kst_dtm)/3600h`는 항상 **12~35 범위**.
- 이 규칙이 하루도 예외 없이 지켜지므로, **`lead_hour`는 `forecast_kst_dtm`의 시각(hour-of-day)의
  단순 아핀 변환(`lead_hour = hour_of_day + 11`)과 완전히 동일한 정보다.** 캐시된
  `prep/baseline7_dataset_train_group1.csv.gz`로 실측한 결과 상관계수 `1.0`, 시각별 `lead_hour`
  고유값 개수 `1`(전 구간 예외 없음)을 확인했다. 이 사실은 4절 아이디어 2 평가에서 핵심적으로 쓰인다.

### 2-3. 전처리 파이프라인 (`Windforce/Preprocessing/`)

```
LDAPS(16격자)/GFS(9격자) 원본
      ↓ transform() — 컬럼 통일, 물리한계 플래그, 파생변수(공기밀도·풍력에너지밀도·풍향 sin/cos)
      ↓ transformToGroupIDW(turbine_meta) — 터빈 위치 기준 역거리가중(IDW, k=4, power=2)으로 16·9격자 → KPX 3그룹
      ↓ transformForecastFeature() — 1h/3h/6h 변화량·램프, GFS 허브풍속 외삽(V_hub = V100×(117/100)^α)
그룹별 시간순 표 (한 행 = 한 그룹 × 한 시각)
```

- `baseline_7`은 `PreprocessedDatasetBuilder`로 이 결과를 `prep/baseline7_dataset_{train,test}_group{1,2,3}.csv.gz`에
  캐시한다(1회 약 52초, 이후 캐시 재사용).
- IDW 결과에 `hour_sin_ldaps/hour_cos_ldaps/target_hour_ldaps`, GFS 쪽도 동일한 4종이 **중복해서**
  존재한다(공간가중은 지정 피처만 통과시키므로 시간 메타를 그룹 단위에서 별도로 복원).
- 그룹별 최종 피처 수는 127개(`LGBM_full`), SHAP 선별 후 82~86개(`*_selected`).

---

## 3. 베이스라인 히스토리 (1 → 7)

`MD/baseline3_6_modeling_comparison.md` 기준 요약.

```
baseline_1 (EDA, 절차적)
      ↓
baseline_2 (OOP 전환, Windforce 패키지 도입)
      ↓
baseline_3/4 ── Persistence·SVR·LSTM 비교, 사실상 같은 모델링(4는 3의 반복본)
      ↓
baseline_5 ── 공통 LSTMPipeline 명시적 학습 설정 + 결과 CSV 기록으로 절차 정리
      ↓
baseline_5_howto ── IDW group 키 버그(정수 vs 문자열) 수정, LightGBM·SHAP 도입 → 총점 0.493→0.659
      ↓
baseline_6 ── LSTM v1(기본)/v2(LayerNorm+Dropout)/v3(Attention) × 피처 4종(all/corr/pca/tree) 비교 실험
      ↓
baseline_7 ── howto의 LightGBM/SHAP + 6의 LSTM v1/v2/v3를 동일 SHAP 피처·동일 검증 조건으로 통합, 제출 파이프라인 완성
```

**현재 주력 노트북은 `baseline_7.ipynb`.** IDW 버그, Step 8 피처 정렬 버그, 첫 24시간 누락
버그 등 `MD/improvement.md`가 지적한 항목은 모두 반영된 상태다(`turbine_meta["group"]` 문자열
정규화 + 0 행렬 가드, 컬럼명 기준 교집합 정렬, 예측 길이 assert).

### baseline_7 구조 (26셀)

| Step | 내용 |
|---|---|
| 0 | 커널·환경 가드 (`.venv` 경로 확인), 재현 시드, `torch.set_num_threads(8)` |
| 1 | `PreprocessedDatasetBuilder.buildAll()` — IDW 전처리 캐시, 0 행렬 가드 |
| 2 | `LGBM_full`/`LGBM_selected` 학습, null importance+상관 중복 기반 SHAP 피처 선별(`FeatureSelector`) |
| 3 | 동일 SHAP 피처로 `LSTM_v1/v2/v3` 학습, 조기종료 기준을 **공식 총점**으로 통일(`OfficialScoreMonitor`) |
| 4 | 그룹·모델별 결과 집계, 그룹별 검증 총점 최고 모델 선택(`best_by_group`) |
| 5 | 검증에서 정한 설정으로 전체(2022~2024) 재학습 → 2025년 8,760시간 예측 → 스키마·물리범위 검증 → 제출 |

---

## 4. 현재 위치 (수치)

`baseline_7_results.csv`/`baseline_7_summary.csv` 기준, 그룹별 **제출 가능 모델 중 검증 총점 최고**:

| 그룹 | 선택 모델 | NMAE | FICR | 총점 |
|---|---|---|---|---|
| 1 | `LGBM_selected` | 0.0804 | 0.3914 | 0.6555 |
| 2 | `LSTM_v2_shap_selected` | 0.0860 | 0.4613 | 0.6877 |
| 3 | `LSTM_v1_shap_selected` | 0.0819 | 0.3985 | 0.6583 |
| — | **실제 제출 기대 총점(3그룹 평균)** | 0.0828 | 0.4171 | **0.6672** |
| — | `Persistence_oracle_lag1`(비교 기준, 제출 불가) | 0.0467 | 0.6398 | 0.7966 |

Persistence oracle과의 격차(0.1294) 중 **NMAE 기여 14% vs FICR 기여 86%** — 남은 격차의
거의 전부가 FICR이다. 이 결론이 기존 액션플랜(`baseline7_howto_next.md` 카드 A~F)과 이번에
검토하는 두 아이디어 모두의 공통 배경이다.

---

## 5. 기존 액션플랜(`baseline7_howto_next.md`) 방향성 요약

FICR 개선에 집중한 6장 카드 체계, 우선순위 순:

| 카드 | 내용 | 상태 |
|---|---|---|
| A | 오차율 밴드(≤6/6~8/>8%) 진단 + 예측값 저장소 확보 | 미실행 (다른 카드의 전제) |
| B | 사후 보정(affine calibration, `a·pred+b`) | 미실행, 저비용 |
| C | LightGBM용 FICR-aware 커스텀 목적함수(`fobj`/`hess`) | 미실행, 개선 폭 최대 예상 |
| D | LSTM `loss_k`/`regression_weight` 그룹별 재탐색 | 미실행 |
| E | `lead_hour` 오차 진단(보조) | 미실행 |
| F | 그룹별 최종 채택 + 제출 반영 | 미실행 |

카드 H/I/K(커널 고정, LSTM 실행시간 87분→34분, 리포 위생)는 이미 `baseline_7.ipynb`에
반영 완료 상태다(`EXPECTED_VENV` 가드, `torch.set_num_threads(8)`, `MODEL_PARAMS["batch_size"]=256`).

이번 문서(`baseline7_idea_evaluation.md`)와 개정된 `baseline7_howto_next.md`는 **이 카드 체계를
대체하지 않고**, VMD 분해(신규 카드 G)와 리드타임/시간대 차등 모델링(신규 카드 H)을
**같은 우선순위 언어로 추가**한다.

---

## 6. 관련 파일

- `BASELINE/baseline_7.ipynb` — 현재 주력 노트북
- `BASELINE/baseline_7_results.csv`, `baseline_7_summary.csv` — 검증 결과
- `Windforce/EvaluationMetrics.py`, `Windforce/ScoreLossFunction.py` — 공식 지표·미분가능 손실
- `Windforce/Preprocessing/LDAPSFeatureEngineer.py`, `GFSFeatureEngineer.py` — IDW·리드타임 피처 생성
- `Windforce/Modeling/LSTMPipelineVersions.py`, `FeatureVariants.py` — v1/v2/v3 LSTM, 피처 변형
- `MD/data_description.md`, `MD/coding_convention.md`, `MD/baseline3_6_modeling_comparison.md`
- `MD/baseline7_howto_next.md` — FICR 개선 액션플랜 (이번에 카드 G/H로 확장)
