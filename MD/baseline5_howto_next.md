# baseline5_howto 후속 개선 작업 지시서

> 작성일: 2026-08-09 · 최종 수정: 2026-08-12 17:35 (카드 H~K 추가 — 문서 하단)
> 선행 산출물: `BASELINE/baseline5_howto.ipynb`
> 이 문서는 **그대로 구현 가능한 작업 카드** 형식이다. 각 카드는 독립적으로 착수할 수 있고,
> 코드 스타일은 반드시 [`MD/coding_convention.md`](coding_convention.md)를 따른다.

---

## 0. 현재 위치 (baseline5_howto 실행 결과)

검증 조건: 그룹별 시간순 마지막 20% hold-out, `testRatio = 0.2` (baseline 5와 동일)

| 모델 | 평균 NMAE | 평균 FICR | 공식 총점 | 제출 가능 |
|---|---|---|---|---|
| `Persistence_oracle_lag1` | 0.0468 | 0.6389 | **0.7961** | ❌ (직전 **실제값** 사용) |
| **`LGBM_selected` (86/86/82 피처)** | 0.0770 | 0.3955 | **0.6592** | ✅ |
| `LGBM_full` (127 피처) | 0.0775 | 0.3903 | 0.6564 | ✅ |
| baseline 5 `LSTM` | 0.1984 | 0.1844 | 0.4930 | ✅ |
| baseline 5 `SVR` | 0.2280 | 0.0828 | 0.4274 | ✅ |

**해결된 것**
- `turbine_meta["group"]` 정수 → `"kpx_group_N"` 문자열 수정으로 IDW 가중치 0 행렬 문제 제거
- `transformForecastFeature()` 램프·변화량 8종 추가
- 제출 가능 모델 기준 총점 0.4930 → 0.6592 (**+33.7%**)

**남은 것 — 이 문서가 다루는 범위**
- NMAE는 0.077까지 내려왔지만 **FICR이 0.396에 머문다.** 총점의 절반이 FICR인데
  Persistence(0.639)와의 격차 대부분이 여기서 발생한다. → 카드 B가 최우선.
- 검증이 단일 hold-out 1회뿐이라 계절 편향을 걸러내지 못한다. → 카드 C.
- LSTM은 **0 행렬로 학습된 기록**이라 0.4930이라는 숫자 자체가 무의미하다. → 카드 D.

---

## 카드 A. [P0] 나머지 노트북·패키지의 IDW group 키 버그 차단

### 문제
`Windforce/utils/spatial_utils.py`의 `compute_group_weight()`는 `turbine_meta["group"]`을
`KPX_GROUPS = ("kpx_group_1", ...)` 문자열과 비교한다.

```python
for j, group in enumerate(KPX_GROUPS):
    selected = (meta["group"] == group).to_numpy()   # 정수 1/2/3을 넘기면 전부 False
```

`baseline_3.ipynb` · `baseline_4.ipynb` · `baseline_5.ipynb`는 모두 `kpx_info["KPX그룹"]`(정수)을
그대로 넘기므로 **가중치 행렬이 전부 0**이 되고, LDAPS·GFS 그룹 피처가 통째로 `0.0`이 된다.
버그가 조용히 통과하기 때문에 이후 모든 실험 결과가 무효가 된다.

### 구현 단계

1. `Windforce/utils/spatial_utils.py`의 `compute_group_weight()` 시작부에 **정규화 + 가드**를 넣는다.

   ```python
   meta = turbine_meta.reset_index(drop = True).copy()
   # 원본 DataFrame 변형 방지

   if not meta["group"].astype(str).str.startswith("kpx_group_").all():
       meta["group"] = "kpx_group_" + meta["group"].astype(int).astype(str)
       # 정수 그룹 번호(1/2/3)를 KPX_GROUPS와 같은 문자열 키로 자동 승격
       # 이 변환이 없으면 매칭이 전부 실패해 가중치가 0 행렬이 된다

   unknown = sorted(set(meta["group"]) - set(KPX_GROUPS))
   if unknown:
       raise ValueError(f"turbine_meta의 group 값이 KPX_GROUPS에 없습니다: {unknown}")
       # 오타·형식 불일치를 조용히 넘기지 않고 즉시 실패시킨다
   ```

2. 반환 직전에 **영행렬 가드**를 추가한다.

   ```python
   if not np.isfinite(weight_group).all() or weight_group.sum() == 0:
       raise PreprocessingError("IDW 가중치가 전부 0입니다. turbine_meta의 group 키를 확인하세요")
   ```

3. `baseline_3/4/5.ipynb`의 `turbine_meta` 생성 셀에 아래 한 줄을 추가한다.

   ```python
   turbine_meta["group"] = "kpx_group_" + turbine_meta["group"].astype(int).astype(str)
   ```

### 검증 기준
- `pytest` 없이도 확인 가능: 정수 group을 넣은 `turbine_meta`로 `compute_group_weight()`를 호출했을 때
  `weight_group.sum(axis = 1)`이 `[1, 1, 1]`이어야 한다.
- `transformToGroupIDW()` 결과에서 `ws10` 컬럼의 `max()`가 0보다 커야 한다.

### 예상 효과
정확도 개선은 아니지만 **이 수정 없이는 다른 모든 카드의 실험 결과를 신뢰할 수 없다.**

---

## 카드 B. [P1] FICR 최적화 — 총점의 절반을 차지하는 미해결 영역

### 문제
총점 = `0.5 × (1-NMAE) + 0.5 × FICR` 인데, 현재 모델은 L1(MAE) 최소화로 학습한다.
그러나 FICR은 **시간별 오차율이 6% 이하인 시간에만 만점(4원)** 을 주는 계단형 보상이다.

| 오차율 구간 | 단가 | 의미 |
|---|---|---|
| ≤ 6% | 4원 | 만점 |
| 6~8% | 3원 | 부분 점수 |
| > 8% | 0원 | **0점** |

평균 오차를 조금 줄이는 것보다 **"6% 밴드 안에 들어가는 시간 수"를 늘리는 것**이 총점에 직접적이다.
현재 그룹1 기준 시간별 오차율 분포를 보면 FICR 0.39는 대략 40% 남짓의 시간만 밴드 안에 있다는 뜻이다.

### 구현 단계

1. **진단 먼저** — `BASELINE/baseline5_howto.ipynb`에 셀을 추가해 시간별 오차율 히스토그램을 그린다.

   ```python
   def plotErrorRatioBand(pred: np.ndarray, actual: np.ndarray, group: int) -> pd.Series:
       """시간별 오차율의 FICR 구간 분포를 계산·시각화하는 함수

       Returns:
           - {"≤6%": 비율, "6~8%": 비율, ">8%": 비율} Series
       """
       ratio = np.abs(pred - actual) / RATED_CAPACITY_KW[group]
       band = pd.cut(ratio, bins = [-np.inf, 0.06, 0.08, np.inf],
                     labels = ["≤6%", "6~8%", ">8%"])
       share = band.value_counts(normalize = True).sort_index()
       # 구간별 시간 비율. ">8%" 비율이 곧 FICR 손실의 크기다
       ...
   ```

2. **사후 보정(post-hoc calibration)을 먼저 시도한다.** 재학습 없이 예측값에 스칼라 변환만 적용해
   검증 구간 총점을 최대화하는 파라미터를 찾는다. 비용이 거의 없고 효과 확인이 빠르다.

   ```python
   class FicrCalibrator:
       """예측값에 단조 보정을 적용해 검증 총점을 최대화하는 클래스

       Logic:
           - pred_calibrated = clip(a * pred + b, 0, capacity_kw)
           - (a, b) 격자탐색으로 검증 구간 총점이 최대인 조합을 찾는다
           - 반드시 **검증 구간에서만** 탐색하고, 평가 기간에는 찾은 값을 그대로 적용한다
       """
       def fit(self, pred: np.ndarray, actual: np.ndarray, group: int) -> "FicrCalibrator":
           ...
       def transform(self, pred: np.ndarray) -> np.ndarray:
           ...
   ```

   - 탐색 범위 권장: `a ∈ [0.85, 1.15]` 21점, `b ∈ [-0.02, 0.02] × capacity_kw` 21점.
   - 그룹별로 따로 적합한다. 설비용량과 오차 분포가 다르다.

3. **커스텀 목적함수** — 보정으로 부족하면 LightGBM에 FICR 근사 손실을 직접 넣는다.
   `Windforce/ScoreLossFunction.py`의 시그모이드 근사를 numpy로 옮겨 1·2차 도함수를 제공한다.

   ```python
   def makeFicrObjective(capacity_kw: float, k: float = 40.0, regression_weight: float = 0.7):
       """LightGBM용 (grad, hess) 반환 목적함수를 만드는 팩토리 함수

       Logic:
           - loss = w × |e| + (1-w) × (1 - softFICR(e))
           - softPrice = 4 - 1×sigmoid(k(r-0.06)) - 3×sigmoid(k(r-0.08)), r = |e|/capacity
           - hess가 0이 되면 LightGBM이 분할을 멈추므로 1e-6 하한을 둔다
       """
   ```

   - `regression_weight`는 0.5 / 0.7 / 0.9를 비교한다. FICR 항만 쓰면 초기 포화로 학습이 멈춘다
     (`improvement.md` 2-1과 같은 함정).

### 검증 기준
- 진단 셀에서 `">8%"` 구간 비율이 **줄어드는지**를 1차 지표로 본다.
- 최종 판단은 3그룹 평균 공식 총점. 현재 `0.6592`를 넘지 못하면 채택하지 않는다.
- 보정 파라미터는 검증 구간에서만 탐색했는지 반드시 확인한다 (평가 구간 정보 유입 금지).

### 예상 효과
FICR 0.396 → 0.45~0.50 도달 시 총점 **0.68~0.71**. 이 문서에서 기대 효과가 가장 크다.

---

## 카드 C. [P1] 시계열 교차검증(rolling-origin) + 하이퍼파라미터 탐색

### 문제
현재는 마지막 20% 단일 구간 검증뿐이다. 그룹3 검증 구간은 2024년 하반기 3,508시간에 불과해
계절 편향이 그대로 점수에 반영된다. 하이퍼파라미터도 손으로 고른 1세트만 썼다.

### 구현 단계

1. `Windforce/Modeling/` 아래 `TimeSeriesValidator.py`를 새로 만든다 (파일명 = 클래스명).

   ```python
   class TimeSeriesValidator:
       """rolling-origin 방식으로 시계열 교차검증을 수행하는 클래스

       사용 순서:
           validator = TimeSeriesValidator(n_splits = 4, valid_months = 3)
           folds = validator.split(df, time_col = "forecast_kst_dtm")
           scores = validator.evaluate(df, feature_cols, target_col, group, fitFn)

       Logic:
           - fold k의 학습 구간은 항상 검증 구간보다 **과거**다. 랜덤 셔플·미래 구간 학습 금지
           - 검증 구간을 3개월 단위로 잡아 계절(겨울/봄/여름/가을)을 모두 포함시킨다
       """
       def split(self, df: pd.DataFrame, time_col: str = "forecast_kst_dtm") -> list[tuple[np.ndarray, np.ndarray]]:
           ...
   ```

2. 각 fold에서 `EvaluationMetrics.summarize()`로 nmae/ficr/score를 계산하고
   **fold 평균과 표준편차를 함께** 보고한다. 표준편차가 크면 그 설정은 불안정한 것이다.

3. 하이퍼파라미터 탐색은 랜덤 서치 30~50회로 충분하다. 탐색 범위 권장값:

   | 파라미터 | 범위 |
   |---|---|
   | `num_leaves` | 31 / 63 / 127 / 255 |
   | `learning_rate` | 0.02 ~ 0.08 (로그 균등) |
   | `min_child_samples` | 20 / 40 / 80 / 160 |
   | `feature_fraction` | 0.6 ~ 1.0 |
   | `bagging_fraction` | 0.6 ~ 1.0 |
   | `lambda_l2` | 0 ~ 10 (로그 균등) |

4. 결과를 `BASELINE/baseline5_howto_cv.csv`에 저장한다
   (컬럼: `group, fold, params_json, nmae, ficr, score`).

### 검증 기준
- fold 간 총점 표준편차가 0.03 미만이어야 "안정적"으로 본다.
- 단일 hold-out 점수와 CV 평균이 0.05 이상 벌어지면 hold-out 결과를 신뢰하지 않는다.

---

## 카드 D. [P2] LSTM 재평가 — 공정한 비교 복원

### 문제
baseline 5의 LSTM 점수 `0.4930`은 **전부 0인 입력**으로 학습한 결과다. 즉 "LSTM이 나쁘다"는 결론을
내릴 근거가 아직 없다. 카드 A 수정 후 동일 조건으로 다시 측정해야 한다.

### 구현 단계

1. `prep/dataset_train_group{g}.csv.gz`(전처리 완료본)를 입력으로 `LSTMPipeline`을 학습한다.
   피처는 `prep/selected_features.csv`의 그룹별 선별 결과를 그대로 쓴다.
2. 스케일링은 `MinMaxScaler().fit(X_train)` — **학습 구간에만 fit**.
3. 파라미터는 baseline 5와 동일하게 두고 시작한다:
   `epochs=80, warmup_epochs=15, loss_k=40.0, regression_weight=0.75, patience=12`.
4. 예측·정답 정렬 시 `y_valid[seq_len:]`로 앞쪽을 잘라내는 것을 잊지 않는다.

### 검증 기준
- LSTM 총점이 `LGBM_selected`(0.6592)를 넘으면 제출 후보를 교체한다.
- 넘지 못하더라도 **카드 E(앙상블)의 후보**로는 가치가 있으므로 결과를 폐기하지 않는다.

---

## 카드 E. [P2] 그룹별 가중 앙상블

### 구현 단계

1. 카드 C의 CV fold별 **out-of-fold 예측**을 그룹별로 모은다 (`LGBM`, `LSTM`, 그 외 후보).
2. 가중치는 단순 격자탐색으로 찾는다 — 모델이 2~3개뿐이라 최적화 라이브러리가 필요 없다.

   ```python
   for w in np.arange(0, 1.01, 0.05):
       blended = w * pred_lgbm + (1 - w) * pred_lstm
       score = metrics.summarize(blended, actual, group)["score"]
       # 그룹마다 최적 w가 다를 수 있으므로 그룹별로 따로 찾는다
   ```

3. 가중치는 **OOF 예측에서만** 탐색한다. 검증 구간에서 직접 고르면 그 구간에 과적합된다.

### 검증 기준
- 앙상블 총점이 단일 최고 모델보다 **모든 그룹에서** 높거나 같아야 채택한다.

---

## 카드 F. [P2] SCADA 파워커브를 물리 사전정보 피처로 추가

### 배경
`baseline_1~5`의 Step 5에서 확인한 S자 파워커브(풍속 → 발전량)는 터빈 모델별로 고정된 물리 특성이다.
현재 모델은 이 관계를 데이터에서 처음부터 학습하고 있는데, 곡선을 미리 피팅해 피처로 주면
적은 데이터로도 일반화가 쉬워진다.

### 구현 단계

1. `Windforce/Preprocessing/SCADAFeatureEngineer.py`의 `transformToTurbineHourly()`로
   터빈별 시간 집계값을 만든다.
2. 제작사·모델별(V126 / U136)로 풍속 0.5 m/s 구간 중앙값을 구해 **경험적 파워커브 테이블**을 만든다.
   ```python
   curve = (scada_hourly.assign(ws_bin = (scada_hourly["ws"] // 0.5) * 0.5)
            .groupby(["model", "ws_bin"], observed = True)["power_kwh"].median())
   # 평균이 아닌 중앙값을 쓰는 이유: 정비·고장 구간의 0 출력이 평균을 끌어내린다
   ```
3. 예보 허브풍속(`gfs_ws_hub`)을 이 테이블로 조회해 `expected_power_kwh` 피처를 만들고,
   그룹 설비용량에 맞춰 터빈 수를 곱한다.
4. 결측 구간(컷인 미만·컷아웃 초과)은 0으로 채운다.

### 검증 기준
- SHAP 중요도 상위 10위 안에 `expected_power_kwh`가 들어오는지 확인한다.
- 들어오지 못하면 곡선 피팅 품질(이상치 제거)을 먼저 점검한다.

---

## 카드 G. [P3] lead_hour 구조 활용

`data_description.md` 기준 실제 운영 조건은 **전날 13시 예보로 다음날 24시간을 일괄 예측**하는
day-ahead 구조이며, `lead_hour`는 12~35 범위를 가진다. 리드타임이 길수록 예보 오차가 커지므로:

1. `lead_hour`를 구간화한 피처(`lead_bucket` = 단기 12-19h / 중기 20-27h / 장기 28-35h)를 추가한다.
2. 리드타임 구간별 잔차 분포를 확인하고, 편향이 뚜렷하면 구간별 보정계수를 적용한다.
3. 효과가 크면 구간별 모델 분리까지 검토한다 (데이터가 1/3로 줄어드는 비용을 감수할 만한지 판단).

---

## 작업 순서 요약

```
카드 A (버그 차단)  ← 다른 모든 실험의 전제. 가장 먼저.
      ↓
카드 B (FICR 최적화)  ← 총점 개선 폭이 가장 큼
      ↓
카드 C (시계열 CV + 튜닝)  ← B의 개선이 진짜인지 검증
      ↓
카드 D (LSTM 재평가) → 카드 E (앙상블)
      ↓
카드 F (파워커브) · 카드 G (리드타임)  ← 추가 신호 확보
```

## 공통 준수 사항

- 코드 스타일: [`MD/coding_convention.md`](coding_convention.md) — 특히 **설명 주석은 코드 줄 아래**,
  단위·shape 명시, `df.copy()` 선행, 시간순 분할.
- 새 실험은 반드시 `BASELINE/baseline5_howto_results.csv`와 같은 스키마
  (`group, model_name, n_features, nmae, ficr, score, mae, rmse`)로 저장해 리더보드에서 재집계 가능하게 한다.
- 제출 파일은 8,760행 · 컬럼 순서 일치 · NaN 없음 · 음수 없음 · 설비용량 이하를 `assert`로 검증한다.
- **Persistence는 비교 기준으로만 쓴다.** 직전 실제값을 사용하므로 2025년 일괄 제출에는 사용할 수 없다.

---
---

# 추가 지시 (2026-08-12 17:35 작성)

> 대상: `BASELINE/baseline_7.ipynb` 실행 환경 진단에서 나온 항목.
> 위 카드 A~G(모델 성능)와 **성격이 다르다.** 여기 카드 H~K는 **환경·실행시간·위생** 문제이며
> 점수를 올리지 않는다. 대신 **카드 B~G를 돌릴 수 있게 만드는 선행 조건**이다.
>
> ### 이 문서를 받는 작업자에게 (Sonnet 실행 전제)
>
> 1. **아래 수치는 이미 실측했다. 다시 측정하지 마라.** 벤치마크 재실행은 시간 낭비다.
> 2. **파일 위치는 줄 번호가 아니라 문자열로 찾아라.** 노트북은 실행하면 셀 인덱스가 밀린다.
>    편집 직전에 반드시 파일을 다시 읽고, **모든 편집을 끝낸 뒤에** 실행한다.
> 3. **새 파일을 만들지 마라.** 각 카드에 명시된 파일만 수정한다. 새 헬퍼 모듈 금지.
> 4. 카드 하나를 끝낼 때마다 그 카드의 "검증" 명령을 실제로 돌리고 **출력을 붙여서 보고**한다.
>    "수정했습니다"만으로는 완료가 아니다.
> 5. 판단이 필요한 지점은 카드 안에 **이미 결정해 두었다.** 대안을 탐색하지 말고 적힌 대로 한다.
>    적힌 대로 했는데 검증이 실패하면 거기서 멈추고 보고한다.

## 실측 근거 (2026-08-12, 로컬 Windows 11 · 12코어 · torch 2.13.0+cpu)

`baseline_7.ipynb` 전체 실행시간 분해:

| 셀 | 내용 | 실측 시간 | 비중 |
|---|---|---|---|
| Step 1 전처리 (`prep_builder.buildAll()`) | LDAPS+GFS IDW | **52초** (1회성, 이후 캐시) | 1% |
| Step 2 LightGBM 9회 + SHAP 6회 | 피처 선별 | **~0.5분** | 0.5% |
| **Step 3 LSTM 9개 (3그룹 × v1/v2/v3)** | 학습 | **~87분** | **98%** |

LSTM 세부 — **연산량 병목이 아니다.** 피처를 45개 → 127개로 늘려도 에폭 시간이 6.7초 → 7.1초로 거의 같다.
실제 원인은 `batch_size = 64`(에폭당 328배치)와 12스레드 조합이다. 배치·스레드 격자 실측:

| threads | batch_size | 에폭 | LSTM 9개 × 80에폭 |
|---|---|---|---|
| **12** | **64** | **8.18초** | **87분** ← 현재 기본값 |
| 8 | 64 | 5.50초 | 59분 |
| 4 | 512 | 4.43초 | 47분 |
| **8** | **256** | **3.18초** | **34분** ← 권장 |

---

## 카드 H. [P0] Jupyter 커널 인터프리터 고정

### 문제

`.venv/share/jupyter/kernels/python3/kernel.json`의 `argv[0]`이 절대경로가 아니라 맨 이름이었다.

```json
"argv": ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
```

Jupyter가 커널을 띄울 때 이 `python`을 PATH로 해석한다. `.venv\Scripts`가 앞에 없으면
전역 파이썬(`C:\Users\<user>\AppData\Local\Programs\Python\Python314`)이 잡힌다. 그 전역 파이썬에는
`ipykernel`·`pandas`·`numpy`·`torch`·`sklearn`이 **있어서 커널은 정상적으로 뜨고**,
`lightgbm`·`shap`만 없어서 Step 0 임포트 셀에서 `ModuleNotFoundError`로 죽는다.
"커널이 죽었다"가 아니라 "패키지만 없다"로 보이기 때문에 원인 추적이 오래 걸린다.

코랩에서 되는 이유는 인터프리터가 하나뿐이라 이 분기 자체가 없기 때문이다. 로컬만의 문제다.

### 상태

**`kernel.json`은 2026-08-12 17:28에 이미 수정·검증 완료했다.** 아래는 재발 방지용 잔여 작업이다.
`.venv`를 다시 만들면 `argv`가 맨 이름으로 되돌아가므로 노트북 자체에 가드가 있어야 한다.

### 구현 단계

1. `BASELINE/baseline_7.ipynb`에서 **문자열 `import lightgbm as lgb`가 있는 셀을 찾는다**(인덱스로 찾지 말 것).
   그 셀의 **맨 위**, 모든 서드파티 import보다 **앞**에 아래 블록을 넣는다.

   ```python
   import sys
   from pathlib import Path

   EXPECTED_VENV = "WINDFORCE/.venv"
   # 이 노트북이 돌아야 하는 가상환경 경로 조각

   if EXPECTED_VENV not in Path(sys.executable).as_posix():
       raise RuntimeError(
           f"잘못된 커널입니다: {sys.executable}\n"
           f"VS Code 우상단 커널 선택에서 WINDFORCE/.venv를 고르세요.\n"
           f"주피터랩이면 kernel.json의 argv[0]이 절대경로인지 확인하세요."
       )
       # 전역 파이썬으로 뜬 커널을 임포트 실패 전에 명확한 메시지로 차단한다
   ```

2. 같은 블록 바로 아래에, 임포트가 끝난 뒤 실행 로그용으로 한 줄 추가한다.

   ```python
   print(f"kernel: {sys.executable}")
   # 어느 인터프리터로 돌았는지 실행 기록에 남긴다
   ```

3. **`BASELINE/baseline_6.ipynb`와 `BASELINE/baseline_5_howto.ipynb`에도 같은 블록을 넣는다.**
   세 노트북 모두 같은 커널을 쓴다.

### 검증

```bash
# (1) kernel.json이 절대경로인지
python -c "import json;print(json.load(open(r'.venv/share/jupyter/kernels/python3/kernel.json'))['argv'][0])"
# 기대: D:\workspaces\WINDFORCE\.venv\Scripts\python.exe   (맨 이름 'python'이면 실패)

# (2) 그 커널이 실제로 lightgbm/shap을 잡는지
.venv/Scripts/python.exe -c "import lightgbm,shap;print(lightgbm.__version__,shap.__version__)"
# 기대: 4.7.0 0.52.0
```

### 하지 말 것

- 전역 파이썬에 `lightgbm`·`shap`을 설치해서 해결하지 마라. 증상만 가리고 버전이 갈라진다.
- `sys.path` 조작으로 우회하지 마라. 인터프리터 자체가 다른 문제라 효과가 없다.

---

## 카드 I. [P1] LSTM 실행시간 87분 → 34분

### 문제

Step 3이 전체 실행시간의 98%다. 카드 B~E는 전부 "고치고 다시 돌려서 총점을 비교"하는 작업인데,
1회 87분이면 반복 자체가 불가능하다. **카드 B 착수 전에 이 카드를 먼저 끝내라.**

### 중요 — 결과가 바뀐다

배치 크기는 하이퍼파라미터다. `batch_size`를 바꾸면 **점수가 달라진다.**
따라서 아래 I-1(무해)과 I-2(결과 변경)를 **반드시 분리해서** 적용한다.
**기존 87분 실행의 점수와 새 점수를 같은 리더보드 표에 섞지 마라.**

### I-1. 스레드 수 조정 — 결과 불변, 87분 → 59분

`torch`가 기본으로 12스레드(전 코어)를 쓰는데, 배치 64에서는 스레드 동기화 비용이 연산량을 넘어선다.
실측상 12스레드가 8스레드보다 **느리다**(8.18초 vs 5.50초).

`baseline_7.ipynb`에서 **문자열 `torch.manual_seed(SEED)`가 있는 셀을 찾아** 그 아래에 추가한다.

```python
torch.set_num_threads(8)
# 12코어 전부를 쓰면 배치 64 구간에서 스레드 동기화 비용이 연산량을 넘어선다.
# 실측: 12스레드 8.18초/epoch vs 8스레드 5.50초/epoch
```

모델 파라미터가 아니므로 점수는 바뀌지 않는다. 부동소수 누적 순서 차이로 마지막 자리 수준의
미세한 변동은 있을 수 있다.

### I-2. 배치 크기 조정 — 결과 변경, 59분 → 34분

**패키지 파일은 건드리지 마라.** `VersionedLSTMPipeline.__init__`이 이미 `batch_size`를 인자로
노출하고 있고(`Windforce/Modeling/LSTMPipelineVersions.py:133`), `make_lstm_pipeline(**kwargs)`가
그대로 전달한다. 노트북의 `MODEL_PARAMS` 딕셔너리만 고치면 된다.

`baseline_7.ipynb`에서 **문자열 `MODEL_PARAMS = {`를 찾아** 아래 두 줄을 추가한다.

```python
MODEL_PARAMS = {
    "epochs": 80,
    "warmup_epochs": 15,
    "loss_k": 40.0,
    "regression_weight": 0.75,
    "patience": 12,
    "batch_size": 256,
    # 64 → 256으로 에폭당 스텝을 1/4로 줄인다. 실측 5.50초 → 3.18초/epoch
    "lr": 2e-3,
    # 배치가 4배가 되면 스텝당 그래디언트 잡음이 줄어드는 만큼 학습률을 sqrt(4)=2배로 올린다.
    # 이걸 빼면 같은 epoch 수에서 덜 수렴해 점수가 떨어진다
    "random_state": SEED,
}
```

### 검증

```
[그룹1][v1] epoch=NN | NMAE=0.xxxx FICR=0.xxxx 총점=0.xxxx
```

1. **속도**: Step 3 셀 전체가 **40분 이내**에 끝나야 한다. 60분을 넘으면 I-1이 안 먹은 것이다.
2. **점수**: 9개 모델의 총점이 **기존 대비 그룹별 0.02 이상 떨어지면 채택하지 않는다.**
   그 경우 `batch_size`를 128, `lr`을 1.4e-3으로 낮춰 한 번만 더 시도하고,
   그래도 떨어지면 **I-1만 남기고 I-2를 되돌린 뒤 보고한다.**
3. `pipeline.best_epoch`이 대부분 80에 붙어 있으면 조기종료가 안 걸린 것이니 그 사실을 보고한다.

---

## 카드 J. [P2] 구버전 `LSTMPipeline`의 하드코딩 배치 노출

### 문제

`Windforce/Modeling/LSTMPipeline.py`는 배치 크기가 두 군데 하드코딩되어 있다.

- `181번 줄` — `trainLoader = DataLoader(trainDs, batch_size=64, shuffle=True)`
- `305번 줄` — `testLoader = DataLoader(testDs, batch_size=64, shuffle=False)`

`baseline_7`은 `VersionedLSTMPipeline`을 쓰므로 영향이 없지만, **baseline 5·6이 이 클래스를 쓴다.**
카드 D(LSTM 재평가)에서 같은 87분 문제를 다시 만나게 된다.

### 구현 단계

`Windforce/Modeling/LSTMPipeline.py` **이 파일 하나만** 수정한다.

1. `__init__`(126번 줄 근처) 시그니처 **맨 끝**에 인자를 추가한다. 기존 인자 순서는 바꾸지 마라.

   ```python
   batch_size: int = 64,
   # 기본값 64는 기존 호출부의 동작을 그대로 보존하기 위한 것이다. 절대 바꾸지 마라
   ```

2. `__init__` 본문에 `self.batch_size = batch_size`를 추가한다.
3. 181번·305번 줄의 `batch_size=64`를 `batch_size=self.batch_size`로 바꾼다.

### 검증

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'D:/workspaces/WINDFORCE')
from Windforce.Modeling import LSTMPipeline
p = LSTMPipeline(capacity_kw=1000.0)
assert p.batch_size == 64, '기본값이 64가 아니면 기존 결과가 재현되지 않는다'
assert LSTMPipeline(capacity_kw=1000.0, batch_size=256).batch_size == 256
print('OK: 기본값 보존 + 인자 주입 동작')
"
```

### 하지 말 것

- 기본값을 256으로 바꾸지 마라. baseline 5·6의 기존 점수가 조용히 재현 불가능해진다.
  **기본값 유지 + 인자 노출**이 이 카드의 전부다.

---

## 카드 K. [P3] 리포지토리 위생

세 항목 모두 독립이고 각각 5분 이내다.

1. **`prep/`가 `.gitignore`에 없다.** 현재 `git status`에 `?? prep/`로 잡히며 약 92MB다.
   `git add .` 한 번이면 생성물 캐시가 통째로 커밋된다.
   `.gitignore`의 `# Jupyter Notebook` 섹션 **위에** 추가한다.

   ```gitignore
   # 전처리 생성물 캐시 (buildAll()이 언제든 재생성한다. 52초 소요)
   prep/
   ```

2. **`lightgbm` 4.7의 `eval_set` deprecation.** `BASELINE/baseline_7.ipynb`와
   `BASELINE/baseline_5_howto.ipynb`의 `model.fit(...)` 호출이 대상이다.
   현재 노트북이 `warnings.filterwarnings("ignore")`로 경고를 가리고 있어 조용히 통과하지만
   다음 메이저에서 깨진다. 각 1군데씩, 총 2군데다. **두 노트북의 표기가 다르니 주의한다** —
   `baseline_7`은 `eval_set=[(...)]`, `baseline_5_howto`는 `eval_set = [(...)]`(공백 있음)이다.

   ```python
   eval_set = [(X_valid, y_valid / capacity_kw)],   # 변경 전
   eval_X = X_valid, eval_y = y_valid / capacity_kw,   # 변경 후
   ```

   **바꾼 뒤 반드시 LightGBM 셀을 재실행해 총점이 그대로인지 확인한다.**

3. **`torch`가 `2.13.0+cpu` 빌드다.** 로컬에서는 GPU를 쓸 수 없다.
   이건 버그가 아니라 사실 확인이며, 카드 I의 34분이 **CPU 기준 하한**이라는 뜻이다.
   더 줄이려면 코랩 GPU 런타임에서 Step 3만 돌리는 쪽을 검토한다. 이번 작업 범위 밖이다.

---

## 카드 H~K 작업 순서

```
카드 H (커널 고정)   ← 이게 안 되면 노트북이 아예 안 돌아간다. 가장 먼저.
      ↓
카드 I (실행시간 87분 → 34분)   ← 카드 B~G의 반복 실험을 가능하게 만드는 전제
      ↓
카드 K (리포 위생)   ← 커밋 사고 방지. 짧으니 여기서 끊고 커밋한다.
      ↓
카드 J (구버전 배치 노출)   ← 카드 D 착수 직전에만 필요하다. 급하지 않다.
```

카드 H·I를 끝내고 **커밋한 뒤에** 위쪽 카드 B(FICR 최적화)로 넘어간다.
