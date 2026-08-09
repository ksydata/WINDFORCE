# baseline5_howto 후속 개선 작업 지시서

> 작성일: 2026-08-09 · 선행 산출물: `BASELINE/baseline5_howto.ipynb`
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
