# baseline7_howto_next — FICR 개선 작업 지시서

> 작성일: 2026-08-12
> 선행 산출물: `BASELINE/baseline_7.ipynb`, `BASELINE/baseline_7_results.csv`, `BASELINE/baseline_7_summary.csv`
> 이 문서는 **그대로 구현 가능한 작업 카드** 형식이다. 코드 스타일은 [`MD/coding_convention.md`](coding_convention.md)를 따른다.

### 실행자(Sonnet)에게

1. **셀은 문자열로 찾아라.** 노트북을 실행하면 셀 인덱스가 밀린다. 각 카드에 인용한
   고유 문자열(`"LGB_PARAMS = {"` 등)로 셀을 찾고, **모든 편집을 끝낸 뒤** 위에서부터 실행한다.
2. **새 패키지 파일을 만들지 마라.** `baseline_5_howto_next.md`의 카드 A~K와 달리, 이 문서의
   클래스·함수는 전부 `baseline_7.ipynb` 안에 새 셀로 추가한다(`FeatureSelector`,
   `OfficialScoreMonitor`가 이미 그렇게 되어 있는 것과 동일한 관례).
3. **탐색은 반드시 검증 구간(뒤 20%)에서만 한다.** 평가(2025년, 라벨 없음) 구간 정보가
   보정·앙상블 가중치 탐색에 섞이면 안 된다.
4. 카드 하나를 끝낼 때마다 그 카드의 "검증" 기준을 실제로 계산해 **수치를 보고**한다.
   "수정했습니다"만으로는 완료가 아니다.
5. 카드 H/I/K(커널 고정·실행시간·리포 위생, `baseline5_howto_next.md` 하단)는
   `baseline_7.ipynb`에 **이미 반영되어 있다** (`EXPECTED_VENV` 가드, `torch.set_num_threads(8)`,
   `MODEL_PARAMS["batch_size"]=256`, `eval_X=/eval_y=` API, `prep/` gitignore 확인함).
   다시 만들지 마라.

---

## 0. 현재 위치 (baseline_7 실행 결과)

검증 조건: 그룹별 시간순 마지막 20% hold-out, `TEST_RATIO = 0.2`, 첫 24시간(SEQ_LEN)은
모든 모델 비교에서 동일하게 제외.

`baseline_7_results.csv` 기준, 그룹별 **제출 가능 모델 중 검증 총점 최고** (Step 4 `best_by_group`):

| 그룹 | 선택 모델 | NMAE | FICR | 총점 |
|---|---|---|---|---|
| 1 | `LGBM_selected` | 0.0804 | 0.3914 | 0.6555 |
| 2 | `LSTM_v2_shap_selected` | 0.0860 | 0.4613 | 0.6877 |
| 3 | `LSTM_v1_shap_selected` | 0.0819 | 0.3985 | 0.6583 |
| — | **3그룹 평균(=실제 제출 기대치)** | **0.0828** | **0.4171** | **0.6672** |
| — | `Persistence_oracle_lag1` (비교 기준, 제출 불가) | 0.0467 | 0.6398 | 0.7966 |

> `baseline_7_summary.csv`의 `official_score`(모델 계열별 3그룹 단순 평균, 최고 0.6627)는
> **실제 제출 총점과 다르다.** 실제 제출은 그룹마다 다른 모델을 섞어 쓰므로(위 표),
> 진짜 기준선은 **0.6672**다. 이 문서의 모든 "개선" 판단은 이 숫자 및 그룹별 개별 값과 비교한다.

**격차 분해** (Persistence oracle 대비 0.1294점 차이):
- NMAE 기여분: `0.5 × (0.0828 − 0.0467) = 0.0181` (14%)
- FICR 기여분: `0.5 × (0.6398 − 0.4171) = 0.1114` (**86%**)

즉 남은 격차의 거의 전부가 FICR이다. NMAE는 이미 오차율 4%p 이내로 근접했지만,
FICR은 계단형 정산단가 구조 때문에 "평균 오차가 조금 준다"만으로는 거의 개선되지 않는다.

---

## 1. 진단: FICR이 아직 낮은 이유

- **LightGBM 계열(`LGBM_full`/`LGBM_selected`)은 FICR을 전혀 모른다.** `LGB_PARAMS["objective"] =
  "regression_l1"`로 순수 절대오차만 최소화한다. 6%/8% 문턱 근처에서의 정산단가 손실은
  학습 목적함수에 어떤 신호도 주지 않는다.
- **LSTM 계열은 이미 `ScoreLossFunction`(softFICR)을 섞어 쓴다** (`warmup_epochs=15` 이후
  `regression_weight=0.75`로 SmoothL1과 혼합, Step 3 `OfficialScoreMonitor`가 공식 총점으로
  조기종료). 그런데도 FICR이 0.38~0.46 수준에 머문다 — 손실함수 자체보다 하이퍼파라미터
  (`loss_k`, `regression_weight`)가 전 그룹 공통 고정값이라는 점, 그리고 사후 보정·앙상블이
  전혀 없다는 점이 남은 문제로 보인다.
- **예측값을 사후 보정(post-hoc calibration)하는 로직이 없다.** 학습된 모델의 출력이
  전반적으로 과소/과대 추정 편향이 있다면, 재학습 없이 스칼라 변환만으로 6% 밴드 안에
  들어가는 시간을 늘릴 수 있는데 이 시도가 baseline 7에 없다.
- **시간별 오차율이 실제로 어느 구간(≤6/6~8/>8%)에 몰려 있는지 측정한 적이 없다.**
  FICR 0.39 vs 0.46이 ">8% 비율이 큰 것"인지 "6~8% 비율이 큰 것"인지에 따라 다음 대응이
  완전히 다른데, 이 진단 자체가 baseline 7 어디에도 없다. → **카드 A가 다른 모든 카드의 전제.**
- **리드타임(`lead_hour`/`gfs_lead_hour`)은 원시 피처로만 들어가 있고**(`WindforceDatasetBuilder
  .featureCols`가 제외하지 않으므로 학습에는 이미 포함됨), 리드타임별 오차 특성을 진단하거나
  활용하는 로직은 없다.

---

## 카드 A. [P0] 오차율 밴드 진단 + 예측값 저장소 확보

### 문제
현재 Step 2/Step 3 루프는 `pred_full`/`pred_selected`/`pred`를 그룹 반복마다 지역 변수로만
쓰고 버린다. 카드 B~F가 전부 "검증 구간 예측값"을 필요로 하므로, 먼저 그룹·모델별 예측을
저장하고 그 위에서 오차율 밴드(≤6%/6~8%/>8%)를 계산해야 한다.

### 구현 단계

1. **문자열 `lgb_models: dict[tuple[int, str], lgb.LGBMRegressor] = {}`가 있는 셀**(Step 2 루프)
   맨 위, 그 선언 바로 아래에 저장소를 하나 추가한다.

   ```python
   predictions_store: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
   # (그룹, 모델명) -> (예측 kWh 배열, 실제 kWh 배열). 카드 A~F가 재학습 없이 재사용한다
   ```

   같은 셀에서 `pred_full`, `pred_selected`, `persistence`를 만든 직후 각각 한 줄씩 추가한다.

   ```python
   predictions_store[(group, "LGBM_full")] = (pred_full, actual_aligned)
   predictions_store[(group, "LGBM_selected")] = (pred_selected, actual_aligned)
   predictions_store[(group, "Persistence_oracle_lag1")] = (persistence, actual_aligned)
   ```

2. **문자열 `MODEL_PARAMS = {`가 있는 셀**(Step 3 루프)에서 `summary = summarizeAll(...)` 바로
   아래에 추가한다.

   ```python
   predictions_store[(group, model_name)] = (pred, actual)
   # LSTM v1/v2/v3의 검증 예측도 같은 저장소에 모아 카드 B~F에서 재사용한다
   ```

3. **Step 4 aggregation 셀**(`def aggregateOfficialScore(result_df: pd.DataFrame) -> pd.DataFrame:`)
   맨 아래, `print(f"✅ 요약 저장: {summary_path}")` 다음에 새 진단 셀을 추가한다.

   ```python
   def summarizeErrorBand(pred: np.ndarray, actual: np.ndarray, group: int) -> pd.Series:
       """시간별 오차율을 FICR 정산단가 구간으로 나눠 비율을 계산하는 함수

       Args:
           - pred, actual: 검증 구간 예측·실제 발전량(kWh) 배열
           - group: KPX 그룹 번호

       Logic:
           - hourlyRatio = |pred - actual| / RATED_CAPACITY_KW[group]
           - EvaluationMetrics.evaluateFICR과 동일한 문턱(0.06/0.08)으로 구간을 나눈다
           - ">8%" 비율이 클수록 그 시간대는 정산단가 0원 → FICR 손실의 직접 원인
       """
       ratio = np.abs(pred - actual) / RATED_CAPACITY_KW[group]
       band = pd.cut(
           ratio,
           bins=[-np.inf, 0.06, 0.08, np.inf],
           labels=["≤6%", "6~8%", ">8%"],
       )
       return band.value_counts(normalize=True).reindex(["≤6%", "6~8%", ">8%"])
       # 세 구간의 시간 비율 합이 1이 되는 Series를 반환한다

   band_rows = []
   for (group, model_name), (pred, actual) in predictions_store.items():
       share = summarizeErrorBand(pred, actual, group)
       band_rows.append({"group": group, "model_name": model_name, **share.to_dict()})
   band_df = pd.DataFrame(band_rows).sort_values(["group", ">8%"])
   display(band_df)
   # 그룹·모델별 밴드 분포를 한 표로 비교한다. best_by_group 모델의 ">8%" 값이 카드 B/C의 목표치다
   ```

### 검증 기준
- `band_df`에서 `best_by_group`에 해당하는 6개 행(그룹×모델)의 `">8%"` 비율을 실행 로그에 보고한다.
- 세 구간 합이 각 행마다 1.0인지 확인한다(반올림 오차 허용).

### 예상 효과
점수를 올리지 않지만, 카드 B(보정)와 카드 C(커스텀 목적함수) 중 **어느 쪽이 더 효과적일지**를
데이터로 판단할 수 있게 한다. `>8%` 비율이 크면 카드 C(근본적 재학습)가, `6~8%`가 크고 `>8%`는
작으면 카드 B(가벼운 보정)만으로도 충분할 가능성이 높다.

---

## 카드 B. [P1] 사후 보정(post-hoc calibration) — 재학습 없이 먼저 시도

### 문제
재학습 비용 없이 예측값에 아핀 변환만 적용해 검증 총점을 높일 수 있는지 확인하지 않았다.
비용이 거의 없으므로 카드 C보다 먼저 시도한다.

### 구현 단계

1. 카드 A 진단 셀 바로 아래에 새 셀을 추가한다.

   ```python
   class FicrCalibrator:
       """예측값에 단조 아핀 보정을 적용해 검증 총점을 최대화하는 클래스

       사용 순서:
           calibrator = FicrCalibrator(capacity_kw=RATED_CAPACITY_KW[group]).fit(pred, actual, group)
           pred_calibrated = calibrator.transform(pred)

       Logic:
           - pred_calibrated = clip(a * pred + b, 0, capacity_kw)
           - (a, b) 격자탐색으로 검증 구간 공식 총점이 최대인 조합을 찾는다
           - 반드시 fit()에 넘긴 예측·실제값(검증 구간)에서만 (a, b)를 찾는다
       """

       A_GRID = np.linspace(0.85, 1.15, 21)
       B_RATIO_GRID = np.linspace(-0.02, 0.02, 21)
       # b는 설비용량 비율로 탐색한 뒤 capacity_kw를 곱해 kWh 단위로 환산한다

       def __init__(self, capacity_kw: float):
           self.capacity_kw = capacity_kw
           self.a_: float = 1.0
           self.b_: float = 0.0

       def fit(self, pred: np.ndarray, actual: np.ndarray, group: int) -> "FicrCalibrator":
           best_score, best_a, best_b = -np.inf, 1.0, 0.0
           for a in self.A_GRID:
               for b_ratio in self.B_RATIO_GRID:
                   b = b_ratio * self.capacity_kw
                   candidate = np.clip(a * pred + b, 0.0, self.capacity_kw)
                   score = metrics.summarize(candidate, actual, group)["score"]
                   if score > best_score:
                       best_score, best_a, best_b = score, a, b
           # 441개 조합은 그룹당 수 초 내에 끝나므로 벡터화 없이 이중 루프로 충분하다
           self.a_, self.b_ = float(best_a), float(best_b)
           return self

       def transform(self, pred: np.ndarray) -> np.ndarray:
           return np.clip(self.a_ * pred + self.b_, 0.0, self.capacity_kw)
   ```

2. 같은 셀 아래에서 `predictions_store`의 모든 항목에 적용해 개선 여부를 표로 만든다
   (`Persistence_oracle_lag1`은 비교 대상에서 제외한다).

   ```python
   calibration_rows = []
   for (group, model_name), (pred, actual) in predictions_store.items():
       if model_name == "Persistence_oracle_lag1":
           continue
       before = metrics.summarize(pred, actual, group)
       calibrator = FicrCalibrator(capacity_kw=RATED_CAPACITY_KW[group]).fit(pred, actual, group)
       after = metrics.summarize(calibrator.transform(pred), actual, group)
       calibration_rows.append({
           "group": group, "model_name": model_name,
           "score_before": before["score"], "score_after": after["score"],
           "ficr_before": before["ficr"], "ficr_after": after["ficr"],
           "a": calibrator.a_, "b_kw": calibrator.b_,
       })
   calibration_df = pd.DataFrame(calibration_rows).sort_values(
       "score_after", ascending=False
   )
   display(calibration_df)
   ```

3. `score_after > score_before`인 (그룹, 모델) 조합의 `(a, b)`를 보관해 두었다가, **카드 F에서
   최종 제출 모델로 채택될 경우** Step 5 최종 셀(`def fitFinalLightGBM(`이 있는 셀)의 예측 생성
   직후에 같은 변환을 적용한다. 검증 구간에서 찾은 `(a, b)`를 그대로 재사용하고, 평가 구간
   데이터로 다시 탐색하지 않는다.

### 검증 기준
- `calibration_df`에서 `score_after`가 `score_before`보다 낮아지는 행이 있으면 그 모델은
  보정을 적용하지 않는다(항등 변환 `a=1, b=0`이 최선이라는 뜻이므로 채택하지 않는 것이 맞다).
- 그룹별 `best_by_group` 후보의 `score_after` 최댓값이 카드 0의 기준선(0.6555/0.6877/0.6583)을
  넘는지가 1차 채택 기준이다.

### 예상 효과
비용이 거의 없는 카드이므로 큰 개선을 기대하기보다는, "얼마나 쉽게 얻을 수 있는 부분인지"를
먼저 확인해 카드 C/D에 들일 노력의 우선순위를 정하는 데 쓴다.

---

## 카드 C. [P1] LightGBM에 FICR-aware 커스텀 목적함수 적용

### 문제
`LGB_PARAMS["objective"] = "regression_l1"`은 FICR 신호를 전혀 받지 않는다. LSTM은 이미
`ScoreLossFunction`으로 FICR을 근사 반영하는데 LightGBM만 빠져 있어 비교가 불공정하다.

### 중요 — capacity_kw로 나눌 필요가 없다

`trainLightGBM`이 `model.fit(X_train, y_train / capacity_kw, ...)`로 **이미 설비이용률(0~1)을
타깃으로 학습한다.** 따라서 LightGBM의 raw 예측값과 타깃 모두 이용률 단위이고, 오차율은
`|pred - actual|`을 capacity_kw로 다시 나눌 필요 없이 **그 자체가 이미 오차율**이다
(`ScoreLossFunction`의 `errorRatio = |pred_kw - actual_kw| / capacity_kw`와 동일한 값).

### 구현 단계

1. **문자열 `LGB_PARAMS = {`가 있는 셀**의 `trainLightGBM` 함수 위에 새 함수를 추가한다.

   ```python
   def makeFicrObjective(k: float = 40.0, regression_weight: float = 0.7):
       """LightGBM용 (grad, hess)를 반환하는 FICR-aware L1 혼합 목적함수를 만드는 팩토리 함수

       Args:
           - k: softFICR 시그모이드 steepness. ScoreLossFunction과 같은 기본값(40.0) 사용
           - regression_weight: L1 항 가중치. 1이면 순수 L1(기존과 동일), 0이면 순수 FICR 근사

       Logic:
           - 타깃·예측 모두 설비이용률(0~1) 단위이므로 오차율 e = pred - actual을 그대로 쓴다
           - softPrice(r) = 4 - sigmoid(k(r-0.06)) - 3*sigmoid(k(r-0.08)), r = |e|
           - loss_i = regression_weight*|e| + (1-regression_weight)*(1 - softPrice(r)/4)
           - L1 항의 2차 도함수는 거의 어디서나 0이므로 hess에 1e-6 하한을 둔다
           - FICR 항의 2차 도함수는 부호가 바뀔 수 있으므로 절댓값을 취한 뒤 하한을 둔다
       """
       def objective(y_true: np.ndarray, y_pred: np.ndarray):
           e = y_pred - y_true
           s = np.sign(e)
           r = np.abs(e)

           g1 = 1.0 / (1.0 + np.exp(-k * (r - 0.06)))
           g2 = 1.0 / (1.0 + np.exp(-k * (r - 0.08)))
           # sigmoid(k(r-0.06)), sigmoid(k(r-0.08))

           d_ficr = s * (k * g1 * (1 - g1) + 3.0 * k * g2 * (1 - g2)) / 4.0
           dd_ficr = (
               k * k * g1 * (1 - g1) * (1 - 2 * g1)
               + 3.0 * k * k * g2 * (1 - g2) * (1 - 2 * g2)
           ) / 4.0
           # loss_ficr = (g1 + 3*g2)/4 이므로 위 두 식은 pred에 대한 1·2차 도함수다
           # (dr/dpred = s이고 s^2 = 1이므로 2차 도함수 연쇄법칙에서 s^2 항은 사라진다)

           grad = regression_weight * s + (1.0 - regression_weight) * d_ficr
           hess = (1.0 - regression_weight) * np.abs(dd_ficr)
           hess = np.maximum(hess, 1e-6)
           # L1 항의 hess(0)는 이 하한에 흡수된다. FICR 항의 hess는 부호가 바뀔 수 있어 abs 처리
           return grad, hess
       return objective
   ```

2. **문자열 `lgb_models: dict[tuple[int, str], lgb.LGBMRegressor] = {}`가 있는 셀**(Step 2 루프)
   에서 `model_selected = trainLightGBM(...)` 블록 다음에 세 번째 후보를 추가한다.

   ```python
   model_ficr = lgb.LGBMRegressor(
       **{**LGB_PARAMS, "objective": makeFicrObjective(k=40.0, regression_weight=0.7)}
   )
   model_ficr.fit(
       X_train[columns], y_train / capacity_kw,
       eval_X=X_valid[columns], eval_y=y_valid / capacity_kw,
       eval_metric="l1",
       callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
   )
   # 커스텀 목적함수는 objective에만 관여하고 조기종료 기준(l1)은 그대로 유지한다
   lgb_models[(group, "LGBM_ficr")] = model_ficr
   pred_ficr = predictKw(model_ficr, X_valid[columns], capacity_kw)[SEQ_LEN:]
   summary_ficr = summarizeAll(pred_ficr, actual_aligned, group)
   predictions_store[(group, "LGBM_ficr")] = (pred_ficr, actual_aligned)
   records.append({
       "group": group, "model_name": "LGBM_ficr", "n_features": len(columns),
       "best_epoch": model_ficr.best_iteration_, **summary_ficr,
   })
   ```

3. `regression_weight`를 0.5 / 0.7 / 0.9 세 값으로 각각 실행해 그룹별로 어느 값이 가장 높은
   검증 총점을 내는지 비교한다(모델명은 `LGBM_ficr_w50`/`w70`/`w90`처럼 구분).
   `improvement.md` 2-1이 지적한 함정(FICR 항만 쓰면 초기 포화로 학습이 멈춤)을 피하려면
   `regression_weight`를 0.5 미만으로 내리지 않는다.

### 검증 기준
- `predictKw`가 여전히 `np.clip(..., 0.0, 1.0)`으로 이용률을 자르는지 확인한다(커스텀
  목적함수를 써도 이 클리핑은 그대로 유지해야 한다).
- 카드 A의 `summarizeErrorBand`를 `LGBM_ficr`에도 적용해 `">8%"` 비율이 `LGBM_selected`보다
  줄었는지 확인한다. 총점이 같아도 밴드 분포가 개선됐다면 카드 F의 앙상블 후보로 유효하다.
- 최종 채택은 검증 총점이 카드 0의 그룹별 기준선을 넘는지로 판단한다.

### 예상 효과
이 문서에서 근본적 개선 폭이 가장 큰 카드다. LGBM이 FICR을 직접 최적화하게 되므로,
카드 B(보정)보다 큰 폭의 `>8%` 비율 감소를 기대할 수 있다.

---

## 카드 D. [P1] LSTM `loss_k` / `regression_weight` 재탐색

### 문제
Step 3의 `MODEL_PARAMS`는 세 그룹·세 버전 모두에 같은 `loss_k=40.0`, `regression_weight=0.75`를
쓴다. 그룹마다 오차 분포와 설비용량이 다른데(그룹1/2 21,600kW, 그룹3 21,000kW; FICR도
0.38~0.46으로 그룹 간 편차가 큼) 하이퍼파라미터를 손으로 고른 1세트만 검증했다.

### 구현 단계

1. **문자열 `MODEL_PARAMS = {`가 있는 셀**을 카드 I(`baseline5_howto_next.md`)의 실행시간
   경험(9모델 80epoch 풀 실행 ≈ 34분)을 고려해, **그룹별로 이미 `best_by_group`에 뽑힌 버전
   1개만** 재탐색한다(9개 전부 다시 돌리지 않는다).
   - 그룹1: `v1`(현재 최고 후보였던 `LGBM_selected`와 근접 경쟁하는 버전)
   - 그룹2: `v2`
   - 그룹3: `v1`

2. 그리드: `regression_weight ∈ {0.6, 0.75, 0.9}` × `loss_k ∈ {20.0, 40.0, 80.0}` = 9조합 ×
   그룹 3개 = 27회 학습(각 회는 최대 80epoch, patience 12로 조기종료). 기존 `MODEL_PARAMS`를
   덮어쓰지 말고 지역 변수로 복사해 그리드 값만 바꾼다.

   ```python
   lstm_grid_rows = []
   grid_targets = {1: "v1", 2: "v2", 3: "v1"}
   # 카드 0의 best_by_group에서 그룹별로 이미 선택된 버전만 재탐색한다
   for group, version in grid_targets.items():
       data = datasets[group]
       columns = selected_features[group]
       X_train = data["X_train"][columns]
       X_valid = data["X_valid"][columns]
       y_train, y_valid = data["y_train"], data["y_valid"]
       variant = AllFeaturesVariant(random_state=SEED)
       train_frame = variant.fit_transform(X_train, y_train)
       valid_frame = variant.transform(X_valid)
       scaler = MinMaxScaler().fit(train_frame)
       X_train_scaled, X_valid_scaled = scaler.transform(train_frame), scaler.transform(valid_frame)
       monitor = OfficialScoreMonitor(metrics, group)

       for regression_weight in (0.6, 0.75, 0.9):
           for loss_k in (20.0, 40.0, 80.0):
               params = {**MODEL_PARAMS, "regression_weight": regression_weight, "loss_k": loss_k}
               pipeline = make_lstm_pipeline(
                   version, capacity_kw=RATED_CAPACITY_KW[group], seq_len=SEQ_LEN, **params
               )
               pipeline.fit(
                   X_train_scaled, y_train, X_val=X_valid_scaled, y_val=y_valid,
                   metrics=monitor, group=group,
               )
               pred = pipeline.predict(X_valid_scaled, y_valid)
               actual = y_valid[SEQ_LEN:]
               summary = summarizeAll(pred, actual, group)
               lstm_grid_rows.append({
                   "group": group, "version": version,
                   "regression_weight": regression_weight, "loss_k": loss_k,
                   "best_epoch": pipeline.best_epoch, **summary,
               })
               # 27회 전체가 baseline5_howto_next.md 카드 I 실측 기준으로 그룹당 약 3~4분 내 끝나야 한다

   lstm_grid_df = pd.DataFrame(lstm_grid_rows).sort_values(
       ["group", "score"], ascending=[True, False]
   )
   display(lstm_grid_df.groupby("group").head(3))
   # 그룹별 상위 3개 조합만 우선 확인한다
   ```

### 검증 기준
- 그룹별 그리드 실행이 **15분 이내**에 끝나야 한다(카드 I의 8스레드/배치256 실측 기준
  27회 ≈ 9~12분 예상). 크게 넘으면 그리드를 6조합으로 줄인다.
- 그룹별 최고 조합의 검증 총점이 카드 0의 기준선(그룹2 0.6877, 그룹3 0.6583)을 넘는지 확인한다.
- `best_epoch`이 대부분 80에 붙어 있으면 조기종료가 걸리지 않은 것이므로, 그 조합은
  `epochs`를 늘려 한 번 더 확인할 가치가 있다고 보고한다.

### 예상 효과
카드 C만큼 근본적이지는 않지만, 이미 FICR 손실을 쓰고 있는 LSTM에서 "공짜로" 얻을 수 있는
개선이다. 그룹2(`v2`, FICR 0.4613)처럼 이미 좋은 조합은 큰 변화가 없을 수 있고,
그룹1/3처럼 FICR이 낮은 쪽에서 개선 여지가 클 것으로 예상한다.

---

## 카드 E. [P2] `lead_hour` 기반 오차 진단 (보조)

### 배경
`data_description.md` 기준 실제 운영은 전날 13시 예보로 다음날 24시간을 일괄 예측하는
day-ahead 구조이고, `lead_hour`/`gfs_lead_hour`는 이미 원시 피처로 학습에 들어가 있다
(`WindforceDatasetBuilder.featureCols`가 제외하지 않음). 하지만 리드타임이 길수록 예보
오차가 커진다는 가정을 실제로 확인한 적이 없다.

### 구현 단계

1. 카드 A의 `predictions_store`와 `datasets[group]["df"]`(또는 `X_valid`)의 `lead_hour` 컬럼을
   시간 인덱스로 정렬해 이어붙인다.
2. `pd.cut(lead_hour, bins=[0, 19, 27, 35], labels=["단기(≤19h)", "중기(20~27h)", "장기(28~35h)"])`로
   구간을 나누고, 구간별로 카드 A의 `summarizeErrorBand`를 다시 계산한다.
3. 장기 구간의 `>8%` 비율이 단기 구간보다 뚜렷이 크면(예: 1.5배 이상), 카드 B의
   `FicrCalibrator`를 리드타임 구간별로 따로 적합하는 확장을 카드 F 착수 전에 검토한다.
   뚜렷하지 않으면 이 카드는 여기서 종료하고 보고만 한다 — 억지로 구간별 모델을 분리하지 않는다.

### 검증 기준
- 구간별 `>8%` 비율 표를 실행 로그에 남긴다. 추가 구현 여부는 이 표의 결과로 판단한다.

---

## 카드 F. [P2] 그룹별 최종 후보 채택 + 제출 반영

### 구현 단계

1. 카드 A~D에서 나온 모든 후보(원본 `LGBM_selected`/`LSTM_*`, 카드 B 보정 적용본, 카드 C
   `LGBM_ficr_*`, 카드 D 재탐색 LSTM)를 그룹별로 검증 총점 내림차순 정렬한다.
2. 앙상블(가중 블렌딩)은 **후보가 이미 2개 이상 카드 0 기준선을 넘었을 때만** 시도한다.
   가중치는 단순 격자탐색으로 충분하다.

   ```python
   for w in np.arange(0.0, 1.01, 0.05):
       blended = w * pred_a + (1 - w) * pred_b
       # pred_a, pred_b는 같은 그룹의 서로 다른 두 후보(검증 구간, predictions_store에서 조회)
       score = metrics.summarize(blended, actual, group)["score"]
   ```

   가중치는 검증 예측에서만 탐색하고, 앙상블 총점이 **두 후보 각각보다 낮으면 채택하지 않는다.**
3. 그룹별 최종 채택 모델·보정 파라미터를 **문자열 `def fitFinalLightGBM(`가 있는 셀**(Step 5)의
   해당 그룹 분기에 반영한다.
   - `LGBM_ficr`가 채택되면: `fitFinalLightGBM` 호출 시 `LGB_PARAMS` 대신
     `{**LGB_PARAMS, "objective": makeFicrObjective(...)}`를 쓰도록 그 그룹 분기만 수정한다.
   - 카드 B 보정이 채택되면: `pred = predictKw(...)` 또는 `pred = final_model.predict(...)`
     직후에 `pred = calibrator.transform(pred)`를 한 줄 추가한다. `calibrator`는 검증 구간에서
     이미 적합한 `(a, b)`를 그대로 재사용하고, 평가 구간에서 다시 `fit()`하지 않는다.
   - 앙상블이 채택되면: 두 최종 모델을 각각 전체 재학습해 `pred = w*pred_a + (1-w)*pred_b`로
     합친다. `w`는 검증 구간에서 찾은 값을 고정 상수로 쓴다.

### 검증 기준
- 기존 Step 5 검증 로직(`len(pred) != len(test_df)` 에러, 스키마·결측·물리범위·첫 24시간
  검증 셀)을 그대로 통과해야 한다. 새 로직을 추가했다고 이 가드들을 느슨하게 풀지 않는다.
- 최종 `submission_baseline7.csv` 재생성 후, 그룹별 값이 0이 아니고 설비용량 이하인지
  마지막 검증 셀로 재확인한다.

---

## 작업 순서 요약

```
카드 A (오차율 밴드 진단 + 예측값 저장소)   ← 다른 모든 카드의 전제. 가장 먼저.
      ↓
카드 B (사후 보정)   ← 비용이 가장 낮다. 다음.
      ↓
카드 C (LightGBM FICR 커스텀 목적함수)   ← 개선 폭이 가장 클 것으로 예상
      ↓
카드 D (LSTM loss_k/regression_weight 재탐색)   ← C와 독립적으로 병행 가능
      ↓
카드 E (lead_hour 진단, 보조)   ← 결과에 따라 선택적으로만 확장
      ↓
카드 F (그룹별 최종 채택 + 제출 반영)   ← 커밋 직전 마지막 단계
```

## 공통 준수 사항

- 코드 스타일: [`MD/coding_convention.md`](coding_convention.md) — 설명 주석은 코드 줄 아래,
  단위 명시, `df.copy()` 선행, 시간순 분할.
- 새 실험 결과는 `baseline_7_results.csv`와 같은 스키마
  (`group, model_name, n_features, best_epoch, nmae, ficr, score, mae, rmse`)로 추가 저장해
  `aggregateOfficialScore`로 재집계 가능하게 한다.
- **모든 보정·앙상블 가중치 탐색은 검증 구간(뒤 20%)에서만 한다.** 평가(2025년) 구간에는
  검증에서 고정한 파라미터를 그대로 적용만 한다.
- **채택 기준은 카드 0의 그룹별 기준선**(그룹1 0.6555 / 그룹2 0.6877 / 그룹3 0.6583)이다.
  이를 넘지 못하면 그 그룹은 기존 `best_by_group` 선택을 유지한다.
- `Persistence_oracle_lag1`은 비교 기준으로만 쓴다. 직전 실제값을 쓰므로 2025년 제출에는
  사용할 수 없다.

---
---

# 추가 지시 (2026-08-12, 2차) — VMD 분해 · 시각대 차등 모델링 검토 반영

> 근거 문서: [`MD/baseline7_workspace_summary.md`](baseline7_workspace_summary.md)(워크스페이스·데이터 구조 재확인),
> [`MD/baseline7_idea_evaluation.md`](baseline7_idea_evaluation.md)(두 아이디어 심층 검증 전문).
> 이 문서는 그 검증 결과를 실행 카드로 옮긴 것이다. **판정 요약만 필요하면 각 카드의 "판정" 줄만 읽는다.**

## Background & Objective

두 아이디어 모두 "FICR이 여전히 Persistence 대비 크게 낮다"(카드 0, 격차의 86%가 FICR)는
같은 문제의식에서 출발했다. 그러나 심층 검증 결과, 둘 다 **제안된 원형 그대로는 적용할 수
없거나 한계효용이 낮다는 구조적 근거**가 있었다(`baseline7_idea_evaluation.md` 참조):

- **VMD**: 이 대회는 평가 구간 실제값을 어떤 형태로도 쓸 수 없는 day-ahead 일괄 제출
  구조라(`data_description.md` 12절), 문헌의 표준적인 "타깃 분해 + 모드별 자기회귀 예측"
  구조는 `Persistence_oracle_lag1`과 같은 이유로 제출 불가능하다. **입력(NWP 예보) 신호
  분해로 범위를 좁혀야만** 시도할 수 있다.
- **리드타임 차등**: 이 데이터셋은 하루 1회(13:00)만 예보를 발행하므로 `lead_hour`가
  대상 시각(hour-of-day)의 결정론적 아핀 변환과 정확히 같다(캐시 데이터로 상관계수 1.0
  실측). 또한 이 원시 정보(`lead_hour`/`target_hour_*`/`hour_sin/cos_*`)는 **이미 SHAP
  선별을 통과해 `LGBM_selected`·`LSTM_v1/v2/v3` 전부의 학습 입력에 들어가 있다.** "리드타임
  열화"라는 인과 프레이밍은 이 데이터로 검증 불가능하며, "완전 분리 모델"은 이미 트리 모델이
  암묵적으로 할 수 있는 일을 데이터 손실(그룹당 1/3)을 감수하며 다시 구조로 강제하는 것에
  가깝다.

**목표**: 두 아이디어를 이론적으로 옳은 형태로 재정의하고, 카드 A~F(오차율 밴드 진단·사후
보정·LightGBM FICR 커스텀 목적함수·LSTM 재탐색)와 **충돌하지 않는 우선순위**로 실행 계획에
편입한다. 두 아이디어 모두 카드 A~F를 대체하지 않는다 — **카드 A~D를 먼저 끝내고, 그 결과가
카드 0 기준선을 넘긴 뒤 여유 자원이 있을 때** 아래 카드 G·H에 착수한다.

---

## 카드 G. [P3] VMD 입력 피처 분해 — 범위를 좁힌 탐색 스파이크

### Background & Objective

`ws10`(LDAPS IDW 대표 풍속)·`gfs_ws_hub`(GFS 허브높이 외삽 풍속)는 이미 1h/3h/6h 변화량
피처(`transformForecastFeature`)로 어느 정도의 다중 시간축 정보를 갖고 있지만, 추세와 고주파
변동을 명시적으로 분리한 피처는 없다. 목표는 VMD로 이 두 풍속류 신호를 K개 모드로 분해해
LightGBM/LSTM의 기존 SHAP 선별 피처셋에 **추가**하고, 카드 0 기준선(그룹1 0.6555 / 그룹2
0.6877 / 그룹3 0.6583) 대비 검증 총점이 오르는지 확인하는 것이다. 목표는 새 모델 계열을
만드는 것이 아니라 **기존 `LGBM_selected` 파이프라인에 피처를 더했을 때의 순수 효과**를
측정하는 것이다.

### Detailed Methodology

**절대 하지 말 것 (`baseline7_idea_evaluation.md` 1-1절 근거):**
- 타깃(발전량) 시계열을 VMD로 분해해 모드별로 자기회귀 예측하지 않는다 — 제출 불가 구조다.
- 학습 구간 + 검증 구간을 합쳐 **한 번에** VMD를 돌리지 않는다 — 검증 구간의 스펙트럼 정보가
  학습 구간 모드 값에 섞여 들어가는 누수다.

**해야 할 것:**

1. **분해 대상**: 그룹별 `ws10`(LDAPS), `gfs_ws_hub`(GFS) 두 컬럼만. 전체 127개 피처를
   전부 분해하지 않는다(연산 비용, 다중공선성 모두 불리).
2. **누수 차단 절차**:
   - 학습 구간(시간순 앞 80%) 피처값은 **그 80% 구간만으로 적합한 VMD 결과**에서 가져온다.
   - 검증 구간(뒤 20%)과 최종 평가(2025년) 구간은 **확장 윈도우(expanding window) 재적합**을
     쓰되, 계산 비용을 감안해 **주 단위(1주일)로만 재적합**한다. 한 번 재적합한 결과를
     다음 재적합 시점까지 그대로 쓴다 — 매 시각마다 다시 돌리지 않는다.
   - 각 재적합 윈도우의 **가장 최근(오른쪽 끝) 지점은 경계 왜곡이 가장 크다**는 것을
     명시적으로 알고 진행한다(완화 불가능한 구조적 한계 — Risks 참조).
3. **파라미터**: `K`는 4/6/8 세 값, `α`는 1,000/2,000 두 값으로 그리드 6조합.
   `vmdpy` 패키지 사용(순수 numpy/scipy 기반, 무거운 의존성 없음). **재구성 오차가 아니라
   카드 0과 동일한 검증 총점**으로 조합을 고른다(`baseline7_idea_evaluation.md` 1-2절 경고).
4. **모델 통합**: 새 컬럼(`vmd_ws10_mode1..K`, `vmd_gfs_ws_hub_mode1..K`)을 `datasets[group]["df"]`
   에 병합한 뒤, Step 2(`LGBM_full`/`LGBM_selected` 학습 셀)를 그대로 재실행해 SHAP 선별
   대상에 자연스럽게 포함시킨다. 별도의 "VMD 전용 모델"을 새로 만들지 않는다 — 기존
   파이프라인에 피처만 추가하는 것이 이 스파이크의 전부다.

### Step-by-Step Action Tasks

| 순서 | Task | 선행 조건 |
|---|---|---|
| G-1 | `pip install vmdpy` (또는 동등 라이브러리) 가능 여부 확인, `.venv`에 설치 | 없음 |
| G-2 | 그룹1 `ws10` 학습 구간(80%)만으로 VMD 1회 적합, K/α 6조합 재구성 오차 확인(폭주 여부만 점검) | G-1 |
| G-3 | 검증 구간(뒤 20%)에 대해 **주 단위 확장 윈도우** 재적합 로직 구현, 경계 왜곡 크기(마지막 24시간 vs 그 이전) 정량 비교 | G-2 |
| G-4 | `datasets[1]["df"]`에 병합, Step 2 재실행, 카드 0 기준선(0.6555) 대비 검증 총점 비교 | G-3 |
| G-5 | 그룹1에서 유의미한 개선(+0.005 이상)이 없으면 **그룹 2·3으로 확장하지 않고 종료**, 있으면 확장 | G-4 |

### Validation & Metrics

- **1차 게이트**: 그룹1 `LGBM_selected` 검증 총점이 VMD 피처 추가 전(0.6555) 대비 **+0.005
  이상** 개선되는지. 이 문턱을 넘지 못하면 이후 카드(그룹 2·3 확장, LSTM 통합)를 진행하지
  않는다 — 탐색 비용 대비 효용이 낮다고 판단한다.
- **2차 확인**: 카드 A의 `summarizeErrorBand`를 VMD 피처 포함/미포함 두 모델에 각각 적용해
  `>8%` 비율이 실제로 줄었는지 확인한다(총점 개선이 NMAE 쪽에서만 온 것이 아닌지 분리).
- **경계 왜곡 진단**: G-3에서 각 재적합 윈도우의 마지막 24시간과 그 이전 구간의 모드 값
  분산을 비교해, 마지막 24시간이 비정상적으로 크면(예: 2배 이상) 그 구간의 VMD 피처를
  결측 처리하고 원본 풍속 피처로 폴백하는 로직이 필요하다는 신호로 기록한다.

### Risks & Fallback

| 리스크 | 대응 |
|---|---|
| 경계 왜곡이 완화되지 않음(구조적 한계) | 최근 24시간 VMD 피처를 결측 처리 → 기존 원본 피처로 자동 폴백하는 가드를 G-4 병합 셀에 넣는다 |
| 주 단위 재적합도 연산 비용이 카드 I 기준(≈34분 실행시간 예산)을 크게 초과 | 재적합 주기를 월 단위로 낮춘다. 그래도 초과하면 카드 G를 P3에서 보류(deprioritize)하고 카드 A~F만 우선 완료 |
| G-4에서 총점 개선이 없거나 악화 | 즉시 폐기(1차 게이트가 이미 이 경우를 다룬다). VMD 컬럼을 `feature_cols`에서 제외하고 기존 파이프라인으로 복귀 |
| `vmdpy` 설치 실패(환경 제약) | PyEMD의 CEEMDAN 등 대체 라이브러리로 1회 대체 시도, 그래도 실패하면 카드 G 전체를 보류하고 보고 |

### 판정 (요약)

**P3.** 카드 A~D가 카드 0 기준선을 넘긴 뒤에만 착수한다. 1차 게이트(그룹1, +0.005)를 넘지
못하면 더 투자하지 않는다.

---

## 카드 H. [P2] 시각대(diurnal) 오차 진단 + 저비용 차등화

> 기존 카드 E("`lead_hour` 기반 오차 진단")를 대체한다. 카드 E의 "리드타임" 프레이밍은
> `baseline7_idea_evaluation.md` 2-1절 근거로 부정확하다 — 이 데이터셋에서 `lead_hour`는
> 대상 시각(hour-of-day)의 아핀 변환과 동일한 정보이므로, 이하 카드는 "시각대"로 정정해
> 진행한다. 카드 E를 이미 실행했다면 결과는 그대로 유효하다(같은 축을 다른 이름으로 부른
> 것뿐이다).

### Background & Objective

목표는 두 가지를 순서대로 확인하는 것이다: (1) 시간별 오차율의 밴드 분포가 시각대별로
실제로 불균일한가, (2) 불균일하다면 **완전 분리 모델이 아니라 저비용 차등**(사후 보정·표본
가중)만으로 그 불균일을 얼마나 완화할 수 있는가. `baseline7_idea_evaluation.md` 2-2절이
확인했듯 `lead_hour`/`target_hour_*`/`hour_sin/cos_*` 8개 컬럼이 이미 두 모델 계열의 학습
입력에 들어가 있으므로, "모델에 이 정보를 준다"가 아니라 **"이미 갖고 있는 정보를 목적함수
레벨에서 더 적극적으로 쓴다"**가 이 카드의 실제 작업이다.

### Detailed Methodology — 진단 매트릭스 및 차등 기준

1. **시각대 구간**: `target_hour_ldaps`(1~24) 또는 `lead_hour`(12~35, 둘은 동일 정보이므로
   해석이 쉬운 쪽을 쓴다) 기준 3구간 — 단기 `01~08시`, 중기 `09~16시`, 장기 `17~24시`.
   구간 경계는 카드 0의 기존 표기(`단기/중기/장기`)와 맞춘다.
2. **진단 매트릭스**: (그룹 × 모델 × 시각대) 조합마다 카드 A의 `summarizeErrorBand` 출력
   (`≤6%`/`6~8%`/`>8%` 비율)을 계산해 하나의 표로 만든다. 총 3그룹 × best_by_group 모델 1개
   × 3시각대 = 9행이면 충분하다(모든 모델×시각대 조합을 다 볼 필요는 없다).
3. **차등 적용 기준**: 어떤 시각대의 `>8%` 비율이 다른 시각대 평균보다 **1.5배 이상** 크면
   "불균일"로 판정하고 아래 저비용 차등을 적용한다. 그 미만이면 카드 H는 진단 단계에서
   종료하고 완전 분리 모델은 시도하지 않는다(`baseline7_idea_evaluation.md` 2-4절 ROI 표의
   (a)/(d) 행 — 데이터 손실 대비 효용이 낮다는 결론을 그대로 따른다).

### Step-by-Step Action Tasks

| 순서 | Task | 선행 조건 |
|---|---|---|
| H-1 | 카드 A의 `predictions_store`와 각 그룹 `X_valid`(또는 `datasets[group]["df"]`)의 `target_hour_ldaps`를 시간 인덱스로 정렬해 이어붙인다 | 카드 A 완료 |
| H-2 | 위 2-2 진단 매트릭스(9행)를 계산해 표로 보고한다 | H-1 |
| H-3 | 불균일 판정이 나오면(1.5배 기준): 카드 B의 `FicrCalibrator`를 시각대 인자를 받도록 확장(`fit(pred, actual, group, hour_bucket)`)해 시각대별 `(a, b)`를 따로 탐색 | H-2, 카드 B |
| H-4 | 카드 C의 `makeFicrObjective` 학습 시, LightGBM `sample_weight` 인자로 `>8%` 비율이 높은 시각대 행에 1.2~1.5배 가중을 주는 버전을 추가로 학습해 비교 | H-2, 카드 C |
| H-5 | H-3/H-4 중 더 큰 개선을 보인 쪽만 채택해 카드 F(최종 제출 반영)에 병합. 완전 분리 모델(그룹×시각대별 별도 LightGBM/LSTM)은 **H-2 결과가 2배 이상 불균일하고 H-3/H-4로도 목표를 못 채울 때만** 마지막 수단으로 검토한다 | H-3, H-4 |

### Validation & Metrics

- H-2 진단표의 시각대별 `>8%` 비율과 그 비율(최대/평균)을 실행 로그에 남긴다 — 이 숫자가
  이후 모든 판단의 근거다.
- H-3/H-4 각각 적용 전후의 그룹별 검증 총점을 카드 0 기준선과 비교한다. **두 방법 모두
  기준선을 넘지 못하면 카드 H는 여기서 종료**하고 완전 분리 모델을 시도하지 않는다.
- 완전 분리 모델까지 간다면(H-5의 마지막 수단), 그룹3처럼 학습 데이터가 이미 적은 그룹은
  시각대 분리 후 표본 수가 과적합 위험 수준(연 단위 데이터 2년 × 1/3 시각대 ≈ 5,800시간)인지
  먼저 확인한다.

### Risks & Fallback

| 리스크 | 대응 |
|---|---|
| H-2에서 불균일이 확인되지 않음(가능성 높음 — 2-2절 근거) | 카드 H를 진단 단계에서 종료. 시간 낭비 방지가 이 카드의 실질적 성과다 |
| 시각대별 보정(H-3)이 특정 시각대에서 과적합(표본 수 부족) | `FicrCalibrator`의 격자탐색 조합 수를 줄이거나(11×11), 인접 시각대와 통합해 2구간으로 낮춘다 |
| 완전 분리 모델까지 갔는데도 데이터 손실로 총점이 악화 | 즉시 폐기하고 H-3/H-4의 저비용 버전으로 복귀. 이 경로가 `baseline7_idea_evaluation.md` 2-4절이 예상한 결과이므로 놀라운 일이 아니다 |

### 판정 (요약)

**P2, 그러나 축소된 형태로.** 원안(완전 분리 모델)은 비권장. 진단(H-2) → 저비용 차등
(H-3/H-4) 순서를 반드시 지키고, 각 단계에서 카드 0 기준선을 넘지 못하면 다음 단계로
확대하지 않는다.

---

## 작업 순서 갱신 (카드 G·H 포함)

```
카드 A (오차율 밴드 진단 + 예측값 저장소)
      ↓
카드 B (사후 보정) ── 카드 C (LightGBM FICR 커스텀 목적함수) ── 카드 D (LSTM 재탐색)   [병행 가능]
      ↓
카드 F (그룹별 최종 채택 + 제출 반영)  ← 여기까지가 이 문서의 핵심 우선순위(카드 0 기준선 갱신)
      ↓
      ├─ 카드 H (시각대 진단 → 저비용 차등)  [P2, H-2 결과에 따라 축소 실행]
      └─ 카드 G (VMD 입력 피처 스파이크)      [P3, 1차 게이트 미달 시 즉시 종료]
```

카드 G·H는 **카드 A~D·F가 카드 0 기준선을 이미 넘긴 뒤**, 그리고 여유 자원이 있을 때만
착수한다. 두 카드 모두 "판정" 절이 명시한 게이트를 넘지 못하면 그 자리에서 종료하고 다음
카드로 넘어가지 않는다 — 이 문서의 목적은 새 기법을 최대한 많이 시도하는 것이 아니라
카드 0의 실제 제출 총점(0.6672)을 올리는 것이다.
