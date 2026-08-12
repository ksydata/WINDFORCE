# baseline7_howto_next — FICR 작업 지시서

> 작성일: 2026-08-12 / 개정: 2026-08-13
> 선행 산출물: `BASELINE/baseline_7.ipynb`, `BASELINE/baseline_7_results.csv`, `BASELINE/baseline_7_summary.csv`
> 근거 문서: [`MD/baseline7_workspace_summary.md`](baseline7_workspace_summary.md)(워크스페이스·데이터 구조),
> [`MD/baseline7_idea_evaluation.md`](baseline7_idea_evaluation.md)(아이디어 1~6 심층 검증 전문 — 이 문서의
> 모든 기술적 판단 근거). **판정 요약만 필요하면 각 카드의 "검증 가설 및 KPI" 절만 읽는다.**
> 코드 스타일은 [`MD/coding_convention.md`](coding_convention.md)를 그대로 따른다.

### 실행자에게

1. 셀 번호가 아니라 부록 A의 **문자열 앵커**로 셀을 찾고, 편집을 끝낸 뒤 위에서부터 실행한다.
2. 신규 클래스·함수는 `baseline_7.ipynb` 안에 추가한다. 새 패키지는 자동 설치하지 않는다.
3. 보정·앙상블·HPO·VMD 설정은 검증 구간(뒤 20%)에서만 정하고, 2025년 평가 구간에는 고정값만 적용한다.
4. 카드마다 검증 총점과 KPI 수치를 남긴다. 기준선보다 좋아지지 않으면 즉시 종료한다.
5. 커널·실행시간·리포 위생 카드(H/I/K)는 이미 반영됐으므로 다시 만들지 않는다.
6. 채택 기준은 항상 `EvaluationMetrics.summarize`의 검증 총점이다.

---

## 1. 개요 및 고도화 목표 (Executive Summary)

### 1-1. 현재 위치 — baseline_7 실행 결과

검증 조건: 그룹별 시간순 마지막 20% hold-out, `TEST_RATIO = 0.2`, 첫 24시간(`SEQ_LEN`)은
모든 모델 비교에서 동일하게 제외.

`baseline_7_results.csv` 기준, 그룹별 **제출 가능 모델 중 검증 총점 최고**(Step 4 `best_by_group`):

| 그룹 | 선택 모델 | NMAE | FICR | 총점 |
|---|---|---|---|---|
| 1 | `LGBM_selected` | 0.0804 | 0.3914 | 0.6555 |
| 2 | `LSTM_v2_shap_selected` | 0.0860 | 0.4613 | 0.6877 |
| 3 | `LSTM_v1_shap_selected` | 0.0819 | 0.3985 | 0.6583 |
| — | **3그룹 평균(= 실제 제출 기대치, 이하 "기준선")** | **0.0828** | **0.4171** | **0.6672** |
| — | `Persistence_oracle_lag1`(비교 기준, 제출 불가) | 0.0467 | 0.6398 | 0.7966 |

**격차 분해**(Persistence oracle 대비 0.1294점 차이):

```
NMAE 기여분 = 0.5 × (0.0828 − 0.0467) = 0.0181   (14%)
FICR 기여분 = 0.5 × (0.6398 − 0.4171) = 0.1114   (86%)
```

### 1-2. 고도화 목표 — "평균 오차 감소"에서 "정산 구간 안착"으로 재정의

남은 격차의 86%가 FICR이라는 사실은, 이 문서의 모든 성공 기준을 **NMAE를 조금 더 낮추는
것**이 아니라 **시간별 오차율이 정산단가 4원 구간(≤6%)에 들어가는 비율(=밴드 안착률)을
높이는 것**으로 재정의해야 한다는 뜻이다. FICR은 발전량 가중 계단함수이므로, 평균 오차가
줄어도 6%를 살짝 넘는 시간이 많으면 점수는 거의 오르지 않는다. 반대로 평균 오차가 그대로여도
"6.5% 오차 시간"을 "5.9% 오차"로 밀어넣으면 점수가 즉시 뛴다. 이 문서의 카드 A~G는 전부
이 재정의를 전제로 설계됐다.

**정량 목표(단계별):**

| 단계 | 목표 | 판단 기준 |
|---|---|---|
| Phase 1 (카드 A/B) | 밴드 분포 가시화 + 저비용 보정으로 얻을 수 있는 상한 확인 | `band_df`, `calibration_df` 실측 |
| Phase 2 (카드 C/D) | 그룹별 검증 총점이 기준선(0.6555/0.6877/0.6583)을 **모두** 상회 | `predictions_store` 재집계 |
| Phase 3 (카드 E/F) | 3그룹 평균 총점이 **0.6672 → 0.68 이상**으로 상승(잠정 목표, 카드 A 실측 후 재조정) | `aggregateOfficialScore` |
| Phase 4 (카드 G, 선택) | 그룹1 VMD 피처 추가로 **+0.005 이상** 추가 개선(1차 게이트) | Step 2 재실행 비교 |

목표 수치(0.68)는 현재 격차의 86%가 FICR이라는 진단에서 나온 **작업 방향성**이지, 이 문서가
보장하는 확정치가 아니다 — 카드 A~F 각각의 실측 결과로 계속 갱신한다.

### 1-3. 5대 고도화 전략 → 카드 매핑

| 고도화 전략 | 대응 카드 | 핵심 |
|---|---|---|
| ① 오차율 구간 진단 + 예측값 저장소 + 자동 분기 | **카드 A** | `predictions_store`, `summarizeErrorBand`, `recommendAction` |
| ② FICR-aware 커스텀 목적함수(LightGBM) + LSTM 정렬 | **카드 C** | `makeFicrObjective`(2단계 웜업 포함), `ScoreLossFunction`과의 설계 원칙 정렬 |
| ③ VMD 입력 분해 + IMF 특징 선택(mRMR/PCC-GRA) | **카드 G** | 입력(NWP) 한정 VMD, mRMR 축소 → 기존 SHAP 선별 |
| ④ 지능형 HPO(메타휴리스틱/베이지안) | **카드 D** | GWO(1순위, 순수 numpy) 기반 `loss_k`/`regression_weight` 연속 탐색 |
| ⑤ 시각대별 사후 보정 + 앙상블 | **카드 B(확장) / 카드 E / 카드 F** | `FicrCalibrator`의 시각대 확장, 진단 매트릭스, 가중 블렌딩 |

각 전략의 이론적 타당성·리스크 검증 전문은 `MD/baseline7_idea_evaluation.md`의 아이디어 1~6에
있다. 이 문서는 그 검증 결과를 **그대로 구현 가능한 카드**로 옮긴 것이다.

---

## 2. 5대 핵심 실행 로드맵 (Action Items: Cards A~G)

### 카드 A. [P0] 오차율 밴드 진단 + 예측값 저장소 + 자동 분기

**[목적]** 카드 B~G 전부가 "검증 구간 예측값"을 필요로 하므로, 먼저 그룹·모델별 예측을
저장하고 오차율 밴드(`≤6%`/`6~8%`/`>8%`)를 계산한 뒤, 그 결과로 카드 B(보정)와 카드 C(재학습)
중 어느 쪽에 자원을 먼저 투입할지 **규칙 기반으로 자동 판단**한다(`baseline7_idea_evaluation.md`
아이디어 3).

**[기술적 세부 사양]**

1. `lgb_models: dict[tuple[int, str], lgb.LGBMRegressor] = {}`가 있는 셀(Step 2 루프) 맨 위에
   저장소를 선언한다.

   ```python
   predictions_store: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
   # (그룹, 모델명) -> (예측 kWh 배열, 실제 kWh 배열). 카드 A~G가 재학습 없이 재사용한다
   ```

   같은 셀에서 `pred_full`, `pred_selected`, `persistence`를 만든 직후 각각 저장한다.

   ```python
   predictions_store[(group, "LGBM_full")] = (pred_full, actual_aligned)
   predictions_store[(group, "LGBM_selected")] = (pred_selected, actual_aligned)
   predictions_store[(group, "Persistence_oracle_lag1")] = (persistence, actual_aligned)
   ```

2. `MODEL_PARAMS = {`가 있는 셀(Step 3 루프)에서 `summary = summarizeAll(...)` 바로 아래에
   LSTM v1/v2/v3의 검증 예측도 동일하게 저장한다.

   ```python
   predictions_store[(group, model_name)] = (pred, actual)
   ```

3. Step 4 aggregation 셀(`def aggregateOfficialScore(result_df: pd.DataFrame) -> pd.DataFrame:`)
   맨 아래, `print(f"✅ 요약 저장: {summary_path}")` 다음에 진단 셀을 추가한다.

   ```python
   def summarizeErrorBand(pred: np.ndarray, actual: np.ndarray, group: int) -> pd.Series:
       """시간별 오차율을 FICR(오차가 작을수록 단가가 높은 점수) 구간으로 나눠 비율을 계산하는 함수

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

   def recommendAction(band_share: pd.Series, gt8_dominance_ratio: float = 1.5,
                        gt8_absolute_floor: float = 0.15, mid_band_floor: float = 0.10) -> str:
       """오차 밴드 비율로부터 카드 B/C 중 무엇을 우선할지 규칙 기반으로 추천하는 함수

       Args:
           - band_share: summarizeErrorBand()의 반환값, 인덱스 ["≤6%","6~8%",">8%"]
           - gt8_dominance_ratio: ">8%"가 "6~8%"의 이 배수 이상이면 카드 C(재학습) 우선
           - gt8_absolute_floor: ">8%" 절대 비율이 이 값 이상이면 배수와 무관하게 카드 C 우선
           - mid_band_floor: ">8%"는 낮지만 "6~8%"가 이 값 이상이면 카드 B(보정)만으로 개선 여지

       Logic:
           - ">8%"는 FICR 기여가 0원이므로 이 비율을 줄이는 것이 총점에 가장 직접적이다.
             그래서 ">8%" 조건을 "6~8%" 조건보다 먼저 검사한다(우선순위 비대칭).
           - 세 조건 모두 해당 없으면 이미 양호하므로 원안(best_by_group) 유지를 권고한다.
           - 이 함수의 출력은 "권고"일 뿐 강제가 아니다. 카드 B는 비용이 거의 없으므로 권고와
             무관하게 항상 먼저 시도하고, 이 함수는 "카드 C에 자원을 더 투자할 가치가 있는가"만
             판단하는 데 쓴다(baseline7_idea_evaluation.md 아이디어 3-3 근거).
       """
       gt8, mid = band_share[">8%"], band_share["6~8%"]
       if gt8 >= gt8_absolute_floor or (mid > 0 and gt8 >= gt8_dominance_ratio * mid):
           return "카드 C 우선 (근본적 재학습 — FICR-aware 목적함수)"
       if mid >= mid_band_floor:
           return "카드 B 우선 (저비용 사후 보정으로 6~8%를 6% 이하로 밀어넣기 시도)"
       return "카드 B/C 착수 전 원안 유지 — 이미 밴드 분포 양호"

   band_rows = []
   for (group, model_name), (pred, actual) in predictions_store.items():
       share = summarizeErrorBand(pred, actual, group)
       band_rows.append({
           "group": group, "model_name": model_name, **share.to_dict(),
           "recommended_action": recommendAction(share),
       })
   band_df = pd.DataFrame(band_rows).sort_values(["group", ">8%"])
   display(band_df)
   # 그룹·모델별 밴드 분포와 권고 조치를 한 표로 비교한다
   ```

**[데이터 누수 방지책]** `predictions_store`는 검증 구간(뒤 20%) 예측만 담는다 — 평가(2025년)
구간 예측을 여기 섞지 않는다. `recommendAction`의 임계값(1.5배/15%/10%)은 사전 추정치(prior)일
뿐이므로, 실측 `band_df`가 그룹 간 편차를 크게 보이면 그룹별로 재조정한다
(`baseline7_idea_evaluation.md` 아이디어 3-2).

**[검증 가설 및 KPI]**
- `band_df`에서 `best_by_group`에 해당하는 6개 행의 `">8%"` 비율과 `recommended_action`을
  실행 로그에 보고한다.
- 세 구간 합이 각 행마다 1.0인지 확인한다(반올림 오차 허용).
- 가설: `>8%` 비율이 크면 카드 C가, `6~8%`가 크고 `>8%`는 작으면 카드 B만으로도 충분할
  가능성이 높다 — 이 가설은 `recommended_action` 컬럼으로 자동 검증된다.

---

### 카드 B. [P1] 사후 보정(post-hoc calibration) — 전역 + 시각대별 확장

**[목적]** 재학습 비용 없이 예측값에 아핀 변환만 적용해 검증 총점을 높일 수 있는지 확인한다.
비용이 거의 없으므로 카드 C보다 먼저 시도하며, 카드 E의 시각대 진단 결과가 불균일하면 시각대
차원으로 확장한다(`baseline7_idea_evaluation.md` 아이디어 2-4의 저비용 대안 (b)).

**[기술적 세부 사양]**

1. 카드 A 진단 셀 바로 아래에 전역 보정 클래스를 추가한다.

   ```python
   class FicrCalibrator:
       """예측값에 단조 아핀(기울기와 절편으로 만드는 직선) 보정을 적용하는 클래스

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

2. 같은 셀 아래에서 `predictions_store`의 모든 항목(`Persistence_oracle_lag1` 제외)에 적용해
   개선 여부를 표로 만든다.

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
   calibration_df = pd.DataFrame(calibration_rows).sort_values("score_after", ascending=False)
   display(calibration_df)
   ```

3. **시각대 확장(카드 E 결과가 "불균일"일 때만 실행).** `FicrCalibrator`를 상속해 시각대
   인자를 받는 버전을 추가한다.

   ```python
   class DiurnalFicrCalibrator(FicrCalibrator):
       """target_hour_ldaps 3구간(단기/중기/장기)별로 (a, b)를 독립적으로 탐색하는 확장 클래스

       Logic:
           - 구간마다 표본 수가 카드 A 대비 1/3로 줄어드므로, 표본이 너무 적으면(예: <500시간)
             과적합 위험이 있다는 것을 fit() 호출부에서 확인해야 한다(카드 E §검증 가설 참고)
           - 구간별 (a_h, b_h)를 dict로 보관하고, transform()은 hour_bucket 배열을 받아
             구간별로 다른 변환을 적용한다
       """
       def fit(self, pred: np.ndarray, actual: np.ndarray, group: int,
               hour_bucket: np.ndarray) -> "DiurnalFicrCalibrator":
           self.params_: dict[str, tuple[float, float]] = {}
           for bucket in np.unique(hour_bucket):
               mask = hour_bucket == bucket
               sub = FicrCalibrator(self.capacity_kw).fit(pred[mask], actual[mask], group)
               self.params_[bucket] = (sub.a_, sub.b_)
           return self

       def transform(self, pred: np.ndarray, hour_bucket: np.ndarray) -> np.ndarray:
           out = pred.copy()
           for bucket, (a, b) in self.params_.items():
               mask = hour_bucket == bucket
               out[mask] = np.clip(a * pred[mask] + b, 0.0, self.capacity_kw)
           return out
   ```

4. `score_after > score_before`인 (그룹, 모델) 조합의 `(a, b)`를 보관해 두었다가, **카드 F에서
   최종 제출 모델로 채택될 경우** Step 5 최종 셀(`def fitFinalLightGBM(`이 있는 셀)의 예측 생성
   직후에 같은 변환을 적용한다. 검증 구간에서 찾은 `(a, b)`를 그대로 재사용하고, 평가 구간
   데이터로 다시 탐색하지 않는다.

**[데이터 누수 방지책]** `(a, b)`(전역·시각대별 모두)는 검증 구간에서만 탐색한다. 평가
구간에서는 `fit()`을 다시 호출하지 않고 검증 구간에서 고정한 파라미터를 `transform()`에만
적용한다. `DiurnalFicrCalibrator`도 동일 원칙 — 시각대 구간 경계(`target_hour_ldaps` 기준)는
카드 E가 정한 고정 구간을 그대로 쓴다.

**[검증 가설 및 KPI]**
- `calibration_df`에서 `score_after < score_before`인 행은 보정을 적용하지 않는다(항등 변환이
  최선이라는 뜻).
- 그룹별 `best_by_group` 후보의 `score_after` 최댓값이 기준선(0.6555/0.6877/0.6583)을 넘는지가
  1차 채택 기준이다.
- 시각대 확장은 카드 E의 "불균일" 판정(§카드 E)이 나온 그룹에서만 시도하고, 전역 보정 대비
  검증 총점이 더 높을 때만 채택한다.

---

### 카드 C. [P1] LightGBM FICR-aware 커스텀 목적함수 + LSTM 정렬

**[목적]** `LGB_PARAMS["objective"] = "regression_l1"`은 FICR 신호를 전혀 받지 않는다. LSTM은
이미 `ScoreLossFunction`으로 FICR을 근사 반영하는데 LightGBM만 빠져 있어 비교가 불공정하다.
이 카드는 (1) 수학적으로 검증된 `fobj`/`hess`를 적용하고, (2) LSTM의 웜업 설계 원칙과 정렬해
학습 안정성을 확보한다(`baseline7_idea_evaluation.md` 아이디어 4).

**[기술적 세부 사양 — 수식]**

`trainLightGBM`이 `model.fit(X_train, y_train / capacity_kw, ...)`로 이미 설비이용률(0~1)을
타깃으로 학습하므로, LightGBM raw 예측·타깃 모두 이용률 단위이고 오차율은 `|pred - actual|`을
capacity_kw로 다시 나눌 필요가 없다.

```
e = pred - actual,  r = |e|,  w = regression_weight
g1 = σ(k(r-0.06)),  g2 = σ(k(r-0.08))
loss(e) = w·|e| + (1-w)·(g1 + 3g2)/4

grad = dloss/dpred = sign(e)·[ w + (1-w)·k·(g1(1-g1) + 3g2(1-g2))/4 ]
hess = d²loss/dpred² = (1-w)·k²·[ g1(1-g1)(1-2g1) + 3g2(1-g2)(1-2g2) ]/4
```

이 유도는 `baseline7_idea_evaluation.md` 4-1절에서 손으로 재검증되어 코드와 정확히 일치함을
확인했다.

```python
def makeFicrObjective(k: float = 40.0, regression_weight: float = 0.7):
    """LightGBM용 grad(오차 방향)와 hess(오차 변화 민감도)를 반환하는 목적함수 생성 함수

    Args:
        - k: softFICR 시그모이드 steepness. ScoreLossFunction과 같은 기본값(40.0) 사용
        - regression_weight: L1 항 가중치. 1이면 순수 L1(기존과 동일), 0이면 순수 FICR 근사

    Logic:
        - 타깃·예측 모두 설비이용률(0~1) 단위이므로 오차율 e = pred - actual을 그대로 쓴다
        - hess 하한을 절대값(1e-6)이 아니라 regression_weight에 비례한 상대 하한으로 잡는다.
          문턱(6%/8%) 근처 소수 표본에서 hess가 수백 배 커지는 비균질 분포가 되면 LightGBM
          리프 가중치(-Σgrad/(Σhess+λ))가 그 소수 표본에 지배되어 발산할 수 있기 때문이다
          (baseline7_idea_evaluation.md 4-2절 근거)
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        e = y_pred - y_true
        s = np.sign(e)
        r = np.abs(e)

        g1 = 1.0 / (1.0 + np.exp(-k * (r - 0.06)))
        g2 = 1.0 / (1.0 + np.exp(-k * (r - 0.08)))

        d_ficr = s * (k * g1 * (1 - g1) + 3.0 * k * g2 * (1 - g2)) / 4.0
        dd_ficr = (
            k * k * g1 * (1 - g1) * (1 - 2 * g1)
            + 3.0 * k * k * g2 * (1 - g2) * (1 - 2 * g2)
        ) / 4.0

        grad = regression_weight * s + (1.0 - regression_weight) * d_ficr
        hess_floor = max(1e-2 * (1.0 - regression_weight), 1e-6)
        # 상대 하한: regression_weight가 클수록(=FICR 항 비중이 작을수록) 하한도 낮춘다
        hess = np.maximum((1.0 - regression_weight) * np.abs(dd_ficr), hess_floor)
        return grad, hess
    return objective
```

**[기술적 세부 사양 — 2단계 웜업(LSTM 정렬)]** LSTM은 `warmup_epochs=15`까지 SmoothL1
단독으로 학습해 초기 큰 오차에서 시그모이드가 포화되는 것을 막는다. LightGBM은 표본별
부스팅이라 "epoch 워밍업" 개념이 없지만, **부스팅 라운드를 2단계로 나눠 동일한 설계 원칙을
적용**한다(`baseline7_idea_evaluation.md` 4-3절 권고).

```python
def trainFicrLightGBM(X_train, y_train, X_valid, y_valid, capacity_kw: float,
                       warmup_rounds: int = 200, k: float = 40.0,
                       regression_weight: float = 0.7) -> lgb.LGBMRegressor:
    """1단계 L1 웜업 → 2단계 FICR 혼합 목적함수로 이어 학습하는 함수

    Logic:
        - 1단계(warmup_rounds): objective="regression_l1"로 대략적인 예측 수준을 먼저 잡는다
          (LSTM의 SmoothL1 워밍업과 동일한 목적)
        - 2단계: init_model=1단계 booster로 이어받아 makeFicrObjective로 계속 부스팅한다
        - min_sum_hessian_in_leaf를 기본값(1e-3)보다 높여(1e-2~1e-1) hess 총합이 작은 리프의
          추가 분할을 억제한다(baseline7_idea_evaluation.md 4-2절 완화책)
    """
    warmup_params = {**LGB_PARAMS, "objective": "regression_l1", "n_estimators": warmup_rounds}
    model_warmup = lgb.LGBMRegressor(**warmup_params)
    model_warmup.fit(
        X_train, y_train / capacity_kw,
        eval_X=X_valid, eval_y=y_valid / capacity_kw,
        eval_metric="l1", callbacks=[lgb.log_evaluation(0)],
    )

    ficr_params = {
        **LGB_PARAMS, "objective": makeFicrObjective(k, regression_weight),
        "min_sum_hessian_in_leaf": 0.05,
    }
    model_ficr = lgb.LGBMRegressor(**ficr_params)
    model_ficr.fit(
        X_train, y_train / capacity_kw,
        eval_X=X_valid, eval_y=y_valid / capacity_kw,
        eval_metric="l1", init_model=model_warmup.booster_,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model_ficr
```

이를 `lgb_models: dict[...] = {}`가 있는 셀(Step 2 루프)에서 `model_selected = trainLightGBM(...)`
블록 다음 세 번째 후보로 추가한다.

```python
model_ficr = trainFicrLightGBM(
    X_train[columns], y_train, X_valid[columns], y_valid,
    capacity_kw=RATED_CAPACITY_KW[group], k=40.0, regression_weight=0.7,
)
lgb_models[(group, "LGBM_ficr")] = model_ficr
pred_ficr = predictKw(model_ficr, X_valid[columns], capacity_kw)[SEQ_LEN:]
summary_ficr = summarizeAll(pred_ficr, actual_aligned, group)
predictions_store[(group, "LGBM_ficr")] = (pred_ficr, actual_aligned)
records.append({
    "group": group, "model_name": "LGBM_ficr", "n_features": len(columns),
    "best_epoch": model_ficr.best_iteration_, **summary_ficr,
})
```

`regression_weight`는 원안대로 0.5/0.7/0.9 세 값으로 우선 비교하되(0.5 미만은
`improvement.md` 2-1절이 지적한 초기 포화 함정 때문에 쓰지 않는다), **카드 D가 이 값과
`k`를 GWO로 연속 탐색하므로 이 카드에서는 초기 감(sense)만 잡고 최종값은 카드 D 결과로
대체한다.**

**[데이터 누수 방지책]** 웜업·본학습 모두 `X_train`/`y_train`(학습 구간)에만 `fit`하고,
`eval_X`/`eval_y`(검증 구간)는 조기종료 판단에만 쓴다 — 이 규칙은 `trainLightGBM`과 동일하다.
`predictKw`의 `np.clip(..., 0.0, 1.0)` 클리핑은 커스텀 목적함수를 써도 그대로 유지한다.

**[검증 가설 및 KPI]**
- 카드 A의 `summarizeErrorBand`를 `LGBM_ficr`에도 적용해 `>8%` 비율이 `LGBM_selected`보다
  줄었는지 확인한다(총점이 같아도 밴드 분포가 개선됐다면 카드 F의 앙상블 후보로 유효하다).
- 학습 로그에서 `model_ficr`의 검증 손실(`l1`)이 발산하지 않고 단조 또는 준단조로 감소하는지
  확인한다 — 발산하면 §4 리스크 관리의 "hess 수렴 실패" 대응을 적용한다.
- 최종 채택은 검증 총점이 기준선을 넘는지로 판단한다.

---

### 카드 D. [P1] 지능형 하이퍼파라미터 최적화 (GWO 기반)

**[목적]** 기존 그리드 탐색(3×3=9격자)은 두 파라미터의 실제 최적점이 격자점 사이에 있으면
놓치고, 카드 C의 LGBM `k`/`regression_weight`까지 같은 방식으로 넓히면 탐색점이 선형으로
늘어나 실행시간 예산을 압박한다. `.venv`에 `optuna`/`scikit-optimize` 등 베이지안 최적화
라이브러리가 설치되어 있지 않음을 실측으로 확인했으므로(§4 리스크 관리), **GWO(Grey Wolf
Optimizer)를 순수 numpy로 구현**해 연속 공간을 탐색한다(`baseline7_idea_evaluation.md`
아이디어 6).

**[기술적 세부 사양]**

1. `MODEL_PARAMS = {`가 있는 셀 근처에 GWO 클래스를 추가한다.

   ```python
   class GreyWolfOptimizer:
       """GWO(후보 설정을 무리처럼 비교하는 방법)로 하이퍼파라미터를 탐색하는 클래스

       사용 순서:
           gwo = GreyWolfOptimizer(bounds=[(10,100),(0.5,0.95)], n_wolves=5, max_iter=4, seed=SEED)
           best_params, best_score, history = gwo.optimize(objective_fn)

       Logic:
           - 개체(늑대) 위치 = 하이퍼파라미터 벡터. alpha/beta/delta(상위 3개체)의 위치를
             기준으로 나머지 개체가 이동하는 군집 지능 알고리즘
           - a는 매 반복 2에서 0으로 선형 감소 (탐색→수렴 전환을 제어하는 유일한 스케줄 파라미터)
           - objective_fn은 "클수록 좋음"(검증 총점)을 반환한다고 가정하고 내부에서 최대화한다
       """
       def __init__(self, bounds: list[tuple[float, float]], n_wolves: int = 5,
                    max_iter: int = 4, seed: int = SEED):
           self.bounds = np.asarray(bounds, dtype=np.float64)
           self.n_wolves, self.max_iter = n_wolves, max_iter
           self.rng = np.random.default_rng(seed)

       def _clip(self, pos: np.ndarray) -> np.ndarray:
           return np.clip(pos, self.bounds[:, 0], self.bounds[:, 1])

       def optimize(self, objective_fn):
           dim = len(self.bounds)
           lo, hi = self.bounds[:, 0], self.bounds[:, 1]
           positions = lo + self.rng.random((self.n_wolves, dim)) * (hi - lo)
           # 탐색 범위 내 균등 무작위 초기화

           scores = np.array([objective_fn(p) for p in positions])
           history = [{"iter": 0, "best_score": float(scores.max())}]

           for it in range(self.max_iter):
               order = np.argsort(-scores)
               # 검증 총점 내림차순: 상위 3개체(alpha/beta/delta)가 나머지를 이끈다
               alpha, beta, delta = positions[order[0]], positions[order[1]], positions[order[2]]
               a = 2.0 - 2.0 * it / max(1, self.max_iter - 1)
               # a: 반복이 진행될수록 2->0으로 선형 감소, 탐색(exploration)에서 수렴(exploitation)으로 전환

               for i in range(self.n_wolves):
                   new_pos = np.zeros(dim)
                   for leader in (alpha, beta, delta):
                       r1, r2 = self.rng.random(dim), self.rng.random(dim)
                       A = 2 * a * r1 - a
                       C = 2 * r2
                       D = np.abs(C * leader - positions[i])
                       new_pos += leader - A * D
                   positions[i] = self._clip(new_pos / 3.0)
                   # alpha/beta/delta 각각이 이끄는 위치의 평균으로 이동 (표준 GWO 갱신식)

               scores = np.array([objective_fn(p) for p in positions])
               history.append({"iter": it + 1, "best_score": float(scores.max())})

           best_idx = np.argmax(scores)
           return positions[best_idx], float(scores[best_idx]), history
   ```

2. 그룹별로 이미 `best_by_group`에 뽑힌 버전 1개만 재탐색한다(원안과 동일한 범위 제한 —
   그룹1: `v1`, 그룹2: `v2`, 그룹3: `v1`). 탐색 공간은 `loss_k ∈ [10, 100]`,
   `regression_weight ∈ [0.5, 0.95]` 2차원.

   ```python
   grid_targets = {1: "v1", 2: "v2", 3: "v1"}
   gwo_rows = []

   for group, version in grid_targets.items():
       data = datasets[group]
       columns = selected_features[group]
       X_train, X_valid = data["X_train"][columns], data["X_valid"][columns]
       y_train, y_valid = data["y_train"], data["y_valid"]
       variant = AllFeaturesVariant(random_state=SEED)
       train_frame = variant.fit_transform(X_train, y_train)
       valid_frame = variant.transform(X_valid)
       scaler = MinMaxScaler().fit(train_frame)
       X_train_scaled, X_valid_scaled = scaler.transform(train_frame), scaler.transform(valid_frame)
       monitor = OfficialScoreMonitor(metrics, group)

       def objective_fn(params, group=group, version=version):
           loss_k, regression_weight = float(params[0]), float(params[1])
           pipeline = make_lstm_pipeline(
               version, capacity_kw=RATED_CAPACITY_KW[group], seq_len=SEQ_LEN,
               **{**MODEL_PARAMS, "loss_k": loss_k, "regression_weight": regression_weight},
           )
           pipeline.fit(X_train_scaled, y_train, X_val=X_valid_scaled, y_val=y_valid,
                        metrics=monitor, group=group)
           pred = pipeline.predict(X_valid_scaled, y_valid)
           actual = y_valid[SEQ_LEN:]
           return metrics.summarize(pred, actual, group)["score"]

       gwo = GreyWolfOptimizer(bounds=[(10.0, 100.0), (0.5, 0.95)], n_wolves=5, max_iter=4, seed=SEED)
       best_params, best_score, history = gwo.optimize(objective_fn)
       gwo_rows.append({
           "group": group, "version": version,
           "loss_k": best_params[0], "regression_weight": best_params[1],
           "best_score": best_score, "n_evals": len(history) * 5,
       })
       # n_evals: 개체 수 x (반복 수 + 초기화 1회)만큼 objective_fn이 호출된 횟수 — 실행시간 추정용

   gwo_df = pd.DataFrame(gwo_rows)
   display(gwo_df)
   ```

3. **LGBM `k`/`regression_weight`도 동일한 GWO 루프로 탐색**(카드 C의 `trainFicrLightGBM` 호출을
   `objective_fn`으로 감싸기만 하면 된다). LightGBM은 LSTM보다 학습이 훨씬 빠르므로
   `n_wolves=8, max_iter=6`으로 예산을 더 크게 잡을 수 있다.

**[데이터 누수 방지책]** `objective_fn`은 검증 구간(`X_valid`/`y_valid`) 총점만 반환한다 —
평가(2025년) 구간은 어떤 형태로도 GWO 탐색에 노출되지 않는다. GWO의 무작위 초기화·개체 이동은
`SEED`로 고정해 재현성을 보장한다.

**[검증 가설 및 KPI]**
- 그룹별 GWO 실행이 예상 시간(§1-2 표, 대략 20~27분/3그룹) 이내에 끝나야 한다. **실측 후 1회
  반복 소요 시간을 확인하고, 예상 총 소요가 20분을 넘으면 즉시 `n_wolves=4, max_iter=3`으로
  축소한다**(`baseline7_idea_evaluation.md` 6-3절 축소 규칙).
- 그룹별 `best_score`가 기존 그리드 탐색 결과(카드 D 원안, 9격자)의 최고값 이상인지 비교한다
  — 낮으면 GWO가 그리드보다 나쁜 지역해에 수렴한 것이므로 초기화를 다르게 하여 재시도하거나
  FA(Firefly Algorithm)로 대체한다(§4 리스크 관리).
- `history`의 `best_score`가 반복이 진행될수록 비감소(non-decreasing)인지 확인한다(엘리트
  보존이 없는 표준 GWO는 이론상 감소할 수 있으므로, 감소가 관측되면 `positions[best_idx]`
  대신 `history` 전체에서 최댓값을 채택하도록 보정한다).

---

### 카드 E. [P2] 시각대(diurnal) 오차 진단 매트릭스

**[목적]** `lead_hour`는 이 데이터셋에서 시각(hour-of-day)의 결정론적 아핀 변환과 동일한
정보이고(`baseline7_idea_evaluation.md` 2-1절, 상관계수 1.0 실측), 이미 SHAP 선별을 통과해
두 모델 계열 모두의 학습 입력에 들어가 있다(2-2절). 따라서 이 카드의 실제 작업은 "모델에
새 정보를 주는 것"이 아니라 **시간별 오차율의 밴드 분포가 시각대별로 실제로 불균일한지
진단**하고, 불균일할 때만 카드 B/C를 시각대 차원으로 확장하는 것이다.

**[기술적 세부 사양]**

1. `predictions_store`와 각 그룹 `X_valid`(또는 `datasets[group]["df"]`)의 `target_hour_ldaps`를
   시간 인덱스로 정렬해 이어붙인다.
2. 3구간(단기 `01~08시`, 중기 `09~16시`, 장기 `17~24시`)으로 나누고, 구간별로 카드 A의
   `summarizeErrorBand`를 재계산한다. (그룹 × best_by_group 모델 1개 × 3시각대 = 9행이면 충분)

   ```python
   hour_bins = pd.cut(
       target_hour, bins=[0, 8, 16, 24],
       labels=["단기(01~08시)", "중기(09~16시)", "장기(17~24시)"],
   )
   diurnal_rows = []
   for bucket in hour_bins.cat.categories:
       mask = hour_bins == bucket
       share = summarizeErrorBand(pred[mask], actual[mask], group)
       diurnal_rows.append({"group": group, "hour_bucket": bucket, **share.to_dict()})
   diurnal_df = pd.DataFrame(diurnal_rows)
   display(diurnal_df)
   ```

3. **불균일 판정**: 어떤 시각대의 `>8%` 비율이 다른 시각대 평균보다 **1.5배 이상** 크면
   "불균일"로 판정한다.
   - 불균일이면: 카드 B의 `DiurnalFicrCalibrator`(시각대별 (a,b))와 카드 C의 `sample_weight`
     가중(`>8%` 비율이 높은 시각대 행에 1.2~1.5배 가중) 두 가지를 각각 시도해 더 큰 개선을
     보인 쪽만 채택한다.
   - 불균일이 아니면: 카드 E는 진단 단계에서 종료하고 완전 분리 모델은 시도하지 않는다
     (`baseline7_idea_evaluation.md` 2-4절 ROI 표 — 데이터 손실 대비 효용이 낮다는 결론을
     그대로 따른다).

**[데이터 누수 방지책]** 시각대 구간 경계(`01~08/09~16/17~24`)는 고정 상수이며 데이터로부터
학습되지 않는다. 시각대별 보정·가중 탐색은 카드 B/C와 동일하게 검증 구간에서만 수행한다.

**[검증 가설 및 KPI]**
- `diurnal_df`의 시각대별 `>8%` 비율과 (최대/평균) 비율을 실행 로그에 남긴다.
- 두 저비용 방법(시각대별 보정, 시각대별 가중) 모두 기준선을 넘지 못하면 카드 E는 여기서
  종료하고 완전 분리 모델을 시도하지 않는다.
- 완전 분리 모델까지 고려하는 경우(불균일이 2배 이상이고 두 저비용 방법 모두 목표 미달일
  때만, 마지막 수단), 그룹3처럼 학습 데이터가 이미 적은 그룹(2023~2024만 2년치)은 시각대
  분리 후 표본 수가 과적합 위험 수준(연 단위 데이터 2년 × 1/3 시각대 ≈ 5,800시간)인지 먼저
  확인한다.

---

### 카드 F. [P2] 시각대별 앙상블 + 그룹별 최종 후보 채택 + 제출 반영

**[목적]** 카드 A~E에서 나온 모든 후보 — 원본(`LGBM_selected`/`LSTM_*`), 카드 B 보정 적용본
(전역·시각대별), 카드 C `LGBM_ficr`, 카드 D로 재탐색한 LSTM/LGBM — 를 그룹별로 비교해 최종
제출 모델을 확정한다.

**[기술적 세부 사양]**

1. 카드 A~E의 모든 후보를 그룹별로 검증 총점 내림차순 정렬한다.
2. **앙상블(가중 블렌딩)은 후보가 이미 2개 이상 기준선을 넘었을 때만 시도한다.** 가중치는
   단순 격자탐색으로 충분하다.

   ```python
   ensemble_rows = []
   for w in np.arange(0.0, 1.01, 0.05):
       blended = w * pred_a + (1 - w) * pred_b
       # pred_a, pred_b는 같은 그룹의 서로 다른 두 후보(검증 구간, predictions_store에서 조회)
       score = metrics.summarize(blended, actual, group)["score"]
       ensemble_rows.append({"group": group, "w": w, "score": score})
   ensemble_df = pd.DataFrame(ensemble_rows).sort_values("score", ascending=False)
   ```

   가중치는 검증 예측에서만 탐색하고, 앙상블 총점이 **두 후보 각각보다 낮으면 채택하지 않는다.**
   카드 E에서 시각대 불균일이 확인된 그룹은, 전역 가중치 하나 대신 시각대별로 다른 `w`를
   탐색하는 것도 고려할 수 있다(표본이 충분한 경우에 한함).
3. 그룹별 최종 채택 모델·보정 파라미터를 Step 5(`def fitFinalLightGBM(`가 있는 셀)의 해당
   그룹 분기에 반영한다.
   - `LGBM_ficr`가 채택되면: `fitFinalLightGBM` 호출 시 `trainFicrLightGBM`(카드 D가 찾은 최적
     `k`/`regression_weight` 사용)으로 그 그룹 분기만 교체한다.
   - 카드 B 보정이 채택되면: `pred = predictKw(...)` 또는 `pred = final_model.predict(...)`
     직후에 `pred = calibrator.transform(pred)`(또는 시각대별이면
     `calibrator.transform(pred, hour_bucket)`)를 추가한다. 검증 구간에서 이미 적합한 파라미터를
     그대로 재사용하고, 평가 구간에서 다시 `fit()`하지 않는다.
   - 앙상블이 채택되면: 두 최종 모델을 각각 전체 재학습해 `pred = w*pred_a + (1-w)*pred_b`로
     합친다. `w`는 검증 구간에서 찾은 값을 고정 상수로 쓴다.

**[데이터 누수 방지책]** 기존 Step 5 검증 로직(`len(pred) != len(test_df)` 에러, 스키마·결측·
물리범위·첫 24시간 검증 셀)을 그대로 통과해야 한다. 새 로직을 추가했다고 이 가드들을 느슨하게
풀지 않는다.

**[검증 가설 및 KPI]**
- 최종 `submission_baseline7.csv` 재생성 후, 그룹별 값이 0이 아니고 설비용량 이하인지
  마지막 검증 셀로 재확인한다.
- 3그룹 평균 검증 총점을 §1-2의 Phase 3 목표(0.68 잠정치)와 비교해 보고한다.

---

### 카드 G. [P3] VMD 입력 피처 분해 + mRMR 기반 IMF 특징 선택 (선택적 탐색 스파이크)

**[목적]** `ws10`(LDAPS IDW 대표 풍속)·`gfs_ws_hub`(GFS 허브높이 외삽 풍속)를 VMD로 K개
모드로 분해해 추세·고주파 성분을 파생 피처로 추가하고, 상관된 모드 다발이 기존 SHAP 선별
(`FeatureSelector`)에서 "집단 탈락"하는 것을 막기 위해 **mRMR로 사전 축소**한 뒤 기존
`LGBM_selected` 파이프라인에 편입한다. 새 모델 계열을 만드는 것이 아니라 **기존 파이프라인에
피처를 더했을 때의 순수 효과**를 측정하는 것이 목적이다(`baseline7_idea_evaluation.md`
아이디어 1, 5).

**[기술적 세부 사양 — 절대 하지 말 것]**
- 타깃(발전량) 시계열을 VMD로 분해해 모드별로 자기회귀 예측하지 않는다 — day-ahead 일괄
  제출 구조상 제출 불가능하다(`Persistence_oracle_lag1`과 동일한 결함).
- 학습 구간 + 검증 구간을 합쳐 **한 번에** VMD를 돌리지 않는다 — 검증 구간의 스펙트럼 정보가
  학습 구간 모드 값에 섞여 들어가는 누수다.

**[기술적 세부 사양 — 절차]**

1. **분해 대상**: 그룹별 `ws10`, `gfs_ws_hub` 두 컬럼만(전체 피처를 다 분해하지 않는다).
2. **VMD 누수 차단**: 학습 구간(80%)만으로 적합한 VMD 결과를 학습 피처로 쓰고, 검증·평가
   구간은 확장 윈도우(주 단위 재적합)를 쓴다. 재적합 윈도우의 가장 최근(오른쪽 끝) 지점은
   경계 왜곡이 가장 크다는 것을 알고 진행한다(완화 불가능한 구조적 한계).
3. **파라미터**: `K ∈ {4,6,8}`, `α ∈ {1000,2000}` 6조합. `vmdpy` 사용(순수 numpy/scipy 기반).
   **재구성 오차가 아니라 검증 총점**으로 조합을 고른다.
4. **mRMR 특징 선택 (신규)** — VMD가 만든 최대 16개(`vmd_ws10_mode1..K`,
   `vmd_gfs_ws_hub_mode1..K`) 신규 컬럼을 기존 SHAP 선별에 바로 태우지 않고, 먼저 학습 구간
   에서만 mRMR로 축소한다.

   ```python
   from sklearn.feature_selection import mutual_info_regression

   def selectImfFeaturesMrmr(imf_df: pd.DataFrame, y_train: np.ndarray,
                              top_k: int = 6, seed: int = SEED) -> list[str]:
       """VMD의 IMF(시계열을 나눈 한 조각)를 mRMR(관련성은 높고 중복은 낮게)로 고르는 함수

       Args:
           - imf_df: 학습 구간의 IMF 컬럼만 담은 DataFrame(vmd_ws10_mode1..K 등)
           - y_train: 학습 구간 타깃(발전량 kWh 또는 이용률)
           - top_k: 최종적으로 남길 IMF 개수

       Logic:
           - relevance: mutual_info_regression(imf, y_train) — 비선형 관계도 포착
           - redundancy: 이미 선택된 IMF들과의 평균 절대 상관계수
           - 매 스텝 (relevance - redundancy)가 최대인 컬럼을 그리디하게 추가
           - 반드시 학습 구간(imf_df, y_train)에서만 계산한다. 선택된 컬럼 이름 집합은
             검증·평가 구간에도 고정 적용하고, 재적합 윈도우가 바뀌어도 다시 뽑지 않는다
             (baseline7_idea_evaluation.md 5-3절 누수 방지 규칙)
       """
       relevance = pd.Series(
           mutual_info_regression(imf_df, y_train, random_state=seed),
           index=imf_df.columns,
       )
       selected: list[str] = [relevance.idxmax()]
       remaining = [c for c in imf_df.columns if c not in selected]

       while len(selected) < min(top_k, len(imf_df.columns)) and remaining:
           redundancy = {
               c: imf_df[selected].corrwith(imf_df[c]).abs().mean() for c in remaining
           }
           mrmr_score = {c: relevance[c] - redundancy[c] for c in remaining}
           best = max(mrmr_score, key=mrmr_score.get)
           selected.append(best)
           remaining.remove(best)
       return selected
       # 선택된 IMF 컬럼명 리스트. 이후 datasets[group]["df"]에 이 컬럼만 병합한다
   ```

   `mutual_info_regression` 결과가 비정상적으로 균일하면(예: 모든 IMF 관련성 점수가 거의
   동일) PCC-GRA로 대체한다(§4 리스크 관리).
5. **모델 통합**: mRMR로 축소한 IMF 컬럼만 `datasets[group]["df"]`에 병합한 뒤, Step 2
   (`LGBM_full`/`LGBM_selected` 학습 셀)를 그대로 재실행해 기존 SHAP 선별 대상에 자연스럽게
   포함시킨다. 별도의 "VMD 전용 모델"을 새로 만들지 않는다.

**[데이터 누수 방지책]**
- VMD: 학습 구간(80%)에서만 최초 적합, 검증·평가 구간은 주 단위 확장 윈도우 재적합(매
  시각마다 재적합하지 않는다).
- mRMR: relevance·redundancy 계산 모두 학습 구간 `imf_df`/`y_train`에서만 수행하고, 선택된
  컬럼 이름 집합을 검증·평가 구간에 고정 적용한다(구간마다 다시 뽑지 않는다).
- `coding_convention.md` 7-3절(피처 선정 시 식별자·타깃 제외)과 동일한 원칙을 적용한다.

**[검증 가설 및 KPI]**
- **1차 게이트**: 그룹1 `LGBM_selected` 검증 총점이 VMD+mRMR 피처 추가 전(0.6555) 대비
  **+0.005 이상** 개선되는지. 이 문턱을 넘지 못하면 이후 카드(그룹 2·3 확장, LSTM 통합)를
  진행하지 않는다.
- **2차 확인**: 카드 A의 `summarizeErrorBand`를 VMD+mRMR 피처 포함/미포함 두 모델에 각각
  적용해 `>8%` 비율이 실제로 줄었는지 확인한다(총점 개선이 NMAE 쪽에서만 온 것이 아닌지 분리).
- **경계 왜곡 진단**: 각 재적합 윈도우의 마지막 24시간과 그 이전 구간의 모드 값 분산을
  비교해, 마지막 24시간이 비정상적으로 크면(예: 2배 이상) 그 구간의 VMD 피처를 결측 처리하고
  원본 풍속 피처로 폴백하는 로직이 필요하다는 신호로 기록한다.
- **mRMR 축소 확인**: `selectImfFeaturesMrmr`가 반환한 컬럼이 최종 SHAP 선별(`FeatureSelector`)
  에서 몇 개나 살아남는지 보고한다 — 0개면 mRMR 축소가 무의미했다는 뜻이므로 `top_k`를
  늘리거나 PCC-GRA로 재시도한다.
- 그룹1에서 유의미한 개선이 없으면 **그룹 2·3으로 확장하지 않고 종료**한다.

---

## 3. 구현 위치와 흐름

기존 전처리와 Step 2~5의 뼈대는 유지한다. 신규 코드는 `baseline_7.ipynb`의 셀로만 추가한다.
실행 흐름은 `A(예측 저장·진단) → B(저비용 보정) 또는 C(재학습) → D(필요 시 HPO) →
E(시각대 진단) → F(최종 채택·제출)`이다. 카드 G는 선택한 경우에만 Step 2 전에 VMD·mRMR 피처를
추가한다.

---

## 4. 리스크 관리 및 Fallback 전략

| 리스크 | 영향 카드 | 감지 신호 | 대응(Fallback) |
|---|---|---|---|
| VMD 연산 지연(재적합 비용이 예산 초과) | G | 주 단위 재적합 전체 소요가 카드 I 실행시간 예산(≈34분)을 크게 초과 | 재적합 주기를 월 단위로 낮춘다. 그래도 초과하면 카드 G를 P3에서 보류(deprioritize)하고 카드 A~F만 우선 완료 |
| `vmdpy` 설치 실패(환경 제약) | G | `pip install vmdpy` 실패 | PyEMD의 CEEMDAN 등 대체 라이브러리로 1회 대체 시도, 그래도 실패하면 카드 G 전체를 보류하고 보고 |
| VMD 경계 왜곡이 완화되지 않음(구조적 한계) | G | 재적합 윈도우 마지막 24시간의 모드 분산이 그 이전 구간 대비 2배 이상 | 최근 24시간 VMD 피처를 결측 처리 → 원본 풍속 피처로 자동 폴백하는 가드를 병합 셀에 넣는다 |
| mRMR 결과가 비정상적으로 균일(상호정보량 추정 불안정) | G | `selectImfFeaturesMrmr`의 relevance 값들이 서로 거의 구분되지 않음(표준편차가 평균의 5% 미만 등) | PCC-GRA(Pearson 상관 + Grey Relational Analysis)로 대체 계산(`baseline7_idea_evaluation.md` 아이디어 5-2) |
| 커스텀 `hess`가 문턱 근처 표본에서 급변해 리프 가중치 발산(수렴 실패) | C | 학습 로그에서 검증 `l1` 손실이 발산하거나 진동, `model_ficr`의 `best_iteration_`이 비정상적으로 작음 | (1) `min_sum_hessian_in_leaf`를 0.05→0.1로 상향 (2) `hess` 상대 하한을 1e-2→5e-2로 상향 (3) 그래도 불안정하면 2단계 웜업 라운드 수를 200→400으로 늘려 1단계에서 더 안정된 초기 모델을 확보 |
| GWO가 그리드 탐색보다 나쁜 지역해에 수렴 | D | `gwo_df`의 `best_score`가 카드 D 원안(그리드 9격자)의 최고값보다 낮음 | (1) `n_wolves`를 늘려 다른 시드로 재시도 (2) Firefly Algorithm(FA)으로 대체 — 광 흡수계수 `γ`, 매력도 `β₀`는 문헌 기본값(`γ=1.0`, `β₀=1.0`)에서 시작 (3) 그래도 실패하면 카드 D 원안(그리드 9격자)으로 폴백 — 이미 전체 문서에 구현되어 있으므로 즉시 전환 가능 |
| GWO/HPO 탐색이 실행시간 예산(20분) 초과 | D | 1회 반복 실측 시간 × `max_iter` × 그룹 수가 20분을 넘을 것으로 예상 | `n_wolves=5→4`, `max_iter=4→3`으로 축소(그룹당 20회→12회 평가). 그래도 초과하면 그룹당 재탐색을 1개 버전에서 가장 시급한(카드 A `recommendAction`이 "카드 C 우선"인) 그룹으로만 한정 |
| 사후 보정(카드 B) 시각대 확장 시 특정 시각대 표본 부족 → 과적합 | B/E | `DiurnalFicrCalibrator`의 특정 구간 표본 수가 500시간 미만 | `A_GRID`/`B_RATIO_GRID`의 격자 수를 21×21→11×11로 줄이거나, 인접 시각대와 통합해 2구간으로 낮춘다 |
| 앙상블(카드 F) 채택 후 총점이 개별 후보보다 낮음 | F | `ensemble_df`의 최고 `score`가 두 후보 각각의 단독 검증 총점보다 낮음 | 즉시 폐기하고 더 높은 단독 후보를 채택한다(앙상블은 항상 선택적 시도이지 필수 단계가 아니다) |
| 신규 의존성(`optuna`, `vmdpy` 등) 설치가 필요한데 자동 설치가 부적절 | D, G | `.venv`에 해당 패키지가 없음을 사전 확인(§본 문서 작성 시점 실측: 없음) | **사용자에게 설치 여부를 먼저 확인한다.** 이 저장소는 신규 의존성을 자동으로 추가하지 않는 것이 원칙이며, GWO(카드 D)·`vmdpy`(카드 G, 이미 별도 fallback 있음) 모두 최소 의존성 경로가 이미 마련되어 있으므로 대부분의 경우 설치 자체가 불필요하다 |

---

## 5. 실행 순서와 중단 기준

| 순서 | 할 일 | 다음 단계로 가는 조건 | 남기는 결과 |
|---|---|---|---|
| P0 | 카드 A: 예측 저장·오차 밴드 진단 | `band_df`와 권고 조치 확인 | `predictions_store`, `band_df` |
| P1 | 카드 B 전역 보정 → 카드 C FICR 목적함수 → 필요 시 카드 D HPO | 그룹별 총점이 기존 기준선보다 높은 후보만 유지 | `calibration_df`, `LGBM_ficr`, `gwo_df` |
| P2 | 카드 E 시각대 진단 → 카드 F 최종 후보·앙상블·제출 | 불균일할 때만 B/C 시각대 확장; 최종 평균이 0.6672보다 높아야 함 | `diurnal_df`, `ensemble_df`, `submission_baseline7.csv` |
| P3 (선택) | 카드 G VMD+mRMR 피처 실험 | 그룹1 총점이 기준선보다 `+0.005` 이상일 때만 그룹 2·3으로 확장 | 게이트 통과 여부와 비교 결과 |

카드 B는 비용이 낮아 먼저 시도한다. 카드 D와 G는 시간이 많이 들 수 있으므로 앞 단계의 총점
개선이 확인된 경우에만 진행한다. 어떤 카드든 기준선을 넘지 못하면 그 카드에서 멈추고 기존
`best_by_group` 후보를 유지한다.

---

## 부록 A. 코드 앵커 문자열 목록

노트북 편집 시 셀을 찾는 데 쓰는 고유 문자열과, 그 문자열이 속한 로직 단위를 정리한다.
(노트북 실행 시 셀 인덱스가 바뀌므로 인덱스가 아니라 이 문자열로 셀을 찾는다.)

| 앵커 문자열 | 위치(로직 단위) | 관련 카드 |
|---|---|---|
| `"lgb_models: dict[tuple[int, str], lgb.LGBMRegressor] = {}"` | Step 2 LightGBM 학습 루프 시작 | A, C |
| `"LGB_PARAMS = {"` | LightGBM 공통 하이퍼파라미터 정의 | C |
| `"class FeatureSelector:"` | null importance + 상관 중복 기반 SHAP 선별 | G |
| `"MODEL_PARAMS = {"` | Step 3 LSTM 학습 루프 시작 | A, D |
| `"class OfficialScoreMonitor:"` | LSTM 조기종료 기준(공식 총점) | D |
| `"def aggregateOfficialScore(result_df: pd.DataFrame) -> pd.DataFrame:"` | Step 4 집계 | A |
| `"best_by_group = ("` | 그룹별 검증 총점 최고 모델 선택 | A, F |
| `"def fitFinalLightGBM("` | Step 5 최종 재학습·제출 생성 | C, F |

## 부록 B. 공통 준수 사항

- 코드 스타일: [`MD/coding_convention.md`](coding_convention.md) — 설명 주석은 코드 줄 아래,
  단위 명시, `df.copy()` 선행, 시간순 분할.
- 새 실험 결과는 `baseline_7_results.csv`와 같은 스키마
  (`group, model_name, n_features, best_epoch, nmae, ficr, score, mae, rmse`)로 추가 저장해
  `aggregateOfficialScore`로 재집계 가능하게 한다.
- **모든 보정·앙상블·HPO 탐색은 검증 구간(뒤 20%)에서만 한다.** 평가(2025년) 구간에는
  검증에서 고정한 파라미터를 그대로 적용만 한다.
- **채택 기준은 §1-1의 그룹별 기준선**(그룹1 0.6555 / 그룹2 0.6877 / 그룹3 0.6583)이다. 이를
  넘지 못하면 그 그룹은 기존 `best_by_group` 선택을 유지한다.
- `Persistence_oracle_lag1`은 비교 기준으로만 쓴다. 직전 실제값을 쓰므로 2025년 제출에는
  사용할 수 없다.
- 신규 의존성(`vmdpy` 등)이 필요한 카드는 설치 전 사용자에게 확인한다 — 이 저장소는 자동
  `pip install`을 원칙으로 하지 않는다.

## 부록 C. 초심자용 주석 용어 정의

노트북 주석이나 독스트링에서 아래 용어가 처음 나올 때 괄호 안의 짧은 설명을 함께 쓴다.

| 용어 | 주석에 붙일 쉬운 정의 |
|---|---|
| FICR | 오차가 작을수록 발전량에 더 높은 정산단가를 적용하는 점수 |
| NMAE | 설비용량으로 나눈 평균 절대 오차. 값이 작을수록 좋다 |
| 기준선(baseline) | 새 방법과 비교하는 현재 최고 결과 |
| 검증 구간(hold-out) | 학습에 사용하지 않고 성능을 확인하는 뒤 20% 데이터 |
| 평가 구간(test) | 실제 제출을 만드는 구간. 정답을 모르므로 튜닝에 쓰지 않는다 |
| 데이터 누수(leakage) | 미래나 정답 정보를 학습·튜닝에 미리 섞는 실수 |
| 보정(calibration) | 예측값에 간단한 변환을 적용해 실제값에 더 가깝게 맞추는 작업 |
| 목적함수(objective) | 모델이 학습 중 더 좋게 만들려고 최소화·최대화하는 기준 |
| 웜업(warm-up) | 본 학습 전에 쉬운 설정으로 모델을 안정시키는 준비 단계 |
| 하이퍼파라미터(HPO) | 학습 전에 사람이 정하는 설정값을 시험하는 과정 |
| GWO | 여러 후보 설정을 늑대 무리처럼 움직이며 탐색하는 최적화 방법 |
| VMD / IMF | 시계열을 느린 흐름과 빠른 흔들림으로 나누는 방법 / 그 결과 조각 |
| mRMR | 정답과 관련이 크면서 서로 비슷하지 않은 피처를 고르는 방법 |
| 앙상블(blending) | 여러 모델의 예측을 가중 평균해 하나로 합치는 방법 |
| 폴백(fallback) | 새 방법이 실패하거나 불안정할 때 기존 방법으로 돌아가는 처리 |
| `fit` / `transform` | `fit`은 규칙을 데이터에서 배우는 단계, `transform`은 배운 규칙을 적용하는 단계 |
