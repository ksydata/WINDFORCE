# 전처리 모듈화 변경사항

[03_Preprocessing.ipynb](../03_Preprocessing.ipynb)에서 만든 전처리 로직을
[Windforce/](../Windforce/) 패키지로 옮겼다. 노트북은 그대로 두고(탐색 기록이라 지우지 않는다)
모델링 노트북이 `from Windforce import ...` 한 줄로 받아쓸 수 있게 만드는 것이 목적이다.

전처리의 **최종 목표는 LSTM이 바로 받는 3차원 배열**이다. 그래서 파이프라인이
격자 → 그룹 → 시간정렬 표 → `(samples, timesteps, features)` 순으로 좁혀 들어간다.

---

## 1. 파일 구조

```
Windforce/
├── FeatureEngineer.py         추상 클래스. 8단계 템플릿 + 공통 물리계산 + LSTM 시퀀스 생성
├── LDAPSFeatureEngineer.py    16격자 예보
├── GFSFeatureEngineer.py      9격자 예보 (신규)
├── SCADAFeatureEngineer.py    10분 실측
├── FeatureEngineerFactory.py  source·group 문자열로 위 셋을 생성
└── __init__.py                공개 API 38개
```

[MD/coding_convention.md](./coding_convention.md)를 따랐다. 클래스 1개 = 파일 1개,
PascalCase 파일명, 주석은 코드 **아래** 줄, 고정 동사 어휘(`compute`/`transform`/`calculate`/`check`),
`Args:`에 단위·shape, `Logic:`에 수식, 키워드 인자 앞뒤 공백.

## 2. 노트북 클래스 → 패키지 대응표

| 노트북 03 | 옮긴 곳 | 비고 |
|---|---|---|
| `WindVectorTransformer` | `FeatureEngineer` | 추상 클래스로 승격 |
| `GapInterpolator` | `FeatureEngineer.interpolateShortGap` | 동작 수정 (아래 3-2) |
| `TimeFeatureEngineer` | `FeatureEngineer` + 각 엔지니어 | 순환 인코딩·변화량 |
| `LDAPSPreprocessor` | `LDAPSFeatureEngineer` | |
| `LDAPSSpatialMapper` | `LDAPSFeatureEngineer.transformToGroupIDW` | 흡수 |
| `LDAPSGridStatsBuilder` | `LDAPSFeatureEngineer.transformToGridStats` | 흡수 |
| `ScadaPreprocessor` | `SCADAFeatureEngineer.transformToLong` | 흡수 |
| `ScadaQualityChecker` | `SCADAFeatureEngineer.checkPhysicalLimit` 외 | 흡수 |
| `ScadaCleaner` | `SCADAFeatureEngineer.interpolateMissing` | 흡수 |
| `ScadaAggregator` | `SCADAFeatureEngineer.transformTo*Hourly` | 흡수 |
| (없음) | `GFSFeatureEngineer` | **신규** |

**안 옮긴 것** — 로더와 검증기는 이번 범위에서 제외했다.

| 노트북 03 | 이유 |
|---|---|
| `WindforceDataLoader`, `TurbineMetaLoader` | 데이터 로더 |
| `DataContractChecker`, `CheckCollector` | 검증기 |
| `PreprocessingValidator` | 검증기 (시각 정합 ±1h, 최종 검증) |
| `MissingPatternAnalyzer` | 진단·시각화 |
| `PreprocessedDataWriter` | I/O |

터빈 메타(`INFO/info.xlsx` 도·분·초 좌표 파싱)는 아직 노트북에 있으므로
패키지를 쓸 때는 노트북에서 만든 `turbine_meta` DataFrame을 넘겨야 한다.

## 3. 동작이 바뀐 부분 (중요)

기존 `FeatureEngineer.py` 초안이 노트북과 다르게 동작하던 곳을 노트북 기준으로 맞췄다.
**이미 만든 `prep/` CSV를 이 코드로 다시 만들면 값이 달라진다.**

### 3-1. `power_clean` — 음수를 전부 지우고 있었다

```python
# 기존: 모든 음수가 사라진다
df["power_clean"] = df["power"].where(df["power"].between(0, 1e6))

# 수정: DROP_FLAGS 3종(정격초과·큰음수·센서고착)만 NaN
df["power_clean"] = df["power_raw"].mask(df[DROP_FLAGS].any(axis = 1))
```

작은 음수는 대기전력·보조전력이라 살려야 하고, 발전량 0과 고풍속 무발전은
실제 운영 상태라 결측으로 바꾸면 고장·정비를 저발전으로 학습하게 된다.

### 3-2. 보간이 긴 구멍의 앞부분을 채우고 있었다

`interpolate(limit = 2)`는 **연속 결측 3칸짜리 구멍의 앞 2칸을 채운다.**
정비로 3일 비어 있는 구간의 앞 20분이 메워지면 그 시간은 실제 저발전으로 학습된다.
구간 **전체 길이**를 먼저 재고 2칸 이하인 구멍만 채우도록 바꿨다.

### 3-3. 시간 집계가 1단계였다

노트북은 터빈×시간 → 그룹×시간 2단계이고 두 단계 모두 `min_count`를 준다.
1단계로 합치면 6기 그룹에서 1기만 살아 있어도 `min_count=6`을 통과해
통신 장애가 발전량 0으로 둔갑한다. 그룹 합계에 `min_count=len(members)`를 넣었다.

### 3-4. `coverage_ratio` 분모

`유효샘플 / 6` → `유효샘플 / (6 × 터빈수)`. 기존 식은 6기 그룹에서 값이 6까지 나왔다.

### 3-5. 그 외

| 항목 | 기존 | 수정 |
|---|---|---|
| `clipPower` | 집계 결과에 적용 | 제거. 정격 초과는 센서 오류 증거라 플래그로 남긴다 (최종 예측 때만 사용) |
| `EPSILON` | 0.622 / 0.378 | 0.62197 / 0.37803. 반올림하면 밀도 4번째 자리가 흔들린다 |
| `e_pa` (수증기압) | 미보존 | 컬럼으로 남김 (밀도가 이상할 때 추적용) |
| `is_rh_over_100` | 없음 | 추가 (LDAPS 상대습도 100% 초과) |
| `month` / `season` | 없음 | 추가 |
| `computeAngleDifference` | 없음 | 추가 (350도와 10도의 차는 340도가 아니라 -20도) |
| 5/6 완화 집계 | 없음 | `power_sum_relaxed` 추가 (6/n 보정) |

### 3-6. 모듈화하면서 새로 잡은 결함 2건

1. **시간 피처가 IDW 집계에서 통째로 사라졌다.** 순환 인코딩을 격자 단위에서 하는데
   공간 가중평균은 지정한 피처만 통과시켜 `hour_sin`·`doy_sin`·`season`·`lead_hour`가 전부 날아갔다.
   `transformGroupTimeFeature`로 그룹 단위에서 복원한다 (피처 169 → 178개).
2. **`season` 원핫 차원이 데이터에 따라 달라졌다.** `get_dummies`가 관측된 범주만 만들어
   겨울만 있는 구간에서는 컬럼이 1개만 생긴다. 학습 때와 입력 차원이 어긋나므로
   `pd.Categorical`로 4범주를 고정했다 (→ 181개).

## 4. 클래스 설계

`transform()`이 순서를 고정하는 **템플릿 메서드**이고, 그 안의 단계가 추상/훅이다.
각 단계가 노트북 Step에 1:1 대응한다.

| 순서 | 메서드 | 종류 | 노트북 Step |
|---|---|---|---|
| 1 | `transformColumnName` | **추상** | Step 2 컬럼 통일 |
| 2 | `transformTimeAxis` | **추상** | Step 2 시간 통일 |
| 3 | `checkPhysicalLimit` | **추상** | Step 4 물리한계 플래그 |
| 4 | `interpolateMissing` | 훅 | Step 5·7 정제 + 보간 |
| 5 | `calculatePhysicalFeature` | **추상** | Step 8 파생변수 |
| 6 | `checkDerivedFlag` | 훅 | Step 8 파생값 플래그 |
| 7 | `transformCyclicFeature` | 훅 | Step 10-3 주기성 인코딩 |
| 8 | `selectGroup` | 구체 | 그룹 분리 |

- **추상 4개**는 소스마다 반드시 달라 기본값을 줄 수 없다 (`@abstractmethod`)
- **훅 3개**는 기본이 `return df`다. LDAPS·GFS는 결측이 없어 4번을 안 쓰고,
  SCADA는 파생 플래그가 없어 6번을 안 쓴다
- `source`·`COLUMN_MAP` 누락은 `__init_subclass__`가 **import 시점에** 잡는다.
  `@abstractmethod`는 메서드만 강제하고 클래스 속성은 못 잡기 때문이다

## 5. GFS 추가 (신규)

노트북 03에서 범위 밖으로 미뤘던 GFS를 붙였다. **쓰는 이유는 고도 하나다.**
LDAPS는 10m와 50m(최대·최소)뿐인데 GFS는 80m·100m 순간값을 준다.

- 7개 레벨 풍속·풍향 (10m / 80m / 100m / PBL / 850 / 700 / 500hPa)
- **전단지수** `alpha = ln(V100/V10) / ln(100/10)`
- **허브 높이 외삽** `V_hub = V100 × (117/100)^alpha` — 평가 기간에 SCADA가 없어
  사실상 유일한 허브풍속 단서다
- 돌풍계수, 850hPa 안정도(`lapse_850`), 벌크 시어

격자가 9개뿐이고 0.25° 간격이라 **그룹 간 공간 차이는 거의 없다**(4.30/4.33/4.34 m/s).
GFS의 값은 공간 해상도가 아니라 고도에 있으므로 `gfs_ws_hub`·`gfs_shear_alpha`를
LDAPS 10m 풍속의 보정 입력으로 쓰는 게 맞다.

## 6. 사용법

```python
from Windforce import FeatureEngineerFactory

# --- 예보 ---
fe = FeatureEngineerFactory.create("ldaps")          # "gfs"도 동일
grid  = fe.transform(ldaps_raw)                       # 격자별 파생 + 품질 플래그
group = fe.transformToGroupIDW(grid, turbine_meta)    # 16격자 -> KPX 그룹 (IDW)
stats = fe.transformToGridStats(grid)                 # 16격자 단순통계 (A/B 대조군)
feat  = fe.transformForecastFeature(group)            # 변화량·램프

# --- SCADA (turbine_meta 필수) ---
sc = FeatureEngineerFactory.create("scada", turbine_meta = turbine_meta)
long_df = sc.transformToLong(vestas_raw, vestas_turbines)
scada   = sc.transform(long_df)
hourly  = sc.transformToGroupHourly( sc.transformToTurbineHourly(scada) )

# --- LSTM 입력 ---
model_input  = fe.transformToModelInput(merged)
feature_cols = fe.computeFeatureColumn(model_input, target_col = "scada_power_kwh")
scale_param  = fe.computeScaleParam(model_input.loc[train_mask], feature_cols)  # 학습구간만
scaled       = fe.transformScale(model_input, scale_param)
x, y, index  = fe.transformToSequence(scaled, feature_cols,
                                      target_col = "scada_power_kwh", seq_len = 24)
# x.shape = (samples, 24, n_features), model.fit(x, y)
```

주의할 점 두 가지.

- **`SCADAFeatureEngineer`는 `turbine_meta`가 필수다.** 터빈마다 정격(VESTAS 600 /
  UNISON 700 kWh)과 컷아웃이 달라 행별 물리한계 기준을 만들 수 없다. 빠뜨리면 생성 시점에 막힌다.
- **`computeScaleParam`은 학습 구간만 넣어야 한다.** 전체 기간으로 통계를 내면 누수다.

## 7. 실데이터 확인 결과 (40일치)

```
LDAPS 그룹별 평균 ws10   5.25 / 5.66 / 5.86 m/s      <- 공간가중 작동
(평균 ws)³ 253.9 vs 평균(ws³) 268.4                  <- 순서 틀리면 5.4% 과소평가
GFS 고도별  10m 2.81 -> 80m 3.92 -> 100m 4.08 -> hub 4.19 m/s
GFS 전단지수 중앙값 0.159                             <- 개활지 0.14, 태백 산지라 더 큼
SCADA coverage_ratio 0.993, 설비이용률 0.386 / 0.416
최종  X (1800, 24, 181) float32, y (1800,), NaN 0개
```

## 8. 남은 일

- 터빈 메타 로더(`info.xlsx` 파싱)를 패키지로 옮길지 결정
- 검증기(시각 정합 ±1h, 최종 검증) 이관 여부 결정
- 10m 예보풍속 → 허브 높이 보정 모델. `gfs_ws_hub`를 입력으로 쓰고 SCADA를 정답으로
  학습하되 **Out-of-Fold 예측값**을 만들어야 간접 누수를 막는다
- 모델링에서 A/B: `transformToGroupIDW`(공간가중) vs `transformToGridStats`(단순통계),
  `scada_power_kwh`(6/6) vs `scada_power_kwh_relaxed`(5/6)
- 최종 예측은 `clipPower`로 `clip(0, 그룹 설비용량)` 필수
