# WINDFORCE 코딩 컨벤션

이 문서는 `BASELINE/baseline_1~3.ipynb` 와 이 노트북들이 사용하는 `Windforce/` 패키지에서
**실제로 쓰이고 있는 코드 스타일을 규칙으로 성문화한 것**이다.

앞으로 이 저장소에 코드를 추가·수정하는 사람(또는 LLM 에이전트)은
**별도 지시가 없어도 이 문서의 규칙을 기본값으로 적용한다.**

| 항목 | 값 |
|---|---|
| 언어 | Python 3.10+ (`dict[str, bool]`, `str \| None` 문법 사용) |
| 주석·문서 언어 | **한국어** (코드 식별자는 영문) |
| 들여쓰기 | 스페이스 4칸 |
| 인코딩 | 소스 UTF-8, CSV 입출력 `utf-8-sig` |
| 최대 줄 길이 | 권장 100자 (표·주석·수식은 초과 허용) |

---

## 0. 최우선 원칙 (Golden Rules)

1. **주석은 코드 아래에 붙인다.** 이 저장소의 가장 두드러진 특징이다. (§4)
2. **모든 수치에는 단위를 명시한다.** kW / kWh / m/s / K / Pa / 도(°). (§8)
3. **시계열 데이터는 절대 랜덤 셔플로 분할하지 않는다.** 스케일러는 train 구간에만 `fit`. (§9)
4. **DataFrame을 받는 함수는 `df.copy()` 로 시작한다.** 원본은 변형하지 않는다. (§7)
5. **주변 코드의 스타일을 우선한다.** 기존 파일을 수정할 때는 그 파일의 관례(공백, 네이밍)를 따른다.
6. **모르는 값을 지어내지 않는다.** 설비용량·좌표·기간 등은 `kpx_info` / 상수 모듈에서 가져온다.

---

## 1. import 규칙

### 1-1. import 블록 순서

표준 라이브러리 → 데이터/과학 스택 → 딥러닝 → 프로젝트 내부 모듈 순서로,
**그룹 사이에 빈 줄 한 줄**을 넣는다.

```python
import os
import sys

import csv
from typing import List, Optional
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as stats

import torch
import torch.nn as nn
```

### 1-2. 별칭(alias) 고정

| 모듈 | 별칭 | 비고 |
|---|---|---|
| pandas | `pd` | |
| numpy | `np` | |
| matplotlib.pyplot | `plt` | |
| plotly.express | `px` | **시각화 기본 라이브러리** |
| statsmodels.api | `stats` | (`sm` 아님 — 이 저장소 관례) |
| torch.nn | `nn` | |

### 1-3. 노트북에서의 패키지 import

노트북은 항상 `ROOT` 상수 → `sys.path` 삽입 → 패키지 import 순서로 쓴다.

```python
ROOT = "/Users/ksydata/WINDFORCE"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Windforce 패키지 탐색 경로를 sys.path에 추가

from Windforce import (
    WindforceDataLoader,
    FeatureEngineer,
    LDAPSFeatureEngineer,
    GFSFeatureEngineer,
    EvaluationMetrics,
    RATED_CAPACITY_KW,
    TIME_STEP_HOURS,
    ScoreLossFunction,
)
# Windforce 루트 패키지에서 주요 클래스·상수 import

from Windforce.Modeling import (
    LSTMPipeline,
    GroupExperimentRunner,
    BaselineModels,
)
# Modeling 서브패키지에서 학습·실험 관련 클래스 import
```

- 다중 import는 **괄호 + 한 줄에 하나 + 마지막 항목에도 콤마**(trailing comma).
- 패키지 내부 모듈끼리는 **상대 import**를 쓴다: `from ..EvaluationMetrics import EvaluationMetrics`.
- 함수 안에서의 지역 import는 스크립트성 검증 코드(`from scipy import stats`)에서만 허용한다.

### 1-4. 노트북 표시 옵션 (첫 셀 다음에 고정 배치)

```python
pd.set_option("display.max_columns", None)
# 데이터프레임 생략되는 열 없이 모든 열 조회
pd.options.display.float_format = "{:,.6f}".format
# 파이썬 지수 e표현 실수로 변환 (원상복구: pd.options.display.float_format = None)
```

---

## 2. 파일·패키지 레이아웃

```
Windforce/
├── __init__.py                  # 주요 클래스·상수 재수출 + __all__
├── WindforceDataLoader.py       # 파일 1개 = 클래스 1개, 파일명 == 클래스명
├── EvaluationMetrics.py
├── ScoreLossFunction.py
├── Preprocessing/               # 소스별 전처리기 (LDAPS / GFS / SCADA)
├── Modeling/                    # 모델·파이프라인·실험 러너
└── utils/                       # 상태 없는 순수 계산 함수 모음
```

규칙:

- **한 파일에 한 개의 주요 클래스**를 두고, **파일명 = 클래스명(PascalCase)** 으로 맞춘다.
  (예: `LSTMPipeline.py` → `class LSTMPipeline`)
- 그 클래스에만 쓰이는 보조 클래스(`SequenceDataset`, `_LSTMModel`)는 같은 파일에 둔다.
- 패키지 `__init__.py` 는 **재수출 전용**이며 반드시 `__all__` 을 명시한다.

```python
from .WindforceDataLoader import WindforceDataLoader
from .EvaluationMetrics import EvaluationMetrics, RATED_CAPACITY_KW, TIME_STEP_HOURS

__all__ = [
    "WindforceDataLoader",
    "EvaluationMetrics",
    "RATED_CAPACITY_KW",
    "TIME_STEP_HOURS",
]
```

- 모듈 최상단 첫 줄에는 **그 모듈이 파이프라인의 몇 단계인지**를 적는다.

```python
# Step 6.2. LSTMPipeline: LSTM 모델의 생성·학습·예측을 하나로 묶어 관리하는 클래스
```

- `utils/` 처럼 순수 계산만 모은 모듈은 파일 최상단에 한 줄 docstring을 쓴다.

```python
"""바람·대기 물리 계산 유틸리티."""
```

---

## 3. 네이밍 규칙

### 3-1. 기본 표

| 대상 | 규칙 | 예시 |
|---|---|---|
| 클래스 | `PascalCase` | `WindforceDataLoader`, `LDAPSFeatureEngineer`, `ScoreLossFunction` |
| 내부 전용 클래스 | `_PascalCase` | `_LSTMModel` |
| **클래스 메서드** | **`camelCase`** | `evaluateNMAE`, `transformToGroupIDW`, `runGroup`, `finalScores` |
| 던더·PyTorch 규약 메서드 | `snake_case` 고정 | `__post_init__`, `__getitem__`, `forward`, `fit`, `predict` |
| 모듈 수준 함수 | `snake_case` | `parseDMS`는 예외(노트북), `compute_wind_speed`(utils) |
| 내부 헬퍼 함수/메서드 | `_camelCase` / `_snake_case` | `_splitAndScale`, `_featureCols`, `_load_data` |
| 지역 변수 | `snake_case` 또는 `camelCase` 혼용 허용 | `curve_df`, `groupResults`, `epochLoss` |
| 상수 | `UPPER_SNAKE_CASE` | `RATED_CAPACITY_KW`, `SEQ_LEN`, `GROUPS`, `SEED`, `R_D` |
| 클래스 상수(ClassVar) | `UPPER_SNAKE_CASE` | `COLUMN_MAP`, `REQUIRED_COLUMNS`, `IDW_FEATURES` |
| private 속성 | `_snake_case` | `_paths`, `_cache` |
| DataFrame 컬럼명 | `snake_case` | `forecast_kst_dtm`, `kpx_group_1`, `ldaps_10m_ws_raw` |

> **혼재 스타일 안내**
> `Windforce/utils/`, `Windforce/Preprocessing/FeatureEngineer.py`(신 ABC 계층)는
> PEP8 `snake_case` 함수명을 쓴다. 그 외 대부분(노트북 + Modeling + EvaluationMetrics)은 `camelCase` 메서드다.
> **새 코드는 "수정 중인 파일이 쓰는 쪽"을 따른다.** 새 파일을 만들 때는
> *클래스의 공개 메서드 = camelCase*, *utils의 자유 함수 = snake_case* 를 기본값으로 한다.

### 3-2. 메서드 동사 접두사 사전 (매우 중요)

메서드 이름은 **동사 + 목적어** 형태이며, 아래 접두사를 의미대로 지켜서 쓴다.

| 접두사 | 의미 | 예시 |
|---|---|---|
| `load` | 파일 → DataFrame 로딩 | `load`, `load_all` |
| `check` | 존재/한계 검사 후 상태·플래그 반환 | `check_paths`, `checkPhysicalLimit`, `checkDerivedFlag` |
| `validate` | 계약 위반 시 **예외를 던짐** | `validateData` |
| `compute` | 입력 → 값 계산 (Series/스칼라 반환) | `computeWindSpeed`, `computeWindDirection`, `computeScaleParam` |
| `calculate` | 물리식 기반 파생변수 계산 (DataFrame 반환) | `calculateAirDensity`, `calculateWindPowerDensity`, `calculatePhysicalFeature` |
| `transform` | DataFrame → DataFrame 변환 (컬럼 추가/이름 변경/구조 변경) | `transformLDAPS`, `transformWindDirection`, `transformToGroupIDW`, `transformToSequence` |
| `interpolate` | 결측 보간 | `interpolateMissing`, `interpolateTimeVariable` |
| `select` | 부분집합 필터링 | `selectGroup` |
| `evaluate` | 평가지표 계산 | `evaluateNMAE`, `evaluateFICR`, `evaluateTotalScore` |
| `build` | 학습용 데이터셋 조립 | `build`, `featureCols` |
| `make` | 새 객체·모델 인스턴스 생성 후 반환 | `makeSVRModel` |
| `run` | 실험 실행(부수효과 있음) | `runGroup`, `runAll` |
| `summarize` | 요약 정보 출력 또는 dict 반환 | `summarize` |
| `fit` / `predict` | scikit-learn 호환 학습·예측 (이름 고정) | `fit`, `predict` |

### 3-3. 변수·컬럼 이름 접미사 규칙

| 접미사 | 의미 | 예시 |
|---|---|---|
| `_df` | pandas DataFrame | `ldaps_transformed_df`, `curve_df`, `groupDf` |
| `_cols` | 컬럼명 리스트 | `wd_cols`, `feat_cols`, `target_cols` |
| `_col` | 단일 컬럼명 문자열 | `ws_col`, `target_col`, `date_col` |
| `_kw` / `_kwh` | 전력(kW) / 전력량(kWh) | `capacity_kw`, `pred_kw`, `generation_kwh` |
| `_m` | 미터 | `height_low`, `HUB_HEIGHT_M` |
| `_raw` | 원시 계산값(후처리 전) | `ldaps_10m_ws_raw`, `gfs_100m_wd_raw` |
| `_sin` / `_cos` | 주기 인코딩 결과 | `{col}_sin`, `{col}_cos` |
| `_ws` / `_wd` | 풍속 / 풍향 | `unison_wtg01_ws`, `unison_wtg01_wd` |
| `_scaled`, `_sc`, `_norm` | 스케일링·정규화 결과 | `X_scaled`, `X_train_sc`, `y_train_norm` |
| `X_` / `y_` 접두 | 모델 입출력 배열 (대문자 X 유지) | `X_train`, `X_test`, `y_train`, `y_test` |

파생 컬럼 이름은 **`{소스}_{고도}_{물리량}_{가공}` 순서**로 만든다.

```
ldaps_10m_ws_raw        # LDAPS · 지상 10m · 풍속 · 원시
gfs_100m_wd_raw_sin     # GFS   · 지상 100m · 풍향 · 원시 · sin 인코딩
gfs_ws_hub              # GFS   · 허브높이 외삽 풍속
```

그룹 키는 문자열 `"kpx_group_1"` 형식이고, 그룹 번호는 정수 `1/2/3` 이다.
둘을 섞지 말고, 필요할 때 `group_key = f"kpx_group_{group}"` 로 변환한다.

---

## 4. 주석 작성 방식 (이 저장소의 핵심 스타일)

### 4-1. 규칙: 설명 주석은 **해당 코드 줄 바로 아래**에 쓴다

```python
capacity = self.rated_capacity_kw[group]
# KPX 그룹별 설비용량(kW) 가져오기
return float(np.mean(np.abs(pred - actual) / capacity))
# NMAE 계산: 시간별 오차율을 평균하여 단일 그룹의 NMAE 반환
```

`return` 문 아래에도 주석을 붙인다(무엇을 왜 반환하는지 설명).

블록의 **의도**를 미리 설명해야 할 때만 위쪽 주석을 허용한다.

```python
# ---------------------------------------------------------------
# (1) Persistence 베이스라인
# ---------------------------------------------------------------
persPred = BaselineModels.persistenceForecast(y_test)[1:]
# 주의: 이 값은 검증 구간의 직전 실제값을 사용한다. 2025년 전체를 한 번에
# 제출하는 조건에서는 사용할 수 없는 oracle 성격의 참고 기준선이다.
```

### 4-2. 주석 밀도

- **한 줄 = 한 설명**을 기본으로 한다. 이 저장소는 의도적으로 **주석 밀도가 높다**(코드 대비 0.7~1.0줄).
- 자명한 코드라도 "왜 이 값인지", "무엇이 위험한지"를 적는다.
- 인자 목록·딕셔너리 항목 각각에도 주석을 붙일 수 있다.

```python
self._paths = {
# 각 데이터 파일의 논리적 이름과 실제 경로를 매핑
    "ldaps_train": f"{self.root_train}/ldaps_train.csv", # [v]
    # LDAPS 기반 기상 예보 데이터 (약 1.5 km 공간해상도의 16개 격자)
    # 2022-01-01 01:00:00 ~ 2025-01-01 00:00:00
}
```

### 4-3. 주석에 반드시 포함할 정보

1. **단위** — `설비용량(kW)`, `발전량(kWh)`, `기온(K)`, `기압(Pa)`, `풍향(0~360도)`
2. **shape** — `shape = (N, n_features)`, `(batch, seq_len, hidden_size)`
3. **함정·주의** — 누수 위험, 인덱스 정렬, 부호 규약
4. **왜 이 값인가** — 하이퍼파라미터·상수의 근거

```python
weight = actual_kw / (actual_kw.sum() + 1e-8)
# 실제 발전량에 대한 가중치 계산 (실제 발전량이 큰 시간대에 더 큰 영향력 부여)
# 0으로 나누는 경우를 방지하기 위해 작은 값 1e-8을 더함
```

### 4-4. 섹션 구분자

- 파이썬 파일 내부: `# ----- ... -----` 또는 `# ─── 제목 ───`
- 노트북 마크다운 셀: `---` 로 단계 구분
- 주석으로 소제목을 달 때: `# ── 학습 데이터 준비 ──────────────────`

### 4-5. 비활성 코드·미완성 코드

```python
# df = df.drop(columns = [col])
# 필요시 주석 해제
```

- 미구현 메서드는 docstring + `pass` 로 남기고, 무엇을 구현해야 하는지 `Logic:` 에 적는다.
- 폐기 예정 코드는 `'''...'''` 로 감싸 두거나 주석 처리하고 **왜 남겨두는지** 적는다.

---

## 5. Docstring 규칙

### 5-1. 형식

- **큰따옴표 3개**, 첫 줄은 한 문장 한국어 요약.
- 요약 끝맺음은 **`~하는 클래스` / `~하는 메서드` / `~하는 함수` / `~한다.`** 로 통일한다.
- 섹션 키워드: `Args:` `Returns:` `Raises:` `Attributes:` `Logic:`(또는 `Logics:`) `사용 순서:`
- 항목은 **`- ` 불릿**으로 시작하고, `이름: 설명` 형태로 쓴다.

```python
def evaluateNMAE(self, pred: np.ndarray, actual: np.ndarray, group: int) -> float:
    """단일 그룹의 NMAE(Normalized Mean Absolute Error)를 계산하는 메서드

    Args:
        - pred: 예측 발전량 배열 (kWh 단위), shape = (시간, )
        - actual: 실제 발전량 배열 (kWh 단위), shape = (시간, )
        - group: KPX 그룹 번호 (1/2/3 중 하나)

    Returns:
        - 단일 그룹의 NMAE 값 (0에 가까울수록 좋음)

    Logic:
        - NMAE = mean(|시간당 예측 발전량 - 시간당 실제 발전량| / 설비용량(kW))
        - 용량 대비 오차율로 정규화한 뒤 전체 시간에 대해 평균을 낸다
    """
```

### 5-2. 클래스 docstring

한 줄 요약 + (필요 시) `Attributes:` 또는 `Logic:` 또는 `사용 순서:`.

```python
class LSTMPipeline:
    """LSTM 모델의 생성·학습·예측을 하나로 묶어 관리하는 클래스

    사용 순서:
        pipeline = LSTMPipeline(capacity_kw = 21600.0)
        pipeline.fit(X_train, y_train)           # 학습 (self.model이 채워짐)
        pred = pipeline.predict(X_test, y_test)  # 예측 (학습된 self.model 사용)
    """
```

- 약어는 **처음 등장할 때 풀어 쓴다**: `NMAE(Normalized Mean Absolute Error)`,
  `FICR(Financial Incentive Capture Rate)`, `IDW(역거리가중)`.
- 수식은 docstring 또는 마크다운에 그대로 적는다: `loss = 0.5*NMAE + 0.5*(1-FICR)`.

### 5-3. 클래스 설계 의사결정은 파일 상단 주석에 남긴다

```python
# 기존 trainLSTM/predictLSTM 함수는 model 객체를 인자로 주고받는 구조였는데,
# 이렇게 하면 "이 model이 어느 그룹용으로 학습된 것인지" 호출부에서 매번
# 변수명으로 관리해야 해서 실수하기 쉽다.
# 이 클래스는 self.model에 학습된 가중치 상태를 직접 들고 있게 해서,
# "학습된 모델"이라는 개념 자체를 하나의 객체로 캡슐화한다.
```

---

## 6. 타입 힌트 규칙

- **모든 공개 함수·메서드의 인자와 반환값에 타입을 붙인다.**
- 소문자 제네릭을 쓴다: `dict[str, bool]`, `list[dict]`, `tuple[float, float]`.
- 선택 인자는 `Optional[np.ndarray]` 또는 `np.ndarray | None` (파일 스타일에 맞춤).
- 반환 스칼라는 **`float()` 로 명시 캐스팅**한다 (numpy 스칼라 누출 방지).

```python
return float(earningSettlement / maxSettlement)
```

자주 쓰는 타입:

| 데이터 | 타입 | 비고 |
|---|---|---|
| 표 데이터 | `pd.DataFrame` | 전처리 계층의 입출력 기본형 |
| 컬럼 1개 | `pd.Series` | 물리 계산 함수의 입출력 |
| 모델 입출력 | `np.ndarray` | `float32` 로 맞춘다 |
| 텐서 | `torch.Tensor` | `dtype=torch.float32` |
| 그룹 번호 | `int` (1/2/3) | 그룹 키는 `str` |
| 설비용량 | `float` (kW) | `21_600.0` 처럼 언더스코어 구분 |
| 결과 누적 | `list[dict]` → `pd.DataFrame` | `{"group":.., "model_name":.., "nmae":..}` |

숫자 리터럴은 자릿수 구분에 `_` 를 쓴다: `21_600.0`, `26_304`.

메서드 체이닝을 지원하려면 `fit()` 은 `-> "LSTMPipeline"` 로 자기 자신을 반환한다.

---

## 7. pandas / numpy 사용 규칙

### 7-1. 필수 관용구

```python
def transformXXX(self, df: pd.DataFrame) -> pd.DataFrame:
    """..."""
    df = df.copy()
    # 원본 DataFrame 변형 방지
    ...
    return df
```

| 상황 | 관용구 |
|---|---|
| 컬럼 존재 확인 | `if u_col in df.columns and v_col in df.columns:` / `if {ws, wd, pw}.issubset(df.columns):` |
| 컬럼 안전 제거 | `df.drop(columns="group", errors="ignore")` |
| 결측 제거 후 인덱스 정리 | `df.dropna(subset=[col]).reset_index(drop=True)` |
| 숫자형 컬럼만 추출 | `df.select_dtypes(include="number").columns.tolist()` |
| 부분 문자열 컬럼 선택 | `df.filter(like=columnName)` |
| 시각 컬럼 변환 | `pd.to_datetime(df["forecast_kst_dtm"])` |
| 조건 분기 배열 | `np.where(cond1, a, np.where(cond2, b, c))` |
| 0 나눗셈 방지 | `+ 1e-8`, `.clip(lower=1e-3)` |
| 무한대 제거 | `df[numeric].replace([np.inf, -np.inf], np.nan)` |
| 시각화용 샘플링 | `df.sample(3000, random_state=SEED)` |

### 7-2. 병합(merge) 규칙

- 기상 데이터끼리는 **`forecast_kst_dtm` 기준 `how="inner"`**, `suffixes=("_ldaps", "_gfs")`.
- 라벨 결합은 `left_on="forecast_kst_dtm", right_on="kst_dtm", how="left"` 후 `kst_dtm` 제거.
- merge 전에 양쪽 키를 반드시 `pd.to_datetime()` 으로 맞춘다.

### 7-3. 피처 컬럼 선정 (누수 방지)

식별자와 **모든 그룹의 타깃**을 제외한다.

```python
exclude = {"kst_dtm", "forecast_kst_dtm"} | {f"kpx_group_{i}" for i in range(1, 4)}
return [c for c in df.columns if c not in exclude]
# kst_dtm / forecast_kst_dtm : 시각 식별자 → 피처에 포함하면 미래 정보 누수 위험
# kpx_group_*: 다른 그룹 타깃도 전부 제외 (그룹 간 누수 방지)
```

### 7-4. 반복 처리 패턴

컬럼 매핑 딕셔너리를 만들고 순회하며 파생변수를 생성한다. 하드코딩 나열보다 이 패턴을 우선한다.

```python
gfs_mapping = {
    # Key: 새롭게 생성될 변수들의 접두사(Prefix)
    # Value: (동서방향 U 성분 컬럼명, 남북방향 V 성분 컬럼명)
    "gfs_10m":  ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "gfs_100m": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
}

ws_cols, wd_cols = [], []
for prefix, (u_col, v_col) in gfs_mapping.items():
    if u_col in df.columns and v_col in df.columns:
        ws_name = f"{prefix}_ws_raw"
        wd_name = f"{prefix}_wd_raw"
        df[ws_name] = self.computeWindSpeed(df[u_col], df[v_col])
        df[wd_name] = self.computeWindDirection(df[u_col], df[v_col])
        ws_cols.append(ws_name)
        wd_cols.append(wd_name)
        # 일괄 파생 변수 생성을 위한 리스트 업로드
```

---

## 8. 도메인·단위 규칙

| 물리량 | 단위 | 규칙 |
|---|---|---|
| 발전량 | kW / kWh | 컬럼·변수명에 `_kw`, `_kwh` 명시. MW 입력은 `* 1000` 으로 즉시 변환 |
| 설비용량 | kW | `RATED_CAPACITY_KW = {1: 21_600.0, 2: 21_600.0, 3: 21_000.0}` |
| 풍속 | m/s | `sqrt(u² + v²)` |
| 풍향 | 0~360도 (기상학적, 북 0 / 동 90 / 남 180 / 서 270) | `(np.degrees(np.arctan2(-u, -v)) + 360) % 360` |
| 기온 | **K** | 섭씨로 넣으면 공기밀도가 통째로 틀림 |
| 기압 | **Pa** | hPa 혼용 금지 |
| 시간 해상도 | `TIME_STEP_HOURS = 1.0` | SCADA 10분값은 `1/6 = 0.1667` |
| 시퀀스 길이 | `SEQ_LEN = 24` | 최근 24시간으로 다음 시점 예측 |

핵심 물리식(주석에 근거로 인용한다):

```
P     = 0.5 · ρ · A · V³ · Cp          풍력 출력
ρ     = (P_sfc - 0.378·e) / (R_d · T)  습윤공기 밀도
e     = q·P / (0.62197 + 0.37803·q)    수증기압
WPD   = 0.5 · ρ · V³                   풍력에너지 밀도 (W/m²)
alpha = ln(V2/V1) / ln(h2/h1)          멱법칙 전단지수
```

**풍향은 반드시 sin/cos 로 인코딩한다.** 0도와 360도가 같은 방향임을 모델이 인지하지 못하기 때문이다.

```python
radian = np.radians(df[col])
df[f"{col}_cos"] = np.cos(radian)
df[f"{col}_sin"] = np.sin(radian)
# 머신러닝의 0도-360도 연속성 인지를 위한 주기성 인코딩
```

각도 평균이 필요하면 U·V 성분으로 바꾼 뒤 평균한다. 각도 차이는 `(a - b + 180) % 360 - 180`.

---

## 9. 모델링 규칙

### 9-1. 재현성

노트북·스크립트 상단에 상수와 시드를 함께 선언한다.

```python
GROUPS  = [1, 2, 3]
# KPX 평가 그룹 번호 목록
SEQ_LEN = 24
# LSTM 입력 시퀀스 길이 (최근 24시간)
SEED    = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# 재현성 확보를 위해 난수 시드 고정
```

- 난수 생성은 `rng = np.random.default_rng(SEED)` 를 우선 사용한다.
- 정렬 대입(`GROUPS  = ...`)처럼 **등호를 세로로 맞추는 스타일**을 허용한다.

### 9-2. 절대 규칙 (위반 시 결과 무효)

1. **시간순 분할만 허용.** `train_test_split(shuffle=True)` 금지.
   ```python
   split = int(n * (1 - self.testRatio))
   # 앞쪽 (1-testRatio)를 train, 뒤쪽 testRatio를 test로 시간순 분할
   ```
2. **스케일러는 train 구간에만 `fit`.** transform은 전체에 적용.
   ```python
   xScaler = MinMaxScaler().fit(X_all[:split])
   X_scaled = xScaler.transform(X_all)
   ```
3. **시퀀스 모델은 앞 `seq_len` 개 타깃을 버린다.** 정답 배열도 동일하게 자른다.
   ```python
   yTestAligned = y_test[self.seqLen:]
   # SequenceDataset이 앞쪽 seqLen개 시점을 버렸으므로 정답도 동일하게 정렬
   ```
4. **`capacity_kw` 는 그룹마다 다르다.** 파이프라인 인스턴스를 그룹 간 재사용하지 않는다.
5. **DataLoader**: 학습 `shuffle=True`, 예측 `shuffle=False`(순서 유지).

### 9-3. 파이프라인 클래스 계약

- 상태(학습 가중치)를 갖는 모델 → **클래스** (`LSTMPipeline`, `self.model`).
- 상태가 없는 모델·유틸 → **`@staticmethod` 전용 네임스페이스 클래스** (`BaselineModels`, `WindUtils`).
- `fit()` 은 `self` 를 반환하고, `predict()` 는 학습 여부를 먼저 검증한다.

```python
if self.model is None:
    raise RuntimeError("fit()을 먼저 호출해 모델을 학습시켜야 합니다.")
```

- 추론 메서드에는 `@torch.no_grad()` 데코레이터를 붙인다.
- `self.model.train()` / `self.model.eval()` 을 명시적으로 호출한다(현재 구조에 dropout이 없어도 관례상 표기).
- 학습 이력은 `self.history: list[dict]` 에 `{"epoch":.., "train_loss":..}` 형태로 누적한다.

### 9-4. 타깃 정규화

발전량은 **설비용량 대비 이용률(0~1)** 로 학습하고, 손실 계산 시 kW로 되돌린다.

```python
y_train_norm = np.clip(np.asarray(y_train, dtype=np.float32) / self.capacity_kw, 0.0, 1.0)
...
scoreLoss = scoreCriterion(predNorm * self.capacity_kw, yb * self.capacity_kw)
```

출력층은 `sigmoid`(이용률) 또는 `ReLU`(kW 직접 예측)로 **음수 발전량을 원천 차단**한다.

---

## 10. 평가지표·손실함수 규칙

대회 정의는 아래 한 가지뿐이며, 임의로 바꾸지 않는다.

```
NMAE_g = mean(|pred - actual| / Cap_g)
UnitPrice_t = 4원 (오차율 ≤ 6%) / 3원 (6~8%) / 0원 (> 8%)
FICR_g = Σ(UnitPrice_t · actual_t) / Σ(4 · actual_t)
총점    = 0.5 × (1 - mean(NMAE)) + 0.5 × mean(FICR)
loss   = 1 - 총점 = 0.5 × NMAE + 0.5 × (1 - FICR)
```

- **평가**(`EvaluationMetrics`)는 numpy로 **계단함수 그대로** 계산한다.
- **학습**(`ScoreLossFunction`)은 계단함수를 시그모이드 2개로 완화해 미분 가능하게 근사한다.
  ```python
  softPrice = 4.0 - 1.0 * torch.sigmoid(self.k * (errorRatio - 0.06)) \
                  - 3.0 * torch.sigmoid(self.k * (errorRatio - 0.08))
  ```
- 손실은 **작을수록 좋은 방향**으로 일관성을 유지한다(`1 - softFICR`).
- 지표 결과는 항상 `{"group":.., "nmae":.., "ficr":.., "score":..}` dict로 반환·누적한다.
- 새 모델을 추가하면 반드시 `Persistence` 기준선과 함께 비교한다.
  단, `Persistence_oracle_lag1` 은 직전 실제값을 쓰는 **참고용 oracle** 이므로 제출 근거로 삼지 않는다.

---

## 11. 예외 처리·검증 규칙

- **예외 메시지는 한국어**로 쓰고, 가능한 값 목록이나 실제 값을 함께 보여 준다.

| 상황 | 예외 |
|---|---|
| 등록되지 않은 키 | `KeyError(f"'{name}'은 등록된 파일이 아닙니다. 사용 가능: {list(self._paths.keys())}")` |
| 필수 컬럼 누락 | `KeyError(f"필수 컬럼이 없습니다: {missing}")` |
| 잘못된 인자 값 | `ValueError(f"group은 {KPX_GROUPS} 중 하나여야 합니다: {group}")` |
| 호출 순서 위반 | `RuntimeError("runAll()을 먼저 호출해 결과를 생성해야 합니다.")` |
| 파일 없음 | `FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")` |
| 도메인 전용 오류 | `PreprocessingError(...)` (`utils.spatial_utils` 정의) |

- 예외 재발생 시 원인을 연결한다: `raise PreprocessingError(...) from exc`.
- **일괄 로딩처럼 "하나가 실패해도 전체를 멈추면 안 되는" 경우에만 `try/except` 로 감싸고 실패를 출력한다.**
  ```python
  try:
      self.load(name, force=force)
      print(f"✅ {name}: shape={self._cache[name].shape}")
  except Exception as e:
      print(f"❌ {name}: {e}")
  ```
- `assert` 는 **노트북의 제출 스키마 검증**에서만 사용한다(라이브러리 코드에서는 예외를 던진다).

---

## 12. 출력·로깅 규칙

`print` + f-string 을 사용한다(로깅 프레임워크 미사용).

| 용도 | 형식 |
|---|---|
| 존재 여부 | `print(f"{'✅' if exists else '❌'}  {name}: {path}")` |
| 단계 시작 | `print("LDAPS 격자 전처리 중 (16격자 × 26,304 시각)...")` |
| 구간 헤더 | `print(f"===== KPX_{group} =====")` / `print("=== SCADA VESTAS ===")` |
| shape 확인 | `print(f"ldaps_group_df shape: {ldaps_group_df.shape}")` |
| 지표 | `print(f"[그룹{g}] NMAE={nmae:.4f}  FICR={ficr:.4f}  총점={score:.4f}")` |
| 학습 로그 | `print(f"  epoch {epoch + 1:02d}/{self.epochs} [{phase}] loss={epochLoss:.5f}")` |
| 결과 정렬 출력 | `print(f"  {model_name:<12}: {score:.4f}")` |

- 소수 자릿수: 지표 `:.4f`, 손실 `:.5f`~`:.6f`, epoch `:02d`.
- 하위 단계 로그는 공백 2칸 들여쓴다.
- 이모지는 `✅ ❌` 두 개만 사용한다.

---

## 13. 시각화 규칙

**plotly express(`px`)가 기본**이며, matplotlib은 상관행렬 등 정적 도표에만 쓴다.

```python
fig = px.scatter(
    curve_df,
    x = ws_col,
    y = pw_col,
    trendline = "lowess",
    trendline_color_override = "red",
    title = f"[{col}] Power Curve (Wind Speed vs Power)",
    labels = {ws_col: "Wind Speed (m/s)", pw_col: "Power (kW)"},
    opacity = 0.4,
    template = "plotly_white",
)
fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 40))
fig.show()
break  # 렌더링 시간 절약을 위해 첫 번째 터빈만 표시
```

규칙:

- `labels` 에 **반드시 단위를 포함**한다: `"Wind Speed (m/s)"`, `"Power (kW)"`, `"Wind Direction (°)"`.
- `title` 은 `f"[{대상}] 설명"` 형식.
- 겹치는 산점도는 `opacity = 0.4~0.7`.
- 컬러스케일: `Viridis`(3D) / `Cividis`(풍속) / `Plasma`(풍향).
- `fig.update_layout(margin=dict(...))` 로 여백을 정리한 뒤 `fig.show()`.
- 대량 데이터는 `sample(3000, random_state=SEED)` 로 줄인다.
- 반복문에서 대표 1개만 그릴 때는 `break` + 주석으로 이유를 남긴다.
- 컬럼이 없으면 그리지 말고 건너뛴 사실을 출력한다.
  ```python
  else:
      print(f"Skipping {col}: Required columns ({ws_col}, {wd_col}, {pw_col}) not found.")
  ```

---

## 14. 노트북(.ipynb) 작성 규칙

### 14-1. 고정 구조

첫 마크다운 셀에 제목과 **Step 표**를 둔다.

```markdown
## Baseline N — 제목

| Step | 내용 |
|------|------|
| 1 | 파일 경로 확인 |
| 2 | 데이터 로드 및 구조 확인 |
| 3 | 터빈 메타 준비 |
| 4 | 데이터 전처리 및 시각화 |
| 5 | 평가지표 계산 및 손실함수 확인 |
| 6 | 그룹별(KPX 1/2/3) 개별 모형 학습 |
| 7 | 스키마 검증 + 제출 CSV 저장 |
```

이후 각 단계는 `## Step N. 제목` 마크다운 셀로 시작하고, 단계 사이는 `---` 로 구분한다.

### 14-2. 셀 작성 규칙

- **한 셀 = 한 가지 일.** import / 옵션 설정 / 상수·시드 / 클래스 정의 / 실행 을 각각 분리한다.
- 실행 셀 첫 줄에 그 셀의 목적을 주석으로 적는다.
  ```python
  # WindforceDatasetBuilder 인스턴스 생성 및 그룹1 데이터 확인
  ```
- 임시 검증 셀은 `# [테스트 코드] Step 5.2. 손실함수 계산` 처럼 표기한다.
- 데이터 확인은 `head(1)`~`head(5)`, `info()`, `describe()`, `shape` 를 사용한다.
- 파이프라인 흐름은 마크다운 코드블록 다이어그램으로 남긴다.
  ```
  ldaps_group_df ─┐
  gfs_group_df  ─┤  WindforceDatasetBuilder.build(group)
  train_labels  ─┘         ↓
                     피처+타깃 DataFrame
  ```
- 도메인 근거(논문·링크·수식)는 마크다운 셀에 출처와 함께 기록한다.

### 14-3. 노트북 → 패키지 승격

노트북에서 클래스가 안정화되면 `Windforce/` 아래 모듈로 옮기고,
노트북은 import만 남긴다(baseline_1 → baseline_2/3 로의 전환이 그 예).
승격 시 **주석과 docstring을 그대로 가져간다.**

---

## 15. 제출 파일 규칙

```python
# ── 스키마 검증 ────────────────────────────────────────
sample = loader["sample_submission"]
assert list(submission.columns) == list(sample.columns), "컬럼 순서 불일치!"
assert len(submission) == len(sample), f"행 수 불일치! 기대: {len(sample)}, 실제: {len(submission)}"
assert submission.isna().sum().sum() == 0, "제출 파일에 NaN 존재!"
for g in GROUPS:
    col = f"kpx_group_{g}"
    assert (submission[col] >= 0).all(), f"{col}에 음수 예측값 존재!"
print("스키마 검증 통과 ✅")

# ── 저장 ───────────────────────────────────────────────
out_path = f"{ROOT}/submission_baselineN.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"제출 파일 저장 완료: {out_path}")
```

체크리스트:

- [ ] `sample_submission.csv` 를 `.copy()` 해서 시작한다 (컬럼 순서 보존).
- [ ] 행 수 8,760 (2025년 시간별).
- [ ] NaN 없음 — 남으면 `ffill().fillna(0)`.
- [ ] 음수 예측값 없음.
- [ ] `index=False`, `encoding="utf-8-sig"`.
- [ ] 파일명은 `submission_baseline{N}.csv` 형식.
- [ ] 예측값은 시각 인덱스로 정렬해 채운다: `pred_series.reindex(submission_idx).values`.

---

## 16. 금지 사항 (Anti-patterns)

| 금지 | 이유 / 대안 |
|---|---|
| 시계열 랜덤 셔플 분할 | 미래 정보 누수 → 시간순 분할 |
| 전체 데이터로 scaler `fit` | 테스트 통계 누수 → train 구간만 `fit` |
| 타깃(`kpx_group_*`)·시각 컬럼을 피처에 포함 | 누수 → `_featureCols()` 로 제외 |
| 인자로 받은 DataFrame 직접 수정 | 부작용 → `df = df.copy()` |
| 풍향을 0~360 원본값 그대로 입력 | 불연속 → sin/cos 인코딩 |
| 각도를 산술평균 | 0/359도 왜곡 → U·V 성분 평균 |
| 피처 개수·설비용량 하드코딩 | `X_train.shape[1]`, `RATED_CAPACITY_KW[group]` 사용 |
| 영어 주석 / 주석 없는 코드 | 이 저장소는 한국어 고밀도 주석이 기본 |
| 주석을 코드 위에만 몰아 쓰기 | 설명 주석은 코드 **아래** |
| 단위 없는 변수·라벨 | `_kw`, `(m/s)` 등 명시 |
| `logging` 모듈 도입 | 현재 관례는 `print` + f-string |
| 라이브러리 코드에서 `assert` | 예외를 던진다 (assert는 노트북 검증 전용) |
| 광범위한 `except Exception: pass` | 최소한 `print(f"❌ {name}: {e}")` 로 알린다 |
| 파일 경로 문자열 산재 | `WindforceDataLoader._paths` 에 등록 후 `loader["name"]` 접근 |

---

## 17. 새 코드를 작성할 때의 템플릿

### 17-1. 전처리 클래스

```python
# Step 4.x. XXXFeatureEngineer: XXX 데이터 전처리를 담당하는 클래스

import numpy as np
import pandas as pd

from .FeatureEngineer import FeatureEngineer


class XXXFeatureEngineer(FeatureEngineer):
    """XXX 데이터 전처리를 담당하는 클래스

    Attributes:
        - 지상 10m : 순간값
        - 지상 100m : 순간값
    """

    def __init__(self):
        super().__init__()
        # 부모 클래스 FeatureEngineer의 초기화 메서드 호출

    def transformXXX(self, df: pd.DataFrame) -> pd.DataFrame:
        """XXX 테이블 명세서 기반 전처리 메서드

        Args:
            - df: 원본 XXX DataFrame

        Returns:
            - 파생변수(풍속·풍향·sin/cos)가 추가된 DataFrame
        """
        df = df.copy()
        # 원본 DataFrame 변형 방지

        df["xxx_10m_ws_raw"] = self.computeWindSpeed(df["u_col"], df["v_col"])
        # u, v 성분으로부터 절대 풍속(m/s) 계산
        df["xxx_10m_wd_raw"] = self.computeWindDirection(df["u_col"], df["v_col"])
        # 북쪽 0도 기준 시계방향 풍향(0~360도) 계산

        df = self.transformWindDirection(df, wind_direction_cols=["xxx_10m_wd_raw"])
        # 풍향 주기성 인코딩 일괄 적용 (sin/cos 추출)

        return df
```

### 17-2. 모델 파이프라인 클래스

```python
class XXXPipeline:
    """XXX 모델의 생성·학습·예측을 하나로 묶어 관리하는 클래스

    사용 순서:
        pipeline = XXXPipeline(capacity_kw = 21600.0)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
    """

    def __init__(self, capacity_kw: float, seq_len: int = SEQ_LEN):
        """
        Args:
            - capacity_kw: 이 파이프라인이 다루는 그룹의 설비용량(kW)
            - seq_len: 입력 시퀀스 길이 (과거 몇 시간을 볼지)
        """
        self.capacity_kw = capacity_kw
        # ScoreLossFunction(capacity_kw=...) 호출 시 그대로 넘길 값
        self.seq_len = seq_len
        self.model = None
        # 학습 전에는 None. fit()이 끝나야 학습된 모델이 여기 저장됨

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XXXPipeline":
        """학습 데이터로 모델을 학습시키는 메서드

        Returns:
            - self : 학습이 끝난 자기 자신 (메서드 체이닝 지원)
        """
        ...
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """학습된 모델로 예측값을 반환하는 메서드

        Raises:
            - RuntimeError: fit()을 먼저 호출하지 않은 경우
        """
        if self.model is None:
            raise RuntimeError("fit()을 먼저 호출해 모델을 학습시켜야 합니다.")
        ...
```

### 17-3. 순수 계산 유틸 함수 (`utils/`)

```python
@staticmethod
def compute_xxx(a: pd.Series, b: pd.Series) -> pd.Series:
    """XXX를 계산한다.

    Args:
        a: 설명 (단위).
        b: 설명 (단위).

    Returns:
        결과 시리즈 (단위). 계산식과 주의사항을 여기에 적는다.
    """
    return ...
```

---

## 18. 코드 작성 후 셀프 체크리스트

- [ ] 모든 공개 함수·메서드에 타입 힌트와 docstring(`Args`/`Returns`/`Logic`)이 있는가?
- [ ] 설명 주석이 **코드 아래**에 한국어로, 충분한 밀도로 달렸는가?
- [ ] 단위(kW/kWh/m/s/K/Pa/도)와 배열 shape이 주석에 명시되었는가?
- [ ] `df.copy()` 로 원본을 보호했는가?
- [ ] 컬럼 존재 여부를 확인하고, 없으면 건너뛰거나 명확한 예외를 던지는가?
- [ ] 시간순 분할 / train-only scaler fit / seq_len 정렬을 지켰는가?
- [ ] 타깃·시각 컬럼이 피처에서 제외되었는가?
- [ ] 그룹별 `capacity_kw` 를 올바르게 주입했는가?
- [ ] 시드(`SEED = 42`)가 고정되었는가?
- [ ] 네이밍이 §3 의 동사 접두사·접미사 사전을 따르는가?
- [ ] 결과 출력 포맷(`✅/❌`, `:.4f`)이 §12 와 일치하는가?
- [ ] 제출 파일이면 §15 스키마 검증을 통과하는가?
