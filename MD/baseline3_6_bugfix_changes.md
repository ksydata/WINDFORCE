# baseline 3~6 IDW 및 제출 추론 버그 수정 내역

> 작성일: 2026-08-12  
> 대상: `BASELINE/baseline_3.ipynb` ~ `BASELINE/baseline_6.ipynb`, `Windforce/` 공통 코드

---

## 1. 수정 목적

다음 두 문제를 소급 수정했다.

1. `turbine_meta["group"]`에 정수 `1/2/3`이 들어오면 문자열 그룹 키와 매칭되지 않아
   IDW(역거리가중) 그룹 가중치가 0 행렬이 되는 문제
2. baseline 3·4 제출 추론에서 LSTM의 과거 24시간 입력을 제공하지 않아
   `2025-01-01 01:00:00`부터 24개 시각의 예측이 누락되고, 이 결측이 `0`으로 저장되는 문제

기존 모델 구조, 피처 생성 방식, 시간 분할 및 하이퍼파라미터는 변경하지 않았다.

---

## 2. IDW 그룹 키 수정

### `Windforce/UTILS/spatial_utils.py`

- 입력 `turbine_meta`를 복사해 호출자의 원본 DataFrame을 변경하지 않도록 했다.
- 그룹 번호 `1/2/3`, `1.0/2.0/3.0`, 숫자 문자열을
  `kpx_group_1/2/3` 형식으로 정규화한다.
- 이미 올바른 문자열 그룹 키는 그대로 허용한다.
- 등록되지 않은 그룹은 `ValueError`를 발생시킨다.
- 그룹 가중치에 비유한값이 있거나 활성 그룹 행의 합이 1이 아니면
  `PreprocessingError`를 발생시킨다.
- `capacity_mw`만 있는 입력은 기존과 같이 `cap_kw`로 환산한다.

이 변경으로 그룹 키 불일치가 다시 발생해도 0 행렬이 조용히 다음 단계로 전달되지 않는다.

### `Windforce/FeatureEngineer.py`

- 공개 메서드 `computeGroupWeight()` 안의 중복 IDW 구현을 제거했다.
- 공통 함수 `compute_group_weight()`를 호출하도록 변경해 그룹 키 정규화와 검증 로직을
  한 곳에서 관리한다.
- 공개 메서드 시그니처와 반환 shape `(3, 격자 수)`는 유지했다.

### baseline 3~6 노트북

- 터빈 메타의 설비용량을 kW로 변환한 직후 그룹 키를 명시적으로
  `kpx_group_N` 형식으로 바꾸도록 수정했다.
- Step 3 설명에도 IDW에 전달하는 그룹 키 계약을 반영했다.
- baseline 3·5·6의 `ROOT`를 현재 작업공간 `D:/workspaces/WINDFORCE`로 맞췄다.
- baseline 3의 구버전 전처리 호출을 현재 API인 `transformLDAPS()`와
  `transformGFS()`로 변경했다.
- baseline 4의 제출 파일명을 `submission_baseline4.csv`로 바로잡아
  baseline 3 결과를 덮어쓰지 않게 했다.

---

## 3. baseline 3·4 첫 24시간 예측 누락 원인과 수정

### 원인

`SequenceDataset`은 길이 24의 과거 구간 `X[i:i+24]`를 사용해 그 다음 시각을 예측한다.
따라서 평가 데이터만 `pipeline.predict(X_test_sc)`에 전달하면 평가 구간의 처음 24개 행은
예측값을 만들 수 없다.

기존 제출 코드는 다음 순서로 이 누락을 숨기고 있었다.

1. 평가 데이터만으로 LSTM 예측
2. 예측 시각을 `test_merged.iloc[SEQ_LEN:]`에 배치
3. 제출 양식으로 재색인하면서 첫 24개 시각이 `NaN`이 됨
4. `submission.ffill().fillna(0)`에서 선행 결측을 `0`으로 변환

그 결과 `2025-01-01 01:00:00`부터 24시간이 세 그룹 모두 `0.0`으로 저장됐다.

### 수정

baseline 3·4 제출 셀을 baseline 5의 정상적인 시계열 문맥 처리 방식과 같게 수정했다.

```python
pred_arr = pipeline.predict(np.vstack([X_train_sc[-SEQ_LEN:], X_test_sc]))
assert len(pred_arr) == len(test_merged), "예측 길이와 평가 시각 수가 다릅니다."
```

- 학습 데이터의 마지막 24시간을 평가 입력 앞에 붙여 평가 첫 시각부터 예측한다.
- 예측값을 `test_merged`의 전체 시각에 직접 대응시킨다.
- 학습 피처 순서를 기준으로 `common_cols`를 구성해 학습·평가 컬럼 의미와 스케일을 맞춘다.
- 예측 길이와 평가 시각 수가 다르면 즉시 중단한다.
- 그룹별 제출값에 `NaN`이 남으면 즉시 중단한다.
- 누락을 숨기던 `ffill().fillna(0)` 후처리를 제거했다.

baseline 5는 이미 학습 마지막 24시간을 붙이는 구조였으며, 결측 은폐를 막기 위해 동일한
`NaN` 검증과 후처리 제거만 적용했다. baseline 6은 제출 파일 생성 단계가 없어 이 수정 대상이 아니다.

---

## 4. 테스트 파일 처리

`tests/test_spatial_utils.py`는 다른 코드에서 import하거나 실행 시 참조하는 파일이 아니라
이번 변경을 확인하기 위한 독립 검증 파일이었다. 요청에 따라 삭제했다.

공통 코드의 런타임 가드는 그대로 남아 있으므로 잘못된 그룹 키나 유효하지 않은 IDW 가중치는
노트북 실행 중 명시적인 예외로 확인할 수 있다.

---

## 5. 현재 실행 및 산출물 상태

- 수정 전 baseline 3 전체 실행에서 IDW 피처가 0이 아닌 값으로 생성되는 것까지 확인했다.
- 그 실행으로 생성된 루트의 `submission_baseline3.csv`에서 첫 24시간 0 문제가 발견됐다.
- 이후 baseline 3·4의 제출 추론 소스를 위 방식으로 수정했다.
- 수정 후 baseline 3 전체 재실행은 사용자 요청에 따라 중단했다.
- baseline 4~6도 수정 후 전체 재실행하지 않았다.

따라서 **현재 루트의 `submission_baseline3.csv`는 첫 24시간 수정 전 산출물**이다.
노트북 소스와 CSV 상태가 서로 다르므로 이 파일을 최종 제출에 사용하면 안 된다.
baseline 3~6 노트북의 기존 출력 영역도 재실행 전 결과가 남아 있을 수 있다.

---

## 6. 사용자 검증 시 확인할 항목

노트북은 baseline 3 → 4 → 5 → 6 순서로 전체 실행한다.

- IDW 그룹 가중치 활성 행의 합이 각각 약 1인지 확인
- LDAPS `ws10`, GFS `gfs_ws_10` 등 IDW 피처의 최댓값이 0보다 큰지 확인
- 제출 CSV가 8,760행이며 샘플 제출 파일과 컬럼 순서가 같은지 확인
- 제출 CSV에 `NaN`과 음수가 없는지 확인
- 그룹별 예측이 설비용량을 넘지 않는지 확인
- baseline 3·4의 첫 24개 행이 실제 모델 예측으로 채워졌는지 확인
- 노트북에 오류 출력이 남지 않았는지 확인

재실행 시 현재 가상환경을 우선 사용하려면 PowerShell에서 다음과 같이 실행할 수 있다.

```powershell
$venvScripts = (Resolve-Path ".venv\Scripts").Path
$env:PATH = "$venvScripts;$env:PATH"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 `
  BASELINE\baseline_3.ipynb
```

나머지 노트북은 마지막 경로만 `baseline_4.ipynb`, `baseline_5.ipynb`,
`baseline_6.ipynb`로 바꿔 순서대로 실행한다.
