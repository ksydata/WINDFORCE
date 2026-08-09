- FeatureEngineer
  - LDAPS 데이터 : 16개 격자(1.5km 해상도), 예보 발행은 매일 13시 한 번
  - 즉, 최근 24시간 예보 시점을 보고 다음 시점의 발전량 예측하는 구조
    - .transform()
      - 풍속 급변 검사 (같은 격자 내 1시간에 15 m/s 이상 편차 발생)
      - 온도(기온 < 이슬점), 기압(<= 0), 비습 정리 후/ 상대습도 100% 이상 여부, (습윤) 공기밀도 계산 
      - 풍향 sin, cos 변환(순환 인코딩)
      - 터빈, 격자(grid_id) 전체 이용하여 파생변수 생성
  - GFS 데이터 : 9개 격자 + 여러 고도 바람 변수, 예보 발행은 매일 13시 한 번
    - .transform()    
      - 10m / 80m / 100m 바람
      - 850 / 750 / 500 pha 바람
      - PBL 바람
      - 풍속 전단 / 허브높이 풍속 외삽
      - 돌풍비율, 공기밀도, 풍력에너지 밀도
  - SCADA 데이터 : 터빈당 10분
    - .transformToLong() : 데이터 품질 검사하여 플래그 + 파워커브(이론값 P = 1/2*rho*AV^3)
      - 정격 초과 출력 / 크거나 작은 음수 발전량
      - 풍속 범위 초과 / 컷아웃 풍속
      - 고풍속 무발전 / 센서 고착
  - Utils(time / wind / spatial)
    - 시간: 시간 주기 인코딩, 짧은 결측 선형보간(30분 단위), LSTM Deep Learning Model용 3차원 sequence 생성
    - 바람: 풍향을 u,v성분에서 복원 / 여러 고도 풍속으로 전단지수 계산 / 습윤 공기밀도 계산 / 풍력에너지밀도 계산
    - 공간: 예보 격자 데이터 가까운 4개를 통해, 거리 제곱의 역수로 가중(IDW)

- LSTMPipeline
  - 타깃을 발전량 / 설비용량을 정규화 + train 구간에만 min max 스케일링
  - 출력은 sigmoid로 0~1로 값 제한
  - 시간순으로 데이터 80:20 분할
  - 초기 warmup epochs동안 Smooth L1 사용(FICR의 계단함수는 미분할 수 없으므로 sigmoid 두 개로 부드럽게 근사)
  - warmup(15로 정의) 후 Smooth L1 + ScoreLossFunction 조합으로 모형 적합(epochs = 80)
  - 가장 높은 검증 총점의 가중치 복원
  - 예측 시 설비용량을 곱하여 단위를 kWh(시간 단위)로 전환
  
  - multi-step day-ahead 구조와 차이: 핵심 개선 방향은 현재의 1시간 단위 LSTM을 실제 운영 조건에 맞는 예보 발행일 단위 24시간 출력 모델로 확장하는 것
  - 출처(REF)
    - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html 
    - https://docs.pytorch.org/docs/2.13/generated/torch.nn.LSTM.html

| Step | 내용 | 사용 클래스 |
|---|---|---|
| 1 | 파일 경로 존재 확인 | `WindforceDataLoader.check_paths()` |
| 2 | SCADA/labels/ldaps/gfs/kpx_info 구조 확인 | `WindforceDataLoader` |
| 3 | DMS 좌표 파싱 → `turbine_meta`(group/lat/lon/cap_kw) 생성 | 커스텀 `parseDMS()` |
| 4 | LDAPS·GFS 격자 전처리 + IDW 그룹 집계, SCADA 파워커브 3D 시각화(LOWESS) | `LDAPSFeatureEngineer`, `GFSFeatureEngineer` |
| 5 | 평가지표·손실함수 더미 검증 | `EvaluationMetrics`, `ScoreLossFunction(k=40)` |
| 6 | 그룹별 Persistence/SVR/LSTM 학습·평가 | `WindforceDatasetBuilder`, `GroupExperimentRunner`, `LSTMPipeline` |
| 7 | 스키마 검증 + `submission_baseline5.1.csv` 저장 | 컬럼명 교집합 기반 재추론 |

- `baseline_3`(38셀 단일 노트북)에서 LSTM 3그룹 평균 총점이 0.3859였는데, `baseline_5`에서는 (0.4740+0.5023+0.5026)/3 ≈ **0.493**으로 약 0.11 상승했습니다. `improvement.md`의 진단대로 손실함수 그래디언트 소실(A)과 제출 피처 정렬 버그(B)를 고치니 LSTM이 최소한 "입력에 반응하는" 모델이 된 것으로 보입니다.
- 다만 여전히 Persistence(직전값 그대로 예측하는 oracle 기준선)가 압도적으로 높습니다. "어제와 비슷하다"는 신호 자체가 매우 강하다는 뜻이고, 아직 LSTM/SVR이 기상 예보에서 그 이상의 가치를 뽑아내지 못하고 있다는 신호로 남아 있습니다. (참고: 이 Persistence는 검증 구간의 직전 *실제값*을 쓰는 oracle 성격이라, 2025년 전체를 한 번에 제출해야 하는 실전 조건에서는 그대로 쓸 수 없는 참고 기준선입니다.)

- 잔차 학습(residual modeling) — Persistence 예측 대비 잔차를 LSTM이 학습하도록 바꾸면 최소 Persistence 수준을 보장하면서 기상 정보로 추가 개선 가능
- 파워커브 물리 피처 — Step 4-2에서 시각화한 S자 파워커브를 터빈 모델별로 피팅해 피처로 추가
- 리드타임(`lead_hour`) 구조 활용 — 예보 발행 후 1~24시간 뒤 예측이라는 특성을 명시적으로 반영
- LightGBM/XGBoost 병행 — 그룹당 2~3만 행 규모면 딥러닝보다 트리 기반이 더 안정적일 가능성
- 시계열 CV — 현재는 단일 20% 홀드아웃뿐, 계절 블록별 rolling-origin CV 필요
- 앙상블 — 위 개선 후 Persistence·GBM·LSTM(잔차)을 그룹별 가중 블렌딩