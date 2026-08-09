import nbformat as nbf

src = nbf.read('BASELINE/baseline_5.ipynb', 4)
cells = [src.cells[i] for i in range(31)]
cells[0] = nbf.v4.new_markdown_cell('# Baseline 6 — 버전별 LSTM·피처 엔지니어링 실험\n\n원본 모델은 보존하고, LSTM v1/v2/v3와 피처 all/corr_filtered/pca/tree_importance 조합을 동일한 시간순 검증으로 비교합니다.')
cells[3].source += '\nfrom Windforce.Modeling import VersionedExperimentRunner\n'
cells.append(nbf.v4.new_markdown_cell('## 실험 조합 설정'))
cells.append(nbf.v4.new_code_cell('''MODEL_VERSIONS = ["v1", "v2", "v3"]\nFEATURE_VERSIONS = ["all", "corr_filtered", "pca", "tree_importance"]\nMODEL_PARAMS = {"epochs": 35, "warmup_epochs": 8, "loss_k": 40.0, "regression_weight": 0.75, "patience": 7}\nFEATURE_PARAMS = {\n    "corr_filtered": {"threshold": 0.95},\n    "pca": {"variance": 0.95},\n    "tree_importance": {"keep_ratio": 0.35, "min_features": 8},\n}\nprint(f"{len(MODEL_VERSIONS) * len(FEATURE_VERSIONS)}개 버전 조합 실행")'''))
cells.append(nbf.v4.new_markdown_cell('## 모델×피처 전체 실험'))
cells.append(nbf.v4.new_code_cell('''runner = VersionedExperimentRunner(\n    dataset_builder=dataset_builder, metrics=metrics, groups=GROUPS,\n    test_ratio=0.2, seq_len=24, model_versions=MODEL_VERSIONS,\n    feature_versions=FEATURE_VERSIONS, model_params=MODEL_PARAMS,\n    feature_params=FEATURE_PARAMS, random_state=SEED,\n)\nresult_df = runner.run_all()\nsummary_df = runner.summary(result_df)\ndisplay(summary_df)\nresult_df.to_csv(f"{ROOT}/BASELINE/baseline_6_results.csv", index=False)\nsummary_df.to_csv(f"{ROOT}/BASELINE/baseline_6_summary.csv", index=False)'''))
cells.append(nbf.v4.new_markdown_cell('## 결과 해석용 피벗'))
cells.append(nbf.v4.new_code_cell('''display(result_df.pivot_table(index="model_version", columns="feature_version", values="score", aggfunc="mean"))\ndisplay(result_df.groupby("feature_version")["n_features"].mean().sort_values())'''))
cells.append(nbf.v4.new_markdown_cell('''## 제출 예측\n\n최고 조합을 확정한 뒤에는 `baseline_6_submission` 셀에서 동일한 피처 변환기를 학습 구간에만 적합해 평가 구간을 예측합니다. 평가 기간 라벨이 없으므로 이 셀은 실험 결과 저장과 분리했습니다.'''))
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=src.metadata), 'BASELINE/baseline_6.ipynb')

lb = [nbf.v4.new_markdown_cell('# Baseline 6 Leaderboard\n\n모델 버전과 피처 버전 조합별 3그룹 평균 성능을 비교합니다.'),
      nbf.v4.new_code_cell('''from pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path("/Users/ksydata/WINDFORCE")\nresults = pd.read_csv(ROOT / "BASELINE/baseline_6_results.csv")\nsummary = (results.groupby(["model_version", "feature_version"], as_index=False)\n           .agg(mean_nmae=("nmae", "mean"), mean_ficr=("ficr", "mean"),\n                mean_score=("score", "mean"), groups=("group", "nunique"),\n                mean_features=("n_features", "mean"))\n           .sort_values("mean_score", ascending=False))\ndisplay(summary)'''),
      nbf.v4.new_code_cell('''print("모델 버전별")\ndisplay(summary.groupby("model_version", as_index=False)[["mean_nmae", "mean_ficr", "mean_score"]].mean().sort_values("mean_score", ascending=False))\nprint("피처 버전별")\ndisplay(summary.groupby("feature_version", as_index=False)[["mean_nmae", "mean_ficr", "mean_score", "mean_features"]].mean().sort_values("mean_score", ascending=False))'''),
      nbf.v4.new_code_cell('''pivot = summary.pivot(index="model_version", columns="feature_version", values="mean_score")\ndisplay(pivot.style.format("{:.4f}").background_gradient(cmap="Blues"))\nax = pivot.plot(kind="bar", figsize=(11, 5), rot=0)\nax.set_ylabel("mean score = 0.5(1-NMAE)+0.5(FICR)")\nax.set_title("Baseline 6 version leaderboard")\nax.legend(title="feature version", bbox_to_anchor=(1.02, 1), loc="upper left")\nplt.tight_layout()''')]
nbf.write(nbf.v4.new_notebook(lb), 'BASELINE/leaderboard.ipynb')
