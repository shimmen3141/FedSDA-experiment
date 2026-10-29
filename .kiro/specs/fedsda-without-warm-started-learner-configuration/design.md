# Design Document

## Overview
最終構成の既定から`new_model_hypothesis_learners`だけを`reinitialized_learner`に替えた構成`fedsda-without-warm-started-learner`を足す。既定・goldenは変えない。

## Architecture
- `sweep/experiment_configurations.py`: 定数`WITHOUT_WARM_STARTED_LEARNER_CONFIGURATION_NAME`（値`fedsda-without-warm-started-learner`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。`build_experiment_run_settings`の分岐で、`fedsda-minimum-interval-multiples-change-interval-search`と同じ形で`training_data_assignment_settings`を`replace`する（値は`training_data_assignment_settings.py`の`REINITIALIZED_LEARNER_POLICY_NAME`）。依存境界の許可に新しいimportを登録する。
- 組合せの検査: `reinitialized_learner`は最小記述長の選択と組み合わせられ（今の検査はMDLでない選び方に`reinitialized_learner`を求めるだけ）、`every_sample_with_reverse_prequential_code`とも組み合わせられる（新しい扱いの分岐は、設定が両方の学習者のときだけwarm startの途中までの和を計算する）。新しい検査は足さない。
- test: 新しいファイル`tests/refactoring/test_fedsda_without_warm_started_learner_configuration.py`（要求2.1・2.2）。`tests/refactoring/behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ登録する。要求2.2の全体runは、条件を固定する（`fedsda-sine2-adoption-without-validation`のケースの条件。版16の既定では、このケースで新しいモデルが選ばれ、必須の経路になっている）。新しい構成でも新しいモデルが1回以上選ばれることを実装の前に確かめ、選ばれなければ主担当へ報告する（条件を探さない）。選択の記録の`description_length`が、記録から再計算した値と一致することまでは求めない（記録の列に前半の損失と切替の符号長が無いため）。代わりに、学習者の名前の列に`warm_started_learner`が無いことと、純粋な計算（`select_new_model_learner`）に学習者1つの辞書を渡すとln 2を足さないことを、部品のtestで確かめる。
- 文書: `docs/experiments/experiment-sweep.md`（構成の一覧）、`docs/components/model-reuse.md`（最小記述長の節。warm startを外して比べる構成がある）、NEW-006（状態の行の後に、版16で外して比べる構成を足したこと）。

## Validation
新しいtest、構成・掃引・識別子・依存境界のtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは、変更がsweepの構成の追加だけなので、関係するtestと基準版の照合で代える（spec Task 2で範囲を記録する）。

## 本規模の追加の実行（要求3.2）
- 6データ: `--configurations fedsda-without-warm-started-learner --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study fedsda-component-ablation-v16 --label without-warm-started-learner-seeds0-9`（240 run）。
- 新しい4データ: まだ実行していない構成要素の比較のコマンド（`--configurations fedsda fedsda-minimum-interval-multiples-change-interval-search fedsda-without-prediction-weight-recalibration fedsda-overall-loss-esr-detection fedsda-without-concept-specific-adapter fedsda-loss-distance-clustering fedsda-local-parameter-prediction fedsda-without-candidate-validation fedsda-single-model-prediction --datasets fmnist2 fmnist4 rotmnist2 rotmnist4 ... --study fedsda-component-ablation-new-datasets-v16 --label core-components-v16-seeds0-9`）の構成に`fedsda-without-warm-started-learner`を加える（10構成×160 = 1,600 run）。比べる相手は同じ実行の`fedsda`。
- 6データの追加分は、実行中の`fedsda-component-ablation-v16`（基準版16、このspecの前のcommit）の`fedsda`と比べる。このspecは既定と`fedsda`の設定・識別子を変えないので、コードのcommitが違っても`fedsda`の値は同じ（基準版16のgoldenの照合で確かめる）。
