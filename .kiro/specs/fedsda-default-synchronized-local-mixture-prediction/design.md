# Design Document

## Overview
最終構成の既定の2つの値を替え、それに合わせて、派生の構成・旧と比べる設定の入口・基準版・test・文書を整える。混ぜる予測とFixed-Shareの動作は変えない。

| 設定 | これまでの既定 | 新しい既定 |
|---|---|---|
| 予測に使うパラメータ（`PredictionCombinationSettings.prediction_model_parameter_source`） | `current_local_parameters` | `mixture_of_synchronized_and_local_parameters` |
| Fixed-Shareの時間尺度T（`fixed_share_weight_redistribution_time_scale_samples`） | 30 | 120 |
| 帰属の保留の容量 | 30 | 変えない |

## Boundary Commitments
### This Spec Owns
`runtime/fedsda_final_configuration_run_settings.py`の2つの既定、`sweep/experiment_configurations.py`（`fedsda-single-model-prediction`の明示、構成`fedsda-local-parameter-prediction`の追加）、旧と比べる設定の入口（`tests/refactoring/test_fedsda_run_client.py`の`use_legacy_compatible_policies`と`assert_run_client_uses_legacy_compatible_policies`）、基準版のgoldenのhelper（版1〜5の明示、版6）、既定に依るtest、文書。
### Out of Boundary
混ぜる予測・Fixed-Share・写しのownerの実装、帰属の保留の容量、FedDrift・FedAvg、固定旧、研究室サーバ用のgolden。
### Allowed Dependencies
新しいmoduleは無い。構成のmoduleは、既存の`_replace_prediction_combination_settings`を使う。依存境界の許可は、変える必要がない見込み（最終構成のmoduleは値を替えるだけ）。
### Revalidation Triggers
既定のbuilderを通るtest、掃引のrunの識別子（FedSDAの全runで変わる）、基準版のhelper、構成の一覧（コマンドで選べる名前が1つ増える）。

## Architecture
### 最終構成と派生の構成
- `build_final_configuration_fedsda_run_settings`: `prediction_model_parameter_source`を`mixture_of_synchronized_and_local_parameters`へ、Tの定数`_FIXED_SHARE_WEIGHT_REDISTRIBUTION_TIME_SCALE_SAMPLES`を120へ。
- `fedsda-single-model-prediction`: 混ぜる値は、現在のモデルだけで予測する設定と組み合わせられない（`FedsdaRunClientSettings`の検査）。そこで、`_replace_prediction_combination_settings`の替えるfieldへ`prediction_model_parameter_source="current_local_parameters"`を足す。Tは既定（120）を引き継ぐ（予測には使わない。下調べで、この構成の全出力が版5と同じだった）。
- 構成`fedsda-local-parameter-prediction`（新規。定数`LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME`）: 既定から`prediction_model_parameter_source`だけを`current_local_parameters`にする。`EXPERIMENT_CONFIGURATION_NAMES`の最後へ足す。goldenの事例にせず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、専用のtest `tests/refactoring/test_fedsda_local_parameter_prediction.py`と登録する。
- `fedsda-synchronized-model-prediction`・`fedsda-synchronized-local-mixture-prediction`: 予測に使うパラメータを明示しているので変えない（Tは120を引き継ぐ）。後者は既定と同じ設定になる（名前だけが違う）。残すのは、これまでのコマンドと文書がこの名前を使っているため。ただし、この名前の過去の結果はT30（容量の軸で替えたrunを除く）で、これからの結果はT120なので、値は連続しない（識別子は設定の内容から作るので、取り違えはない）。説明の文に、このことを書く。
- ほかの派生の構成（ADWINの2構成、独立のモデル、adapterなし、AdaHedgeの重みほか）は、既定を引き継ぐ。全構成の組立てと、小さい条件での実行が例外を出さないことを、testで確かめる（下調べでは、goldenの全ケースの構成で実行できた）。

### 旧と比べる設定の入口
- `use_legacy_compatible_policies`: 替える項目へ、`prediction_combination_settings`の`prediction_model_parameter_source="current_local_parameters"`と、Tを`training_data_assignment_settings.pending_assignment_buffer_capacity_samples`と同じ値に、を足す（旧の分け合いの割合は1/N_FIFO）。
- `assert_run_client_uses_legacy_compatible_policies`: clientの設定（`run_client._run_client_settings`。testの補助なので非公開の属性を読む。理由をコメントに書く）の`prediction_combination_settings.prediction_model_parameter_source`が`current_local_parameters`であることと、`fixed_share_weight_redistribution_time_scale_samples`が`training_data_assignment_settings.pending_assignment_buffer_capacity_samples`と同じことを確かめる（今は登録・保留・警報の3つだけ）。
- `make_run_client_settings`は、既定のbuilderを通らず、`valid_run_settings_mapping`の明示の値（局所、T＝容量）を使うので、変えない。
- 既定を替えた後の全pytestで、実旧と比べるtestが失敗したら、期待値は変えず、明示の漏れを直す。

### 基準版のgolden（`tests/refactoring/behavior_golden_cases.py`）
- 版1〜5の扱い:
  - 予測に使うパラメータは、すでに「版の後に足された項目」の表（`_PREDICTION_MODEL_PARAMETER_SOURCE_DEFAULTS`）にある。`build_case_settings`は今、この項目を明示していない（既定が同じ値だったため）。この項目が表にあれば、`current_local_parameters`を明示して組み立てるように広げる。
  - Tは、版1〜5の保存済みの設定が持つ項目（値30）なので、「版の後に既定の値が変わった項目」の表へ、版1〜5のT＝30を足し、`build_case_settings`がこの経路も扱うようにする（今は登録と警報の経路だけ）。
  - `fedsda-single-model-prediction`のケースも、版1〜5ではT＝30を明示する（保存済みの設定はT30。新しい既定のT120を引き継ぐと、比べられない）。予測に使うパラメータは、構成が局所を明示するので、版1〜5でも同じ値になる。「変わった項目」の表は、FedSDAの全ケース（単一モデルのケースを含む）に当てる。
- 版6を足す（`SUPPORTED_BASELINE_VERSIONS`、`BASELINE_VERSION`、`CASE_NAMES_BY_BASELINE_VERSION`。ケースは版5と同じ19）。設定は、新しい既定のまま。版6は予測の側だけを変えるので、必須の経路（`assert_required_paths`）は版5と同じ分岐を使う（下調べで、全ケースの`coverage`が版5と同じだった）。
- `BASELINE_VERSION`を6へ替えるのは、版6のWindowsとWSLのgoldenを生成して置いた後にする（前の版と同じ）。sourceとtestをcommitしてから生成する（この中間のcommitでは、版6のgoldenの存在を確かめるtestが失敗する。前の版と同じく避けられない）。

### 版5→6の差の説明（`tests/refactoring/test_important_behavior_regression_goldens.py`）
前の版の差は「直した挙動の事象の回数」で説明したが、版6は動作の修正ではなく予測の方式の変更なので、別のtest `test_baseline_6_differs_from_5_only_on_the_prediction_side`で、全ケースについて次を確かめる（`BASELINE_VERSION_DIFFERENCES`には足さない）:
1. `global_parameters`と`coverage`は、版5と同じ。
2. 記録の表は、予測の側の列の一覧`_PREDICTION_SIDE_RECORD_TABLE_COLUMNS`以外が、版5と同じ。
3. 指標は、予測の側の項目の一覧`_PREDICTION_SIDE_RUN_METRIC_PATHS`以外が、版5と同じ。
4. 重み付けの予測を使うFedSDAのケース（`fedsda-single-model-prediction`以外のFedSDAのケース）は、ケースごとに、予測の側の一覧の項目のどれかが版5と違う（下調べで、小さい条件のケース——`fedsda`ほか——は、予測の記録と精度が同じで、推論の回数だけが違った）。さらに、`sample_prediction`の表が版5と違うケースが、少なくとも1つある（下調べで、sine2・mnist2の4ケース）。
5. FedDrift・FedAvgのケースと`fedsda-single-model-prediction`は、全部が版5と同じ。

予測の側の一覧（下調べ、2026-10-16、Windows、goldenの全19ケース。一覧は、prefixで持つ）:
- 記録の表: `sample_prediction.`の全列、`server_round.local_processing_*_inference_example_count`と`server_round.local_processing_*_inference_forward_multiply_accumulate_count`（混ぜる予測が、局所の結合予測も計算する分の推論の回数）、`client_weight_update_counts.fixed_share_weight_leader_switch_count`、`model_registration.maximum_weight_prediction_sample_count`（予測の重みの診断）。
- 指標: `prediction_accuracy`、`stable_period_prediction_accuracy`、`prediction_diagnostic_metrics.`の全項目、`model_computation_counts.`の`shared_part_inference_example_count`・`shared_part_inference_forward_multiply_accumulate_count`・`concept_specific_part_inference_example_count`・`concept_specific_part_inference_forward_multiply_accumulate_count`の4つ、`computation_cost_summary.`の、下調べで違った9項目（完全な名前で持つ。`client_forward_multiply_accumulate_count`ほか。研究の記録の`research.md`に一覧）。
- `computation_cost_summary`の9項目は、学習の順伝播・逆伝播も含む合計なので、それだけでは学習の側の変化を隠しうる。そこで、合計のもとになる`model_computation_counts`の学習の側の項目（推論の4つ以外の全項目）は、一覧に入れず、版5と同じであることを確かめる（学習の側が変われば、そこで失敗する）。
- 通信量（`communication_volume`）・検出・帰属・学習・モデルの統合の項目は、一覧に入れない（違えば失敗する）。

## 既定に依る既存のtestの扱い

既定を替えると、意図して失敗するtest（仕様レビュー1回目で洗い出した。実装で、全pytestの失敗と照らし合わせる）:

| test | 今の前提 | 新しい期待 |
|---|---|---|
| `test_fedsda_final_configuration_run_settings.py`の、既定が「旧の固定設定＋列挙した差」と一致するtest（225〜387行） | 予測は局所、T30 | 列挙した差に、混ぜる予測とT120を足す（docstringと選択の一覧も） |
| `test_fedsda_configuration_variants.py`の、構成の変形の表（単一モデルの`modify`、225〜241行）と、除外の集合（952〜960行） | 既定の予測に使うパラメータは局所 | 単一モデルの変形に局所の明示を足す。除外の集合へ新しい構成名を足す |
| `test_fedsda_synchronized_local_mixture_prediction.py`の、混ぜる構成が既定と予測に使うパラメータだけ違うtest（333〜351行） | 既定は局所 | 既定と同じ設定であることを確かめるtestへ変える（`fedsda-local-parameter-prediction`との比較で、予測に使うパラメータだけが違うことを確かめる） |
| `test_fedsda_synchronized_model_prediction.py`の、既定は写しを作らないtest（471〜488行）と、既定を局所と仮定するtest（514〜534行） | 既定は局所 | 比べる相手を`fedsda-local-parameter-prediction`（局所を明示）へ替える |
| `test_fedsda_fixed_share_time_scale_axis.py`（77・95〜98・178行の、構成の値のT＝30） | 既定のT30 | 既定のT120へ |
| 掃引のrunの識別子の期待値（`sine2_interval3_seed0_…`ほか）、構成の既定のtest、fresh scriptの既定の名前 | 今の既定の設定 | 新しい既定の値へ（全pytestで失敗したものから合わせる） |

実旧と比べるtestは、上の表に入れず、期待値を変えない（失敗したら、旧と比べる入口の明示の漏れを直す）。

## 旧と違う点
既定を使う実行（掃引を含む）の、FedSDAの構成の予測と、推論の計算回数が変わる。挙動の修正の比較script（直下の`run_behavior_fix_comparison.py`）は、既定の`fedsda`を土台に3つの扱いだけを明示するので、これからは混ぜる予測・T120の上で比べる（局所・T30には固定しない。比べる3つの扱いは予測と独立で、既定の上で比べるのが目的に合う。説明の文に書く）。モデルのパラメータ、検出・帰属・学習・統合・通信量は変わらない（下調べで、goldenの全ケースで確かめた。版5→6のtestで固定する）。`fedsda-single-model-prediction`は、値が変わらない。

## Validation Design
- 構成: FedSDAの全構成の、予測に使うパラメータとTの値を1つのtestで確かめる（`fedsda-single-model-prediction`と`fedsda-local-parameter-prediction`は局所、同期の構成は同期、ほかは混ぜる値。Tは全部120）。全構成の組立てが例外を出さないこと。掃引のコマンドの引数の一覧が変わらないこと。
- `fedsda-local-parameter-prediction`の専用のtest: 既定と比べて、予測に使うパラメータだけが違う（保存の形で比べる）。
- 旧と比べる入口: 既存の実旧との照合（期待値は変えない）。入口の検査が、混ぜる予測の設定を拒否すること。
- 基準版: 版1〜5の照合（値を変えずに）、版6の照合、版5→6の差の説明、版6の必須の経路。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜6と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`、`src/federated_learning_experiments/sweep/experiment_configurations.py`。
- `tests/refactoring/`: `behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v6/`、`test_fedsda_run_client.py`（旧と比べる入口）、新規`test_fedsda_default_synchronized_local_mixture_prediction.py`（構成の既定）と`test_fedsda_local_parameter_prediction.py`、既定に依るtest（掃引のrunの識別子、最終構成の既定の照合、構成の派生、混ぜる予測の構成のtest、fresh script）。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/refactoring-baseline.md`、`docs/research/research-backlog.md`、`docs/overview/proposed-method.md`、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.4 | 最終構成の2つの値、単一モデルの構成の明示、局所の構成の追加、構成の既定のtest |
| 2.1 | 旧と比べる入口の明示と検査、実旧との照合の既存test |
| 2.2 | 版1〜5の明示と照合 |
| 3.1〜3.3 | 版6、版5→6の差の説明のtest |
| 4.1 | 文書の更新 |
