# Design Document

## Overview
候補の学習の設定に「共有部を更新するかどうか」の項目を足し、最終構成の既定を「候補検証なし＋候補の共有部の固定」へ替える。候補検証の処理は消さず、これまでの既定を構成`fedsda-with-candidate-validation`として残す。

| 設定 | これまでの既定 | 新しい既定 |
|---|---|---|
| 候補の採否の方針（`CandidateModelTrainingAndAcceptanceSettings.candidate_model_acceptance_policy`） | `current_model_first_reuse_then_two_segment_candidate_validation` | `current_model_first_reuse_then_candidate_adoption_without_validation` |
| 候補の共有部の更新（新しい項目`CandidateEpochTrainingSettings.candidate_shared_feature_update_policy`） | （項目なし。`update_shared_features_with_candidate`に当たる） | `keep_shared_features_fixed` |

### 候補検証を残す理由（主担当の判断）
1. 論文のablation（「候補検証あり」との比較）と、これまでの結果との連続に要る。
2. 実旧との照合（旧の提案構成は候補検証あり）と、基準版1〜6が、候補検証の経路を通る。消すと、それらの照合の土台が無くなる。
3. 残す費用はほぼ無い（処理は既にあり、testで守られている）。既定から外れるだけで、新しい分岐は増えない。

処理を消す判断は、別の機会に、ユーザーが決める（このspecの対象外）。

## Boundary Commitments
### This Spec Owns
`CandidateEpochTrainingSettings`の新しい項目と、`candidate_epoch_training.py`での使い方、`FedsdaRunClientSettings`の組合せの検査、最終構成の既定、`sweep/experiment_configurations.py`（新しい構成、`fedsda-without-candidate-validation`の明示）、旧と比べる設定の入口（`tests/refactoring/test_fedsda_run_client.py`の`use_legacy_compatible_policies`と`assert_run_client_uses_legacy_compatible_policies`、旧のgoldenの条件`make_golden_condition_run_settings`ほか）、基準版のhelper（版1〜6の明示、版7）、既定に依るtest、研究ツールの構成名、文書。
### Out of Boundary
候補検証の処理、採用の手順（`adopt_candidate_as_current_training_model`と共有部の反映）、候補の初期値・エポック・早期終了、FedDrift・FedAvg、固定旧、研究室サーバ用のgolden。
### Allowed Dependencies
新しいmoduleは無い。`fedsda_run_client_settings.py`は、既に両方の設定の型をimportしている見込み（実装で確かめる。足すなら依存境界の許可へ登録する）。
### Revalidation Triggers
候補の学習を通る全test、既定のbuilderを通るtest、掃引のrunの識別子（FedSDAの全runで変わる）、基準版のhelper、構成の一覧（コマンドで選べる名前が1つ増える）。

## Architecture
### 候補の共有部の更新（Requirement 1）
- `CandidateEpochTrainingSettings`に、必須の項目`candidate_shared_feature_update_policy`（`allowed_parameter_values`は`update_shared_features_with_candidate`と`keep_shared_features_fixed`）を足す。既存の項目と同じく既定値を持たない（全部の生成箇所で明示する）。
- `candidate_epoch_training.py`: 入力の検査で設定を組み立て直す箇所へ新しい項目を足す。`_train_candidate_dataset_epochs`は、共同更新`perform_joint_model_parameter_update`へ、`update_shared_features=(policy == "update_shared_features_with_candidate")`を渡す（今は`True`の固定）。この関数は、固定エポックと早期終了の両方の経路から呼ばれるので、両方に効く。
  - 「固定する」とき: 共同更新は、共有部の順伝播を勾配なしで行い、共有部のoptimizerを進めない（`perform_joint_model_parameter_update`の既存の分岐。研究ツールの`_frozen_candidate_backbone_policy`と同じ形）。計算回数の記録は、勾配なしの順伝播を推論として数える（`model_computation_measurement`の既存の扱い）。乱数の消費（DataLoaderのshuffle、早期終了の分割）は変わらない。
  - 「候補と一緒に更新する」とき: これまでと同じ（引数の値が同じ）。
  - 共有部を持たない構造: 共有部にパラメータが無いので、どちらでも同じ（共同更新の既存の分岐。testで確かめる）。
- 組合せの検査（`FedsdaRunClientSettings`の検査）: 採否の方針が`current_model_first_reuse_then_two_segment_candidate_validation`で、共有部が`keep_shared_features_fixed`なら拒否する。理由: 検証の期間の後に採用すると、候補の共有部（警報の時点の値）が、期間中に学習したclientの共有部を、警報の時点へ戻す。この組合せは研究ツールで不安定だった（MNIST4で−1.31ほか）。

### 最終構成と派生の構成（Requirement 2）
- `build_final_configuration_fedsda_run_settings`: 採否の方針を`current_model_first_reuse_then_candidate_adoption_without_validation`、候補の共有部を`keep_shared_features_fixed`へ。検証の標本数（10）は、そのまま持つ（検証なしでは使わない）。
- 構成`fedsda-with-candidate-validation`（新規。定数`CANDIDATE_VALIDATION_CONFIGURATION_NAME`）: 既定から、採否の方針を候補検証ありへ、候補の共有部を`update_shared_features_with_candidate`へ替える（これまでの既定と同じ設定）。2つの要素を替えるのは、上の組合せの制約のため（説明の文に書く）。`EXPERIMENT_CONFIGURATION_NAMES`の最後へ足す。
- `fedsda-without-candidate-validation`: 既定から、候補の共有部だけを`update_shared_features_with_candidate`へ替える（採否の方針は既定と同じ値になる。替える処理は、両方を明示する）。設定は、これまでと同じ（新しい項目を除く）。名前は、過去の結果との連続のために残す（説明の文に書く）。
- ほかの派生の構成は、既定を引き継ぐ（候補検証なし＋固定）。全構成の組立てと、小さい条件での実行が例外を出さないことを、testで確かめる。

### 旧と比べる設定の入口（Requirement 3.1）
- `use_legacy_compatible_policies`: 替える項目へ、`candidate_epoch_training_settings.candidate_shared_feature_update_policy="update_shared_features_with_candidate"`を足す（旧には共有部の固定が無い）。採否の方針は替えない（候補検証を外した構成の照合は、この入口を通さず、`valid_run_settings_mapping`から組み立てる。採否の方針を決めるのは、呼ぶ側）。
- `assert_run_client_uses_legacy_compatible_policies`: clientの設定の`candidate_epoch_training_settings.candidate_shared_feature_update_policy`が`update_shared_features_with_candidate`であることを確かめる。
- 旧のgoldenの条件（`make_golden_condition_run_settings`と、`test_fedsda_run_metric_derivation.py`の`make_golden_condition_settings`）は、旧の提案構成なので、採否の方針を候補検証ありに明示する。
- `valid_run_settings_mapping`と、test・fresh scriptの`CandidateEpochTrainingSettings`の生成は、`update_shared_features_with_candidate`を明示する（これまでと同じ動作）。
- 構成の実旧との照合（`test_fedsda_configuration_variants.py`）: `fedsda-without-candidate-validation`の変形（`modify_run_participant_settings`）へ、共有部の明示を足す。`fedsda-with-candidate-validation`は、実旧の提案構成と同じ設定なので、構成の照合の表へ入れず（提案構成の照合が覆う）、除外の集合へ足す。

### 基準版のgolden（Requirement 3.2、4。`tests/refactoring/behavior_golden_cases.py`）
- 版1〜6の扱い:
  - 共有部の更新は、版1〜6の保存済みの設定に無い項目なので、「版の後に足された項目」の表へ、版1〜6とも`update_shared_features_with_candidate`を足し、`build_case_settings`がこの項目を明示する。
  - 採否の方針は、版1〜6の保存済みの設定が持つ項目なので、「版の後に既定の値が変わった項目」の表へ、版1〜6の候補検証ありを足す。ただし、`fedsda-without-candidate-validation`のケースは、構成が検証なしを明示するので、この項目を当てない（当てると別の設定になる）。
- 版7を足す（`SUPPORTED_BASELINE_VERSIONS`、`BASELINE_VERSION`、`CASE_NAMES_BY_BASELINE_VERSION`）。ケースは、版6の19に、新しい既定の大きい条件のケースを2つ足した21:
  - 版6の大きい条件のFedSDAの4ケース（`fedsda-sine2-events`、`fedsda-mnist2-decisions`、`fedsda-sine2-reference-reuse`、`fedsda-sine2-alternative-reference-reuse`）は、候補検証の経路を守るケースなので、版7では構成`fedsda-with-candidate-validation`で実行する（条件は版6と同じ）。設定がこれまでの既定と同じなので、値は版6と同じになる。
  - 新しいケース`fedsda-sine2-adoption-without-validation`と`fedsda-mnist2-adoption-without-validation`は、新しい既定（`fedsda`）で、`fedsda-sine2-events`・`fedsda-mnist2-decisions`と同じ条件（seedは、必須の経路を通るものを下調べで選ぶ。選んだ理由をhelperの注記へ書く）。
  - 小さい条件の構成のケース（`BASELINE_CONFIGURATION_CASE_NAMES`）は、各構成のまま（`fedsda`ほかは新しい既定を引き継ぐ）。
- 経路の記録（`coverage`）: 版7から、FedSDAのケースに「採用した候補の数」（`adopted_candidate_count`。適応の行動の指標`change_response_metrics.adaptation_action_metrics.adopt_candidate_model_count`。検証ありの採用も含む。新しい既定では、全部が検証なしの採用）を足す。必須の経路（`assert_required_paths`）は、新しい2ケースで、この数、学習帰属の切替え、予測の重みの再較正が0でないこと、sine2では統合も0でないことを求める。候補検証の4ケースの必須の経路は、版6と同じ。
- `BASELINE_VERSION`を7へ替えるのは、版7のWindowsとWSLのgoldenを生成して置いた後にする。`core/run_provenance.py`の`BEHAVIOR_BASELINE_VERSION`も同時に7へ上げる。sourceとtestをcommitしてから生成する（この中間のcommitでは、版7のgoldenの存在を確かめるtestが失敗する。前の版と同じく避けられない）。

### 版6→7の差の説明（`tests/refactoring/test_important_behavior_regression_goldens.py`）
新しいtest `test_baseline_7_keeps_baseline_6_outside_the_new_default`で、次を確かめる:
1. 候補検証の4ケース、FedDrift・FedAvgのケース、`fedsda-without-candidate-validation`のケースは、指標・記録の表・`global_parameters`・`coverage`（版7で足した項目を除く）が版6と同じ。設定は、新しい項目を除いて版6と同じ。
2. 新しい2ケースは、版6に無い（比べない）。必須の経路で守る。
3. 新しい既定を引き継ぐ小さい条件のケース（上の1以外のFedSDAのケース）は、差があってもよい（候補の経路を通らないケースは同じ値になりうるので、差を求めない）。

## 既定に依る既存のtestの扱い
既定を替えると、意図して失敗するtestがある。実装で、全pytestの失敗と照らし合わせて合わせる:

| test | 今の前提 | 新しい期待 |
|---|---|---|
| `test_fedsda_final_configuration_run_settings.py`の、既定が「旧の固定設定＋列挙した差」と一致するtest | 候補検証あり | 列挙した差に、採否の方針と候補の共有部を足す（docstringも） |
| `test_fedsda_configuration_variants.py`の、構成の一覧と除外の集合、`fedsda-without-candidate-validation`の変形 | 一覧は17、既定は検証あり | 一覧へ新しい構成、除外の集合へ新しい構成、変形へ共有部の明示 |
| `test_fedsda_default_synchronized_local_mixture_prediction.py`ほか、構成ごとの値を確かめるtest | 構成の一覧 | 新しい構成を含めて合わせる |
| 掃引のrunの識別子の期待値、構成の既定のtest、fresh scriptの既定の名前 | 今の既定の設定 | 新しい既定の値へ（全pytestで失敗したものから合わせる） |

実旧と比べるtestは、上の表に入れず、期待値を変えない（失敗したら、旧と比べる入口の明示の漏れを直す）。候補検証の部品のtestは、`CandidateEpochTrainingSettings`の生成に新しい項目を足すだけで、期待値を変えない。

## 旧と違う点
既定を使う実行（掃引を含む）のFedSDAの構成で、候補を検証せずに採用し、候補の学習では共有部を更新しない。そのため、学習・帰属・登録・統合・通信量・予測が変わる（研究記録の本規模の比較のとおり）。挙動の修正の比較script（直下の`run_behavior_fix_comparison.py`）は、既定の`fedsda`を土台に3つの扱いを明示するので、これからは新しい既定の上で比べる（説明の文に書く）。`fedsda-without-candidate-validation`と`fedsda-with-candidate-validation`は、値が変わらない。

## Validation Design
- 候補の学習: 「固定する」で、学習の後の共有部のパラメータが学習の前と同じで、概念固有部は変わること（固定エポックと早期終了の両方）。同じ入力で2つの値を学習し、学習の後のtorch・Python・NumPyの乱数の状態が同じこと（乱数の消費が変わらない）。「固定する」で、候補の共有部のoptimizerの状態（Adamのstep数・moment）が学習の前と同じで、概念固有部のoptimizerは進むこと。「候補と一緒に更新する」で、これまでの値と同じこと（既存の実旧との照合のtestが、明示した値で通る）。共有部を持たない構造で、2つの値の結果が同じこと。組合せの検査が、候補検証あり＋固定を拒否し、ほかの3つの組合せを受け付けること。
- 構成: FedSDAの全構成の、採否の方針と候補の共有部の値を1つのtestで確かめる（`fedsda-with-candidate-validation`は検証あり・更新、`fedsda-without-candidate-validation`は検証なし・更新、ほかは検証なし・固定）。全構成の組立てと小さい実行が例外を出さないこと。掃引のコマンドの引数の一覧が変わらないこと。`fedsda-with-candidate-validation`の専用のtest: 既定と比べて、2つの項目だけが違う（保存の形で比べる）。
- 旧と比べる入口: 既存の実旧との照合（期待値は変えない）。入口の検査が、共有部の固定の設定を拒否すること。
- 基準版: 版1〜6の照合（値を変えずに）、版7の照合、版6→7の差の説明、版7の必須の経路。
- 研究ツール: 構成名を替えた研究ツールの、明示実行のchecks（`tools/research_*/checks.py`のうち、替えた2つ）。
- 全pytest、Ruff、Pyright、WSL（goldenの版1〜7と、このspecのtest）。

## File Structure Plan
- `src/federated_learning_experiments/learning/training/candidate_epoch_training_settings.py`、`candidate_epoch_training.py`。
- `src/federated_learning_experiments/runtime/fedsda_run_client_settings.py`（組合せの検査）、`fedsda_final_configuration_run_settings.py`、`post_alarm_candidate_validation_session_start.py`（設定の組立て直しがあれば）。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`、`src/federated_learning_experiments/core/run_provenance.py`。
- `tests/refactoring/`: 新規`test_fedsda_candidate_shared_feature_update_policy.py`（候補の学習と組合せの検査）と`test_fedsda_with_candidate_validation.py`（構成）、`behavior_golden_cases.py`、`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v7/`、`test_fedsda_run_client.py`（旧と比べる入口）、`CandidateEpochTrainingSettings`を生成するtestとfresh script、既定に依るtest。
- 研究ツール: `tools/research_provisional_candidate_expert/run.py`、`tools/research_prediction_initial_mass/run.py`・`checks.py`、`tools/research_candidate_validation_waiting/checks.py`、`run_candidate_validation_waiting_diagnostic.py`。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/refactoring-baseline.md`、`docs/overview/proposed-method.md`、`docs/research/fedsda-provisional-candidate-expert-20261019.md`、`.kiro/steering/resume.md`。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1.1〜1.5 | 新しい項目、候補の学習の分岐、組合せの検査、候補の学習のtest |
| 2.1〜2.4 | 最終構成の2つの値、新しい構成、`fedsda-without-candidate-validation`の明示、構成の値のtest |
| 3.1 | 旧と比べる入口の明示と検査、旧のgoldenの条件、実旧との照合の既存test |
| 3.2 | 版1〜6の明示と照合 |
| 3.3 | 候補検証の部品の既存test |
| 4.1〜4.3 | 版7、版6→7の差の説明のtest、新しい2ケースの必須の経路 |
| 5.1 | 研究ツールの構成名と、そのchecks |
| 5.2 | 文書の更新 |
