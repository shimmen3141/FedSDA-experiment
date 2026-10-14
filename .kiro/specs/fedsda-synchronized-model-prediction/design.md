# Design Document

## Overview
予測の設定（`PredictionCombinationSettings`）へ`prediction_model_parameter_source`を足す。clientのownerに、同期の時点の保有モデルの写しを持つ`SynchronizedPredictionModelSnapshot`を足し、`parameters_at_last_synchronization`のときだけ、配布の反映の直後に写しを取る。標本1件の予測（`predict_observed_sample_and_update_prediction_weights`と、その中の`_predict_with_current_training_model_only`）は、写しのownerを受け取ったとき、評価に使う分類器を、写しのownerが選んだものに替える。ほかの処理は、引数を受け取らず、今のまま。

## Boundary Commitments
### This Spec Owns
- `methods/fedsda/prediction_combination/prediction_combination_settings.py`: 項目と許す値。
- 新しいmodule `methods/fedsda/prediction_combination/synchronized_prediction_model_snapshot.py`: 写しのowner。
- `runtime/observed_sample_prediction.py`: 予測の関数の、省略可能な引数`synchronized_prediction_model_snapshot`（省略かNoneなら今と同じ）。
- `runtime/observed_sample_processing.py`: 同じ名前の省略可能な引数を、予測へ渡すだけ。ownerの検査の表（`_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME`。全entryでexactな型を求め、Noneを拒否する）へは入れず、別の分岐で、Noneか`SynchronizedPredictionModelSnapshot`のexactな型かを、状態を変える前に確かめる。予測の関数の検査（`_validate_observed_sample_prediction_inputs`の表）も同じ扱い。
- `runtime/fedsda_run_client.py`: 写しのownerを、ownerの束（`FedsdaRunClientOwners`）へは入れず、client（`FedsdaRunClient`）の属性として持つ（下の「client」）。配布の反映（`apply_global_model_distribution`）の後に、設定が`parameters_at_last_synchronization`なら写しを取る。標本の処理へ、その設定のときだけownerを渡す（`current_local_parameters`ならNone）。
- `runtime/fedsda_final_configuration_run_settings.py`、`tests/refactoring/test_run_settings_validation.py`、`tests/refactoring/fresh_process_smoke.py`（`PredictionCombinationSettings`を作る箇所の全部）: 項目を明示する（設定の項目は既定値を持たない規約）。
- `sweep/experiment_configurations.py`: 構成名と組み立て。
- `tests/refactoring/behavior_golden_cases.py`: 新しい項目を、版1〜5の「後に足された項目」に登録する。新しい構成を`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ登録する。
- 構成の一覧を使うtest（`test_fedsda_configuration_variants.py`の完全なtupleと`CONFIGURATION_VARIANTS`の除外）。FedSDAのrunの識別子の期待値。
- 新しいtest `tests/refactoring/test_fedsda_synchronized_model_prediction.py`。
- 文書。
### Out of Boundary
学習・検出・候補の検証・統計・アップロード・集約後の再較正・サーバ。FedDrift・FedAvg。既定の値。
### Allowed Dependencies
新しいmodule（`methods/`の層）は、`copy.deepcopy`と、`learning/`の層の`HeldModelTrainingStateRegistry`（`learning/training/held_model_training_state_registry.py`）・`ModelArchitectureClassifier`（`learning/models/model_architecture_classifier.py`）を使う（methodsからlearningへのimportは、`methods/fedavg/fedavg_client.py`ほかに前例がある）。依存境界のtestの許可の一覧へ、新しいmoduleの許可（`copy.deepcopy`と上の2つの名前とmodule）と、それを読むmodule（予測・標本の処理・client）の`SynchronizedPredictionModelSnapshot`への依存を足す。
### Revalidation Triggers
FedSDAのrun設定の保存の形（FedSDAのrunの識別子のhashが全部変わる）、goldenの設定の比較、標本の処理と予測の引数。

## Architecture
### 設定
`prediction_model_parameter_source: str`（`allowed_parameter_values`: `current_local_parameters`、`parameters_at_last_synchronization`）。既定値は持たない（規約）。最終構成は`current_local_parameters`を明示する。

### 写しのowner（`SynchronizedPredictionModelSnapshot`）
- 状態: 写しの対応`model_id → (取った時点の分類器への参照, 写した分類器)`。最初は空。
- `capture_held_models(*, held_model_training_state_registry)`: 保有モデルの分類器の全部を、tupleにまとめて1回で`deepcopy`する（分類器どうしが共有する共有部は、写しの中でも1つのオブジェクトのまま）。前の写しは捨てる。
- `select_prediction_classifiers(*, held_model_training_state_registry) -> tuple[ModelArchitectureClassifier, dict[int, ModelArchitectureClassifier]]`: 保有モデルごとに、写しを持ち、取った時点の分類器と今の分類器が同じオブジェクト（`is`）なら写した分類器、そうでなければ今の分類器を返す。共有特徴を計算する分類器は、写した分類器を返したモデルがあれば、そのうちID昇順で最初のもの、無ければ（最初の同期の前など、写しを持つモデルが無いとき）、ID昇順で最初の保有モデルの今の分類器。
- 診断との対応: 診断は、写しを持つ分類器にstate_dictを読み込み、共有部は分類器どうしで同じモジュールなので、写しを持たないモデルも、共有部は写しの値で評価された。共有特徴はID昇順で最初のモデルで1回だけ計算する（今の予測の関数）。写しを持たないモデルの共有部が、写しを持つモデルと同じモジュールである限り、上の選び方は診断と同じ値になる。違いうるのは、写しを持たないモデルが別の共有部のオブジェクトを持つ場合だけだが、現行コードでは起きない: 配布の反映は、保有モデルを1つの共有部へつなぎ直し（`runtime/global_model_distribution_application.py`の363〜365行付近）、採用した候補も同じ共有部へつながる（`learning/training/adopted_candidate_shared_feature_integration.py`の105行付近）。保有モデルを1つの共有部へつなぐ経路は、配布の反映（上）と、配布の後のつなぎ直し（`learning/training/held_model_shared_feature_reconnection.py`の149〜150行付近）、採用した候補の共有部の選び方（`runtime/adopted_candidate_initial_local_registration.py`の142〜145行付近）と統合（`learning/training/adopted_candidate_shared_feature_integration.py`の102〜107行付近）。adapterを持たない構造でも、共有部は隠れ層を持ち、学習するパラメータを持つ（差し替わるのはモデル固有部の`IdentityFeatureAdapter`だけ）。この構造で診断と一致する理由も、上の共有部のオブジェクトの同一性である。共有部を持たない構造だけは、共有部が隠れ層0の恒等写像で、パラメータを持たない（`learning/models/model_architecture_classifier.py`の113〜120行付近、`learning/models/shared_feature_extractor.py`）ので、どの分類器で共有特徴を計算しても同じ値になる。要求1.5のtestで、3つの構造で、実際の流れで一致することを確かめる。
- 計算量: 写しは同期ごとに1回（保有モデルのパラメータの合計に比例）。予測の計算の回数は変わらない。記憶は、保有モデルのパラメータがもう1組。

### 予測
- `predict_observed_sample_and_update_prediction_weights(..., synchronized_prediction_model_snapshot=None)`: Noneでなければ、上の選び方で、共有特徴の分類器と、モデルごとの分類器を決める。Noneなら今と同じ（ID昇順で最初の保有モデルの分類器で共有特徴を計算し、各モデルの今の分類器で評価）。型の検査（Noneか、exactな`SynchronizedPredictionModelSnapshot`）は、乱数や状態を変える前（入力の検査と同じ時点）に、ownerの検査の表とは別の分岐で行う。
- `_predict_with_current_training_model_only`: 写しのownerを受け取ったら、現在のモデルの分類器と、共有特徴の分類器を、同じ選び方で決める。
- 予測の後の処理（記録、重みと証拠の更新）は、今のまま、予測で得た確率と損失を使う。

### client
写しのownerは、ownerの束（`FedsdaRunClientOwners`）へは入れない。この束は、項目の名前が標本1件の処理の引数名と同じで、`tests/refactoring/test_fedsda_run_client.py`の`collect_processing_arguments`が、束の全部（`vars(owners)`）を標本の処理の引数として読み、旧実装との状態の照合にも使うため（入れると、既存のtestが壊れる）。

- `FedsdaRunClient`の構成子へkeyword引数`synchronized_prediction_model_snapshot`を足し、属性（`_synchronized_prediction_model_snapshot`）として持つ。`FedsdaRunClient(`を呼ぶのは`assemble_fedsda_run_client`だけで、そこで空のownerを作って渡す（どちらの設定でも持つ。`current_local_parameters`では使わない）。testが状態を読むための読取りのproperty（`synchronized_prediction_model_snapshot`）を置く。
- `apply_global_model_distribution`: 今の処理の後、設定が`parameters_at_last_synchronization`なら`capture_held_models`。
- `process_observed_sample`: 標本の処理へ、設定が`parameters_at_last_synchronization`ならowner、そうでなければNoneを渡す。

### 構成
`SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME = "fedsda-synchronized-model-prediction"`。一覧の末尾へ足す（testの並列の組は、一覧の位置で決まるので、既存の組は変わらない）。組み立ては、`_replace_prediction_combination_settings(prediction_model_parameter_source="parameters_at_last_synchronization")`。

### golden
`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`の版1〜5へ、経路`("prediction_combination_settings", "prediction_model_parameter_source")`と値`current_local_parameters`を足す。基準版は増やさない（`BEHAVIOR_BASELINE_VERSION`は5のまま）。

## Validation Design
`tests/refactoring/test_fedsda_synchronized_model_prediction.py`:
1. 設定: 許す値の受理と、不正な値の拒否（task 1.1）。新しい構成のrun設定が、提案構成と、この項目だけ違う（保存の形で比べる。構成名が要るので、task 2.3）。
2. owner: 写しを取った後に、今の分類器のパラメータを変えても、選ばれた分類器の値は写しの値のまま。分類器を差し替えたモデル・写しの後に足したモデルは、今の分類器が選ばれ、共有特徴の分類器は写した分類器。写しが空なら、全部、今の分類器。共有部は、写しの中でも1つのオブジェクト。
3. 診断との一致（要求1.5）: 小さい条件（同期の後に新しいモデルが作られる条件を選ぶ）で、新しい構成のrunと、提案構成に診断の差し替えをtestの中で当てたrunの、予測の記録の全部と主な指標が一致する。写しを持たないモデルを含む予測が少なくとも1回あったことを確かめる（経路を通ったこと）。モデルの構造は、既定（共有部＋adapter）に加えて、共有部を持たない構造（`fedsda-independent-models`と同じ）と、adapterを持たない構造（`fedsda-without-concept-specific-adapter`と同じ）でも、それぞれ1つの条件で同じ照合をする（予測の設定の項目だけを`parameters_at_last_synchronization`へ替えたrun設定と、その構成に診断の差し替えを当てたrunを比べる）。診断の差し替え（oracle）は、gitの対象外の結果のディレクトリのscriptには依らず、testの中で同じ処理を書き直す。testの中で、標本の処理が呼ぶ予測の関数（`runtime/observed_sample_processing.py`の名前）を、`results/.../fedsda_distributed_prediction_probe.py`と同じ処理——配布の後に保有モデル全部のstate_dictを控え、予測の間だけ読み込み、終わったら戻す——で包んだものに替え、新しい経路（写しのowner）は通らない（oracleのrunは`current_local_parameters`の設定で走らせ、引数は`**kwargs`でそのまま渡す）。この2つの構造は新しい構成名を持たないが、設定の組合せとしては作れるので、拒否せずに確かめる。
4. 検出力: 同じ条件の提案構成（`current_local_parameters`）では、予測の記録が少なくとも1標本で違う。
5. 予測の後の処理は現在の値（要求1.4）: 実装では、標本の処理の名前空間の`evaluate_classifier_per_sample_bounded_losses`（損失の監視とADWINの窓の損失の、標本の処理の中の唯一の計算の経路）を包み、渡された分類器が、写しの分類器のどれでもないことを、runの全体で確かめた（写しは予測へだけ渡り、候補の検証・学習ほかの経路は写しを受け取らないので、構造上、現在の値を使う）。以下は、仕様の段階の案: 写しを持つ状態で、写しの後に現在の分類器のパラメータを変えてから、標本の処理（`process_observed_sample`）を1回呼ぶ。損失の監視が受け取る損失は、`OverallAndTrueClassLossMonitor.observe_loss_after_label_observation`を、動きを変えずに引数を記録する関数で包んで読む（標本の処理の戻り値の`LossMonitoringObservation`は、損失を持たないため）。記録した`observed_loss`が、現在の分類器で計算した損失と同じで、写しの分類器での損失と違うことを確かめる。写しの分類器での損失は、戻り値の`observed_sample_prediction.observed_losses_by_model_id`と同じ。
6. Noneの既定の経路: 既存の標本の処理・予測・clientのtestが、引数を渡さないまま通る。省略可能な引数へ、None・正しい型以外を渡すと、状態を変える前に`TypeError`。

既存のtest: goldenの版1〜5、`test_important_behavior_regression_goldens.py`、構成の一覧を使うtest、依存境界、全pytest。WSLでは、新しいtestとgoldenの版1〜5を実行する。

## 旧と違う点
FedSDAのrun設定の保存の形に1つの項目が増えるので、既定の`fedsda`を含むFedSDAのrunの識別子が全部変わる。前の結果の出力先へ、新しいコードで実行すると、前のrunは計画に入らない。比べるときは、別の出力先を使う（実験の入口の文書に1行書く）。基準版の照合（`experiment-run-code-provenance`の、済んだrunの基準版の確認）は、識別子が一致するrunだけを対象にするので、前のrunとは照合されず、影響しない。既定の動作と値は変わらない。
