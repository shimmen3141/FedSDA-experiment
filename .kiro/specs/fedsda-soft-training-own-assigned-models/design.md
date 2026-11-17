# Design Document

## Overview
**Purpose**: 柔らかい学習の構成`fedsda-soft-training`から、FIFOから放す標本の責任度を求める対象だけを、そのclientが自分で固く割り当てた標本を持つモデル（と、いま固く割り当てる先のモデル）に制限した構成`fedsda-soft-training-own-assigned-models`を足す。
**Users**: 固い学習が連合学習でだけ効く理由を切り分ける研究者。
**Impact**: 既存の構成の挙動・設定の記録・runの識別子は変えない。新しい設定のfieldは既定値を保存しない宣言にする。

### Goals
- 配られただけのモデル（そのclientの固い割り当ての標本を持たない保有モデル）を、柔らかい学習の責任度の対象から外す。それ以外は`fedsda-soft-training`と同じ。
- 既存の掃引コマンドで実行・保存・一覧・再開でき、既存の固い学習・柔らかい学習の結果と対にできる。

### Non-Goals
- 予測・検出・選択・採用・集約・責任度の式の変更。固い学習の構成の変更。
- uraniumでの本規模の実行と集計の研究ツール（spec外）。

## Boundary Commitments

### This Spec Owns
- clientの設定の新しいfield（責任度の対象）と、その検査。
- 固い割り当ての標本を持つモデルIDを返す、`ModelTrainingSampleStore`の読取りだけの公開メソッド。
- 責任度の提供の関数（`build_released_sample_responsibility_provider`）の、対象を制限する任意の引数。
- client（`FedsdaRunClient`の組立て）からの接続、構成名の定数、構成の一覧への登録、構成から設定への組立て、構成の一覧の文書の1行。

### Out of Boundary
- `calculate_training_responsibilities`の式、`ResponsibilityWeightedTrainingStore`の保持と抽出（責任度が0のモデルは、既存の規則どおり参加しない）。
- 集約（重みは今どおり責任度の合計。対象外のモデルは学習量が0になり、既存の規則どおり集約に参加しない）。

### Allowed Dependencies
- `runtime/released_sample_training_responsibilities.py`は、新しい引数の型として`learning.training.model_training_sample_store.ModelTrainingSampleStore`をimportする（runtimeからlearningへの向き。依存の境界の許可集合へ登録する）。

### Revalidation Triggers
- 柔らかい学習で固い割り当ての標本の保持の規則が変わったとき。
- 責任度の提供の関数の呼出し方（`fallback_model_id`の意味）が変わったとき。

## Architecture

### 今の責任度の求め方（変えない部分の確認）
`ModelTrainingSampleStore.append_model_training_samples(model_id=m, ...)`は、柔らかい学習のとき、各標本について`training_responsibility_provider(m, sample)`を呼んで責任度を求め、責任度の記録に足してから、固い割り当ての標本の列`_training_samples_by_model_id[m]`へ足す。`m`は、その標本をいま固く割り当てる先のモデル（固い学習で学習するモデル）。提供の関数は、全保有モデルの分類器を読み、`weight_controller.get_weights_before_label_observation(model_ids=全保有モデル)`で重みを得て（集合が変わっていれば、予測と同じ規則で同期する）、全保有モデルの真のラベルの確率を求め、`calculate_training_responsibilities(prediction_weights, true_class_probabilities, fallback_model_id=m)`を返す。

### 変更
1. `ModelTrainingSampleStore`に`snapshot_model_ids_with_assigned_samples() -> frozenset[int]`を足す。固い割り当ての標本を1件以上持つモデルIDの集合を返す（標本の列はcopyしない。読むだけ）。
2. `build_released_sample_responsibility_provider`に、キーワード引数`own_assignment_sample_store: ModelTrainingSampleStore | None = None`を足す。
   - `None`（既定）なら今と同じ。
   - 与えられたら、重みと確率は今どおり全保有モデルで求め（重みの同期・forwardの順と回数を変えない）、`calculate_training_responsibilities`へ渡す前に、両方の辞書を「`own_assignment_sample_store.snapshot_model_ids_with_assigned_samples()`に含まれるモデル」と`fallback_model_id`だけに絞る。
   - 絞った後の集合は`fallback_model_id`を含むので空にならない。`calculate_training_responsibilities`の検査（重みと確率の集合が同じ、`fallback_model_id`が含まれる）はそのまま通る。正規化は絞った集合の中で行われる。
   - 標本は提供の関数を呼んだ後で固い割り当ての列へ足されるので、初めて固く割り当てられるモデルは`fallback_model_id`として含まれる（要求1.5）。
3. `FedsdaRunClientSettings`に`training_responsibility_model_scope: str`を足す。値は`"all_held_models"`（既定）と`"models_with_own_assigned_samples"`。metadataは`omit_default_from_saved_settings=True`と`allowed_parameter_values`。`__post_init__`で型・値と、`"models_with_own_assigned_samples"`で`training_assignment_strategy != "responsibility_weighted_training"`の拒否（設定の組立て時なので、runの準備・乱数の消費より前）。
4. client（`fedsda_run_client.py`の、柔らかい学習の提供の関数を組み立てる2か所）で、設定が`"models_with_own_assigned_samples"`なら`own_assignment_sample_store=owners.training_sample_store`を渡す。既定なら引数を渡さない（今の呼出しと同じ）。
5. `experiment_configurations`に`SOFT_TRAINING_OWN_ASSIGNED_MODELS_CONFIGURATION_NAME = "fedsda-soft-training-own-assigned-models"`を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾に置く。組立ては`fedsda-soft-training`の分岐に含め、`training_responsibility_model_scope="models_with_own_assigned_samples"`だけを追加で替える。
6. 文書`docs/experiments/experiment-sweep.md`の構成の表に1行を足す。

### 乱数・計算量・状態の順
- 提供の関数は乱数を使わない。絞る処理は辞書の選択だけで、forwardの回数・重みの同期は変えないので、計算量の計数（MAC）は`fedsda-soft-training`の同じ状態と同じ。
- 対象外のモデルは責任度の辞書に入らないので、学習量の記録・抽出の参加（正の責任度の標本数がbatch以上）に現れず、学習されない。予測・選択・採用は保有モデル全部を今どおり使う。

### 要求と処理の対応
- 要求1.1〜1.3、1.5: 上の2。単体testで、(a) 対象外のモデルが辞書に入らない、(b) 対象の中で既存の式と同じ値（全保有モデルで求めた値を、対象の中で正規化し直した値と一致）、(c) 重みの同期が全保有モデルで行われる（Fixed-Shareの重みの状態が既定と同じ）、(d) 初めて割り当てる先が含まれる、を確かめる。
- 要求1.4・2.2: 全保有モデルが対象なら絞っても集合は変わらず、同じ値になる。単体testと、runのtest（2 client、各release時点で全保有モデルが対象である間、責任度の列が`fedsda-soft-training`と一致）で確かめる。
- 要求2.1・3.1〜3.3: 組立てが1 fieldだけ違うtest、保存の記録のtest、記録の表のtest。
- 要求4.1〜4.3: 計画・実行・一覧・再開のtest、既存の構成の固定のtest（fixtureの照合に名前を足す）、文書の行。

## Components and Interfaces

| 部品 | 変更 | 公開するもの |
|---|---|---|
| `learning/training/model_training_sample_store.py` | メソッド追加 | `ModelTrainingSampleStore.snapshot_model_ids_with_assigned_samples` |
| `runtime/released_sample_training_responsibilities.py` | 任意の引数を追加 | `build_released_sample_responsibility_provider`（引数`own_assignment_sample_store`） |
| `runtime/fedsda_run_client_settings.py` | field追加と検査 | `FedsdaRunClientSettings.training_responsibility_model_scope` |
| `runtime/fedsda_run_client.py` | 提供の関数の組立てで引数を渡す | — |
| `sweep/experiment_configurations.py` | 定数・一覧・組立て | `SOFT_TRAINING_OWN_ASSIGNED_MODELS_CONFIGURATION_NAME` |
| `docs/experiments/experiment-sweep.md` | 構成の表に1行 | — |

## File Structure Plan
- 変更（src）: 上の5ファイル。
- 新規（test）: `tests/refactoring/test_fedsda_soft_training_own_assigned_models.py`（設定・メソッド・提供の関数・run・掃引）。
- 変更（test）: `test_soft_training_single_model_prediction.py`の一覧の末尾、`test_fedsda_configuration_variants.py`の列挙と変種の集合の差、`behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、`test_fedsda_soft_training_assignment_count_aggregation_weights.py`の「一覧の末尾2つ」の期待、`test_single_run_dependency_boundaries.py`の許可集合（提供の関数のmoduleに1件）、`fresh_process_smoke.py`に新しい構成の小さいrun。

## Error Handling
- 不正な`training_responsibility_model_scope`、柔らかい学習でない構成での制限の指定は、設定の組立て時に`RunSettingsValidationError`。
- `own_assignment_sample_store`に`ModelTrainingSampleStore`以外が渡されたら、提供の関数を作る時点で`TypeError`。

## Testing Strategy
- 設定: 新しい構成の設定が`fedsda-soft-training`と`training_responsibility_model_scope`だけ違う。保存の記録にその値が出て、既定の構成には出ない。不正な値・固い学習との組合せの拒否。
- メソッド: 標本を持つモデルIDだけを返す（空列で登録しただけのモデル、付け替え・remapの後の扱いを含む）。
- 提供の関数: 上の(a)〜(d)と、全保有モデルが対象のとき既定と`==`で同じ値。
- run（2 client、sine2とmnist4、小さい標本数、`slow`）: 新しい構成のrunで、責任度の記録（学習量の記録）に、そのclientの固い割り当ての標本を持たないモデルの量が現れない（提供の関数を包み、各呼出しで返した辞書のキーが対象に含まれることを確かめる）。対象外の保有モデルが実際に生じる（制限が効いている）。各clientで、対象外の保有モデルが初めて生じるまでの責任度の列が`fedsda-soft-training`と一致する。記録の表に学習量のコンセプト別の記録がある。
- 掃引: 計画で新しい構成のrunが増え、既存の構成の識別子が変わらない。小さい実掃引（`slow`）で保存・一覧のCSV・再開・競合の拒否。
- 既存: 既存の構成の設定・hash・識別子・goldenの固定（既存のtest）。全pytest。
