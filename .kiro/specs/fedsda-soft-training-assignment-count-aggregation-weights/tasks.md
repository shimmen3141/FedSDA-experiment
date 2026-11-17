# Implementation Plan

- [ ] 1. 集約の重みの元を設定とclientに足す
  - `FedsdaRunClientSettings`に`model_aggregation_weight_source`（既定`"training_assignment_amounts"`、もう1つの値`"single_model_assignment_sample_counts"`、既定値を保存しない宣言、`allowed_parameter_values`）を足し、`__post_init__`で型・値と、件数の指定が柔らかい学習以外と組み合わされたときの拒否を行う。`FedsdaRunClient`に読取りだけのpropertyを足す。
  - 完了条件: 新しいtestファイルで、不正な値・型と、固い学習との組合せが`RunSettingsValidationError`になり、柔らかい学習との組合せは通る。既存の`test_soft_training_single_model_prediction.py::test_existing_configuration_mapping_order_identity_and_protected_files`（既存の全構成の設定の写像・hash・識別子が変わらない）が通る。
  - _Boundary: runtime（clientの設定とclient）_
  - _Requirements: 3.1, 3.2, 4.2_

- [ ] 2. 集約の重みを決める関数を足し、集約から呼ぶ
  - `derive_client_model_aggregation_weights`を足し、`aggregate_client_models_into_global_models`の重みを決める数行をこの関数の呼出しに置き換える（参加の判定・平均・反映は変えない）。
  - 完了条件: 新しいtestファイルの単体testで、(a) 2モデルへ分かれた責任度の記録では既定が学習量・件数の指定が件数、(b) 1モデルのときの実際の`calculate_training_responsibilities`の責任度で作った記録で、両者の値が`==`で等しい（近似の比較を使わない）、(c) 責任度の提供が無い記録ではどちらも件数、(d) 不正な値は状態を読む前に`ValueError`、が通る。既存の集約のtest（`test_server_model_registration_and_aggregation`系、`test_fedsda_soft_training_configuration.py`、`test_global_model_distribution.py`、`test_model_clustering_and_consolidation.py`、`test_model_cross_evaluation.py`）が通る。
  - _Boundary: runtime（サーバの集約）_
  - _Depends: 1_
  - _Requirements: 1.3, 2.1, 2.5_

- [ ] 3. 構成を登録し、掃引と文書へつなぐ
  - 構成名の定数を一覧の末尾へ足し、`fedsda-soft-training`の組立ての分岐で`model_aggregation_weight_source="single_model_assignment_sample_counts"`だけを替える。構成の名前を固定で照合している既存のtest（`test_soft_training_single_model_prediction.py`の一覧の末尾、`test_dynamic_weighted_majority_experiment_sweep.py`の`dwm-neural-network`の位置、`test_fedsda_configuration_variants.py`の列挙と変種の集合の差）と`behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、design「File Structure Plan」のとおり足す。`docs/experiments/experiment-sweep.md`の構成の表に、目的と`fedsda-soft-training`との違いを書く。
  - 完了条件: 新しいtestファイルで、新しい構成の設定が`fedsda-soft-training`と`model_aggregation_weight_source`だけ違い、保存の記録にその値が出る。計画で新しい構成のrunが増え、既存の構成の識別子が変わらない。小さい実掃引（`slow`）で保存・一覧のCSV・再開（完了済みの飛ばし）・競合の拒否が通る。上の既存のtestと、構成の一覧の先頭や固定の範囲を照合している既存のtest（`test_fedsda_soft_training_configuration.py`、`test_feddrift_fedccfa_configuration.py`、`test_feddrift_ifca_window_configuration.py`、`test_feddrift_window_and_adaptive_fedavg_configuration.py`）、`test_important_behavior_regression_goldens.py`が通る。
  - _Boundary: sweep（構成の登録）、test の構成一覧、文書_
  - _Depends: 2_
  - _Requirements: 1.1, 3.1, 4.1, 4.2, 4.3_

- [ ] 4. runの挙動を確かめ、fresh processへ足す
  - 2 client・sine2とmnist4の小さいrun（`slow`）で、集約の呼出しを包んで期待と比べるtestを書く。`fresh_process_smoke.py`に新しい構成の小さいrunを足す。
  - 完了条件: (a) 集約の各呼出しで、重み付きの和に渡る重みの列が各clientの件数から作った期待（参加は保有かつ件数>0、共有部は合計）と一致、(b) 上りのモデル数が期待と一致、(c) どのclientも件数を持たない保有モデルのパラメータが集約で変わらない、(d) 少なくとも1回の集約で件数と学習量が異なる、(e) 最初の同期より前の標本の予測の記録が同じ条件の`fedsda-soft-training`と一致、(f) 記録の表に学習量のコンセプト別の記録がある、が通る。fresh process smokeが通る。
  - _Boundary: tests（run）、fresh process smoke_
  - _Depends: 3_
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.3_

- [ ] 5. 全体の検証
  - naming.mdを実装に合わせ、`spec_checks.py names`を通す。独立CLIの実装レビューの指摘を反映した後、全pytest（goldenを含む）・Ruff・pyrightを通す。
  - 完了条件: 全pytestが通り（件数を前の件数＋追加のtest数と照合）、既存のgoldenと識別子が変わらない。件数とcommitをImplementation Notesへ記録する。
  - _Depends: 4_

## Implementation Notes

### 仕様レビュー
- 1回目（GPT-6 Luna、effort medium、独立CLI、session `01a1ea75-bf01-7ae0-9035-3b191c335c97`、対象 `f63f0a82`）: FAIL。
  - Major 1（要求1.3の一致が浮動小数点の丸めで崩れうる）: 採用。1モデルの責任度はちょうど1.0で、学習量は1.0の和なので件数と厳密に等しいことをdesignに書き、要求を「厳密に一致」、task 2の完了条件を実際の責任度の関数で作った記録の`==`の比較にした。
  - Minor 2（構成の一覧の先頭を照合する`test_fedsda_soft_training_configuration.py`がtaskに無い）: 採用。task 3の完了条件に足した。
  - 任意 3（FedDrift系のtestの固定の範囲）: 採用。task 3の完了条件に足した。
- 2回目（同じLuna、session `01a1ea77-5056-7842-b32d-3ba1a232a519`、対象 `f63f0a82..3fa890d2`）: PASS（指摘なし）。
