# Implementation Plan

- [x] 1. 責任度の対象の設定と、固い割り当ての標本を持つモデルIDの読取りを足す
  - `FedsdaRunClientSettings`に`training_responsibility_model_scope`（既定`"all_held_models"`、もう1つの値`"models_with_own_assigned_samples"`、既定値を保存しない宣言、`allowed_parameter_values`）を足し、`__post_init__`で型・値と、柔らかい学習以外との組合せの拒否を行う。`ModelTrainingSampleStore`に`snapshot_model_ids_with_assigned_samples`を足す。
  - 完了条件: 新しいtestファイルで、不正な値・型と、固い学習・独立したFixed-Shareの重みとの組合せが`RunSettingsValidationError`になり、柔らかい学習との組合せは通る。メソッドが標本を持つモデルIDだけを返す（空列で登録しただけのモデルは含まない。付け替え・remapの後のIDを返す）。既存の`test_soft_training_single_model_prediction.py::test_existing_configuration_mapping_order_identity_and_protected_files`が通る。
  - _Boundary: runtime（clientの設定）、learning（学習標本の保持）_
  - _Requirements: 1.1, 3.1, 3.2, 4.2_

- [x] 2. 責任度の提供の関数に対象の制限を足し、clientから渡す
  - `build_released_sample_responsibility_provider`に`own_assignment_sample_store`を足し、重みと確率を全保有モデルで求めた後で対象に絞る。`fedsda_run_client.py`の2か所で、設定が制限のときだけstoreを渡す。依存の境界の許可集合に、既存の形に合わせてmodule名と型名の両方を登録する。
  - 完了条件: 新しいtestファイルの単体testで、(a) 対象外のモデルが責任度の辞書に入らない、(b) 対象の中の値が、全保有モデルで求めた値を対象の中で正規化し直した値と一致、(c) Fixed-Shareの重みの状態が制限なしと同じ（全保有モデルで同期）、(d) 初めて割り当てる先（`fallback_model_id`）が含まれる、(e) 全保有モデルが対象なら制限なしと`==`で同じ、(f) storeに不正な型を渡すと`TypeError`、が通る。既存の`test_fedsda_soft_training_configuration.py`と依存の境界のtestが通る。
  - _Boundary: runtime（責任度の提供、client）_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. 構成を登録し、掃引と文書へつなぐ
  - 構成名の定数を一覧の末尾へ足し、`fedsda-soft-training`の組立ての分岐で`training_responsibility_model_scope="models_with_own_assigned_samples"`だけを替える。構成の名前を固定で照合している既存のtest（`test_soft_training_single_model_prediction.py`の一覧の末尾、`test_fedsda_soft_training_assignment_count_aggregation_weights.py`の末尾2つの期待、`test_fedsda_configuration_variants.py`の列挙と変種の集合の差）と`behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ足す。`docs/experiments/experiment-sweep.md`の構成の表に、目的と`fedsda-soft-training`との違いを書く。
  - 完了条件: 新しいtestファイルで、新しい構成の設定が`fedsda-soft-training`と`training_responsibility_model_scope`だけ違い、保存の記録にその値が出る。計画で新しい構成のrunが増え、既存の構成の識別子が変わらない。小さい実掃引（`slow`）で保存・一覧のCSV・再開・競合の拒否が通る。上の既存のtest、構成の一覧の先頭や固定の範囲を照合している既存のtest（`test_fedsda_soft_training_configuration.py`、`test_feddrift_fedccfa_configuration.py`、`test_feddrift_ifca_window_configuration.py`、`test_feddrift_window_and_adaptive_fedavg_configuration.py`、`test_dynamic_weighted_majority_experiment_sweep.py`）、`test_important_behavior_regression_goldens.py`が通る。
  - _Boundary: sweep（構成の登録）、testの構成一覧、文書_
  - _Depends: 2_
  - _Requirements: 2.1, 3.1, 4.1, 4.2, 4.3_

- [x] 4. runの挙動を確かめ、fresh processへ足す
  - 2 client・sine2とmnist4の小さいrun（`slow`）で、提供の関数を包んで確かめるtestを書く。`fresh_process_smoke.py`に新しい構成の小さいrunを足す。
  - 完了条件: (a) 新しい構成の各呼出しで、返した責任度の辞書のキーが「そのclientの固い割り当ての標本を持つモデル」と`fallback_model_id`に含まれる、(b) 対象外の保有モデルが実際に生じる呼出しがある（制限が効いている）、(c) 各clientで、対象外の保有モデルが初めて生じる呼出しより前の責任度の列が、同じ条件の`fedsda-soft-training`と一致する、(c2) 各呼出しの対象が、そのclientがそれまでに`fallback_model_id`として渡したIDを付け替え・統合で移した集合（と今回の`fallback_model_id`）と一致する、(d) 記録の表に学習量のコンセプト別の記録がある、が通る。fresh process smokeが通る。
  - _Boundary: tests（run）、fresh process smoke_
  - _Depends: 3_
  - _Requirements: 1.1, 1.4, 1.5, 2.1, 2.2, 3.3_

- [ ] 5. 全体の検証
  - naming.mdを実装に合わせ、`spec_checks.py names`を通す。独立CLIの実装レビューの指摘を反映した後、全pytest（goldenを含む）・Ruff・pyrightを通す。
  - 完了条件: 全pytestが通り（件数を前の件数＋追加のtest数と照合）、既存のgoldenと識別子が変わらない。件数とcommitをImplementation Notesへ記録する。
  - _Depends: 4_

## Implementation Notes

### 仕様レビュー
- GPT-6 Lunaは使用量の上限（2026-11-22 23:07まで回復しない、とCLIが表示）で使えず、Claude Haiku 5.5（effort medium、読むだけ、独立CLI）へ代えた。
- 1回目（Haiku 5.5、session `7de891e7-bdf5-44be-aa93-d33a5b46e6d2`、JSONの`is_error=false`・`modelUsage`は`claude-haiku-5-5`、対象 `d9d556bd`）: PASS（Blocker・Majorなし）。
  - Minor 1（対象外のモデルが学習・集約から外れ、run全体のMACと上りの通信量が変わる）: 採用。designに書いた。
  - Minor 2（絞る向き）: 採用。保有モデルの辞書の側から残す向きと明記した。
  - Minor 3（「持つ」が「過去に割り当てた」と同じであることの前提とtest）: 採用。designに前提を書き、task 4の完了条件(c2)を足した。
  - Minor 4（乱数の順が同じなのは全保有モデルが対象の間だけ）: 採用。designに限定を書いた。
  - Minor 5（naming.mdの引数の行、役割の広がり、値の名前の注記）: 採用。引数の行を消し、関数の役割を広げたことと、値の注記を書いた。
  - Minor 6（依存の許可集合はmodule名と型名の両方）: 採用。design・tasksを直した。
  - Minor 7（spec.jsonの仕様レビューの記録、作成の時刻）: 採用。記録を足し、作成の時刻を実際の時刻（UTC）に直した。
  - 任意 8（独立したFixed-Shareの重みとの組合せ）: 採用。拒否にし、要求3.2・design・task 1の完了条件に足した（要求の小さい追加。Minorとして扱い、再レビューは求めない）。

