# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `fedsda-soft-training` | 構成 | `fedsda`の学習の割り当てだけを、責任度による全モデルの柔らかい学習に替えた構成 | `fedsda-single-model-prediction`は予測を1モデルにする（学習は固いまま） |
| `fedsda-training-mass-diagnostics` | 構成 | `fedsda`に学習の量のコンセプト別の記録だけを足した構成（挙動は`fedsda`と同じ）。純度の比べる相手 | 学習の割り当ては固いまま |
| 責任度 $q_{i,m}$ | 共有の学習ストアの標本ごとの値 | 標本を放す時点の、重み×真のラベルの予測確率の正規化 | Fixed-Shareの重み $w_m$（clientのモデルごとの重み）とは別の量 |
| 純度 | 診断の指標 | 各モデルの学習の量のうち最多のコンセプトの割合（clientで集計） | FedEMの`max_responsibility_model_id`の一致とは別の定義 |

実装担当が決めた設定の項目・表・関数は、実装の後にこの表へ追記する。

## 実装で追加した名前

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `SOFT_TRAINING_CONFIGURATION_NAME` / `TRAINING_MASS_DIAGNOSTICS_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 追加する2構成の正式名の定数 | 既存の構成の名前・順は維持し末尾へ追加 |
| `training_assignment_strategy` | `FedsdaRunClientSettings` | 学習での標本の割当方式。`single_model_assignment`（既定）と`responsibility_weighted_training` | 監視とMDLの帰属先の選択は変更しない |
| `record_training_mass_by_concept` | 同上 | 真の概念別の学習量の保存を有効にするbool。既定False | 学習と乱数を変更しない。両項目の既定値は保存から省く |
| `responsibility_weighted_training` | `learning/training/` | 責任度の式、共有ストア、抽出、純度の導出 | Fixed-Shareの予測重みの管理ではない |
| `calculate_training_responsibilities` | 同上 | 解放時の重み×真のラベルの確率を正規化。積のunderflowを回避、分母0は選択先にone-hot | 観測時の予測記録は参照しない |
| `ResponsibilityWeightedTrainingStore` | 同上 | 上限のない共有標本と固定した責任度・概念別の量の所有者 | 固いストアは監視と統合の判断用に維持し、柔らかい学習と集約には使わない |
| `sample_count` / `training_masses_by_model_id` / `support_counts_by_model_id` / `mass_by_model_and_concept` | 同classのproperty | 標本数、責任度の総量、正の責任度の標本数、概念別の量のコピー | 学習の参加条件はsupport、サーバの集約はmassを使う |
| `append` / `remap` / `discard` / `sample_batches` | 同class | 解放時の追加、一回対応の列の加算、列の削除、重み付き復元抽出 | 列の変更後に責任度を正規化しない |
| `derive_training_mass_purity` | 同module | clientの純度と全部を1つのモデルへ入れた基準。分母0はNone | 責任度の実効モデル数とは別の指標 |
| `derive_run_training_mass_purity` | 同module | 有効なclientの等しい重みの平均と欠損client数。全欠損はrunもNone | clientの学習量で重み付けしない |
| `released_sample_training_responsibilities` / `build_released_sample_responsibility_provider` | `runtime/` | 解放時の局所分類器とFixed-Shareの現在のownerを読む接続を作る | storeへ実体を組み立てて渡し、learning層はruntimeをimportしない |
| `training_mass_by_concept` | 保存する表 | runの最後の`client_id`, `model_id`, `observed_concept_id`, `training_mass` | 真の概念が無い標本は除外。診断構成が増やす唯一の表 |
| `training_responsibility` | 保存する表 | 追加する標本ごとの`client_id`, `effective_model_count` | 追加後にモデルが統合されても追加時の値を保持 |
| `soft_training_update` | 保存する表 | 共同更新ごとの`client_id`, `participating_model_count` | 候補学習は含めない |
| `soft_training_batch` | 保存する表 | モデルのbatchごとの`client_id`, `duplicate_sample_fraction` | `1 - 異なる共有ストアの標本位置数 / B`。同じpayloadでも別標本位置は区別 |
