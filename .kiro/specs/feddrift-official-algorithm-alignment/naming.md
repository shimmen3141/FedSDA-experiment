# Naming

新しい設定の項目と値の名前（案）。実装担当は、既存の命名（`FeddriftRunClientSettings`・`FeddriftServerSettings`の項目名、値の名前の定数`*_POLICY_NAME`）に合わせて調整してよく、
調整したら、この表を実装と同じに直す。

| 項目（案） | 置き場所（案） | 今の値（基準版1〜13） | 新しい値（基準版14の既定） |
|---|---|---|---|
| `model_comparison_criterion` | client・サーバの両方（同じ値） | `bounded_loss` | `classification_error_rate` |
| `new_model_parameter_source` | client | `current_training_model` | `minimum_criterion_held_model` |
| `detection_reference_after_drift` | client | `new_model_batch_value` | `minimum_held_model_batch_value` |
| `training_model_eligibility` | サーバ（clientへ伝える） | `held_models_with_training_samples` | `models_assigned_in_current_timestep` |
| `consolidation_weight_timing` | サーバ | `weights_before_current_timestep` | `weights_including_current_timestep_assignments` |
| `maximum_cross_evaluation_client_count`（既存） | サーバ | 3 | clientの数 |
| `loss_increase_threshold`・`maximum_same_cluster_loss_increase`（既存。値の意味は基準の単位） | client・サーバ | 0.1 | データ別の既定（SEA 0.04、SINE 0.20、CIRCLE 0.10、MNIST 0.10） |

文書での呼び方: 「基準を誤り率にする」「最小のモデルから複写する」「最小の値を基準値にする」「今の時刻に割り当てたモデルだけを学習する」「今の時刻の割当を含む統合の重み」。
