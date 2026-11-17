# 命名と役割

「自分で固く割り当てた」は、そのclientの`ModelTrainingSampleStore`に固い割り当ての標本（`_training_samples_by_model_id`の列）を1件以上持つことを指す。配られてきただけのモデルは、選ばれて固い割り当ての先になるまで、この集合に入らない。

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `fedsda-soft-training-own-assigned-models` | 構成 | `fedsda-soft-training`の責任度の対象だけを、そのclientが自分で固く割り当てた標本を持つモデル（と、いま割り当てる先）に制限した構成 | `fedsda-soft-training`は全保有モデルへ責任度を分ける。`fedsda-soft-training-assignment-count-aggregation-weights`は集約の重みだけを替える |
| `SOFT_TRAINING_OWN_ASSIGNED_MODELS_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 | 既存の構成名の定数と同じ扱い。一覧の末尾に足す |
| `FedsdaRunClientSettings.training_responsibility_model_scope` | `runtime/fedsda_run_client_settings.py` | 柔らかい学習の責任度を分ける対象。`"all_held_models"`（既定）か`"models_with_own_assigned_samples"`（固い割り当ての標本を持つモデルに加え、いま固く割り当てる先のモデルを常に含む） | `training_responsibility_weight_update_policy`は責任度に使う重みの更新方式で、対象のモデルは決めない |
| `ModelTrainingSampleStore.snapshot_model_ids_with_assigned_samples` | `learning/training/model_training_sample_store.py` | 固い割り当ての標本を1件以上持つモデルIDのfrozenset。読むだけ | `snapshot_ordered_model_training_samples`は標本の列そのもの（空列のモデルも含む）を返す |
| `build_released_sample_responsibility_provider` | `runtime/released_sample_training_responsibilities.py` | FIFOから放す時点の責任度を求める関数を作る。任意の引数で、責任度の対象を、自分で固く割り当てた標本を持つモデルといま割り当てる先に絞れるようにする（役割を広げる。既存の名前は残す） | — |
