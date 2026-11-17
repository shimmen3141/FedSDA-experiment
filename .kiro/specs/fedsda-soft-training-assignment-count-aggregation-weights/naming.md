# 命名と役割

「固い割り当て」は製品の既存の名前`single_model_assignment`（学習の割当方式）に合わせる。「学習量」は、学習の割当方式が決める、集約で今使っている量（固い学習は件数、柔らかい学習は責任度の合計）を指す。

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `fedsda-soft-training-assignment-count-aggregation-weights` | 構成 | `fedsda-soft-training`の集約の重みと参加の判定だけを、固い割り当ての標本の件数に替えた構成 | `fedsda-soft-training`は責任度の合計で集約する。`fedsda-training-mass-diagnostics`は学習も固い |
| `SOFT_TRAINING_ASSIGNMENT_COUNT_AGGREGATION_WEIGHTS_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 | 既存の構成名の定数と同じ扱い。一覧の末尾に足す |
| `FedsdaRunClientSettings.model_aggregation_weight_source` | `runtime/fedsda_run_client_settings.py` | 集約で使う、clientのモデルごとの重みの元。`"training_assignment_amounts"`（既定。学習の割当方式による今の量）か`"single_model_assignment_sample_counts"`（固い割り当ての件数） | `training_assignment_strategy`は局所の学習の割当方式で、集約の重みは決めない |
| `FedsdaRunClient.model_aggregation_weight_source` | `runtime/fedsda_run_client.py` | 上の設定の読取りだけのproperty。集約が読む | — |
| `derive_client_model_aggregation_weights` | `runtime/server_model_registration_and_aggregation.py` | 1 clientの学習標本の記録と重みの元から、集約に使うモデルIDごとの重みを返す。状態は読むだけ | `aggregate_client_models_into_global_models`は全clientの平均と反映を行う。重みの決め方だけをこの関数へ分ける |
