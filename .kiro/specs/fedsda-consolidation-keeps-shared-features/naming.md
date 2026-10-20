# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `AVERAGE_SHARED_FEATURES_WITH_CONCEPT_SPECIFIC_PARTS_POLICY_NAME`、`KEEP_IDENTICAL_SHARED_FEATURES_POLICY_NAME`、`CONSOLIDATED_SHARED_FEATURE_POLICY_NAMES` | `runtime/model_clustering_and_consolidation.py`の定数 | 統合での共有部の扱いの2つの値の名前と、その組 |
| `validate_consolidated_shared_feature_policy` | 同じmoduleの関数 | 統合での共有部の扱いが、builtin strの2つの値のどちらかであることを検査する |
| `parameters_have_the_same_bits` | `runtime/server_model_registration_and_aggregation.py`の関数 | 2つのパラメータの辞書が、同じ名前で、各tensorのdtype・形・byte列が同じか（値の比較と違い、+0.0と−0.0を区別し、同じビットのNaNを同じとする）。v8の`server_round_synchronization._tensors_have_the_same_bits`を、統合と共有して使えるよう移す |
| 基準版9 | goldenの版 | 統合で同じ共有部を保つ扱い（新しい既定）の値 |

設定の項目`consolidated_shared_feature_policy`と、関数の引数は、設計に書く。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `cluster_and_consolidate_global_models` | 共有部も加重平均する | 引数で、同じ共有部を保てる |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 8 | 9 |
