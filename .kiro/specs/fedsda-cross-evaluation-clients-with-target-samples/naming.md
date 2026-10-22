# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `CLIENTS_HOLDING_TARGET_MODEL_POLICY_NAME`、`CLIENTS_WITH_TARGET_MODEL_SAMPLES_POLICY_NAME`、`CROSS_EVALUATION_CLIENT_SELECTION_POLICY_NAMES` | `runtime/model_cross_evaluation.py`の定数 | クロス評価で評価させるclientの選び方の2つの値（「対象のモデルを保有するclient」——今の扱い——と「対象のモデルの標本を返せるclient」）の名前と、その組 |
| `validate_cross_evaluation_client_selection_policy` | 同じmoduleの関数 | 選び方が、builtin strの2つの値のどちらかであることを検査する |
| `target_model_cross_evaluation_samples_are_available` | `runtime/client_model_cross_evaluation.py`の関数 | clientが、対象のモデルのクロス評価の依頼に、標本を返せるか（評価が標本を選ぶ規則と同じ定義から作る真偽）。状態を変えず、乱数を使わない |
| `FedsdaRunClient.get_model_ids_with_cross_evaluation_samples` | `runtime/fedsda_run_client.py`のメソッド | このclientが、クロス評価の依頼に標本を返して評価できるモデルのID（保有モデルのIDのうち、上の関数が真のもの） |
| 基準版12 | goldenの版 | クロス評価で対象の標本を返せるclientだけを評価させる扱い（新しい既定）の値 |

設定の項目`cross_evaluation_client_selection_policy`、関数の引数、経路の記録の項目`zero_sample_cross_evaluation_count`は、設計に書く。

似た名前との違い:
- `get_cross_evaluation_held_model_ids`（既存）: 評価標本・学習データ・保有モデル・現在の学習帰属の、どれかにあるID（「保有する」）。標本を返せないIDも含む。
- `get_model_ids_with_cross_evaluation_samples`（新）: 保有モデルのうち、標本を返せるIDだけ（評価を必ず行える）。
- `CLIENTS_WITH_TARGET_MODEL_SAMPLES_POLICY_NAME`の「samples」は、評価に使う標本（評価標本か、現在の学習帰属の学習データ）を指す。研究ツールの変種名の「sample-holding」と同じ扱い。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `cross_evaluate_global_models` | 対象を保有するclientから選ぶ | 引数で、対象の標本を返せるclientから選べる |
| `synchronize_models_in_server_round` | — | 引数で、クロス評価のclientの選び方を受け取る |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 11 | 12 |
