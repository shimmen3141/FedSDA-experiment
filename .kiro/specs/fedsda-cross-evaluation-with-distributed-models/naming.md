# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `record_global_model_distribution_volume` | `runtime/global_model_distribution.py`の関数 | 全グローバルモデルを全clientへ配布するときの下りの通信量（共有部1個と全モデルの概念固有部を、client数だけ。モデルの転送回数）を記録する。状態は変えない。`distribute_global_models_to_clients`（記録と反映の両方を行う）は、この関数を呼ぶ形にする |
| `validate_cross_evaluation_model_transfer_policy` | `runtime/server_round_synchronization.py`の関数 | クロス評価のモデルの送り方が、builtin strの2つの値のどちらかであることを検査する（サーバの組立てと、1ラウンドの同期の入口で使う） |
| `SEND_CANDIDATE_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME`、`REUSE_DISTRIBUTED_MODELS_FOR_CROSS_EVALUATION_POLICY_NAME`、`CROSS_EVALUATION_MODEL_TRANSFER_POLICY_NAMES` | 同じmoduleの定数 | 2つの値の名前と、その組（参加者の設定の検査も、この組を使う） |
| 基準版8 | goldenの版 | クロス評価で配布するモデルを使う扱い（新しい既定）の値 |

新しいmodule・classは無い。設定の項目`cross_evaluation_model_transfer_policy`（参加者の設定）と、関数の引数`record_model_transfers`（`cross_evaluate_global_models`）・`record_communication_volume`（`distribute_global_models_to_clients`）は、設計に書く。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `cross_evaluate_global_models` | 評価と、メッセージ・モデルの転送の記録 | 引数で、モデルの転送の記録を省ける（メッセージは常に記録する） |
| `distribute_global_models_to_clients` | 配布の記録と反映 | 引数で、記録を省ける（反映は常に行う） |
| `BASELINE_VERSION`、`BEHAVIOR_BASELINE_VERSION` | 7 | 8 |
