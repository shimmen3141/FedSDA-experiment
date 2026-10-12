# 命名と役割

公開名の追加・変更は無い。

| 既存公開名 | 役割 | 類似名との違い |
|---|---|---|
| runtime.model_cross_evaluation | サーバのモデル対ごとの評価の進行と通信量 | client_model_cross_evaluationはclient1件の評価 |
| cross_evaluate_global_models | 1回の全モデル対の評価と転送の計数 | クラスタリングの統合判断は含まない |

実装の公開名は上記のまま。testのhelper・局所名は表の対象外。
