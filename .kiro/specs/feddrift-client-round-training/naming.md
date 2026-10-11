# Naming
公開source名だけを対象とし、field/引数/private/test名は含めない。最終独立レビューで確認する。

| 名前 | 役割・似た概念との違い |
| --- | --- |
| learning/training/independent_model_parameter_update.py | 一独立モデルの平均損失によるparameter更新。複数モデルの共同損失とは別 |
| validate_independent_model_parameter_update_inputs | forward/zero_grad前の学習契約検査。損失判定/更新はしない |
| update_independent_model_parameters | 借用した一モデルとoptimizerを一回更新し更新前lossを返す |
| runtime/feddrift_client_round_training.py | FedDriftの一roundのモデル順/抽出/計数を接続する |
| train_feddrift_held_models_for_round | store順で各モデルを予算回連続更新する。観測時のpending要求は扱わない |
