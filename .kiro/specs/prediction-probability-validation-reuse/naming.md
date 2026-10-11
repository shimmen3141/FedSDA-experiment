# 命名表: prediction-probability-validation-reuse

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `prediction_probabilities_are_validated` | `combine_model_prediction_probabilities`・`compute_model_mean_bounded_losses_after_label_observation`の引数 | 渡すモデル別の確率が、呼出し側で検査済みであることを伝える |
| `prediction_scores_are_validated` | `predict_class_labels_from_prediction_scores`の引数 | 渡す予測のtensorが、呼出し側で検査済みであることを伝える |
