# 統合検証: 重み付き分類予測

## 対象と判定

対象はCPU float32のモデル別出力確率化、重み正規化、混合、クラス判定、観測後モデル別平均損失。モデルforward・学習・ClassESR・候補・FIFO・server同期・新FedSDA全体runは後続範囲。
現在はtask 3とfeature統合の最終レビュー中。最終承認はspec.jsonを参照する。

## 要件の証拠

以下の検証名はtests/refactoring/test_class_probability_calculations.pyのtest_class_probability_接頭辞を省略している。

| 条件 | 実装・実行証拠 |
|---|---|
| 1.1, 1.2, 1.3, 1.4 | model_outputs_become_probabilities_before_combination、invalid_inputs_are_rejected_without_mutation、functions_require_explicit_keyword_arguments。二値コピー・モデル別softmax・入力契約・ラベルなしAPI |
| 2.1, 2.2, 2.3, 2.4 | normalization_matches_reference_without_input_mutation、combination_matches_reference_in_sorted_model_order、for_prediction。昇順通常sum、単一/ゼロ重み、負ID、ID対応拒否。浮動小数点の中間値に近い重みで、混合中の再正規化が値を変えるケースも照合 |
| 3.1, 3.2, 3.3, 3.4 | class_predictions_preserve_threshold_and_tie_rules、after_mixture_rounding、for_prediction。閾値隣接・同率・返却形・有限スコア、ラベル観測後の再予測一致 |
| 4.1, 4.2, 4.3, 4.4 | observed_model_losses_match_reference、for_labels、fixed_share_prediction_and_observation_match_reference。旧float32平均への完全一致、不正ラベル時の入力と外部状態不変 |
| 5.1, 5.2, 5.3, 5.4 | outputs_are_independent_and_have_no_gradient_history、numeric_calls_preserve_caller_random_states、fixed_share_prediction_and_observation_match_reference、AST全走査・注入検査・独立smoke。30標本×2/3/10クラス、モデル集合3→2→1→10の各更新後に全重み状態を照合。完成範囲を数値部品へ限定 |

旧oracleは固定した旧clientのstatic 4関数とSwitchingExpertRouter。新productionへの旧import・controller import・設定読み込みはない。
public APIだけで予測前重み取得→正規化→モデル別確率化→混合/予測→ラベル観測→損失→同じ正規化済みsnapshotで更新を接続した。
勾配履歴・入力storage・入力grad、Python/NumPy/torchのグローバル乱数の不変も確認する。

## 検証コマンド

worktreeルートから共有Windows golden環境（Python 3.13.15、torch 2.12.1+cpu、NumPy 2.4.6）を使用。

```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
../../venv/Scripts/python.exe -m pytest tests/refactoring -q -p no:cacheprovider
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/weighted-prediction-final-20261004a
```

basetempは今回未使用の場所を指定した。再実行時は別の未使用名にする。Windows一時ディレクトリの権限制限を避けて全testsを実行した。
対象統合検証は1500 passed / 3.67s、全refactoringは1996 passed / 5.54s、exit 0。Lunaの独立再検証は1996 passed / 5.98s、exit 0。
全testsは2281 passed / 3 skipped / 116.75s、exit 0。既存11ケースと最終提案3ケースのgoldenは全成功、値・許容誤差は更新していない。スキップはWindowsで実行しない既存の以下3ケースだけである。

- test_server_sweep_wrapper_resolves_runtime_options
- test_server_sweep_wrapper_rejects_owned_options
- test_main_ablation_suite_lists_individual_variants

## 独立プロセスsmoke

pytest fixture・旧moduleなしに5つのpublic数値APIと予測重みcontrollerをimportし、二値/3クラスで各3標本の予測と更新を実行した。返却形と更新後総和をassertし、sys.modulesに旧packageがないことをassertした。
出力: PASS: binary/multiclass public numerical API + weight update; no legacy imports。exit 0。

```powershell
$env:PYTHONPATH='src'
@'
import sys
import torch
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    convert_model_outputs_to_prediction_probabilities,
    normalize_model_prediction_weights,
    combine_model_prediction_probabilities,
    predict_class_labels_from_prediction_scores,
    compute_model_mean_bounded_losses_after_label_observation,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import PredictionCombinationSettings
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import FixedSharePredictionWeightController
prediction_combination_settings = PredictionCombinationSettings(
    prediction_combination_strategy="fixed_share_weighted_prediction",
    prediction_mixture_activation_policy="always",
    prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
    prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
    fixed_share_weight_redistribution_time_scale_samples=30,
)
for class_count in (2, 3):
    controller = FixedSharePredictionWeightController(prediction_combination_settings=prediction_combination_settings)
    for observation_index in range(3):
        prediction_weights_by_model_id = controller.get_prediction_weights_before_label_observation(model_ids=(7, -3))
        normalized_prediction_weights_by_model_id = normalize_model_prediction_weights(prediction_weights_by_model_id=prediction_weights_by_model_id)
        model_outputs_by_model_id = {7:torch.tensor([[0.8]]), -3:torch.tensor([[0.2]])} if class_count == 2 else {7:torch.tensor([[0.0, 1.0, 2.0]]), -3:torch.tensor([[2.0, 1.0, 0.0]])}
        prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count)
        combined_prediction_probabilities = combine_model_prediction_probabilities(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, prediction_weights_by_model_id=normalized_prediction_weights_by_model_id, class_count=class_count)
        predicted_class_labels = predict_class_labels_from_prediction_scores(prediction_scores=combined_prediction_probabilities, class_count=class_count)
        assert tuple(predicted_class_labels.shape) == (1, 1)
        observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, observed_class_labels=torch.tensor([1]), class_count=class_count)
        controller.update_weights_after_loss_observation(observed_losses_by_model_id=observed_losses_by_model_id, prediction_weights_by_model_id=normalized_prediction_weights_by_model_id)
        assert abs(sum(controller.weights_by_model_id.values())-1) < 1e-12
assert not any(name == 'federated_drift_experiment' or name.startswith('federated_drift_experiment.') for name in sys.modules)
print('PASS: binary/multiclass public numerical API + weight update; no legacy imports')
'@ | ../../venv/Scripts/python.exe -
```

## 変更範囲

旧productionとgoldenは次の固定基準との差分なし。新srcの数値moduleはstdlib/torchのみ。ASTではtorch許可をexact moduleに限定し、別のprediction module・NumPy・上位runtime・controller・旧package・configの注入を拒否した。

```powershell
git diff --name-only 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json
git diff --check
```

差分対象一覧は空、diff checkエラーなし。新しいplaceholder・hardcoded secretはなし。コミット保留3資料とresultsは変更対象外。

