# 統合検証: 全体・正解クラス別の損失監視

## 対象と判定

対象は単一有界損失のe-SR状態と全体・正解クラス系列の固定重み混合、global候補位置、reset、観測結果と診断copy。
モデルforward・学習・モデル統計からのbaseline推定・FIFO帰属・候補の将来損失判定・警報後操作・server同期・新FedSDA全体runは後続範囲。
最終feature判定はGO、承認の正本はspec.json。task 1・2・3のLunaレビューはAPPROVEDで、具体的な未解決指摘はない。Lunaが20/20条件・全体接続・設計境界・blockedなしを確認し、独立smokeも再実行してGO。主担当も最新の全回帰・smoke・承認hashと完成範囲を確認し、FEATURE_GO VERIFIEDとした。

## 要件の証拠

以下はtests/refactoring/test_loss_change_monitoring.pyのtest_loss_monitoring_接頭辞を省略する。

| 条件 | 実装・実行証拠 |
|---|---|
| 1.1, 1.2, 1.3, 1.4 | single_series_matches_reference_after_each_observation、candidate_ties_and_retention_match_reference、reset_and_baseline_limits_match_reference。保持上限1/7/1000 × baseline0/0.2/0.6/1 × bet5個/単一/重複の各130観測で、capital・内部番号・log-e・警報・幅・件数を完全一致照合 |
| 2.1, 2.2, 2.3, 2.4 | overall_and_class_series_match_reference、first_class_observation_freezes_current_baseline。class数2/3/10 × 上限1/7/1000の各420観測で、全体/該当classだけの更新・遅延baseline・最新他class値・未観測class分の非再配分・混合閾値・delta計数を旧実メソッドへ照合 |
| 3.1, 3.2, 3.3, 3.4 | overall_and_class_series_match_reference、component_ties_preserve_reference_priority、class_positions_survive_retention_and_reset。global候補位置・非連続class位置・位置切捨て・全体→今回class→他class優先・警報後非resetを照合。同率は両者へ同じ成分寄与を注入して優先順だけを分離検証 |
| 4.1, 4.2, 4.3, 4.4 | invalid_inputs_preserve_complete_state、invalid_inputs_preserve_complete_state_for_mixture、class_positions_survive_retention_and_reset、snapshots_and_instances_are_independent、public_calls_preserve_random_states、public_functions_require_explicit_keyword_arguments。明示条件・理由付き拒否・全状態不変・reset後任意global位置・frozen copy・別実体/RNG独立 |
| 5.1, 5.2, 5.3, 5.4 | 上記direct oracle、current_model_observed_loss_connects_without_prediction_state、全src AST・禁止依存注入・全tests・独立smoke。指定現行モデルの観測後有界損失public APIを監視へ渡し、重みcontrollerやモデル状態を所有させない。完成範囲は監視部品のみ |

旧oracleはBoundedMeanEDetectorとClassConditionalESRFedSDAClientの実メソッドをテスト側から呼ぶ。新productionへの旧importや互換窓口はない。
旧clientの不正class入力でoverallだけ進む挙動は引き継がず、全入力の更新前検証を新API契約とした。正常な有限損失系列の完全一致とは別に検証する。

## 検証コマンド

worktreeルートから共有Windows golden環境（Python 3.13.15、torch 2.12.1+cpu、NumPy 2.4.6）を使用。

```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
../../venv/Scripts/python.exe -m pytest tests/refactoring/test_loss_change_monitoring.py tests/refactoring/test_single_run_dependency_boundaries.py tests/refactoring/test_class_probability_calculations.py tests/refactoring/test_run_settings_validation.py -q -p no:cacheprovider
../../venv/Scripts/python.exe -m pytest tests/refactoring -q -p no:cacheprovider
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/class-monitoring-final-20261004a
```

basetempは今回未使用の場所を指定した。再実行時は別の未使用名にする。Windows一時ディレクトリの権限制限を避けて全testsを実行する。
task 3のAST許可更新前は4 failed / 1529 passed、更新後は1533 passed / 6.64s、exit 0。
Lunaの独立確認は対象1533 passed / 5.19s、全refactoring 2105 passed / 9.12s、exit 0。主担当の全refactoringは2105 passed / 6.85s、レビュー後の対象再検証は1533 passed / 5.32s、exit 0。
全testsは2390 passed / 3 skipped / 123.81s、exit 0。既存11ケースと最終提案3ケースのgoldenは全成功、golden値・許容誤差は更新していない。スキップはWindowsで実行しない既存の以下3ケースだけである。

- test_server_sweep_wrapper_resolves_runtime_options
- test_server_sweep_wrapper_rejects_owned_options
- test_main_ablation_suite_lists_individual_variants

## 独立プロセスsmoke

pytest fixtureなしに単一検出器・混合監視・監視設定のpublic APIをimportし、2/10クラスの各30観測、候補件数、snapshot保持、reset後global位置再開、単一閾値の等号判定をassertした。
sys.modulesに旧packageとtorchがないこともassertした。出力はPASS、exit 0。

```powershell
$env:PYTHONPATH='src'
@'
import sys
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import LossChangeDetectionSettings
from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import BoundedLossESRDetector
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor
loss_change_detection_settings = LossChangeDetectionSettings(drift_detector_name="e_sr", loss_monitoring_scope="overall_and_true_class_losses", e_sr_false_alarm_control_alpha=0.05)
for class_count in (2, 10):
    monitor = OverallAndTrueClassLossMonitor(loss_change_detection_settings=loss_change_detection_settings, class_count=class_count, initial_baseline_loss_mean=0.2, maximum_retained_candidate_count=7, betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8))
    for sample_index in range(100, 130):
        observation = monitor.observe_loss_after_label_observation(observed_loss=0.8, observed_class_id=sample_index % class_count, sample_index=sample_index, current_model_baseline_loss_mean=0.2)
        assert observation.component_update_count == 2
        assert observation.estimated_change_span_sample_count >= 1
    state_snapshot = monitor.get_state_snapshot()
    assert state_snapshot.last_sample_index == 129
    assert len(state_snapshot.class_esr_states_by_class_id) == class_count
    monitor.reset(baseline_loss_mean=0.6)
    assert monitor.last_observation is None
    assert state_snapshot.last_sample_index == 129
    monitor.observe_loss_after_label_observation(observed_loss=0.2, observed_class_id=0, sample_index=500, current_model_baseline_loss_mean=0.6)
detector = BoundedLossESRDetector(baseline_loss_mean=0.2, false_alarm_control_alpha=0.5, maximum_retained_candidate_count=7, betting_fractions=(0.1,))
assert not detector.observe_loss(observed_loss=0.2).drift_detected
assert detector.observe_loss(observed_loss=0.2).drift_detected
assert not any(name == "federated_drift_experiment" or name.startswith("federated_drift_experiment.") for name in sys.modules)
assert "torch" not in sys.modules
print("PASS: single/mixed loss monitoring, reset and snapshots; no legacy/torch imports")
'@ | ../../venv/Scripts/python.exe -
```

## 変更範囲と完成境界

単一kernelはstdlib/NumPyだけ、混合monitorはstdlib・同機能kernel・既存設定だけを参照する。ASTはNumPy許可をexact単一moduleに限定し、混合側のNumPy/torch・上位runtime・隣接controller・旧package・別detectorのNumPy注入を拒否する。
旧productionとgoldenの固定基準差分を確認する。

```powershell
git diff --name-only 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json
git diff --check
```

旧production・両goldenの差分対象一覧は空、diff checkエラーなし。新しいplaceholder・hardcoded secretはなし。コミット保留3資料とresultsは変更対象外。
alphaと監視対象は既存LossChangeDetectionSettingsが所有する。モデル統計・FIFO・履歴collector・client調整は監視へ吸収していない。後続未作成specをこの完了範囲へ含めない。

