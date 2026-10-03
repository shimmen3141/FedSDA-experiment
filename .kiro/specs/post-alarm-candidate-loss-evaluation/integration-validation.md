# 統合検証: 警報後の候補損失評価

## 対象と判定

対象は外部収集済みloss列による既存参照の適合選択、現行優先、全参照の比較対象選択、二分区間の候補採否と診断値。結果はimmutableで可変stateを蓄積しない。
モデルforward・学習・履歴平均推定・収集session・FIFO・採否後の登録/割当・新FedSDA全体runは後続範囲。最終判定と承認状態の正本はspec.json。

## 要件の証拠

以下はtests/refactoring/test_post_alarm_candidate_loss_evaluation.pyのtest_post_alarm_candidate_接頭辞を省略する。

| 条件 | 実装・実行証拠 |
|---|---|
| 1.1, 1.2, 1.3 | reference_selection_matches_legacy、invalid_reference_inputs_are_rejected_without_mutation、invalid_evaluation_inputs_are_rejected_without_mutation。既存固定条件と明示閾値、非空同長tuple損失、ID/履歴/最低件数、理由付き拒否、入力不変 |
| 2.1, 2.2, 2.3 | reference_selection_matches_legacy。12入力で現行優先・代替・低ID同率・履歴欠落・消えたID・負ID・許容値等号を旧pure関数へ照合 |
| 3.1, 3.2, 3.3, 3.4 | evaluation_matches_legacy_finalization、rounding_boundary_preserves_decision_reason。2/3/10/11件×9分岐の36系列と境界4ケースを旧finalizeへ直接照合。元Python sumと入力順tie、適合参照置換、消えた初期参照保持、奇数split、独立したfloat32採否/理由演算と両方向の不一致 |
| 4.1, 4.2 | 上記全診断照合、results_inputs_and_random_states_are_independent、public_functions_require_explicit_keyword_arguments。frozen結果、反復評価、入力/共有乱数/defaultdtype/device不変。float64/meta環境中でも明示CPUfloat32結果が一致、finally復元 |
| 5.1, 5.2, 5.3 | directoracle、public_loss_connection_matches_legacy、全src ASTと禁止注入、全refactoring/全tests/golden/独立smoke。2/3/10class×3/10標本のpublic損失を観測順に渡し、model/collector状態を評価へ取り込まない |

旧oracleはselect_forward_fitting_referenceとFedSDAClient._finalize_forward_validationの実メソッド。後者はテスト専用sessionへ損失を渡し、decision appendをcaptureして専用例外で中断する。診断構築まで実行し、実モデル登録・割当操作は実行しない。
旧sessionはappend時にfloat化するため整数端点もテスト側で対応する。旧理由文字列と新診断理由、履歴欠落の旧NaNと新Noneはテスト側で明示対応し、productionにaliasを設けない。

## 検証コマンド

worktreeルート、共有Windows golden環境（Python3.13.15、torch2.12.1+cpu、NumPy2.4.6）。

```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
../../venv/Scripts/python.exe -m pytest tests/refactoring/test_post_alarm_candidate_loss_evaluation.py tests/refactoring/test_run_settings_validation.py tests/refactoring/test_single_run_dependency_boundaries.py tests/refactoring/test_class_probability_calculations.py -q -p no:cacheprovider
../../venv/Scripts/python.exe -m pytest tests/refactoring -q -p no:cacheprovider
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/candidate-loss-final-20261004b
```

basetempは今回未使用の場所、再実行は別の未使用名を使う。Windows一時ディレクトリの権限制限を避けて全testsを実行する。
task 3 REDは3 AST failed / 1546 passed、exact許可追加後GREENは1549 passed / 5.02s、exit 0。全refactoringは2219 passed / 11.40s、exit 0。
初回全testsは2504 passed / 3 skipped / 287.20sだが、device setter復元が余分なDeviceContextを残す検証側の問題を確認した。一時torch.deviceスコープへ修正し、対象1549 passed / 5.05sを確認してから上記bの未使用basetempで全回帰を再実行した。新productionの演算は変更していない。命名revision 2/3の追加役割はLuna PASSを反映した。
修正後の全testsは2504 passed / 3 skipped / 129.94s、exit 0。旧11ケース・最終提案3ケースのgoldenを値・許容誤差とも更新せず通過した。既存のWindows対象外の以下3ケースのみスキップ。
Lunaは修正後diffと対象を再検証しtask 3 APPROVED。主担当の最終対象再検証は1549 passed / 3.73s、独立smoke再実行exit 0。

- test_server_sweep_wrapper_resolves_runtime_options
- test_server_sweep_wrapper_rejects_owned_options
- test_main_ablation_suite_lists_individual_variants

## 独立プロセスsmoke

pytest/旧moduleなしで新固定条件・参照選択・候補評価を起動し、候補採用・現行優先再利用・丸め棄却と理由合格・適合同率の小ID選択をassert。sys.modulesに旧packageがないことも確認した。
出力PASS、exit 0。

```powershell
$env:PYTHONPATH='src'
@'
import sys
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses, select_available_reference_within_historical_loss_tolerance
candidate_model_training_and_acceptance_settings = CandidateModelTrainingAndAcceptanceSettings(candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation", candidate_post_alarm_validation_sample_count=2)
evaluation_arguments = dict(candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings, candidate_losses=(.2,.2,.2), reference_losses_by_model_id={9:(.8,.8,.8), -3:(.4,.4,.4)}, reference_historical_mean_losses_by_model_id={}, available_reference_model_ids=(9,-3), current_training_model_id=9, maximum_reference_mean_loss_increase=.1, minimum_candidate_mean_loss_improvement=.01)
result = evaluate_candidate_using_post_alarm_losses(**evaluation_arguments)
assert result.candidate_accepted and result.comparison_reference_model_id == -3 and result.validation_sample_count == 3
result = evaluate_candidate_using_post_alarm_losses(**(evaluation_arguments | dict(reference_historical_mean_losses_by_model_id={9:.8,-3:.4})))
assert not result.candidate_accepted and result.reusable_reference_model_id == 9
assert result.decision_reason == "current_reference_within_historical_loss_tolerance"
result = evaluate_candidate_using_post_alarm_losses(**(evaluation_arguments | dict(candidate_losses=(.2,.2),reference_losses_by_model_id={9:(.8,.8)},minimum_candidate_mean_loss_improvement=.60000001)))
assert not result.candidate_accepted and result.decision_reason == "both_segment_margins_passed"
assert select_available_reference_within_historical_loss_tolerance(reference_losses_by_model_id={9:(.125,.125),-3:(.125,.125)},reference_historical_mean_losses_by_model_id={9:.1,-3:.1},available_reference_model_ids=(9,-3),current_training_model_id=0,maximum_reference_mean_loss_increase=.1) == -3
assert not any(name == "federated_drift_experiment" or name.startswith("federated_drift_experiment.") for name in sys.modules)
print("PASS: candidate acceptance, current-first reuse, rounding disagreement and reference ties; no legacy imports")
'@ | ../../venv/Scripts/python.exe -
```

## 変更範囲と修正候補

評価の直接依存はstdlib/torch/同機能固定条件のみ。torch許可はexact評価moduleに限定、NumPy・旧/config・上位runtime・隣接予測/監視・別candidateのtorch注入を拒否する。
旧productionとgoldenは固定基準との差分なし。

```powershell
git diff --name-only 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json
git diff --check
```

旧差分対象一覧は空、diff checkエラーなし。新placeholder・hardcodedsecretなし。コミット保留3資料/resultsは変更対象外。
採否とmargin理由が矛盾する丸め境界は旧不整合の修正候補としてresearch.mdへ記録した。今回の移植で式を統一せず、過去実験での発生有無も未確認。今後の修正は別変更で影響を確認する。
