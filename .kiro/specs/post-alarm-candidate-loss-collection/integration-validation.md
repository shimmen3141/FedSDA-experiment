# 統合検証の証拠

## 完成範囲

警報後の候補と固定参照の損失系列、提案次位置からの連続順、規定件数到達、immutable copy・atomic拒否だけを所有する。
モデル・参照snapshot生成/学習、履歴baseline、採否起動、payload/held_data、episode・正式登録/切替・終端回収は後続。新FedSDA全体runの完成ではない。

## 基準環境と全回帰

docs/experiments/refactoring-baseline.mdに従う共有venv。Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnistを明示。

```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/loss-collection-final-20261004b
```

2026-10-04: exit 0、2664 passed /3 skipped /157.32s。旧11/最終3goldenとschema・対象機能を含む。3 skipは既存Windows shell非対応のserver wrapper2件・ablation suite1件。旧production、golden/許容誤差に748c3aaから差分なし。

## 対象と依存境界

対象72件＋AST104件＝176 passed /exit 0。AST更新前は2 failed /174 passed、exact同機能設定のみの許可追加後成功。NumPy/torch/旧client/globalconfig/runtime/採否/FIFO/別collectionを禁止注入で確認。
task 2はtest-only接続なのでRED非該当、productionに数値評価の依存を増やさない。

## 条件対応

| 条件 | 証拠 |
|---|---|
| 1.1, 1.2, 1.3 | 既存設定/開始位置/固定ID、型・値域・空/重複と改変設定の拒否 |
| 2.1, 2.2, 2.3, 2.4 | 旧sessionへtarget2/3/5・proposal0/100/大整数、逆順dict・float系列、全不正入力の原子的拒否 |
| 3.1, 3.2, 3.3 | 各回count/ready比較、完了後拒否と繰返しsnapshot保持、採否/モデル/終端callbackなし |
| 4.1, 4.2 | frozen copy/inputdict変更・別実体独立、共有RNG/defaultdtype/device不変 |
| 5.1, 5.2 | 旧client直接observe→同target回finalize、live参照消失後も固定順、既存採否への明示接続と旧pure適合/二分採否 |
| 5.3 | 全AST/golden/独立smoke、LEGACY-004記録・正本とpartial範囲 |

15/15条件の証拠あり。src・test構成は設計と一致、収集値→snapshot→既存数値評価の境界をtestで接続する。設定だけのproduction依存、共有状態整合、blockedなしを確認。旧sessionの不足参照時部分更新は正常client影響を未確認として記録し、旧productionは修正しない。

## 独立起動

新しいPython processで以下を実行しexit 0 /COLLECTION_SMOKE_PASS。旧/torch/NumPyをimportせず、入力拒否・観測順・完了/不変状態まで起動確認した。

```python
import sys
sys.path.insert(0,'src')
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection
collection=PostAlarmCandidateLossCollection(
    candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy='current_model_first_reuse_then_two_segment_candidate_validation',
        candidate_post_alarm_validation_sample_count=2),
    proposal_sample_index=100,reference_model_ids=(9,-2))
state_before_call=collection.get_state_snapshot()
try:
    collection.observe_losses_after_label_observation(sample_index=101,candidate_loss=.2,reference_losses_by_model_id={9:.5})
except ValueError:
    pass
else:
    raise AssertionError('missing reference must be rejected')
assert collection.get_state_snapshot()==state_before_call
collection.observe_losses_after_label_observation(sample_index=101,candidate_loss=.2,reference_losses_by_model_id={-2:.6,9:.5})
assert not collection.ready_for_acceptance_evaluation
collection.observe_losses_after_label_observation(sample_index=102,candidate_loss=.3,reference_losses_by_model_id={9:.4,-2:.5})
state_snapshot=collection.get_state_snapshot()
assert state_snapshot.ready_for_acceptance_evaluation
assert state_snapshot.candidate_losses==(.2,.3)
assert state_snapshot.reference_losses_by_model_id==((9,(.5,.4)),(-2,(.6,.5)))
assert state_snapshot.last_validation_sample_index==102
try:
    collection.observe_losses_after_label_observation(sample_index=103,candidate_loss=.2,reference_losses_by_model_id={9:.5,-2:.6})
except RuntimeError:
    pass
else:
    raise AssertionError('completed collection must reject append')
assert collection.get_state_snapshot()==state_snapshot
assert not any(name=='federated_drift_experiment' or name.startswith('federated_drift_experiment.') for name in sys.modules)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
print('COLLECTION_SMOKE_PASS: ordered series, atomic rejection, readiness; no legacy/numeric imports')

```
