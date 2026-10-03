# 統合検証の証拠

## 範囲

新部品は保留標本位置のFIFOだけを所有する。payload参照、モデル帰属、学習、候補session・episode・警報後client進行と終端方針は後続spec。旧丸め差・短い警報後の残留・終端末尾の扱いは共有発見記録へ接続し、今回修正しない。

## 基準環境・コマンド

docs/experiments/refactoring-baseline.mdのローカルvenvを共用し、Python3.13.15 / torch2.12.1+cpu / NumPy2.4.6 / pytest9.1.1。MPLCONFIGDIRは../../venv/matplotlib-cache、TMP/TEMPは../../venv/refactoring-tests。全回帰はOMP_NUM_THREADS=1、MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnistを指定する。

```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/pending-assignment-final-20261004a
```

2026-10-04実行：exit 0、2580 passed / 3 skipped / 119.22s。既存Windows shell実行に関する3 skip（server wrapperの2件、ablation suiteの1件）は前specと同じ。旧golden11件・最終golden3件を含む。golden/許容誤差と旧productionには748c3aaから差分なし。

## 対象・依存検証

対象FIFOとAST：161 passed / exit 0。うちFIFO67件、AST94件。AST更新前は2 failed /159 passedで同機能設定を拒否した。exact許可を追加して、torch/NumPy/旧client/config/runtime/監視/別FIFO禁止注入も通過。

LEGACY-002再現：10 passed /52 deselected。旧pending/duplicate/insufficient分岐を整数tokenで直接比較。短い警報区間では割当済みprefixを含む全FIFOを保持する。

## 独立smoke

以下をworktreeで新しいPython processへ渡して実行し、exit 0 / FIFO_SMOKE_PASSを確認した。追加・参照・解放・全消費・連続位置拒否後の再開までを起動確認する。

```python
import sys
sys.path.insert(0, 'src')
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import TrainingDataAssignmentSettings
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer
buffer = PendingTrainingAssignmentBuffer(training_data_assignment_settings=TrainingDataAssignmentSettings(pending_assignment_buffer_capacity_samples=2))
for sample_index in range(71, 74):
    buffer.append_observed_sample_index(sample_index=sample_index)
state_snapshot = buffer.get_state_snapshot()
assert state_snapshot.pending_sample_indices == (71, 72, 73)
result = buffer.get_change_interval_partition(estimated_change_span_sample_count=2)
assert result.earlier_sample_indices == (71,)
assert result.change_interval_sample_indices == (72, 73)
assert result.change_interval_start_sample_index == 72
assert buffer.get_state_snapshot() == state_snapshot
assert buffer.release_sample_indices_exceeding_capacity() == (71,)
assert buffer.drain_pending_sample_indices() == (72, 73)
assert buffer.get_state_snapshot().last_observed_sample_index == 73
try:
    buffer.append_observed_sample_index(sample_index=0)
except ValueError:
    pass
else:
    raise AssertionError('drain must not reset observation sequence')
buffer.append_observed_sample_index(sample_index=74)
assert buffer.get_state_snapshot().pending_sample_indices == (74,)
assert not any(name == 'federated_drift_experiment' or name.startswith('federated_drift_experiment.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
print('FIFO_SMOKE_PASS: append/partition/release/drain/continuity; no legacy or numeric imports')

```

## 条件対応と設計一致

| 条件 | 証拠 |
|---|---|
| 1.1, 1.2, 1.3 | 既存容量設定・旧process_one_step直接比較、容量1/3/30、開始0/71、C+1と超過順 |
| 2.1, 2.2, 2.3 | 16系列の空/切詰め/過大span、旧global開始位置、非破壊参照・短い警報保持 |
| 3.1, 3.2, 3.3 | 明示drainとlast維持、3警報分岐、別実体/frozen copy/共有乱数不変 |
| 4.1, 4.2, 4.3 | 型/負/連続性/容量/span拒否state不変、旧client各oracle、public監視結果のspan接続 |
| 5.1, 5.2 | exact ASTと全golden・旧importなしsmoke、共有発見記録・部分完成明記 |

14/14条件に証拠あり。実ファイルは設計と一致。FIFO部品がstdlibと同機能設定を参照し、monitorの結果をtestで明示入力するためproduction依存方向は変わらない。3操作間の状態整合と観測順を確認、blockedなし。実際のモデル帰属と新全体runの完成は未主張。

