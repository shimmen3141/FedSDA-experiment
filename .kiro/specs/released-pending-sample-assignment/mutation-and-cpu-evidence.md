# 検出力の証拠

Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）、source/test commit `cbf38b6`（sourceの最終変更は`86cd7ed`。`cbf38b6`はtestだけを追加した）。証拠ファイルは元checkoutの`venv/refactoring-tests/released-pending-sample-assignment-mutation-evidence/`（Git管理外、このPCだけ）。

## 汎用の変異tool

### 確定の関数

```text
python .kiro/settings/scripts/mutation_check.py --source src/federated_learning_experiments/runtime/released_pending_sample_assignment.py --functions assign_released_pending_samples_to_current_training_model --tests tests/refactoring/test_released_pending_sample_assignment.py --evidence <.../assignment>
```

結果: 21/22検出。復元後は47 passed。`report.json`は`assignment/`。

未検出の1種は等価と判断した: 位置の並びの検査を、「容量を超える位置を読む代入」（`released_sample_observations = ...`）の直後へ移す変異。この検査は自分の検査の最後で、移した先との間にあるのはその代入だけである。代入は読取りだけで、ここまでの検査を通った入力（exact型の保留位置のownerと、exact tupleの保留標本）では例外を出さない。移した先は、解放がないときの早期returnと吸収の呼出しより前なので、どの入力でも、同じ例外が状態の更新や戻り値より前に起こる。

Task 1の独立レビュー（1回目）の前は17/20で、未検出の残り2種（現在の学習帰属のownerの型検査、保留標本の組がexact tupleであることの検査を、同じ代入の後へ移す変異）も等価と判断していた。レビューで、型の不正と並びの不一致が同時にある入力では例外の型が変わる（元はTypeError、変異ではValueError）ので等価ではないと指摘された。型の検査が並びの検査より先に拒否することを確かめるtestを足し、この2種は検出されるようになった。

Task 2の独立レビュー（1回目）の前は19/20だった。レビューで、toolの「消す」変異が関数の最上位の文だけを対象にしており、保留標本の要素ごとの検査（loopの中の、要素の型と位置のintの検査）は「緩める」変異しか実行されていないと指摘された。toolへ「最上位のfor/whileの本体の中の文を1つずつ消す」変異を足し（汎用の変更）、実行し直した。足された2種（要素の型検査を消す、位置のint検査を消す）はどちらも検出された。

検出された変異には、保留位置のownerの型検査を後へ移す・各検査を消す（loopの中の2つを含む）・exact型検査（owner、tuple、要素、位置）をisinstanceへ緩める・解放がないときの早期returnを消す・吸収を消す・解放を消す・吸収と解放を入れ替える、が含まれる。

仕様化の段階（作業ツリーの複製）で、保留標本の組のexact tuple検査をisinstanceへ緩める変異が未検出だったので、tupleの派生型の拒否条件を足した。

### 保留位置のownerの読取りの操作

`get_sample_indices_exceeding_capacity`の本体は代入とreturnだけで、toolが機械的に作れる変異がない。`--extra`で3種を足した: 1件多く返す、新しい側から返す、容量を使わず件数を決める。

```text
python .kiro/settings/scripts/mutation_check.py --source src/federated_learning_experiments/methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py --functions get_sample_indices_exceeding_capacity --tests tests/refactoring/test_pending_training_assignment_buffer.py --evidence <.../buffer> --extra ...（3種）
```

結果: 3/3検出。復元後は69 passed。

## 共用のfresh process script

`tests/refactoring/fresh_process_smoke.py`へ、警報の前に「容量＋1件が保留された時点で最古の1件が現在のモデルへ確定する」流れを足した（確定した標本が渡した最古の標本であること、保留に残る位置、学習標本が現在のモデルの1件だけであることを確かめる）。全16の流れが成功し、`test_fresh_process_smoke.py`が全pytestから別processで実行する。
