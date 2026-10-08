# 警報のない標本での帰属確定 — 調査記録

2026-10-09、主担当Claude Code。基点commit `0f075c0`。固定旧基準`748c3aa`。Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 旧処理の事実

`federated_drift_experiment/clients/fedsda.py::process_one_step`（461〜534行）の、警報のない分岐（510〜525行）:

```python
while len(self.buffer) > self.fifo_size:
    old_data = self.buffer.popleft()
    old_x, old_y = old_data[:2]
    self._record_model_compute("statistics", len(old_x))
    loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
    class_id = int(old_y.view(-1)[0].item())
    self._update_model_stats(self.current_model_id, loss_val, class_id=class_id)
    self.train_data_store[self.current_model_id].append(old_data)
    self._record_model_concept(self.current_model_id, old_data[2])
self.train_step()
```

- 標本は、この分岐より前（480行）で保留へ追加済み。警報のあった標本ではこの分岐を通らない（警報の処理が保留を扱う）。
- 旧の吸収`_absorb_into_store`（`clients/base.py` 162〜173行）は、標本ごとに「標本追加→概念計数→損失評価→統計」の順。上のinlineは「損失評価→統計→標本追加→概念計数」の順。どちらも、評価に使うモデルは確定の間に変わらず、評価は標本の保存先や計数を読まないので、標本ごとの順の違いは最終状態に現れない。
- `train_step`（`clients/base.py` 198〜206行）は学習要求を1つ数え、間隔に達したら全保有モデルの共同学習を行う。新実装には、学習要求の件数管理（`LocalTrainingRequestSchedule`）と共同学習の反復（`perform_held_model_joint_training_iterations`）があるが、両者をつなぐ処理はまだない（srcのどこからも呼ばれていない。2026-10-09にgrepで確認）。学習の接続は別specにする。

## 新実装の現状

- 保留位置のowner（`PendingTrainingAssignmentBuffer`）は位置だけを持つ。追加は容量で切り捨てず、`release_sample_indices_exceeding_capacity`が容量を超えた最古の位置を解放して返す。容量を超える位置を解放せずに読む操作はない。
- 保留標本そのもの（特徴・ラベル・概念ID）を持つownerはない。警報のときの既存の応答は、呼出し側から全保留標本を位置つきで受け取り、保留位置のownerの並びと一致することを確かめる。本specも同じ形にする。
- 吸収（`absorb_assigned_training_samples_into_held_model`）は、全標本の検査と損失評価を済ませてから、標本ごとに標本追加→概念計数→統計を行う。

## 判断

- **吸収の後で保留を解放する**: 先に解放すると、吸収が解放対象の標本を拒否したとき、保留だけが減った状態が残る。吸収は全検査を更新より前に行うので、「読む→吸収→解放」の順にすれば、拒否のときに何も変わらない。そのために、保留位置のownerへ読取りの操作を足す。
- **解放がないときは吸収を呼ばない**: 設計4節。初めは空の標本列で呼ぶ形にしていたが、仕様レビューで、旧にない例外（現在のモデルが未保有のとき）になると指摘され、旧に合わせた。
- **保留標本を引数で受ける**: 警報のときの応答と同じ。保留標本を持つownerをどこに置くかは、標本1件の処理全体のspecで決める。
- **保留標本の検査を本moduleに書く**: 警報のときの応答のmoduleに同じ趣旨の検査（非公開のhelper）がある。共通化は既存moduleの変更になるので、本specでは行わず、改善候補へ記録する（[IMPROVE-010](../../../docs/research/improvement-candidates/improve-010-share-pending-observation-validation.md)）。

## oracleの実行可能性

吸収のoracle（`tests/refactoring/test_assigned_training_sample_absorption.py`の`build_absorption_oracle`）が、実モデル・統計・標本・計数を持つ実旧clientと、同じ状態の新ownerを返す。その実旧clientへ、標本処理が読む属性（保留のFIFO、容量、処理済み件数、時間計測の入れ物、履歴）を与え、予測・候補検証の観測・検出・学習・計算量の記録を止めて、実旧`FedSDAClient.process_one_step`を呼ぶ。保留位置のownerのtest（`make_legacy_processing_client`）が同じ方法で実行済み（そちらは統計・概念もstub）。リポジトリ外の下書きを作業ツリーの複製で実行し、2/4 class×6条件が成功することを確かめた。

## 手順上の事実

- 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、`git archive HEAD`で作った作業ツリーの複製へ置いて実行した（worktreeは変更していない。対象test・依存境界・共用scriptが成功、Ruff・Pyright成功）。汎用の変異toolも複製で試行し、未検出の1種（保留標本の組のexact tuple検査をisinstanceへ緩める）に対して拒否条件を1件足した。
- 共用のfresh process scriptへ流れを足す際、警報の前に1標本が現在のモデルへ確定するようになるので、流れの前提（警報時の履歴基準）が動かないよう、scriptの履歴統計の件数を大きくした。
