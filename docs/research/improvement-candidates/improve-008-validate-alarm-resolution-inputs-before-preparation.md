# IMPROVE-008. 警報応答で、区間解決の引数の検査を区間の準備より前に行う

- 記録日: 2026-10-09。種別: 簡略化（契約外入力での部分更新をなくす）。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/runtime/alarm_buffer_response.py::respond_to_alarm_with_buffered_samples`と、その中で呼ぶ`alarm_training_interval_preparation.py`・`alarm_change_interval_resolution.py`。発見spec: alarm-occurrence-handling（Task 1の独立レビュー、Claude Haiku 5.5の任意の指摘。検証commit `e80b368`）。

## 確認した事実

警報応答は、(1)区間の準備（前区間の評価標本の保存と現行モデルへの吸収）、(2)変化区間の件数の判定、(3)区間解決（再利用評価、切替または候補検証の開始）の順に行う。区間解決だけが使う引数（許容損失増加量`maximum_alarm_interval_mean_loss_increase`など）の検査は(3)の中にあり、(1)の更新の後に行われる。そのため、これらの引数が不正なとき、前区間の吸収と評価標本の保存が残ったまま例外になる。

これはalarm-buffer-responseが決めた既存の契約で、`tests/refactoring/test_alarm_buffer_response.py::test_failure_after_preparation_keeps_completed_earlier_updates`が固定している。旧`_resolve_drift`も、前区間の吸収の後で区間の評価に進む。

確認していないこと: (3)でだけ検査される引数の全一覧（候補の設定類が含まれるか）。通常の経路でこれらの引数が不正になることがあるか（設定は組立時に検査済みのはずで、契約外入力だけの問題と見ている）。

## 案と効果の仮説

応答の冒頭で、区間解決の引数の検査（読取りだけ）を先に済ませる。不正な設定で応答が失敗したとき、どのownerも更新されていない状態になり、呼出し側（`handle_alarm_occurrence`、標本1件の処理）の「検査はすべて最初の更新より前」という説明が例外なしで成り立つ。

## 挙動への影響

正常な入力での数値・選択・乱数は変わらない見込み（検査の位置だけ）。契約外入力での例外の時点と、残る状態が変わる。上のtestは期待を書き換える必要がある。変化区間が不足する警報（区間解決へ進まない）でも不正な設定を拒否するようになる点は、挙動の変化として扱う。

## 必要な検証

検査を先に移す変更と、既存の実旧対照（正常経路）が変わらないことの確認、拒否のtestの更新、変異（検査を準備の後へ戻す）の検出。採否・変更commitは未定。旧挙動を維持する現在の移植へは混ぜない。
