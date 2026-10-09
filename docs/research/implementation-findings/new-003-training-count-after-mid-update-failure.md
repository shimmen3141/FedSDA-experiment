# NEW-003: 1回の共同更新の途中で失敗したときの学習量の計数が、旧と違う

- 発見日: 2026-10-09。対象: `src/federated_learning_experiments/runtime/held_model_training_request_handling.py`の`train_held_models_for_pending_training_requests`（commit `1a03667`）。発見の経緯: spec held-model-training-request-handlingの仕様レビュー。
- 種別: 旧実装との挙動の差（失敗の経路だけ）。状態: 許容（2026-10-09、ユーザー決定。現状問題がないのでそのままにする）。

## 内容

旧（`federated_drift_experiment/clients/shared_backbone.py`の`_train_heads_together`）は、1回の共同更新の中で、モデルごとに個別部のoptimizerを進めた直後に、そのモデルの学習標本数と更新回数を足す。その後に計算量の診断を記録する。

新は、1回の共同更新が完了してから、参加したモデルの計数を足す（共同学習の反復は損失しか返さないので、完了を損失の有無で知る）。

| 場合 | 旧 | 新 |
| --- | --- | --- |
| 正常終了 | 一致 | 一致 |
| 共同更新の合間（抽出、次の回の検査）で例外 | 完了した回のぶんの計数が残る | 同じ（実旧との対照testあり） |
| 1回の共同更新の途中（個別部のoptimizerを順に進めている間、旧ではその後の診断の記録の間）で例外 | そこまでに進めたモデルの計数が残る | その回の計数は残らない |

## 影響

旧も新も、この例外は標本処理の外まで伝わる。旧は試行単位で例外を捕捉して次の試行へ進む（`federated_drift_experiment/trials.py`）が、失敗した試行の結果は返らない。現在の流れでは、差のある計数が読まれる経路はない。失敗の後に実行を続けて計数を使う仕組みを作る場合は、この差を見直す。

## 関連

- 設計の記録: `.kiro/specs/held-model-training-request-handling/design.md` 2節「旧と違う点」。
- 合わせる方法の案: [IMPROVE-011](../improvement-candidates/improve-011-report-participating-models-from-joint-training.md)（共同学習の反復が参加したモデルを返す）。未採用。
