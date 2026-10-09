# IMPROVE-011. 共同学習の反復が、参加したモデルを返す

- 記録日: 2026-10-09。種別: 簡略化（条件の重複の解消）。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/learning/training/held_model_joint_training_iterations.py`（`perform_held_model_joint_training_iterations`）と、`runtime/held_model_training_request_handling.py`の学習量の計数。発見spec: held-model-training-request-handling（基点commit `35e7af0`）。

## 確認した事実

共同学習の反復は、完了した共同更新の損失だけを返し、各回にどのモデルが参加したかを返さない。学習量の計数（モデル別の学習標本数と更新回数）には参加モデルが必要なので、学習要求の処理は、反復を共同更新1回ぶんずつ呼び、損失が返った回ごとに、参加の条件「保有していて、標本がbatchの件数以上あるモデル」を、反復へ渡したのと同じsnapshotからもう一度求めている。反復は呼出しのたびに保有モデルの対応を検査し直す。同じ条件は、batchの抽出（`held_model_training_batch_sampling.py`の`_validate_sampling_request`）にある。

反復の間は保有モデルと標本列が変わらないので、現在は両者が食い違わない（実旧との対照testで計数の一致を確かめている）。抽出の参加条件を変えた場合は、計数の側も合わせて変える必要がある。

## 案

共同学習の反復の戻り値を、損失に加えて各回の参加モデルID（とbatchの件数）を持つ記録にする（または、回ごとに呼出し側へ通知する）。学習要求の処理は、その記録から計数を足し、反復を1度の呼出しで済ませる。

## 挙動への影響

正常終了の後と、共同更新の合間で失敗した後の値は変わらない（現在も旧と一致している）。1回の共同更新の途中（個別部のoptimizerを進めている間と、旧ではその後の診断の記録の間）で失敗した場合の計数（現在は旧と違う）を合わせるかどうかは、採用のときに決める（この差は2026-10-09にユーザーが許容した。記録は[NEW-003](../implementation-findings/new-003-training-count-after-mid-update-failure.md)）。

## 必要な検証

反復の実旧対照（損失・parameter・乱数）と、学習要求の処理の実旧対照（計数）が変わらず成功すること。採否・変更commitは未定。
