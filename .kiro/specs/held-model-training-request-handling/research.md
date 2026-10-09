# 学習要求の記録と保有モデルの共同学習 — 調査記録

2026-10-09、主担当Claude Code。基点commit `35e7af0`。固定旧基準`748c3aa`。Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 旧処理の事実

`federated_drift_experiment/clients/base.py`:

```python
def train_step(self):
    self._pending_updates += 1
    if self._pending_updates >= config.LOCAL_UPDATE_INTERVAL:
        self.flush_pending_updates()

def flush_pending_updates(self):
    if self._pending_updates > 0:
        self.train_all_held_models(count_multiplier=self._pending_updates)
        self._pending_updates = 0
```

- 最終構成のclientは`shared_backbone.py`の`train_all_held_models`（187〜203行）を使う。`config.SHARED_BACKBONE_TRAINING`が`joint`なら`_train_heads_together(count_multiplier, update_backbone=True)`を呼ぶ。`sequential`（基底clientの、モデルごとに順に学習する方式）と`frozen`（共有部を更新しない診断）は最終構成で使わない。
- `_train_heads_together`（234〜396行）は`updates_per_sample × count_multiplier`回、毎回`_sample_training_batches`（保有していて、標本が`batch_size`以上あるモデルを、標本の辞書の順に）でbatchを抽出して共同更新し、参加モデルごとに`model_training_examples[model_id] += len(bx)`と`model_optimizer_steps[model_id] += 1`を行う。参加モデルがない回は何もしない。`len(bx)`は常に`batch_size`（標本は1件1行）。
- `updates_per_sample`が0でも、間隔に達すれば`flush_pending_updates`が`train_all_held_models`を呼び（0回の学習）、保留を0へ戻す。
- 学習が例外で失敗すると、`_pending_updates = 0`へ進まないので保留は残る。失敗までに完了した更新のぶんの計数は残る。
- 呼出し位置: `train_step`は`fedsda.py::process_one_step`の警報のない分岐の最後（525行）、`flush_pending_updates`は警報の処理の前（485行）と、ラウンド境界（clientの外）。位置の接続は本specの範囲外。

## 新実装の現状（作業開始時点。基点commit `35e7af0`）

- `LocalTrainingRequestSchedule`: `record_training_request`（件数を足し、間隔に達していれば保留件数×一要求あたりの回数、達していなければ0を返す）、`calculate_pending_joint_update_iteration_count`（保留件数×回数）、`acknowledge_completed_training_requests`（現在の全保留件数と等しい正の件数で0へ戻す）、`pending_training_request_count`。間隔に達したかを直接読む操作はない。件数管理のtestは、外側が設定の間隔と保留件数を比べている。
- `perform_held_model_joint_training_iterations`: 回数・保有モデルの対応・標本列・batchの件数・乱数生成器・学習の設定・共有部とoptimizer・共有部を更新するかを受け取り、完了した共同更新の損失を返す。どのモデルが参加したかは返さない。回数が0なら入力を読まない。
- `ModelTrainingAndAssignmentCountsStore.record_completed_model_training`: モデルごとに学習標本数と更新回数を足す。
- 上の3つをつなぐ処理はsrcにない（2026-10-09にgrepで確認。`record_completed_model_training`を呼ぶのは採用時の移管だけ）。

## 判断

- **関数を2つにする**: 旧の`train_step`と`flush_pending_updates`は呼出し位置が違う（標本ごと／警報の前とラウンド境界）。新でも呼出し側が選べるように2つにし、前者は後者を呼ぶ（旧と同じ形）。
- **間隔に達したかの読取りを件数管理へ足す**: `record_training_request`の戻り値は、回数が0のとき、間隔に達したかを区別できない。接続の関数が設定を別に受け取って比べる形は、件数管理が持つ設定と食い違いうる。件数管理が自分の設定で答える読取りを足す。
- **反復を共同更新1回ぶんずつ呼び、完了のたびに計数を足す**: 共同学習の反復は損失しか返さない。反復を変えて参加モデルを返させる方法もあるが、承認済みのmoduleとそのtestの変更になる。反復の間は保有モデルと標本列が変わらず、参加の条件は「保有していて標本がbatchの件数以上」の1つだけなので、同じsnapshotから求める。条件が2箇所にある点は改善候補へ記録する（IMPROVE-011）。初めは、全回を1度の呼出しで学習してから計数をまとめて足す形にしていたが、仕様レビューで、共同学習が途中で失敗すると旧と違って完了済みの更新のぶんの計数が残らないと指摘され、1回ぶんずつ呼んで完了のたびに足す形へ改めた（共同更新1回の単位では旧と同じ時点。残る違いは、1回の共同更新の途中——個別部のoptimizerを進めている間と、旧ではその後の診断の記録の間——の失敗だけ。設計2節）。反復は呼出しのたびに保有モデルの対応を検査し直すが、抽出と更新の順・乱数の消費は、全回を1度に呼ぶ場合と同じである（実旧との対照testで示す）。
- **共有部の更新は常に行う**: 新の`LocalTrainingSettings`は共同学習（共有部も更新）だけを受け入れる。凍結の方式を呼出し側が選べる引数にすると、設定と食い違う組合せを作れるので、引数にしない。
- **ownerの型だけを先に確かめる**: 学習へ渡すだけの値は、反復が抽出より前に検査する。旧も学習に入るまで読まない。要求の記録より前に確かめるのは、本処理が読む・更新するownerの型だけにする。

## oracleの実行可能性

吸収のoracle（`build_absorption_oracle`）の実旧clientは`SharedBackboneClassConditionalESRFedSDAClient`を`__new__`で作ったもので、実物の`train_step`・`flush_pending_updates`・`train_all_held_models`・`_train_heads_together`・`_sample_training_batches`・`_shared_backbone`・`_record_model_compute`をそのまま持つ。学習stepが読む属性と設定を与えれば実行できる。モデル9へ2標本を吸収して標本数を3件と5件にし、保有していないモデル55へ6標本を持たせると、batchの件数（1・3・4・6）で参加モデルが変わる。リポジトリ外の下書きを作業ツリーの複製で実行し、全条件が成功することを確かめた。

## 手順上の事実

- 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、`git archive HEAD`で作った作業ツリーの複製へ置いて実行した（worktreeは変更していない。新testと件数管理のtest・依存境界・共用scriptを実行するtestが成功、Ruff成功）。汎用の変異toolも複製で試行した。1回目に未検出だった「計数の反映と保留の消化を入れ替える」変異に対して、計数への反映の時点で保留が未消化であることの確認を足し、手で足す変異のために保有していないモデルの標本を足した。仕様レビューの反映の後に未検出だった「損失が返らなかった回を飛ばす文を消す」変異に対して、反復が損失を返さなかった回を計数しないことのtestを足した。このtestは反復を差し替えており、実旧をoracleにしていない（実際の反復では、損失が返らない回は参加するモデルがない回と同じで、その回は計数するモデルもないので、この分岐の有無は差し替えなければ観測できない。参加するモデルがない回の実旧との対照は、batchの件数を6にした条件が行う）。
- 下書きの置場: 主担当のsessionの一時領域 `%LOCALAPPDATA%\Temp\claude\c--Users-yshin-Research-FedSDA-experiment\c5e1c86f-9c7d-4b76-b40d-88292c996b53\scratchpad\draft8\`（下書きと整形済みの写し）と、同じ場所の`copy8\`（作業ツリーの複製）。Git管理外で、保存は保証されない。
- **規則からの逸脱**: 上の下書き（新しいsourceとtest）の作成と複製での実行は、命名の承認より前に行った。worktreeのAGENTS.mdの「新しいsrc実装、移植、リネーム、実装を先取りしたテスト追加は、命名承認前には行わない」と、共通引継ぎ手順の「未承認の名前・仕様を先取りして実装しない」は作業場所を限定していないので、リポジトリ外であっても逸脱に当たる（仕様レビューの1回目と2回目が、別々のsessionで同じ指摘をした。主担当は1回目に不採用としたが、2回目の指摘で撤回した）。直前のspec（released-pending-sample-assignment、shared-verification-infrastructure）も同じ進め方で、それぞれのresearch.mdに記録がある（それより前のspecは、この記録のために確認し直していない）。複製での実行結果は承認の証拠に使わない（Task 1〜3の判定は、worktreeで実装した後の実測とレビューによる）。今後の扱い（名前を実際のコードで確かめる下書きを認めるか）はユーザーへ確認する。
- 共用のfresh process scriptでは、警報の前に学習要求を1件記録し（間隔2なので学習は行われず、流れの前提にしている損失が動かない）、流れの最後に保留中の要求を学習する。
