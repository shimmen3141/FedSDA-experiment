# Design Document

## Overview
`ModelTrainingSampleStore`が、モデルごとに最新の区切り（保存の列の位置）を持つ。区切りは、警報の後に同じモデルで続けると決まった2か所の処理が、標本を足す直前に付ける。snapshotの`ModelTrainingSampleCollection`が区切りを運び、抜き出しの関数が、2つの新しい抜き出し方で使う。設定（`LocalTrainingSettings.training_sample_selection_policy`）の許す値と、掃引の軸の書き方を足す。

## Boundary Commitments
### This Spec Owns
- `learning/training/model_training_sample_records.py`: `ModelTrainingSampleCollection`へ`latest_alarm_boundary_position: int | None = None`。
- `learning/training/model_training_sample_store.py`: 区切りの記録、IDの付け替え・統合での扱い、snapshot。
- `runtime/alarm_change_interval_resolution.py`、`runtime/post_alarm_candidate_validation_resolution.py`: 区切りを付ける。
- `learning/training/held_model_training_batch_sampling.py`: 2つの新しい抜き出し方。
- `learning/training/local_training_settings.py`: 許す値と組合せ。
- `sweep/experiment_sweep_plan.py`: 軸の書き方（解析と表示）。
- `sweep/experiment_sweep_command.py`: 引数の説明の文（今の3つの書き方を並べているので、5つにする）。
- `sweep/experiment_sweep_records.py`: 一覧の表の列（表示の関数を通るので、新しい書き方が表に出ることをtestで確かめる）。
- 文書。
### Out of Boundary
FedDrift（同じ保存と記録の型を使うが、区切りを付けない。記録の新しい項目は既定のNone）、ほかの区切りの付け方、既定。
### Allowed Dependencies
新しいmoduleは無い。2つのruntimeのmoduleは、すでに保存の型を読んでいる（新しいimportは無い）。
### Revalidation Triggers
保存のsnapshotの形（記録の項目が1つ増える）、抜き出しの関数、2つの警報の後の処理、FedSDAのrunの識別子（許す値の追加だけでは保存の形は変わらないので、既定の識別子は変わらない）。

## Architecture
### 記録（`ModelTrainingSampleCollection`）
`latest_alarm_boundary_position: int | None = None`を足す（記録の型なので、設定の規約——既定値なし——の対象外。FedDriftと既存の作る箇所を変えないため既定値をNoneにする）。Noneでなければ、0以上かつ`len(training_samples)`以下のbool以外のint（`__post_init__`で検査。今の記録は`__post_init__`を持たないので、足す）。

### 保存（`ModelTrainingSampleStore`）
- `_latest_alarm_boundary_position_by_model_id: dict[int, int]`を持つ。
- `count_model_training_samples(*, model_id) -> int`: そのモデルの列の標本の数を返す（列が無ければ0）。区切りの位置を、吸収の前に読むために使う。
- `mark_latest_alarm_boundary(*, model_id, position)`: 指定の位置を、そのモデルの最新の区切りとして記録する。そのモデルの列があり、位置がbool以外のintで、0以上かつ列の長さ以下であること（違えば拒否し、何も変えない）。吸収の後に付けるので、列はある（列の無いモデルへ区切りだけを残さない）。
- `reassign_model_training_samples_id`: 元のIDの列があるときだけ、区切りも新しいIDへ移す（元のIDの区切りは消す。先のIDの区切りは、元のIDの区切りで上書きし、元のIDに区切りが無ければ消す——列の上書きと同じ扱い）。元のIDの列が無いときは、区切りも含めて何も変えない（今の列の扱いと同じ）。
- `remap_model_training_sample_collections`: 写像の先ごとに、つながった元の列の数（`mapping.get(元のID, 元のID)`がその先に一致し、かつ保存に列がある元のIDの数。代表自身への恒等の対も1つと数え、列の無い元のIDは数えない）を数え、0なら何もせず、1つなら区切りを移し、2つ以上なら区切りを消す。
- `snapshot_ordered_model_training_samples`: 記録へ区切りを入れる。

### 区切りを付ける処理
- `resolve_alarm_change_interval`: 選ばれたモデルが今のモデル（`current_training_model_assignment.current_training_model_id`と同じ）なら、区間の標本を吸収する前に、そのモデルの列の長さを`count_model_training_samples`で読み（列が無ければ0）、吸収が成功した後に、その長さを区切りとして付ける（`mark_latest_alarm_boundary(model_id=..., position=...)`。位置は0以上、今の列の長さ以下）。吸収が例外を出したら、区切りは付かない。
- `apply_post_alarm_candidate_validation_resolution`: 候補が退けられた（`candidate_rejected`。吸収先は検証の前の今のモデル）とき、または参照のモデルが今のモデルと同じ（`assign_model_for_training`がNoneを返し、結果が`current_model_maintained`）とき、同じ形で付ける。吸収の前に吸収先の長さを読み、吸収が成功し、吸収先が今のモデルのまま（後者は`assign_model_for_training`の後で確かめる）のときだけ、その長さで区切りを付ける。

### 抜き出し（`sample_training_batches_for_held_models`）
- 振り分けは、`training_sample_selection_policy`の名前で行う（今の実装は、Kとαの値の有無で振り分けているので、αを持つ`recency_decayed_before_latest_alarm_boundary`が、今の指数減衰の経路に入ってしまう。名前で振り分け直す）。区切りの位置は、各モデルの記録（`latest_alarm_boundary_position`）から渡す。今の`recency_decayed_stored_samples`は、区切りを使わない今の経路（区間の方法の下請け）のまま。
- 乱数の前の検査に、区切りの型と範囲（Noneか、bool以外のintで、0以上かつ列の長さ以下）も含める。対象は、渡された全部の列（抜き出しの対象でない列も含む）。区切りの範囲の検査は列の長さを使うので、標本の列がtupleであることの検査を、今の保有モデルだけから、全部の列へ広げ、その後に置く。記録の`__post_init__`の迂回に備える。
- `uniform_after_latest_alarm_boundary`: 区切りがあれば、`start = min(区切り, 列の長さ − batchの件数)`、`python_random_generator.sample(training_samples[start:], batch_sample_count)`。区切りが無ければ全部から。
- `recency_decayed_before_latest_alarm_boundary`（α）: 末尾から何件前かをaとし、区切りより後の件数を`f = 列の長さ − 区切り`（区切りが無ければ`f = 列の長さ`）とする。重みは、`a < f`なら1、`a ≥ f`なら`α^(a − f + 1)`。1件ずつ、`random()`を1回使って選ぶ: 未選択の平らな位置の数`c`、減衰の範囲（[f, 列の長さ)）の未選択の区間の重みの和を、減衰の範囲の未選択の最新の位置`d0`を1とする相対値で`R`、実の倍率`s = α^(d0 − f + 1)`とする。`c > 0`なら、`s`が正規化数の最小値より小さいときは0とみなす（減衰の範囲は選ばれない。倍精度で表せないほど小さい確率）。全体`c + s·R`に対する`t = u·(c + s·R)`で、`t < c`なら平らな範囲の未選択の位置の`floor(t)`番目（新しい順）、そうでなければ`(t − c)/s`を減衰の範囲の相対の残りとして、held-model-training-sample-recencyの区間の方法で選ぶ（`s`が0へ丸められれば、減衰の範囲は選ばれない）。`c = 0`なら、減衰の範囲だけを相対の重みで選ぶ（区切りが列の末尾のとき——区切りより後の件数f=0——、または平らな範囲を選び尽くしたとき）。区切りが0のときは、全部の位置が平らな範囲（f=列の長さ）で、一様と同じ。減衰の範囲に未選択の位置が無いとき（区切りが無いとき、または減衰の範囲を選び尽くしたとき）は、`R = 0`とし、`d0`と`s`は使わず、平らな範囲だけで選ぶ。丸めの保護は、held-model-training-sample-recencyと同じ（平らな範囲の番目は、未選択の数−1で上から抑える）。手数は1件あたり選んだ件数に比例する。
- 区間の方法（区間の列、相対の重みの和、区間内の残りと逆関数、丸めの保護）は、今の`_sample_recency_decayed_positions`から、範囲[lo, hi)と基準の位置を引数にとる下請けの関数へ切り出し、今の指数減衰と共用する（今の指数減衰の値と乱数の使い方は変えない。testで確かめる）。

### 設定と掃引の軸
- 許す値へ`uniform_after_latest_alarm_boundary`と`recency_decayed_before_latest_alarm_boundary`を足す。組合せの辞書（`LocalTrainingSettings.__post_init__`の、選び方→値を持つ項目）へも足す: 前者はKもαもNone、後者はαだけ。
- 抜き出しの関数の乱数前の検査も、2つの値を受ける。
- 軸の書き方: `alarm`（区切りより後から一様）、`alarm-decay:<α>`（αの書き方は`decay:<α>`と同じ）。表示も同じ。

## 旧と違う点
既定の動作と値は変わらない。区切りは、既定の抜き出し方でも付ける（保存の状態だけで、学習の値には効かない）。記録の型に項目が1つ増える。

## Validation Design
- 記録: 区切りの範囲の検査。
- 保存: 区切りの記録、IDの付け替えで移る、統合で1つの列なら移り・2つ以上なら消える、snapshotが運ぶ。
- 区切りを付ける処理: 今のモデルを維持したとき・候補を退けたとき・参照に今のモデルが選ばれたときに、吸収の前の長さで付く（列が無いモデルでは0）。ほかの結果（別のモデルの再利用、候補の採用、短い警報、候補の検証の間の警報、runの終わりの検証の打切り）では付かない。吸収が失敗したら付かない（警報の区間の評価の既存のtestの、失敗の前後の状態の比較へ、区切りも入れる）。候補の検証の間に別の警報が来たあと、候補を退けたとき、区切りは検証の窓の標本の後・区間の標本の前に付く（列の順と区切りの位置をtestで固定する）。
- 抜き出し: 区切りより後から一様の母集団（区切りより後がbatchより少ないとき、区切りが無いとき）。区切りより前を指数減衰で、区切りが無い・区切りが0・区切りが列の末尾・途中の4つの条件について、小さな列（例: 長さ4、batch 1と2）の選ばれ方の確率を、要求2.2の重みから全数列挙で計算した値と比べる（乱数の値uを細かい格子で動かして、uの区間の長さから確率を厳密に求める。頻度の許容幅に依らない）。α=5e-324で例外を出さない。区切りより前を指数減衰の、重複なし・件数・条件付き確率（小さな列で頻度が重みの比に近い）・αが極端に小さいとき・乱数の消費が列の長さに依らない・区切りが無いときは全部の重みが1（一様と同じ分布）。今の指数減衰の値と乱数の使い方が、切り出しの前と同じ（切り出しの前に、今の実装で、seed・α・列の長さ・batchの件数の複数の組合せの抜き出しの位置の列を採取し、testへ値として書いてから、切り出す）。
- 設定と軸: 許す値、組合せの拒否、書き方の解析と表示、表の列。
- 回帰: goldenの版1〜5、実旧との照合、全pytest。
- 実装の後に、ユーザーへ確認する項目: 別のモデルの再利用で、そのモデルに残る古い区切りの扱い（再利用の後の標本と、区切りの後の古い標本が、同じ扱いになる）。
- 実行の配線: 共用のfresh process smoke（`tests/refactoring/fresh_process_smoke.py`）に、警報の後に今のモデルを維持して区切りが付き、`uniform_after_latest_alarm_boundary`の抜き出しが区切りより後の標本だけを使う、最小の流れを足す。

## File Structure Plan
- 変更: 上の`This Spec Owns`のsourceと、対応するtest（新しいtest file、保存のtest、2つの警報の後の処理のtest、抜き出しのtest、設定の規約のtest、依存境界）。
- 文書: `docs/experiments/experiment-sweep.md`、研究バックログの7、再開案内。

## Requirements Traceability
| 要求 | 設計 |
|---|---|
| 1.1–1.4 | 記録、保存、区切りを付ける処理 |
| 2.1–2.3 | 抜き出し、設定 |
| 3.1 | 掃引の軸 |
| 4.1 | 回帰 |
| 5.1 | 文書 |
