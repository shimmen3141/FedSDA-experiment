# 候補検証sessionの保持と進行 — 調査記録

2026-10-08、主担当Claude Code。基点commit `5dd639d`。固定旧基準`748c3aa`。直前のspec（candidate-validation-adaptation-recording）はTask 1まで承認済みで、Task 2・3はWindows基準の検証待ち。本specはその記録関数を使う。

## 旧処理の事実（`federated_drift_experiment/clients/fedsda.py`）

`_forward_validation`を読み書きする箇所は次のとおり（リポジトリ全体のgrepで確認）。

- 52行: 初期値None。
- 152行: `_begin_forward_validation`の最後で新しい`ForwardValidationSession`を代入する。呼ばれるのは`_resolve_drift`の新規作成の経路（864行）だけ。
- 741行: `_resolve_drift`の冒頭。Noneでなければ、FIFO全件を現行モデルへ吸収し、`forward_validation_pending`のイベントを記録して終わる。sessionは変えない。
- 171行: `_observe_forward_validation`。Noneなら0を返す。観測し、要求件数に達したら`_finalize_forward_validation`（191行）。
- 355行: `_finalize_forward_validation`の最後。適応イベントの記録（346行）の後でNoneにし、`_on_drift_resolution`を呼ぶ。
- 361行・393行: `finalize_incomplete_forward_validation`。Noneなら何もしない。回収とイベント記録（384行）の後でNoneにする。
- 508行: `process_one_step`で、警報処理の後にNoneなら`_on_drift_resolution`を呼ぶ（通知。範囲外）。
- 1450行: soft routingの有効化で読む（範囲外）。
- `experiment.py` 2284行: 実験の終端で`finalize_incomplete_forward_validation`を呼ぶ。

したがって、sessionを持つのは「候補検証を開始した警報の後」から「到達時の確定」または「終端回収」までで、その間の警報では変わらない。

## oracleの実行可能性

- 警報応答5種類: `test_alarm_adaptation_recording.py`の`build_completed_recording_oracle`が、実旧`_resolve_drift`を実行済みのclientと、新の完了情報（応答を含む）を返す。候補検証中の条件は、1回目の警報で開始したsessionを2回目の警報へ渡す。
- 到達時・未到達: `build_validation_progress_oracle`と`set_scripted_validation_losses`。実旧`_observe_forward_validation`を実行する。未到達は、新は損失収集を空のものへ差し替え、実旧はsessionの観測済み損失を空にして、3標本を観測させる（損失を固定するhelperは実旧sessionの先頭の損失を読むので、空にする前に呼ぶ）。
- 終端: `build_incomplete_validation_finalization_oracle`。実旧`finalize_incomplete_forward_validation`を実行する。

いずれも完了済みspecが同じ形で実行しているoracleで、新しい旧メソッドは使わない。

## 判断

- **holderを独立したownerにする**: 変数1つで足りる内容だが、「進行中のsessionを黙って置き換えない」「空で解除しない」を1箇所で守り、後のclient進行が同じownerを使えるようにする。
- **警報応答の反映は応答（`AlarmBufferResponse`）を入力にする**: 必要なのは結果種別とsessionだけ。完了処理・警報応答の記録との前後は、3つを並べる組立側（後続のclient進行）が決める。
- **候補検証中の応答で同一性を確かめる**: 応答が指すsessionと保持中のsessionが別物なら、応答を作った呼出しが別の状態を見ていたことになる。
- **進行と終端回収は、既存の関数へsessionだけを差し込む形にする**: 引数は既存の関数と同じ名前・同じ役割で、検査も既存の関数に任せる。本処理が検査するのは、自分が直接使うholderと適応記録のownerの型だけ（上流が更新を始める前）。
- **記録→解除の順**: 旧と同じ。どちらも他方の状態を読まないので結果は順序に依らないが、旧の順を保つ。

## 手順上の事実

- 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、作業ツリーの一時複製へ置いて実行した（worktreeは変更していない。対象38件が成功）。
- 実行環境はWSL 2 Ubuntu（Python 3.14.4、torch 2.12.1+cpu）。Windowsの基準環境は、torchの読込みがスマートアプリコントロールでブロックされている。WSLの結果はWindows基準の検証ではない。
