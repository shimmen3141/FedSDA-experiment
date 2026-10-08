# 候補検証の確定に伴う診断通知 — 調査記録

2026-10-09、主担当Claude Code。基点commit `4b2fc15`。固定旧基準`748c3aa`。実行環境はWindowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 旧処理の事実（`federated_drift_experiment/clients/fedsda.py`）

`_set_local_current_model`（84〜89行）は、IDが変わるときだけ`_on_local_model_change`を呼ぶ。最終構成のclassでは、これがAdaHedgeの再始動（2052〜2062行）になる。呼出し箇所は次のとおり（ファイル内のgrepで確認）。

- `_resolve_drift`の再利用の分岐（828行）。alarm-occurrence-handlingで接続済み。
- `_resolve_drift`の即時作成の分岐（890行）。最終構成では通らない（候補検証を開始する方針のため）。
- `_finalize_forward_validation`の3箇所: 採用（312行）、shadowが勝った場合（321行。移植しない。[UNPORTED-002](../../../docs/research/implementation-findings/unported-002-reference-shadow-tournament.md)）、再適合した参照（332行。他モデルなら切替、現行なら同じIDなのでhookなし）。棄却の分岐は呼ばない。
- `finalize_incomplete_forward_validation`は呼ばない。

したがって、候補検証の確定で再始動が起こるのは、採用と他モデルの再利用の2つ。

新実装の確定（`runtime/post_alarm_candidate_validation_resolution.py`）は、結果の`training_model_assignment_change`に、採用と他モデルの再利用では変更を、棄却と現行の維持ではNoneを入れる。既存の通知（`notify_diagnostics_of_training_assignment_change`）は、変更がNoneまたは同じIDなら何もせず、違うIDならglobalの診断証拠を1回再始動する。

## 新実装の現状（変更前）

`advance_held_candidate_validation`は、上流の進行→（確定なら）適応記録→保持の解除を行い、通知しない。警報のときの通知だけが`handle_alarm_occurrence`で接続されている。このままだと、候補の採用と検証後の再利用で、診断証拠の再始動が旧より少なくなる。

## 判断

- **既存の進行の関数へ足す**（別の関数で包まない）: 設計1節。通知のない入口を残さない。
- **診断のownerの型を上流の前に確かめる**: 通知は上流の更新の後に呼ばれる。保持がないときも確かめる（不正なownerを、候補検証が始まるまで見逃さない）。
- **通知は解除の後**: 設計2節。警報のときと同じ並び。
- **NEW-001の文言を同時に直す**: このファイルを次に変更するときに直すと記録していた。挙動は変えない。
- **終端回収は変えない**: 学習帰属を変えない。

## oracleの実行可能性

既存の到達時のtest（`build_validation_progress_oracle`、実旧`_observe_forward_validation`）の実旧clientは、帰属変更を記録するだけのhookを持つ（上流の照合helperがその記録を読む）。その記録を残したまま、実旧`RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change`を同じclientに対して呼ぶhookを置き、実`AdaHedgeRouter`を与える（alarm-occurrence-handlingで実行済みの方法）。再始動が観測できるよう、確定の前に新旧へ同じ損失を1回与える。リポジトリ外の下書きを作業ツリーの複製で実行し、2/4class×4条件が成功することを確かめた。

## 手順上の事実

- 命名の事前登録のため、sourceとtestの変更をリポジトリ外のpatchとして下書きし、`git archive HEAD`で作った作業ツリーの複製へ適用して、Windowsの基準環境のPythonで実行した（worktreeは変更していない。対象47件、依存境界と合わせて成功、Ruff・Pyright成功）。
- 既存のtestの変更点: 進行の呼出しへ診断のownerを渡す。ownerの型の拒否のtestは、操作と不正にする引数の組を明示する形へ改めた（進行だけが診断のownerを受け取るため。受け取らない引数を渡したときのTypeErrorを拒否と取り違えないよう、例外の文言で対象の引数を確かめる）。派生型の条件を足した。
