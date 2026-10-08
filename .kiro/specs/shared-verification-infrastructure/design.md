# 検証基盤の整理 — 設計 revision1

## 1. Boundary Commitments

- `tests/refactoring/fresh_process_smoke.py`: 新実装だけをimportして流れを実行するscript。test moduleではない（pytestは収集しない）。
- `tests/refactoring/test_fresh_process_smoke.py`: 上のscriptを別processで実行するtest。
- `tests/refactoring/test_single_run_dependency_boundaries.py`: 許可集合を読むhelperとtestを1つずつ追加。使っていない許可を1件外す。
- `tests/refactoring/test_held_candidate_validation_progress.py`: 拒否条件と順序の確認を追加。

Out of Boundary: `src/`、既存の注入契約testの整理、依存の登録方式の一本化、新しい接続。

## 2. 共用のfresh process script

- 置き場所を`tests/refactoring/`にする理由: Ruffの対象で、Git管理下で、test（同じdirectory）から場所を決められる。名前が`test_`で始まらないのでpytestは収集しない。
- `src`を`sys.path`の先頭へ自分で足す（pytestの設定に依らず単独で実行できる）。importは`federated_learning_experiments`と標準library・torchだけ。
- 流れは、2026-10-09までの個別script（alarm-occurrence-handlingとheld-candidate-validation-diagnostic-notificationのもの。Git管理外）を1つにまとめたもの。各流れは`run_smoke_scenario`が実行し、食い違いはAssertionErrorにする。`main`が2/4 class×全流れを実行し、最後に、必要な結果（警報の5種類と確定の4種類）をすべて観測したことと、`sys.modules`に旧実装・test moduleがないことを確かめて、目印の1行を出力する。
- test側は、`sys.executable`でscriptを実行する。環境変数は引き継ぐが`PYTHONPATH`は外す。終了codeが0であることと、目印の行が出力にあることを確かめる。失敗時は子processの出力をassertの説明に出す。
- 以後の運用: 新しい接続を移植するspecは、その接続を通る流れをこのscriptへ足す（共通引継ぎ手順）。新全体runを接続したら廃止する。

確かめないこと: 実旧との一致（各部品の対照testの役割）。このscriptは、新実装が旧実装なしで最後まで動くことと、ownerの状態の整合だけを見る。

## 3. 許可集合が広すぎないことの検査

- 既存の登録は、test fileの`dependency_is_allowed`の中に、`if source_module_path == "<module>": return imported_module_name in (<名前>, ...)`の形で書かれている。この形の分岐を、test file自身のASTから読む（登録を別の場所へ移さない）。
- moduleごとに、実際のsourceのimport文から名前の集合を作る: 既存の`resolve_imported_module_names`の結果に、`from m import a`の`m`と`m.a`を足したもの（検査の方式がmoduleによって違い、許可集合に書かれる名前が「module名」「module名.symbol」のどちらの場合もあるため、両方を含める）。
- 許可集合の各名前がこの集合に入っていなければ、使っていない許可として集める。先頭が`__future__`の名前は対象外（依存先ではない）。全moduleぶんを集めてから、空であることを1回assertする（失敗時に全件が見える）。
- 読取りの空振りの検査: 読めたmoduleの数が、現在の登録数以上であること。
- この検査で見つかった1件（`candidate_parameter_initialization.py`の`torch.Tensor`。sourceは`import torch`だけを使う）を許可から外し、その書き方（`from torch import Tensor`）を許可例のtestから除く。

確かめないこと: 上の形で書かれていない分岐（層ごとの規則、条件つきの許可）。それらは対象外で、既存のtestのまま。

## 4. NEW-002の修正

- 派生型の拒否: 応答は、同じfieldを持つ派生型の値を作って拒否条件へ足す（exact型の検査だけが拒否する入力）。保持のownerは、別の型と、同じsessionを保持した派生型の両方を、例外の文言で対象を確かめて拒否させる。拒否の後、保持が不変であることを確かめる。
- 終端回収の順: 進行のtestと同じ方法（解除を記録用wrapperで包み、その時点の適応記録の件数を記録する）で、解除の時点で記録が追加済みであることを確かめる。
- あわせて、ownerの型の拒否のtestのコメントを、mockへ差し替えた条件に合う説明へ改める。

## 5. Testing Strategy / Requirements Traceability

| 要求 | 証拠 |
| --- | --- |
| 1.1, 1.3 | scriptを単独で実行して成功すること。流れの一覧と、各段の後の確認はscriptの内容 |
| 1.2 | 全pytestに含まれるtestが別processで実行して成功すること。検出力の確認として、scriptへ一時的に旧実装のimportを足す・流れを1つ外す・sourceの進行の関数の引数名を変える、のそれぞれでtestが失敗すること（確認後に元へ戻す） |
| 2.1, 2.3 | `torch.Tensor`を外す前はtestが失敗し、外した後は成功すること。検出力の確認として、任意のmoduleの許可集合へ使っていない名前を1つ足すと失敗すること |
| 2.2 | 検出力の確認として、読取りの条件を満たさないよう一時的に変えると失敗すること |
| 3.1, 3.2 | 汎用の変異tool（`mutation_check.py`）で、保持と進行のmoduleの3関数の未検出が、等価と判断済みの変異だけになること |
| 4.1 | `git diff`で`src/`・旧実装・goldenの差分が空、全pytest、Ruff |

未検証として残す: 新全体runのgolden一致。依存testの方式の一本化（行わない。IMPROVE-009）。
