# Implementation Plan

- [x] 1. クロス評価のモデルの送り方を選べるようにする（既定は、まだ変えない）
  - 通信量の記録のtest（2つの値の全体runで、通信量以外が同じ、下りのbyteが新しい扱いの式どおり、クロス評価をしないラウンドは同じ）と、設定の検査のtestを先に書き、REDを確かめる。
  - 参加者の設定の新しい項目、サーバへの受け渡し、`synchronize_models_in_server_round`の分岐、クロス評価と配布の、通信量を記録するかどうかの引数、配布の通信量を記録する関数。全部の参加者の設定の組立て（最終構成の既定、test、fresh script）で、旧の扱いを明示する。旧と比べる入口と、基準版1〜7の「後に足された項目」（参加者の設定）の表。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜7）が値を変えずに通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2_

- [x] 2. 最終構成の既定を替える
  - 既定の値と、全FedSDA構成が引き継ぐことのtestを先に書き、REDを確かめる。最終構成の既定を替え、既定に依るtest（設計の表）を、失敗したものから合わせる（実旧と比べるtestの期待値は変えない）。
  - 完了条件: 構成のtestが通り、基準版1〜7と実旧との照合が値を変えずに通る。
  - _Requirements: 2.1, 2.2_

- [x] 3. 基準版8を足す
  - 版8の定義と、版7→8の差の説明のtestを足す。sourceとtestをcommitしてから、`--baseline-version 8`で、Windowsの基準環境とWSLで生成して足し、照合する。現在の版を8へ替えるのは、goldenを置いた後。
  - 完了条件: 版1〜8の照合と、版7→8の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2_

- [x] 4. 文書を合わせる
  - 実験の入口、指標（通信量の定義）、基準（版8）、最終提案構成（サーバの1ラウンドの順）、IMPROVE-020、再開案内。
  - 完了条件: 文書が新しい扱いと、過去の結果の通信量の扱いを説明する。
  - _Requirements: 5.1_

- [x] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL（goldenの版1〜8と、このspecのtest）。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合の結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 仕様レビュー1回目（Luna high）: Major 1件（統合は共有部を含む全パラメータを加重平均するので、共有部がビット単位で変わりうる。送り直しの対象を保証していない）、Minor 1件（代表の損失統計の扱い）。統合の後に配布する共有部が先に送った値とビット単位で違えば共有部も送り直したと数える、損失統計は送り直す内容に含めbyteは数えない、と設計・要求を直した。あわせて、ID対応が空でないことと統合が起きたことが同じであることを、コードで確かめて注記した。2回目（再確認）PASS。
- Task1: 新しいtestを先に書き、新しい名前が無いことによる収集の失敗をREDとした。全体runのtestは、goldenの`fedsda-sine2-adoption-without-validation`（版7）の条件で、旧の扱いのラウンドごとの記録（クロス評価の下り、統合の後の配布の下り、先の配布の量、代表と共有部の送り直し）から新しい扱いの下りのbyteとモデルの転送回数を計算し、新しい扱いの実測と一致することを確かめた（通信量以外の記録の表・指標・最終パラメータも一致）。共有部の送り直しは、部品のtestで、変わる場合と変わらない場合を確かめた。関数の新しい引数（`record_model_transfers`、`record_communication_volume`、`cross_evaluation_model_transfer_policy`）は、既存の呼出しを変えないよう、旧の扱いを既定値に持たせた（設計では必須の書き方をしていなかった。参加者の設定とサーバの組立ての項目は必須）。依存境界に、新しくimportした4つの名前を登録した。全pytestで、runの識別子の期待値2件と依存境界の1件だけが失敗し（どちらも意図した変化と登録の漏れ）、基準版1〜7と実旧との照合は値を変えずに通った。
- Task2: 既定と全FedSDA構成の値のtestのREDを確かめてから既定を替えた。runの識別子`52189bd930df`→`d34424af3358`。全pytest 14517 passed / 8 skipped / 失敗0。
- Task3: 生成元`29c6406`。Windows（Python 3.13.15）とWSL（Python 3.14.4）で、独立2processの全snapshot一致で21ケースを生成し、`v8/`へ新規に置いた（旧版は上書きしていない）。設置後に`BASELINE_VERSION`と`BEHAVIOR_BASELINE_VERSION`を8へ上げた。版7→8の差の説明のtestは、全21ケースで通信量以外が同じ、クロス評価が起きたFedSDAのケースで下りが減ることを確かめた。Windowsの版1〜8の照合と来歴のtest 304 passed、WSLの版1〜8とこのspecのtest 320 passed（`../../venv/refactoring-tests/cross-eval-wsl.xml`）。
- Task4: 指標（通信量の定義と、過去の結果の通信量が旧の扱いであること）、最終提案構成（サーバ同期の順）、実験の入口、基準（版8）、IMPROVE-020（採用と実装の経緯）、再開案内を更新した。
- 実装レビュー1回目（Luna high、session `01a156c1-2df6-72e3-a53f-e73100b90745`、対象`dffd54f..9f6d15b`）: CHANGES REQUIRED。Major 2件（共有部の比較が`torch.equal`の値の比較で、符号付きゼロとNaNでビット単位にならない。全体runのtestの期待値が、実装と同じ配布の量の関数を使っていた）、Minor 1件（命名表に、公開した検査の関数と値の名前の定数が無い）。全部採用し、dtype・形・uint8のbyte列で比べる関数にし、符号付きゼロとNaNのtestを足し、期待値を分けたパラメータから独立に計算して、実装の関数の値がそれと一致することも確かめる形にした。修正の後も、版8のgolden（21ケース）の値は変わらない。2回目（session `01a15724-aee6-72a1-ae53-b4d382bad72c`、修正の再確認）PASS。レビュー担当は、1回目に対象test・依存境界3156件（1件はsandboxの一時ファイルの権限エラー）と版8ほか10件を独立に実行し、fresh smokeは権限エラーで実行できなかった（環境制約の例外）。2回目は対象test 13件。全pytest・WSL・Pyrightは独立に実行していない。
- 主担当の最終検証（`f7d0e7f`、source hash `82f39ed95a19fd4d8fccb4adfd6ad6932f78a4253c22aa3a9ef47f6e21839328`、498パス）: Windows全pytest 14543 passed / 8 skipped / 失敗0（2314秒、JUnit `../../venv/refactoring-tests/cross-eval-full.xml`）。旧の回帰2件も成功。fresh smoke 16フロー成功（`../../venv/refactoring-tests/cross-eval-fresh.log`）。Pyright 0 errors、pip check・Ruff成功。WSLの版1〜8とこのspecのtestは320件成功（Task3）。`tools/research_*`を除いた旧実装・旧goldenの差分は0行。基準版のgoldenは`v8/`の2ファイルの追加だけ。
