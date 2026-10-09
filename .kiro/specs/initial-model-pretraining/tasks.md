# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 事前学習
- [x] 1.1 事前学習の条件の設定型を実装する
  - 標本数、epoch数、batchの件数を、既存の検査の仕組みで宣言する。依存の許可集合を登録する。
  - 完了: 設定のtest（各値の境界、型と範囲の不正）と、依存境界のsuiteが成功する。
  - _Boundary: InitialModelPretrainingSettings_
  - _Requirements: 3.1, 4.2_
- [x] 1.2 事前学習の関数を、実旧の事前学習との対照つきで実装する
  - 実`_pretrain_initial_model`と新の関数を、同じseedで実行して照合する対照testを先に書く（2値・多クラス、標本数・epoch数・batchの件数の組、optimizerの種類）。
  - 引数の検査、分類器と2つのoptimizerの状態の生成、標本の生成、epochごとのshuffleとbatchごとの更新、統計の逐次更新、結果の記録を実装する。依存の許可集合を登録する。
  - 完了: 対照testが、全条件で、全パラメータ・2つのoptimizerの状態・損失統計・実行後の3つの乱数の状態の一致を示す。拒否のtest（各不正で、3つの乱数の状態が変わらない）、結果の形のtest、依存境界のsuiteが成功する。
  - _Boundary: pretrain_initial_model、PretrainedInitialModel_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 3.2, 3.3, 4.2_

- [x] 2. clientの組立てへの接続
- [x] 2.1 事前学習の結果からclientを組み立て、実旧のclientと照合する
  - 事前学習の結果を`assemble_fedsda_run_client`へ渡し、実旧の事前学習の結果から実`__init__`で作った実旧のclientと、生成直後と、続く標本列・ラウンド境界の処理の後に照合するtestを追加する（sourceの変更が要らないことを確かめる）。
  - 完了: 生成直後と、標本列の処理の後の全状態と乱数が、実旧と一致する。
  - _Depends: 1.2_
  - _Boundary: pretrain_initial_modelとassemble_fedsda_run_clientの接続_
  - _Requirements: 2.2_
- [x] 2.2 共用のfresh process scriptの、clientの流れの初期モデルを、事前学習で作る
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、事前学習→clientの組立て→実行の枠での標本列の処理、を最後まで行って成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Boundary: 共用のfresh process script_
  - _Requirements: 4.1_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 2.1, 4.1, 4.2_

## Implementation Notes

- 手順からの逸脱: 1.1・1.2・2.1は、1つのtestファイルにまとめたので、1つのcommitにした（`cd38743`）。sourceの下書きの後にtestを書いた（testを先に書く手順からの逸脱）。
- 独立レビュー1回目（2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。NNの数値と乱数の順序の移植なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `e1b4caaf-923e-4614-833f-610184f4ac15`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`c14d733..a323a06`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、Minor 2件）。対象test（98 passed）、依存境界とfresh process（3044 passed）、Ruff、共用script、`spec_checks.py names`を独立に実行した。指摘の数値と挙動は、コードの読みに基づく（許可したコマンドでは確かめられない）と申告している。
- 指摘の採否（3件とも採用、`197ebb1`）: (1)Major: 旧の既定の設定は最後のbatchが20件になるが、対照に20件のbatchがなく、共同更新の「損失×件数÷件数」が旧の更新と違いうる→対照を広げた。旧の既定の設定（500標本・10 epoch・batch 32）と、batchの件数1〜33のすべてで、実旧と全パラメータ・2つのoptimizerの状態・損失統計が一致した。sourceの更新の方式は変えていない。34件以上のbatchは照合していない。(2)Minor: 隠れ層なしが、分類器の生成の後で拒否される→引数の検査で拒否する。(3)Minor: 勾配の計算が無効のとき、分類器と標本の生成の後で拒否される→epoch数が1以上なら、引数の検査で拒否する（epoch数0は受け入れる）。
- 独立レビュー2回目（修正の確認。別session）: Claude Haiku 5.5（同上）。session `c1d5606a-b375-4dce-b778-8bec2db0087c`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`a323a06..197ebb1`。判定は`IMPLEMENTATION: APPROVED`（指摘なし。任意の注記1件: 件数ごとのtestが、失敗したときに件数を示さない。変更していない）。対象test（110 passed）、依存境界、Ruffを独立に実行した。
- 検証（Windows基準環境、`197ebb1`）: 全pytest 10830 passed / 3 skipped / 2 warnings、exit 0（前spec 10720＋事前学習 110）。JUnitは10833 testcase、failure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。source hash（306パス）は`4d6404d5773a3bc87a9b1cb1fdf754731a7e2cfbab91c1f4b54afea23be2a06c`。レビュー前の`a323a06`でも全pytestを実行した（10818 passed / 3 skipped）。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 対照は、SINE（2値のラベル）だけである（多クラスは、クラス数4の分類器へ0/1のラベルを与えて確かめた）。34件以上のbatchの更新は、実旧と照合していない。全clientとサーバの準備（`prepare_run`）へはつないでいない。WSLでは実行していない。

