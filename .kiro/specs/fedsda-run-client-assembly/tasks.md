# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. clientの設定の束
- [x] 1.1 置き場所のない値の型と、clientの設定の束を実装する
  - 数値と文字列の値を、既存の検査の仕組みで宣言する型にまとめ、機能別の設定・賭け率と合わせた束にする。型、範囲、賭け率、3組の等値（Fixed-Shareの時間尺度と保留の容量、最小改善量と早期終了の最小改善量、ローカル学習と候補の学習のbatchの件数）を、生成時に確かめる。
  - 新しいmoduleの依存の許可集合を、既存と同じ形で登録する。
  - 完了: 束の単独のtest（各値の境界、別の型、賭け率の不正、等値の違反、frozenを回避した不正の拒否）と、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunClientScalarSettings、FedsdaRunClientSettings_
  - _Requirements: 5.1, 5.2, 5.3, 6.3_

- [x] 2. clientの組立てと操作
- [x] 2.1 clientの組立てを、実旧の生成直後の状態との対照つきで実装する
  - 実`_pretrain_initial_model`と実`__init__`で実旧の最終構成のclientを作り、新側を同じ値の初期モデルから組み立てる対照testの土台を先に作る（旧の設定の差し替えを1箇所にまとめる）。
  - 組立ての引数の検査、初期モデルの写し、全ownerの生成、ownerの記録とclientの生成を実装する。
  - 完了: 生成直後の全状態が実旧と一致するtest（2値・多クラス）、組立ての拒否のtest（各不正で、渡した初期モデル・optimizerの状態・乱数が変わらない）、写しの独立と乱数を使わないことのtestが成功する。
  - _Boundary: assemble_fedsda_run_client、FedsdaRunClientOwners_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.4, 5.5_
- [x] 2.2 clientの5つの操作を実装する
  - 観測標本の変換と標本1件の処理の呼出し、ラウンド境界での保留中の学習、登録できるモデルの有無、送信待ちの進行、終端での未完了の候補検証の回収（概念IDの保持の解除を含む）を実装する。新しいmoduleの依存の許可集合を登録する。
  - 完了: 対照testが、2値・多クラス、複数の標本列で、標本ごと・ラウンド境界ごと・終端の後に、全状態と乱数の一致を示し、要求4.2の経路をすべて通ったことを確かめるtestが成功する。操作の入力の拒否のtest、概念IDなしのtest、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunClient_
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 5.6, 6.3_

- [x] 3. 実行の枠への接続
- [x] 3.1 組み立てたclientを、実行の枠の参加者として動かす
  - 組み立てたclient 2つと、何もしないサーバの代役で参加者を作り、参加者の検査と区間の進行を実行するtestを追加する（sourceの変更が要らないことを確かめる）。
  - 完了: 参加者の検査が通り、区間の進行が最後まで実行され、実行の記録の段の列が契約の順で、各clientの予測の記録の件数が処理した標本数と同じである。
  - _Depends: 2.2_
  - _Boundary: FedsdaRunClient（実行の枠との適合）_
  - _Requirements: 6.1_
- [x] 3.2 共用のfresh process scriptへ、clientを組み立てて実行の枠で進める流れを足す
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、clientを組み立て、区間の進行で標本列を最後まで進め、予測の記録の件数と、警報が1回以上起きたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.2_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.2_
  - _Requirements: 4.1, 6.1, 6.2, 6.3_

## Implementation Notes

- 手順からの逸脱: 2.1・2.2・3.1は、1つのmoduleと1つのtestファイルなので、まとめて1つのcommitにした（`3b7567b`）。clientのsourceの下書きの後にtestを書いた（testを先に書く手順からの逸脱）。
- 仕様を書いた後の変更: 束が「同じ値」と確かめる組が、2組から3組になった（旧は、ローカル学習と候補の学習のbatchの件数に、同じ`CLIENT_BATCH_SIZE`を使う）。終端の回収で、候補検証を保持しているのに概念IDを保持していない状態を、何も変えずに拒否するようにした。どちらも要求・設計へ反映済み。
- 対照の標本列: 自作の2概念の標本列では、採用・再利用・候補検証中の警報に達しなかったので、実旧のデータ生成（概念が4つ）で標本列を作り、候補検証の標本数と標本数を変えた12条件にした。候補検証の確定での他モデルの再利用（`post_alarm_validation_held_model_reused`）は、この対照では通っていない（標本1件の処理の対照が通している）。
- 旧は、送信保留のモデルのIDを持たない（採用の後に現行モデルが別のモデルへ戻ると、送信保留のモデルと現行モデルが違う）。対照testは、新の送信保留のモデルIDを、「保有している一時IDのモデルである」ことで確かめている。サーバ同期のspecで、正式IDの確認のときの扱いを確かめる（再開案内の「次の候補」）。
- 独立レビュー1回目（2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。初期状態の写しとoptimizerの状態、複数部品の接続なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `5b3126b6-3c65-417c-b28a-e83d9b285039`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`72aea4d..7ed2364`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、Minor 3件、任意 1件）。対象test（182 passed）、依存境界とfresh process（3044 passed）、Ruff、共用script、`spec_checks.py names`を独立に実行した。
- 指摘の採否（`053f221`）: (1)Major: 初期モデルのIDの範囲を確かめていない（負のIDで組立てが通り、評価標本が黙って捨てられる）→採用。負を`ValueError`で拒否し、testの場合と、要求5.4・設計の範囲を足した。(2)Minor: 最終構成で旧が更新するのに記載のない診断（`backbone_gradient_diagnostics`、`provisional_model_decisions`）→採用。要求の範囲外と設計の「旧と違う点」へ書いた。(3)Minor: docstringの「全部の検査の後」は不正確（分類器のパラメータの型は、登録が写しに対して確かめる）→採用。docstringと設計を直し、写しに対する拒否のtestの場合を足した。(4)Minor: 対照testが候補検証のsessionの内部を標本ごとに比べない、網羅のtestが全条件を実行したときだけ確かめる→不採用。sessionの内部は上流のspecの対照testが実旧と照合しており、本specの対照は、確定後の保有モデル・学習データ・適応記録を標本ごとに照合する。網羅のtestの形は、既存の標本1件の処理のtestと同じ。(5)任意: 契約は`-> None`だが、2操作は値を返す→変更しない。clientは契約をimportせず、構造的に満たす（実行の枠の検査と区間の進行を通すtestで確かめた）。Pyrightは0 errors。
- 独立レビュー2回目（Majorの修正の確認。別session）: Claude Haiku 5.5（同上）。session `de43fbf2-24c7-4c5a-bf1b-be02fb5024f6`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`7ed2364..053f221`。判定は`IMPLEMENTATION: APPROVED`（指摘なし。任意の注記2件: float64の場合のtestが例外の文言を確かめていない、helperが共有部もfloat64にしている。どちらも変更していない）。対象test（184 passed）とRuffを独立に実行した。指摘4の不採用の理由のうち、「標本ごとに照合している」ことは、oracleの比較箇所を読んでいないので検証していない、と申告している。
- 検証（Windows基準環境、`053f221`）: 全pytest 10720 passed / 3 skipped / 2 warnings、exit 0（前spec 10536＋束 131＋client 53）。JUnitは10723 testcase、failure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtestはskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。source hash（303パス）は`f4078be9437605e03b9cfa29ab175715435d9bc2ad0f1efb46e25a6b3605c56e`。レビュー前の`7ed2364`でも全pytestを実行した（10718 passed / 3 skipped）。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 1回目のレビュー担当が、作業の途中で、旧ファイルの写し3つ（`legacy_fedsda.py`・`legacy_shared.py`・`legacy_experiment.py`）を`venv/refactoring-tests/claude/`（worktreeの外、Git管理外、このPCだけ）へ作った。削除していない。2回目の依頼文では、写しを作らないことを明記した。
- 未検証・残る制約: サーバなしの対照である（正式IDの確認、モデルの配布、集約後の再較正を通っていない）。初期モデルは、対照testでは実旧の事前学習の結果を写して作った（新の事前学習は未移植）。真の概念IDを渡さない経路は、実旧と対照できない。新全体runのgolden一致は範囲外で、未確認。WSLでは実行していない。

