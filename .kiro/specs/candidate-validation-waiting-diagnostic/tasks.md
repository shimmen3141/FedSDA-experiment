# Implementation Plan

- [x] 1. 待機後強制採用の診断介入を検証して実装する
  - テストを先に作り、元評価の保持・強制理由・RNG不変・例外時復元と9/10件境界を確認する。
  - _Boundary:_ 診断介入と既存検証の接続。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 3.2, 3.3_
- [x] 2. 実行・保存・計画表示と実run対照を接続する
  - 不正/既存出力の拒否、読取り専用計画、元V/Oとの全配列・指標・終了RNG一致を確認する。
  - source/script来歴と介入内容を保存し、診断形式を通常結果と分ける。
  - _Boundary:_ CLIと実行・成果物。
  - _Requirements: 2.2, 2.3, 3.1, 3.4, 3.5, 3.6_
- [x] 3. 独立レビュー・品質/回帰検証・実験コマンドと引継ぎを完了する
  - 独立CLIの実装レビューで契約と証拠を照合し、要求対応を確認する。
  - 対象test、依存境界、fresh、全pytestとRuffを実測し、既存goldenを維持する。
  - 計画確認済みの診断実験コマンドを用意し、実行した範囲と未実行を記録する。
  - _Boundary:_ 検証と研究記録。
  - _Requirements: 1.1, 2.2, 2.3, 3.1, 3.4, 3.5, 3.6_

## Implementation Notes
- 2026-10-18: 要求・設計・命名・tasksをユーザーの研究方向に基づいて作成。仕様レビュー待ち。実装/新規学習は未着手。

- 仕様レビュー初回: Major3（同時起動のNPZ上書き競合、JSON来歴schema、RNG対照方法）を採用。専用runディレクトリの原子的予約、全出力field、seed初期化と同process直接対照を設計へ追記。同じレビューsessionへ再確認を依頼。

- 仕様再確認: RNG全状態の範囲が曖昧というMajorを採用。NumPy legacy global RandomStateの全5要素（Gaussianキャッシュ含む）、Python全tuple、torch CPU全byteと直列化方法・同一環境の比較制約を明示した。

- 仕様レビューの全Major解消後に実装着手。未実装stubでNotImplementedErrorのREDを確認後に実装。Python3.10サーバ対応のためExecutorでなく同じspawn/1run契約を持つmultiprocessing.Pool(maxtasksperchild=1)を使用。公開境界・振る舞いは設計通り。
- 対象テストの初回は不足していた既存helper引数class_countを修正。主担当の通常sandboxではpytest一時領域の権限で停止したため、既知の問題を繰り返し回避せず許可済み昇格で検証。

- 独立実装レビュー（初回1回）Luna high PASS、Blocker/Major0、Minor1（dirty判定で未追跡を除外）を採用し除外オプションを削除。Minorのみなので再レビュー不要。独立は3135test成功、I/O6件とfreshは権限で未実行。主担当3141件・fresh成功を照合し、既存の権限例外ルールで判定。全pytestは指摘反映後に主担当が実行する。

- 手元WSLの起動前検証で、Windows作成worktreeの.git内のC:/参照をLinux gitが解決できないことを確認。Git管理ファイルは変更せず、起動shellのGIT_DIR/GIT_WORK_TREEを検証済みLinuxパスへ明示して実行する。サーバの通常Linux checkoutには不要。初回起動shellのCRLFもLFへ修正。いずれも学習開始前でrun出力の上書きなし。

- 最終検証: 全pytest14441 passed/8 skipped/0 failure/error（既存14427＋追加14）、旧・提案goldenを含む。意味変更後の全pytest開始後にRuffによる改行のみ整形し、最終対象＋依存3141件とfreshを再実測して成功。Ruff374 files・Pyright0 errors・pip check成功。元src/golden差分は空。ログ/JUnitはspec.json参照。
- 小規模パイロット: Ubuntu1000/client・6条件×seed0×3方式の18run全件完走。D131件のうち元非採用115件。全JSON/NPZ・元/適用評価の多重集合・10件境界・同一設定/環境/hash・RNGを照合。性能採否なし。本規模54runは未実行。
- 同じ実装レビュワーの最終証拠照合PASS、追加指摘なし。全JUnit・18run計36ファイルのhashと判定値を独立照合。全pytest/freshは主担当証拠で、独立学習・全pytest再実行はしていない。

- 実装commit `40bf6b8124e67f4210c6a3d8f268e2dc0f4d9ea3`。コミット後identity照合PASS、source486パスのhashをspec.jsonへ記録。固定旧差分空・clean・全JUnit14441/8skip・旧/提案golden test成功と一致。
