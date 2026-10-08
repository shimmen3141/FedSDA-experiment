# 実装タスク revision1

逐次実行。既存公開部品を接続する1つの応答runtimeと、実旧oracleの再利用による検証。役割を跨ぐ後始末ownerは作らない。runtimeの外部下書きは約260行、testは約760行で、多くは既存NN/状態照合helperとparametrizeの接続。対象CPU実測は数十秒、全回帰は直前232秒。各taskを独立レビューし、最後に別fresh feature最終GOを行う。

- [x] 1. 不変応答recordと初期依存境界を統合する
  - record-onlyのfoundation統合task。正式5値、frozen/kw_only、active/不足/解決のfield組、FIFO消費判断をtestのRED後に実装。
  - 同moduleのrecord定義だけを作り、その時点で使用するexact import集合と注入契約を両resolverへ登録する。未来のruntime importをまだ許可しない。
  - 完了: record対象と依存suiteが成功、独立レビュー承認。命名表は全taskに先行して承認する。
  - _Boundary: 不変応答とrecord依存境界の統合_
  - _Requirements: 2.1, 2.2_

- [x] 2. 警報応答の分岐を実旧対照つきで組み立てる
  - runtimeの構造検査→active全件吸収、または準備→最小件数→公開解決を実装前REDから実装する。必要なexact importと注入testを同時に拡張する。
  - 実NNの2/4class×3解決条件×span/minimum境界で実旧と比較。active正規/負ID・空/非空、未使用設定、同じsessionのparameter/optimizer/履歴・pending参照不変、構造拒否/部分更新境界を確認。
  - このtaskは明示的な統合task。準備と解決を接続しているため、同じoracleで後続共同更新・候補観測まで照合し、別の本番学習処理は追加しない。
  - 完了: 対象test・exact依存suite・Ruffが成功、独立レビュー承認。
  - _Boundary: 既存準備/吸収/解決への応答の統合_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2_

- [x] 3. 実source変異で検出力を確認する
  - test-only。active無視、前区間省略、最小件数の丸め違い（<を<=）、session置換、変化区間へ全FIFOを渡す、余分な乱数、FIFO消費、消費指示逆転、元metadataの上書きを代表変異とする。
  - 変異は各回finallyで元byteへ復元。syntax/collection失敗を検出成功に数えず、意味を持つassertion失敗と復元後hash/GREENを記録する。
  - 完了: 代表変異検出・byte復元・最終対象成功を独立reviewerが確認して承認。
  - _Boundary: 応答の検出力検証_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2, 2.3, 3.2_

- [x] 4. 依存境界と新CPUでの接続を検証する
  - 明示的なtest-only統合検証task。実importとexact集合一致、direct/alias/relative許可、wholemodule/private/star/child/upward拒否を照合する。
  - 新実装だけのfresh CPU processで不足・十分（現行維持）・activeの応答を実行し、FIFO保持と消費判断、実吸収、session参照、未使用入力と乱数を確認する。
  - 完了: 対象＋AST成功とfresh CPUの実測を記録、独立reviewerが再現/証拠確認して承認。
  - _Boundary: 応答の依存/独立動作の統合証拠_
  - _Requirements: 3.2_

- [ ] 5. 固定基準の全回帰と証拠を確定する
  - 全pytest/JUnit、旧11・最終3golden、Ruff/Pyright/pip check、固定旧から空diff、承認hash・source hash・tested commitを実測する。
  - integration-validation.mdに9要件の対応、各taskレビュー、未検証の新全体run/後始末、借用/部分更新の制約を残す。
  - 完了: 主担当の全実測とJUnitを独立担当が確認し承認（独立全pytest再実行は既存ユーザー決定に従い必須にしない）。その後別fresh feature GOを行い、再開案内を次の候補へ更新する。
  - _Boundary: 全回帰/品質と統合証拠_
  - _Requirements: 3.2_
