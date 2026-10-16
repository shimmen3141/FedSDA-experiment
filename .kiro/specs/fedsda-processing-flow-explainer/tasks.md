# Implementation Plan

- [x] 1. 現行処理に対応した全体像と用語を表示する
  - 基準commitと構成を明記し、client/serverとモデル構造の静的な全体図、用語、処理順を作る。
  - 起点と受け手、現行モデルと予測用のモデル、二種類の重みを区別して読める状態にする。
  - _Boundary: 説明コンテンツ_
  - _Requirements: 1, 2, 3, 4, 5.4, 5.5_
- [x] 2. 機構ごとの受渡しをコマ送りで追えるようにする
  - 章と構成・分岐の選択、コマ移動・再生停止、固定配置の図と文章の受渡しを接続する。
  - 全章と分岐に到達でき、端で停止、選択変更で再生停止、キーボードと狭い表示で読める状態にする。
  - _Depends: 1_
  - _Boundary: インタラクティブ表示_
  - _Requirements: 1.3, 2, 3, 4, 5.1, 5.2, 5.3_
- [x] 3. 内容・操作・表示を検証して依頼元へ反映する
  - 独立CLI実装レビュー、構造/操作/視覚確認を行い、実施範囲と未実施を記録する。
  - Claude側の関連変更と元HTMLの他者編集を確認し、バックアップ後に完成HTMLを反映する。保留資料をstageしない。
  - 元と作業コピーが一致し、Python/golden/Claude側の作業ツリーに本作業の差分がないことを確認する。
  - _Depends: 2_
  - _Boundary: 統合と検証_
  - _Requirements: 1, 2, 3, 4, 5_

## Implementation Notes
task plan review gateとinline sanityはPASS（順序・境界・成果物・検証の依存を確認）。独立CLI仕様レビュー: Luna medium。初回Major3・Minor2を反映し候補の検証中の学習という下書き誤りも訂正。再確認で内容整合PASS、残ったtask3依存を明記し最小再確認PASS。sessionはspec.json。HTML専用scopeによりPython全pytestは対象外とし、実施しないことを記録する。

Task1: 静的な全体図・モデル構造・用語・損失用途・参照版の表示を作成。既存HTMLの32KBを機械的に流用せず責務ごとに再構成。構造checkerは実装前にファイル未存在で期待どおり失敗。操作はまだ接続していない。

Task2: 章・3構成・警報3場面・検証4結果のコマを接続。構造checkerとJS構文はPASS。キーボードは図にfocusがある場合だけ扱い、select/buttonの操作を横取りしない。次はヘッドレスChromeの操作検証と描画確認。内蔵ブラウザは利用不可（inventory空、iab生成がBrowser is not available）。利用者のブラウザを操作せず独立プロファイルのヘッドレスで検証する。

Task3主担当検証（レビュー前）: HTML構造checkerとnode構文PASS。独立Chromeプロファイルで全章・3構成・警報3分岐・検証4結果の151通りのコマ、端/直接移動/キーボード/フォーム非横取り/手動停止/選択変更時停止/自動終端停止を確認。実viewport1440・390px、低モーションtrueで各151通りPASS、JS例外0・横はみ出し/図カード重なり/検査対象内の文字はみ出し0。描画3場面×2幅を保存し目視確認。Chrome通常モーションも別実行PASS。ログ/検証script/画像は../../venv/refactoring-tests/flow-doc-*（Git管理外）。初回のレイアウトcheckerはカード外の矢印pseudo要素を文字はみ出しと誤認し、実文字子要素の検査へ訂正。最初のChrome起動はsandbox EPERMのため、許可されたescalationで独立プロファイルを起動。

最終: Luna mediumの独立実装レビューPASS（sessionはspec.json）、指摘なし。現行sourceと内容、JS構文、保存済みブラウザ証拠を独立確認。レビュー側のPython構造checkerは一時JSの書込み権限で失敗したため独立成功とは扱わず、再試行なし。主担当の構造checkerは成功。ブラウザの独立再実行なし。

狭幅の入力→出力→正解の並びを縦に整えた最終HTMLで再検証。実viewport1440/390各151通り、通常/低モーション、実行時エラー0。全体・予測・平時・警報・検証・同期の狭幅描画も保存。JavaScript無効で全体図と用語が表示されることを確認。Python全pytest/数値実験はscope外につき未実行。

並行側はb68b7a2へ進んだが、参照3e6153cからsrc/.kiro/specs差分は空（研究文書と改善候補の追加のみ）。元HTMLが作業前バックアップと同一であることを確認して更新し、元と作業コピーのバイト一致を確認した。元のバックアップは元checkoutのvenv/refactoring-tests/flow-doc-original-20261016.html。HTMLはGit管理外・コミット保留のまま、spec文書だけを専用branchに保存する。Claude側の作業ツリーは変更していない。
