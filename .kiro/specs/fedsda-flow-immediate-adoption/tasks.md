# Implementation Plan

- [x] 1. 仕様レビュー後、新モデル作成と即時採用の図・遷移へ限定変更する。
  - Requirements: 1〜4
- [x] 2. 全経路・操作・代表画面・実装対応を確認し、独立実装レビューを反映する。
  - Requirements: 1〜4

## Implementation Notes
ユーザーが方針3（旧検証方式を図に残さず現在のHTMLを更新）を選択。refactoringの並行編集へは触れない。

- 独立Luna highの仕様レビューはPASS、Blocker/Major/Minor 0。現行srcとの対応と変更範囲を確認後に実装した。
- 受入検査を先に準備し、旧HTMLが`Legacy acceptance UI remains`で失敗することを確認した。
- 旧候補検証領域を新モデル作成へ転用。C01〜C03と関連線・選択状態・説明だけを変更し、将来標本V01〜V06/F01を除去した。残るframeの契約で変更したものはC01/C02/C03のみ。候補採用後のサーバ登録待ちは保持した。
- Chrome headlessで1280×720・1440×900・390×844の各6組合せ（警報3状態×応答2選択）、計18経路444コマを検証。actor参照、順序、即時採用表示、対象外分岐、左右循環、フェーズ移動、全体図、再開、実時間再生、補足開閉が成功しpageerror 0。新規作成C02/C03と狭幅全体図のPNGを目視した。既存の狭幅縮小表示は変更しない。
- 実行用補助はGit管理外の元checkout `venv/flow-browser-check/`。`check.cjs`と`validation.json`、画面PNGを保存した。最初のブラウザ起動はsandboxのEPERMで止まったため、専用profileのheadless Chromeを権限承認下で実行した。既存ブラウザのprofileは使っていない。
- HTMLと本specのみの変更。実験コード・goldenは未変更であり、数値pytest/Ruffは対象外。
- 独立Luna highの実装レビューはBlocker/Major 0・Minor 1。SVGの読み上げ用descに旧名称「候補検証」が残っていた指摘を採用し、「新モデル作成」へ修正した。初回判定FAILの記録を保持し、Minorのみのため再レビューは行わない。レビュワーは18経路の成功証拠を照合し、ブラウザ独立再実行は未実施。
- HTMLParserによる実id143件の重複なし、全36frameのactor/edge参照解決、外部assetなし、Node構文、差分の空白検査を確認。残る候補以外のframeと左右送り・再生・キーボード・補足handlerは旧版と同じ。最初の正規表現id検査はdata-model-idまで拾った検査側の誤りであり、HTMLParserで再確認した。
- Minor修正後、実ブラウザで読み上げ用descの新名称と、先頭で左キー→末尾・右キー→先頭の循環を再確認して成功。mainやrefactoringへのマージは行わない。

## ユーザー追記：コピー元・追加先の明示
- [x] 3. 追記仕様レビュー後、コピー矢印・候補の名称・新モデルと新ストアの限定表示・直接標本線へ修正し、全経路と代表画面を確認して独立レビューを反映する。
  - Requirements: 5〜7

- 追記仕様レビューは独立Luna highでPASS、Blocker/Major/Minor 0。
- 実装前の追記受入検査が`Obsolete adoption box remains`で失敗することを確認後、名称とコピー2線・追加先の共有接続・標本枠からの直接ストア線へ変更した。変化区間枠は処理へ切り出した標本の模式表示であり、別の永続ストアではない旨を補足へ明記。
- C03/R01の新規モデルとC03のみのnストアをCSSで限定表示し、全体図・他コマで非表示にした。旧「＋新規モデル」と即時追加専用枠は除去。新モデルとデータの追加先を1′/1の標本から切り離した。
- Chromeの3画面サイズで延べ18経路444コマ、追加枠の限定表示・actor/edge参照・コピー2線・追加/標本線の接続・共有接続・既存操作を確認し、issues/pageerror 0。C01/C03/R01のPNGを目視した。共有ラベルが追加線に重なりやすかったため短い「共有」へ位置を調整した。
- 残存frameで変更した契約はC01/C02/C03/R01のみ。検査はGit管理外`venv/flow-browser-check/check-connections.cjs`、結果は`connections-validation.json`、PNGは`connections-<幅>-<frame>.png`。
- 独立Luna highの追記実装レビューはPASS、指摘なし。全18経路の結果を照合し、C01/C03/R01のPNGを独立目視。ブラウザの独立再実行は未実施。
- 追加の文字枠検査で「バックボーンコピー（固定）」の横幅が余白を超えたため、その見出しだけ18pxへ調整。`geometry.cjs`の再検査で候補/新モデル/新ストアの文字収まりと共有ラベルの追加線との非交差が成功した。最終PNGは`connections-final-<frame>.png`、結果は`geometry-validation.json`。
