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
