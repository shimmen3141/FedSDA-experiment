# Tasks

- [x] 1. フェーズと枠内表示を整える
  - [x] 1.1 先頭枠/6色/結合記号のbbox/候補枠余白の検査を用意し、v4へ表示と接続を反映する。
  - _Requirements: 1, 2, 3_
- [x] 2. 通信と配布後コピーを接続する
  - [x] 2.1 sourceで確認した順序に基づくコマ列・送信/コピー矢印とマーク状態の検査を用意し、全枝へ接続する。
  - _Depends: 1.1_
  - _Requirements: 4_
- [x] 3. 最終表示を確認する
  - [x] 3.1 全組合せ・画面・代表PNGを検査し、独立CLI実装レビューを受け、指摘の採否/証拠をtasks.md、session/判定をspec.jsonへ記録する。commit/pushしない。
  - _Depends: 2.1_
  - _Requirements: 5_

## Implementation Notes
- 専用worktreeの別v4。既存成果物のhashを保持確認する。
- 実装調査: refactoring HEAD c4222ff。FedsdaRunClientの配布反映後captureと、snapshotの全classifier一括deepcopyを確認。登録は採番のみ、通信/平均は同じ集約関数内。該当source3ファイルのHEADとの差分は空。図は処理概念の分解でPython内部命令列の再現ではない。
- 初回仕様レビューでR12/R03の抽象度をMajorとして指摘（同じ出力内にPASSも併記）。全specに同一関数の入力/計算の図解と明示し、限定再レビューで単一PASS。組合せ数の説明も追記。
- v4を新規作成。6色、初回太枠、独立した確率記号、候補枠の余白、全共有枝の送信、右→左コピー、R10の選択マーク非表示を反映。R02の登録でパラメータを重複送信する旧表現を修正。
- 数学用文字の実測幅が約32で105幅では余白不足だったため、箱を左右5ずつ拡張して115幅へ、中心/主要接続を維持して記号を個別配置。狭幅縮小ではfont hintingでbboxが広がるため枠内を確認し、拡大後は通常と同じ6余白を確認。枠でモデル本体を覆わないfill:noneも明示。
- 主担当headless Chrome（専用profile）で3画面を各108組合せ2310コマ検証しissuesなし。表示/全枝/コピーと送信の向き/濃淡/余白/操作/固定位置/通常travel/低motionを確認、代表PNGを目視。v3に対する専用検査のREDを別名reportで確認。
- 独立Luna medium実装レビューPASS、Major/Minorなし。仕様4ファイル・実SVG/JS/CSS・source抜粋・代表PNG5枚・3監査reportを確認、JS構文とHTML hashを独立確認。独立ブラウザ再実行はなし。Python数値試験は対象外。
- 旧版/v2/v3/元checkoutのhash不変、外部依存なし。既存spec変更なし。stage/commit/pushなし。
