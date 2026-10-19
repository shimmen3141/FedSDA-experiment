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

## ユーザー追記：コピーコマの交差解消と同列追加の強調
- [x] 4. 追記仕様レビュー後、標本受渡しとコピーを分離し、標本系の色を統一、共有矢印を除去して追加コマのモデル/標本群を濃くする。全分岐・限定表示・操作と代表画面を確認し、独立レビューを反映する。
  - Requirements: 8〜9

- 仕様レビュー初回Major 1は対象「candidate枠」の略記に関する指摘。現行HTMLのid=candidateは変化区間の標本枠であり、候補モデル本体はcandidate-head、外枠はcandidate-shellであることを明示し、色変更対象を限定した。既存ストアの緑は保持。独立Luna high再確認でPASS。
- 先行受入検査は旧版に対して`Shared arrow remains`で失敗。C00を標本受渡し専用、C01をコピー専用に分け、標本枠/受渡し2線を青緑にした。重複する変化区間ラベルと共有線/ラベルを除去した。C03はmodels/stores、R01はmodelsを追加活性化し、配布時側と新n枠の限定表示は保持した。
- Chromeの1280×720・1440×900・390×844で各6分岐、延べ18経路447コマに成功（issues/pageerror 0）。順序、色差、活性線の分離、追加群の濃淡、限定表示、循環、フェーズ移動、再生、補足を確認。C00/C01/C03のPNGを目視し、コピー2線の全直交segmentが交差しないことも計算で確認。全frameの参照・Node構文・diff --checkも成功。
- 変更frameはC00の追加とC01/C03/R01のみ。検査は元checkoutのGit管理外`venv/flow-browser-check/check-declutter.cjs`、結果は`declutter-validation.json`、PNGは`declutter-<幅>-<frame>.png`。
- 独立Luna highの実装レビューはPASS、指摘なし。差分・18経路447コマの証拠とC00/C01/C03のPNGを独立確認した。ブラウザの独立再実行は未実施。

## ユーザー追記：ID登録コマの表示
- [x] 5. 追記仕様レビュー後、新規モデル文字を拡大し、R02でも表示・活性化して配布時モデルを薄くする。限定差分・文字の収まり・全分岐表示を確認し独立レビューを反映する。
  - Requirements: 10

- 独立Luna highの仕様レビューはPASS、指摘なし。サーバ登録は保有状態等のID付替えであり、配布時snapshotを更新しないことを実装で確認した。
- 先行検査は`New model title remains small`で失敗。新モデル名・CSS表示限定・R02 actors・localOnlyの4箇所だけ変更した。4置換の逆変換で変更前HTMLの全テキストに一致することを確認。
- 当初の18px案は文字幅127.82に対し枠幅125で収まらなかったため、実測113.75の16px（以前13px）へ調整した。枠寸法と他の文字は変更しない。最初の逆変換検査は別の既存18px指定まで拾ったため、タイトル文字列で対象を限定して再確認した。
- 狭幅390では16px文字幅116.72・左余白1.77pxで枠内に収まっていたが、検査側の2px余白条件に触れたため、実際の枠内＋1pxの確認へ訂正した。HTMLの追加変更はしていない。Chromeの3画面・18経路447コマで表示限定・R02の活性と配布時opacity・文字の収まり・既存操作が成功しissues/pageerror 0。R02のPNGを目視した。検査と証拠はGit管理外`venv/flow-browser-check/check-registration.cjs`、`registration-validation.json`、`registration-<幅>-<frame>.png`。
- 独立Luna highの実装レビューはPASS、指摘なし。4置換の逆変換一致・全経路の成功証拠・R02の1280/390 PNGを独立確認した。ブラウザの独立再実行は未実施。

## ユーザー追記：複数処理をまとめた中継枠の整理
- [x] 6. 実装照合と追記仕様レビュー後、区間分割・再利用判定の処理記号と直接データ線へ整理し、対象外表示を保持して全分岐/画面を検証し独立レビューを反映する。
  - Requirements: 11〜13

- 実装の容量超過分吸収→解放、前区間吸収→変化区間評価→適合先吸収/帰属切替または候補という順を読取確認。独立Luna highの仕様レビューはPASS、指摘なし。
- 先行受入検査が旧枠に対し`Generic assignment hub remains`で失敗後、assignmentを区間分割/再利用判定の菱形2個へ置換し、4種類の標本受渡しを直接線へ変えた。評価結果だけを再利用判定へ渡す。★の切替、FIFO分割と該当区間の濃淡、実装上の処理順は保持した。
- 全frame参照とNode構文、対象外frame契約の不変を照合。変更契約はN04/A03〜A07/C00のみ。対象の評価結果/再利用コマA06/A07も更新用側へ焦点を合わせた。A05はbackboneを活性化し、共同学習の説明param-localは非活性へ変更して推論と学習を区別した。モデル/ストア/FIFO/サーバ/予測結合の配置と新規追加フロー、操作handlerは変更しない。
- Chromeの3画面・各6分岐、延べ18経路447コマを検証しissues/pageerror 0。新記号・直接線端点・★・限定表示・既存操作を確認。最終座標の新ラベルと活性線の非交差、菱形文字の収まりも3画面で検査成功。N04/A03/A06/A07/C00のPNGを目視した。補足へ処理記号が保存場所ではない旨だけ追記した。
- 検査はGit管理外`venv/flow-browser-check/check-assignment.cjs`・`assignment-geometry.cjs`、結果は`assignment-validation.json`・`assignment-geometry.json`、最終PNGは`assignment-final-<幅>-<frame>.png`。
- A05の非学習表示とA06/A07の更新用側焦点も最終追加確認し、それぞれ3画面で成功。結果は`assignment-evaluation.json`・`assignment-focus.json`、対応PNGを更新した。
- 独立Luna highの実装レビューはPASS、Blocker/Major/Minor 0。全経路の証拠、新線・処理記号の代表PNG、A05/A06/A07の最終限定検査と対応PNGを独立確認。ブラウザの独立再実行は未実施。実験コード・goldenは未変更で数値pytestは対象外。

## ユーザー追記：矢印の色・向き・結果とタイムラインの左右キー
- [x] 7. 追記仕様レビュー後、要求14〜18の矢印表現とタイムラインのキー処理を限定変更し、全分岐・色/方向/結果・実キー操作と代表画面を確認して独立レビューを反映する。
  - Requirements: 14〜18

- 独立Luna highの仕様レビューはPASS、指摘なし。先行受入検査は旧版の矢印ラベルの色不一致で失敗した。
- 全矢印ラベルを対応する線色に統一し、A03を区間分割→FIFOの赤破線へ反転、A05の評価線を更新用バックボーンの右端へ接続した。A06に分岐に応じた判定結果の出力線を追加した。frame契約の変更はA03/A06のみで、モデル等の配置と他frameの順序・表示契約は保持した。
- タイムラインのコマボタンに限定した左右キーハンドラを追加し、移動後のコマへフォーカスを追従させた。既存のcanvasキー処理・ラジオ・補足モーダルは変更しない。
- Chromeの1280×720・1440×900・390×844、各6分岐の18経路447コマでissues/pageerror 0。全ラベルの色、破線と向き、評価端点、分岐結果、実クリック後の左右キー・先頭末尾循環・連続移動とフォーカス、ラジオとモーダルのキー操作を確認した。全frame参照・Node構文・diff --checkも成功。新ラベルと活性線の非交差、処理記号文字の収まりは3画面で追加検査成功し、A03/A05/A06のPNGを目視した。
- 証拠は元checkoutのGit管理外`venv/flow-browser-check/check-arrows.cjs`・`arrows-validation.json`・`arrows-geometry.cjs`・`arrows-geometry.json`・`arrows-<幅>-<frame>.png`・`arrows-final-<幅>-<frame>.png`。実験コード・goldenは未変更で数値pytestは対象外。
- 独立Luna highの実装レビューはPASS、Blocker/Major/Minor 0。検証スクリプト・全経路の成功記録・A03/A05/A06と既存再利用分岐の最終PNGを独立照合した。ブラウザの独立再実行は未実施。
