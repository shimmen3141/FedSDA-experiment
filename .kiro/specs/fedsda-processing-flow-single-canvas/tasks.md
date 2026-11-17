# Implementation Plan

- [x] 1. 固定した全構成図の配置と専用portを組み立てる
  - client/server境界と全機構を同一SVGに置き、モデル内部の共有/固有構造、FIFO/ストア/予測の関係を図示する。data-actor/data-edgeのIDと機構名/送り元/受け手の一覧を抽出し、designの有効集合と一致することを確認する。
  - ズーム100%の1440×900/1280×720で、操作部と図の全機構が一画面に収まる構成を作る。
  - _Boundary: 固定構成図_
  - _Requirements: 1, 2_
- [x] 2. 全経路のコマ送り・濃淡・受渡しアニメーションを接続する
  - 平時と分岐、現在モデル・候補・登録の状態を同一座標の上で表し、同時強調する線を交差させない。
  - 再生/端停止/選択変更/直接移動/キーボード/拡大と低モーションが操作できる。
  - _Depends: 1_
  - _Boundary: frameとアニメーション_
  - _Requirements: 2, 3, 4_
- [x] 3. 図の意味・操作・描画を検証して元HTMLへ反映する
  - designのJSONと全経路/全構成のframe IDの順序を照合し、共通基底表/上書き表のID和集合との一致（活性集合には和集合を使わない）、node/edge実在と固定座標・交差・箱横断、実viewportの全図/操作bbox、主文字12px/補助10px以上、実装との意味を検証して成功を記録する。
  - 元の他者編集と並行側source差分を確認し、バックアップ後に完成HTMLを反映。保留は維持。その反映後の最終HTMLに対し、同じID一覧・frame順序の検査を再実行して記録する。
  - _Depends: 2_
  - _Boundary: 検証と反映_
  - _Requirements: 1, 2, 3, 4_

## Implementation Notes
要件/設計gate、順次1→2→3のtask sanityを主担当が確認。内蔵子エージェントなし、独立CLI仕様レビュー前。前回のspecの完了記録は書き換えない。このspecはユーザーの訂正を新しい表示契約として扱う。

仕様レビューMajor3・Minor2に対して、UI各行の高さ上限/残るSVG領域/主文字の実寸下限、必須frame契約表、交差と箱横断の幾何判定の範囲、caption非重畳をdesignへ追記。

Haikuの指摘反映: designの重複した順序表を削除し全構成JSON一つを正本化。frame→actor/edge対応を列挙、警報応答領域を座標表へ追加。補助文字10px以上と余白合計の意味、操作契約、task3の観測条件を明記。旧版はGit管理外でcommitはないと明記。

構成別/結果別対応は共通表を置換する完全な上書き表を追加。task1で抽出/一致を検査しtask3で最終HTMLに同じ検査を再実行。O01例外・狭幅390×844・ラベル対応も明記。

Task1/2: 旧形式の固定SVGへ再構成。client/serverとモデル/ストア/FIFO/検証/予測/監視を常時配置し、frameの濃淡・線・状態札を接続。検証はこれから。

最終検証・レビュー:
- 独立仕様レビューで曖昧だった寸法/完全frame順序/基底と上書きの優先/ID検査の範囲を整理。Haiku mediumで内容通過、残った反映後検査のtask明記をLuna mediumで限定PASS（spec.json）。初回Lunaの存在しないN06引用は採用せず、実ファイルをHaikuが確認してN06なしと判定した。
- 41種類のframeから、3構成×9経路の全352コマを検査。1440×900/1280×720/390×844でissuesなし。通常サイズは主文字12px/補助10px以上、全図と操作が画面内、ページスクロールなし。狭幅は全図維持/拡大/全図復帰。活性線の内部交差/重複・機構箱横断なし、固定actor座標/opacity/ID/順序一致。
- 初回検証scriptはWindows改行でJSON抽出失敗しLF正規化で修正。図の初回検証は特徴線が新規札を横断し専用通路へ修正。実描画で入力/FIFOラベルとセル、ストアの塗り、見出しと札を修正。FIFOの前/変化区間を仕切り、将来損失を前/後半の箱で示した。
- 通常モーションはtransitionだけを検査用に止めて全352コマを検査、flowAnimation=travelを確認。低モーションもPASS。キーボード、フォーム非横取り、端停止、経路変更停止、自動最終停止を確認。
- Luna medium独立実装レビューPASS、指摘なし。実SVG/仕様/overview・learningの画像とレポート、node構文を独立確認。ブラウザの独立再実行は未実施。主担当は各通常viewportの代表コマ画像を目視。Python全pytest/数値実験はscope外で未実施。
- 並行側c4222ffまでのsrc差分なし。元HTMLと前回ページのバックアップが同一で他者編集がないことを確認して反映。反映後の元HTMLに同じID/順序/幾何/画面/操作検査を再実行し全352コマPASS。元と作業コピー一致。保留HTMLはstageしない。
- 証拠は元checkoutのvenv/refactoring-tests/flow-canvas-*、検査scriptはcheck-flow-canvas.cjsとbuild-flow-canvas.py（Git管理外）。旧SVGはflow-doc-original-20261016.html、不採用ページはflow-doc-rejected-page-before-single-canvas.htmlに保持。
