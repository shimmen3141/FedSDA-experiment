# 作業

- [x] 1. 実装照合・仕様レビュー後、先行配布と依頼を分離し、受信保持・変更分再送・内部反映を図示する。
- [x] 2. 全分岐/3画面と代表PNG、対象外不変と既存操作を検証し、独立レビューを反映する。

## 実装記録
- server_round_synchronization.pyは既定で先行配布量を記録し、クロス評価のモデル転送を数えず、統合後は代表固有部と必要な共有部の再送を記録する。状態更新は従来順を維持。fedsda_run_client.pyは最終反映の後に予測用snapshotを保存する。5d58741で読取確認。

- 独立Luna highの仕様レビューPASS、指摘なし。先行受入検査は旧版で`Missing separate evaluation-request frame`に失敗した。
- R05先行配布、R05aのID依頼を追加し、R05〜R09に受信モデル群を保持。R08の送信内容は変更分だけ、R09は受信枠から更新用モデル群への内部反映線にした。受信枠の補助文も反映待ち/反映中を区別。R09だけも右群の対象範囲点線枠を表示し、線の終点を明示した。R10コピーとR11再較正を保持した。
- frame契約の差分はR04〜R09と新規R05aだけ（R10/R11/R12および通常予測/FIFO/ドリフト/新規作成は不変）。新しい条件説明は通常配布へ進む場合を示すが、分岐UIを増やさない。既存操作handlerは変更しない。
- Chromeの1280×720・1440×900・390×844、各6分岐の18経路471コマでissues/pageerror 0。配布元・依頼時のモデル非送信・受信保持・旧snapshot非表示・変更分ラベル・内部反映・順序と既存のキー/循環/focus/分岐/再生/補足を確認。
- 代表コマの新ラベルと活性線の非交差・caption収まりは3画面で成功。代表PNGを目視。最終追加のR09対象枠/反映中補助文は3画面の限定検査でも成功した。
- 証拠は元checkoutのGit管理外`venv/flow-browser-check/check-distributed-flow.cjs`・`distributed-flow-validation.json`・`distributed-flow-<幅>-<frame>.png`、`distributed-geometry.cjs`・`distributed-geometry.json`・`distributed-final-<幅>-<frame>.png`、`distributed-apply.cjs`・`distributed-apply.json`・`distributed-apply-<幅>-R09.png`。Node構文・diff --checkも成功。実験コード/golden未変更で数値pytestは対象外。
- 独立Luna high実装レビューPASS、指摘なし。全経路と配置検査の記録、R05〜R10の代表PNGを独立照合。R09の最終表示追加も別途確認しPASS。ブラウザ独立再実行は未実施。

## ユーザー追記：サーバ内処理と通信の分離
- [x] 3. 要求8〜10を仕様レビュー後に反映し、サーバ処理列・条件ひし形・配布2線と強調を整理する。全分岐、3画面、接続/枠内/線ラベル、対象外不変を検証して実装レビューを反映する。

- 追記8〜10の仕様レビューはPASS（Minor 1）。旧design/namingのreceive-modelsとdistribution起点の記載は、追記で置換済みと注記して解消。Minorのみで再レビュー不要。
- 先行受入はR04が条件ひし形以外の2処理を強調することで失敗。サーバ5段（条件ひし形を含む）・常時薄い順序ガイドへ整理し、early-modelsを集約から、changed-modelsを統合から送る線へ分離。配布箱と旧内部活性線を除去した。
- 全18経路471コマ、3画面でissues/pageerror 0。R04条件のみ、R03/R07の実行処理、R05/R05a/R06/R08の送受信元と線、旧箱/線の除去、受信保持、順序と既存のキー/循環/focus/再生/補足を確認した。
- frame契約変更はR04/R05/R05a/R06/R07/R08だけ。buildと操作handler全文が変更前に一致することをassert確認。クライアント側モデル/ストア/FIFO配置・コマ数/順序は不変。
- 配置検査で文字bboxの逸脱を確認し、20→18px、baseline修正を試したが、画面高によるSVG縮尺と文字の境界差で別画面に逸脱が残った。検査条件は緩和せず、専任fixerにサーバ箱4つの文字/余白だけの根本修正を委譲した。
- Fixerの実測で、390px時は18px見出しbbox高23.684＋16px説明18.947で41.684になり40px箱に収まらないことを確認。4箱を高さ48px、各行のbaselineを調整しdata-boxも追随。箱のx/y/幅と通信ポート・ガイドは不変。全配置検査が成功しissues[]。最小の上下余白は390pxでも2.263 SVG単位を確保した。検査条件は緩和していない。
- 最終の配置証拠はGit管理外`venv/flow-browser-check/server-layout-geometry.cjs`・`server-layout-geometry.json`・`server-layout-final-<幅>-<frame>.png`。全経路の証拠は`check-server-layout.cjs`・`server-layout-validation.json`（18経路471コマ、errors0）。
- 独立Luna highの最終実装確認で、最新4箱48pxの配置、全経路/geometry成功記録、1280のR04/R05/R08と390のR04 PNG、対象外不変を照合し問題なし。ブラウザ独立再実行は未実施。
- 最終レビューのMinor 1（確認時点のtask3未チェック）は、完了チェックと記録を同期して解消。実装への指摘はなく、Minorのみのため再レビュー不要。
