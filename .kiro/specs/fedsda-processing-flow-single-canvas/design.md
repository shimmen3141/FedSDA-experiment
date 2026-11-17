# Design Document

## Overview
旧SVG版の固定全構成図、濃淡と流れる破線、コマ送りを土台にする。前回の説明ページ形式は撤回。元の旧SVGはGit管理外でcommitは存在しない。バックアップ`../../venv/refactoring-tests/flow-doc-original-20261016.html`を参照する（現在のHTMLは前回の不採用ページ）。

## Boundary Commitments
所有はこのspecとGit管理外HTML、検証後の元checkoutの同HTMLへの反映。Python/test/golden/既定/並行Claude worktreeと前specの承認履歴は対象外。外部依存なしのHTML/CSS/JS/SVG。配置・経路・処理順・操作を変えたら再検証する。

## Architecture
SVG viewBox 1440×790の固定図。server左上、clientは右側と左下を含むL形で包む。client内はモデル群中央上、予測右上、モデル別ストア中央中、帰属・警報応答中央下、FIFO中央最下、候補検証左下、損失監視/警報ゲート右下。同じノードをコマや構成で移動/別図へ置換しない。

| 機構 | SVG座標範囲の目安 | port/描画 |
|---|---|---|
| server | x20〜310,y100〜415 | 登録→集約→条件付き統合→配布。モデル上りy175、配布下りy320へ別通路 |
| 入力 | x1100〜1370,y25〜85 | 特徴x/正解yを一個の標本ノード。特徴は上部、正解と標本は右端の通路 |
| モデル群 | x420〜1020,y140〜395 | 共有Bと3つの固有部（名称は略さず表示）。上が特徴、下が学習、左が通信/検証、右が確率/現行損失 |
| 予測 | x1110〜1370,y125〜410、出力y427〜475 | 同期/局所の二列→共通w→各結合→v→出力。監視用損失はここから出さない |
| ストア | x420〜1020,y450〜535 | モデルと対応順の3ストア。上へ学習batch、下から帰属標本 |
| 帰属/再利用/警報応答 | x420〜1020,y565〜620 | FIFOから上へ、ストアへ上へ、右から警報、左へ候補保留。既存評価はモデル右下専用通路 |
| FIFO | x420〜1020,y660〜740 | 右から新着標本、上へ未確定標本、右の外周から重み再較正へ |
| 監視/警報ゲート | x1110〜1370,y495〜650 | 局所現行損失→監視→最小件数のゲート→帰属/警報応答 |
| 候補検証 | x20〜320,y475〜735 | 候補/固定参照→将来損失→判定。モデルからの参照コピーと採用の帰りを別portの平行通路に分ける |

送信/学習矢印はモデル枠のportへ結び、内部の共有特徴→固有部の構造線を横断しない。共有Bは一つ。パラメータの「局所/同期の写し」は同じ構造の別状態として札を分ける。★は現行の学習帰属、wの最大とは別。新規採用は追加札で示し既存モデルを消さない。各actorは常時見える。通常frameは非対象opacity約0.28、対象1、全体frameは全actor1/線は薄い。対象edgeのみ流す。

## Viewportと文字
通常viewportは1440×900/1280×720、ズーム100%。header最大44、toolbar最大48、caption最大52、timeline最大36、legend最大24 CSS px。**外側余白と枠の合計**は最大16px。残り（720−44−48−52−36−24−16=500px以上）をcanvasへ割当。captionはSVGへ重ねない。閉じた補足もcanvasを覆わない。
全機構/操作bboxがviewport内で、ページ縦横スクロール0を確認。SVG文字は主ラベル`text.primary`を20px以上、補助も16px以上。各eの`parseFloat(getComputedStyle(e).fontSize)*Math.abs(e.getScreenCTM().a)`を実CSS pxとして主>=12、補助>=10と検査。s=getScreenCTM().aは**実canvas**の変換で、画面全体の比でない。500/790×20≈12.66、×16≈10.13で、実測でも下限を確認。狭幅は全図維持＋明示的な拡大でスクロール可、全図へ戻せる。

## 操作と状態
コマ前後は端で停止、番号で直接移動、再生/停止、最終コマで停止。経路/構成/採否変更時はtimerを止め先頭へ。select/buttonのキーを横取りしない。図focusの矢印キー/spaceのみ操作。低モーション対応。frameはnode/edge/短いcaption/現行★/候補/判定/登録札の宣言から毎回再描画し、戻っても正しく再構築。node座標は不変。

## Frame contract
次のJSONだけをframe順序の正本とする。順序そのものを検査する。ここでaccept/reject/reuseのV05は、警報時変化区間の帰属保留の解決。後続N03/N04は今回到着した標本の通常FIFO処理であり別の標本/処理で、重複帰属ではない。検証後のその標本で新たな警報がない場面を追う。V06が監視で、N01を重ねない。

```json
{
  "mixture": {
    "overview": [
      "O01"
    ],
    "steady": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "N01",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "wait": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "W01",
      "N03",
      "N04",
      "N05"
    ],
    "known": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "A07"
    ],
    "unknown": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "C01",
      "C02",
      "C03"
    ],
    "accept": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reject": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reuse": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P05",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "server": [
      "R01",
      "R02",
      "R03",
      "R04",
      "R05",
      "R06",
      "R07",
      "R08",
      "R09",
      "R10",
      "R11"
    ]
  },
  "sync": {
    "overview": [
      "O01"
    ],
    "steady": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "wait": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "W01",
      "N03",
      "N04",
      "N05"
    ],
    "known": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "A07"
    ],
    "unknown": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "C01",
      "C02",
      "C03"
    ],
    "accept": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reject": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reuse": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "server": [
      "R01",
      "R02",
      "R03",
      "R04",
      "R05",
      "R06",
      "R07",
      "R08",
      "R09",
      "R10",
      "R11"
    ]
  },
  "local": {
    "overview": [
      "O01"
    ],
    "steady": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "wait": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "W01",
      "N03",
      "N04",
      "N05"
    ],
    "known": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "A07"
    ],
    "unknown": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "N01",
      "N02",
      "A01",
      "A02",
      "A03",
      "A04",
      "A05",
      "A06",
      "C01",
      "C02",
      "C03"
    ],
    "accept": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reject": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V04",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "reuse": [
      "P01",
      "P02",
      "P03",
      "P04",
      "P06",
      "P07",
      "V01",
      "V02",
      "V03",
      "V05",
      "V06",
      "N02",
      "N03",
      "N04",
      "N05"
    ],
    "server": [
      "R01",
      "R02",
      "R03",
      "R04",
      "R05",
      "R06",
      "R07",
      "R08",
      "R09",
      "R10",
      "R11"
    ]
  }
}
```

### frameからactor/edgeへの共通の基底表
ALLは全actor。IDが実在することを全frameで確認する。同期/局所では反対側のprob/combine/paramとmixture-weightsを強調せず対応edgeも省く。P06の出力は選んだ側の結合→outputへ専用の迂回edgeを用意する（`single-output-sync`/`single-output-local`）。V05はacceptでadopt、reuse/rejectでは`validation-return`で帰属判断に戻し、対象ストアを示す。矢印は各frameのedge集合だけを強調する。

| ID | 活性actor | 活性edge |
|---|---|---|
| O01 | ALL | なし |
| P01 | input | なし |
| P02 | input, backbone, models | feature, features-0, features-1, features-2 |
| P03 | models, param-local, param-sync, prob-sync, prob-local | probabilities |
| P04 | prob-sync, prob-local, model-weights, combine-sync, combine-local | prob-sync-w, prob-local-w, w-combine-sync, w-combine-local |
| P05 | combine-sync, combine-local, mixture-weights | mix-sync, mix-local |
| P06 | mixture-weights, output | output |
| P07 | input, model-weights, mixture-weights | label-weights |
| N01 | models, param-local, monitor | current-loss |
| N02 | input, fifo | sample-fifo |
| N03 | fifo, assignment | fifo-assign |
| N04 | assignment, stores | assign-store |
| N05 | stores, models, backbone, param-local | train |
| A01 | monitor, alarm | signal |
| A02 | stores, models, backbone, param-local | train |
| A03 | fifo, alarm, assignment | alarm-assign, fifo-assign |
| A04 | assignment, stores | assign-store |
| A05 | assignment, models, param-local | assess |
| A06 | models, assignment | assessment-result |
| A07 | assignment, stores, models | assign-store |
| W01 | alarm, monitor, fifo | なし |
| C01 | assignment, candidate | assign-validation |
| C02 | models, references | copy-reference |
| C03 | candidate, references, loss-log | なし |
| V01 | input, candidate, references | future-validation |
| V02 | candidate, references, loss-log | candidate-loss, reference-loss |
| V03 | references, loss-log, decision | loss-decision |
| V04 | candidate, references, loss-log, decision | なし |
| V05 | decision, candidate, models, adopted | adopt |
| V06 | models, param-local, monitor | current-loss |
| R01 | models, registration, adopted | なし |
| R02 | models, registration | upload |
| R03 | registration, aggregation | register-aggregate |
| R04 | aggregation, clustering | なし |
| R05 | clustering, models, eval-data | evaluation-request |
| R06 | models, eval-data, clustering | evaluation-result |
| R07 | aggregation, clustering | aggregate-cluster |
| R08 | clustering, distribution | cluster-distribute |
| R09 | distribution, models, backbone, param-local | distribute |
| R10 | models, param-sync | なし |
| R11 | fifo, models, model-weights | replay |

段階の意味: P=特徴→共有/固有→確率→w結合→v混合→出力→正解/次回重み。N=局所現行損失→FIFO→古い分帰属→ストア→学習。A=警報ゲート→保留更新→分割→前区間確定→既存評価→結果→選択ストア。W=最小件数不足で証拠を保持し待つ。C=候補初期学習と変化区間のsession保留→参照固定→将来検証待ち。V=将来標本→候補/参照損失→既存再適合→無ければ前半後半比較→採否→通常監視へ。R=現行として待つ→送信登録→集約→統合条件→評価依頼→評価結果→統合→配布準備→配布→写し→FIFO再較正。候補は検証中追加学習しない。Tはwとvの分け合いに共用、FIFO容量と独立、設定値は図に載せない。

## File Structure Plan / Traceability
標準6文書はこのspec内、作業HTMLはdocs/overview/fedsda-processing-flow.html（Git管理外）、完成後に元checkoutの同HTMLへ反映。要求1→固定図/viewport、2→portと機構、3→frame contract、4→操作と濃淡。前回ページ/旧版ともバックアップし、反映前に他者編集と並行source差分を確認。

## Testing Strategy
全経路×全構成で上記JSONとframe ID順序が一致、actor/edge参照が存在、actor bbox不変、対象opacity1/背景0<opacity<1、全要素と操作のviewport内/文字下限を検査。強調edgeの全線分の内部交点と重複を禁止（共通端点/同じportでの分岐合流は許容）。outer境界と送り元/受け手の機構を除き、塗りのある他機構の矩形を横断しない。等比変換なのでSVG座標の幾何検査を全frameへ適用。実viewport1440×900/1280×720と狭幅をヘッドレスChromeで描画し、代表frameを目視。構文/外部依存/操作/低モーションも確認。独立CLI仕様/実装レビュー必須。Python変更なしのため全pytest/数値実験は対象外、未実施を記録。

## 対応表の構成別/結果別の上書き（この表を共通表より優先する）
共通表はmixtureの基底。以下は条件付き規則の完全な上書きで、同じframeの共通表を適用した後の追加ではない。ここに該当する行は活性集合全体を置換する。

| 構成/結果 | frame | 活性actorの完全な集合 | 活性edgeの完全な集合 |
|---|---|---|---|
| sync | P03 | models,param-sync,prob-sync | probabilities |
| local | P03 | models,param-local,prob-local | probabilities |
| sync | P04 | prob-sync,model-weights,combine-sync | prob-sync-w,w-combine-sync |
| local | P04 | prob-local,model-weights,combine-local | prob-local-w,w-combine-local |
| sync | P06 | combine-sync,output | single-output-sync |
| local | P06 | combine-local,output | single-output-local |
| sync | P07 | input,model-weights | label-weights |
| local | P07 | input,model-weights | label-weights |
| accept（全構成） | V05 | decision,candidate,models,adopted | adopt |
| reject（全構成） | V05 | decision,assignment,stores,models | validation-return |
| reuse（全構成） | V05 | decision,assignment,stores,models | validation-return |

actor IDの有効集合は共通表と上書き表の和集合（ALLは集合展開記号）とする。edge IDの有効集合も両表の和集合である。task1の成果物として、**SVGに実装したdata-actor/data-edgeの全IDと機構名/送り元/受け手の一覧を抽出し、両表の有効集合と一致すること**を検査する。frameで活性化しない構造線はIDもdata-edgeも持たせない。従って除外ID一覧は空集合[]。新たな機構を無断追加してID集合を拡げない。

通常frameの濃淡検査はO01を除き、O01は全actorが1であることを検査。狭幅検証は390×844。経路ラベルはoverview=全体、steady=平時、wait=警報待ち、known=既存再利用、unknown=候補開始、accept/reuse/reject=検証の採用/再利用/棄却、server=同期。高さ900pxでも同じ上限合計220pxでcanvas680px以上。720pxで補助文字下限に余裕が小さいので、上限変更時は実測を必須とする。

### 限定再確認の確定事項
- 共通表は基底、上書き表が該当frameの完全な置換である。「両表の和集合」はSVG IDの存在・余分なIDの検査にだけ使い、各frameの活性集合には決して使わない。
- frameで一度も活性化しないedgeのIDは空集合[]。装飾の構造線にID/data-edgeを付けないので、除外処理は不要。
- 文字12/10pxとスクロール0の条件は通常viewport（1440×900/1280×720）だけ。390×844では全図表示で全機構bboxがcanvas内にあることだけを判定し、文字下限は課さない。拡大を明示的に選んだ場合、SVGの実寸幅を1440 CSS px以上（s>=1）にし、主20/補助16px以上を確認。canvas内の横縦スクロールを許し、全図へ戻すと全機構bboxが再びcanvas内に戻ることを確認する。ページ自体はスクロールさせない。
- task1でID一覧抽出と一致を確認、task3で同じ検査を最終HTMLに再実行する。送り元/受け手の抽出は参考情報であり、期待値の機械比較はID集合だけ。意味と矢印の方向は主担当と独立実装レビューで確認する。
