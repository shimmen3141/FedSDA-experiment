# IMPROVE-007: 最終予測に不要な比較予測診断を選択実行する

- 発見日: 2026-10-08。基準: 旧`748c3aa`。発見spec: alarm-adaptation-recording。
- 種別: 計算効率・処理の簡略化。状態: 未検証・未採用。

## 確認した事実

旧clients/fedsda.pyの最終Switching予測は1763–1765でFixed-Shareのscoresを使う一方、global AdaHedge比較の重み/正解数も1609–1614、1651–1653、1830–1847で計算する。比較結果はexperiment.py 1189–1194のglobal gain、1554–1556のconcept restart count等へ保存する。LOO診断のfallbackにもAdaHedge proposalが使われる（diagnostics/routing.py 328–342）。これらは最終3goldenの比較指標には含まれないが、既存成果物には残る。最終予測だけを見て通知や比較状態を削除すると保存診断が変わる。

## 案と仮説

比較予測やLOOなどを明示的な診断設定にし、不要な診断を無効にしたrunでは、そのためだけのAdaHedge状態の更新・再始動・比較scores生成を省く。分類器のscoresはFixed-Shareでも使うので、forward計算全部が省けるわけではない。CPU処理・状態保持・設定分岐を削減できる可能性があるが、効果の大きさは未測定。

## 検証と採否

診断有効時は既存NPZ/CSV診断と数値・順序が一致すること、無効時は最終予測・学習・RNGが一致することを確認する。診断無効をゼロ値で表現せず、未計測として保存schemaへ明示する。時間・メモリ・精度・通信量を同条件で比較し、処理の削減が実測でき、診断を必要とする研究比較を損なわない場合に採用する。今回の構造移植には混ぜず、別specで判断する。
