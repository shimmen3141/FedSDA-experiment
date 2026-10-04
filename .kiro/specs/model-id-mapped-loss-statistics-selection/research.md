# 調査・設計判断
## Summary
旧clients/base.py:662–738と既存ModelAndClassLossStatistics/Storeを直接確認した。統計選択だけを独立させる。
## Research Log
- 旧678–694: ID対応は一回、target初出順、max(n)でfirsttie、server missing/zeroのみwhole置換。class全値/順序維持。
- 周辺currentID/適応記録・storeddata/trainingdata・モデル再構築は別責務。サーバの重み付き統計集計も別機能。
- モデル実体なしstubで旧apply_server_mappingを直接実測。chain/cycle/firsttie/server larger/localpositive/zero補完と順序を確認。
- 旧localwinnerは元dictを共有、server補完はdeepcopy。新APIは既存不変型の方針に従い全結果を独立コピーする。選択の数値/順序変更ではない。
- pending登録の参照所有は別の将来調査。今回正常経路の新不具合は観測していないため台帳候補を推測で追加しない。
## Synthesis
store snapshotを直接受け、同形式を返すpure関数を採用。dict入力ではなくtupleにしてID重複を検査可能にする。
全recordを公開constructorでコピー/検査するためprivate helper依存を避けられる。新snapshot型/merge枠組み/既存store全置換APIは不要。
cc-sdd requirements/design prewrite gateと命名規約、fable-methodの旧oracle実測・受入証拠先決めを適用。
