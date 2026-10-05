# 統合検証: optimizer状態所有と明示リセット

## 範囲と正本
Task1はLuna独立APPROVEDで完了、最終完成判定未実施。
固定parameter/設定を借用し、現在optimizerと成功後reset交換だけ所有する。
モデル生成・共有接続・reset時機・parameter差し替え・学習・state移送・空共有optimizerは後続。
spec.jsonが承認/進捗、naming revision3が正式名、review.mdが採否の正本。

## 実測
- Task1: 新module未存在でImportError、1 collection error/3.35秒/exit1。実装後23 passed/4.56秒。
- 実旧SharedBackboneMLP.reset_optimizerへ6設定×更新0/2回を直接比較。groups/state/parameter値/grad一致、reset後3回同勾配（None混在）でもexact一致。
- 値/grad参照/parameter順保持、readonly現在参照、外側更新の反映、固定条件へ戻す再生成、旧optimizer state保持、繰り返しresetを確認。
- 不正params/settings・不正dtypeの再検証、生成例外で現optimizer/state/値/grad保持、Python/Torch RNG不変を確認。
- 対象Ruff/format成功、Pyright0 errors/0 warnings、diff-check成功。固定748c3aaとの旧production/golden/旧回帰test差分なし。

## 旧所見と限界
今回新たな旧正常系不具合を確認していない。既存LEGACY-010/空共有optimizerの拒否は変更しない。
reset後の旧bindingは旧optimizerのまま。上位が現在参照を取得し新bindingを作る。
parameter/settingsは借用。frozen/private回避や並行resetは通常契約外。
旧全体goldenは旧コード固定値の確認、新旧同値性は部品対照で確認する。新全体run golden完成を主張しない。
