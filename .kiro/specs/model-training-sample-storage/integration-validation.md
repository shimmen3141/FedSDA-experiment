# 統合検証: モデル別学習標本の保持

## 範囲と正本
Task1はLuna独立APPROVEDで完了。最終完成判定は未実施。
保持/追加/snapshot/サーバID一回対応のみ。正式登録pop/上書き、統計、評価store、NN/optimizer、全体runは対象外。
spec.jsonが承認/進捗、naming revision1が正式名、review.mdが独立判定と採否の正本。

## 実測
- Task1: 新module未存在でModuleNotFoundError、1 collection error/3.23秒/exit1。実装後45 passed/1.97秒/exit0。
- 空/混合列×8対応表の16条件を実旧BaseClient追加/emptyextend/apply_server_mappingへ全標本参照と順序で照合。
- snapshotの構造分離とpayload借用、空列作成、型派生/bool拒否、後尾不正入力でstate保持、巨大ID、容量制限なし/重複保存を確認。
- Task1対象Ruff成功、Pyright0 errors/0 warnings、git diff --check成功。固定748c3aaとの旧production/両golden/旧回帰test差分なし。

## 旧所見と限界
- Task2: 94 passed/2.20秒/exit0。48条件（4対応表×batch1/3/7×seed2×保有集合2）で各3反復の実旧samplerと新snapshot→samplerを照合。全batch Tensor/モデル順/終端Random一致、保持器stateとglobal Random不変。
- 無効Tensorは保持時に参照だけ保存し、未保有/不足時skip、参加時にsamplerがRNG消費前に拒否する。接続test以外のproduction変更なし。
- Task2追加後のRuff unused loop指摘は不要なループ名を除去して解消。全Ruff/diff-check成功。
新たな旧正常系の不具合は今回確認していない。正式登録の上書きは別責務として後続確認する。
payloadの完全不変性/deepcopyや、MemoryError・非同期更新のrollbackを保証しない。
旧全体goldenの確認と新旧部品対照を区別する。新全体run golden移植の完了とは主張しない。
