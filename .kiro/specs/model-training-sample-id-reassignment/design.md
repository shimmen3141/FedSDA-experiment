# 設計: モデル学習標本のID付替え

revision: 1

## 境界/依存

既存ModelTrainingSampleStoreの所有辞書のみを更新する。既存依存（__future__.annotations、同階層公開record2型）を増やさず、既存ASTを検証する。
評価標本/容量/NN/統計/保留・他ownerのID対応・sampler/学習実行は上位。private改変/並行変更/MemoryError rollbackは通常契約外。

## Interface

`reassign_model_training_samples_id(*, original_model_id:int, reassigned_model_id:int)->None`。
両IDをforで検査（exact int、不正TypeError、引数名と理由）。元欠落no-op。
登録済み元は内部listをpopして先へ代入。list/record/Tensorをコピーせず、内容も触らない。
既存先は位置維持で上書き。新先/同IDは末尾。既存append/snapshot/remapの契約は変更しない。

## File Structure Plan

|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/training/model_training_sample_store.py|変更|単一ID付替え|
|tests/refactoring/test_model_training_sample_id_reassignment.py|新規|実旧順序/参照/拒否/後続追加、抽出と共同更新接続|
|対象spec・steering resume/roadmap|新規/更新|承認/証拠/現在地|

## 検証trace

|要件|証拠|
|---|---|
|1.1–1.4|実旧負元confirmの先既存/欠落/同ID/元欠落、空/重複/非空列、Tensoridentityとモデル/標本順、signed直接期待値|
|2.1|両引数×不正6型×元先有無、一覧参照/全Tensor値・環境保持|
|2.2|過去一覧ID/列保持、付替え後追加と再取得の構造分離/借用参照|
|2.3|契約外payload（opaque object/float64/meta等）も移動のみ。samplerが数値契約を担う既存テストを再検証|
|2.4|12条件（class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結）で各3抽出・共同更新、最初の更新後に旧confirmと新store付替え。上位でbinding IDを更新し、batch/全loss/param/grad/両optimizer/Random終端をexact照合。3共有RNG/defaults保持、既存AST/fresh新CPU/固定旧golden・品質/hash/JUnit|

旧production/golden/許容差は固定。接続はtest-onlyで、新client/正式登録全体の完成を主張しない。
