# 設計: 保有モデル学習状態のID付替え

revision: 1

## 境界・依存

既存HeldModelTrainingStateRegistryの所有辞書内だけを変更。NN/optimizer生成・clone・reset、他owner・現在帰属ID・正式登録/通信は上位。
既存のexact依存を増やさず、既存ASTを再検証する。private改変/並行変更/MemoryErrorのrollbackは通常契約外。

## Interface

`reassign_held_model_training_state_id(*, original_model_id:int, reassigned_model_id:int)->None`。
既存private `_validate_model_id`へ`parameter_name:str="model_id"`を加え、既存呼出のmessageを維持しながら両新引数の拒否項目を区別する。
両ID検査→元欠落ならreturn→元record取得→変更先IDと同じNN/管理器の新HeldModelTrainingState作成→元pop→先代入。
新record生成は辞書変更前。NN/parameter/optimizer対応は既存登録済み参照をそのまま使い、再登録/再検証・resetを呼ばない。
異なる既存先は位置保持/上書き。同IDでも新wrapperを作りpop後末尾。古いwrapper/bindingは変更しない。戻り値None。

## File Structure Plan

|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/training/held_model_training_state_registry.py|変更|単一ID付替えと拒否項目名|
|tests/refactoring/test_held_model_training_state_id_reassignment.py|新規|実旧順序/参照/拒否/学習継続と統計上位接続|
|対象spec・steering resume/roadmap|新規/更新|承認/実測/現在地|

## 検証trace

|要件|検証|
|---|---|
|1.1–1.4|実旧confirmの負元ID×既存/未登録/欠落/同ID、独立NN/owner先上書き、空、汎用signed/同ID期待順|
|2.1|両IDのbool/float/str/None/int派生/NumPy int拒否、空/元有/先有、一覧record/NN値grad/optimizer保持|
|2.2|古いstate/binding/一覧ID保持、新wrapperのみID変更、借用NN/owner identity|
|2.3|蓄積state保持→通常owner resetを新bindingへ反映/古いbinding固定、実旧class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件・3共同更新で付替え前後loss/全parameter/grad/optimizerをexact照合|
|2.4|統計store付替えは上位testだけで明示実行、registry操作前後の統計/pending保持、3乱数/defaults、既存AST/fresh新CPU/旧非import/全golden・品質/hash/JUnit|

旧実モデルのpopを含む実BaseClient.confirm_model_registrationをtest-only oracleとして使用する。旧NN構造のprefix対応は既存helperへ委譲。
新全体runのgoldenを主張せず、固定旧production/golden/許容差は変更しない。
