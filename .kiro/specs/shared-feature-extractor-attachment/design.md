# 設計: 共有特徴抽出部への再接続

## Overview / Boundary Commitments
分類器は適合する既存抽出部への参照を交換する。概念固有adapter/分類層/activationを維持する。
所有範囲は型/構造/寸法の事前検証と参照交換だけ。optimizer/RNG/モデル一覧/共有元選択/parameterロード/全体runは所有しない。
Allowed Dependenciesは既存モデルmoduleとTorch公開型/演算の現行境界を維持、新importなし。
Revalidation Triggersは接続寸法・参照所有・検査/交換順変更。共同更新/optimizer owner/将来clientを再検証する。

## API / Contracts
ResidualAdapterClassifier.attach_shared_feature_extractor(*,shared_feature_extractor:SharedFeatureExtractor)->Noneを追加。
exact型を検査し、元feature_extractorのinput_feature_count/hidden_layer_widthsを期待寸法として接続先validate_structureを呼び、成功後にfeature_extractor参照を交換する。
既存constructorのoptional Noneとは違い、再接続はNone不可。ValueErrorで拒否し、値/grad/接続/外側optimizerは変更しない。
概念固有部を作り直さず、同参照再接続も検査して参照を保つ。新Tensor/NN生成なし。
元モデルは正常構築済みで、層/宣言寸法の外側改変は通常契約外。
外側が新接続先parameterの共有ownerを選択する。個別ownerは同じ概念固有parameter列を保持するためresetだけできる。
旧attach_backboneの再生成までを照合するときは新接続＋個別owner reset＋新借用batchを明示する。接続単体のoptimizer無操作と混同しない。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/models/residual_adapter_classifier.py|変更|公開接続操作のみ追加|
|tests/refactoring/test_shared_feature_extractor_attachment.py|新規|参照/拒否/forwardと実旧attach/共同更新対照|
|tests/refactoring/test_single_run_dependency_boundaries.py|参照のみ|既存モデルguardでoptimizer/上位非依存を確認|
|.kiro/specs/shared-feature-extractor-attachment/*|新規|承認/仕様/証拠|
|.kiro/steering/roadmap.md|変更|進捗|

## Requirements Traceability / Testing Strategy
|要件|証拠|
|---|---|
|1.1/1.2/1.3/1.5|class2/4×hidden3構成×同/別参照12条件、concept/双方parameter値grad参照保持とforward新経路|
|1.4|None/型派生/寸法違い/層違い/float64/meta/shape違い拒否時の接続/値grad保持|
|2.1|接続だけで個別/旧共有owner状態保持、Python/Torch RNG不変|
|2.2|実旧attach_backboneと新接続＋個別reset、class2/4×optimizer3×接続先既存optimizer有無2の12条件で共同更新3stepの全loss/値grad/state一致|
接続先は既存SharedFeatureExtractorを用い、旧SharedFeatureBackbone.netのstateをnew.hidden_layersへtest内で対応付ける。
未実装methodのAttributeErrorで実RED後に実装GREEN。依存importは変更しないので既存guardを実行し、架空のAST REDは作らない。
全pytest/旧11最終3golden、fresh新CPU接続/学習smoke、Ruff/format/Pyright/pip、旧固定差分/源hashを保存する。
Pyrightは共有venvの絶対pythonpathとrequire_escalatedで子Pythonを起動する。
