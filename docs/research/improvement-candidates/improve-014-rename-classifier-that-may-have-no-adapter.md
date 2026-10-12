# IMPROVE-014. adapterを持たない構造も扱う分類器の、クラスの名前

- 記録日: 2026-10-11。種別: 名前。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/learning/models/residual_adapter_classifier.py`の`ResidualAdapterClassifier`と、それを型として受け取る箇所（学習・runtimeの多くのmodule）。

## 確認した事実

構成`fedsda-without-concept-specific-adapter`（モデル構造`shared_backbone_without_residual_adapter`）では、分類器は、残差adapterを持たない（`residual_adapter`の場所に、共有特徴をそのまま返す`IdentityFeatureAdapter`を置く）。クラスの名前と、属性の名前（`residual_adapter`）は、提案構成の構造のままで、この構造の中身と合わない。

## 案

クラスを、構造に依らない名前（例: `SharedBackboneClassifier`）へ、属性を`feature_adapter`へ、改める。moduleの名前も合わせる。

## 挙動への影響

値は変わらない（名前だけ）。依存境界のtestの登録と、多くのimportが変わる。

## 必要な検証

全pytest（依存境界、goldenの3ケース、構成の照合）が、変わらず成功すること。採否・変更commitは未定。

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

採用（整理。値は変えない）。挙動を直す変更の後に行う。共有部を持たない構造も含めて、名前を決める。
