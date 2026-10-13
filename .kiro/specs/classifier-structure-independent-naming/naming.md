# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `ModelArchitectureClassifier` | `learning/models/model_architecture_classifier.py`（前は`ResidualAdapterClassifier`、`residual_adapter_classifier.py`） | 1つのモデルの分類器。構造（共有部＋残差adapter、共有部だけ、共有部なし）を、`ModelArchitectureSettings`のモデル構造の名前で決める。特徴抽出部、モデルごとの部品、分類層を持つ | `ModelArchitectureSettings`は、構造を決める設定。`NonlinearResidualAdapter`は、提案構成の構造で、モデルごとの部品として入る、残差adapterそのもの。`SharedFeatureExtractor`は、モデルの間で共有する特徴抽出部 |
| `ModelArchitectureClassifier.model_owned_feature_transform` | 同上 | 特徴抽出部の出力を、分類層の入力へ変える、モデルごとに持つ部品。構造により、`NonlinearResidualAdapter`、`IdentityFeatureAdapter`（そのまま返す）、`ModelOwnedHiddenLayers`（モデルごとの隠れ層）の、どれか。パラメータ名の一部になる | `feature_extractor`は、共有する側（共有部を持たない構造では、隠れ層の無いもの）。`classification_layer`は、最後の線形層。概念固有部（モデルごとのパラメータ）は、この部品と`classification_layer`のパラメータを合わせたもの |
| `INDEPENDENT_MODELS_ARCHITECTURE_NAME` | 同上（moduleだけが変わる） | 共有部を持たない構造の、モデル構造の名前（設定の値）。値は変えない | — |

## 無くなる公開名

| 公開名 | 代わり |
|---|---|
| `ResidualAdapterClassifier`、`learning/models/residual_adapter_classifier.py` | `ModelArchitectureClassifier`、`learning/models/model_architecture_classifier.py` |
| `ResidualAdapterClassifier.residual_adapter` | `ModelArchitectureClassifier.model_owned_feature_transform` |

## 名前を変えないもの

`NonlinearResidualAdapter`・`IdentityFeatureAdapter`・`ModelOwnedHiddenLayers`（`learning/models/nonlinear_residual_adapter.py`）、`ModelArchitectureClassifier`の、ほかの属性とメソッド（`feature_extractor`、`classification_layer`、`output_activation`、`attach_shared_feature_extractor`、`extract_shared_features`、`forward_from_shared_features`）、設定の値と項目（`shared_backbone_residual_adapter`、`shared_backbone_without_residual_adapter`、`independent_models_without_shared_backbone`、`residual_adapter_requested_rank`）。

## 比べた名前

- クラス: `ConfiguredModelArchitectureClassifier`（仕様レビューの候補。この分類器は、どの構造でも、設定で構造が決まるので、`Configured`を足しても、ほかの名前との区別は増えない。短い方を採る）、`SharedBackboneClassifier`（IMPROVE-014の記録の例。共有部を持たない構造に合わない）、`ConceptModelClassifier`（FedAvg・FedDriftのモデルは、概念に対応しない）、`ArchitectureConfiguredClassifier`（意味は同じだが、設定の型`ModelArchitectureSettings`と、語がそろわない）。
- 属性: `feature_adapter`（記録の例。共有部を持たない構造では、隠れ層の全部が入るので、adapterと呼べない）、`concept_specific_feature_layers`（概念固有部は、この部品と分類層を合わせたもの、という既存の用語とぶつかる。恒等の部品は、層を持たない）。
