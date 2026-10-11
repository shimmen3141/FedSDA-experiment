# 命名表: fedsda-without-concept-specific-adapter

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `shared_backbone_without_residual_adapter` | `ModelArchitectureSettings.model_architecture_name`の値 | 共有部と、概念別の分類層だけの構造 |
| `IdentityFeatureAdapter` | `learning/models/nonlinear_residual_adapter.py` | 共有特徴をそのまま返す、パラメータを持たない部品 |
| `WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME`（値`fedsda-without-concept-specific-adapter`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
