# 命名表: fedsda-independent-models

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `independent_models_without_shared_backbone` | `ModelArchitectureSettings.model_architecture_name`の値 | 共有部を持たない構造 |
| `INDEPENDENT_MODEL_ARCHITECTURE_NAME` | `learning/models/residual_adapter_classifier.py` | 上の値（構造での分岐が、読む） |
| `ModelOwnedHiddenLayers` | `learning/models/nonlinear_residual_adapter.py` | モデルごとの隠れ層（Linear/ReLUの列） |
| `input_feature_count`・`hidden_layer_widths` | `ResidualAdapterClassifier`の属性 | 構築のときに指定された寸法（同じ構造の分類器を作るときに使う） |
| `parameter_optimizer_if_any` | `ParameterOptimizerState` | 現在のoptimizer。パラメータが空なら、None |
| `sequential_per_model_training` | `LocalTrainingSettings.local_model_parameter_update_strategy`の値 | モデルごとに順に学習する |
| `INDEPENDENT_MODELS_CONFIGURATION_NAME`（値`fedsda-independent-models`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
