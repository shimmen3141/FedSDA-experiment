# Design Document: fedsda-without-concept-specific-adapter

## 変更

- `ModelArchitectureSettings.model_architecture_name`へ、値`shared_backbone_without_residual_adapter`を足す（要求rankは、この構造では使わない）。
- `learning/models/nonlinear_residual_adapter.py`へ、`IdentityFeatureAdapter`を足す: 共有特徴をそのまま返す部品で、パラメータを持たない。入力の検査は、残差adapterと同じ。
- `ResidualAdapterClassifier`は、この構造のとき、`residual_adapter`へ、`IdentityFeatureAdapter`を置く。
- 計算量の計測（`learning/models/model_computation_measurement.py`）は、概念固有部へ入力された標本数を、どちらの部品でも数える。
- 掃引: 構成`fedsda-without-concept-specific-adapter`を足す。

## 部品を置き換える形にした理由

分類器の概念固有部のパラメータは、多くの場所が、`residual_adapter`と`classification_layer`のパラメータの連結として読む（学習状態の登録、候補の学習、共同更新、配布の適用、事前学習ほか）。構造ごとに分類器のクラスを分けると、それらの全部を、構造で分けることになる。`residual_adapter`の場所へ、パラメータを持たない部品を置けば、連結は、分類層のパラメータだけになり、読む側は変えずに済む。パラメータの名前（`state_dict`）にも、adapterの項目が出ない。

分類器のクラスの名前（`ResidualAdapterClassifier`）は、この構造では、中身と合わない。名前を変えると、型として受け取る多くの関数と、依存境界の登録が変わるので、今回は変えず、改善候補として記録する。

## 検証

- `tests/refactoring/test_classifier_without_residual_adapter.py`（新規）: 同じ乱数の状態から作った分類器の、パラメータ・出力・構築の後の乱数の状態が、旧の`SharedBackboneMLP`と一致する（2クラスと多クラス）。提案構成の構造は、旧の`ResidualAdapterMLP`と同じ数の乱数を消費する。部品の入力の検査と、計算量の計測。
- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、構成名を替える）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestを足す。
