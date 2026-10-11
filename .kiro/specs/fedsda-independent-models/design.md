# Design Document: fedsda-independent-models

## 考え方: 空の共有部と、モデルごとの隠れ層

新実装は、「clientが、共有部を1つ持ち、保有モデルは、概念固有部だけを独立に持つ」ことを、広い範囲で前提にしている。この前提を保ったまま、共有部を持たないモデルを、次の形で表す。

- 共有の特徴抽出部（`feature_extractor`）は、隠れ層を持たない（入力を、そのまま返す。パラメータなし）。
- 隠れ層は、分類器の`residual_adapter`の場所に、モデルごとに独立に持つ（`ModelOwnedHiddenLayers`）。
- 概念固有部（`residual_adapter`と`classification_layer`のパラメータ）が、モデルの全パラメータになる。

こうすると、予測（共有特徴を1回計算して、各モデルの概念固有部へ渡す）、サーバの集約・配布・統合（共有部を1回、概念固有部をモデルごと）、最終のパラメータの数、パラメータの距離は、今の実装のまま、旧の、共有部を持たない構成と、同じ結果になる（共有部が空なので、「共有部を1回」が、何もしない）。

変更が要るのは、次の点だけ。

| 点 | 変更 |
|---|---|
| モデル構造 | `ModelArchitectureSettings`へ`independent_models_without_shared_backbone`。分類器（`learning/models/residual_adapter_classifier.py`）は、この構造で、空の特徴抽出部と`ModelOwnedHiddenLayers`を作る。分類器へ、構築のときの`input_feature_count`と`hidden_layer_widths`を持たせ、同じ構造の分類器を作る箇所は、特徴抽出部の寸法でなく、これらを使う |
| 空の共有部のoptimizer | `ParameterOptimizerState`は、パラメータが空なら、optimizerを持たない（`parameter_optimizer_if_any`がNone。`parameter_optimizer`は拒否）。共有部のoptimizerを取り出す箇所は、`parameter_optimizer_if_any`を使う（共同更新は、もともと、空の共有部とNoneのoptimizerの組を受け付ける） |
| 学習 | `LocalTrainingSettings`へ`sequential_per_model_training`。学習要求の処理（`runtime/held_model_training_request_handling.py`）は、この方式のとき、モデルごとに、1モデルだけを渡して、共同学習の繰り返しを呼ぶ（batchの抽出と更新は、既存の部品。乱数は、そのモデルのbatchの抽出に、1回ずつ使われる） |
| 採用と配布 | 採用した候補の統合（`adopted_candidate_shared_feature_integration.py`）と、配布の後のつなぎ直し（`held_model_shared_feature_reconnection.py`）は、この構造では、概念固有部のoptimizerを作り直さない。配布の適用は、作り直すモデルのoptimizerを、どれも、作り直しの設定で作る |
| 候補 | 候補の構築と学習は、この構造では、空の共有部を受け付ける |
| サーバ | パラメータの分け方（`split_shared_and_concept_specific_parameters`）は、空の共有部を受け付ける。クロス評価の通信量は、共有部が空のとき、重複を省かない（[LEGACY-018](../../../docs/research/implementation-findings/legacy-018-cross-evaluation-download-volume-without-shared-part.md)） |
| 計算量の計測 | 隠れ層を持たない特徴抽出部は、数えない。`ModelOwnedHiddenLayers`の順伝播を、概念固有部の入口として数える |
| 設定の組 | 参加者の設定（`FedsdaRunParticipantSettings`）が、構造・学習の方式・再較正の方針の組を確かめる |
| 掃引 | 構成`fedsda-independent-models` |

空のパラメータを受け付けるのは、`ParameterOptimizerState`（パラメータの列とoptimizerだけを持つ、下位の部品。モデル構造を知らない）。空の共有部を、どの構造で許すかは、構造を知る側（候補の構築、候補の学習、参加者の設定の検査）が決める。共有部が空でないのに、optimizerが無い組は、共同更新・候補の学習・保有モデルのつなぎ直しの検査が、拒否する（独立レビューの指摘1への判断）。

## 旧と同じになる理由（主なもの）

- 1モデルの更新: 共同更新（`perform_joint_model_parameter_update`）へ、1モデルのbatchだけを渡す。損失は、（損失×件数）÷件数で、旧の`model.update`の損失と、同じ勾配になる（候補の学習と事前学習が、同じ形で、旧と一致している）。optimizerは、旧は、全パラメータに1つ。新は、概念固有部のoptimizerが、全パラメータを持つ（共有部が空なので）。
- optimizerの状態: 旧の、共有部を持たないモデルは、配布で作り直すと、新しいoptimizer（基準の学習率）を持つ。採用した候補と、一時モデルは、学習で蓄積した状態を、そのまま持つ。新の、共有部を持つ構造は、つなぎ直しで、概念固有部のoptimizerを作り直すので、この構造では、それをしない。
- 再較正: 旧のサーバは、この構成で、集約後の再較正を呼ばない。構成は、再較正をしない方針との組にする。

## 計算量の指標の意味

この構造では、共有部が無いので、共有部の計数（標本数、optimizerの更新、積和演算）は、全部0。旧は、共有部の標本数として、概念固有部と同じ数（モデル×標本の延べ数）を数える。照合では、旧の、共有部の標本数を、新の、概念固有部の標本数と比べる。

## 進め方の変更（調査の文書との差）

調査の文書は、3つのspec（clientの予測と学習／サーバ／構成と照合）に分ける案で、規模を、ADWINの移植の3〜4倍と見積もっていた。上の表し方で、サーバと予測の変更が、ほぼ不要になったので、1つのspecにまとめた。差分は、ADWINの移植と同じ程度。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、構成名を替える）。sine2の7条件で、指標と離散列を照合し、mnist2の2条件で、全体runの全状態を照合する。固有の経路のtestと、設定の組の拒否のtestを足す。
- `tests/refactoring/test_classifier_without_residual_adapter.py`: 分類器を、旧の`SimpleMLP`と照合する（パラメータ、出力、乱数の状態、計算量の計測）。
- `tests/refactoring/test_parameter_optimizer_state.py`: 空のパラメータの状態。
- 照合の土台: 実旧のモデルが共有部を持たない構成（`legacy_models_share_feature_extractor`が偽）では、旧の、共有部の標本数を、新の、概念固有部の標本数と比べ、新の、共有部の計数が0であることを確かめる。旧のパラメータの名前（`net.N`）から、新の名前への変換と、保有モデルの状態の照合（optimizerが1つのモデル）を、広げた。
