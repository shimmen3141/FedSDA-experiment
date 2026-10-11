# 構成`independent`（モデルが共有部を持たない構成）の、移植の規模の調査

- 調査日: 2026-10-11。状態: 調査だけ。移植するかどうかは、未定（ユーザーの判断）。
- 対象: 旧の`tools/run_main_ablation_suite.sh`の`independent`（新実装での構成名の案は`fedsda-independent-models`）。
- 基準: 旧実装は`748c3aa`、新実装は`7a472e9`。

## 旧実装での指定

構成名`FedSDA_NoCached_ClassESR_RestartingSoftRouting`。クラスタリングと候補の検証は、提案構成と同じ。予測は、Fixed-Shareの重み（`--soft-routing-context switching`）。共有部に関する引数（共同学習、勾配の結合、adapterのrank、集約後の再較正）は、渡さない。

提案構成との違いは、3つのクラスだけ。

| | 提案構成 | `independent` |
|---|---|---|
| モデル | `ResidualAdapterMLP`（共有部＋adapter＋出力層） | `SimpleMLP`（隠れ層＋出力層が、1つのまとまり） |
| client | 共有表現のmixin（`_SharedRepresentationFedSDAClientMixin`）つき | mixinなし |
| サーバ | `SharedBackboneFedSDANoCachedServer` | `FedSDANoCachedServer` |

したがって、違いは、mixin・サーバのsubclass・モデルのクラスが、上書きしている処理の全部で、それ以外には無い（旧のコードで、モデルの種類によって分かれる箇所は、ほかに、クラスタリングの診断の、パラメータの距離の対象だけ）。

## 違いの一覧（旧実装）

### モデル

1. パラメータは、1つのまとまり（共有部と概念固有部に、分けない）。optimizerは、モデルごとに1つで、全パラメータを更新する。
2. adapterが無い。構築のときの乱数の消費は、隠れ層→出力層の順（adapterを持たない構造と同じ）。

### client

3. 保有モデルは、共有部を持たない。配布を受け取った後に、全保有モデルを1つの共有部へつなぎ直す処理が、無い。
4. 採用した候補の共有部を、保有モデルの共有部へ写す処理が、無い（候補は、自分のパラメータを、そのまま持つ）。
5. 予測: モデルごとに、全体の順伝播を行う（共有特徴を1回だけ計算して使い回すことを、しない）。
6. ローカル学習: モデルごとに、順に学習する。データが足りるモデルについて、更新の回数だけ、batchを抽出して、そのモデルだけを更新する（提案構成は、更新ごとに、全モデルのbatchを抽出し、損失を平均して、共有部を1回、各概念固有部を1回ずつ更新する）。batchの抽出の順（乱数の消費の順）が、「モデル→更新」になる（提案構成は「更新→モデル」）。共有部の勾配の診断は、無い。
7. 集約後の、予測の重みの再較正: 無い（サーバが呼ばない。旧の`no-recalibration`と同じ動きが、この構成に含まれる）。
8. 計算量の計数: 共有部と概念固有部を分けた、optimizerの更新回数の計数が、無い。順伝播の標本数は、共有部・概念固有部とも、モデル×標本の延べ数。

### サーバ

9. 集約: モデルのIDごとに、全パラメータを、標本数で加重平均する（提案構成は、共有部を、clientごとに1回、全モデルの標本数で加重平均し、概念固有部を、IDごとに加重平均する）。
10. 配布: 全モデルの全パラメータを、全clientへ配る。
11. 統合: クラスタの、全パラメータを、加重平均する（提案構成でも、同じ関数だが、共有部は、どのモデルでも同じ値）。
12. 通信量: 集約・配布・クロス評価の、どの転送も、モデルごとに、全パラメータを数える。クロス評価では、同じclientへの、重複した送信を、省かない（提案構成は、共有部を、clientごとに1回だけ数える）。
13. 最終のパラメータの数: 全モデルの、全パラメータの合計（提案構成は、共有部を1回だけ数える）。
14. クラスタリングの診断の、パラメータの距離: 全パラメータが対象（提案構成は、共有部を除く）。

## 新実装で、変更が要る箇所

新実装は、「clientが、共有部を1つ持ち、保有モデルは、概念固有部だけを独立に持つ」ことを、学習・予測・集約・配布・統合・通信量の計数の、全部で前提にしている。共有部に関する名前（共有部の型、共有特徴、共有部のoptimizer、パラメータの分け方）を参照するmoduleは、`src`に26、`tests/refactoring`に50ある。

| 違い | 新実装の、変更が要る箇所 | 変更の中身 |
|---|---|---|
| 1・2 | 分類器（`learning/models/residual_adapter_classifier.py`）、モデル構造の設定 | 分類器は、共有部を渡さなければ、自分の特徴抽出部を持てる（今の実装で可能）。adapterを持たない構造（`IdentityFeatureAdapter`）と組み合わせれば、旧の`SimpleMLP`と、同じ構造になる。新しい構造の名前を足す |
| 3 | 保有モデルの共有部のつなぎ直し（`learning/training/held_model_shared_feature_reconnection.py`）、配布の適用（`runtime/global_model_distribution_application.py`） | この構造では、つなぎ直さない。配布の適用は、モデルごとに、全パラメータを置く |
| 4 | 採用した候補の共有部の統合（`learning/training/adopted_candidate_shared_feature_integration.py`）、採用の登録（`runtime/adopted_candidate_initial_local_registration.py`） | この構造では、統合しない |
| 5 | 標本1件の予測（`runtime/observed_sample_prediction.py`）、損失の評価（`learning/prediction/classifier_bounded_loss_evaluation.py`の、共有特徴からの評価）、クロス評価（`runtime/client_model_cross_evaluation.py`） | モデルごとに、特徴抽出から行う |
| 6 | 共同更新（`learning/training/joint_model_parameter_update.py`）、その繰り返し（`held_model_joint_training_iterations.py`）、学習要求の処理（`runtime/held_model_training_request_handling.py`）、共有部のoptimizerの保持（`shared_parameter_optimizer_state_holder.py`）、保有モデルの学習状態（`held_model_training_state_registry.py`） | モデルごとに順に学習する関数を、新しく作る。モデルごとに、特徴抽出部のoptimizerを持たせる（旧の「1つのoptimizer」と、数値は同じになる見込み: Adamは、パラメータごとに独立）。乱数の消費の順を、旧と合わせる |
| 7 | 設定 | 再較正をしない方針（`keep_weights_without_recalibration`）との組にする（追加の実装は、不要） |
| 8 | 計算量の計測（`learning/models/model_computation_measurement.py`）、指標の導出 | 計測は、特徴抽出部の順伝播を、モデルごとに数えるので、そのまま使える見込み。optimizerの更新回数の、旧との対応を、確かめる必要がある |
| 9〜13 | 登録と集約（`runtime/server_model_registration_and_aggregation.py`）、配布（`runtime/global_model_distribution.py`）、統合（`runtime/model_clustering_and_consolidation.py`）、クロス評価（`runtime/model_cross_evaluation.py`）、通信量の記録、指標の導出（`runtime/fedsda_run_metric_derivation.py`） | 「共有部を1回、概念固有部をモデルごと」の扱いを、「全パラメータを、モデルごと」へ分ける。通信量の数え方（重複を省かない）を、旧と合わせる |
| 14 | 統合（同上） | パラメータの距離の対象を、全パラメータにする |
| 全体 | clientの組立て（`runtime/fedsda_run_client.py`）、参加者の準備（`runtime/fedsda_run_participant_factory.py`）、事前学習（`runtime/initial_model_pretraining.py`）、標本1件の処理（`runtime/observed_sample_processing.py`）、候補の学習と構築 | 共有部と、そのoptimizerを、関数の引数として受け渡している。この構造での扱いを、決める必要がある |

## 規模の見積り（実測ではない）

- これまでの9構成のうち、変更が最も大きかったのは、ADWINの2構成（差分は、約1000行、26ファイル）。`independent`は、違いの項目が14あり、そのうち、学習（6）と、サーバの4つの処理（9〜13）が、それぞれ、ADWINの移植と同じ程度か、それ以上の変更になる見込み。全体で、ADWINの移植の、3〜4倍の見込み。
- 危険: 学習の、乱数の消費の順と、optimizerの分け方が、旧と、ビット単位で合うかどうかは、実装して照合するまで、分からない。サーバの集約の、加算の順（モデル→client）も、合わせる必要がある。
- 影響: 共有部を前提にした型の検査と、単体testが多い（testは50ファイル）。提案構成の経路を変えない形（構造での分岐を足す）にしても、引数の受け渡しが、広い範囲で変わる。

## 進め方の案

1. **移植する**: 上の項目を、3つのspecに分ける（(a)モデル構造と、clientの予測・学習、(b)サーバの集約・配布・統合・通信量、(c)構成と、実旧の全体runとの照合）。(a)と(b)は、単体では、実旧の全体runと照合できないので、部品の対照（実旧のclient・サーバと、同じprocessで照合）を、それぞれに置く。
2. **移植しない**: この構成だけ、旧実装の結果を使う（旧実装を、この構成のために、残す）か、論文のablationから外す。
3. **形を変えて移植する**: 新実装の構造のまま、「共有部を持つが、モデルごとに、特徴抽出部を持つ」ことに相当する構成を、別に定義する（旧の`independent`と、数値は一致しない。旧実装との照合が、できない）。

主担当の考え: 論文で、「共有部を持つこと」の効果を示すのに、この構成が要るなら、1。要らないなら、2（外す）。3は、旧実装と照合できないので、勧めない。
