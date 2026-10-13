# Design Document

## Overview
分類器のクラス・module・属性`residual_adapter`の名前を、3つの構造のどれにも当てはまる名前へ替える。動作は変えない。属性の名前は、パラメータ名の一部なので、重要動作のgoldenの、パラメータ名のキーを、機械的に書き換える（値は作り直さない）。

### Goals / Non-Goals
- Goals: 名前から、分類器の中身を取り違えないこと。値を変えないこと。goldenを、作り直さずに、新しい名前と合わせること。
- Non-Goals: 分類器の構造・乱数の消費・順伝播の変更、残差adapterの部品の改名、設定の項目と値の改名、ほかの属性（`feature_extractor`、`classification_layer`）の改名。

## Boundary Commitments
- This Spec Owns: 分類器のクラス名・module名・属性名、その全部の使用箇所の付替え、依存境界の登録、goldenの6ファイルのキーの書き換え、文書。
- Out of Boundary: 分類器の本体の処理（1行も変えない。名前だけ）、`NonlinearResidualAdapter`・`IdentityFeatureAdapter`・`ModelOwnedHiddenLayers`の名前と処理（docstringの中の、分類器の属性の名前は、替える）、設定、固定旧。
- Revalidation Triggers: 分類器へ、4つ目の構造を足すとき。パラメータ名を、保存する成果物へ入れるとき。

## Architecture

### いま
- `learning/models/residual_adapter_classifier.py`: `ResidualAdapterClassifier`（`Module`）。属性は、登録の順に、`feature_extractor`（共有の特徴抽出部。共有部を持たない構造では、隠れ層の無いもの）、`residual_adapter`（下の3つのどれか）、`classification_layer`、`output_activation`。同じmoduleに、定数`INDEPENDENT_MODELS_ARCHITECTURE_NAME`。
- `residual_adapter`の場所に入る部品（`learning/models/nonlinear_residual_adapter.py`）:

| モデル構造の名前（設定の値） | 部品 | パラメータ |
|---|---|---|
| `shared_backbone_residual_adapter`（提案構成） | `NonlinearResidualAdapter` | `feature_compression`、`feature_expansion` |
| `shared_backbone_without_residual_adapter` | `IdentityFeatureAdapter`（共有特徴を、そのまま返す） | なし |
| `independent_models_without_shared_backbone` | `ModelOwnedHiddenLayers`（モデルごとの隠れ層） | `hidden_layers.<番号>` |

- 使う箇所: クラス名とmodule名は、`src`の32ファイル、`tests/refactoring`と直下の22ファイル。属性`residual_adapter`の読取りは、`src`の10ファイル（概念固有部のパラメータを集める箇所）と、testの約30ファイル。testには、旧のパラメータ名（`adapter.down.`・`adapter.up.`）と、新のパラメータ名（`residual_adapter.feature_compression.`・`residual_adapter.feature_expansion.`）の対応を、文字列で書いた箇所がある。
- パラメータ名を、文字列として読む処理: サーバの集約が、接頭辞`feature_extractor.`で、共有部を見分ける（`runtime/server_model_registration_and_aggregation.py`）。`residual_adapter.`を、文字列として読む処理は、`src`に無い。パラメータ名の並べ替え（名前の順）に依る計算も、`src`に無い（パラメータは、登録の順で扱う）。
- パラメータ名は、実験の成果物（runの保存ファイル）には、入らない。重要動作のgolden（`tests/refactoring/behavior_regression_goldens/v1〜v3/`の、WindowsとLinuxの6ファイル）には、グローバルモデルのパラメータの項目（`cases.<ケース>.global_parameters`）のキーとして、入る（`<モデルID>.residual_adapter.…`。1ファイルあたり96）。値のdigest（`sha256`・`shape`・`dtype`）は、配列の中身だけから作り、名前を含まない。

### これから

| いま | これから |
|---|---|
| `learning/models/residual_adapter_classifier.py` | `learning/models/model_architecture_classifier.py` |
| `ResidualAdapterClassifier` | `ModelArchitectureClassifier`（構造を、`ModelArchitectureSettings`で決める分類器） |
| 属性`residual_adapter` | 属性`model_owned_feature_transform`（特徴抽出部の出力を、分類層の入力へ変える、モデルごとに持つ部品） |
| パラメータ名`residual_adapter.…` | `model_owned_feature_transform.…`（ほかの部分は同じ） |

- 分類器の本体は、名前の付替えだけ（属性の代入の順——`feature_extractor`→間の部品→`classification_layer`→`output_activation`——を変えないので、パラメータの登録の順と、乱数の消費の順は、同じ）。順伝播の中の局所名（`adapted_features`）も、部品に合わせて替える。
- 定数`INDEPENDENT_MODELS_ARCHITECTURE_NAME`は、新しいmoduleに、そのまま置く。
- `nonlinear_residual_adapter.py`の3つの部品は、名前も処理も変えない。docstringの「分類器の`residual_adapter`の場所」を、新しい属性の名前へ替える。
- testのファイル名のうち、分類器のクラス名を含むもの（`test_residual_adapter_model_architecture.py`——中身は、分類器の全体を、旧のモデルと照合するtest——）は、`test_model_architecture_classifier.py`へ替える。`test_classifier_without_residual_adapter.py`は、adapterを持たない構造のtestで、名前は実態と合うので、変えない。testの関数名のうち、`test_residual_adapter_classifier_…`・`test_residual_adapter_model_…`は、新しいクラス名に合わせて替える（testのidが変わる。数は同じ）。

### goldenの書き換え
- 対象: `tests/refactoring/behavior_regression_goldens/v{1,2,3}/`の6ファイル。
- 方法: JSONを読み、`cases.<ケース>.global_parameters`のキーのうち、`<モデルID>.residual_adapter.`で始まる部分を、`<モデルID>.model_owned_feature_transform.`へ替え、goldenを生成するときと同じ書式（`ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False`、末尾に改行、改行はLF）で書く。値は読んだまま（実行し直さない）。
- 確かめること:
  1. 書式が同じこと: 書き換えの前のファイルを、読んで、そのまま同じ書式で書いたものが、もとのファイルと、バイト単位で一致する（書き換えで変わるのが、キーの名前と、キーの並び——名前の順——だけになる）。
  2. 内容が同じこと: 書き換えの後のJSONの、キーの名前を、もとへ戻したものが、書き換えの前のJSONと、等しい。替えたキーの数が、1ファイルあたり96。`residual_adapter.`を含むキーが、`global_parameters`の外に無い。
  3. Windowsの基準版1〜3の照合（`test_important_behavior_regression_goldens.py`）が、書き換えた後のファイルで通る。変えたgoldenを検出するtest（同じファイルの、実際のgoldenを変えたら検出するtest）も通る。
  4. Linuxのファイルは、同じ書き換えを適用する。WSLで、goldenのtestを実行して、結果を記録する（WSLは、Linuxの基準環境と、数値計算の版が違うので、これまでも、環境が合わないケースは、照合せずに飛ばす）。
- goldenの中の、生成の記録（生成したときのsourceのcommitなど）は、変えない（値を作り直していないので）。書き換えたことは、基準の文書と、このspecの記録へ書く。
- 書き換えのscriptは、1回だけ使う。repoへは置かず、手順と確かめた結果を、記録へ書く。

### 変えないもの（確かめる点）
- パラメータの登録の順: `named_parameters()`・`state_dict()`の順は、属性の代入の順で決まる。名前の順ではない。改名で、順は変わらない。
- パラメータ名を、キーとして持つ・戻す処理（`classifier_parameter_snapshot.py`の`named_parameters()`・`state_dict()`、各所の`load_state_dict(..., strict=True)`、サーバのグローバルモデルのパラメータ）: 取り出しと、戻しは、同じ実行の中の、同じクラスの分類器の間で行う（保存したファイルから、パラメータを読み戻す処理は、無い）。名前が、取り出す側と戻す側で同時に変わるので、対応は保たれる。確かめ方: パラメータのsnapshot・配布の適用・候補の構築・参照モデルの固定のtest（旧との照合）が、期待値を変えずに通ること。3つの構造のそれぞれで、`named_parameters()`の名前の並びが、属性の部分を除いて、改名の前と同じこと（新しい名前の確認のtestで、並びを、文字列の列として確かめる）。
- 接頭辞`feature_extractor.`で共有部を見分ける処理: 新しい属性の名前は、`feature_extractor.`で始まらない。
- 設定の値`shared_backbone_residual_adapter`・`shared_backbone_without_residual_adapter`、項目`residual_adapter_requested_rank`、部品`NonlinearResidualAdapter`と、そのmodule名: 変えない（機械的な置換の対象から、外す）。
- 型の検査の文言に、クラス名が入るもの（「classifierはexact ResidualAdapterClassifier…」）は、新しい名前へ替わる（例外の型は同じ）。文言を照合するtestがあれば、合わせる。

## Testing Strategy
- 分類器の全体のtest（旧のモデルとの、構築の順・全状態・順伝播・勾配の照合）、adapterを持たない構造・共有部を持たない構造のtest、パラメータのsnapshotのtest: 名前だけを替えて、期待値を変えずに通る。旧のパラメータ名との対応（`adapter.down.`↔`…feature_compression.`）の、新の側の文字列を替える。
- 新しい名前の確認: 3つの構造のそれぞれで、分類器の、間の部品の型と、パラメータ名の接頭辞（`feature_extractor.`、`model_owned_feature_transform.`、`classification_layer.`だけ。前の名前が無い）を確かめるtestを足す（既存のtestに同じ確認があれば、それを替える）。
- 依存境界のtest: moduleの名前とクラスの名前を替える（期待は同じ）。
- 重要動作のgolden: 上の「goldenの書き換え」の1〜3。
- 全pytest（実旧との照合、構成の照合、掃引の識別子の期待値）、Ruff、Pyright。
- 前の名前の`grep`: `ResidualAdapterClassifier`、`residual_adapter_classifier`、属性としての`residual_adapter`（`.residual_adapter`、`residual_adapter.`で始まるパラメータ名）が、`src`・`tests/refactoring`・直下のscript・goldenに、0件。残してよいのは、`NonlinearResidualAdapter`・`nonlinear_residual_adapter`・設定の値と項目の名前、旧実装のクラス名（`ResidualAdapterMLP`）を指す箇所。

## Risks
- 機械的な置換が、変えない名前（設定の値、残差adapterの部品）まで替える → 置換は、クラス名・module名（完全な語）と、属性の形（`.residual_adapter`、文字列の中の`residual_adapter.`）に限り、置換の後、設定の値と部品の名前が、変わっていないことを、`git diff`と`grep`で確かめる。保存するrun設定と、掃引の識別子の期待値のtestが、設定の値の変化を検出する。
- goldenの書き換えの誤り → 上の確かめること1・2（書式と内容）と、3（照合）。
- Linuxのgoldenを、Linuxの基準環境で確かめられない → 書き換えは、値に触れない（確かめること2）。WSLでの実行結果と合わせて、未検証の範囲を記録する。
