# Requirements Document

## Introduction
新実装の分類器`ResidualAdapterClassifier`（`learning/models/residual_adapter_classifier.py`）は、設定のモデル構造の名前で、3つの構造を作る: 共有部＋残差adapter（提案構成）、共有部だけ（adapterを持たない構成）、共有部を持たない構造（独立したモデル。FedDrift・FedAvgも使う）。クラスの名前と、属性の名前`residual_adapter`は、提案構成の構造のままで、ほかの2つの構造の中身と合わない（adapterを持たない構造では、共有特徴をそのまま返す部品が、共有部を持たない構造では、モデルごとの隠れ層が、`residual_adapter`の場所に入る）。

このspecは、クラス・module・属性の名前を、構造に依らない名前へ改める（[IMPROVE-014](../../../docs/research/improvement-candidates/improve-014-rename-classifier-that-may-have-no-adapter.md)。2026-10-13、ユーザーが、整理として採用を了承。同日、属性名も変えること——重要動作のgoldenの、パラメータ名のキーを、機械的に書き換えること——を、ユーザーが選んだ）。**値は変えない**（名前だけ）。

## Boundary Context
- 対象: 新実装の分類器のクラス名・module名・属性名`residual_adapter`と、それを使う`src`・`tests/refactoring`・直下のscriptの全箇所、依存境界の登録、重要動作のgolden（基準版1〜3、WindowsとLinuxの6ファイル）の中の、パラメータ名のキー、文書。
- 対象外: 分類器の動作（構造の作り方、乱数の消費の順、順伝播）、残差adapterの部品そのものの名前（`NonlinearResidualAdapter`・`nonlinear_residual_adapter.py`。提案構成のadapterを指す名前として、正しい）、設定の項目と値の名前（`model_architecture_name`の3つの値、`residual_adapter_requested_rank`。保存するrun設定とrunの識別子を変えない）、属性`feature_extractor`・`classification_layer`の名前、固定旧。
- 用語: 「パラメータ名」は、分類器のパラメータの、PyTorchの名前（属性の名前をつないだもの。例: `residual_adapter.feature_compression.weight`）。サーバのグローバルモデルのパラメータは、この名前をキーに持つ。

## Requirements
### Requirement 1: 構造に依らない名前
**Objective:** As a 開発者, I want 分類器の名前が、3つの構造のどれにも当てはまること, so that 名前から、中身を取り違えない。
#### Acceptance Criteria
1. The FedSDA system shall 分類器のクラスとmoduleを、特定の構造（残差adapter）を指さない名前にする。
2. The FedSDA system shall 分類器の、特徴抽出部と分類層の間の部品（残差adapter、共有特徴をそのまま返す部品、モデルごとの隠れ層の、どれかが入る場所）の属性を、3つのどれにも当てはまる名前にする。
3. The FedSDA system shall 前のクラス名・module名・属性名を、残さない（別名を設けない）。
4. The FedSDA system shall 残差adapterの部品、設定の項目と値の名前、ほかの属性の名前を、変えない。

### Requirement 2: 値と動作を変えない
**Objective:** As a 研究者, I want この改名が、実験の値を変えないこと, so that これまでの結果と、そのまま比べられる。
#### Acceptance Criteria
1. The FedSDA system shall 全部の構成で、このspecの前と、同じ結果（全部の状態・指標・記録の表・乱数の消費）にする。パラメータの登録の順（特徴抽出部→間の部品→分類層）を、変えない。
2. The FedSDA system shall 保存するrun設定と、runの識別子を、変えない。
3. The FedSDA system shall パラメータ名のうち、改名した属性の部分だけが変わり、ほかの部分（`feature_extractor.`、`classification_layer.`、部品の中の名前）は変わらないようにする。パラメータ名の接頭辞で、共有部を見分ける処理（`feature_extractor.`）は、影響を受けない。
4. The validation shall 実旧との照合test、構成の照合、掃引のrunの識別子の期待値を、期待値を変えずに、通し続ける（旧のパラメータ名との対応を書いたtestの中の、新の側の名前は、替える）。

### Requirement 3: goldenは、キーの名前だけを書き換える
**Objective:** As a 開発者, I want 重要動作のgoldenが、値を作り直さずに、新しいパラメータ名と合うこと, so that 改名の前後で、同じ値を照合し続けられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜3の、WindowsとLinuxのgoldenの、6つのファイルについて、パラメータ名のキーの、改名した属性の部分だけを、機械的に書き換える（実行し直して作り直さない）。
2. The validation shall 書き換えの前後で、キーの名前の対応を除いて、内容（値のdigest・形・型、ほかの全部の項目）が同じことと、書き換えた数（1ファイルあたり96）を、確かめる。ファイルの書式（キーの並べ方、字下げ）は、goldenを生成するときと同じにする。
3. The validation shall 書き換えた後、Windowsの基準版1〜3のgoldenの照合が、通ること（Linuxのgoldenは、同じ書き換えを適用する。Linuxでの照合は、WSLで、通る範囲を確かめる）。
4. The report shall goldenを書き換えたこと（キーの名前だけ。値は同じ）と、その確かめ方を、基準の文書へ記録する。

### Requirement 4: 依存の境界と、記録を合わせる
**Objective:** As a 開発者, I want 依存の登録と文書が、新しい名前と一致すること, so that 次の変更で、名前を取り違えない。
#### Acceptance Criteria
1. The validation shall 依存境界のtestの登録を、新しいmoduleの名前とクラスの名前へ替える（期待を、弱めない）。
2. The validation shall 前の名前が、`src`・`tests/refactoring`・直下のscript・goldenに残っていないことを、確かめる（残差adapterの部品と、設定の名前として現れる箇所は、除く）。
3. The report shall IMPROVE-014の記録と一覧、いまの構造を説明する文書、再開案内を更新する。
