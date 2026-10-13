# Requirements Document

## Introduction
AdaHedgeの重みの証拠（累積損失と、混合性のgap）は、提案構成では比較用（予測にも学習の判断にも使わない）なので、評価の層（`evaluation/`）に置いた。その後、構成`fedsda-adahedge-prediction-weights`（予測の結合の方式が`adahedge_weighted_prediction`）で、全標本の証拠から作る重みが、予測に使う重みになった。予測に使う重みの出どころが、評価の層にある形になっている（[IMPROVE-013](../../../docs/research/improvement-candidates/improve-013-move-adahedge-weight-evidence-out-of-evaluation-layer.md)。2026-10-13、ユーザーが、整理として採用を了承）。

証拠の集まり（`AdaHedgeWeightEvidenceCollection`）は、全標本の証拠（予測に使いうる）と、真の概念別の証拠（真の概念を使うので、常に比較用）を、1つのownerに持つ。このspecは、証拠の計算の部品と、全標本の証拠のownerを、手法の層へ移し、真の概念別の証拠のownerを、評価の層に残して、2つに分ける。**値は変えない**（置き場所・ownerの分け方・名前だけ）。

## Boundary Context
- 対象: 新実装の、AdaHedgeの証拠の2つのmodule（`evaluation/adahedge_weight_evidence.py`、`evaluation/adahedge_weight_evidence_collection.py`）と、それを受け取る・組み立てる・読む`runtime/`のmodule、対応するtest、依存境界の登録、文書。
- 対象外: 証拠の計算の中身（式、初期化、再始動、再生）、予測・警報・再較正の処理の順序、設定の項目（増やさない・変えない）、保存するrun設定とrunの識別子、記録の表の列名、指標の名前、Fixed-Shareの重みの部品、固定旧。
- 用語: 「全標本の証拠」は、clientが観測した全部の標本で更新する、AdaHedgeの証拠（1つ）。「真の概念別の証拠」は、標本の真の概念IDごとに分けて更新する、診断用のAdaHedgeの証拠（概念IDごとに1つ。真の概念は、手法の判断には使えない情報）。

## Requirements
### Requirement 1: 予測に使いうる証拠を、手法の層が持つ
**Objective:** As a 開発者, I want 予測に使う重みの出どころが、手法の層にあること, so that 評価の層を読まずに、予測の結合の動作を追える。
#### Acceptance Criteria
1. The FedSDA system shall AdaHedgeの証拠の計算の部品（`AdaHedgeWeightEvidence`）を、手法の層の、予測の結合の場所（`methods/fedsda/prediction_combination/`）に置く。
2. The FedSDA system shall 全標本の証拠と、「予測の処理が証拠を更新する構成かどうか」を持つownerを、同じ場所に置く。このownerは、真の概念別の証拠を、持たず、作らず、返さない。
3. The FedSDA system shall 手法の層の、この2つのmoduleから、評価の層（`evaluation/`）と`runtime/`を、importしない。

### Requirement 2: 真の概念別の証拠を、評価の層の別のownerが持つ
**Objective:** As a 開発者, I want 真の概念を使う比較用の証拠が、手法の判断へ渡らないことが、引数の形で分かること, so that 比較用の値が、判断に混ざらないことを、呼び出しの形で確かめられる。
#### Acceptance Criteria
1. The FedSDA system shall 真の概念別の証拠のownerを、評価の層に置き、全標本の証拠を、持たせない。
2. The FedSDA system shall 真の概念別の証拠のownerを、それを更新・再始動する処理（標本の予測、集約後の再較正）と、その処理へ渡すだけの処理（標本1件の処理、client）だけが受け取るようにする。警報の処理、保持中の候補検証の進行、学習帰属の変更に伴う証拠の再始動は、受け取らない。
3. The FedSDA system shall 前の2つのmodule（`evaluation/adahedge_weight_evidence.py`、`evaluation/adahedge_weight_evidence_collection.py`）と、前のownerの名前を、残さない（別名を設けない）。
4. The FedSDA system shall 学習帰属の変更に伴う証拠の再始動の、moduleと関数の名前を、全標本の証拠だけを始め直すことが分かる名前にする（役割は変えない。前の名前を残さない）。

### Requirement 3: 値と動作を変えない
**Objective:** As a 研究者, I want この整理が、実験の値を変えないこと, so that これまでの結果と、そのまま比べられる。
#### Acceptance Criteria
1. The FedSDA system shall 全部の構成で、このspecの前と、同じ結果（全部の状態・指標・記録の表・乱数の消費）にする。状態の更新の順序（全標本の証拠の更新・再生、真の概念別の証拠の更新・再始動、Fixed-Shareの重みの更新・再生の、互いの順）を、変えない。
2. The FedSDA system shall 保存するrun設定と、runの識別子を、変えない（設定の項目を、増やさない）。
3. The FedSDA system shall 2つのownerの型の検査（exactな型だけを受け付ける）を、これまでと同じく、最初の状態の更新より前に行う。
4. The validation shall 実旧との照合test、重要動作のgoldenの基準版1〜3、構成の照合、掃引のrunの識別子の期待値を、期待値とgoldenのファイルを変えずに、通し続ける。

### Requirement 4: 依存の境界と、記録を合わせる
**Objective:** As a 開発者, I want 依存の登録と文書が、新しい置き場所と一致すること, so that 次の変更で、置き場所を取り違えない。
#### Acceptance Criteria
1. The validation shall 依存境界のtestの登録を、新しいmoduleと、受け取るmoduleについて、実際のimportと一致させる（前のmoduleの登録を、残さない）。移した証拠の部品と、新しい2つのownerの、3つのmoduleを、importの形を厳しく確かめるmoduleの一覧へ入れる。
2. The FedSDA system shall 評価の層から、手法の層を読む依存を、真の概念別の証拠のownerが、証拠の計算の部品（`AdaHedgeWeightEvidence`）を読む、1つだけにする。
3. The validation shall 2つのownerが、互いの証拠を持たないことを、testで確かめる（手法の層のownerに、真の概念別の証拠を得る口が無い。評価の層のownerに、全標本の証拠が無い）。
4. The validation shall 警報の処理、保持中の候補検証の進行、学習帰属の変更に伴う証拠の再始動の、入口の関数が、真の概念別の証拠のownerを受け取る引数を持たないことを、testで確かめる。
5. The report shall IMPROVE-013の記録と一覧、再開案内を、対応済みへ更新する。
