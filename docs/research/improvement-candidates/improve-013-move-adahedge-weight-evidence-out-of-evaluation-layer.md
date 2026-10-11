# IMPROVE-013. AdaHedgeの重みの証拠を、評価の層から移す

- 記録日: 2026-10-11。種別: 層の置き場所。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/evaluation/adahedge_weight_evidence.py`・`evaluation/adahedge_weight_evidence_collection.py`と、それを読む箇所（`runtime/`の7つのmodule）。

## 確認した事実

AdaHedgeの重みの証拠は、提案構成では、比較用（予測と学習の判断に使わない）なので、評価の層（`evaluation/`）に置いた。構成`fedsda-adahedge-prediction-weights`（予測の結合の方式が`adahedge_weighted_prediction`）では、AdaHedgeの重みが、予測に使う重みになる。予測に使う重みの出どころが、評価の層にある形になった。

Fixed-Shareの重みの保持（`FixedShareWeightController`）は、手法の層（`methods/fedsda/prediction_combination/`）にある。

証拠の集まり（`AdaHedgeWeightEvidenceCollection`）は、全標本の証拠と、真の概念別の証拠（真の概念を使うので、常に比較用）の両方を持つ。

## 案

- 証拠そのもの（`AdaHedgeWeightEvidence`）を、`methods/fedsda/prediction_combination/`へ移す。
- 集まりは、全標本の証拠（予測に使いうる）と、真の概念別の証拠（比較用）を、別のownerに分けるか、評価の層に残して、手法の層の型を使う形にする。

## 挙動への影響

値は変わらない（置き場所とimportだけ）。依存境界のtest（moduleごとの許可集合）の登録が変わる。

## 必要な検証

全pytest（依存境界、goldenの3ケース、構成の照合）が、変わらず成功すること。採否・変更commitは未定。
