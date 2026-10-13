# IMPROVE-013. AdaHedgeの重みの証拠を、評価の層から移す

- 記録日: 2026-10-11。種別: 層の置き場所。状態: 対応済み（2026-10-13。下の「対応」）。
- 対象: 新実装`src/federated_learning_experiments/evaluation/adahedge_weight_evidence.py`・`evaluation/adahedge_weight_evidence_collection.py`と、それを読む箇所（`runtime/`の10のmodule。記録の時点では、7つと数えていた）。

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

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

採用（整理。値は変えない）。挙動を直す変更の後に行う。

## 対応（2026-10-13）

[adahedge-weight-evidence-owner-separation](../../../.kiro/specs/adahedge-weight-evidence-owner-separation/tasks.md)で対応した。

- 証拠の計算の部品（`AdaHedgeWeightEvidence`）は、中身を変えずに、`methods/fedsda/prediction_combination/adahedge_weight_evidence.py`へ移した。
- 全標本の証拠のowner（`AllSampleAdaHedgeWeightEvidenceHolder`）は、同じ場所（手法の層）。真の概念別の証拠は持たない。
- 真の概念別の、診断用の証拠のowner（`TrueConceptAdaHedgeDiagnosticEvidenceCollection`）は、評価の層（`evaluation/true_concept_adahedge_diagnostic_evidence_collection.py`）。受け取るのは、標本の予測、集約後の再較正と、その通り道（標本1件の処理、client）だけ。警報の処理・候補検証の進行・証拠の再始動は、受け取らない。
- 前の集まり（`AdaHedgeWeightEvidenceCollection`）と、前の2つのmoduleは、削除した。証拠の再始動は、名前へ`all_sample`を足した。
- 値は変えていない（全pytest——実旧との照合、基準版1〜3、構成の照合——が、期待値を変えずに通過）。
