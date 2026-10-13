# Research & Design Decisions

## Summary
- Feature: adahedge-weight-evidence-owner-separation（IMPROVE-013）。2026-10-13、ユーザーが、整理（値を変えない）として採用を了承し、挙動を直す変更（LEGACY-016・002）の後に行う、と決めた。同日、「次に進んでください」の指示で着手。

## Research Log
- [IMPROVE-013の記録](../../../docs/research/improvement-candidates/improve-013-move-adahedge-weight-evidence-out-of-evaluation-layer.md): 証拠そのものを、手法の層へ移す。集まりは、「全標本の証拠と、真の概念別の証拠を、別のownerに分ける」か、「評価の層に残して、手法の層の型を使う」。決定の表には、「真の概念別の比較用証拠を、手法の判断へ流さない」とある。
- コード（`7ac8e16`）: 集まりを受け取る`runtime/`のmoduleは10。真の概念別の口（`get_true_concept_diagnostic_evidence`、`created_true_concept_ids`）を使うのは、標本の予測と、集約後の再較正の2つだけ。警報の処理・候補検証の進行・証拠の再始動は、全標本の証拠と、`evidence_is_updated_by_prediction`だけを使う。指標・記録の表は、全標本の証拠の、再較正の計数だけを読む。
- `AdaHedgeWeightEvidence`は、標準ライブラリだけに依存する。手法の層へ、そのまま移せる。
- 手法の層から、評価の層を読む依存は、いまも無い。真の概念ID（`true_concept`・`concept_id`）を扱う、手法の層のmoduleは、保留の標本の保持（`pending_sample_observation_store.py`。診断用に持つだけ）と、FedAvgのclientだけ。
- 依存境界のtestは、moduleごとの許可集合（明示の登録）で確かめる。新しいmoduleは、登録を足す。
- testの参照: `adahedge_weight_evidence`を含む行は、`tests/refactoring`の12ファイルで、約340行。

## Design Decisions
- 集まりを、2つのownerに分ける（記録の、1つ目の案）。理由: 「評価の層に残して、手法の層の型を使う」案では、予測に使う重みを持つownerが、評価の層に残り、記録が挙げた問題（予測に使う重みの出どころが、評価の層にある）が、直らない。集まりごと、手法の層へ移す案は、真の概念を使う比較用の証拠を、手法の層のownerが持つことになる。分ける案は、真の概念別の証拠が、手法の判断（警報の処理・候補検証の進行・証拠の再始動）へ渡らないことを、引数の形で示せる。
- 証拠の計算の部品（`AdaHedgeWeightEvidence`）は、手法の層に1つだけ置き、評価の層の、真の概念別のownerが、それを読む。同じ計算を、2か所に置かない。評価の層→手法の層の依存が1つ増えるが、評価の層→`learning/`の依存は、すでにある（`computation_cost_summary.py`）。
- 証拠の計算の部品の置き場所は、`methods/fedsda/prediction_combination/`にする（IMPROVE-013の記録の案）。比べた案: `learning/prediction/`（分類器の確率・損失の計算がある。評価の層→`learning/`の依存は、既にあるので、評価の層→手法の層の依存を足さずに済む）。採らない理由: AdaHedgeの証拠は、FedSDAの、保有モデルの予測を結合する重みの方式で、同じ役割のFixed-Shareの部品（`fixed_share_weights.py`）が、この場所にある。`learning/prediction/`の部品は、1つの分類器の計算で、役割が違う。手法の層に、診断用の場所（`methods/fedsda/diagnostics`のような）を作って、真の概念別のownerを置く案は、真の概念を使う比較用の状態を、手法の層へ入れることになるので、採らない。評価の層のownerが、予測の中で更新される状態を持つ点は、前の集まりと同じ（新しく増える形ではない）。構造の原則（[構造と命名](../../steering/structure.md)の「評価は記録を読み」）からは外れるが、この診断は、標本ごとの計算の途中の値（保有モデルごとの損失）を、その場で積む必要があり、記録から後で作り直せない（集約で、モデルが変わるため）。この点は、このspecでは変えない。
- 全標本の側のownerは、前の集まりの2つのproperty（`all_sample_weight_evidence`、`evidence_is_updated_by_prediction`）を、名前を変えずに持つ。`evidence_is_updated_by_prediction`を、引数として配る案は、警報の処理・候補検証の進行まで、引数が増えるので、採らない。
- 証拠の計算、処理の順序、設定は、変えない。提案構成で、使わない全標本の証拠を計算しない、という最適化は、値（記録の表の、AdaHedgeの重みの列）を変えるので、対象にしない。

## Risks & Mitigations
- 値を、気づかずに変える → 実旧との対照test、基準版1〜3、構成の照合、掃引の識別子の期待値が、期待値を変えずに通ること（全pytest）。
- 替え漏れ → 前の名前（`adahedge_weight_evidence_collection`、`AdaHedgeWeightEvidenceCollection`、`evaluation.adahedge_weight_evidence`）を、`grep`で、0件まで確かめる。

## References
- [IMPROVE-013の記録](../../../docs/research/improvement-candidates/improve-013-move-adahedge-weight-evidence-out-of-evaluation-layer.md)、[構造と命名](../../steering/structure.md)
- 利用skill: kiro-spec-init/requirements/design/tasks、kiro-impl（--review off）、kiro-validate-impl。共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
