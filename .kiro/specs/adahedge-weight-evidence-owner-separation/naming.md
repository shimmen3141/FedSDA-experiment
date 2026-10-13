# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `AdaHedgeWeightEvidence` | `methods/fedsda/prediction_combination/adahedge_weight_evidence.py`（前は`evaluation/`） | 1つのAdaHedgeの証拠（累積損失、gap、再始動・再較正の計数）と、そこから重みを作る計算。名前と中身は変えない | `FixedShareWeightController`は、Fixed-Shareの重みそのものを持つ。こちらは、重みを作るもとの証拠を持つ |
| `AllSampleAdaHedgeWeightEvidenceHolder` | `methods/fedsda/prediction_combination/all_sample_adahedge_weight_evidence_holder.py` | clientの、全標本の証拠（1つ）と、「予測の処理が証拠を更新する構成かどうか」を持つ。真の概念別の証拠は持たない | 前の`AdaHedgeWeightEvidenceCollection`は、真の概念別の証拠も持つ集まりだった。こちらは、証拠を1つだけ持つので、collectionと呼ばない |
| `AllSampleAdaHedgeWeightEvidenceHolder.all_sample_weight_evidence` | 同上 | 全標本の証拠（前の集まりの、同じ名前のproperty） | — |
| `AllSampleAdaHedgeWeightEvidenceHolder.evidence_is_updated_by_prediction` | 同上 | 予測の処理が、証拠を更新する構成かどうか（前の集まりの、同じ名前のproperty。偽なら、学習帰属の変更でも、証拠を始め直さない） | — |
| `TrueConceptAdaHedgeDiagnosticEvidenceCollection` | `evaluation/true_concept_adahedge_diagnostic_evidence_collection.py` | 標本の真の概念IDごとの、診断用のAdaHedgeの証拠の集まり（比較用。手法の判断には使わない） | 全標本の側のholderは、予測に使いうる証拠。こちらは、真の概念を使うので、常に比較用 |
| `TrueConceptAdaHedgeDiagnosticEvidenceCollection.get_true_concept_diagnostic_evidence` | 同上 | 真の概念IDの証拠を返す（初回だけ作る。前の集まりの、同じ名前の口） | — |
| `TrueConceptAdaHedgeDiagnosticEvidenceCollection.created_true_concept_ids` | 同上 | 証拠を作った真の概念IDの、作成順の並び（前の集まりの、同じ名前のproperty） | — |
| `restart_all_sample_adahedge_weight_evidence_after_training_assignment_change` | `runtime/training_assignment_all_sample_adahedge_weight_evidence_restart.py` | 学習帰属が、別のモデルへ変わったときに、全標本の証拠を始め直す。予測の処理が証拠を更新しない構成（ownerの`evidence_is_updated_by_prediction`が偽）では、何もしない | 真の概念別の証拠は、学習帰属の変更では始め直さない（集約後の再較正`recalibrate_prediction_state_after_aggregation`が、再始動する） |

## 無くなる公開名

| 公開名 | 代わり |
|---|---|
| `AdaHedgeWeightEvidenceCollection`（`evaluation/adahedge_weight_evidence_collection.py`） | 上の2つのowner |
| `evaluation/adahedge_weight_evidence.py` | `methods/fedsda/prediction_combination/adahedge_weight_evidence.py` |

## 名前を変えるもの（役割は同じ）

| いま | これから | 理由 |
|---|---|---|
| `restart_adahedge_weight_evidence_after_training_assignment_change`（`runtime/training_assignment_adahedge_weight_evidence_restart.py`） | `restart_all_sample_adahedge_weight_evidence_after_training_assignment_change`（`runtime/training_assignment_all_sample_adahedge_weight_evidence_restart.py`） | AdaHedgeの証拠のownerが2つになる。この関数が始め直すのは、全標本の証拠だけ（真の概念別の証拠は、学習帰属の変更では始め直さない）。名前だけで、対象が分かるようにする |

## 名前を変えないもの

`recalibrate_prediction_state_after_aggregation`、`predict_observed_sample_and_update_prediction_weights`、記録の表の列名（`client_weight_update_counts.adahedge_weight_…`）、指標の名前。
