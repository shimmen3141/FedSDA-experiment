"""一つのクライアントの全標本のAdaHedgeの証拠と、真の概念別の診断の証拠を独立して保持する。"""

from federated_learning_experiments.methods.fedsda.prediction_combination.adahedge_weight_evidence import (
    AdaHedgeWeightEvidence,
)


class AdaHedgeWeightEvidenceCollection:
    """予測や学習帰属を持たず、AdaHedgeの証拠のlive ownerを返す。

    `evidence_is_updated_by_prediction`が偽の集まりは、予測の処理が証拠を更新しない構成（現在のモデルだけで
    予測する構成）のもの。その証拠は、学習帰属の変更でも、始め直さない（使われない証拠の計数を、動かさない）。
    """

    def __init__(self, *, evidence_is_updated_by_prediction: bool = True) -> None:
        if type(evidence_is_updated_by_prediction) is not bool:
            raise TypeError("evidence_is_updated_by_predictionはbuiltin boolが必要です。")
        self._evidence_is_updated_by_prediction = evidence_is_updated_by_prediction
        self._all_sample_weight_evidence = AdaHedgeWeightEvidence()
        self._true_concept_diagnostic_evidence_by_id: dict[int, AdaHedgeWeightEvidence] = {}

    @property
    def evidence_is_updated_by_prediction(self) -> bool:
        return self._evidence_is_updated_by_prediction

    @property
    def all_sample_weight_evidence(self) -> AdaHedgeWeightEvidence:
        return self._all_sample_weight_evidence

    def get_true_concept_diagnostic_evidence(
        self, *, true_concept_id: int
    ) -> AdaHedgeWeightEvidence:
        """IDを先行検証し、初回だけ独立した証拠を作成する。"""
        if type(true_concept_id) is not int:
            raise TypeError("真の概念IDはbool・派生型以外のbuiltin intが必要です。")
        if true_concept_id not in self._true_concept_diagnostic_evidence_by_id:
            self._true_concept_diagnostic_evidence_by_id[true_concept_id] = AdaHedgeWeightEvidence()
        return self._true_concept_diagnostic_evidence_by_id[true_concept_id]

    @property
    def created_true_concept_ids(self) -> tuple[int, ...]:
        """証拠を生成せず、作成順の不変snapshotを返す。"""
        return tuple(self._true_concept_diagnostic_evidence_by_id)
