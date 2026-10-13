"""一つのクライアントの、全標本のAdaHedgeの証拠を保持する。"""

from federated_learning_experiments.methods.fedsda.prediction_combination.adahedge_weight_evidence import (
    AdaHedgeWeightEvidence,
)


class AllSampleAdaHedgeWeightEvidenceHolder:
    """予測や学習帰属を持たず、全標本のAdaHedgeの証拠のlive ownerを返す。真の概念別の証拠は持たない。

    `evidence_is_updated_by_prediction`が偽のholderは、予測の処理が証拠を更新しない構成（現在のモデルだけで
    予測する構成）のもの。その証拠は、学習帰属の変更でも、始め直さない（使われない証拠の計数を、動かさない）。
    """

    def __init__(self, *, evidence_is_updated_by_prediction: bool = True) -> None:
        if type(evidence_is_updated_by_prediction) is not bool:
            raise TypeError("evidence_is_updated_by_predictionはbuiltin boolが必要です。")
        self._evidence_is_updated_by_prediction = evidence_is_updated_by_prediction
        self._all_sample_weight_evidence = AdaHedgeWeightEvidence()

    @property
    def evidence_is_updated_by_prediction(self) -> bool:
        return self._evidence_is_updated_by_prediction

    @property
    def all_sample_weight_evidence(self) -> AdaHedgeWeightEvidence:
        return self._all_sample_weight_evidence
