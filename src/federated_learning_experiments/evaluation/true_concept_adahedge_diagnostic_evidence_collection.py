"""一つのクライアントの、真の概念別の、診断用のAdaHedgeの証拠を保持する（比較用。手法の判断には使わない）。"""

from federated_learning_experiments.methods.fedsda.prediction_combination.adahedge_weight_evidence import (
    AdaHedgeWeightEvidence,
)


class TrueConceptAdaHedgeDiagnosticEvidenceCollection:
    """真の概念IDごとに独立した、AdaHedgeの証拠のlive ownerを返す。全標本の証拠は持たない。"""

    def __init__(self) -> None:
        self._true_concept_diagnostic_evidence_by_id: dict[int, AdaHedgeWeightEvidence] = {}

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
