"""一つのクライアントのglobal・真の概念別診断証拠を独立して保持する。"""

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence import (
    AdaHedgeDiagnosticEvidence,
)


class AdaHedgeDiagnosticEvidenceCollection:
    """予測や学習帰属を持たず、診断証拠のlive ownerを返す。"""

    def __init__(self) -> None:
        self._global_diagnostic_evidence = AdaHedgeDiagnosticEvidence()
        self._true_concept_diagnostic_evidence_by_id: dict[int, AdaHedgeDiagnosticEvidence] = {}

    @property
    def global_diagnostic_evidence(self) -> AdaHedgeDiagnosticEvidence:
        return self._global_diagnostic_evidence

    def get_true_concept_diagnostic_evidence(
        self, *, true_concept_id: int
    ) -> AdaHedgeDiagnosticEvidence:
        """IDを先行検証し、初回だけ独立した証拠を作成する。"""
        if type(true_concept_id) is not int:
            raise TypeError("真の概念IDはbool・派生型以外のbuiltin intが必要です。")
        if true_concept_id not in self._true_concept_diagnostic_evidence_by_id:
            self._true_concept_diagnostic_evidence_by_id[true_concept_id] = (
                AdaHedgeDiagnosticEvidence()
            )
        return self._true_concept_diagnostic_evidence_by_id[true_concept_id]

    @property
    def created_true_concept_ids(self) -> tuple[int, ...]:
        """証拠を生成せず、作成順の不変snapshotを返す。"""
        return tuple(self._true_concept_diagnostic_evidence_by_id)
