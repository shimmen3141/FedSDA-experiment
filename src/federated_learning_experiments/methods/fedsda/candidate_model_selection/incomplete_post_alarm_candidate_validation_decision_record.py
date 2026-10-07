"""件数不足による候補棄却と、比較未成立の診断情報を保持する。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class IncompletePostAlarmCandidateValidationDecisionRecord:
    """不足用の不変出力。比較参照・検証平均・通常採否評価を保持しない。"""

    proposal_sample_index: int
    finalization_sample_index: int
    detector_name: str
    candidate_training_interval_sample_count: int
    validation_sample_count: int

    @property
    def finalization_delay_sample_count(self) -> int:
        return self.finalization_sample_index - self.proposal_sample_index
