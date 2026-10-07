"""警報後の候補検証の判定根拠を、状態操作から独立した不変記録で保持する。"""

from dataclasses import dataclass

from .post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateValidationDecisionRecord:
    """検証のmetadataと計算済み評価。分類器・session・可変ownerを保持しない。"""

    proposal_sample_index: int
    resolution_sample_index: int
    detector_name: str
    candidate_training_interval_sample_count: int
    post_alarm_candidate_loss_evaluation: PostAlarmCandidateLossEvaluation

    @property
    def validation_completion_delay_sample_count(self) -> int:
        return self.resolution_sample_index - self.proposal_sample_index

    @property
    def full_validation_mean_loss_advantage(self) -> float:
        return (
            self.post_alarm_candidate_loss_evaluation.reference_full_interval_mean_loss
            - self.post_alarm_candidate_loss_evaluation.candidate_full_interval_mean_loss
        )

    @property
    def second_segment_mean_loss_advantage(self) -> float:
        return (
            self.post_alarm_candidate_loss_evaluation.reference_second_segment_mean_loss
            - self.post_alarm_candidate_loss_evaluation.candidate_second_segment_mean_loss
        )

    @property
    def reference_mean_loss_difference_from_history(self) -> float | None:
        if self.post_alarm_candidate_loss_evaluation.reference_historical_mean_loss is None:
            return None
        return (
            self.post_alarm_candidate_loss_evaluation.reference_full_interval_mean_loss
            - self.post_alarm_candidate_loss_evaluation.reference_historical_mean_loss
        )
