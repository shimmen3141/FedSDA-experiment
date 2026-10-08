"""完成した警報応答を適応記録に写し、記録ownerへ一件追加する。"""

from typing import cast

from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationOutcome,
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.runtime.alarm_response_completion import AlarmResponseCompletion


def record_completed_alarm_response(
    *,
    alarm_response_completion: AlarmResponseCompletion,
    detector_name: str,
    adaptation_record_store: AdaptationRecordStore,
) -> AdaptationRecord:
    if type(alarm_response_completion) is not AlarmResponseCompletion:
        raise TypeError("alarm_response_completion must be exact AlarmResponseCompletion")
    if type(adaptation_record_store) is not AdaptationRecordStore:
        raise TypeError("adaptation_record_store must be exact AdaptationRecordStore")
    # frozen fieldの手動破壊も、上流の読取り専用検査と記録の検査で更新前に拒否する。
    alarm_response_completion.__post_init__()
    adaptation_record = AdaptationRecord(
        adaptation_sample_index=alarm_response_completion.alarm_sample_index,
        detector_name=detector_name,
        adaptation_outcome=cast(
            AdaptationOutcome, alarm_response_completion.alarm_buffer_response.response_outcome
        ),
        previous_training_model_id=alarm_response_completion.previous_training_model_id,
        current_training_model_id=alarm_response_completion.current_training_model_id,
        estimated_change_point_sample_index=alarm_response_completion.estimated_change_point_sample_index,
        detection_episode_id=alarm_response_completion.detection_episode_id,
    )
    adaptation_record_store.append_adaptation_record(adaptation_record=adaptation_record)
    return adaptation_record
