"""S1の研究用採用。予測後にD>0なら、現行共有部へ接続して既存の登録経路へ渡す。"""

from copy import deepcopy


def adopt_shadow(*, state, arguments):
    from torch import cat
    from federated_learning_experiments.runtime.adopted_candidate_local_adoption import (
        adopt_candidate_as_current_training_model,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_state import (
        ParameterOptimizerState,
    )
    from federated_learning_experiments.evaluation.adaptation_record_store import (
        AdaptationRecord,
    )
    from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
        select_loss_monitoring_baseline_mean_loss,
    )
    from federated_learning_experiments.runtime.training_assignment_all_sample_adahedge_weight_evidence_restart import (
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change,
    )

    a, e = arguments, state["episode"]
    learner = state["learner"]
    current_id = a["current_training_model_assignment"].current_training_model_id
    current = (
        a["held_model_training_state_registry"]
        .get_held_model_training_state(model_id=current_id)
        .classifier
    )
    # 共有部は巻き戻さない。判定時の凍結共有部と採用時の共有部は異なり得る。
    learner.attach_shared_feature_extractor(
        shared_feature_extractor=current.feature_extractor
    )
    optimizer_state = deepcopy(state["optimizer"].state_dict())
    holder = ParameterOptimizerState(
        parameters=tuple(
            p
            for n, p in learner.named_parameters()
            if not n.startswith("feature_extractor.")
        ),
        optimizer_settings=a["parameter_optimizer_settings"],
    )
    pending = a[
        "pending_sample_observation_store"
    ].snapshot_pending_sample_observations()
    observations = state["observations"]
    assignment = adopt_candidate_as_current_training_model(
        **{
            k: a[k]
            for k in (
                "temporary_model_id_allocator",
                "upload_delay_round_count",
                "held_model_training_state_registry",
                "loss_statistics_store",
                "training_sample_store",
                "model_training_and_assignment_counts_store",
                "current_training_model_assignment",
                "pending_model_upload_state",
            )
        },
        adopted_candidate_classifier=learner,
        candidate_concept_specific_parameter_optimizer_state=holder,
        initial_statistics_input_features=cat(
            [o.training_sample.input_features for o in observations]
        ),
        initial_statistics_observed_class_labels=cat(
            [o.training_sample.observed_class_labels for o in observations]
        ),
        candidate_trained_sample_count=len(observations),
        candidate_parameter_update_step_count=len(observations),
        pending_assignment_training_samples=tuple(o.training_sample for o in pending),
    )
    # 共通登録はoptimizerをリセットするため、影が実際に学んだ状態を戻す。
    holder.parameter_optimizer.load_state_dict(optimizer_state)
    for o in pending:
        a["model_training_and_assignment_counts_store"].record_assigned_sample_concept(
            model_id=assignment.current_model_id,
            observed_concept_id=o.observed_concept_id,
        )
    a["pending_training_assignment_buffer"].drain_pending_sample_indices()
    a["pending_sample_observation_store"].retain_latest_pending_sample_observations(
        retained_sample_indices=()
    )
    statistics = a["loss_statistics_store"].get_model_loss_statistics(
        model_id=assignment.current_model_id
    )
    a["loss_change_monitor"].reset(
        baseline_loss_mean=select_loss_monitoring_baseline_mean_loss(
            loss_moments=statistics.overall_loss_moments
        )
    )
    restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
        assignment_change=assignment,
        all_sample_adahedge_weight_evidence_holder=a[
            "all_sample_adahedge_weight_evidence_holder"
        ],
    )
    a["adaptation_record_store"].append_adaptation_record(
        adaptation_record=AdaptationRecord(
            adaptation_sample_index=a["indexed_observation"].sample_index,
            detector_name="research_shadow_score",
            adaptation_outcome="post_alarm_validation_candidate_adopted",
            previous_training_model_id=assignment.previous_model_id,
            current_training_model_id=assignment.current_model_id,
            estimated_change_point_sample_index=e["alarm_sample_index"]
            - e["change_interval_sample_count"]
            + 1,
            detection_episode_id=None,
        )
    )
    e.update(
        adopted_model_id=assignment.current_model_id,
        adoption_sample_index=a["indexed_observation"].sample_index,
        adoption_pending_sample_count=len(pending),
        adoption_shared_policy="current_shared",
        adoption_optimizer_policy="continue_shadow",
    )
