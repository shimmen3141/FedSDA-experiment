"""警報・帰属の選択・統合の判定だけを製品の外から置き換える。"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from itertools import combinations
from unittest.mock import patch

ORACLE_VARIANT = "product-default-oracle-concept-assignment"
# 警報だけを真の変化の5標本目に置き換え、帰属の選択と統合は製品のまま。検出だけで取れる分を測る。
ORACLE_CHANGE_ALARM_VARIANT = "product-default-oracle-change-alarm"
ORACLE_VARIANTS = (ORACLE_VARIANT, ORACLE_CHANGE_ALARM_VARIANT)
VARIANTS = ("product-default", *ORACLE_VARIANTS)


def _unique_majority(counts):
    if not counts:
        return None
    concepts = [concept for concept, count in counts.items() if count == max(counts.values())]
    return concepts[0] if len(concepts) == 1 else None


@dataclass
class _ConceptChangeAlarm:
    previous_concept_id: int | None = None
    change_sample_index: int | None = None

    def observe(self, *, sample_index, concept_id):
        if concept_id is None:
            raise ValueError("oracleには真の概念が必要です")
        changed = self.previous_concept_id is not None and concept_id != self.previous_concept_id
        if changed:
            if self.change_sample_index is not None and sample_index - self.change_sample_index < 5:
                raise ValueError("前の概念の変化から5標本以内に次の変化が起きました")
            self.change_sample_index = sample_index
        self.previous_concept_id = concept_id
        return (
            changed,
            self.change_sample_index is not None and sample_index == self.change_sample_index + 4,
        )


def _oracle_pair_scores(*, model_ids, majority_concept_ids, threshold):
    return {
        (min(left, right), max(left, right)): (
            0.0
            if left in majority_concept_ids
            and right in majority_concept_ids
            and majority_concept_ids[left] == majority_concept_ids[right]
            else max(1.0, threshold + 1.0)
        )
        for left, right in combinations(model_ids, 2)
    }


def _select_destination_model_id(*, held_model_ids, assignment_counts_store, concept_id):
    candidates = []
    for model_id in held_model_ids:
        counts = assignment_counts_store.get_model_assigned_sample_concept_counts(model_id=model_id)
        if _unique_majority(counts) == concept_id:
            candidates.append((sum(counts.values()), model_id))
    # 同数なら、製品の保有順で先のモデル。
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


@contextmanager
def oracle_concept_assignment_policy(*, variant, diagnostics):
    if variant not in VARIANTS:
        raise ValueError(f"変種の名前が不正です: {variant}")
    if variant == "product-default":
        yield
        return

    from federated_learning_experiments.runtime import (
        alarm_interval_minimum_description_length_selection as selection,
        fedsda_run_client,
        model_clustering_and_consolidation as consolidation,
        observed_sample_processing as processing,
        server_round_synchronization,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )
    from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
        MinimumDescriptionLengthAssignmentSelection,
    )
    from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
        ModelTrainingAndAssignmentCountsStore,
    )

    diagnostics.update(
        changes=[],
        alarms=[],
        selections=[],
        adaptation_outcomes={},
        suppressed_monitor_signals=0,
        assigned_sample_count=0,
        processed_sample_count=0,
        mixed_assignment_count=0,
        consolidations=[],
    )
    change_alarms = {}
    assignment_stores = {}
    pending_buffers = {}
    process_original = fedsda_run_client.process_observed_sample
    observe_original = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
    select_original = processing.select_alarm_interval_assignment
    compare_original = selection.select_assignment_by_minimum_description_length
    consolidate_original = server_round_synchronization.cluster_and_consolidate_global_models
    cluster_original = consolidation.cluster_model_ids_by_average_linkage
    assign_original = ModelTrainingAndAssignmentCountsStore.record_assigned_sample_concept

    def record_assignment(self, **arguments):
        # 保留の解放を含め、実際に帰属した標本を全件調べる。run末尾の保留は製品と同じく残す。
        before = self.get_model_assigned_sample_concept_counts(model_id=arguments["model_id"])
        majority = _unique_majority(before)
        assign_original(self, **arguments)
        diagnostics["assigned_sample_count"] += 1
        diagnostics["mixed_assignment_count"] += int(
            bool(before) and majority != arguments["observed_concept_id"]
        )
        assignment_stores[self] = True

    def process(**arguments):
        observation = arguments["indexed_observation"]
        client_id = arguments["client_id"]
        diagnostics["processed_sample_count"] += 1
        pending_buffers[client_id] = arguments["pending_training_assignment_buffer"]
        alarm = change_alarms.setdefault(client_id, _ConceptChangeAlarm())
        changed, raised = alarm.observe(
            sample_index=observation.sample_index,
            concept_id=observation.observed_concept_id,
        )
        event = dict(
            client_id=client_id,
            sample_index=observation.sample_index,
            concept_id=observation.observed_concept_id,
        )
        if changed:
            diagnostics["changes"].append(event)
        if raised:
            diagnostics["alarms"].append(event)
        selected_interval = ()

        def observe(self, **monitor_arguments):
            result = observe_original(self, **monitor_arguments)
            if result.overall_adwin_window_sample_count is not None:
                raise ValueError("oracleは既定のe-SR専用です")
            diagnostics["suppressed_monitor_signals"] += int(result.drift_detected)
            # 監視の計算と基準の引下げはそのまま。製品の警報へ渡す合図だけを置き換える。
            return replace(
                result,
                drift_detected=raised,
                estimated_change_span_sample_count=5
                if raised
                else result.estimated_change_span_sample_count,
                detector_candidate_start_sample_index=alarm.change_sample_index
                if raised
                else result.detector_candidate_start_sample_index,
            )

        def select(**selection_arguments):
            nonlocal selected_interval
            selected_interval = selection_arguments["pending_sample_observations"][-5:]
            concept_id = observation.observed_concept_id
            store = arguments["model_training_and_assignment_counts_store"]
            current_id = arguments["current_training_model_assignment"].current_training_model_id
            # IDは毎回、統合・同期後のownerから取る。
            destination = _select_destination_model_id(
                held_model_ids=tuple(
                    state.model_id
                    for state in selection_arguments[
                        "held_model_training_state_registry"
                    ].snapshot_ordered_held_model_training_states()
                ),
                assignment_counts_store=store,
                concept_id=concept_id,
            )

            def compare(**comparison_arguments):
                compare_original(**comparison_arguments)
                hypothesis = (
                    "new_model"
                    if destination is None
                    else "current_model"
                    if destination == current_id
                    else "held_model"
                )
                lengths = comparison_arguments[
                    "new_model_code_lengths_by_change_interval_sample_count"
                ]
                names = comparison_arguments[
                    "new_model_learner_names_by_change_interval_sample_count"
                ]
                current_losses = comparison_arguments["current_model_log_losses"]
                if hypothesis == "current_model":
                    description_length = sum(current_losses)
                else:
                    interval_length = (
                        lengths[5]
                        if hypothesis == "new_model"
                        else sum(
                            dict(comparison_arguments["held_model_log_losses_by_model_id"])[
                                destination
                            ][-5:]
                        )
                    )
                    description_length = (
                        sum(current_losses[:-5]) + comparison_arguments["switching_code_length"]
                    ) + interval_length
                # 5標本の学習者の選択は製品の計算の結果を使う。oracleの符号長はMDL最短の意味を持たない。
                result = MinimumDescriptionLengthAssignmentSelection(
                    selected_hypothesis=hypothesis,
                    change_interval_sample_count=5,
                    destination_model_id=destination if hypothesis == "held_model" else None,
                    description_length=float(description_length),
                    new_model_learner_name=names[5] if hypothesis == "new_model" else "",
                )
                diagnostics["selections"].append(
                    dict(**event, hypothesis=hypothesis, destination_model_id=destination)
                )
                return result

            with patch.object(
                selection, "select_assignment_by_minimum_description_length", compare
            ):
                return select_original(**selection_arguments)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    OverallAndTrueClassLossMonitor,
                    "observe_loss_after_label_observation",
                    observe,
                )
            )
            if variant == ORACLE_VARIANT:
                stack.enter_context(
                    patch.object(processing, "select_alarm_interval_assignment", select)
                )
            result = process_original(**arguments)
        if result.alarm_occurrence_handling is not None:
            # 応答の内訳の正本はraw。ここでは製品の応答の結果も保存する。
            outcome = result.alarm_occurrence_handling.alarm_response_completion.alarm_buffer_response.response_outcome
            if (
                variant == ORACLE_VARIANT
                and outcome == "alarm_interval_candidate_adopted_as_new_model"
            ):
                # 製品の即時採用は、区間を学習標本へ足すが、概念別の割当計数へは足さない。
                # oracleの多数派を採用直後から使えるよう、診断の計数だけを補う（学習は繰り返さない）。
                for assigned in selected_interval:
                    arguments[
                        "model_training_and_assignment_counts_store"
                    ].record_assigned_sample_concept(
                        model_id=arguments[
                            "current_training_model_assignment"
                        ].current_training_model_id,
                        observed_concept_id=assigned.observed_concept_id,
                    )
            counter = diagnostics["adaptation_outcomes"]
            counter[outcome] = counter.get(outcome, 0) + 1
        return result

    def consolidate(**arguments):
        majority = consolidation._get_unique_majority_concept_ids(
            run_clients=arguments["run_clients"],
            model_ids=arguments["model_cross_evaluation"].cross_evaluated_model_ids,
        )

        def cluster(**cluster_arguments):
            scores = cluster_arguments["decision_scores_by_model_pair"]
            scores.clear()
            scores.update(
                _oracle_pair_scores(
                    model_ids=cluster_arguments["model_ids"],
                    majority_concept_ids=majority,
                    threshold=cluster_arguments["maximum_same_cluster_decision_score"],
                )
            )
            return cluster_original(**cluster_arguments)

        with patch.object(consolidation, "cluster_model_ids_by_average_linkage", cluster):
            result = consolidate_original(**arguments)
        if result.absorbed_model_ids:
            diagnostics["consolidations"].append(
                dict(
                    round_index=arguments["round_index"],
                    model_clusters=result.model_clusters,
                    majority_concept_ids=majority,
                    absorbed_model_ids=result.absorbed_model_ids,
                )
            )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(fedsda_run_client, "process_observed_sample", process))
        if variant == ORACLE_VARIANT:
            stack.enter_context(
                patch.object(
                    server_round_synchronization,
                    "cluster_and_consolidate_global_models",
                    consolidate,
                )
            )
        # 帰属の記録は、どちらの変種でも元の処理を呼んだ後に数えるだけ。
        stack.enter_context(
            patch.object(
                ModelTrainingAndAssignmentCountsStore,
                "record_assigned_sample_concept",
                record_assignment,
            )
        )
        yield
    diagnostics["final_assignment_concept_counts"] = [
        store.snapshot_model_training_and_assignment_counts().assigned_sample_counts_by_model_and_concept_id
        for store in assignment_stores
    ]
    diagnostics["pending_sample_count"] = sum(
        len(buffer.get_state_snapshot().pending_sample_indices)
        for buffer in pending_buffers.values()
    )
