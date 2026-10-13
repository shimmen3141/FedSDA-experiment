"""IMPROVE-006の案の試行（sourceは変えず、実行の間だけ差し込む）。

引数: <worktreeのルート> <variant> <dataset> <seed> [<標本数> [<集約間隔>]]（結果は、improve-006の記録の「試行」）
variant:
  current        いまの既定（短い警報のあと、前区間を外し、検出器を再開する）
  defer          推定変化区間が最小件数より短い間は、警報を出さずに待つ（検出器の証拠は保つ。警報の処理は何もしない）
  keep_evidence  短い警報の処理（前区間の取込みと記録）は行うが、検出器を再開しない（証拠を保つ）
"""

import collections
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
variant, dataset, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
sample_count = int(sys.argv[5]) if len(sys.argv) > 5 else 2500
interval = int(sys.argv[6]) if len(sys.argv) > 6 else 50

import federated_learning_experiments.runtime.alarm_occurrence_handling as occ  # noqa: E402
from federated_learning_experiments.configuration.experiment_run_conditions import ExperimentRunConditions  # noqa: E402
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (  # noqa: E402
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import execute_fedsda_experiment_run  # noqa: E402
from federated_learning_experiments.sweep.experiment_configurations import build_experiment_run_settings  # noqa: E402

settings = build_experiment_run_settings(
    configuration_name="fedsda",
    experiment_run_conditions=ExperimentRunConditions(
        dataset_name=dataset, random_seed=seed, client_count=10, per_client_sample_count=sample_count,
        server_aggregation_interval_per_client_samples=interval,
    ),
)
MINIMUM = settings.run_participant_settings.run_client_settings.scalar_settings.minimum_change_interval_sample_count
SHORT = "alarm_change_interval_too_short"
counters = collections.Counter()

if variant == "defer":
    real_observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation

    def observe(self, **kw):
        observation = real_observe(self, **kw)
        if observation.drift_detected and observation.estimated_change_span_sample_count < MINIMUM:
            counters["deferred_alarm"] += 1
            return replace(observation, drift_detected=False)
        return observation

    OverallAndTrueClassLossMonitor.observe_loss_after_label_observation = observe
elif variant == "keep_evidence":
    real_reset = OverallAndTrueClassLossMonitor.reset
    real_complete = occ.complete_alarm_buffer_response
    state = {"skip_reset": False}

    def reset(self, **kw):
        if state["skip_reset"]:
            counters["skipped_reset"] += 1
            return None
        return real_reset(self, **kw)

    def complete(**kw):
        state["skip_reset"] = kw["alarm_buffer_response"].response_outcome == SHORT
        try:
            return real_complete(**kw)
        finally:
            state["skip_reset"] = False

    OverallAndTrueClassLossMonitor.reset = reset
    occ.complete_alarm_buffer_response = complete
elif variant != "current":
    raise SystemExit(f"unknown variant: {variant}")

run = execute_fedsda_experiment_run(run_settings=settings)
m = run.run_metrics
chain_lengths, chain_spans, alarm_outcomes = [], [], collections.Counter()
repeated = 0
for rc in run.measured_run.participants.client_operations:
    owners = rc.owners
    ids = [id(s) for c in owners.training_sample_store.snapshot_ordered_model_training_samples() for s in c.training_samples]
    repeated += len(ids) - len(set(ids))
    chain = []
    for record in owners.adaptation_record_store.get_state_snapshot().adaptation_records:
        outcome = record.adaptation_outcome
        if not outcome.startswith("alarm_"):
            continue
        alarm_outcomes[outcome] += 1
        if outcome == SHORT:
            chain.append(record.adaptation_sample_index)
        elif chain:
            chain_lengths.append(len(chain)); chain_spans.append(record.adaptation_sample_index - chain[0]); chain = []
    if chain:
        chain_lengths.append(len(chain)); chain_spans.append(chain[-1] - chain[0])
d = m.training_model_switch_detection_metrics
cr = m.change_response_metrics
print("PROTO", json.dumps(dict(
    variant=variant, dataset=dataset, seed=seed, sample_count=sample_count, interval=interval,
    accuracy=m.prediction_accuracy, stable=m.stable_period_prediction_accuracy,
    down=m.communication_volume.downloaded_byte_count, global_models=m.final_global_model_count,
    detection_count=d.detection_count, precision=d.detection_precision, recall=d.detection_recall, f1=d.detection_f1,
    switch_delay=cr.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count,
    alarm_precision=cr.loss_change_alarm_detection_metrics.detection_precision,
    alarm_recall=cr.loss_change_alarm_detection_metrics.detection_recall,
    alarms=sum(alarm_outcomes.values()), short=alarm_outcomes[SHORT],
    chain_count=len(chain_lengths), max_chain=max(chain_lengths, default=0),
    samples_in_chains=sum(chain_spans), repeated=repeated,
    adopted=cr.adaptation_action_metrics.adopt_candidate_model_count,
    reused=cr.adaptation_action_metrics.reuse_held_model_count,
    **dict(counters),
)))
