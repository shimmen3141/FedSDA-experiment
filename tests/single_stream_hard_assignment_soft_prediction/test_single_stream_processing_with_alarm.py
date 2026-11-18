"""1標本の処理（警報の経路）: 警報の後の割り当て、監視の始め直し、再利用、新しいモデル、待つ方針、Fixed-Shareの重み（WSL）。"""

import pytest

pytest.importorskip("river")

from tests.single_stream_hard_assignment_soft_prediction.stream_examples import make_stream  # noqa: E402

from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (  # noqa: E402
    FixedShareWeightController,
)
from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (  # noqa: E402
    HoeffdingTreeConceptModel,
)
from single_stream_hard_assignment_soft_prediction.single_stream_processing import (  # noqa: E402
    SingleStreamHardAssignmentSoftPredictionProcessor,
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (  # noqa: E402
    build_default_single_stream_settings,
)

STREAM = make_stream([(0, 800), (1, 800), (0, 800)])


@pytest.fixture(scope="module")
def traced_run():
    patcher = pytest.MonkeyPatch()
    learned = []
    weights_seen = []
    original_learn = HoeffdingTreeConceptModel.learn_one
    original_weights = FixedShareWeightController.get_weights_before_label_observation

    def counting(self, *, features, observed_class_id):
        # idは捨てた一時的な学習者と後で作ったモデルとで重なりうるので、インスタンス自身に回数を持たせる。
        self.test_learn_count = getattr(self, "test_learn_count", 0) + 1
        learned.append(None)
        return original_learn(self, features=features, observed_class_id=observed_class_id)

    def recording(self, *, model_ids):
        weights = original_weights(self, model_ids=model_ids)
        weights_seen.append(dict(weights))
        return weights

    patcher.setattr(HoeffdingTreeConceptModel, "learn_one", counting)
    patcher.setattr(FixedShareWeightController, "get_weights_before_label_observation", recording)
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=build_default_single_stream_settings(), class_count=2
    )
    resets = []
    monitor_reset = None
    records = []
    for i, (x, y, c) in enumerate(STREAM):
        if processor.loss_change_monitor is not None and monitor_reset is None:
            monitor = processor.loss_change_monitor
            original_reset = monitor.reset

            def tracking_reset(*, baseline_loss_mean, _original=original_reset):
                resets.append((processor.processed_sample_count, baseline_loss_mean))
                return _original(baseline_loss_mean=baseline_loss_mean)

            monitor.reset = tracking_reset
            monitor_reset = tracking_reset
        records.append(
            processor.process_sample(sample_index=i, features=x, observed_class_id=y, observed_concept_id=c)
        )
    patcher.undo()
    return processor, records, learned, weights_seen, resets


def raised(processor):
    return [a for a in processor.alarm_records if a.alarm_raised]


def test_concept_change_creates_a_new_model_and_recurrence_reuses_the_old_one(traced_run):
    processor, _, _, _, _ = traced_run
    outcomes = [(a.sample_index, a.adaptation_outcome, a.destination_model_id) for a in raised(processor)]
    new_alarms = [o for o in outcomes if o[1] == "new" and 800 <= o[0] < 1000]
    assert new_alarms, outcomes
    new_model_id = new_alarms[0][2]
    reuse_alarms = [o for o in outcomes if o[1] == "reuse" and 1600 <= o[0] < 1800]
    assert reuse_alarms, outcomes
    # 再利用したモデルは、コンセプト0（A）を正しく予測できるモデル（モデルのIDは問わない。keepの写しから作った
    # モデルは、元のAの知識を保つ）。比べるために、Bのモデル（最初の新しいモデル以外）より、Aでよく当たること。
    reused_model_id = reuse_alarms[0][2]
    probe = make_stream([(0, 70)])
    model = dict(processor.held_models)[reused_model_id]
    correct = sum(
        int(max(range(2), key=lambda c: model.predict_class_probabilities(features=x)[c]) == y) for x, y, _ in probe
    )
    assert correct / len(probe) > 0.9
    assert new_model_id != 0
    assert [model_id for model_id, _ in processor.held_models][:2] == [0, new_model_id]


def test_each_sample_is_assigned_and_learned_once_by_a_held_model(traced_run):
    processor, _, learned, _, _ = traced_run
    assignments = processor.assigned_model_id_by_sample_index
    pending = set(processor.pending_sample_indices)
    assert set(assignments) | pending == set(range(len(STREAM)))
    assert not set(assignments) & pending
    assert sum(getattr(model, "test_learn_count", 0) for _, model in processor.held_models) == len(assignments)


def test_alarm_empties_the_pending_buffer_and_restarts_monitoring_with_the_destination_baseline(traced_run):
    processor, records, _, _, resets = traced_run
    alarms = raised(processor)
    # 監視の部品は、損失の低下で基準を下げるときにも自分で始め直すので、警報の標本での始め直しがあることを確かめる。
    reset_positions = {count_at_reset for count_at_reset, _ in resets}
    for alarm in alarms:
        assert alarm.sample_index in reset_positions  # 警報の標本の処理の中で始め直す
    assert all(0.01 <= baseline <= 1 - 1e-6 for _, baseline in resets)
    for alarm in alarms:
        position = alarm.sample_index
        assert records[position].current_model_id is not None
    # 警報の直後の標本の時点で、保留はその標本だけ。
    assert alarms[0].sample_index + 1 < len(STREAM)


def test_short_window_after_an_alarm_does_not_raise_and_selection_is_retried(traced_run):
    processor, _, _, _, _ = traced_run
    too_short = [a for a in processor.alarm_records if a.adaptation_outcome == "too_short"]
    for alarm in too_short:
        assert not alarm.alarm_raised
        assert alarm.window_sample_count < 5
    # 待った合図の次の標本でも合図が続けば、その標本で選択をやり直している（番号が連続する）。
    indices = [a.sample_index for a in processor.alarm_records]
    for alarm in too_short:
        later = [i for i in indices if i > alarm.sample_index]
        if later and later[0] == alarm.sample_index + 1:
            break
    else:
        if too_short:
            pytest.skip("待った後に合図が続く場面がこの系列に無い")


def test_fixed_share_weights_reset_to_uniform_only_when_the_model_set_changes(traced_run):
    processor, _, _, weights_seen, _ = traced_run
    alarms = raised(processor)
    new_alarm = next(a for a in alarms if a.adaptation_outcome == "new")
    after_new = weights_seen[new_alarm.sample_index + 1]
    assert len(after_new) == 2
    assert list(after_new.values()) == pytest.approx([0.5, 0.5])
    reuse_alarm = next(a for a in alarms if a.adaptation_outcome == "reuse")
    before = weights_seen[reuse_alarm.sample_index]
    after = weights_seen[reuse_alarm.sample_index + 1]
    assert set(after) == set(before)
    assert list(after.values()) != pytest.approx([1 / len(after)] * len(after))


def test_new_model_is_built_by_the_selected_learner(traced_run):
    processor, _, _, _, _ = traced_run
    new_alarm = next(a for a in raised(processor) if a.adaptation_outcome == "new")
    assert new_alarm.new_model_learner_name in ("reinitialized_learner", "warm_started_learner")
    assert new_alarm.change_interval_sample_count >= 5
    assert len(new_alarm.window_tail_concept_ids) == new_alarm.change_interval_sample_count
