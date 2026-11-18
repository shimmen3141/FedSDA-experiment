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
    # 監視の部品は損失の低下で基準を下げるときにも自分でresetを呼ぶので、警報の処理の中のresetだけを記録する。
    resets = []
    in_alarm_handling = [False]
    original_handle = processor._handle_detector_signal

    def handling(*, sample_index):
        in_alarm_handling[0] = True
        try:
            return original_handle(sample_index=sample_index)
        finally:
            in_alarm_handling[0] = False

    processor._handle_detector_signal = handling
    # 割り当てた監視用の損失を、モデルごとに独立に記録する（基準値の期待値を、統計の部品を使わずに作るため）。
    assigned_losses = {}
    original_record = processor.loss_statistics_store.record_assigned_loss

    def recording_loss(*, model_id, observed_loss, observed_class_id):
        assigned_losses.setdefault(model_id, []).append(observed_loss)
        return original_record(model_id=model_id, observed_loss=observed_loss, observed_class_id=observed_class_id)

    processor.loss_statistics_store.record_assigned_loss = recording_loss
    monitor_wrapped = False
    records = []
    pending_after_alarm = {}
    for i, (x, y, c) in enumerate(STREAM):
        if processor.loss_change_monitor is not None and not monitor_wrapped:
            monitor = processor.loss_change_monitor
            original_reset = monitor.reset

            def tracking_reset(*, baseline_loss_mean, _original=original_reset):
                if in_alarm_handling[0]:
                    losses = assigned_losses.get(processor.current_model_id, [])
                    expected = 0.01 if not losses else min(1.0 - 1e-6, max(0.01, sum(losses) / len(losses)))
                    resets.append((processor.processed_sample_count, baseline_loss_mean, expected))
                return _original(baseline_loss_mean=baseline_loss_mean)

            monitor.reset = tracking_reset
            monitor_wrapped = True
        alarm_count_before = len(processor.alarm_records)
        records.append(
            processor.process_sample(sample_index=i, features=x, observed_class_id=y, observed_concept_id=c)
        )
        new_alarms = processor.alarm_records[alarm_count_before:]
        if any(a.alarm_raised for a in new_alarms):
            pending_after_alarm[i] = processor.pending_sample_indices
    patcher.undo()
    return processor, records, learned, weights_seen, (resets, pending_after_alarm)


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
    processor, _, _, _, (resets, pending_after_alarm) = traced_run
    alarms = raised(processor)
    assert alarms
    # 警報の処理の中の始め直しは、警報1回につき1回で、警報の標本の処理の中。
    assert [count for count, _, _ in resets] == [a.sample_index for a in alarms]
    # 基準値は、帰属先に割り当てた監視用の損失の平均を[0.01, 1−1e-6]に制限した値（統計の部品を使わずに作った期待値）。
    for _, baseline, expected in resets:
        assert baseline == pytest.approx(expected)
    # 警報の標本の処理の直後に、保留は空。
    for alarm in alarms:
        assert pending_after_alarm[alarm.sample_index] == ()


def test_short_window_after_an_alarm_waits_and_retries_the_selection(monkeypatch):
    """検出器の合図を差し替えて、警報の直後（保留が少ない）に合図が続く場面を必ず作る。"""
    from dataclasses import replace

    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )

    forced = {300, 301, 302, 303, 304}
    original_observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation

    def forcing(self, **kwargs):
        observation = original_observe(self, **kwargs)
        return replace(observation, drift_detected=kwargs["sample_index"] in forced)

    monkeypatch.setattr(OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", forcing)
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=build_default_single_stream_settings(), class_count=2
    )
    resets = []
    stream = make_stream([(0, 310)])
    pending_after = {}
    for i, (x, y, c) in enumerate(stream):
        if processor.loss_change_monitor is not None and not resets and not hasattr(processor, "_wrapped"):
            monitor = processor.loss_change_monitor
            original_reset = monitor.reset

            def tracking_reset(*, baseline_loss_mean, _original=original_reset):
                resets.append(processor.processed_sample_count)
                return _original(baseline_loss_mean=baseline_loss_mean)

            monitor.reset = tracking_reset
            processor._wrapped = True
        processor.process_sample(sample_index=i, features=x, observed_class_id=y, observed_concept_id=c)
        pending_after[i] = processor.pending_sample_indices
    by_index = {a.sample_index: a for a in processor.alarm_records}
    # 300: 保留が31件で選択でき、警報になり、保留が空になる。
    assert by_index[300].alarm_raised and by_index[300].window_sample_count == 31
    assert pending_after[300] == ()
    # 301〜304: 保留が1〜4件（<5）なので警報とせず、毎標本、その標本を最後に含む列で選択をやり直す。
    for offset, index in enumerate((301, 302, 303, 304), start=1):
        alarm = by_index[index]
        assert alarm.adaptation_outcome == "too_short" and not alarm.alarm_raised
        assert alarm.window_sample_count == offset
        assert pending_after[index] == tuple(range(301, index + 1))
    # 待った標本では検出器を始め直さない（始め直しは300の警報の1回だけ。内部の基準の引き下げは除く）。
    assert 300 in resets and not any(index in resets for index in (301, 302, 303, 304))
    # 305以降は合図が無く、保留は容量まで溜まり続ける。
    assert pending_after[309] == tuple(range(301, 310))


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
