"""1標本の処理（警報の無い経路）: 予測、重みの更新、ウォームアップと監視、保留、割り当てと学習を確かめる（WSL）。"""

import math

import pytest

pytest.importorskip("river")

from tests.single_stream_hard_assignment_soft_prediction.stream_examples import make_stream  # noqa: E402

from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (  # noqa: E402
    HoeffdingTreeConceptModel,
)
from single_stream_hard_assignment_soft_prediction.single_stream_processing import (  # noqa: E402
    SingleStreamHardAssignmentSoftPredictionProcessor,
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (  # noqa: E402
    build_default_single_stream_settings,
)


def run(stream, concept_offset=0):
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=build_default_single_stream_settings(), class_count=2
    )
    records = [
        processor.process_sample(
            sample_index=i, features=x, observed_class_id=y, observed_concept_id=c + concept_offset
        )
        for i, (x, y, c) in enumerate(stream)
    ]
    return processor, records


def test_warm_up_has_no_monitoring_and_monitoring_starts_after_it():
    processor, records = run(make_stream([(0, 600)]))
    assert all(math.isnan(r.monitored_log_e_value) for r in records[:200])
    assert all(math.isfinite(r.monitored_log_e_value) for r in records[200:])
    assert not processor.alarm_records


def test_only_the_oldest_sample_beyond_the_capacity_is_assigned_and_learned_once(monkeypatch):
    learned = []
    original = HoeffdingTreeConceptModel.learn_one

    def counting(self, *, features, observed_class_id):
        self.test_learn_count = getattr(self, "test_learn_count", 0) + 1
        learned.append(tuple(features.items()))
        return original(self, features=features, observed_class_id=observed_class_id)

    monkeypatch.setattr(HoeffdingTreeConceptModel, "learn_one", counting)
    stream = make_stream([(0, 400)])
    processor, _ = run(stream)
    assert not processor.alarm_records
    assignments = processor.assigned_model_id_by_sample_index
    # 最後の30件は保留に残り、それより前の全件が、古い順に1回ずつ現在のモデルへ割り当てられる。
    assert sorted(assignments) == list(range(370))
    assert set(assignments.values()) == {0}
    assert sum(getattr(model, "test_learn_count", 0) for _, model in processor.held_models) == 370
    assert processor.pending_sample_indices == tuple(range(370, 400))


def test_prediction_is_the_argmax_of_the_weighted_mixture_with_one_model():
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=build_default_single_stream_settings(), class_count=2
    )
    for i, (x, y, c) in enumerate(make_stream([(0, 300)])):
        model = processor.held_models[0][1]
        probabilities = model.predict_class_probabilities(features=x)
        record = processor.process_sample(sample_index=i, features=x, observed_class_id=y, observed_concept_id=c)
        expected = 0 if probabilities[0] >= probabilities[1] else 1
        assert record.predicted_class_id == expected
        assert record.prediction_is_correct == (expected == y)
        assert record.held_model_count == 1 and record.current_model_id == 0


def test_true_concept_ids_do_not_change_predictions_or_learning():
    stream = make_stream([(0, 500), (1, 500)])
    first, first_records = run(stream)
    second, second_records = run(stream, concept_offset=7)
    assert [r.predicted_class_id for r in first_records] == [r.predicted_class_id for r in second_records]
    assert first.assigned_model_id_by_sample_index == second.assigned_model_id_by_sample_index
    assert [(a.sample_index, a.adaptation_outcome) for a in first.alarm_records] == [
        (a.sample_index, a.adaptation_outcome) for a in second.alarm_records
    ]


@pytest.mark.parametrize("bad", [dict(sample_index=-1), dict(observed_class_id=2), dict(observed_class_id=True)])
def test_invalid_inputs_are_rejected_before_any_update(bad):
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=build_default_single_stream_settings(), class_count=2
    )
    arguments = dict(sample_index=0, features={0: 1.0}, observed_class_id=1, observed_concept_id=0) | bad
    with pytest.raises((ValueError, TypeError)):
        processor.process_sample(**arguments)
    assert processor.processed_sample_count == 0
    assert processor.pending_sample_indices == ()
