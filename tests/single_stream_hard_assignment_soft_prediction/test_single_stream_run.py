"""系列1本のrun: 記録と指標が処理の記録と一致すること、事前学習の標本を使わないことを確かめる（WSL）。"""

import numpy as np
import pytest

pytest.importorskip("river")

from tests.single_stream_hard_assignment_soft_prediction.stream_examples import make_stream  # noqa: E402

from single_stream_hard_assignment_soft_prediction.single_stream_input import SingleStreamInput  # noqa: E402
from single_stream_hard_assignment_soft_prediction.single_stream_metrics import (  # noqa: E402
    compute_prequential_metrics,
)
from single_stream_hard_assignment_soft_prediction.single_stream_run import run_single_stream  # noqa: E402
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (  # noqa: E402
    build_default_single_stream_settings,
)


def stream_input():
    stream = make_stream([(0, 700), (1, 700), (0, 700)])
    return SingleStreamInput(
        dataset_name="sine2",
        random_seed=0,
        class_count=2,
        features=np.array([[x[0], x[1]] for x, _, _ in stream], dtype=np.float32),
        labels=np.array([y for _, y, _ in stream], dtype=np.int64),
        concept_ids=np.array([c for _, _, c in stream], dtype=np.int64),
        array_sha256="0" * 64,
    )


@pytest.fixture(scope="module")
def result():
    return run_single_stream(stream=stream_input(), settings=build_default_single_stream_settings())


def test_records_cover_every_sample_and_metrics_match_the_records(result):
    tables, summary = result.record_tables, result.summary
    assert tables["sample.predicted_class_id"].shape == (2100,)
    expected = compute_prequential_metrics(
        predicted_class_ids=tables["sample.predicted_class_id"],
        observed_class_ids=tables["sample.observed_class_id"],
        concept_ids=tables["sample.observed_concept_id"],
        class_count=2,
    )
    for name, value in expected.items():
        assert summary[name] == pytest.approx(value, nan_ok=True)
    assert summary["prequential_accuracy"] == pytest.approx(tables["sample.prediction_is_correct"].mean())
    assert summary["sample_count"] == 2100 and summary["elapsed_seconds"] > 0


def test_alarm_and_model_summaries_match_the_records(result):
    tables, summary = result.record_tables, result.summary
    raised = tables["alarm.alarm_raised"]
    assert summary["alarm_count"] == int(raised.sum()) >= 2
    assert summary["detector_signal_count"] == len(raised)
    assert summary["created_model_count"] == len(tables["model.model_id"]) - 1 >= 1
    assert summary["final_held_model_count"] == len(tables["model.model_id"])
    assert summary["final_tree_node_count"] >= 1
    identification = summary["concept_identification"]
    assert sum(identification.values()) == summary["alarm_count"]
    # 保留に残った最後の30件以外は、どれかのモデルへ割り当てられている。
    assert (tables["sample.assigned_model_id"] >= 0).sum() >= 2100 - 30


def test_warm_up_samples_have_nan_monitoring_values(result):
    values = result.record_tables["sample.monitored_log_e_value"]
    assert np.isnan(values[:200]).all() and np.isfinite(values[200:]).all()
