"""最終Residual Adapter＋Switching構成の数値・離散イベント回帰。

旧11ケースのgoldenとは別に保存する。更新前には必ず差分の原因を確認する。
"""

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_drift_experiment import run_random_drift_experiment
from federated_drift_experiment.data.mnist import default_data_dir
from federated_drift_experiment.experiment_spec.configuration import (
    AlgorithmOptions, ExperimentConfiguration, ParameterAssignment, temporary_config,
)

GOLDEN_PATH = Path(__file__).with_name("proposed_regression_golden.json")
MODE = "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting"
ALGORITHM = AlgorithmOptions(
    clustering_policy="on_new_model", clustering_decision="class_functional_confidence",
    detection_episodes=False, new_model_creation_policy="forward_persistent",
    fifo_size=30, new_model_validation_fraction=0.2,
    new_model_forward_validation_samples=10, shared_backbone_training="joint",
    shared_backbone_routing_recalibration="fifo_replay", shared_adapter_rank=8,
    shared_backbone_gradient_strategy="mean", soft_routing_context="switching",
    soft_routing_activation_policy="always", cluster_linkage="average",
)
# 学習条件も固定し、他テストが変更したグローバル設定を引き継がない。
COMMON = dict(
    N_CLIENTS=3, PRETRAIN_SAMPLES=100, PRETRAIN_EPOCHS=10, PRETRAIN_BATCH_SIZE=32,
    MIN_STABLE_PERIOD=100, DRIFT_PROB=0.015, STABLE_WINDOW=50,
    OPTIMIZER="adam", BASE_LR=0.01, NEW_MODEL_LR=0.01, WEIGHT_DECAY=0.001,
    AMSGRAD=True, NEW_MODEL_EPOCHS=30, NEW_MODEL_TRAINING="early_stopping",
    NEW_MODEL_INITIALIZATION="best_candidate", NEW_MODEL_EARLY_STOPPING_PATIENCE=3,
    NEW_MODEL_EARLY_STOPPING_MIN_DELTA=1e-4, UPDATES_PER_SAMPLE=1,
    LOCAL_UPDATE_INTERVAL=1, E_DETECTOR_ALPHA=0.001,
    STORED_DATA_LIMIT=50, EVAL_STORE_SAMPLE_SIZE=20, EVAL_MAX_SAMPLES=50,
    DELAY_TOLERANCE=100, FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS=1, MIN_DRIFT_DATA=5,
    CROSS_EVAL_MAX_CLIENTS=3,
    CLIENT_BATCH_SIZE=32, FEDSDA_CLUSTERING_CONFIDENCE=0.95,
    SEA_LABEL_NOISE=0.10, SEA_THRESHOLDS={0: 9.0, 1: 8.0, 2: 7.0, 3: 9.5},
)
CASES = {"sine2": {"TOTAL_DATA_POINTS": 1500}, "sea2": {"TOTAL_DATA_POINTS": 600},
         "mnist2": {"TOTAL_DATA_POINTS": 100, "N_CLIENTS": 2,
                    "PRETRAIN_EPOCHS": 2, "MIN_STABLE_PERIOD": 30}}
METRICS = (
    "accuracy", "stable_accuracy", "final_model_count", "precision", "recall", "f1",
    "total_detect", "comm_models_up", "comm_models_down", "comm_models_total",
    "comm_messages_up", "comm_messages_down", "comm_messages_total",
    "comm_parameter_values_up", "comm_parameter_values_down", "comm_parameter_values_total",
    "comm_bytes_up", "comm_bytes_down", "comm_bytes_total",
    "final_parameter_values", "final_parameter_bytes",
    "compute_inference_examples_total", "compute_training_examples_total",
    "compute_optimizer_steps_total", "compute_backbone_examples_total",
    "compute_head_examples_total", "compute_drift_detector_updates_total",
    "compute_drift_detector_hypotheses_total", "provisional_proposal_count",
    "provisional_forward_count", "routing_soft_prediction_sample_count",
    "routing_switching_recalibration_sample_count",
    "routing_aggregation_recalibration_sample_count",
)
TRACES = (
    "history_accuracy", "history_model_id", "drift_client_ids", "drift_positions",
    "switch_client_ids", "switch_positions", "adaptation_client_ids",
    "adaptation_positions", "adaptation_actions", "adaptation_old_model_ids",
    "adaptation_new_model_ids", "provisional_client_ids", "provisional_positions",
    "provisional_accepted", "provisional_reasons", "provisional_resolution_positions",
    "model_registration_ids", "model_registration_rounds", "model_registration_final_active",
    "clustering_rounds", "clustering_model_ids", "clustering_representative_model_ids",
    "clustering_participated_in_merge", "clustering_absorbed",
    "history_routing_switching_correct", "history_routing_switching_leader_id",
    "history_routing_soft_active", "clustering_pair_rounds",
    "clustering_pair_left_model_ids", "clustering_pair_right_model_ids",
    "clustering_pair_same_cluster",
)


def environment():
    return dict(system=platform.system(), machine=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                device="cpu", dtype=str(torch.get_default_dtype()), threads=1)


def definition():
    return dict(mode=MODE, algorithm=ALGORITHM.config_overrides(), common=COMMON,
                scale=CASES, seed=0, aggregation_interval=50, distance_threshold=0.1,
                concept_schedule="random")


def run_case(dataset, directory):
    configuration = ExperimentConfiguration(
        mode=MODE, dataset=dataset, seed=0, concept_schedule="random",
        series="proposed_regression", sweep_parameter=None, sweep_value=None,
        parameters=(ParameterAssignment("aggregation_interval", 50),
                    ParameterAssignment("fedsda_distance_threshold", 0.1)), algorithm=ALGORITHM,
    )
    raw = directory / f"{dataset}.npz"
    with temporary_config({**COMMON, **CASES[dataset]}), configuration.activated(), redirect_stdout(io.StringIO()):
        result = run_random_drift_experiment(
            mode=MODE, random_seed=0, verbose=False, show_plot=False, raw_path=str(raw),
        )
    metrics = {key: float(result[key]) for key in METRICS}
    assert all(math.isfinite(value) for value in metrics.values())
    traces = {}
    with np.load(raw, allow_pickle=False) as arrays:
        for key in TRACES:
            values = arrays[key]
            serialized = json.dumps(values.tolist(), separators=(",", ":"), ensure_ascii=True)
            traces[key] = dict(shape=list(values.shape), sha256=hashlib.sha256(serialized.encode()).hexdigest())
        coverage = dict(
            registered_models=int(arrays["model_registration_ids"].size),
            accepted_candidates=int(arrays["provisional_accepted"].sum()),
            rejected_candidates=int((~arrays["provisional_accepted"]).sum()),
            absorbed_models=int(arrays["clustering_absorbed"].sum()),
            switches=int(arrays["switch_positions"].size),
        )
    if dataset == "sine2":
        assert coverage["registered_models"] > 1, "複数expertの経路を検証する"
        assert coverage["accepted_candidates"] > 0, "候補採用の経路を検証する"
        assert coverage["absorbed_models"] > 0, "モデル統合の経路を検証する"
        assert metrics["routing_switching_recalibration_sample_count"] > 0
    return dict(metrics=metrics, traces=traces, coverage=coverage)


def compute_all():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="fedsda-regression-") as directory:
            return {dataset: run_case(dataset, Path(directory)) for dataset in CASES}
    finally:
        torch.set_num_threads(previous_threads)


def compare(actual, golden):
    assert actual.keys() == golden.keys()
    for dataset, case in actual.items():
        assert case["coverage"] == golden[dataset]["coverage"], f"{dataset}: イベント件数が変化"
        assert case["traces"].keys() == golden[dataset]["traces"].keys()
        for trace, value in case["traces"].items():
            assert value == golden[dataset]["traces"][trace], f"{dataset}/{trace}: 離散列が変化"
        assert case["metrics"].keys() == golden[dataset]["metrics"].keys()
        for metric, value in case["metrics"].items():
            expected = golden[dataset]["metrics"][metric]
            assert math.isclose(value, expected, rel_tol=0, abs_tol=1e-9), (
                f"{dataset}/{metric}: {value} != {expected}"
            )


def check_golden():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    # JSONのキー正規化後に比較する（SEA_THRESHOLDSの整数キーを含む）。
    assert golden["definition"] == json.loads(json.dumps(definition()))
    if environment() != golden["_env"]:
        warnings.warn(f"golden生成環境と実行環境が異なる: {golden['_env']} / {environment()}")
    compare(compute_all(), golden["cases"])


def test_proposed_regression():
    check_golden()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    arguments = parser.parse_args()
    if arguments.update:
        actual = compute_all()
        payload = dict(
            _env=environment(), definition=definition(), cases=actual,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            legacy_golden_sha256=hashlib.sha256(
                GOLDEN_PATH.with_name("regression_golden.json").read_bytes()).hexdigest(),
            mnist_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(default_data_dir().glob("train-*.gz"))},
        )
        GOLDEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value["coverage"] for key, value in actual.items()}, indent=2))
    else:
        check_golden()
        print("最終構成の回帰: PASS")
