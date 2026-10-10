"""旧実装の計算量の計数の検査: 外側からの数え直し（PyTorchのhook）と、旧の計数が一致すること。

旧のclientは、処理の各所で、計数（`compute_counters`）を足す。このtestは、旧の全体runの間、共有部と
概念固有部の順伝播へ入力された標本数と、optimizerの更新回数を、旧の計数とは独立に数えて、旧の計数に、
漏れ・重複がないことを確かめる。旧実装は変えない。

数え方の前提: 旧のモデルは、共有部を`self.backbone(x)`、概念固有部を`self.adapter(features)`と
`self.head(...)`で呼ぶ（どれも`__call__`を通る）。概念固有部の標本数は、アダプタの順伝播で数える。
分類層（素の`nn.Linear`）は、同じ関数（`forward_from_features`）の中で、アダプタの出力を1回受けるので、
同じ標本数になる（分類層そのものは、数えていない）。
"""

import io
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.modules.module import register_module_forward_hook
from torch.optim.optimizer import register_optimizer_step_post_hook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proposed_regression as legacy_regression

from federated_drift_experiment import config, experiment
from federated_drift_experiment.experiment_spec.configuration import (
    ExperimentConfiguration,
    ParameterAssignment,
    temporary_config,
)
from federated_drift_experiment.models import ResidualConceptAdapter, SharedFeatureBackbone

# 旧の計数のうち、学習以外の用途（旧の集計が「推論」として足す6つ）。
LEGACY_INFERENCE_PHASE_NAMES = (
    "prediction",
    "detection",
    "statistics",
    "cross_evaluation",
    "initialization",
    "routing_recalibration",
)


def run_real_legacy_whole_run_counting_model_computation(dataset_name):
    """goldenの条件で、旧の全体run（参加者の準備の後から、終端まで）を、hookつきで実行する。

    戻り値: (hookで数えた値, 旧のclientの計数の合計, 旧の指標の集計の結果, 準備の直後の旧の計数の合計)。
    呼出し側の乱数は進めない。
    """
    configuration = ExperimentConfiguration(
        mode=legacy_regression.MODE,
        dataset=dataset_name,
        seed=0,
        concept_schedule="random",
        series="proposed_regression",
        sweep_parameter=None,
        sweep_value=None,
        parameters=(
            ParameterAssignment("aggregation_interval", 50),
            ParameterAssignment("fedsda_distance_threshold", 0.1),
        ),
        algorithm=legacy_regression.ALGORITHM,
    )
    observed_counts = Counter()
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    previous_thread_count = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with (
            temporary_config({**legacy_regression.COMMON, **legacy_regression.CASES[dataset_name]}),
            configuration.activated(),
            redirect_stdout(io.StringIO()),
            torch.random.fork_rng(devices=[]),
        ):
            config.DATASET = experiment.normalize_dataset_name(config.DATASET)
            legacy_mode_specification = experiment.MODE_SPECS[legacy_regression.MODE]
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            legacy_server, legacy_clients = experiment._setup_server_and_clients(
                legacy_mode_specification, config.FEDSDA_DISTANCE_THRESHOLD, False
            )
            legacy_counts_after_setup = Counter()
            for legacy_client in legacy_clients:
                legacy_counts_after_setup.update(legacy_client.compute_counters)
            # 順伝播した共有部が持つoptimizer（旧の共有部は、自分のoptimizerを属性に持つ）。
            shared_part_optimizers = []

            def count_forward(module, inputs, output):
                if type(module) is SharedFeatureBackbone:
                    part_name = "shared_part"
                    if not any(
                        module.optimizer is shared_part_optimizer
                        for shared_part_optimizer in shared_part_optimizers
                    ):
                        shared_part_optimizers.append(module.optimizer)
                elif type(module) is ResidualConceptAdapter:
                    part_name = "concept_specific_part"
                else:
                    return
                mode_name = "training" if torch.is_grad_enabled() else "inference"
                observed_counts[f"{part_name}_{mode_name}_examples"] += len(inputs[0])
                observed_counts[f"{part_name}_forward_calls"] += 1

            def count_optimizer_step(optimizer, arguments, keyword_arguments):
                # 共有部のoptimizerかどうかは、順伝播した共有部が持つoptimizerとの同一性で見分ける。
                part_name = (
                    "shared_part"
                    if any(
                        optimizer is shared_part_optimizer
                        for shared_part_optimizer in shared_part_optimizers
                    )
                    else "concept_specific_part"
                )
                observed_counts[f"{part_name}_optimizer_steps"] += 1

            forward_hook_handle = register_module_forward_hook(count_forward)
            optimizer_hook_handle = register_optimizer_step_post_hook(count_optimizer_step)
            try:
                chunk_sample_count = getattr(config, legacy_mode_specification.chunk_attr)
                round_count = config.TOTAL_DATA_POINTS // chunk_sample_count
                legacy_concept_schedules = experiment.make_concept_schedules(
                    config.N_CLIENTS, config.TOTAL_DATA_POINTS
                )
                legacy_data_streams = experiment.build_data_streams(legacy_concept_schedules)
                for round_index in range(round_count):
                    start_index = round_index * chunk_sample_count
                    end_index = start_index + chunk_sample_count
                    legacy_mode_specification.run_timestep(
                        legacy_clients,
                        legacy_server,
                        [stream[start_index:end_index] for stream in legacy_data_streams],
                        [schedule[start_index:end_index] for schedule in legacy_concept_schedules],
                        round_index,
                        legacy_mode_specification.use_server,
                        False,
                    )
                for legacy_client in legacy_clients:
                    legacy_client.finalize_incomplete_forward_validation()
                legacy_server.finalize_protocol(round_count)
            finally:
                forward_hook_handle.remove()
                optimizer_hook_handle.remove()
            legacy_counts = Counter()
            for legacy_client in legacy_clients:
                legacy_counts.update(legacy_client.compute_counters)
            # 旧の指標の集計（計算量の項目）。
            legacy_results = {}
            experiment._add_telemetry_results(
                legacy_results, legacy_clients, experiment._new_round_telemetry()
            )
            processed_sample_count = sum(
                legacy_client.processed_samples for legacy_client in legacy_clients
            )
    finally:
        torch.set_num_threads(previous_thread_count)
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    return (
        observed_counts,
        legacy_counts,
        legacy_results,
        legacy_counts_after_setup,
        processed_sample_count,
    )


@pytest.mark.parametrize("dataset_name", tuple(legacy_regression.CASES))
def test_legacy_computation_counters_match_independent_forward_and_optimizer_counts(dataset_name):
    """goldenの3ケースで、旧の計数が、hookで数えた、実際の順伝播とoptimizerの更新と一致する。"""
    (
        observed_counts,
        legacy_counts,
        legacy_results,
        legacy_counts_after_setup,
        processed_sample_count,
    ) = run_real_legacy_whole_run_counting_model_computation(dataset_name)
    # 参加者の準備（事前学習を含む）は、clientの計数に入らない。
    assert not any(legacy_counts_after_setup.values())
    # 共有部・概念固有部を通った標本数（学習と推論の合計）。
    assert (
        observed_counts["shared_part_training_examples"]
        + observed_counts["shared_part_inference_examples"]
        == legacy_counts["backbone_examples"]
    )
    assert (
        observed_counts["concept_specific_part_training_examples"]
        + observed_counts["concept_specific_part_inference_examples"]
        == legacy_counts["head_examples"]
    )
    # 学習（勾配つきの順伝播）。共有部と概念固有部で、同じ標本数。
    assert (
        observed_counts["shared_part_training_examples"]
        == observed_counts["concept_specific_part_training_examples"]
        == legacy_counts["training_examples"]
    )
    # 推論（勾配なしの順伝播）: 概念固有部を通った標本数が、旧の、学習以外の用途の標本数の合計。
    assert observed_counts["concept_specific_part_inference_examples"] == sum(
        legacy_counts[f"{phase_name}_examples"] for phase_name in LEGACY_INFERENCE_PHASE_NAMES
    )
    # 概念固有部の順伝播の回数が、旧の、用途別の呼出しの回数の合計。
    assert observed_counts["concept_specific_part_forward_calls"] == sum(
        legacy_counts[f"{phase_name}_forward_calls"]
        for phase_name in (*LEGACY_INFERENCE_PHASE_NAMES, "training")
    )
    # optimizerの更新回数。旧の`optimizer_steps`は、概念固有部の更新回数。
    assert (
        observed_counts["concept_specific_part_optimizer_steps"]
        == legacy_counts["optimizer_steps"]
        == legacy_counts["head_optimizer_steps"]
    )
    assert (
        observed_counts["shared_part_optimizer_steps"] == legacy_counts["backbone_optimizer_steps"]
    )
    # 旧の計数の用途は、上で照合したものだけ（照合していない、モデルの計算の計数がない）。
    assert set(legacy_counts) <= {
        "backbone_examples",
        "head_examples",
        "optimizer_steps",
        "backbone_optimizer_steps",
        "head_optimizer_steps",
        "drift_detector_updates",
        "drift_detector_hypotheses",
        *(
            f"{phase_name}_{count_name}"
            for phase_name in (*LEGACY_INFERENCE_PHASE_NAMES, "training")
            for count_name in ("examples", "forward_calls")
        ),
    }
    # 旧の指標の集計（goldenが比べる、モデルの計算の5項目）も、hookで数えた値と一致する。
    assert (
        legacy_results["compute_inference_examples_total"]
        == observed_counts["concept_specific_part_inference_examples"]
    )
    assert (
        legacy_results["compute_training_examples_total"]
        == observed_counts["concept_specific_part_training_examples"]
    )
    assert (
        legacy_results["compute_optimizer_steps_total"]
        == observed_counts["concept_specific_part_optimizer_steps"]
    )
    assert legacy_results["compute_backbone_examples_total"] == legacy_counts["backbone_examples"]
    assert legacy_results["compute_head_examples_total"] == legacy_counts["head_examples"]
    # 検出器の更新は、標本1件につき、全体と正解クラスの2つ。
    assert legacy_counts["drift_detector_updates"] == 2 * processed_sample_count
    assert (
        legacy_results["compute_drift_detector_updates_total"]
        == legacy_counts["drift_detector_updates"]
    )
    # 中身のある照合である。
    assert processed_sample_count > 0
    assert legacy_counts["training_examples"] > 0
    assert observed_counts["concept_specific_part_inference_examples"] > processed_sample_count
    assert legacy_counts["optimizer_steps"] > 0


def test_legacy_computation_counters_match_golden_for_all_cases():
    """上の検査の旧の実行が、goldenの計算量の7項目を再現している（Windowsの基準環境）。"""
    if sys.platform != "win32":
        pytest.skip("Windows用のgoldenとの照合は、Windowsでだけ行う")
    import json

    golden = json.loads(legacy_regression.GOLDEN_PATH.read_text(encoding="utf-8"))
    for dataset_name in legacy_regression.CASES:
        _, _, legacy_results, _, _ = run_real_legacy_whole_run_counting_model_computation(
            dataset_name
        )
        for legacy_metric_name in legacy_regression.METRICS:
            if legacy_metric_name.startswith("compute_"):
                assert float(legacy_results[legacy_metric_name]) == pytest.approx(
                    golden["cases"][dataset_name]["metrics"][legacy_metric_name], rel=0, abs=1e-9
                ), (dataset_name, legacy_metric_name)
