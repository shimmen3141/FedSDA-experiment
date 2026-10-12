"""新実装の重要動作を基準版・数値環境ごとに固定するtest専用helper。"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (  # noqa: E402
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.data.mnist.mnist_training_data import (  # noqa: E402
    resolve_mnist_data_directory,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (  # noqa: E402
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (  # noqa: E402
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_record_tables import (  # noqa: E402
    build_fedavg_run_record_tables,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (  # noqa: E402
    FedavgRunSettings,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (  # noqa: E402
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_record_tables import (  # noqa: E402
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_settings import (  # noqa: E402
    FeddriftRunSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (  # noqa: E402
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (  # noqa: E402
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

BASELINE_VERSION = 1
SCHEMA_VERSION = 1
GOLDEN_DIRECTORY = Path(__file__).with_name("behavior_regression_goldens") / "v1"
CASE_NAMES = (
    *EXPERIMENT_CONFIGURATION_NAMES,
    "fedsda-sine2-events",
    "fedsda-mnist2-decisions",
    "feddrift-drift-tail",
    "fedavg-no-complete-interval",
)
ENVIRONMENT_KEYS = {"system", "machine", "python", "numpy", "torch", "device", "dtype", "threads"}
CASE_KEYS = {"run_settings", "run_metrics", "record_tables", "global_parameters", "coverage"}
DIGEST_KEYS = {"shape", "dtype", "sha256"}
TOP_LEVEL_KEYS = {
    "baseline_version",
    "schema_version",
    "source_commit",
    "environment",
    "mnist_sha256",
    "cases",
}


def _plain(value):
    return json.loads(
        json.dumps(convert_settings_to_plain_mapping(value, non_finite_float_as_none=True))
    )


def build_case_settings(case_name):
    """builderの方式を維持して、明示した代表条件にだけ置換する。"""
    if case_name not in CASE_NAMES:
        raise ValueError(f"unknown golden case: {case_name}")
    rich = case_name in ("fedsda-sine2-events", "fedsda-mnist2-decisions")
    mnist = case_name == "fedsda-mnist2-decisions"
    configuration = (
        "fedsda"
        if rich
        else "feddrift"
        if case_name == "feddrift-drift-tail"
        else "fedavg"
        if case_name == "fedavg-no-complete-interval"
        else case_name
    )
    conditions = ExperimentRunConditions(
        dataset_name="mnist2" if mnist else "sine2",
        random_seed=0 if rich else 71 if case_name == "feddrift-drift-tail" else 37,
        client_count=3 if case_name == "fedsda-sine2-events" else 2,
        per_client_sample_count=100
        if mnist
        else 1500
        if rich
        else 17
        if case_name == "feddrift-drift-tail"
        else 9
        if case_name == "fedavg-no-complete-interval"
        else 35,
        server_aggregation_interval_per_client_samples=50
        if rich
        else 4
        if case_name == "feddrift-drift-tail"
        else 10,
    )
    settings = build_experiment_run_settings(
        configuration_name=configuration, experiment_run_conditions=conditions
    )
    schedule = replace(
        settings.execution_settings.concept_schedule_settings,
        minimum_sample_index_gap_before_change_trial=30 if mnist else 100 if rich else 1,
        per_eligible_sample_concept_change_probability=0.015
        if rich
        else 1.0
        if case_name == "feddrift-drift-tail"
        else 0.5,
    )
    participant = settings.run_participant_settings
    pretraining = InitialModelPretrainingSettings(
        pretraining_sample_count=100 if rich else 24,
        pretraining_epoch_count=2 if mnist or not rich else 10,
        pretraining_batch_sample_count=32 if rich else 8,
    )
    changes = dict(
        initial_model_pretraining_settings=pretraining,
        hidden_layer_widths=participant.hidden_layer_widths if rich else (5, 4),
    )
    if type(settings) is FedavgRunSettings:
        changes.update(
            local_training_batch_sample_count=4,
            updates_per_sample=2,
            local_update_interval_per_client_samples=3,
        )
    elif type(settings) is FeddriftRunSettings:
        threshold = 0.03 if case_name == "feddrift-drift-tail" else 0.1
        changes.update(
            run_client_settings=replace(
                participant.run_client_settings,
                loss_increase_threshold=threshold,
                local_training_batch_sample_count=4,
                added_evaluation_batch_sample_count=3,
            ),
            server_settings=replace(
                participant.server_settings,
                maximum_same_cluster_loss_increase=threshold,
                initial_model_aggregation_weight=24,
                local_training_round_count=3 if case_name == "feddrift-drift-tail" else 2,
            ),
        )
    else:
        client = participant.run_client_settings
        if not rich:
            client = replace(
                client,
                scalar_settings=replace(
                    client.scalar_settings, local_training_batch_sample_count=4
                ),
                candidate_epoch_training_settings=replace(
                    client.candidate_epoch_training_settings, maximum_batch_sample_count=4
                ),
            )
        if mnist:
            client = replace(
                client,
                parameter_optimizer_settings=replace(
                    client.parameter_optimizer_settings, learning_rate=0.01
                ),
                rebuilt_model_parameter_optimizer_settings=replace(
                    client.rebuilt_model_parameter_optimizer_settings, learning_rate=0.01
                ),
            )
        changes["run_client_settings"] = client
    return replace(
        settings,
        execution_settings=replace(settings.execution_settings, concept_schedule_settings=schedule),
        run_participant_settings=replace(participant, **changes),
        run_metric_settings=replace(
            settings.run_metric_settings, post_change_recovery_window_sample_count=50
        )
        if rich
        else settings.run_metric_settings,
    )


def numerical_environment():
    """比較対象のCPU数値計算環境。CPU型番差はCI診断の対象とする。"""
    return dict(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        device="cpu",
        dtype=str(torch.get_default_dtype()),
        threads=torch.get_num_threads(),
    )


def golden_file_name(environment):
    return (
        f"{environment['system']}-{environment['machine']}-python{environment['python']}-numpy{environment['numpy']}-torch{environment['torch']}.json"
    ).lower()


def mnist_sha256():
    directory = resolve_mnist_data_directory()
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    }


def _canonical_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return value


def array_snapshot(array):
    """値・shape・dtypeを、ZIPやnative byte orderと無関係に固定する。"""
    if type(array) is not np.ndarray or array.dtype.kind not in "biufU":
        raise TypeError("numeric/bool/unicode ndarray is required")
    payload = dict(
        shape=list(array.shape), dtype=array.dtype.name, values=_canonical_value(array.tolist())
    )
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return dict(shape=payload["shape"], dtype=payload["dtype"], sha256=digest)


def assert_case_matches(case_name, actual, expected, path=""):
    """相違した設定・指標・列・parameterのfield pathを示す。"""
    label = f"{case_name}: {path}"
    if isinstance(expected, dict):
        assert isinstance(actual, dict), label
        assert set(actual) == set(expected), (
            f"{label} keys missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}"
        )
        for key in sorted(expected):
            assert_case_matches(case_name, actual[key], expected[key], f"{path}.{key}")
    else:
        assert type(actual) is type(expected) and actual == expected, (
            f"{label}: actual={actual!r}, expected={expected!r}"
        )


def _coverage(case_name, run, tables):
    result = run.measured_run.run_result
    coverage = dict(
        generated_sample_count_per_client=result.generated_sample_count_per_client,
        processed_sample_count_per_client=result.processed_sample_count_per_client,
        unprocessed_tail_sample_count_per_client=result.unprocessed_tail_sample_count_per_client,
        synchronization_round_count=result.synchronization_interval_count,
    )

    def column(name):
        return tables.get(name, np.asarray([], dtype=np.int64))

    accepted = column("candidate_validation_decision.candidate_accepted").astype(np.bool_)
    complete = ~column("candidate_validation_decision.is_incomplete_at_run_end").astype(np.bool_)
    coverage.update(
        candidate_accepted_count=int(np.count_nonzero(accepted & complete)),
        candidate_rejected_count=int(np.count_nonzero(~accepted & complete)),
        decision_reasons=sorted(
            set(column("candidate_validation_decision.decision_reason").tolist())
        ),
        training_model_switch_count=int(column("training_model_switch.client_id").size),
        merged_model_count=int(
            np.count_nonzero(column("model_clustering.merged_with_other_models"))
        ),
        recalibration_replayed_sample_count=int(
            column(
                "client_weight_update_counts.fixed_share_weight_aggregation_recalibration_replayed_sample_count"
            ).sum()
        ),
        registered_model_count=int(column("model_registration.model_id").size),
        detection_count=run.run_metrics.training_model_switch_detection_metrics.detection_count,
    )
    return coverage


def assert_required_paths(case_name, coverage):
    """イベントが空のケースを重要経路の基準に採用しない。"""
    if case_name.startswith("fedsda-") and case_name in (
        "fedsda-sine2-events",
        "fedsda-mnist2-decisions",
    ):
        for key in (
            "candidate_accepted_count",
            "candidate_rejected_count",
            "training_model_switch_count",
            "recalibration_replayed_sample_count",
        ):
            assert coverage[key] > 0, f"{case_name}: required {key}"
        reasons = set(coverage["decision_reasons"])
        if case_name == "fedsda-sine2-events":
            assert {
                "first_segment_margin_failed",
                "alternative_reference_within_historical_loss_tolerance",
            } <= reasons, (case_name, reasons)
            assert coverage["merged_model_count"] > 0, case_name
        else:
            assert "second_segment_margin_failed" in reasons, (case_name, reasons)
    if case_name == "feddrift-drift-tail":
        assert coverage["detection_count"] > 0 and coverage["registered_model_count"] > 1, coverage
        assert coverage["processed_sample_count_per_client"] == 16, coverage
        assert coverage["unprocessed_tail_sample_count_per_client"] == 1, coverage
        assert coverage["synchronization_round_count"] == 4, coverage
    if case_name == "fedavg":
        assert coverage["synchronization_round_count"] == 3, coverage
        assert (
            coverage["processed_sample_count_per_client"] == 30
            and coverage["unprocessed_tail_sample_count_per_client"] == 5
        ), coverage
    if case_name == "fedavg-no-complete-interval":
        assert (
            coverage["processed_sample_count_per_client"]
            == coverage["synchronization_round_count"]
            == 0
        ), coverage
        assert coverage["unprocessed_tail_sample_count_per_client"] == 9, coverage


def capture_case(case_name):
    settings = build_case_settings(case_name)
    if type(settings) is FedavgRunSettings:
        run = execute_fedavg_experiment_run(run_settings=settings)
        tables = build_fedavg_run_record_tables(experiment_run=run)
        parameters = {0: run.measured_run.participants.server_operations.global_parameters}
    elif type(settings) is FeddriftRunSettings:
        run = execute_feddrift_experiment_run(run_settings=settings)
        tables = build_feddrift_run_record_tables(experiment_run=run)
        assert any(
            event.stage_name == "observed_stream_finalization"
            for event in run.measured_run.run_result.execution_events
        ), case_name
        assert settings.run_participant_settings.server_settings.local_training_round_count == (
            3 if case_name == "feddrift-drift-tail" else 2
        ), case_name
        repository = run.measured_run.participants.run_server.global_model_repository
        parameters = {
            mid: repository.get_global_model_parameters(model_id=mid)
            for mid in repository.global_model_ids
        }
    else:
        run = execute_fedsda_experiment_run(run_settings=settings)
        tables = build_fedsda_run_record_tables(experiment_run=run)
        repository = run.measured_run.participants.server_operations.owners.global_model_repository
        parameters = {
            mid: repository.get_global_model_parameters(model_id=mid)
            for mid in repository.global_model_ids
        }
    coverage = _coverage(case_name, run, tables)
    assert_required_paths(case_name, coverage)
    return dict(
        run_settings=_plain(settings),
        run_metrics=_plain(run.run_metrics),
        record_tables={name: array_snapshot(value) for name, value in tables.items()},
        global_parameters={
            f"{mid}.{name}": array_snapshot(value.detach().cpu().numpy())
            for mid, values in parameters.items()
            for name, value in values.items()
        },
        coverage=coverage,
    )


def load_golden(path):
    """未知版や破損を環境選択の前に拒否する。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(document, allow_nan=False)
        assert set(document) == TOP_LEVEL_KEYS
        for key, version in (
            ("baseline_version", BASELINE_VERSION),
            ("schema_version", SCHEMA_VERSION),
        ):
            assert type(document[key]) is int and document[key] == version, key
        assert re.fullmatch(r"[0-9a-f]{40}", document["source_commit"]), "source_commit"
        assert set(document["environment"]) == ENVIRONMENT_KEYS, "environment"
        assert all(
            isinstance(v, str) for k, v in document["environment"].items() if k != "threads"
        ), "environment types"
        assert (
            type(document["environment"]["threads"]) is int
            and document["environment"]["threads"] == 1
        )
        assert (
            document["environment"]["device"] == "cpu"
            and document["environment"]["dtype"] == "torch.float32"
        )
        assert set(document["mnist_sha256"]) == {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
        }
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value) for value in document["mnist_sha256"].values()
        )
        assert set(document["cases"]) == set(CASE_NAMES), "cases"
        for name, case in document["cases"].items():
            assert set(case) == CASE_KEYS, name
            assert_case_matches(
                name, case["run_settings"], _plain(build_case_settings(name)), "run_settings"
            )
            assert isinstance(case["run_metrics"], dict) and case["run_metrics"], name
            for section in ("record_tables", "global_parameters"):
                assert isinstance(case[section], dict) and case[section], (name, section)
                for item, digest in case[section].items():
                    assert set(digest) == DIGEST_KEYS, (name, section, item)
                    assert isinstance(digest["shape"], list) and all(
                        type(n) is int and n >= 0 for n in digest["shape"]
                    )
                    assert isinstance(digest["dtype"], str) and (
                        (
                            re.fullmatch(r"str[0-9]+", digest["dtype"])
                            and int(digest["dtype"][3:]) % 32 == 0
                        )
                        or np.dtype(digest["dtype"]).kind in "biufU"
                    )
                    assert re.fullmatch(r"[0-9a-f]{64}", digest["sha256"])
            assert_required_paths(name, case["coverage"])
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid behavior golden: {path}: {error}") from error
    return document


def find_golden_path(environment, directory=GOLDEN_DIRECTORY):
    matches = [
        path
        for path in sorted(directory.glob("*.json"))
        if load_golden(path)["environment"] == environment
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate golden environment: {matches}")
    return matches[0] if matches else None


def _write_exclusive(path, document):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generate_golden(output, source_commit):
    """2つの新processの全結果一致後に、新規出力へ保存する。"""
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full commit hash")
    git_command = ["git"]
    mounted_windows_root = re.fullmatch(r"/mnt/([a-z])/(.+)", ROOT.as_posix())
    if sys.platform == "linux" and mounted_windows_root and (ROOT / ".git").is_file():
        git_reference = (ROOT / ".git").read_text(encoding="utf-8").strip()
        if re.match(r"gitdir: [A-Za-z]:[/\\]", git_reference):
            drive, remainder = mounted_windows_root.groups()
            git_command = ["git.exe", "-C", f"{drive.upper()}:/{remainder}"]
    head = subprocess.check_output([*git_command, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if source_commit != head:
        raise ValueError("source_commit must match HEAD")
    source_changes = subprocess.check_output(
        [*git_command, "status", "--porcelain", "--untracked-files=all", "--", "src"],
        cwd=ROOT,
        text=True,
    )
    if source_changes.strip():
        raise ValueError("src must match the source commit, including untracked files")
    with tempfile.TemporaryDirectory(prefix="behavior-golden-") as temporary:
        documents = []
        for iteration in (1, 2):
            candidate = Path(temporary) / f"run-{iteration}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    str(candidate),
                    "--source-commit",
                    source_commit,
                ],
                cwd=ROOT,
                check=True,
            )
            documents.append(load_golden(candidate))
        assert_case_matches("independent-processes", documents[0], documents[1])
        _write_exclusive(output, documents[0])
    return output


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if bool(args.output) == bool(args.worker_output):
        parser.error("exactly one output is required")
    environment = numerical_environment()
    if (
        environment["threads"] != 1
        or environment["dtype"] != "torch.float32"
        or any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    ):
        parser.error(
            "float32 CPU, OMP_NUM_THREADS=MKL_NUM_THREADS=1 and torch threads=1 are required"
        )
    if args.worker_output:
        cases = {}
        for name in CASE_NAMES:
            cases[name] = capture_case(name)
            print(f"captured {name}", flush=True)
        _write_exclusive(
            args.worker_output,
            dict(
                baseline_version=BASELINE_VERSION,
                schema_version=SCHEMA_VERSION,
                source_commit=args.source_commit,
                environment=environment,
                mnist_sha256=mnist_sha256(),
                cases=cases,
            ),
        )
    else:
        print(generate_golden(args.output, args.source_commit))


if __name__ == "__main__":
    _main()
