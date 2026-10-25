"""clean commit・全設定・環境を確認したネイティブ既定結果だけを再利用する。"""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import platform
import re
import subprocess
import sys

from tools.research_s3_validation.run import ROOT, sha


def environment():
    return dict(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=version("numpy"),
        torch=version("torch"),
    )


def strip_types(value):
    if isinstance(value, dict):
        return {k: strip_types(v) for k, v in value.items() if k != "settings_type"}
    if isinstance(value, (tuple, list)):
        return [strip_types(v) for v in value]
    return value


def native_tree_proof(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("既定結果のcommitが不正です")
    prefix = "src/federated_learning_experiments/"
    result = subprocess.run(
        ["git", "ls-tree", "-r", commit, "--", prefix],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    blobs = {}
    for line in result.stdout.splitlines():
        info, name = line.split("\t", 1)
        if name.endswith(".py"):
            blobs[name] = info.split()[2]
    current = {p.relative_to(ROOT).as_posix() for p in (ROOT / prefix).rglob("*.py")}
    if not blobs or set(blobs) != current:
        raise ValueError("既定結果の製品ソース集合が現在と異なります")
    for name, blob in blobs.items():
        data = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            != blob
        ):
            raise ValueError(f"既定結果の製品ソースが現在と異なります: {name}")
    return dict(commit=commit, native_file_count=len(blobs), native_tree_verified=True)


def expected_settings(job):
    sys.path.insert(0, str(ROOT / "src"))
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name=job["dataset"],
        random_seed=job["seed"],
        client_count=10,
        per_client_sample_count=job["sample_count"],
        server_aggregation_interval_per_client_samples=job["interval"],
    )
    return strip_types(
        asdict(
            build_experiment_run_settings(
                configuration_name="fedsda", experiment_run_conditions=conditions
            )
        )
    )


def validate_record(record, job, proofs, expected_environment=None):
    provenance = record["run_provenance"]
    commit = provenance["code_commit"]
    if (
        provenance["has_uncommitted_changes"] is not False
        or provenance["behavior_baseline_version"] != 12
    ):
        raise ValueError("既定結果はcleanな基準版12である必要があります")
    if commit not in proofs:
        proofs[commit] = native_tree_proof(commit)
    if strip_types(record["run_settings"]) != expected_settings(job):
        raise ValueError("既定結果の全run設定が現在と一致しません")
    if (
        expected_environment is not None
        and record["environment"] != expected_environment
    ):
        raise ValueError(
            "既定結果と実行環境が異なります。--baselineを外して既定も再計測してください"
        )
    m = record["run_metrics"]
    counts = m["prediction_diagnostic_metrics"]["prediction_correct_counts"]
    n = 10 * job["sample_count"]
    if (
        m["processed_sample_count"] != n
        or counts["predicted_sample_count"] != n
        or abs(
            counts["combined_prediction_correct_count"] / n - m["prediction_accuracy"]
        )
        > 1e-12
    ):
        raise ValueError("既定結果の全標本数・精度が一致しません")


def convert(record, job, digest):
    m = record["run_metrics"]
    volume = m["communication_volume"]
    metrics = dict(
        dataset=job["dataset"],
        interval=job["interval"],
        sample_count=job["sample_count"],
        seed=job["seed"],
        variant="product-default",
        check=False,
        research_variant="product-default",
        fixed_concept=None,
        accuracy=m["prediction_accuracy"],
        record_accuracy=m["prediction_accuracy"],
        uploaded_bytes=volume["uploaded_byte_count"],
        downloaded_bytes=volume["downloaded_byte_count"],
        model_computation_counts=strip_types(m["model_computation_counts"]),
        preparation_model_computation_counts=None,
        seconds=None,
        concept_accuracies=None,
        prediction_period_accuracies=strip_types(
            m["prediction_diagnostic_metrics"]["combined_prediction_period_accuracies"]
        ),
        research_events=[],
    )
    return dict(
        job=job,
        metrics=metrics,
        input_signature=None,
        reused_default=True,
        baseline_record_sha256=digest,
    )


def load_records(directory, jobs, require_environment):
    expected = {
        (j["dataset"], j["interval"], j["sample_count"], j["seed"]): j
        for j in jobs
        if j["arm"] == "product-default"
    }
    records, proofs = {}, {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        conditions = record["run_settings"]["execution_settings"][
            "experiment_run_conditions"
        ]
        key = (
            conditions["dataset_name"],
            conditions["server_aggregation_interval_per_client_samples"],
            conditions["per_client_sample_count"],
            conditions["random_seed"],
        )
        if key not in expected:
            continue
        if key in records:
            raise ValueError("再利用する既定結果が重複しています")
        validate_record(
            record,
            expected[key],
            proofs,
            environment() if require_environment else None,
        )
        records[key] = (record, sha(path))
    if set(records) != set(expected):
        raise ValueError("再利用する同条件の既定結果が不足しています")
    envs = {json.dumps(r[0]["environment"], sort_keys=True) for r in records.values()}
    if len(envs) != 1:
        raise ValueError("既定結果内で実行環境が混在しています")
    return [(expected[k], *records[k]) for k in sorted(records)], dict(
        native_proofs=proofs, baseline_environment=json.loads(next(iter(envs)))
    )
