"""新IDの予測初期質量を調べる独立した研究用CLI。共通設定への登録はしない。"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
_MODES = ("validation", "immediate", "ratio-preserved", "small-new-mass")
_DATASETS = ("sea2", "sea4", "sine2", "circle2", "mnist2", "mnist4")


@contextmanager
def _prediction_mass_policy(mode, events):
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )

    if mode not in ("ratio-preserved", "small-new-mass"):
        raise ValueError("研究用重みmodeが不正です。")
    original = FixedShareWeightController._synchronize_model_pool

    def synchronize(controller, *, validated_model_ids):
        old_weights = controller.weights_by_model_id
        retained = sorted(set(validated_model_ids) & set(old_weights))
        added = sorted(set(validated_model_ids) - set(old_weights))
        retained_mass = math.fsum(old_weights[i] for i in retained)
        original(controller, validated_model_ids=validated_model_ids)
        if not added or not retained or retained_mass <= 0:
            return
        scale = controller._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        new_mass = 1.0 / (len(validated_model_ids) * (scale if mode == "small-new-mass" else 1))
        retained_total = 1.0 - len(added) * new_mass
        weights = {
            i: retained_total * old_weights[i] / retained_mass if i in old_weights else new_mass
            for i in validated_model_ids
        }
        controller._weights_by_model_id = weights
        events.append(
            dict(
                mode=mode,
                model_ids=list(validated_model_ids),
                new_model_ids=added,
                old_weights=old_weights,
                new_weights=dict(weights),
                time_scale=scale,
            )
        )

    with patch.object(FixedShareWeightController, "_synchronize_model_pool", synchronize):
        yield


def _rng_fingerprint():
    import numpy as np
    import torch

    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes())
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def _run_with_policy(settings, mode):
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    if mode not in _MODES:
        raise ValueError("診断modeが不正です。")
    events = []
    if mode in ("ratio-preserved", "small-new-mass"):
        with _prediction_mass_policy(mode, events):
            result = execute_fedsda_experiment_run(run_settings=settings)
    else:
        result = execute_fedsda_experiment_run(run_settings=settings)
    return result, events


def _code_identity():
    digest = hashlib.sha256()
    for path in sorted((_ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(_ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return {
        "source_tree_sha256": digest.hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _execute_job(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch

    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    if _code_identity() != job["code_identity"]:
        raise RuntimeError("計画後にsource/scriptが変更されました。")
    directory = Path(job["output_directory"]) / job["run_name"]
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    settings = build_experiment_run_settings(
        configuration_name="fedsda-without-candidate-validation"
        if job["mode"] != "validation"
        else "fedsda",
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=job["dataset"],
            random_seed=job["seed"],
            client_count=10,
            per_client_sample_count=job["sample_count"],
            server_aggregation_interval_per_client_samples=job["interval"],
        ),
    )
    rng_before = _rng_fingerprint()
    result, events = _run_with_policy(settings, job["mode"])
    rng_after = _rng_fingerprint()
    if rng_before != rng_after:
        raise RuntimeError("runが外側の乱数状態を変更しました。")
    arrays = build_fedsda_run_record_tables(experiment_run=result)
    repository = result.measured_run.participants.server_operations.owners.global_model_repository
    final_global_parameters = {
        str(model_id): {
            key: dict(
                dtype=str(value.dtype),
                shape=list(value.shape),
                sha256=hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            )
            for key, value in repository.get_global_model_parameters(model_id=model_id).items()
        }
        for model_id in repository.global_model_ids
    }
    if _code_identity() != job["code_identity"]:
        raise RuntimeError("実行中にsource/scriptが変更されました。")
    npz = directory / "records.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    payload = {
        "format": "prediction-initial-mass-diagnostic-v1",
        "final_global_parameters": final_global_parameters,
        **{
            key: job[key]
            for key in ("dataset", "interval", "seed", "sample_count", "mode", "purpose")
        },
        **job["provenance"],
        **job["code_identity"],
        "environment": {
            "system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            **{
                key: os.environ[key]
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "processed_sample_count": result.run_metrics.processed_sample_count,
        "run_settings": convert_settings_to_plain_mapping(settings),
        "run_metrics": convert_settings_to_plain_mapping(
            result.run_metrics, non_finite_float_as_none=True
        ),
        "record_table_file_name": npz.name,
        "record_table_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "intervention_events": events,
        "rng_before_sha256": rng_before,
        "rng_after_sha256": rng_after,
    }
    with (directory / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return {"run_name": job["run_name"], "accuracy": result.run_metrics.prediction_accuracy}


def _main(argv=None):
    from federated_learning_experiments.sweep.seed_argument_parsing import SeedListArgumentAction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, help="dataset:集約間隔")
    parser.add_argument(
        "--seeds",
        action=SeedListArgumentAction,
        nargs="+",
        default=[0, 1, 2],
        help="seed列挙と両端込み範囲（0-2 5）。重複不可、展開後100000個以内",
    )
    parser.add_argument("--modes", choices=_MODES, nargs="+", default=list(_MODES))
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.purpose.strip() or not 1 <= args.workers <= 14 or args.sample_count <= 0:
        raise ValueError("目的・worker数・標本数が不正です。")
    if min(args.seeds) < 0 or max(args.seeds) >= 2**32:
        raise ValueError("seedは0以上2**32未満が必要です。")
    if any(len(items) != len(set(items)) for items in (args.cases, args.seeds, args.modes)):
        raise ValueError("case/seed/modeの重複は許可しません。")
    cases = []
    for value in args.cases:
        dataset, text = value.split(":")
        interval = int(text)
        if dataset not in _DATASETS or interval <= 0 or args.sample_count % interval:
            raise ValueError("dataset・間隔が不正、または末尾に未処理標本が残ります。")
        if (dataset, interval) in cases:
            raise ValueError("同じcaseが重複しています。")
        cases.append((dataset, interval))
    output = args.output_directory.resolve()
    if output.is_relative_to(_ROOT / "src"):
        raise ValueError("src内には結果を保存できません。")
    identity = _code_identity()
    provenance = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
        "has_uncommitted_changes": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT).strip()
        ),
    }
    jobs = []
    for dataset, interval in cases:
        for seed in args.seeds:
            for mode in args.modes:
                name = f"{dataset}_{mode}_i{interval}_s{seed}_n{args.sample_count}"
                if (output / name).exists():
                    raise FileExistsError(output / name)
                jobs.append(
                    dict(
                        dataset=dataset,
                        interval=interval,
                        seed=seed,
                        mode=mode,
                        sample_count=args.sample_count,
                        purpose=args.purpose,
                        output_directory=str(output),
                        run_name=name,
                        code_identity=identity,
                        provenance=provenance,
                    )
                )
    if args.print_plan:
        print(json.dumps({"workers": args.workers, "jobs": jobs}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    # Python3.10のサーバでも、各子processは1runで必ず終了する。
    with multiprocessing.get_context("spawn").Pool(
        processes=args.workers, maxtasksperchild=1
    ) as pool:
        for result in pool.imap_unordered(_execute_job, jobs, chunksize=1):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
