"""1クライアントのFedHASP（製品の構成`fedsda`）が受け取る入力の系列を、そのまま書き出す。

製品のrunを、初期準備（事前学習の標本の生成を含む）と、評価の概念列・観測の系列の生成まで動かし、
標本の処理を始める直前で止める。乱数の順序を手で再現しないので、同じ条件の製品のrunと同じ系列になる。
書き出すもの（1つのdataset・seedにつき1つのNPZ）:
- pretraining_features・pretraining_labels: 初期モデルの事前学習に使った標本（生成の順）。
- features・labels・concept_ids: 評価の系列（client 0、標本の順）。concept_idsは評価にだけ使う真値。

特徴の表現（`--features`）:
- raw（既定）: 生成器が出した特徴そのもの。
- pretrained-shared-features: 事前学習した初期モデル（FedHASPの全モデルの出発点）の共有の特徴抽出器を通した値
  （合成データは32次元、画像は1,568次元）。比較手法に、FedHASPと同じ事前学習の表現を与えるための選択肢。
  抽出器は事前学習の直後の状態で固定する（FedHASPは系列の処理中に抽出器も学習するので、完全な同条件ではない）。
  入力は、製品のclientと同じくfloat32にしてから通す。pretraining_featuresも同じ抽出器を通す。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_single_client_stream_baselines.stream_export \
        --datasets sine2 mnist4 --seeds 0-29 --sample-count 5000 --output <出力先> --workers 8
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from hashlib import sha256
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

import copy  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from federated_learning_experiments.runtime import (  # noqa: E402
    fedsda_run_participant_factory,
    single_run_execution,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (  # noqa: E402
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (  # noqa: E402
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)

# 系列は集約の間隔に依らない（間隔は標本の処理の進め方だけで使う）。計画を作るために1つ決める。
_AGGREGATION_INTERVAL_FOR_PLAN = 50
_EXPORT_FORMAT_VERSION = 1
RAW_FEATURES = "raw"
PRETRAINED_SHARED_FEATURES = "pretrained-shared-features"
FEATURE_REPRESENTATIONS = (RAW_FEATURES, PRETRAINED_SHARED_FEATURES)


class _StreamCaptured(Exception):
    """系列を受け取ったので、標本の処理を始めずに止める合図。"""


def capture_single_client_stream(
    *, dataset_name, random_seed, sample_count, feature_representation=RAW_FEATURES
):
    """製品の`fedsda`の1クライアントのrunを系列の生成まで動かし、事前学習の標本と系列を返す。"""
    if feature_representation not in FEATURE_REPRESENTATIONS:
        raise ValueError(f"知らない特徴の表現です: {feature_representation}")
    sweep_plan = ExperimentSweepPlan(
        configuration_names=("fedsda",),
        dataset_names=(dataset_name,),
        random_seeds=(random_seed,),
        client_count=1,
        per_client_sample_count=sample_count,
        server_aggregation_intervals_per_client_samples=(_AGGREGATION_INTERVAL_FOR_PLAN,),
    )
    (planned_run,) = build_planned_experiment_runs(sweep_plan=sweep_plan)
    generated_before_stream = []
    captured = {}
    original_create = single_run_execution.create_observed_sample_generator
    original_intervals = single_run_execution.run_stream_protocol_intervals
    original_pretrain = fedsda_run_participant_factory.pretrain_initial_model

    def recording_pretrain(**arguments):
        pretrained = original_pretrain(**arguments)
        if "pretrained_classifier" in captured:
            raise RuntimeError("事前学習が2回呼ばれました")
        # 事前学習の直後の状態を複製して残す（その後の準備や処理の影響を受けない）。
        captured["pretrained_classifier"] = copy.deepcopy(pretrained.classifier)
        return pretrained

    def create_recording_generator(**arguments):
        generator = original_create(**arguments)
        generate_sample = generator.generate_sample

        def recording_generate_sample(*, concept_id):
            sample = generate_sample(concept_id=concept_id)
            if "streams" not in captured:
                generated_before_stream.append((concept_id, sample))
            return sample

        # 型の判定（type(generator)）は変えず、このインスタンスの呼び出しだけを記録する。
        generator.generate_sample = recording_generate_sample
        return generator

    def stop_before_processing(*, observed_client_streams, evaluation_concept_traces, **_):
        captured["streams"] = observed_client_streams
        captured["traces"] = evaluation_concept_traces
        raise _StreamCaptured

    single_run_execution.create_observed_sample_generator = create_recording_generator
    single_run_execution.run_stream_protocol_intervals = stop_before_processing
    fedsda_run_participant_factory.pretrain_initial_model = recording_pretrain
    try:
        execute_fedsda_experiment_run(run_settings=planned_run.run_settings)
    except Exception:
        # 製品の側で包まれた例外も来るので、系列を受け取れたかで成否を決める。
        if "streams" not in captured:
            raise
    else:
        raise RuntimeError("標本の処理の前で止まりませんでした")
    finally:
        single_run_execution.create_observed_sample_generator = original_create
        single_run_execution.run_stream_protocol_intervals = original_intervals
        fedsda_run_participant_factory.pretrain_initial_model = original_pretrain

    (stream,) = captured["streams"]
    (trace,) = captured["traces"]
    stream_samples = stream.observed_samples
    if len(stream_samples) != sample_count or len(trace.concept_ids_by_sample_index) != sample_count:
        raise RuntimeError("系列の長さが標本数と違います")
    # 系列の標本は、記録の末尾のsample_count件（系列の生成で作られたもの）と同じobjectでなければならない。
    pretraining_records = generated_before_stream[: len(generated_before_stream) - sample_count]
    stream_records = generated_before_stream[len(generated_before_stream) - sample_count :]
    if len(stream_records) != sample_count:
        raise RuntimeError("記録した系列の標本の数が標本数と違います")
    if any(record[1] is not sample for record, sample in zip(stream_records, stream_samples)):
        raise RuntimeError("記録した標本と系列の標本が対応しません")
    if any(concept_id != 0 for concept_id, _ in pretraining_records):
        raise RuntimeError("事前学習の標本に概念0以外があります")
    pretraining_features = np.array(
        [sample.feature_values for _, sample in pretraining_records], dtype=np.float64
    )
    features = np.array([sample.feature_values for sample in stream_samples], dtype=np.float64)
    if feature_representation == PRETRAINED_SHARED_FEATURES:
        if "pretrained_classifier" not in captured:
            raise RuntimeError("事前学習した分類器を受け取れませんでした")
        extractor = captured["pretrained_classifier"].feature_extractor
        extractor.eval()
        with torch.no_grad():
            pretraining_features, features = (
                extractor(torch.tensor(values, dtype=torch.float32)).numpy().astype(np.float64)
                for values in (pretraining_features, features)
            )
    return dict(
        pretraining_features=pretraining_features,
        pretraining_labels=np.array(
            [sample.class_label for _, sample in pretraining_records], dtype=np.int64
        ),
        features=features,
        labels=np.array([sample.class_label for sample in stream_samples], dtype=np.int64),
        concept_ids=np.array(trace.concept_ids_by_sample_index, dtype=np.int64),
    )


def stream_file_name(*, dataset_name, random_seed, sample_count):
    return f"{dataset_name}_seed{random_seed}_n{sample_count}.npz"


def array_digest(arrays):
    """書き出した配列の内容のhash（名前の順に、形・型・値を連結）。"""
    digest = sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _export_one(task):
    dataset_name, random_seed, sample_count, output_directory, feature_representation = task
    path = Path(output_directory) / stream_file_name(
        dataset_name=dataset_name, random_seed=random_seed, sample_count=sample_count
    )
    if path.exists() and path.with_suffix(".json").exists():
        return f"skip {path.name}"
    arrays = capture_single_client_stream(
        dataset_name=dataset_name,
        random_seed=random_seed,
        sample_count=sample_count,
        feature_representation=feature_representation,
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    metadata = dict(
        export_format_version=_EXPORT_FORMAT_VERSION,
        dataset_name=dataset_name,
        random_seed=random_seed,
        sample_count=sample_count,
        client_count=1,
        source_configuration="fedsda",
        feature_representation=feature_representation,
        array_sha256=array_digest(arrays),
        concept_change_count=int(np.count_nonzero(np.diff(arrays["concept_ids"]))),
        pretraining_sample_count=int(len(arrays["pretraining_labels"])),
        feature_count=int(arrays["features"].shape[1]),
    )
    temporary_metadata = path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_metadata.replace(path.with_suffix(".json"))
    return f"done {path.name}"


def parse_seeds(values):
    """`0-29`（両端込み）と整数の並びを受け付ける。"""
    seeds = []
    for value in values:
        if "-" in value:
            start, end = (int(part) for part in value.split("-", 1))
            if start > end:
                raise ValueError(f"seedの範囲が逆順です: {value}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(value))
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seedに重複か負の値があります")
    return seeds


def _source_record():
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--features", choices=FEATURE_REPRESENTATIONS, default=RAW_FEATURES)
    arguments = parser.parse_args(command_arguments)
    seeds = parse_seeds(arguments.seeds)
    if arguments.workers < 1 or arguments.sample_count < 1:
        raise ValueError("workersとsample-countは正の整数にしてください")
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    (arguments.output / f"export_provenance_{started}.json").write_text(
        json.dumps(
            dict(
                datasets=arguments.datasets,
                seeds=seeds,
                sample_count=arguments.sample_count,
                feature_representation=arguments.features,
                research_source_sha256=_source_record(),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tasks = [
        (dataset_name, seed, arguments.sample_count, str(arguments.output), arguments.features)
        for dataset_name in arguments.datasets
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for index, message in enumerate(executor.map(_export_one, tasks), start=1):
            print(f"[{index}/{len(tasks)}] {message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
