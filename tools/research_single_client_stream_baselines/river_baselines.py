"""書き出した1クライアントの系列で、Riverのストリーム学習の比較手法を予測→学習の順（prequential）で評価する。

プロジェクトのコードは読み込まない（numpyとRiverだけの別のvenvで動かす。プロジェクトの固定のvenvへRiverを
入れると、numpyなどの版が変わりうるため）。各手法は、FedHASPの初期モデルと同じ事前学習の標本（概念0）を
先に学習し（評価しない）、その後の系列の全標本で、予測してから学習する。精度は全標本の正解の割合で、
製品の`prediction_accuracy`（全標本の正解の割合）と同じ定義。

手法（Riverの既定の値。DWMはRiverに無いので、Kolter & Maloof (2007) のアルゴリズム1を実装する）:
- arf: forest.ARFClassifier
- srp: ensemble.SRPClassifier
- leveraging_bagging: ensemble.LeveragingBaggingClassifier（基本はtree.HoeffdingTreeClassifier）
- hoeffding_adaptive_tree: tree.HoeffdingAdaptiveTreeClassifier
- dwm: Dynamic Weighted Majority（β=0.5、θ=0.01、p=1、基本はtree.HoeffdingTreeClassifier）。
  元論文は基本の学習器にNaive Bayesと決定木を使う。RiverのGaussianNBは画素をそのまま入れたMNISTで
  単独でも当て推量の精度（2026-11-17の手元の確認で0.11）になるので、決定木の側を使う。

使い方（Riverのvenv）:
    python tools/research_single_client_stream_baselines/river_baselines.py \
        --streams <書き出し先> --output <出力先> --methods arf srp leveraging_bagging hoeffding_adaptive_tree dwm --workers 8
"""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np

METHOD_NAMES = ("arf", "srp", "leveraging_bagging", "hoeffding_adaptive_tree", "dwm")
_RESULT_FORMAT_VERSION = 1


class DynamicWeightedMajority:
    """Dynamic Weighted Majority（Kolter & Maloof, JMLR 2007, Algorithm 1）。

    元論文は専門家が常に予測を出す前提。Riverの学習器は、まだ何も学習していないときに予測としてNoneを返す。
    ここでは、Noneを返した専門家は誤りとして重みを減らし、全体の予測がNone（票なし）なら誤りとして専門家を足す。
    事前学習の標本を学習した後は、新しく足した専門家以外ではNoneは起きない。

    各標本で、(1) 各専門家の予測に重みを付けて投票し、全体の予測を返す（predict_one）。
    ラベルを受け取ったら（learn_one）、(2) 周期pの標本で、誤った専門家の重みにβを掛け、
    最大の重みで正規化し、重みがθ未満の専門家を消し、全体の予測が誤っていれば重み1の専門家を足す。
    (3) 全部の専門家をその標本で学習する。
    """

    def __init__(self, *, make_expert, beta=0.5, theta=0.01, period=1):
        self.make_expert = make_expert
        self.beta = beta
        self.theta = theta
        self.period = period
        self.experts = [make_expert()]
        self.weights = [1.0]
        self.sample_index = 0
        self._last_predictions = None
        self._last_global = None

    def _expert_predictions(self, x):
        return [expert.predict_one(x) for expert in self.experts]

    def predict_one(self, x):
        predictions = self._expert_predictions(x)
        votes = {}
        for prediction, weight in zip(predictions, self.weights):
            if prediction is not None:
                votes[prediction] = votes.get(prediction, 0.0) + weight
        global_prediction = max(votes, key=votes.get) if votes else None
        self._last_predictions = (x, predictions)
        self._last_global = global_prediction
        return global_prediction

    def learn_one(self, x, y):
        if self._last_predictions is not None and self._last_predictions[0] is x:
            predictions, global_prediction = self._last_predictions[1], self._last_global
        else:
            # 事前学習のように予測を経ない学習でも、同じ規則で重みを更新する。
            global_prediction = self.predict_one(x)
            predictions = self._last_predictions[1]
        if self.sample_index % self.period == 0:
            self.weights = [
                weight * self.beta if prediction != y else weight
                for prediction, weight in zip(predictions, self.weights)
            ]
            maximum_weight = max(self.weights)
            self.weights = [weight / maximum_weight for weight in self.weights]
            kept = [index for index, weight in enumerate(self.weights) if weight >= self.theta]
            self.experts = [self.experts[index] for index in kept]
            self.weights = [self.weights[index] for index in kept]
            if global_prediction != y:
                self.experts.append(self.make_expert())
                self.weights.append(1.0)
        for expert in self.experts:
            expert.learn_one(x, y)
        self.sample_index += 1
        self._last_predictions = None
        self._last_global = None


def make_model(method_name, random_seed):
    from river import ensemble, forest, tree

    if method_name == "arf":
        return forest.ARFClassifier(seed=random_seed)
    if method_name == "srp":
        return ensemble.SRPClassifier(seed=random_seed)
    if method_name == "leveraging_bagging":
        return ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), seed=random_seed
        )
    if method_name == "hoeffding_adaptive_tree":
        return tree.HoeffdingAdaptiveTreeClassifier(seed=random_seed)
    if method_name == "dwm":
        return DynamicWeightedMajority(make_expert=tree.HoeffdingTreeClassifier)
    raise ValueError(f"知らない手法です: {method_name}")


def array_digest(arrays):
    """書き出し側（stream_export.array_digest）と同じ定義の内容のhash。"""
    digest = sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _as_feature_dictionary(row):
    return {index: float(value) for index, value in enumerate(row)}


def evaluate_method_on_stream(*, method_name, arrays, random_seed):
    """事前学習の標本を学習してから、系列を予測→学習の順で処理し、各標本の正誤を返す。"""
    model = make_model(method_name, random_seed)
    for row, label in zip(arrays["pretraining_features"], arrays["pretraining_labels"]):
        model.learn_one(_as_feature_dictionary(row), int(label))
    correct = np.zeros(len(arrays["labels"]), dtype=bool)
    for index, (row, label) in enumerate(zip(arrays["features"], arrays["labels"])):
        x = _as_feature_dictionary(row)
        prediction = model.predict_one(x)
        correct[index] = prediction is not None and int(prediction) == int(label)
        model.learn_one(x, int(label))
    return correct


def _source_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _run_one(task):
    stream_path, method_name, output_directory = task
    stream_path = Path(stream_path)
    metadata = json.loads(stream_path.with_suffix(".json").read_text(encoding="utf-8"))
    stem = f"{stream_path.stem}_{method_name}"
    result_path = Path(output_directory) / "runs" / f"{stem}.json"
    if result_path.exists():
        return f"skip {stem}"
    with np.load(stream_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if array_digest(arrays) != metadata["array_sha256"]:
        raise RuntimeError(f"系列の内容が書き出しの記録と違います: {stream_path}")
    started = time.perf_counter()
    correct = evaluate_method_on_stream(
        method_name=method_name, arrays=arrays, random_seed=metadata["random_seed"]
    )
    elapsed = time.perf_counter() - started
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_path.with_suffix(".npz"), prediction_is_correct=correct)
    result = dict(
        result_format_version=_RESULT_FORMAT_VERSION,
        dataset_name=metadata["dataset_name"],
        random_seed=metadata["random_seed"],
        sample_count=metadata["sample_count"],
        method_name=method_name,
        prediction_accuracy=float(correct.mean()),
        elapsed_seconds=elapsed,
        stream_array_sha256=metadata["array_sha256"],
        # 書き出しの特徴の表現（記録の無い古い書き出しは生の特徴）。
        feature_representation=metadata.get("feature_representation", "raw"),
        river_baselines_source_sha256=_source_sha256(),
    )
    temporary = result_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return f"done {stem} {result['prediction_accuracy']:.4f} ({elapsed:.0f}s)"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, required=True)
    parser.add_argument("--workers", type=int, required=True)
    arguments = parser.parse_args(command_arguments)
    if arguments.workers < 1:
        raise ValueError("workersは正の整数にしてください")
    import river

    stream_paths = sorted(
        path for path in arguments.streams.glob("*.npz") if not path.name.endswith(".tmp.npz")
    )
    if not stream_paths:
        raise ValueError(f"系列がありません: {arguments.streams}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y%m%d_%H%M%S")
    (arguments.output / f"river_provenance_{started}.json").write_text(
        json.dumps(
            dict(
                river_version=river.__version__,
                numpy_version=np.__version__,
                python_version=platform.python_version(),
                platform=platform.platform(),
                methods=arguments.methods,
                streams=str(arguments.streams),
                river_baselines_source_sha256=_source_sha256(),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # 重いMNIST系を先に始める（並列の終わりの待ちを減らす）。
    tasks = sorted(
        (
            (str(path), method_name, str(arguments.output))
            for path in stream_paths
            for method_name in arguments.methods
        ),
        key=lambda task: (not Path(task[0]).name.startswith(("mnist", "fmnist", "rotmnist")), task),
    )
    failures = 0
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for index, (task, future) in enumerate(zip(tasks, futures), start=1):
            try:
                message = future.result()
            except Exception as error:  # 1つの失敗で全体を止めない。失敗は記録して数える。
                failures += 1
                message = f"failed {Path(task[0]).stem}_{task[1]}: {type(error).__name__}: {error}"
            print(f"[{index}/{len(tasks)}] {message}", flush=True)
    print(f"failed: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
