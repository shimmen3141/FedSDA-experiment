"""モデルの混合の重みの規則を、再実験なしに手元で計算し直すための記録（研究ツール、製品のコードは変えない）。

1 run の記録（`ProbabilityTraceRecorder`）: clientごとに、予測に使うモデルの混合の重みの制御（clientの
`owners.fixed_share_weight_controller`）への呼出しの列を、呼ばれた順に残す。

- `weights`: 観測前の重み（`get_weights_before_label_observation`）。モデルの ID の組。
- `update`: 損失での更新（`update_weights_after_loss_observation`）。モデルの ID と損失 1 − p_y、渡された重み。
- `replay`・`replay_after_aggregation`: 損失の列の再生。`reset_after_aggregation`: 集約の後の消去。
- `predict_begin`・`predict`: 標本1件の予測（`predict_observed_sample_and_update_prediction_weights`）の始まりと終わり。
  終わりに、標本の番号、真のクラス、同期の版のモデル別の確率（float32 のまま）、使った Fixed Share の重みを残す。

呼出しの中から呼ばれる呼出し（再生の中の更新など）は記録しない（計算し直すときは、外側の呼出しを同じ制御へ渡せば、
中の呼出しも同じに起きる）。学習の責任度の重みの制御（別の実体）は記録しない。
記録は run ごとに `<変種>/prediction_probability_trace/<run>.pkl.xz`（LZMA で圧縮した pickle）へ書き、
失敗の無い掃引の後に `consolidate_probability_traces` で `<変種>/prediction_probability_trace.zip` へまとめる。
"""

import lzma
import os
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from federated_learning_experiments.methods.fedsda.prediction_combination import fixed_share_weights as _weights_module
from federated_learning_experiments.runtime import fedsda_run_client as _client_module
from federated_learning_experiments.runtime import observed_sample_processing as _processing_module

TRACE_DIRECTORY_NAME = "prediction_probability_trace"
TRACE_ARCHIVE_NAME = "prediction_probability_trace.zip"
TRACE_SUFFIX = ".pkl.xz"
_Controller = _weights_module.FixedShareWeightController
_RECORDED_METHODS = {
    "get_weights_before_label_observation": "weights",
    "update_weights_after_loss_observation": "update",
    "replay_observed_losses": "replay",
    "replay_observed_losses_after_aggregation": "replay_after_aggregation",
    "reset_weights_after_aggregation": "reset_after_aggregation",
}


def _loss_rows(observed_loss_sequence):
    return [tuple((int(model_id), float(loss)) for model_id, loss in row.items()) for row in observed_loss_sequence]


class ProbabilityTraceRecorder:
    """1 run の間だけクラスのメソッドを包んで、呼出しの列を残す。"""

    def __init__(self):
        self._events_by_controller = defaultdict(list)
        self._client_by_controller = {}
        self._depth = 0
        self._current_client_id = None
        self._settings = None
        self._originals = {}

    def _wrap_controller_method(self, name, kind, original):
        recorder = self

        def wrapper(controller, **arguments):
            if recorder._depth == 0:
                recorder._events_by_controller[id(controller)].append(recorder._encode(kind, arguments))
            recorder._depth += 1
            try:
                return original(controller, **arguments)
            finally:
                recorder._depth -= 1

        wrapper.__name__ = name
        return wrapper

    @staticmethod
    def _encode(kind, arguments):
        if kind == "weights":
            return ("weights", tuple(int(model_id) for model_id in arguments["model_ids"]))
        if kind == "update":
            return (
                "update",
                _loss_rows([arguments["observed_losses_by_model_id"]])[0],
                tuple((int(k), float(v)) for k, v in arguments["prediction_weights_by_model_id"].items()),
            )
        if kind in ("replay", "replay_after_aggregation"):
            return (kind, _loss_rows(arguments["observed_loss_sequence"]))
        return (kind,)

    def install(self):
        for name, kind in _RECORDED_METHODS.items():
            original = getattr(_Controller, name)
            self._originals[(_Controller, name)] = original
            setattr(_Controller, name, self._wrap_controller_method(name, kind, original))
        recorder = self
        original_predict = _processing_module.predict_observed_sample_and_update_prediction_weights
        self._originals[(_processing_module, "predict_observed_sample_and_update_prediction_weights")] = original_predict

        def predict(**arguments):
            controller = arguments["fixed_share_weight_controller"]
            if recorder._depth != 0:
                raise RuntimeError("予測が重みの制御の呼出しの中から呼ばれました（想定していない順序）")
            key = id(controller)
            client_id = recorder._client_by_controller.setdefault(key, recorder._current_client_id)
            if client_id != recorder._current_client_id or client_id is None:
                raise RuntimeError("予測の重みの制御とclientの対応が変わりました")
            recorder._settings = controller._prediction_combination_settings
            recorder._events_by_controller[key].append(("predict_begin",))
            result = original_predict(**arguments)
            observation = arguments["indexed_observation"]
            probabilities = result.prediction_probabilities_by_model_id
            model_ids = tuple(sorted(probabilities))
            recorder._events_by_controller[key].append(
                (
                    "predict",
                    int(observation.sample_index),
                    int(observation.training_sample.observed_class_labels.view(-1)[0].item()),
                    model_ids,
                    np.stack([probabilities[model_id].detach().cpu().numpy() for model_id in model_ids]),
                    tuple(float(result.fixed_share_weights_by_model_id[model_id]) for model_id in model_ids),
                )
            )
            return result

        _processing_module.predict_observed_sample_and_update_prediction_weights = predict
        original_process = _client_module.FedsdaRunClient.process_observed_sample
        self._originals[(_client_module.FedsdaRunClient, "process_observed_sample")] = original_process

        def process_observed_sample(client, *positional_arguments, **arguments):
            previous = recorder._current_client_id
            recorder._current_client_id = client.client_id
            try:
                return original_process(client, *positional_arguments, **arguments)
            finally:
                recorder._current_client_id = previous

        _client_module.FedsdaRunClient.process_observed_sample = process_observed_sample

    def uninstall(self):
        for (owner, name), original in reversed(list(self._originals.items())):
            setattr(owner, name, original)
        self._originals = {}

    def write(self, output_directory, run_identifier, variant):
        clients = {
            client_id: self._events_by_controller[key] for key, client_id in sorted(self._client_by_controller.items(), key=lambda item: item[1])
        }
        if len(set(clients)) != len(self._client_by_controller):
            raise RuntimeError("1つのclientに重みの制御が2つ以上ありました")
        directory = Path(output_directory) / TRACE_DIRECTORY_NAME
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(run_identifier=run_identifier, variant=variant, settings=self._settings, clients=clients)
        path = directory / f"{run_identifier}{TRACE_SUFFIX}"
        temporary = path.with_name(path.name + ".tmp")
        with lzma.open(temporary, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)


def consolidate_probability_traces(variant_directory):
    """runごとの記録を1つの zip（圧縮しない格納。中身は LZMA 済み）へまとめ、同じ中身を読み返せたら元を消す。"""
    variant_directory = Path(variant_directory)
    directory = variant_directory / TRACE_DIRECTORY_NAME
    if not directory.is_dir():
        return
    paths = sorted(directory.glob(f"*{TRACE_SUFFIX}"))
    archive = variant_directory / TRACE_ARCHIVE_NAME
    existing = {}
    if archive.exists():
        with zipfile.ZipFile(archive) as packed:
            existing = {name: packed.read(name) for name in packed.namelist()}
    contents = dict(existing)
    for path in paths:
        contents[path.name] = path.read_bytes()
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as packed:
        for name in sorted(contents):
            packed.writestr(name, contents[name])
    with zipfile.ZipFile(temporary) as packed:
        if packed.testzip() is not None or {name: packed.read(name) for name in packed.namelist()} != contents:
            raise RuntimeError(f"まとめた記録を読み返せません: {temporary}")
    os.replace(temporary, archive)
    for path in paths:
        path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()


def iterate_probability_traces(variant_directory):
    """(run の識別子, 記録) を run の識別子の順に返す。まとめた zip と、まとめる前のファイルの両方を読む。"""
    variant_directory = Path(variant_directory)
    blobs = {}
    archive = variant_directory / TRACE_ARCHIVE_NAME
    if archive.exists():
        with zipfile.ZipFile(archive) as packed:
            for name in packed.namelist():
                blobs[name] = ("zip", name)
    directory = variant_directory / TRACE_DIRECTORY_NAME
    if directory.is_dir():
        for path in directory.glob(f"*{TRACE_SUFFIX}"):
            blobs[path.name] = ("file", path)
    for name in sorted(blobs):
        source, location = blobs[name]
        if source == "zip":
            with zipfile.ZipFile(archive) as packed:
                data = packed.read(location)
        else:
            data = location.read_bytes()
        payload = pickle.loads(lzma.decompress(data))
        yield name[: -len(TRACE_SUFFIX)], payload


__all__ = [
    "ProbabilityTraceRecorder",
    "consolidate_probability_traces",
    "iterate_probability_traces",
    "TRACE_ARCHIVE_NAME",
    "TRACE_DIRECTORY_NAME",
]
