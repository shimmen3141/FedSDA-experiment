"""検出器を外し、帰属の保留（FIFO）の列で毎標本MDLの選択をする木の版（案A）の研究ツール。

木の版（`single_stream_hard_assignment_soft_prediction`）は、損失の監視（e-detector）の合図があったときだけ、保留の
列で変化点と帰属先（留まる・保有モデルの再利用・新しいモデル）をMDLで選ぶ。案Aでは監視を使わず、ウォームアップ
（既定200件）の後は毎標本、同じ選択をする。
- 「留まる」（現在のモデル）が選ばれたら、警報としない。保留を保ち、容量を超えた最古の標本だけを現在のモデルへ
  割り当てる（木の版の警報の無い標本と同じ）。記録もしない。
- 「保有モデル」「新しいモデル」が選ばれたときの規則（--decision）:
  - `any-change`: min(保留の件数, 変化区間) が最小の変化区間（5）以上なら、木の版の警報と同じ処理（前の n−k 件を
    現在のモデルへ、後の k 件を帰属先へ、現在のモデルを替え、保留を空にする）をする。5未満なら待つ（`too_short`）。
    毎標本、重なった窓で選択をやり直すので、切り替えすぎる（2026-11-19の手元の確認でMNIST4が5,000標本で485回）。
  - `full-window`（固定遅延）: 保留が満杯（容量＋1件）で、変化区間が窓全体（k＝窓の件数）のときだけ切り替え、
    窓全体を帰属先へ割り当てる。ほかは留まる（最古の標本だけを現在のモデルへ）。保留を出る標本についてだけ決める
    規則で、変化の後の標本が窓を埋めた時点で切り替わる（学習の割り当ては約30件遅れ、予測はFixed-Shareで切り替わる）。
予測（Fixed-Shareの混合）・学習（固い割り当て、1件1回）・MDLの計算は木の版の部品をそのまま使う。
--grace-period・--deltaを付けると、概念モデルの木をその値にする（`single_stream_tree_hyperparameters`と同じ差し替え）。
成果物は木の版のコマンドと同じ形式（`runs/*.json`・`.npz`と`run_summary.csv`）で、出力先に`detector_free.json`を書く。
`summary.alarm_count`は「留まる」以外が選ばれて切り替えた回数。

使い方（WSLの~/river-env、worktreeのルートで）:
    PYTHONPATH=src python tools/research_single_stream_detector_free/run.py --streams <系列> --output <出力先> --workers 8 \\
        [--grace-period 50 --delta 0.01]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(_REPOSITORY_ROOT))


class _AlwaysSignalAfterWarmUp:
    """監視の部品の代わり。ウォームアップの後は毎標本「合図あり」を返す（e値は記録しない）。"""

    def __init__(self, processor):
        self._processor = processor

    def observe_loss_after_label_observation(self, **_):
        from types import SimpleNamespace

        return SimpleNamespace(
            log_e_value=float("nan"),
            drift_detected=self._processor.processed_sample_count >= self._processor.settings.warm_up_sample_count,
        )

    def reset(self, *, baseline_loss_mean):
        del baseline_loss_mean


def make_detector_free_processor_class(decision="any-change"):
    if decision not in ("any-change", "full-window"):
        raise ValueError(f"知らない規則です: {decision}")
    from single_stream_hard_assignment_soft_prediction.alarm_interval_assignment import (
        select_alarm_interval_assignment,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_processing import (
        AlarmRecord,
        SingleStreamHardAssignmentSoftPredictionProcessor,
    )

    class DetectorFreeProcessor(SingleStreamHardAssignmentSoftPredictionProcessor):
        def __init__(self, **arguments):
            super().__init__(**arguments)
            # 監視の部品を最初から差し替える（ウォームアップの終わりに作る処理は、Noneでないので走らない）。
            self.loss_change_monitor = _AlwaysSignalAfterWarmUp(self)

        def _handle_detector_signal(self, *, sample_index):
            settings = self.settings
            window = tuple((features, class_id) for _, features, class_id, _ in self._pending)
            selection = select_alarm_interval_assignment(
                pending_samples=window,
                held_models=tuple(self.held_models),
                current_model_id=self.current_model_id,
                model_ids_with_loss_statistics=frozenset(
                    model_id
                    for model_id, _ in self.held_models
                    if self.loss_statistics_store.get_model_loss_statistics(model_id=model_id) is not None
                ),
                switching_probability=settings.switching_probability,
                minimum_change_interval_sample_count=settings.minimum_change_interval_sample_count,
            )
            if selection.selected_hypothesis == "current_model":
                return False
            k = selection.change_interval_sample_count
            if decision == "full-window" and not (
                len(window) == settings.pending_assignment_buffer_capacity_samples + 1 and k == len(window)
            ):
                return False
            if min(len(window), k) < settings.minimum_change_interval_sample_count:
                self.alarm_records.append(
                    AlarmRecord(
                        sample_index=sample_index,
                        selected_hypothesis=selection.selected_hypothesis,
                        change_interval_sample_count=k,
                        window_sample_count=len(window),
                        destination_model_id=self.current_model_id,
                        new_model_learner_name=selection.new_model_learner_name,
                        description_length=selection.description_length,
                        alarm_raised=False,
                        adaptation_outcome="too_short",
                        window_tail_concept_ids=tuple(c for _, _, _, c in self._pending[len(window) - k :]),
                    )
                )
                return False
            # 切り替える。木の版の警報の処理は選択を計算し直すので、同じ結果を返す関数に差し替えて呼ぶ。
            from single_stream_hard_assignment_soft_prediction import single_stream_processing

            original = single_stream_processing.select_alarm_interval_assignment
            single_stream_processing.select_alarm_interval_assignment = lambda **_: selection
            try:
                return super()._handle_detector_signal(sample_index=sample_index)
            finally:
                single_stream_processing.select_alarm_interval_assignment = original

    return DetectorFreeProcessor


def _install(grace_period, delta, decision):
    from single_stream_hard_assignment_soft_prediction import single_stream_run

    single_stream_run.SingleStreamHardAssignmentSoftPredictionProcessor = make_detector_free_processor_class(decision)
    if grace_period is not None:
        from tools.research_single_client_stream_baselines.single_stream_tree_hyperparameters import _install_tree

        _install_tree(grace_period, delta)


def _run_one(task):
    from single_stream_hard_assignment_soft_prediction.single_stream_artifacts import save_single_stream_run
    from single_stream_hard_assignment_soft_prediction.single_stream_input import load_single_stream_input
    from single_stream_hard_assignment_soft_prediction.single_stream_run import run_single_stream
    from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
        build_default_single_stream_settings,
    )

    stream_path, output_directory = Path(task[0]), Path(task[1])
    runs_directory = output_directory / "runs"
    json_path, npz_path = runs_directory / f"{stream_path.stem}.json", runs_directory / f"{stream_path.stem}.npz"
    if json_path.exists() and npz_path.exists():
        return f"skip {stream_path.stem}"
    if json_path.exists() or npz_path.exists():
        raise RuntimeError(f"成果物が片方だけあります: {stream_path.stem}")
    stream = load_single_stream_input(stream_path=stream_path)
    settings = build_default_single_stream_settings()
    result = run_single_stream(stream=stream, settings=settings)
    save_single_stream_run(
        result=result, settings=settings, stream=stream, output_directory=runs_directory, file_stem=stream_path.stem
    )
    return f"done {stream_path.stem} {result.summary['prequential_accuracy']:.4f}"


def main(command_arguments=None):
    from single_stream_hard_assignment_soft_prediction.single_stream_command import (
        select_stream_paths,
        write_run_summary_table,
    )

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--grace-period", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--decision", choices=("any-change", "full-window"), required=True)
    arguments = parser.parse_args(command_arguments)
    if (arguments.grace_period is None) != (arguments.delta is None):
        parser.error("--grace-periodと--deltaは両方付けるか、両方付けないでください。")
    paths = select_stream_paths(streams=arguments.streams, datasets=arguments.datasets, seeds=arguments.seeds)
    if not paths:
        raise SystemExit("処理する系列がありません。")
    arguments.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        method="detector_free_fifo_mdl_every_sample",
        decision=arguments.decision,
        grace_period=arguments.grace_period,
        delta=arguments.delta,
        tool_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    record_path = arguments.output / "detector_free.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if (previous["grace_period"], previous["delta"], previous.get("decision")) != (
            arguments.grace_period,
            arguments.delta,
            arguments.decision,
        ):
            raise SystemExit("出力先に別の設定の結果があります。")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(
        max_workers=arguments.workers, initializer=_install,
        initargs=(arguments.grace_period, arguments.delta, arguments.decision),
    ) as executor:
        for message in executor.map(_run_one, [(str(p), str(arguments.output)) for p in paths]):
            print(message, flush=True)
    print(f"wrote {write_run_summary_table(output_directory=arguments.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
