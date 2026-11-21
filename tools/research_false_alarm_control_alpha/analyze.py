"""αの変種と既定・FedDriftを、同じデータ×間隔×seedで対にして集計し、`analysis_false_alarm_control_alpha.md`へ保存する。

精度は全体・変化の直後（各clientの真の変化からI件）・安定した期間（それ以外）。標準指標（上下通信のバイト数、
clientの順伝播＋推定逆伝播のMAC、runの経過時間）と、警報の数・新しいモデルの数を併記する。
runのNPZは`runs/`か、まとめた`runs.zip`（`tools.research_results_archive`）から読む。
経過時間は、各出力の実行ログの`[i/n] fedsda/<run>: completed (Ns)`から読む（ログが無ければ未測定）。
"""

import argparse
import csv
import io
import re
import statistics
import zipfile
from pathlib import Path

import numpy as np

from tools.research_post_change_accuracy_breakdown.analyze import post_change_mask

csv.field_size_limit(10**9)
_CORRECT_NAMES = (
    "sample_prediction.combined_prediction_is_correct",
    "sample_prediction.prediction_is_correct",
)
_ALARMS = "change_response_metrics.loss_change_alarm_detection_metrics.detection_count"
_NEW_MODELS = "change_response_metrics.adaptation_action_metrics.adopt_candidate_model_count"
_UP = "communication_volume.uploaded_byte_count"
_DOWN = "communication_volume.downloaded_byte_count"
_MAC = (
    "computation_cost_summary.client_forward_multiply_accumulate_count",
    "computation_cost_summary.client_estimated_backward_multiply_accumulate_count",
)


def _elapsed_seconds(log_paths):
    seconds = {}
    for path in log_paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = re.search(r"\] \w[\w-]*/(\S+): completed \((\d+)s\)", line)
                if match:
                    seconds[match.group(1)] = int(match.group(2))
    return seconds


def read_configuration(directory, log_paths, row_filter=None):
    """構成のフォルダ（`run_summary.csv`がある）の各runを、(データ, 間隔, seed)ごとの値にする。

    `row_filter`は(列の名前, 値の文字列)で、その列がその値のrunだけを読む（例: FIFOの容量で分ける）。
    """
    archive_path = directory / "runs.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    elapsed = _elapsed_seconds(log_paths)
    rows = {}
    with open(directory / "run_summary.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row_filter and row.get(row_filter[0]) != row_filter[1]:
                continue
            name = f"runs/{row['run_identifier']}.npz"
            data = archive.read(name) if archive else (directory / name).read_bytes()
            interval = int(row["server_aggregation_interval_per_client_samples"])
            with np.load(io.BytesIO(data)) as arrays:
                correct_name = next(n for n in _CORRECT_NAMES if n in arrays.files)
                correct = arrays[correct_name].astype(bool)
                mask = post_change_mask(arrays["sample_prediction.observed_concept_id"], interval)
            if abs(correct.mean() - float(row["prediction_accuracy"])) > 1e-9:
                raise ValueError(f"正誤の配列と記録の精度が合いません: {row['run_identifier']}")

            def number(key):
                value = row.get(key, "")
                return float(value) if value not in ("", None) else None

            mac = [number(key) for key in _MAC]
            rows[(row["dataset_name"], interval, int(row["random_seed"]))] = dict(
                overall=100 * correct.mean(),
                post=100 * correct[mask].mean(),
                stable=100 * correct[~mask].mean(),
                alarms=number(_ALARMS),
                new_models=number(_NEW_MODELS),
                up=number(_UP),
                down=number(_DOWN),
                mac=None if None in mac else sum(mac),
                seconds=elapsed.get(row["run_identifier"]),
            )
    return rows


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _format(value, digits=2):
    return "未測定" if value is None else f"{value:.{digits}f}"


def analyze(sources, baseline_name, keys):
    lines = []
    cells = sorted({(d, i) for d, i, _ in keys})
    seeds = sorted({s for _, _, s in keys})
    lines.append(f"seed {seeds[0]}〜{seeds[-1]}の平均。精度は%、差はpp（括弧は既定を上回ったseedの数）。")
    lines.append("")
    lines.append("| データ | 間隔 | 構成 | 全体 | 直後 | 安定 | 警報 | 新しいモデル | 上り MB | 下り MB | client MAC (1e12) | 経過 秒 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for dataset, interval in cells:
        for name, rows in sources.items():
            values = [rows[(dataset, interval, s)] for s in seeds]
            mean = {k: _mean([v[k] for v in values]) for k in values[0]}
            lines.append(
                f"| {dataset} | {interval} | {name} | {_format(mean['overall'])} | {_format(mean['post'])} | "
                f"{_format(mean['stable'])} | {_format(mean['alarms'], 0)} | {_format(mean['new_models'], 0)} | "
                f"{_format(None if mean['up'] is None else mean['up'] / 1e6, 1)} | "
                f"{_format(None if mean['down'] is None else mean['down'] / 1e6, 1)} | "
                f"{_format(None if mean['mac'] is None else mean['mac'] / 1e12, 3)} | {_format(mean['seconds'], 0)} |"
            )
    lines.append("")
    lines.append(f"既定（{baseline_name}）との差（pp、seedで対）")
    lines.append("")
    lines.append("| データ | 間隔 | 構成 | 全体 | 直後 | 安定 |")
    lines.append("|---|---|---|---|---|---|")
    baseline = sources[baseline_name]
    for dataset, interval in cells:
        for name, rows in sources.items():
            if name == baseline_name:
                continue
            parts = []
            for key in ("overall", "post", "stable"):
                differences = [
                    rows[(dataset, interval, s)][key] - baseline[(dataset, interval, s)][key]
                    for s in seeds
                ]
                parts.append(
                    f"{statistics.mean(differences):+.2f}（{sum(d > 0 for d in differences)}/{len(seeds)}）"
                )
            lines.append(f"| {dataset} | {interval} | {name} | " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True, help="この研究ツールの出力先（変種のフォルダの親）")
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "DIRECTORY", "COLUMN=VALUE"),
        help="ほかの構成のフォルダ。COLUMN=VALUEでrunを絞る（絞らないときは-）。ログは<DIRECTORY>/../run.log",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="既定の構成のフォルダ（fedsda）")
    parser.add_argument("--baseline-log", type=Path, nargs="*", default=[])
    parser.add_argument("--feddrift", type=Path, help="FedDriftの構成のフォルダ（任意）")
    parser.add_argument("--feddrift-log", type=Path, nargs="*", default=[])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-name", default="analysis_false_alarm_control_alpha.md")
    arguments = parser.parse_args(command_arguments)
    keys = [(d, i, s) for d in arguments.datasets for i in arguments.intervals for s in arguments.seeds]
    sources = {"既定（α=0.001）": read_configuration(arguments.baseline, arguments.baseline_log)}
    for variant in arguments.variants:
        sources[variant] = read_configuration(
            arguments.root / variant / "fedsda", [arguments.root / "run.log"]
        )
    for name, directory, condition in arguments.source:
        directory = Path(directory)
        sources[name] = read_configuration(
            directory,
            [directory.parent / "run.log"],
            None if condition == "-" else tuple(condition.split("=", 1)),
        )
    if arguments.feddrift:
        sources["FedDrift"] = read_configuration(arguments.feddrift, arguments.feddrift_log)
    for name, rows in sources.items():
        missing = [key for key in keys if key not in rows]
        if missing:
            raise ValueError(f"{name}に無い条件があります: {missing[:3]}")
    text = analyze(sources, "既定（α=0.001）", keys)
    (arguments.root / arguments.output_name).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
