"""現行FedSDAの保存済みrunを照合し、構成要素の対比較を報告する。

入力はformat2 JSONと同じruns内のNPZ。NPZ内部は展開しない。
学習コード・結果・既定値を変更せず、採否は人が判断する。
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (  # noqa: E402
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep.experiment_configurations import (  # noqa: E402
    build_experiment_run_settings,
)

_CONFIGURATIONS = (
    "fedsda-without-candidate-validation",
    "fedsda-without-prediction-weight-recalibration",
    "fedsda-loss-distance-clustering",
    "fedsda-overall-loss-esr-detection",
    "fedsda-without-concept-specific-adapter",
    "fedsda-independent-models",
)
_DATASETS = ("sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4")
_CONDITION_FIELDS = (
    "dataset_name",
    "server_aggregation_interval_per_client_samples",
    "random_seed",
    "client_count",
    "per_client_sample_count",
)
_Key = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class _Metric:
    path: str
    label: str
    unit: str
    direction: int
    scale: float = 1
    per_sample: bool = False

    @property
    def _identifier(self) -> str:
        return self.path + ("/processed_sample_count" if self.per_sample else "")


_COST = "computation_cost_summary."
_DETECTION = "change_response_metrics.loss_change_alarm_detection_metrics."
_METRICS = (
    _Metric("prediction_accuracy", "全体精度", "% / 差pp", 1, 100),
    _Metric("stable_period_prediction_accuracy", "安定期精度", "% / 差pp", 1, 100),
    _Metric(
        "prediction_diagnostic_metrics.combined_prediction_period_accuracies.recovery_period_accuracy",
        "回復期精度",
        "% / 差pp",
        1,
        100,
    ),
    _Metric("communication_volume.uploaded_byte_count", "上り通信総量", "byte", -1),
    _Metric("communication_volume.downloaded_byte_count", "下り通信総量", "byte", -1),
    _Metric(
        "communication_volume.uploaded_byte_count",
        "標本当たり上り通信",
        "byte/sample",
        -1,
        per_sample=True,
    ),
    _Metric(
        "communication_volume.downloaded_byte_count",
        "標本当たり下り通信",
        "byte/sample",
        -1,
        per_sample=True,
    ),
    _Metric(_COST + "client_forward_multiply_accumulate_count", "client前向き計算総量", "MAC", -1),
    _Metric(
        _COST + "client_estimated_backward_multiply_accumulate_count",
        "client推定逆向き計算総量",
        "MAC",
        -1,
    ),
    _Metric(
        _COST + "client_forward_and_backward_multiply_accumulate_count_per_processed_sample",
        "標本当たりclient前向き+推定逆向き計算",
        "MAC/sample",
        -1,
    ),
    _Metric(_COST + "server_multiply_accumulate_count", "server計算総量", "MAC", -1),
    _Metric(_COST + "mean_held_model_count", "平均保有モデル数", "model", 0),
    _Metric("final_global_model_count", "最終globalモデル数", "model", 0),
    _Metric(_DETECTION + "detection_precision", "警報適合率", "% / 差pp", 1, 100),
    _Metric(_DETECTION + "detection_recall", "警報再現率", "% / 差pp", 1, 100),
    _Metric(_DETECTION + "detection_f1", "警報F1", "% / 差pp", 1, 100),
    _Metric(
        "change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count",
        "対応スイッチ平均遅延",
        "sample",
        -1,
    ),
)


@dataclass(frozen=True)
class _Run:
    path: Path
    key: _Key
    payload: dict
    settings_hash: str


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _expected_settings(configuration: str, key: _Key) -> dict:
    return convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name=configuration,
            experiment_run_conditions=ExperimentRunConditions(**dict(zip(_CONDITION_FIELDS, key))),
        )
    )


def _key_description(key: _Key) -> str:
    return f"{key[0]}, interval={key[1]}, seed={key[2]}, clients={key[3]}, samples/client={key[4]}"


def _validate_artifacts(path: Path, payload: dict, key: _Key) -> None:
    if type(payload.get("format_version")) is not int or payload["format_version"] != 2:
        raise ValueError("JSON形式はformat_version=2が必要")
    name = payload.get("record_table_file_name")
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or any(character in name for character in ("/", "\\", ":"))
        or Path(name).suffix != ".npz"
    ):
        raise ValueError("NPZ参照名が安全な単一ファイル名ではない")
    record_path = path.parent / name
    if record_path.resolve().parent != path.parent.resolve() or not record_path.is_file():
        raise ValueError("NPZが存在しない、またはruns外を参照")
    metrics = payload.get("run_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("JSONのrun_metricsが辞書ではない")
    count = metrics.get("processed_sample_count")
    if type(count) is not int or count != key[3] * key[4]:
        raise ValueError("処理標本数が予定のclient数×標本数と一致しない")
    provenance = payload.get("run_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("来歴がない")
    commit = provenance.get("code_commit")
    version = provenance.get("behavior_baseline_version")
    if (
        not isinstance(commit, str)
        or not commit.strip()
        or type(version) is not int
        or version < 1
        or provenance.get("has_uncommitted_changes") is not False
        or "code_revision_unavailable_reason" not in provenance
        or provenance["code_revision_unavailable_reason"] is not None
    ):
        raise ValueError("来歴が不正、dirty、または版が不明")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(name), str) or not environment[name].strip()
        for name in ("system", "machine", "python", "numpy", "torch")
    ):
        raise ValueError("環境の保存項目が不足または不正")
    _canonical(environment)


def _read_runs(
    directory: Path, configuration: str, expected: set[_Key], diagnostics: list[str], audit: list
) -> dict[_Key, _Run]:
    candidates: dict[_Key, list[_Run]] = {}
    for path in sorted((directory / "runs").glob("*.json")):
        try:
            if path.resolve().parent != (directory / "runs").resolve():
                raise ValueError("JSONがruns外を参照")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSONのrootが辞書ではない")
            # 除外した入力も来歴と環境を監査表へ残す。
            audit.append((path, payload))
            settings = payload["run_settings"]
            conditions = settings["execution_settings"]["experiment_run_conditions"]
            key = tuple(conditions[name] for name in _CONDITION_FIELDS)
            if type(key[0]) is not str or any(type(value) is not int for value in key[1:]):
                raise ValueError("設定不一致: 実行条件の型が不正")
            if key not in expected:
                diagnostics.append(f"格子外: {path}: {_key_description(key)}")
                continue
            serialized = _canonical(settings)
            if serialized != _canonical(_expected_settings(configuration, key)):
                diagnostics.append(f"設定不一致: {path}: {_key_description(key)}")
                continue
            run = _Run(path, key, payload, hashlib.sha256(serialized.encode()).hexdigest())
            candidates.setdefault(key, []).append(run)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, OverflowError) as error:
            diagnostics.append(f"JSON/設定の読取り不可: {path}: {error}")
    result = {}
    for key, runs in candidates.items():
        if len(runs) != 1:
            diagnostics.append(
                f"重複: {configuration}: {_key_description(key)}: {[str(run.path) for run in runs]}"
            )
            continue
        run = runs[0]
        try:
            _validate_artifacts(run.path, run.payload, key)
            result[key] = run
        except (OSError, ValueError, TypeError) as error:
            diagnostics.append(f"除外: {run.path}: {error}")
    return result


def _provenance_identity(run: _Run) -> tuple[str, int]:
    provenance = run.payload["run_provenance"]
    return provenance["code_commit"], provenance["behavior_baseline_version"]


def _metric_value(run: _Run, metric: _Metric) -> float | None:
    value = run.payload["run_metrics"]
    for name in metric.path.split("."):
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
    if type(value) not in (float, int):
        return None
    try:
        result = float(value) * metric.scale
        if metric.per_sample:
            result /= run.payload["run_metrics"]["processed_sample_count"]
        return result if math.isfinite(result) else None
    except (ValueError, OverflowError):
        return None


def _format(value: float | None) -> str:
    return "未定義" if value is None else f"{value:.6f}"


def _summarize(values: list[tuple[float, float]], direction: int) -> list[str]:
    if not values:
        return ["未定義"] * 8 + ["0/0/0"]
    baseline, variant = zip(*values)
    differences = [right - left for left, right in values]
    count = len(values)
    stats = [
        statistics.mean(baseline),
        statistics.stdev(baseline) if count > 1 else None,
        statistics.mean(variant),
        statistics.stdev(variant) if count > 1 else None,
        statistics.mean(differences),
        statistics.stdev(differences) / math.sqrt(count) if count > 1 else None,
        min(differences),
        max(differences),
    ]
    signs = [difference * (direction or 1) for difference in differences]
    counts = f"{sum(value > 0 for value in signs)}/{sum(value == 0 for value in signs)}/{sum(value < 0 for value in signs)}"
    return [*map(_format, stats), counts]


def _analysis_revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    result = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["has_uncommitted_changes"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["code_revision_unavailable"] = True
    return result


def _escaped(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _build_report(arguments, command: list[str], created: datetime) -> tuple[str, list[dict], bool]:
    expected = set(
        itertools.product(
            arguments.datasets,
            arguments.aggregation_intervals,
            arguments.seeds,
            [arguments.client_count],
            [arguments.per_client_sample_count],
        )
    )
    diagnostics: list[str] = []
    audit: list[tuple[Path, dict]] = []
    baseline = _read_runs(arguments.baseline_directory, "fedsda", expected, diagnostics, audit)
    rows = []
    sections = []
    total_pairs = 0
    for configuration in arguments.configurations:
        variants = _read_runs(
            arguments.ablation_directory / configuration,
            configuration,
            expected,
            diagnostics,
            audit,
        )
        pairs = {}
        for key in sorted(expected):
            if key not in baseline or key not in variants:
                missing = "/".join(
                    name
                    for name, found in (("基準", baseline), (configuration, variants))
                    if key not in found
                )
                diagnostics.append(f"対の不足: {missing}: {_key_description(key)}")
                continue
            left, right = baseline[key], variants[key]
            if _canonical(left.payload["environment"]) != _canonical(right.payload["environment"]):
                diagnostics.append(f"環境不一致: {configuration}: {_key_description(key)}")
                continue
            if _provenance_identity(left) != _provenance_identity(right):
                if not arguments.allow_provenance_difference:
                    diagnostics.append(
                        f"来歴差（明示許可なし）: {configuration}: {_key_description(key)}"
                    )
                    continue
                diagnostics.append(
                    f"来歴差を許可: {configuration}: {_key_description(key)}: {_provenance_identity(left)} → {_provenance_identity(right)}"
                )
            pairs[key] = (left, right)
        total_pairs += len(pairs)
        sections.append(f"## {configuration}\n\nrun対: {len(pairs)}/{len(expected)}\n")
        for dataset, interval in itertools.product(
            arguments.datasets, arguments.aggregation_intervals
        ):
            sections.extend(
                [
                    f"### {dataset} / interval={interval}\n",
                    "| metric | 指標（単位、改善方向） | 有効対/予定seed | 基準平均 | 基準SD | 変更平均 | 変更SD | 差平均 | 差SE | 差最小 | 差最大 | 改善/同値/悪化（優劣なしは増/同/減） |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|",
                ]
            )
            group = [
                (key, pair) for key, pair in sorted(pairs.items()) if key[:2] == (dataset, interval)
            ]
            for metric in _METRICS:
                values = []
                for key, (left, right) in group:
                    left_value, right_value = (
                        _metric_value(left, metric),
                        _metric_value(right, metric),
                    )
                    if left_value is None or right_value is None:
                        missing_side = "/".join(
                            side
                            for side, value in (("基準", left_value), ("変更", right_value))
                            if value is None
                        )
                        diagnostics.append(
                            f"指標欠測: {configuration}: {_key_description(key)}: {metric._identifier}: {missing_side}"
                        )
                        continue
                    values.append((left_value, right_value))
                    rows.append(
                        dict(
                            configuration=configuration,
                            dataset=dataset,
                            aggregation_interval=interval,
                            seed=key[2],
                            client_count=key[3],
                            per_client_sample_count=key[4],
                            metric=metric._identifier,
                            unit=metric.unit,
                            baseline=left_value,
                            variant=right_value,
                            delta=right_value - left_value,
                            baseline_path=str(left.path),
                            variant_path=str(right.path),
                        )
                    )
                direction = {1: "大", -1: "小", 0: "優劣なし"}[metric.direction]
                cells = [
                    metric._identifier,
                    f"{metric.label}（{metric.unit}、{direction}）",
                    f"{len(values)}/{len(arguments.seeds)}",
                    *_summarize(values, metric.direction),
                ]
                sections.append("| " + " | ".join(cells) + " |")
            sections.append("")
    expected_pairs = len(expected) * len(arguments.configurations)
    complete = total_pairs == expected_pairs
    lines = [
        "# FedSDA構成要素の比較\n",
        f"run格子: {'完全' if complete else '不足・比較不能あり'} / 有効対 {total_pairs}/{expected_pairs}。指標の欠測は別途表示。\n",
        "許容精度低下幅は未設定・自動採否なし。5seedの比較は探索的であり、非劣性・有意差・要素の不要性を認定しない。",
        "差は変更−基準。SDは標本標準偏差、SEは同seedの差の標準誤差。平均は指標の有効対だけで双方を揃えて計算。データセットをまたぐ総平均は出さない。",
        "回復期精度は区間精度であり回復時間ではない。MACは計算量の見積りで実測時間ではない。モデル数には優劣を付けない。",
        "独立モデル構成は共有部・学習方法・再較正の複合変更であり、共有部だけの効果を分離できない。",
        "NPZは存在・参照先だけを確認し内部を検証していない。runの欠落は失敗原因を表さず、過去のsweep失敗履歴は解析しない。",
        "環境は保存されたsystem/machine/python/numpy/torch等の一致だけを確認し、CPU型番・BLAS・backend設定までの同等性を保証しない。",
        f"来歴差の明示許可: {arguments.allow_provenance_difference}。許可は異なるコードの同等性を保証しない。\n",
        f"分析日時UTC: {created.isoformat()}",
        f"分析コード: `{_escaped(_canonical(_analysis_revision()))}`",
        f"コマンド（POSIX表記）: `{_escaped(shlex.join(command))}`",
        f"基準: `{_escaped(arguments.baseline_directory)}`",
        f"変更側: `{_escaped(arguments.ablation_directory)}`",
        f"格子: datasets={arguments.datasets}, intervals={arguments.aggregation_intervals}, seeds={arguments.seeds}, clients={arguments.client_count}, samples/client={arguments.per_client_sample_count}",
        "\n## 診断\n",
        f"診断件数: {len(diagnostics)}（指標欠測・格子外・来歴差許可を含む）",
        *["- " + _escaped(message) for message in diagnostics],
        "",
        *sections,
        "## 入力の監査表\n",
        "設定SHA256はJSONのキーをソートし空白を除いたUTF8のhash。除外入力も来歴を残す。",
        "| 入力JSON | 設定SHA256 | 来歴 | 環境 |",
        "|---|---|---|---|",
    ]
    for path, payload in audit:
        try:
            fingerprint = hashlib.sha256(
                _canonical(payload.get("run_settings")).encode()
            ).hexdigest()
        except (ValueError, TypeError, OverflowError):
            fingerprint = "不正"
        cells = [
            str(path),
            fingerprint,
            json.dumps(payload.get("run_provenance"), ensure_ascii=False),
            json.dumps(payload.get("environment"), ensure_ascii=False),
        ]
        lines.append("| " + " | ".join(map(_escaped, cells)) + " |")
    return "\n".join(lines) + "\n", rows, complete


def main(argv: list[str] | None = None) -> int:
    """引数を検査し、完全な対比較は0、不足は1、入力誤りはSystemExit(2)。

    stdoutでは入力を変更しない。通常出力も新規ディレクトリへの保存のみ。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", required=True, type=Path)
    parser.add_argument("--ablation-directory", required=True, type=Path)
    parser.add_argument(
        "--configurations", nargs="+", choices=_CONFIGURATIONS, default=list(_CONFIGURATIONS)
    )
    parser.add_argument("--datasets", nargs="+", choices=_DATASETS, default=list(_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--aggregation-intervals", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--per-client-sample-count", type=int, default=5000)
    parser.add_argument("--allow-provenance-difference", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    tokens = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(tokens)
    for name in ("baseline_directory", "ablation_directory"):
        directory = getattr(arguments, name).resolve()
        if not directory.is_dir():
            parser.error(f"入力ディレクトリが存在しない: {directory}")
        setattr(arguments, name, directory)
    if not (arguments.baseline_directory / "runs").is_dir():
        parser.error("基準ディレクトリにruns/が必要")
    for name in ("configurations", "datasets", "seeds", "aggregation_intervals"):
        values = getattr(arguments, name)
        if len(set(values)) != len(values):
            parser.error(f"格子引数の重複: {name}")
    if any(value < 0 for value in arguments.seeds) or any(
        value <= 0
        for value in [
            *arguments.aggregation_intervals,
            arguments.client_count,
            arguments.per_client_sample_count,
        ]
    ):
        parser.error("seedは0以上、間隔・client数・標本数は正整数が必要")
    created = datetime.now(UTC)
    report, rows, complete = _build_report(
        arguments, [sys.executable, str(Path(__file__).resolve()), *tokens], created
    )
    if arguments.stdout:
        sys.stdout.write(report)
    else:
        output = arguments.ablation_directory / ("analysis_" + created.strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(exist_ok=False)
        with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report)
        fields = (
            "configuration",
            "dataset",
            "aggregation_interval",
            "seed",
            "client_count",
            "per_client_sample_count",
            "metric",
            "unit",
            "baseline",
            "variant",
            "delta",
            "baseline_path",
            "variant_path",
        )
        with (output / "paired.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"analysis: {output}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
