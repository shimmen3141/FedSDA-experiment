"""保存済み4構成を照合して、seedを単位とする対の差を集計する。"""

import argparse
import csv
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from federated_learning_experiments.core.run_provenance import read_run_provenance
from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
    load_fedsda_run_artifacts,
)

_NEW = "fedsda-soft-training-single-model-prediction"
_DATASETS = (
    "sine2",
    "circle2",
    "sea2",
    "sea4",
    "mnist2",
    "mnist4",
    "fmnist2",
    "fmnist4",
    "rotmnist2",
    "rotmnist4",
)
_NAMES = {
    "Ssingle": _NEW,
    "Smix": "fedsda-soft-training",
    "Hsingle": "fedsda-single-model-prediction",
    "Hmix": "fedsda",
}
_AXES = {
    _NEW: (
        "responsibility_weighted_training",
        True,
        "independent_fixed_share_weight_updates",
        "current_training_model_prediction",
        "current_local_parameters",
        "keep_weights_without_recalibration",
    ),
    "fedsda-soft-training": (
        "responsibility_weighted_training",
        True,
        "prediction_weight_updates",
        "fixed_share_weighted_prediction",
        "mixture_of_synchronized_and_local_parameters",
        "recompute_buffer_losses_and_replay_weight_updates",
    ),
    "fedsda-single-model-prediction": (
        "single_model_assignment",
        False,
        "prediction_weight_updates",
        "current_training_model_prediction",
        "current_local_parameters",
        "keep_weights_without_recalibration",
    ),
    "fedsda": (
        "single_model_assignment",
        False,
        "prediction_weight_updates",
        "fixed_share_weighted_prediction",
        "mixture_of_synchronized_and_local_parameters",
        "recompute_buffer_losses_and_replay_weight_updates",
    ),
}
_AXES["fedsda-training-mass-diagnostics"] = ("single_model_assignment", True, *_AXES["fedsda"][2:])
# 版15の別名は分析だけに適用し、成果物やrun名を書き換えない。
_ALIASES = {"fedsda-every-sample-change-interval-search": "fedsda"}


def _flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        if key == "settings_type":
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def _normalized_settings(settings, configuration):
    result = deepcopy(settings)
    client = result["run_participant_settings"]["run_client_settings"]
    prediction = client["prediction_combination_settings"]
    fields = (
        "training_assignment_strategy",
        "record_training_mass_by_concept",
        "training_responsibility_weight_update_policy",
    )
    defaults = ("single_model_assignment", False, "prediction_weight_updates")
    prediction_fields = (
        "prediction_combination_strategy",
        "prediction_model_parameter_source",
        "prediction_weight_recalibration_after_aggregation_policy",
    )
    observed = tuple(client.get(key, default) for key, default in zip(fields, defaults)) + tuple(
        prediction[key] for key in prediction_fields
    )
    if observed != _AXES[configuration]:
        raise ValueError(f"{configuration}: 許可された軸の値と異なる: {observed}")
    for key in fields:
        client.pop(key, None)
    for key in prediction_fields:
        prediction.pop(key)
    return result


def _check_settings(new, old, configuration):
    if _normalized_settings(new, _NEW) != _normalized_settings(
        old, _ALIASES.get(configuration, configuration)
    ):
        raise ValueError("保存設定に許可表以外の差がある")


def _check_inputs(left, right):
    for required in (
        "sample_prediction.observed_class_id",
        "sample_prediction.observed_concept_id",
    ):
        if required not in left or required not in right:
            raise ValueError(f"入力系列の必須列が無い: {required}")
    for column in (
        "sample_prediction.client_id",
        "sample_prediction.observed_class_id",
        "sample_prediction.observed_concept_id",
        "sample_prediction.sample_index",
        "input.feature_digest",
    ):
        if (column in left) != (column in right):
            raise ValueError(f"入力系列の保存有無が違う: {column}")
        if column in left and (
            left[column].dtype != right[column].dtype
            or not np.array_equal(left[column], right[column], equal_nan=False)
        ):
            raise ValueError(f"入力系列が違う: {column}")
    missing = [
        column
        for column in (
            "sample_prediction.client_id",
            "sample_prediction.sample_index",
            "input.feature_digest",
        )
        if column not in left
    ]
    return f"入力の直接照合は未検証（未保存: {', '.join(missing)}）" if missing else ""


def _paired_summary(values, *, accuracy=False):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("10 seedの有限な対の値が必要")
    values = values * (100 if accuracy else 1)
    return dict(
        mean=float(values.mean()),
        half_width_95=float(2.2621571628540993 * values.std(ddof=1) / np.sqrt(10)),
        positive_seeds=int((values > 0).sum()),
        zero_seeds=int((values == 0).sum()),
        negative_seeds=int((values < 0).sum()),
    )


def _metrics(summary, tables):
    flat = _flatten(summary["run_metrics"])
    metrics = {
        key: value
        for key, value in flat.items()
        if isinstance(value, (int, float)) or value is None
    }
    communication = summary["run_metrics"].get("communication_volume", {})
    up, down = communication.get("uploaded_byte_count"), communication.get("downloaded_byte_count")
    metrics.update(
        communication_upload_bytes=up,
        communication_download_bytes=down,
        communication_total_bytes=None if up is None or down is None else up + down,
        communication_total_MiB=None if up is None or down is None else (up + down) / 1024**2,
        peak_memory_bytes=None,
        elapsed_seconds=summary.get("_elapsed_seconds"),
        training_mass_purity=None,
        training_mass_single_model_baseline=None,
        training_mass_missing_clients=None,
        final_held_model_count=None,
    )
    if "client_held_model.client_id" in tables:
        client_count = summary["run_settings"]["execution_settings"]["experiment_run_conditions"][
            "client_count"
        ]
        counts = np.bincount(
            tables["client_held_model.client_id"].astype(np.int64), minlength=client_count
        )
        metrics["final_held_model_count"] = float(counts.mean())
    columns = ("client_id", "model_id", "observed_concept_id", "training_mass")
    if all(f"training_mass_by_concept.{c}" in tables for c in columns):
        from federated_learning_experiments.learning.training.responsibility_weighted_training import (
            derive_run_training_mass_purity,
        )

        masses = [
            {}
            for _ in range(
                summary["run_settings"]["execution_settings"]["experiment_run_conditions"][
                    "client_count"
                ]
            )
        ]
        for client, model, concept, mass in zip(
            *(tables[f"training_mass_by_concept.{c}"] for c in columns)
        ):
            masses[int(client)].setdefault(int(model), {})[int(concept)] = float(mass)
        purity, baseline, missing = derive_run_training_mass_purity(client_masses=tuple(masses))
        metrics.update(
            training_mass_purity=purity,
            training_mass_single_model_baseline=baseline,
            training_mass_missing_clients=missing,
        )
    for key in (
        "training_responsibility.effective_model_count",
        "soft_training_update.participating_model_count",
        "soft_training_batch.duplicate_sample_fraction",
    ):
        metrics[key] = float(tables[key].mean()) if key in tables and tables[key].size else None
    return metrics


def _key(summary):
    c = summary["run_settings"]["execution_settings"]["experiment_run_conditions"]
    return (
        c["dataset_name"],
        c["server_aggregation_interval_per_client_samples"],
        c["random_seed"],
        c["client_count"],
        c["per_client_sample_count"],
    )


def _identifier(configuration, settings):
    c = settings["execution_settings"]["experiment_run_conditions"]
    text = json.dumps(
        dict(configuration_name=configuration, run_settings=settings),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"{c['dataset_name']}_interval{c['server_aggregation_interval_per_client_samples']}_seed{c['random_seed']}_{hashlib.sha256(text.encode('ascii')).hexdigest()[:12]}"


def _read_group(roots, role, expected, checks):
    runs = {}
    expected_configuration = (
        "fedsda-training-mass-diagnostics" if role == "Diagnostic" else _NAMES[role]
    )
    for root in roots:
        if _ALIASES.get(root.name, root.name) != expected_configuration:
            checks.append(
                dict(role=role, path=str(root), status="error", detail="構成ルートが違う")
            )
            continue
        # ルートは構成ディレクトリ。JSONと同じ構成のsummaryを相互照合する。
        summary_path = root / "run_summary.csv"
        if not summary_path.is_file():
            checks.append(
                dict(role=role, path=str(root), status="error", detail="run_summary.csvが無い")
            )
            continue
        try:
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, ValueError, csv.Error) as error:
            checks.append(
                dict(role=role, path=str(summary_path), status="error", detail=str(error))
            )
            continue
        by_id = {}
        for row in rows:
            configuration = row.get("configuration_name")
            if _ALIASES.get(configuration, configuration) != expected_configuration:
                checks.append(
                    dict(
                        role=role,
                        path=str(summary_path),
                        status="error",
                        detail=f"構成が違う: {configuration}",
                    )
                )
                continue
            identifier = row.get("run_identifier")
            if not identifier:
                checks.append(
                    dict(
                        role=role,
                        path=str(summary_path),
                        status="error",
                        detail="summaryのrun識別子が無い",
                    )
                )
                continue
            if identifier in by_id:
                checks.append(
                    dict(
                        role=role,
                        path=str(summary_path),
                        status="error",
                        detail=f"summary重複: {identifier}",
                    )
                )
            by_id[identifier] = row
        # 成功状態は掃引の記録から読む。成果物があることだけでは成功とはしない。
        outcomes = {}
        for record_path in sorted(root.parent.glob("sweep_record*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                for outcome in record["run_outcomes"]:
                    outcomes[(outcome["configuration_name"], outcome["run_identifier"])] = outcome[
                        "run_status"
                    ]
            except (OSError, ValueError, KeyError, TypeError) as error:
                checks.append(
                    dict(role=role, path=str(record_path), status="error", detail=str(error))
                )
        for path in sorted(root.rglob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if "run_settings" not in raw:
                    continue
                row = by_id.get(path.stem)
                if row is None:
                    raise ValueError("summaryに対応する対象構成の行が無い")
                configuration = row["configuration_name"]
                canonical = _ALIASES.get(configuration, configuration)
                _normalized_settings(raw["run_settings"], canonical)
                key = _key(raw)
                if key not in expected:
                    # 既存の間隔100・200だけを明示的に対象外とする。
                    if (
                        key[1] in (100, 200)
                        and key[0] in _DATASETS
                        and 0 <= key[2] <= 9
                        and key[3:] == (10, 5000)
                    ):
                        continue
                    raise ValueError(f"対象外の実行条件: {key}")
                summary, tables = load_fedsda_run_artifacts(summary_file_path=path)
                if outcomes.get((configuration, path.stem)) not in (
                    "completed",
                    "skipped_existing",
                ):
                    raise ValueError("掃引の成功記録が無い・未成功")
                if path.stem != _identifier(configuration, summary["run_settings"]):
                    raise ValueError("run識別子と完全設定が不一致")
                provenance = read_run_provenance(summary.get("run_provenance"))
                if (
                    provenance.code_commit is None
                    or not re.fullmatch(r"[0-9a-f]{40}", provenance.code_commit)
                    or provenance.has_uncommitted_changes
                ):
                    raise ValueError("実装の来歴が不明または未コミット")
                if provenance.behavior_baseline_version != 16 and not (
                    configuration in _ALIASES and provenance.behavior_baseline_version == 15
                ):
                    raise ValueError("基準版の来歴が契約外")
                if not isinstance(summary.get("environment"), dict) or any(
                    not summary["environment"].get(k)
                    for k in ("system", "machine", "python", "numpy", "torch")
                ):
                    raise ValueError("実行環境が不明")
                conditions = summary["run_settings"]["execution_settings"][
                    "experiment_run_conditions"
                ]
                expected_row = {
                    **conditions,
                    **_flatten(summary["run_metrics"]),
                    **summary["run_provenance"],
                }
                for column, value in expected_row.items():
                    if column in ("settings_type", "code_revision_unavailable_reason"):
                        continue
                    if column not in row:
                        raise ValueError(f"summaryに列が無い: {column}")
                    text = "" if value is None else str(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not row[column] or float(row[column]) != float(value):
                            raise ValueError(f"summaryとJSONの不一致: {column}")
                    elif row[column] != text:
                        raise ValueError(f"summaryとJSONの不一致: {column}")
                if key in runs:
                    raise ValueError(f"実行キーの重複: {key}")
                if not np.isfinite(summary["run_metrics"]["prediction_accuracy"]):
                    raise ValueError("精度が有限でない")
                # 保存済みのラベルと予測の件数・正解数もJSONと照合する。
                correctness = tables["sample_prediction.combined_prediction_is_correct"]
                if (
                    correctness.shape != (key[3], key[4])
                    or correctness.size != summary["run_metrics"]["processed_sample_count"]
                    or float(correctness.mean()) != summary["run_metrics"]["prediction_accuracy"]
                ):
                    raise ValueError("NPZの予測とJSONの精度・件数が不一致")
                for column in (
                    "sample_prediction.observed_class_id",
                    "sample_prediction.observed_concept_id",
                ):
                    if column not in tables or tables[column].shape != correctness.shape:
                        raise ValueError(f"入力系列の形・必須列が不一致: {column}")
                runs[key] = (summary, tables, configuration, path)
                checks.append(
                    dict(
                        role=role,
                        path=str(path),
                        status="loaded",
                        detail=json.dumps(
                            dict(
                                environment=summary["environment"],
                                provenance=summary["run_provenance"],
                            ),
                            ensure_ascii=False,
                        ),
                    )
                )
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
                checks.append(dict(role=role, path=str(path), status="error", detail=str(error)))
        present_ids = {item[3].stem for item in runs.values()}
        for identifier, row in by_id.items():
            if (
                row.get("server_aggregation_interval_per_client_samples") in ("50", "500")
                and identifier not in present_ids
            ):
                checks.append(
                    dict(
                        role=role,
                        path=str(summary_path),
                        status="error",
                        detail=f"summary行のJSONが無い・読めない: {identifier}",
                    )
                )
    for key in sorted(expected - runs.keys()):
        checks.append(dict(role=role, path=str(key), status="error", detail="実行キー欠損"))
    return runs


def _write_csv(path, rows):
    # 分析の再実行も、既存分析を無言で上書きしない。
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root", type=Path, required=True, help="新規掃引のルート（分析出力先）"
    )
    for role in ("soft", "single", "hard", "diagnostic"):
        parser.add_argument(
            f"--{role}-roots",
            type=Path,
            nargs="+",
            required=role != "diagnostic",
            help="構成ディレクトリを指定（runsの親）",
        )
    args = parser.parse_args(argv)
    expected = {(d, i, s, 10, 5000) for d in _DATASETS for i in (50, 500) for s in range(10)}
    checks = []
    roots = {
        "Ssingle": [args.new_root / _NEW],
        "Smix": args.soft_roots,
        "Hsingle": args.single_roots,
        "Hmix": args.hard_roots,
    }
    if args.diagnostic_roots:
        roots["Diagnostic"] = args.diagnostic_roots
    groups = {role: _read_group(paths, role, expected, checks) for role, paths in roots.items()}
    for key in sorted(expected):
        if any(key not in runs for runs in groups.values()):
            continue
        reference, tables, _, _ = groups["Ssingle"][key]
        for role, runs in groups.items():
            summary, other, configuration, path = runs[key]
            try:
                _check_settings(reference["run_settings"], summary["run_settings"], configuration)
                if summary["environment"] != reference["environment"]:
                    raise ValueError("実行環境が異なる")
                detail = _check_inputs(tables, other)
                checks.append(
                    dict(
                        role=role,
                        path=str(path),
                        status="unverified" if detail else "paired",
                        detail=detail,
                    )
                )
            except ValueError as error:
                checks.append(dict(role=role, path=str(path), status="error", detail=str(error)))
    _write_csv(args.new_root / "analysis_pairing_check.csv", checks)
    if any(row["status"] in ("error", "unverified") for row in checks):
        print("照合不成立。analysis_pairing_check.csvを確認してください。集計は停止しました。")
        return 2
    metrics = {
        role: {key: _metrics(item[0], item[1]) for key, item in runs.items()}
        for role, runs in groups.items()
    }
    rows = []
    names = sorted(
        set().union(*(value.keys() for runs in metrics.values() for value in runs.values()))
    )
    contrasts = {
        "Ssingle-Smix": {"Ssingle": 1, "Smix": -1},
        "Ssingle-Hsingle": {"Ssingle": 1, "Hsingle": -1},
        "Ssingle-Hmix": {"Ssingle": 1, "Hmix": -1},
        "Smix-Ssingle": {"Smix": 1, "Ssingle": -1},
        "Hmix-Hsingle": {"Hmix": 1, "Hsingle": -1},
        "interaction": {"Smix": 1, "Ssingle": -1, "Hmix": -1, "Hsingle": 1},
    }
    for metric in names:
        is_accuracy = "accuracy" in metric
        for cell in [(d, i) for d in _DATASETS for i in (50, 500)] + [("all", 0)]:
            keys_by_seed = [
                [
                    (d, i, s, 10, 5000)
                    for d in _DATASETS
                    for i in (50, 500)
                    if cell == ("all", 0) or cell == (d, i)
                ]
                for s in range(10)
            ]
            for comparison, coefficients in {
                **{role: {role: 1} for role in groups},
                **contrasts,
            }.items():
                values = []
                missing = 0
                for keys in keys_by_seed:
                    paired = []
                    for key in keys:
                        parts = [metrics[role][key].get(metric) for role in coefficients]
                        if any(value is None or not np.isfinite(value) for value in parts):
                            missing += 1
                        else:
                            paired.append(
                                sum(
                                    coef * value
                                    for coef, value in zip(coefficients.values(), parts)
                                )
                            )
                    values.append(
                        float(np.mean(paired)) if len(paired) == len(keys) else float("nan")
                    )
                result = (
                    _paired_summary(values, accuracy=is_accuracy)
                    if not missing
                    else dict(
                        mean=None,
                        half_width_95=None,
                        positive_seeds=None,
                        zero_seeds=None,
                        negative_seeds=None,
                    )
                )
                rows.append(
                    dict(
                        dataset=cell[0],
                        interval=cell[1],
                        metric=metric,
                        comparison=comparison,
                        unit="pp" if is_accuracy else "保存単位",
                        missing_cells=missing,
                        **result,
                    )
                )
    _write_csv(args.new_root / "analysis_soft_training_single_model_prediction.csv", rows)
    lines = [
        "# 柔らかい学習と単体予測の対の集計",
        "",
        "監視の定義は4構成で共通の現在モデルの局所有界損失。学習後のモデル・警報の値は変わりうる。",
        "差はseed内の対で計算。全体行は20セルをseed内で等重み平均した10値からt(9)の95%区間を求めた。差の差は相互作用の記述であり、多様性の因果効果ではない。",
        "補助評価と再生を含む保存済み総MACを使用。上下byteと合計・MiBを併記。共有部・概念固有部、推定backwardの内訳はCSVの保存指標。",
        "時間は記述値。runごとの経過時間・peak RSSは現成果物に未保存ならN/A。未保存診断を0で補わない。純度はclient平均→run→seed平均、1モデル基準と欠損client数を別指標にした。診断構成は補助欄でありHmixを置換しない。",
        "版15のfedsda-every-sample-change-interval-searchは分析側だけでHmixへ対応付けた。実環境・commit・版の来歴は照合CSVに保存。",
        "概念別・期間別指標は保存済みrun_metricsの範囲。特徴digest未保存の直接入力照合と、未保存診断は未検証。入力生成の不変testの今回の証拠を確認し、主担当が再利用可否を判断する。",
        "",
        "|データ|間隔|比較|精度平均差/平均 (pp)|95%区間半幅|欠損|",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["metric"] == "prediction_accuracy":
            lines.append(
                "|"
                + "|".join(
                    str(row[k]) if row[k] is not None else "N/A"
                    for k in (
                        "dataset",
                        "interval",
                        "comparison",
                        "mean",
                        "half_width_95",
                        "missing_cells",
                    )
                )
                + "|"
            )
    with (args.new_root / "analysis_soft_training_single_model_prediction.md").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write("\n".join(lines) + "\n")
    print("分析を保存しました。未検証と再利用の判断事項はMarkdownと照合CSVを確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
