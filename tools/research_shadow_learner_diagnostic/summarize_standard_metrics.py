"""S1対照の標準指標と診断指標を、同じ条件・seed単位で一覧にする。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


def describe(values):
    """対象無しは0へ置換せず、母数と欠測数を残す。"""
    valid = [v for v in values if v is not None]
    return dict(
        n=len(valid),
        unavailable=len(values) - len(valid),
        mean=mean(valid) if valid else None,
        sd=stdev(valid) if len(valid) > 1 else None,
        minimum=min(valid) if valid else None,
        maximum=max(valid) if valid else None,
    )


def total_mac(counts):
    """保存済みの順伝播と推定逆伝播のMACを合計する。"""
    return sum(v for k, v in counts.items() if k.endswith("multiply_accumulate_count"))


def extract_pair(row):
    """既定と診断の標準値を残し、非介入の一致と追加計算も照合する。"""
    job, baseline, measured, diag = (
        row[k] for k in ("job", "baseline", "measured", "shadow_diagnostic")
    )
    if not row["baseline_parity"]:
        raise ValueError("既定との照合に失敗した条件です")
    for key in set(baseline) | set(measured):
        if key not in {"seconds", "model_computation_counts"} and baseline.get(
            key
        ) != measured.get(key):
            raise ValueError(f"既定の保存値と不一致: {key}")
    extra = diag["extra_model_computation_counts"]
    if {
        k: v - extra.get(k, 0) for k, v in measured["model_computation_counts"].items()
    } != baseline["model_computation_counts"]:
        raise ValueError("追加計算を除いた計数が不一致です")
    out = {k: job[k] for k in ("dataset", "stream", "interval", "seed", "sample_count")}
    for label, result in (("baseline", baseline), ("measured", measured)):
        for field in ("accuracy", "uploaded_bytes", "downloaded_bytes", "seconds"):
            out[f"{label}_{field}"] = result[field]
        out[f"{label}_mac"] = total_mac(result["model_computation_counts"])
        out[f"{label}_preparation_mac"] = total_mac(
            result["preparation_model_computation_counts"]
        )
        for field in ("stable_period_accuracy", "recovery_period_accuracy"):
            value = result["prediction_period_accuracies"][field]
            # 指標の実装は回復期の分母0を0.0で返す。無変化の回復期は対象無し。
            if (
                field == "recovery_period_accuracy"
                and result["training_model_switch_detection_metrics"][
                    "concept_change_count"
                ]
                == 0
            ):
                value = None
            out[f"{label}_{field}"] = value
        for field in (
            "concept_accuracies",
            "communication_volume",
            "model_computation_counts",
            "preparation_model_computation_counts",
            "prediction_period_accuracies",
        ):
            out[f"{label}_{field}_json"] = json.dumps(result[field], sort_keys=True)
    out.update(
        accuracy_delta_pp=100 * (measured["accuracy"] - baseline["accuracy"]),
        extra_mac=total_mac(extra),
        extra_mac_fraction=total_mac(extra) / out["baseline_mac"],
        shadow_episodes=len(diag["episodes"]),
        same_concept_d_crossings=sum(
            e["first_crossing_same_alarm_concept"] is True for e in diag["episodes"]
        ),
        same_concept_c_crossings=sum(
            e["control_first_crossing_same_alarm_concept"] is True
            for e in diag["episodes"]
        ),
    )
    return out


def summarize(output):
    from tools.research_shadow_learner_diagnostic.analyze_controls import analyze

    # 実行時ソース・条件・全軌跡の整合性を既存の解析器で先に確認する。
    analyze(output)
    source = output / "results.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    pairs = sorted(
        (extract_pair(r) for r in original),
        key=lambda r: (r["dataset"], r["stream"], r["interval"], r["seed"]),
    )
    groups = []
    for key in sorted({(r["dataset"], r["stream"], r["interval"]) for r in pairs}):
        rs = [r for r in pairs if (r["dataset"], r["stream"], r["interval"]) == key]
        item = dict(
            dataset=key[0],
            stream=key[1],
            interval=key[2],
            seeds=[r["seed"] for r in rs],
        )
        fields = [
            k
            for k, v in rs[0].items()
            if (k.startswith(("baseline_", "measured_")) and not k.endswith("_json"))
            or k in ("accuracy_delta_pp", "extra_mac_fraction")
        ]
        item["metrics"] = {f: describe([r[f] for r in rs]) for f in fields}
        item["pooled_extra_mac_fraction"] = sum(r["extra_mac"] for r in rs) / sum(
            r["baseline_mac"] for r in rs
        )
        item["diagnostics"] = {
            f: sum(r[f] for r in rs)
            for f in (
                "shadow_episodes",
                "same_concept_d_crossings",
                "same_concept_c_crossings",
            )
        }
        concepts = sorted(
            {
                c
                for row in original
                if (row["job"]["dataset"], row["job"]["stream"], row["job"]["interval"])
                == key
                for c in row["baseline"]["concept_accuracies"]
            }
        )
        item["concept_accuracy"] = {
            c: describe(
                [json.loads(r["baseline_concept_accuracies_json"]).get(c) for r in rs]
            )
            for c in concepts
        }
        groups.append(item)
    with (output / "standard_metrics_by_seed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = [
        "# S1対照の標準指標と診断の併記",
        "",
        "各行は同一入力・集約間隔のseed別平均。精度はmean ± sample SD、MB=10^6 byte、G MAC=10^9 MAC。標本ごとの重み付けではなくseedの単純平均。",
        "",
        "| データ | 入力 | 間隔 | seed数 | 精度%（両者同値） | 上りMB（同値） | 下りMB（同値） | 既定G MAC | 診断G MAC | 追加MAC%（総量比） |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        m = g["metrics"]
        a = m["baseline_accuracy"]
        sd = "N/A" if a["sd"] is None else f"{100 * a['sd']:.3f}"
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {len(g['seeds'])} | {100 * a['mean']:.3f} ± {sd} | {m['baseline_uploaded_bytes']['mean'] / 1e6:.3f} | {m['baseline_downloaded_bytes']['mean'] / 1e6:.3f} | {m['baseline_mac']['mean'] / 1e9:.3f} | {m['measured_mac']['mean'] / 1e9:.3f} | {100 * g['pooled_extra_mac_fraction']:.3f} |"
        )
    lines += [
        "",
        "## 期間別精度・経過時間・診断",
        "",
        "| データ | 入力 | 間隔 | 定常期% | 回復期% | 既定秒 | 診断秒 | 全影の数 | 同概念中D交差 | 同概念中C交差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        m, d = g["metrics"], g["diagnostics"]
        recovery = m["baseline_recovery_period_accuracy"]["mean"]
        recovery = "N/A" if recovery is None else f"{100 * recovery:.3f}"
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {100 * m['baseline_stable_period_accuracy']['mean']:.3f} | {recovery} | {m['baseline_seconds']['mean']:.2f} | {m['measured_seconds']['mean']:.2f} | {d['shadow_episodes']} | {d['same_concept_d_crossings']} | {d['same_concept_c_crossings']} |"
        )
    lines += [
        "",
        "無変化入力の回復期は分母0なのでN/A。元の保存値0.0はCSVのraw JSONに残す。概念別精度はsummary_standard_metrics.json、seed別の全標準値・計数はstandard_metrics_by_seed.csv。",
        "",
        "MACはmodel_computation_counts内の順伝播＋推定逆伝播の和。preparation_model_computation_countsは別列に保存し、表には別途加算しない。追加計算には影と検証用オンライン継続対照の両方を含み、将来S1を採用した場合の計算量ではない。",
        "",
        "経過秒数は64workersで実行した各runの平均で、実験全体のwall timeやCPU時間ではない。既定を先に診断を後に実行する順序と並列負荷があり、時間差を因果的なオーバーヘッドとは解釈しない。",
        "",
        "全対で精度・通信量は同値、追加計算を除いた計数も同値。これは影を予測・採用に使わない診断の確認であり、S1を採用した後の改善を測ったものではない。再実験はしていない。",
    ]
    (output / "analysis_standard_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    result = dict(
        groups=groups,
        pair_count=len(pairs),
        input_sha256={
            f: hashlib.sha256((output / f).read_bytes()).hexdigest()
            for f in ("results.json", "research-source.json")
        },
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "summary_standard_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
