"""S1対照の全軌跡を照合し、同じ入力ディレクトリへ分析を保存する。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median


def analyze(output):
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    for name, expected in metadata["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"実行時と分析時のソースが異なります: {name}")
    if (output / "failures.json").exists():
        raise ValueError(
            "失敗のある実験は最終集計しません。failures.jsonを確認してください"
        )
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    def key(j):
        return j["dataset"], j["stream"], j["interval"], j["seed"]
    assert len(rows) == len(metadata["jobs"])
    assert Counter(key(r["job"]) for r in rows) == Counter(
        key(j) for j in metadata["jobs"]
    )
    groups, summaries = defaultdict(list), []
    for row in rows:
        job, diag = row["job"], row["shadow_diagnostic"]
        baseline, measured = row["baseline"], row["measured"]
        assert row["baseline_parity"] is True
        assert all(
            baseline.get(k) == measured.get(k)
            for k in set(baseline) | set(measured)
            if k not in {"seconds", "model_computation_counts"}
        )
        extra = diag["extra_model_computation_counts"]
        assert {
            k: v - extra.get(k, 0)
            for k, v in measured["model_computation_counts"].items()
        } == baseline["model_computation_counts"]
        if job["fixed_concept"] is not None:
            assert set(measured["concept_accuracies"]) == {str(job["fixed_concept"])}
        traces = [
            json.loads(line)
            for line in (output / diag["trace_file"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(traces) == diag["trace_row_count"]
        assert extra.get(
            "concept_specific_parameter_optimizer_step_count", 0
        ) == 2 * len(traces)
        by_episode = defaultdict(list)
        for t in traces:
            by_episode[t["episode_id"]].append(t)
        for e in diag["episodes"]:
            trajectory = by_episode[e["episode_id"]]
            assert len(trajectory) == e["observed_sample_count"]
            d = e["current_code_length"] - e["new_code_length"]
            assert d == e["initial_score"] <= 0
            c, advantage, same, first, control_first = d, 0.0, True, None, None
            joint = False
            for age, t in enumerate(trajectory, 1):
                assert (
                    t["age"] == age
                    and t["sample_index"] == e["alarm_sample_index"] + age
                )
                assert t["client_id"] == e["client_id"]
                d += t["current_log_loss"] - t["shadow_log_loss"]
                c += t["online_current_log_loss"] - t["shadow_log_loss"]
                advantage += t["current_log_loss"] - t["online_current_log_loss"]
                same = same and t["observed_concept_id"] == e["alarm_concept_id"]
                assert t["same_alarm_concept_so_far"] == same
                assert abs(t["score"] - d) < 1e-10
                assert abs(t["control_score"] - c) < 1e-10
                assert abs(t["current_control_advantage"] - advantage) < 1e-10
                assert abs(d - c - advantage) < 1e-10
                if first is None and d > 0:
                    first = (age, same)
                if control_first is None and c > 0:
                    control_first = (age, same)
                joint = joint or (same and d > 0 and c > 0)
            assert e["first_crossing_age"] == (first[0] if first else None)
            assert e["first_crossing_same_alarm_concept"] == (
                first[1] if first else None
            )
            assert e["control_first_crossing_age"] == (
                control_first[0] if control_first else None
            )
            assert e["control_first_crossing_same_alarm_concept"] == (
                control_first[1] if control_first else None
            )
            assert (
                abs(e["last_score"] - d) < 1e-10 and abs(e["control_score"] - c) < 1e-10
            )
            assert e["end_reason"] is not None
            if e["end_reason"] == "horizon":
                assert len(trajectory) == job["horizon_samples"]
            groups[
                (job["dataset"], job["stream"], job["interval"], e["alarm_concept_id"])
            ].append(dict(seed=job["seed"], episode=e, joint=joint))
        mac_keys = [
            k
            for k in baseline["model_computation_counts"]
            if k.endswith("multiply_accumulate_count")
        ]
        mac_fraction = sum(extra.get(k, 0) for k in mac_keys) / sum(
            baseline["model_computation_counts"][k] for k in mac_keys
        )
        summaries.append(
            dict(
                **{k: job[k] for k in ("dataset", "stream", "interval", "seed")},
                episodes=len(diag["episodes"]),
                observations=len(traces),
                extra_mac_fraction=mac_fraction,
            )
        )
    strata = []
    lines = [
        "# S1の無変化・オンライン継続対照",
        "",
        f"{len(rows)}条件、各条件の既定と診断の全保存値を照合（秒数・追加計算を除く）。失敗0。",
        "",
        "Dは製品の現在のモデルに対する差、Cは同じオンライン手順で継続した現在の写しに対する差。どちらも初期差を含む。",
        "",
        "| データ | 入力 | 間隔 | 開始概念 | 影 | D交差 | C交差 | 同時にD,C>0 | 同時到達のseed数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (dataset, stream, interval, concept), entries in sorted(groups.items()):
        es = [x["episode"] for x in entries]
        item = dict(
            dataset=dataset,
            stream=stream,
            interval=interval,
            concept=concept,
            episodes=len(es),
            d_crossings=sum(e["first_crossing_same_alarm_concept"] is True for e in es),
            c_crossings=sum(
                e["control_first_crossing_same_alarm_concept"] is True for e in es
            ),
            joint_crossings=sum(x["joint"] for x in entries),
            joint_seeds=sorted({x["seed"] for x in entries if x["joint"]}),
            initial_gap_median=median(-e["initial_score"] for e in es),
            end_reasons=dict(Counter(e["end_reason"] for e in es)),
        )
        strata.append(item)
        lines.append(
            f"| {dataset} | {stream} | {interval} | {concept} | {len(es)} | {item['d_crossings']} | {item['c_crossings']} | {item['joint_crossings']} | {len(item['joint_seeds'])} |"
        )
    lines += [
        "",
        "交差は開始概念が変わる前のみ。影0件のrunは下のrun一覧に含む。警報・上限・run末による打切りを含み、到達率の推定ではない。同じseed/client内の影は独立ではない。",
        "",
        "現在の写しは警報の処理後のパラメータとoptimizerを引き継ぎ、以後は共有部を固定して影と同じ標本で毎回1回学習する。初期状態・過去の学習は影と異なり、この対照で全ての交絡を除いたとは主張しない。Cは診断用の差であり、新しいMDL採用規則ではない。採用、精度の改善、サーバでの保持は未検証。",
        "",
        "## run別の生成数と追加計算",
        "",
        "追加MACは診断とオンライン対照の両方を含む見積り。実時間の増加率ではない。",
        "",
        "```json",
        json.dumps(summaries, indent=2),
        "```",
    ]
    (output / "analysis_controls.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_controls.json").write_text(
        json.dumps(dict(runs=summaries, strata=strata), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
