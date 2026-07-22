"""数値リグレッションテスト: リファクタで実験の挙動(メトリクス)が変わっていないか検証する。

固定シード・小規模設定で各モード×データセットを実行し、記録済みの golden 値と
一致するかを確認する。前セッションで「quick 実行時の精度が完全一致するか」で確認していた
手順を、複数モード・複数指標に広げて自動化・恒久化したもの。

使い方:
    # 依存: FedSDA パッケージ(torch/numpy)。pytest は任意(下記のどちらでも可)
    python tests/test_regression.py                # 単体実行(golden と比較)
    python tests/test_regression.py --update       # 現在の結果を golden として保存/更新
    python tests/test_regression.py --tol 1e-6     # 許容誤差を緩める
    pytest tests/test_regression.py                # pytest でも同じ検証を実行

golden を最初に作る/意図的に挙動を変えたときは `--update` で更新する。
差分が出たら「意図した変更か」を確認し、意図通りなら --update、そうでなければリグレッション。

**バージョン依存の注意**: golden 値は、それを生成したときの torch / numpy の数値に固定される
(生成環境は golden ファイル tests/regression_golden.json の "_env" フィールドに記録され、
比較実行時に現在の環境と食い違えば警告を出す)。torch / numpy を更新すると浮動小数演算
(BLAS・RNG 実装)が変わり、ロジック無変更でもわずかにズレて誤検知することがある。その場合は
環境を揃えるか、更新後に `--update` で golden を取り直す。pytest のバージョンは数値に影響しない
(検証の実行だけ)。
"""
import argparse
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout

# tests/ から実行してもパッケージをimportできるようにリポジトリルートを追加
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config, run_random_drift_experiment  # noqa: E402
from federated_drift_experiment.mode_names import normalize_legacy_mode  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_golden.json")

# 固定マトリクス: (mode, dataset, config上書き)。seed=0・小規模なので決定的かつ短時間。
# これでアルゴリズム本体(clients/server/adwin/metrics/実験ループ)の全モード経路を網羅する。
# データ生成の別分布や別入力次元(sea=3次元)まで固定したい場合はケースを追加する。
# FedSDA_NoCached_ADWIN は sine(マージが発生する)+ τ=10 でNoCached固有の
# 経路(FedAvg先行サーバ・τ バッチ更新)を固定する。
CASES = [
    ("FedSDA_NoCached_ClassADWIN", "circle", {}),
    ("FedSDA_NoCached_ESR", "circle", {}),
    ("FedSDA_NoCached_ClassESR", "circle", {}),
    ("FedSDA_Cached_ADWIN", "blobs", {}),
    ("FedSDA_Cached_ClassADWIN", "circle", {}),
    ("FedSDA_Cached_ESR", "circle", {}),
    ("FedSDA_Cached_ClassESR", "circle", {}),
    ("FedDrift", "blobs", {
        "TOTAL_DATA_POINTS": 300,
        "N_CLIENTS": 4,
        "PRETRAIN_SAMPLES": 100,
        "MIN_STABLE_PERIOD": 50,
        "DRIFT_PROB": 0.03,
    }),
    ("FedSDA_without_server", "blobs", {}),
    ("Oblivious", "blobs", {}),
    ("FedSDA_NoCached_ADWIN", "sine", {
        "TOTAL_DATA_POINTS": 1500, "LOCAL_UPDATE_TAU": 10}),
]
SEED = 0
TOTAL_DATA_POINTS = 600

# 比較対象の指標(runtime など非決定的な値は含めない)
METRIC_KEYS = [
    "accuracy", "stable_accuracy",
    "comm_models_up", "comm_models_down", "comm_models_total",
    "comm_messages_up", "comm_messages_down", "comm_messages_total",
    "final_model_count", "precision", "recall", "f1",
    "avg_delay", "total_detect",
    "change_point_mae", "change_point_bias", "change_point_estimate_count",
]

DEFAULT_TOL = 1e-9  # 「完全一致」に近い厳密さ。FP 揺れで誤検知するなら --tol で緩める


def _run_case(mode, dataset, overrides=None):
    """1 ケースを実行し、比較対象メトリクスの dict を返す(実験ログは抑止)。

    overrides で指定した config 属性はケース実行中のみ上書きし、終了後に復元する
    (後続ケースへ設定が漏れないように)。
    """
    config.DATASET = dataset
    original_schedule = config.CONCEPT_SCHEDULE
    config.CONCEPT_SCHEDULE = "random"
    config.TOTAL_DATA_POINTS = TOTAL_DATA_POINTS
    # 新しい再割当機構を無効化し、既存手法の従来結果を継続監視する。
    overrides = {
        "RECENT_ASSIGNMENT_JOURNAL_SIZE": 0,
        **(overrides or {}),
    }
    saved = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        with redirect_stdout(io.StringIO()):
            r = run_random_drift_experiment(mode=mode, random_seed=SEED,
                                            verbose=False, show_plot=False)
    finally:
        config.CONCEPT_SCHEDULE = original_schedule
        for k, v in saved.items():
            setattr(config, k, v)
    return {k: r.get(k) for k in METRIC_KEYS}


def compute_all():
    """全ケースを実行し {"mode/dataset": {metric: value}} を返す。"""
    return {f"{mode}/{ds}": _run_case(mode, ds, ov) for mode, ds, ov in CASES}


def _env():
    import numpy
    import torch
    return {"torch": torch.__version__, "numpy": numpy.__version__,
            "python": sys.version.split()[0]}


def _values_close(a, b, tol):
    if a is None or b is None:
        return a == b
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return math.isclose(fa, fb, rel_tol=0.0, abs_tol=tol)


def compare(current, golden, tol):
    """current と golden の差分を [(name, kind, current, golden)] で返す(空なら一致)。"""
    diffs = []
    gold_cases = {k: v for k, v in golden.items() if not k.startswith("_")}
    for case in sorted(set(current) | set(gold_cases)):
        cur, gold = current.get(case), gold_cases.get(case)
        if cur is None or gold is None:
            diffs.append((case, "ケース欠落", cur is not None, gold is not None))
            continue
        for k in METRIC_KEYS:
            if not _values_close(cur.get(k), gold.get(k), tol):
                diffs.append((f"{case}.{k}", "変化", cur.get(k), gold.get(k)))
    return diffs


def _load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        stored = json.load(f)
    normalized = {key: value for key, value in stored.items() if key.startswith("_")}
    for key, value in stored.items():
        if key.startswith("_"):
            continue
        mode, dataset = key.split("/", 1)
        # FedSDA は旧v1の曖昧な名称なので比較対象外とする。
        # FedDrift は現在の正式モード名であり、除外してはいけない。
        if mode == "FedSDA":
            continue
        normalized[f"{normalize_legacy_mode(mode)}/{dataset}"] = value
    return normalized


def main():
    ap = argparse.ArgumentParser(description="FedSDA 数値リグレッションテスト")
    ap.add_argument("--update", action="store_true", help="現在の結果を golden として保存し直す")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"浮動小数の許容絶対誤差(default {DEFAULT_TOL})")
    args = ap.parse_args()

    current = compute_all()

    if args.update:
        payload = {"_env": _env(), **current}
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"golden 更新: {GOLDEN_PATH} ({len(current)} ケース, env={_env()})")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"golden がありません。まず `python {os.path.relpath(__file__, _REPO_ROOT)} "
              f"--update` を実行してください。")
        return 2

    golden = _load_golden()
    diffs = compare(current, golden, args.tol)
    if diffs:
        print(f"REGRESSION: {len(diffs)} 件の相違 (tol={args.tol})")
        if "_env" in golden and golden["_env"] != _env():
            print(f"  [注意] golden 生成環境と不一致: golden={golden['_env']} / 現在={_env()}")
            print("        torch/numpy 差なら FP 揺れの可能性。--tol を緩めるか --update を検討。")
        for name, kind, cur, gold in diffs:
            print(f"  {name}: {kind}  current={cur}  golden={gold}")
        return 1

    print(f"OK: {len(current)} ケース一致 (tol={args.tol})")
    return 0


def test_regression():
    """pytest 用エントリ: golden と一致することを assert する。"""
    assert os.path.exists(GOLDEN_PATH), (
        "golden 未作成。`python tests/test_regression.py --update` を先に実行してください。")
    diffs = compare(compute_all(), _load_golden(), DEFAULT_TOL)
    assert not diffs, "数値リグレッション:\n" + "\n".join(
        f"  {n}: {k} current={c} golden={g}" for n, k, c, g in diffs)


if __name__ == "__main__":
    sys.exit(main())
