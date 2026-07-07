# FedSDA 実験コード

FedSDA (Statistical Drift-aware Assignment for Federated Learning) の比較実験。
アルゴリズムの詳細は [ALGORITHM.md](ALGORITHM.md)、論文本体は [main_jp.tex](main_jp.tex) を参照。

## セットアップ

```
pip install -r requirements.txt
```

(Python 3.14 + torch 2.12 / numpy / matplotlib で動作確認済み)

## 実行方法

```powershell
# 単発実験(提案手法、seed=0、図は results/ に保存)
python run_experiment.py --mode FedSDA --seed 0 --plot-dir results

# 3モード × 10シードの比較試行
python run_comparative_trials.py --n-trials 10 --plot-dir results

# オプション確認
python run_experiment.py --help
python run_comparative_trials.py --help
```

`--plot-dir` を省略すると図はウィンドウ表示(`plt.show()`)になる。

### 実験モード

| mode | 内容 |
|---|---|
| `FedSDA` | **提案手法**: ADWIN逐次検出 + FIFOバッファ + サーバ集約 |
| `FedDrift` | ベースライン: 固定バッチ検出 + サーバ集約 |
| `FedSDA_without_server` | 提案手法のローカルのみ版(サーバ集約なし) |

## コード構成

```
.
├── run_experiment.py            # CLI: 単発実験
├── run_comparative_trials.py    # CLI: 複数シード比較試行
├── FedSDA/                      # 実験パッケージ
│   ├── config.py                # ★ハイパーパラメータの一元管理
│   ├── data.py                  # 合成データ生成・ドリフトスケジュール
│   ├── models.py                # SimpleMLP(2次元入力の二値分類)
│   ├── adwin.py                 # FullScanADWIN(全分割点走査のADWIN)
│   ├── clients.py               # BaseClient / AdwinClient(提案) / PeriodicClient(FedDrift)
│   ├── server.py                # サーバ(FedAvg集約・階層的クラスタリングマージ)
│   ├── experiment.py            # 実験本体 run_random_drift_experiment
│   ├── metrics.py               # 検出性能メトリクス(TP/FP/FN, 遅延など)
│   ├── plotting.py              # 可視化(保存 or 表示)
│   └── trials.py                # 複数試行の実行・集計
├── colab_original/              # 分割前のColab版スクリプト(参照用バックアップ)
├── ALGORITHM.md                 # 論文の実装用仕様書
└── main_jp.txx                  # 論文原稿(LaTeX)
```

> 参照用に `microsoft/FedDrift` を `FedDrift/` へ clone している場合、それは
> `.gitignore` によりこのリポジトリの追跡対象外となる。

## ハイパーパラメータ

全ハイパーパラメータは [FedSDA/config.py](FedSDA/config.py) に一元管理されている
(論文の記号 K, τ, L, E_init, δ_adwin, N_FIFO, γ_dist との対応もコメントに記載)。

既定値の概要:

- 実験規模: クライアント数 10、総データ数 3000/クライアント、1ラウンド = 50 ステップ
- データ: コンセプト数 4(ガウス2塊×ラベル反転、同心円×ラベル反転)
- ドリフト: 安定期間300サンプル経過後、毎サンプル確率0.0015で別コンセプトへ遷移
- 検出: ADWIN δ=0.05、FIFOバッファ長 30、距離閾値 γ_dist=0.1

変更方法:

```python
# ファイルを直接編集するか、コードから上書きする(各モジュールは実行時に参照)
from FedSDA import config
config.TOTAL_DATA_POINTS = 300   # 例: 縮小実験
config.ADWIN_DELTA = 0.01        # 例: 検出感度の変更
```
