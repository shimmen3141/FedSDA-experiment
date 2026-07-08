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

# データセットを SEA-4(FedDrift互換)に切り替え
python run_experiment.py --mode FedSDA --seed 0 --dataset sea

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
| `Oblivious` | ベースライン: 単一モデル・FedAvg・無適応(FedDrift の Oblivious) |

### データセット(`--dataset`)

| dataset | 概念数 | 内容 |
|---|---|---|
| `blobs`(既定) | 4 | 独自の2次元合成(ガウス塊 / 同心円) |
| `sea` | 4 | **FedDrift SEA-4**。x1,x2,x3 ~ U[0,10](x3 ノイズ)、label = 1 iff (x1+x2) ≤ 閾値 |
| `circle` | 2 | **FedDrift CIRCLE-2**。x1,x2 ~ U[0,1]²、概念別の円の外側を label=1(小さな概念変化) |
| `sine` | 2 | **FedDrift SINE-2**。x1,x2 ~ U[0,1]²、概念0: x2≤sin(x1) を label=1、概念1: 反転(大きな概念変化) |

`sea` の閾値・ノイズ率は [FedSDA/config.py](FedSDA/config.py) の `SEA_THRESHOLDS`(FedDrift論文 appendix の A,B,C,D = `{0:9, 1:8, 2:7, 3:9.5}`)/ `SEA_LABEL_NOISE`(0.10)で定義。約10%の内在ラベルノイズがあるため精度の上限は約0.90。

> **データセット定義の注記**: FedDrift 同梱の `concept4.csv` は実測 θ≈9.0 で `concept2` と重複する(データ配布側のズレ)。本実装は**論文 appendix の正準定義** θ_D=9.5 に合わせ、4概念が区別されるようにしている。信号特徴は論文どおり x1+x2(x3 がノイズ)。

> **注**: ドリフトスケジュール(いつ・何回ドリフトするか)は FedDrift と異なり、本実装は各サンプルで確率的に複数回発生させる方式のまま(`make_concept_schedules`)。データセットのみ FedDrift 互換にしている。

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
