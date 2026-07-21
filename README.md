# FedSDA 実験コード

FedSDA (Statistical Drift-aware Assignment for Federated Learning) の比較実験。
アルゴリズムの詳細は [docs/fedsda-algorithm.md](docs/fedsda-algorithm.md)、論文本体は [main_jp.tex](main_jp.tex) を参照。

## セットアップ

```
pip install -r requirements.txt                     # 実行時依存(版固定)
pip install -r requirements.txt -r requirements-dev.txt   # テストも回すなら(pytest 追加)
```

(Python 3.14 + torch 2.12.1 / numpy 2.4.6 / matplotlib 3.11.0 で動作確認済み。版は
requirements に固定。数値リグレッションテストの golden はこの版で生成した値に依存する)

## テスト

```powershell
python tests/test_regression.py        # 数値リグレッション(golden と一致するか)
pytest tests/test_regression.py        # pytest でも可
python tests/test_regression.py --update   # 意図的に挙動を変えた/初回のみ golden 再生成
```

固定シード・小規模で全モードを実行し、記録済み `tests/regression_golden.json` と主要指標
(精度・通信量・モデル数・検出性能)が一致するかを検証する。torch/numpy を更新すると
浮動小数演算が変わり誤検知し得るため、その場合は `--tol` を緩めるか `--update` で取り直す。

## 実行方法

```powershell
# 単発実験(提案手法、seed=0、図は results/ に保存)
python run_experiment.py --mode FedSDA_NoCached_ADWIN --seed 0 --plot-dir results

# データセットを SEA-4(FedDrift互換)に切り替え
python run_experiment.py --mode FedSDA_NoCached_ADWIN --seed 0 --dataset sea

# 同じデータ分布でFedDrift論文の固定ドリフト系列を使用
python run_experiment.py --mode FedSDA_NoCached_ADWIN --seed 0 --dataset sea --concept-schedule feddrift_fixed

# FedDrift の検出バッチサイズを掃引(精度–通信量トレードオフの観察)
python run_experiment.py --mode FedDrift --feddrift-batch 500 --seed 0

# 3モード × 10シードの比較試行
python run_comparative_trials.py --n-trials 10 --plot-dir results

# 精度–通信量の掃引(FedSDA δ_adwin / FedDrift バッチ・δ)。結果は実験内容がわかる名前で保存
python run_pareto_sweep.py --datasets sea circle sine --seeds 0

# プロトコルと検出器を限定して掃引
python run_pareto_sweep.py --datasets sea --seeds 0 1 2 `
  --fedsda-modes FedSDA_NoCached_ADWIN FedSDA_Cached_ADWIN --feddrift-modes FedDrift

# 掃引をシードごとに分割して積み増した後、複数CSVをシード平均で1枚に再描画
python run_pareto_sweep.py --plot-csvs "results/pareto/pareto_sea-circle-sine_seed*_n5000.csv"

# 回復曲線(適応速度)分析: 生データを保存 → acc(Δ) を集計・描画
python run_pareto_sweep.py --datasets sine --seeds 0 1 2 --raw-dir results/raw
python recovery_analysis.py --npz "results/raw/*sine*.npz"

# オプション確認
python run_experiment.py --help
python run_comparative_trials.py --help
```

### CPUスレッド設定

CLI実験では、小さいPyTorch演算を大量に繰り返す際の過剰並列化を避けるため、
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`を既定値とする。PyTorchのintra-opスレッド数も
`OMP_NUM_THREADS`に合わせ、inter-opスレッド数は1に固定する。外部で指定した環境変数は
上書きしないため、必要なら実行環境ごとに変更できる。

Linuxサーバでユーザーサイトのパッケージを混入させず実行する例:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
PYTORCH_NVML_BASED_CUDA_CHECK=1 \
python -u run_pareto_sweep.py 2>&1 | tee run_pareto_sweep.log
```

`PYTHONNOUSERSITE`はPython起動時に処理されるため、必要な場合はコマンド側で指定する。
`PYTHONWARNINGS=ignore`は数値警告や非推奨APIの警告も隠すため、通常の実験では設定しない。

Pareto 図では `FedSDA_without_server` と `Oblivious` を通信量に依存しない基準として横線で描く。
横線はシード平均、半透明の帯はシード間の ±1 標準偏差である。1 シードだけの場合は標準偏差が
0 のため帯は表示されない。凡例には前者の固定 `δ_adwin`、後者の固定 `AGG_INTERVAL` も示す。
通信量 0 の点を対数軸に置く表現ではなく、精度の参照線として表示する。

`--plot-dir` を省略すると図はウィンドウ表示(`plt.show()`)になる。

### 実験モード

| mode | 内容 |
|---|---|
| `FedSDA_NoCached_ADWIN` | 現ラウンドのFedAvg済みモデルをクロス評価し、全体損失ADWINで検出 |
| `FedSDA_NoCached_ClassADWIN` | NoCached + 全体・正解クラス別ADWIN |
| `FedSDA_NoCached_ESR` | NoCached + bounded mean向け混合Shiryaev–Roberts型e-detector |
| `FedSDA_NoCached_ClassESR` | NoCached + 全体・正解クラス別e-SRの固定重み混合 |
| `FedSDA_NoCached_HDDMA` / `FedSDA_NoCached_HDDMW` | NoCached + 全体損失HDDM-A / HDDM-W |
| `FedSDA_NoCached_ClassHDDMA` | NoCached + 全体・正解クラス別HDDM-A |
| `FedSDA_Cached_ADWIN` | 配布済みモデルのキャッシュでクロス評価し、全体損失ADWINで検出 |
| `FedSDA_Cached_ClassADWIN` | Cached + 全体・正解クラス別ADWIN |
| `FedSDA_Cached_ESR` | Cached + bounded mean向け混合e-SR |
| `FedSDA_Cached_ClassESR` | Cached + 全体・正解クラス別e-SRの固定重み混合 |
| `FedSDA_Cached_HDDMA` / `FedSDA_Cached_HDDMW` | Cached + 全体損失HDDM-A / HDDM-W |
| `FedSDA_Cached_ClassHDDMA` | Cached + 全体・正解クラス別HDDM-A |
| `*_ESR_UCB` / `*_ClassESR_UCB` | 対応するe-SRの基準平均をempirical Bernstein上側信頼限界へ置換するアブレーション |
| `FedDrift` | 論文準拠フロー: 固定バッチ検出、新規モデル隔離、選択可能linkage、正確にR回のFedAvg |
| `FedSDA_without_server` | 提案手法のローカルのみ版(サーバ集約なし) |
| `Oblivious` | ベースライン: 単一モデル・FedAvg・無適応(FedDrift の Oblivious) |

### データセット(`--dataset`)

| dataset | 概念数 | 内容 |
|---|---|---|
| `blobs`(既定) | 4 | 独自の2次元合成(ガウス塊 / 同心円) |
| `sea` | 4 | **FedDrift SEA-4**。x1,x2,x3 ~ U[0,10](x3 ノイズ)、label = 1 iff (x1+x2) ≤ 閾値 |
| `sea2` | 2 | **FedDrift SEA-2**。SEAの概念0/1 |
| `circle` | 2 | **FedDrift CIRCLE-2**。x1,x2 ~ U[0,1]²、概念別の円の外側を label=1(小さな概念変化) |
| `sine` | 2 | **FedDrift SINE-2**。x1,x2 ~ U[0,1]²、概念0: x2≤sin(x1) を label=1、概念1: 反転(大きな概念変化) |
| `mnist2` | 2 | **FedDrift MNIST-2**。元ラベルと1↔2置換 |
| `mnist4` | 4 | **FedDrift MNIST-4**。元ラベル、1↔2、3↔4、5↔6置換 |

`sea` の閾値・ノイズ率は [federated_drift_experiment/config.py](federated_drift_experiment/config.py) の `SEA_THRESHOLDS`(FedDrift論文 appendix の A,B,C,D = `{0:9, 1:8, 2:7, 3:9.5}`)/ `SEA_LABEL_NOISE`(0.10)で定義。約10%の内在ラベルノイズがあるため精度の上限は約0.90。

> **FedDrift 元論文との相違点**（データセット定義・ドリフトスケジュール・学習パラメータ・
> 評価指標の細かな差異）は [docs/differences-from-feddrift.md](docs/differences-from-feddrift.md) に
> 一元的にまとめている。要点: sine/circle/sea の**生成規則は論文・参照コードに忠実**、
> 全データセットで`--concept-schedule random`（既定）と`feddrift_fixed`を選択できる。
> `random`は従来どおりper-sampleの確率的ドリフト、`feddrift_fixed`は2概念で
> `A.cp`、4概念で`B.cp`相当の10時点系列を使う。MNISTは初回実行時にIDXファイルを
> `data/mnist` へ取得し、`FDE_MNIST_DATA_DIR` で保存先を変更できる。

### 評価指標(結果 dict のキー)

| キー | 意味 |
|---|---|
| `accuracy` | prequential(逐次)精度: 各サンプルを予測→即学習した際の当否の平均。**全期間の総合精度** |
| `stable_accuracy` | **定常精度**: 各真ドリフト直後 `STABLE_WINDOW`(=W)サンプル(回復中)を除外した prequential 精度。回復曲線 acc(Δ) の Δ≥W の裾に相当 |
| `recall` / `precision` / `f1` | ドリフト検出の質(ローカル切替を検出とみなし真のドリフトと照合) |
| `avg_delay` | 平均検出遅延(サンプル数) |
| `final_model_count` | 最終モデル数(サーバ集約あり)/ クライアント平均保持数(なし) |
| `comm_models_up` / `comm_models_down` / `comm_models_total` | モデルパラメータ転送数。up=クライアント→サーバ、down=サーバ→クライアント |
| `comm_messages_up` / `comm_messages_down` / `comm_messages_total` | 割当・ドリフト要約、クロス評価依頼・統計返送、ID割当、マージ通知などの軽量メッセージ数 |
| `compute_inference_examples_total` / `compute_training_examples_total` | 推論系 / 学習でモデルに入力した延べサンプル数 |
| `compute_optimizer_steps_total` | 全クライアントの optimizer 更新回数 |
| `compute_drift_detector_hypotheses_total` | ADWINの候補分割、またはe-SRの候補変化点×賭け率を評価した延べ数 |
| `change_point_mae` / `change_point_bias` | 検出器が推定した変化開始位置の平均絶対誤差 / 符号付き平均誤差 |
| `mean_model_count` / `max_model_count` / `model_count_auc` | ラウンド末モデル数の平均 / 最大 / 全ラウンド和 |
| `client_compute_seconds_sum` / `client_compute_seconds_max` | クライアント処理時間の総和 / クライアント別最大値(実行環境依存) |

定常精度の回復除外窓 W は [federated_drift_experiment/config.py](federated_drift_experiment/config.py) の `STABLE_WINDOW`(既定200)で設定。最も遅い回復のプラトーを越える大きめ固定値で、`MIN_STABLE_PERIOD` 未満(次ドリフトを跨がない)。適応の**速さ**は回復曲線 acc(Δ)([recovery_analysis.py](recovery_analysis.py))で別途評価する。

モデル通信量は「1単位 = 1モデルのパラメータを1回転送」とする。軽量メッセージはモデル転送と
サイズが異なるため別指標で数え、両者を単純加算しない。

### 回復曲線(適応速度)分析

検出遅延(recall/precision/avg_delay)は手法ごとに検出の定義が揺れ、特に大バッチ FedDrift では比較に使いにくい。そこで **outcome ベースの適応速度指標**として回復曲線 acc(Δ) を用意している。

- `--raw-dir` を付けて実験を回すと、各 run の per-sample 生データに加え、クライアント別・ラウンド別の計算カウンタ、処理時間、モデル数時系列を軽量な `.npz`(gitignore 対象)に保存する。`run_experiment.py` / `run_pareto_sweep.py` の両方で使える。
- `recovery_analysis.py` が `.npz` を読み込み、各ドリフト発生点を Δ=0 に揃えて「ドリフト後 Δ サンプル時点の平均精度 acc(Δ)」を集計する。

出力:

| 出力 | 内容 |
|---|---|
| 図 (`recovery_*.png`) | データセット別に acc(Δ) を手法ごとに重ね描き(シード間 std を帯で表示) |
| 表 (`recovery_*.md`) | 固定オフセット精度 `acc@Δ`(既定 Δ=20/50/100)と、固定窓 `[0, W)` の平均精度 |

固定窓の平均精度は「適応リグレット = 基準精度 − 平均精度」に相当し、値が高いほど適応が速い。窓幅 W(既定200)と Δ 上限(既定250)はいずれも `MIN_STABLE_PERIOD`(300)未満に取るため、集計窓が必ず単一の安定区間に収まり次ドリフトが混入しない(ランダムなドリフト間隔に値が振り回されない)。生データを保存しておけば、数時間かかる掃引を再実行せずに事後的に指標を計算できる。

## コード構成

```
.
├── run_experiment.py            # CLI: 単発実験
├── run_comparative_trials.py    # CLI: 複数シード比較試行
├── federated_drift_experiment/ # 連合ドリフト実験パッケージ
│   ├── config.py                # ★ハイパーパラメータの一元管理
│   ├── data.py                  # 合成データ生成・ドリフトスケジュール
│   ├── models.py                # SimpleMLP(2次元入力の二値分類)
│   ├── adwin.py                 # FullScanADWIN(全分割点走査のADWIN)
│   ├── hddm.py                  # HDDM-A/W（平均・EWMAの有界損失検出）
│   ├── e_detector.py            # bounded mean向け混合e-SR検出器
│   ├── clients/                 # 共通基底とFedSDA検出器別 / FedDriftクライアント
│   ├── servers/                 # Base / 共通クラスタリング / FedSDA / FedDrift サーバ
│   ├── server.py                # 旧サーバimportパスの互換窓口
│   ├── experiment.py            # 実験本体 run_random_drift_experiment
│   ├── metrics.py               # 検出性能メトリクス(TP/FP/FN, 遅延など)
│   ├── plotting.py              # 可視化(保存 or 表示)
│   └── trials.py                # 複数試行の実行・集計
├── docs/                        # ドキュメント
│   ├── fedsda-algorithm.md      # FedSDA 実装仕様書
│   ├── differences-from-feddrift.md # FedDrift 元論文との相違点まとめ
│   ├── drift-detection.md       # ADWIN/e-detector の理論・実装・比較条件
│   ├── hyperparameters.md       # 全変数の意味・使用手法・仕様の一覧
│   └── sequence-diagrams.md     # FedSDA/FedDrift の処理フロー(mermaid)
├── results/                     # 実験成果物(results_<実行時刻>/ 単位。.gitignore 済み)
└── main_jp.tex                  # 論文原稿(LaTeX)
```

> 参照用に `microsoft/FedDrift` を `FedDrift/` へ clone している場合、それは
> `.gitignore` によりこのリポジトリの追跡対象外となる。

## ハイパーパラメータ

全ハイパーパラメータは [federated_drift_experiment/config.py](federated_drift_experiment/config.py) に一元管理されている
(論文の記号 K, R, τ, L, E_init, δ_adwin, N_FIFO, γ_dist との対応もコメントに記載)。
**各変数の意味・使用手法(FedSDA / FedDrift / 共通)・仕様は
[docs/hyperparameters.md](docs/hyperparameters.md) に一覧化している。**

既定値の概要:

- 実験規模: クライアント数 10、総データ数 5000/クライアント、1ラウンド = 50 ステップ
- データ: コンセプト数 4(ガウス2塊×ラベル反転、同心円×ラベル反転)
- ドリフト: 安定期間300サンプル経過後、毎サンプル確率0.0015で別コンセプトへ遷移
- 検出: FedSDA=ADWIN δ=0.05・FIFOバッファ長 30・距離閾値 γ_dist=0.1、FedDrift=検出バッチ 50・通信ラウンド R=1

変更方法:

```python
# ファイルを直接編集するか、コードから上書きする(各モジュールは実行時に参照)
from federated_drift_experiment import config
config.TOTAL_DATA_POINTS = 300   # 例: 縮小実験
config.ADWIN_DELTA = 0.01        # 例: 検出感度の変更
```
