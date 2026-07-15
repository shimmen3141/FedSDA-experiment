# FedDrift 元論文との相違点

本実装の **FedDrift ベースライン**と、FedDrift 元論文（Jothimurugesan et al., 2023）および公開実装 `microsoft/FedDrift` との相違点を一元的にまとめる。照合元は `FedDrift_paper/`（論文 tex、特に `appendix-expt-setup.tex`）と `FedDrift/`（clone した参照コード）。

> このファイルは **論文/参照の FedDrift ↔ 本実装の FedDrift** の対応に絞る。提案手法 FedSDA の
> 詳細は [fedsda-algorithm.md](fedsda-algorithm.md)、全変数の意味・手法間の使い分けは
> [hyperparameters.md](hyperparameters.md) を参照。FedSDA は「なぜ FedDrift をこう
> 実装したか」を説明する範囲でのみ登場する。

区分:
- ✅ **忠実**（論文・参照コードと一致することを確認済み）
- ⚠️ **意図的な相違**（本研究の実験設計による）
- ❌ **未実装**（本研究のスコープ外）

---

## 1. データセット定義

FedDrift 由来の 3 データセット（sine / circle / sea）の**生成規則は論文・参照コードに忠実**。実装は `FedSDA/data.py`。

| dataset | 生成規則 | 参照との照合 | 判定 |
|---|---|---|---|
| **sine** | x₁,x₂~U[0,1]²、概念0: y=1 iff x₂≤sin(x₁)、概念1: 反転 | 参照 `sine/data_loader.py::generate_sine_sample` と完全一致 | ✅ |
| **circle** | x₁,x₂~U[0,1]²、円 (0.2,0.5,0.15)/(0.6,0.5,0.25) の**外側=1**（z>0） | 参照 `circle/data_loader.py::generate_circle_sample` と z 式・極性・中心/半径すべて一致 | ✅ |
| **sea** | x₁,x₂,x₃~U[0,10]（x₃ ノイズ特徴）、y=1 iff x₁+x₂≤θ、θ={9,8,7,9.5}、10% ラベルノイズ | 論文 appendix の定義と一致 | ✅ |
| **blobs**（既定） | 独自の 2 次元合成（ガウス2塊×ラベル反転 / 同心円×ラベル反転） | **論文に存在しない独自データ** | ⚠️ |

### 注記

- **sea の閾値**: 参照コードは同梱 CSV（`concept1..4.csv`）を読むが、`concept4.csv` は実測 θ≈9.0 で別概念と潰れるズレがある。本実装は**論文 appendix の正準定義 θ_D=9.5** を採用し、4 概念が区別されるようにしている（`config.SEA_THRESHOLDS`）。信号は論文どおり x₁+x₂（x₃ はノイズ）。
- **ノイズ**: ラベルノイズ 10% は論文どおり **sea のみ**。sine / circle にはノイズなし。
- **blobs**: FedDrift には無い独自データ。概念差が劇的（ラベル反転＋共変量シフト）で、ドリフトが最も検出しやすいケースとして用いている。論文の CIRCLE/SINE とは別物。
- ❌ **MNIST-2/4・FMoW** は未実装（合成データの sine/circle/sea と独自 blobs のみ）。

---

## 2. 実験セットアップ（⚠️ 意図的な相違）

データ定義は忠実だが、ドリフトの**出し方・処理粒度**が論文と異なる。本実装が連続ストリームを扱うための設計判断であり、バグではない。

| 項目 | 論文 / FedDrift | 本実装 | 実装箇所 |
|---|---|---|---|
| ドリフト系列 | 10 時刻 × 500 サンプル/クライアント、**固定 staggered パターン**、切替は時刻境界のみ | `MIN_STABLE_PERIOD`(300) 経過後、毎サンプル確率 `DRIFT_PROB`(0.0015) で**ランダム**切替、連続ストリーム `TOTAL_DATA_POINTS`(5000)/クライアント | `data.py::make_concept_schedules` |
| 切替回数 | 2 概念は原則 **1 回**（default→第2概念） | **再帰的に複数回**（2 概念なら 0↔1 を反復） | 同上 |
| 検出単位 | 時刻（500 サンプル）単位で検出 | `FEDDRIFT_DETECT_BATCH`(既定 50) 件ごと。完了時にのみ集約するため通信間隔も兼ねる（§3） | `clients/feddrift.py` |
| 学習の反復 | 固定 500 サンプルに R=100 ラウンド（≒反復エポック） | 既定は単一パス相当（`FEDDRIFT_ROUNDS`=1）。R を上げれば論文どおりの反復学習も可（§3） | `experiment.py` |

この「切替回数」の相違は recovery 分析に直結する: 本実装は 1 ランで**多数のドリフトイベント**が生じ、それを Δ=0 起点に集計する（論文は原則 1 回）。

---

## 3. 学習ハイパーパラメータ

論文 `appendix-expt-setup.tex` の Table（training-params）と本実装の **FedDrift** の対応。

| 記号 | 説明 | 論文（合成データ） | 本実装（FedDrift） | 判定 |
|---|---|---|---|---|
| K | 1 ラウンドあたりのローカル学習ステップ数 | 50 | `FEDDRIFT_DETECT_BATCH × UPDATES_PER_SAMPLE`（既定 50×1） | ✅ |
| η | 学習率（SINE/CIRCLE/SEA） | 10⁻² | `BASE_LR=0.01` | ✅ |
| — | Adam weight_decay / amsgrad | 10⁻³ / True | `WEIGHT_DECAY=1e-3` / `AMSGRAD=True` | ✅ |
| B | ミニバッチサイズ | 50 | `CLIENT_BATCH_SIZE=32` | ⚠️ |
| L | 1 データ点あたりのローカル更新回数 | （明記なし） | `UPDATES_PER_SAMPLE=1`（両手法共通） | — |
| （検出/時刻粒度） | 1 時刻あたりのデータ数＝検出単位 | 500 | `FEDDRIFT_DETECT_BATCH=50`（＝1ラウンドの処理サンプル数） | ⚠️ |
| R | 1 時刻あたりの通信ラウンド数 | 100 | `FEDDRIFT_ROUNDS=1`（既定。下記） | ⚠️ |

### 補足: 検出粒度・学習量・R の扱い

- **検出単位 ＝ `FEDDRIFT_DETECT_BATCH`（＝1ラウンドで処理するサンプル数）**。元 FedDrift は**時刻(500 サンプル)単位で検出**する。これを独立パラメータ化し既定 50 とした（概念は論文の時刻粒度そのもの、値のみ相違）。主ループはこの件数を 1 ラウンドで処理し、溜まるたびに検出+割り当て+**通信**を行う（＝**集約(通信)間隔も兼ねる**。掃引で大きくすると通信量が反比例して減る＝精度–通信量トレードオフの軸）。
- **1 ラウンドの学習量（論文 K）＝ `FEDDRIFT_DETECT_BATCH × UPDATES_PER_SAMPLE`**。バッチの各データ点に対し FedSDA と同じ `UPDATES_PER_SAMPLE` 回の更新を行うのと同予算。これにより**総ローカル更新数は `FEDDRIFT_DETECT_BATCH` に依らず `TOTAL_DATA_POINTS × UPDATES_PER_SAMPLE`（R=1 なら FedSDA と一致）**となり、検出バッチを掃引しても学習量が変わらない（通信だけを変数にできる）。
- **R ＝ `FEDDRIFT_ROUNDS`**。論文はバッチを R=100 ラウンド収束まで反復学習する。本実装は**既定 R=1**で、これは比較対象の FedSDA が**オンライン単一パス**（各データ点 1 回）で、その学習・通信予算と揃えるため。`FEDDRIFT_ROUNDS=100` にすれば論文忠実な FedDrift（バッチ収束学習）を再現できるが、総更新数・総通信量が R 倍になり FedSDA との予算一致は崩れる。R は `FEDDRIFT_DETECT_BATCH`（検出粒度↔通信）と**直交する第2の通信軸**（バッチあたり収束度↔通信）。

FedSDA 側の対応変数（`AGG_INTERVAL` 等）や手法間の使い分けは [hyperparameters.md](hyperparameters.md) を参照。

---

## 4. 評価指標

| 項目 | 論文 | 本実装 |
|---|---|---|
| 主指標 | 各時刻 τ の学習後、τ+1 の（次概念）held-out で test accuracy（ドリフト時刻除外） | prequential `accuracy`（全期間の総合精度）+ `stable_accuracy`（定常精度: 真ドリフト直後 `STABLE_WINDOW`=W サンプルを除外した prequential）+ 回復曲線 acc(Δ)（適応速度）の3層。論文流の「次時刻 held-out・ブロック評価」は per-sample ランダムドリフト設定に噛み合わない（＆ prequential と役割が重複）ため標準実行では用いない |
| 検出閾値 | 損失/距離の閾値で検出 | `DISTANCE_THRESHOLD`（損失増分の閾値）。`run_pareto_sweep.py` で掃引し感度を提示 |

**実装忠実性の検証**: 論文式（各時刻の学習後に次時刻コンセプトの held-out で評価・ドリフト時刻除外）で本実装の FedDrift 再実装は論文値（SINE ≈97.4%）にほぼ一致することを確認済み。厳密な論文再現が必要な場合は、境界ドリフト・固定バッチの専用設（`FEDDRIFT_ROUNDS`=100 等）で別途評価する。

回復除外窓 W（`STABLE_WINDOW`=200）は最も遅い回復のプラトーを越える大きめ固定値で、
`MIN_STABLE_PERIOD` 未満（次ドリフトを跨がない）。`recovery_analysis.py` の回復窓 window と同値。

---

## 5. 実装上の設計判断

- **サーバ側モデル間距離の評価**: FedDrift 公開実装は aggregator が全クライアントの生データ（`all_data`）を中央に集めて距離を計算する**シミュレーション近道**を取る。本実装はこれに倣わず、評価対象モデルをデータ保持クライアントへ配布して現地評価させ、**集約統計量 (n, Σℓ, Σℓ²) のみ**を返す federated な形にしている（サブサンプリング `CROSS_EVAL_MAX_CLIENTS` / `EVAL_MAX_SAMPLES` による近似）。詳細と根拠は [fedsda-algorithm.md](fedsda-algorithm.md) §3 を参照。
- **FedDrift は再実装**であり、参照コードをそのまま呼んでいるわけではない。sea も CSV を読まず生成器で論文定義を再現している。

## FedDrift v1 / v2 の位置付け

- モード `FedDrift` は既存結果の再現用に従来フローを維持する。検出時のクラスタリング付き集約1回と、
  その後の `FEDDRIFT_ROUNDS` 回を実行し、クラスタリングは閾値グラフの連結成分を使う。
- モード `FedDrift_v2` は監査結果を反映した論文準拠フローである。新規モデルを
  `FEDDRIFT_ISOLATION_TIMESTEPS` 時刻だけ隔離し、FedAvgは正確に `FEDDRIFT_ROUNDS` 回だけ実行する。
- v2のクラスタリングは共通設定 `CLUSTER_LINKAGE` で `complete`（論文のmax-linkage、既定）または
  `connected`（従来方式）を選べる。原手法との比較値には `complete` を使い、`connected` は
  クラスタリング方式のアブレーションとして報告する。
- v2のクロス評価は配布済みモデルのクライアントキャッシュを再利用する。モデルパラメータ通信は
  `comm_*`、評価依頼・統計返送などは `control_*` へ分離して数える。
