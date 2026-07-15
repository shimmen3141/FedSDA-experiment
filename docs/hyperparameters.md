# ハイパーパラメータ・変数リファレンス

本実装([FedSDA/config.py](../FedSDA/config.py))の全ハイパーパラメータを、**意味・使用手法・既定値**の観点で一覧化する。実行時はコードから `from FedSDA import config; config.X = ...`で上書きできる(各モジュールは呼び出し時に `config.X` を参照する)。

論文(FedDrift 元論文)との対応・相違は [differences-from-feddrift.md](differences-from-feddrift.md)、アルゴリズム詳細は [fedsda-algorithm.md](fedsda-algorithm.md) を参照。

---

## 0. 前提: FedSDA と FedDrift の処理モデルの違い

両手法は「1 ラウンド」の意味が構造的に異なる。これを押さえると各変数の役割が明確になる。

| | **FedSDA**(提案) | **FedDrift**(ベースライン) |
|---|---|---|
| 処理単位 | **per-sample 逐次**(オンライン単一パス) | **バッチ**(サンプルを溜めてから処理) |
| ドリフト検出 | ADWIN(サンプルごとに統計検定) | 損失増分(検出バッチ完了ごと) |
| 1ラウンドで処理するサンプル数 | `AGG_INTERVAL` | `FEDDRIFT_DETECT_BATCH`(=検出=通信の単位) |
| 1 ラウンドの学習量 | `AGG_INTERVAL` × `UPDATES_PER_SAMPLE` 更新 | `FEDDRIFT_DETECT_BATCH` × `UPDATES_PER_SAMPLE` 更新 |
| 集約(通信)の契機 | 毎ラウンド末(= `AGG_INTERVAL` サンプルごと) | 検出バッチ完了時のみ(= `FEDDRIFT_DETECT_BATCH` ごと) |
| バッチあたり反復 | なし(各サンプル1回) | `FEDDRIFT_ROUNDS` 回(論文 R。既定 1) |

**重要な不変量**: 1 モデルあたりの総ローカル更新数は、FedSDA・FedDrift(R=1)とも `TOTAL_DATA_POINTS × UPDATES_PER_SAMPLE` で**一定**(`AGG_INTERVAL` / `FEDDRIFT_DETECT_BATCH` の値に依存しない)。この予算一致が両手法の公平比較の土台になっている。`FEDDRIFT_ROUNDS` を 1 より大きくすると FedDrift 側だけ更新数・通信量が R 倍になり、この一致は崩れる(論文忠実な FedDrift 再現用)。

論文記号との対応:

| 論文記号 | 意味 | 本実装 |
|---|---|---|
| K | 1 ラウンドのローカル学習ステップ数 | 処理サンプル数 × `UPDATES_PER_SAMPLE`(FedSDA: `AGG_INTERVAL`、FedDrift: `FEDDRIFT_DETECT_BATCH`) |
| R | 1 時刻の通信ラウンド数 | `FEDDRIFT_ROUNDS`(FedDrift のみ、既定 1)。FedSDA は対応物なし |
| B | ミニバッチサイズ | `CLIENT_BATCH_SIZE` |
| L | 1 データ点あたり更新回数 | `UPDATES_PER_SAMPLE`(両手法共通) |
| η | 学習率 | `BASE_LR` / `NEW_MODEL_LR` |
| δ_adwin | ADWIN 信頼度 | `ADWIN_DELTA`(FedSDA) |
| γ_dist | モデル適合/マージ距離閾値 | `DISTANCE_THRESHOLD` |

---

## 1. 実験規模・ラウンド構造

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `N_CLIENTS` | クライアント数 C | 共通 | 10 |
| `TOTAL_DATA_POINTS` | クライアントあたり総データ数(単一パス) | 共通 | 5000 |
| `AGG_INTERVAL` | **FedSDA/Oblivious**: 集約までのサンプル数(=1 ラウンド長。集約間隔でもある) | FedSDA / Oblivious | 50 |
| `LOCAL_UPDATE_TAU` | ローカル更新間隔 τ(論文の「t mod τ = 0」)。τ サンプルごとに τ×`UPDATES_PER_SAMPLE` 回まとめて更新(総更新回数は不変)。1=毎サンプル(v1 挙動)。v1/v2 比較の掃引軸 | FedSDA / Oblivious | 1 |

---

## 2. データセット

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `DATASET` | 使用データ `blobs`(既定・独自2D)/ `sea` / `circle` / `sine`(後3つは FedDrift 由来) | 共通 | `blobs` |
| `SEA_THRESHOLDS` | SEA-4 各概念の閾値 θ(論文 appendix 準拠 {9,8,7,9.5}) | sea のみ | `{0:9,1:8,2:7,3:9.5}` |
| `SEA_LABEL_NOISE` | SEA の内在ラベルノイズ率(標準 10%) | sea のみ | 0.10 |
| `CIRCLE_PARAMS` | CIRCLE-2 各概念の円 (cx,cy,r) | circle のみ | `{0:(0.2,0.5,0.15),1:(0.6,0.5,0.25)}` |

補助関数: `input_dim()` = 特徴次元、`num_concepts()` = 概念数(データセットに追従)。

---

## 3. ドリフトスケジュール(データ生成)

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `MIN_STABLE_PERIOD` | ドリフト後の最小安定期間(サンプル数)。次ドリフトまでの下限 | 共通 | 300 |
| `DRIFT_PROB` | 安定期間経過後、1 サンプルごとのドリフト発生確率 | 共通 | 0.0015 |

> 本実装のドリフトは per-sample ランダム位置で発生(論文の時刻境界固定とは異なる)。
> `STABLE_WINDOW` はこの `MIN_STABLE_PERIOD` 未満に取ること(定常精度が次ドリフトを跨がない)。

---

## 4. モデル・最適化(共通)

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `OPTIMIZER` | 最適化器 `adam`(FedDrift 準拠)/ `sgd` | 共通 | `adam` |
| `BASE_LR` | 通常学習の学習率 η(合成データは論文 1e-2) | 共通 | 0.01 |
| `NEW_MODEL_LR` | 新規モデル初期学習時の学習率 | 共通 | 0.01 |
| `WEIGHT_DECAY` | Adam の weight_decay(論文設定) | 共通 | 1e-3 |
| `AMSGRAD` | Adam の amsgrad(論文設定) | 共通 | True |
| `NEW_MODEL_EPOCHS` | 新規モデル作成時の初期学習エポック数 (E_init) | 共通 | 30 |
| `CLIENT_BATCH_SIZE` | ローカル更新のミニバッチサイズ B | 共通 | 32 |
| `UPDATES_PER_SAMPLE` | 1 データ点あたりの勾配更新回数 L(学習強度)。両手法共通=公平比較の予算なので分けない | 共通 | 1 |
| `PRETRAIN_SAMPLES` / `PRETRAIN_EPOCHS` / `PRETRAIN_BATCH_SIZE` | 初期モデル(モデル0)の事前学習設定 | 共通 | 500 / 10 / 32 |

---

## 5. ドリフト検出 — FedSDA(ADWIN)

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `ADWIN_DELTA` | ADWIN 信頼度パラメータ δ_adwin(小さいほど検出保守的) | FedSDA | 0.05 |
| `ADWIN_MAX_WINDOW` | ADWIN ウィンドウ幅の上限 | FedSDA | 1000 |
| `ADWIN_MIN_WIDTH` | 検定を開始する最小ウィンドウ幅 | FedSDA | 10 |

---

## 6. ドリフト解決・データ管理 — FedSDA(一部共通)

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `DISTANCE_THRESHOLD` | モデル適合判定の距離閾値 γ_dist(サーバのマージ判定と共用) | FedSDA / サーバ | 0.1 |
| `FIFO_BUFFER_SIZE` | FIFO 遅延バッファ長 N_FIFO(検知遅延中の混合防止) | FedSDA | 30 |
| `MIN_DRIFT_DATA` | ドリフト解決に必要な新概念データの最小数 | FedSDA | 5 |
| `STORED_DATA_LIMIT` | モデルごとの評価用データストア上限 | 共通(クライアント) | 50 |
| `EVAL_STORE_SAMPLE_SIZE` | 評価用ストアへ 1 回に追加するサンプル数上限 | 共通(クライアント) | 20 |
| `EVAL_MAX_SAMPLES` | サーバ評価依頼時に使う最大サンプル数 | 共通(クライアント) | 50 |

---

## 7. FedDrift ベースライン

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `FEDDRIFT_DETECT_BATCH` | 検出バッチサイズ ＝ **1ラウンドで処理するサンプル数**(論文の時刻粒度 500 を独立化)。検出粒度・集約(通信)間隔・1ラウンドの学習量(`× UPDATES_PER_SAMPLE`)を兼ねる | FedDrift / FedDrift_v2 | 50 |
| `FEDDRIFT_ROUNDS` | 1 検出バッチあたりの通信ラウンド数(論文 R)。v2 は {ローカル学習 → 集約 → 配布} を正確に R 回実行する。**既定 1 は FedSDA と予算一致の公平比較用** | FedDrift / FedDrift_v2 | 1 |
| `FEDDRIFT_ISOLATION_TIMESTEPS` | 新規モデルをクロス評価・マージから外す時刻数 W。1なら作成時刻だけ隔離し、次時刻から対象。参照実装の既定構成に対応 | FedDrift_v2 | 1 |

> `FEDDRIFT_DETECT_BATCH`(検出粒度 ↔ 通信)と `FEDDRIFT_ROUNDS`(バッチあたり収束度 ↔ 通信)は
> **直交する 2 つの通信軸**。前者は「どの粒度で検出・通信するか」、後者は「1 バッチをどれだけ学習し切るか」。パレート分析では独立に掃引できる。

---

## 8. サーバ(クラスタリング・マージ)

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `CROSS_EVAL_MAX_CLIENTS` | クロス評価で 1 モデルあたりに使うクライアント数上限 | サーバ | 3 |
| `CLUSTER_MIN_EVAL_N` | マージ判定に必要な評価サンプルの最小数 | サーバ | 5 |
| `CLUSTER_LINKAGE` | 共通クラスタリング戦略。`complete`=FedDrift論文のmax-linkage、`connected`=閾値グラフの連結成分(single-linkage cut相当) | FedDrift_v2（他のクラスタリング手法にも再利用可能） | `complete` |

> サーバは生データを集めず、配布モデルを現地評価させて**集約統計量 (n, Σℓ, Σℓ²) のみ**を
> 集める federated 設計(詳細は DIFFERENCES §5)。`DISTANCE_THRESHOLD` をマージ判定に共用。

> **v1/v2 の切替**: サーバ処理順序の v2(FedAvg先行・加重平均マージ・配布1回)は config ノブ
> ではなく**モード `FedSDA_v2`**(`ClusteringServerV2`)で選択する。τ(`LOCAL_UPDATE_TAU`)と
> 直交しており、{`FedSDA`, `FedSDA_v2`} × {τ=1, τ>1} の4構成でアブレーションできる。
> 詳細は [sequence-diagrams.md](sequence-diagrams.md)。

---

## 9. 評価メトリクス・可視化

| 変数 | 意味 | 使用 | 既定 |
|---|---|---|---|
| `DELAY_TOLERANCE` | 真のドリフトと検出をマッチングする許容遅延(サンプル数) | 評価 | 100 |
| `STABLE_WINDOW` | 定常精度 `stable_accuracy` の回復除外窓 W。各真ドリフト直後 W サンプルを prequential 平均から除外。`MIN_STABLE_PERIOD` 未満・最も遅い回復のプラトーを越える大きめ固定値 | 評価 | 200 |
| `PLOT_SMOOTH_WINDOW` | 精度曲線の移動平均ウィンドウ | 可視化 | 50 |

### 結果 dict の主な指標

| キー | 意味 |
|---|---|
| `accuracy` | prequential(逐次)精度 = **全期間の総合精度**(予測→即学習の当否平均) |
| `stable_accuracy` | **定常精度** = 回復窓 W を除外した prequential(回復曲線 acc(Δ) の Δ≥W の裾) |
| `recall` / `precision` / `f1` | ドリフト検出の質(ローカル切替を検出とみなし真ドリフトと照合) |
| `avg_delay` | 平均検出遅延(サンプル数) |
| `final_model_count` | 最終モデル数(集約あり)/ クライアント平均保持数(なし) |
| `comm_upload` / `comm_download` / `comm_total` | 通信量(モデル転送数) |
| `control_upload` / `control_download` / `control_total` | 割当通知・クロス評価依頼・評価統計・ID割当などの軽量制御メッセージ数 |

適応の**速さ**(回復曲線 acc(Δ)・`T90` 等)は [recovery_analysis.py](../recovery_analysis.py) で別途評価する。
