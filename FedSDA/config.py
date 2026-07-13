"""実験ハイパーパラメータの一元管理。

全モジュールはここから値を参照する。実験時にコードから上書きする場合は
`from FedSDA import config; config.TOTAL_DATA_POINTS = 300` のように
モジュール属性を書き換えればよい(各モジュールは呼び出し時に参照する)。

括弧内は論文 (main_jp.tex / ALGORITHM.md) の記号との対応。
"""

# ==========================================
# 実験規模
# ==========================================
N_CLIENTS = 10              # クライアント数 C
# 論文 R(=固定バッチを収束まで反復学習するラウンド数)とは別概念。本実装はストリームを
# 単一パスするため、R_ROUNDS はストリームを「時刻」に区切る粒度(data_per_time =
# R_ROUNDS × K_STEPS)を決めるだけ。総集約・総学習は R_ROUNDS に依らず一定で、変わるのは
# paper_accuracy の評価間隔のみ(実質、汎用化のための残置ノブ)。詳細は DIFFERENCES_FROM_FEDDRIFT.md §3。
R_ROUNDS = 1                # 1タイムステップあたりの通信ラウンド数
# 論文 K =「1ラウンド・1モデルあたりのローカル学習ステップ数」に対応(全手法共通の学習量)。
# FedSDA: 1更新/サンプル × K_STEPS、FedDrift: local_train(k_steps=K_STEPS)。
# 集約(通信)間隔としても使うのは FedSDA / Oblivious のみ(K_STEPS サンプルごとに集約)。
# FedDrift の集約間隔は FEDDRIFT_DETECT_BATCH 側(検出バッチ完了時に集約)なので混同しないこと。
K_STEPS = 50                # 1ラウンドあたりのローカル学習ステップ数 (論文 K)
TOTAL_DATA_POINTS = 5000    # クライアントあたりの総データ数(FedDrift: 10時刻 × 500 に合わせる)

# ==========================================
# データセット
# ==========================================
DATASET = 'blobs'          # 'blobs'(2次元合成) / 'sea' / 'circle' / 'sine'(いずれもFedDrift)

# SEA-4 (FedDrift論文 appendix の定義): 特徴 x1,x2,x3 ~ U[0,10]。x3 はノイズ特徴で、
# label = 1 iff (x1 + x2) <= 閾値。閾値は概念 A,B,C,D で {9, 8, 7, 9.5}。
# 注: FedDrift 同梱の concept4.csv は実測 θ≈9.0 で concept2 と重複する(データ配布側のズレ)
#     が、論文の正準定義に合わせ θ_D=9.5 として4概念を区別する。
SEA_THRESHOLDS = {0: 9.0, 1: 8.0, 2: 7.0, 3: 9.5}
SEA_LABEL_NOISE = 0.10     # 各概念に内在するラベルノイズ率(SEAベンチマークの標準10%)

# CIRCLE-2 (FedDrift): 特徴 x1,x2 ~ U[0,1]^2。概念ごとに異なる円の内外でラベル。
# label = 1 iff (x1-cx)^2 + (x2-cy)^2 > r^2(円の外側)。(cx, cy, r) は概念別。
CIRCLE_PARAMS = {0: (0.2, 0.5, 0.15), 1: (0.6, 0.5, 0.25)}

# SINE-2 (FedDrift): 特徴 x1,x2 ~ U[0,1]^2。
# 概念0: label = 1 iff x2 <= sin(x1)、概念1: そのラベルを反転。

_FEATURE_DIMS = {'blobs': 2, 'sea': 3, 'circle': 2, 'sine': 2}
_DATASET_CONCEPTS = {'blobs': 4, 'sea': 4, 'circle': 2, 'sine': 2}


def input_dim(dataset=None):
    """現在のデータセットの入力特徴次元を返す(モデルの入力層サイズに使用)。"""
    return _FEATURE_DIMS[dataset if dataset is not None else DATASET]


def num_concepts(dataset=None):
    """現在のデータセットのコンセプト数を返す。"""
    return _DATASET_CONCEPTS[dataset if dataset is not None else DATASET]


# ==========================================
# ドリフトスケジュール
# ==========================================
MIN_STABLE_PERIOD = 300     # ドリフト後の最小安定期間(サンプル数)
DRIFT_PROB = 0.0015         # 安定期間経過後、1サンプルごとのドリフト発生確率

# ==========================================
# モデル・最適化 (FedDrift論文: Adam lr 1e-2, weight_decay 1e-3, amsgrad=True)
# ==========================================
OPTIMIZER = 'adam'         # 'adam'(FedDrift準拠) or 'sgd'
BASE_LR = 0.01             # 通常学習の学習率(論文: 合成データは 1e-2)
NEW_MODEL_LR = 0.01        # 新規モデル初期学習時の学習率
WEIGHT_DECAY = 1e-3        # Adam の weight_decay(論文設定)
AMSGRAD = True             # Adam の amsgrad(論文設定)
NEW_MODEL_EPOCHS = 30       # 新規モデル作成時の初期学習エポック数 (E_init)
CLIENT_BATCH_SIZE = 32      # ローカル更新のミニバッチサイズ
UPDATES_PER_STEP = 1        # 1ステップあたりのローカル更新回数 (L)

# 初期モデル(モデル0)の事前学習
PRETRAIN_SAMPLES = 500
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 32

# ==========================================
# ドリフト検出 (ADWIN)
# ==========================================
ADWIN_DELTA = 0.05          # 信頼度パラメータ (delta_adwin)
ADWIN_MAX_WINDOW = 1000     # ウィンドウ幅の上限
ADWIN_MIN_WIDTH = 10        # 検定を開始する最小ウィンドウ幅

# ==========================================
# クライアント(ドリフト解決・データ管理)
# ==========================================
DISTANCE_THRESHOLD = 0.1    # モデル適合判定の距離閾値 (gamma_dist)。サーバのマージ判定と共用
FIFO_BUFFER_SIZE = 30       # FIFOバッファ長 (N_FIFO)
MIN_DRIFT_DATA = 5          # ドリフト解決に必要な新概念データの最小数
STORED_DATA_LIMIT = 50      # モデルごとの評価用データストア上限
EVAL_STORE_SAMPLE_SIZE = 20 # 評価用ストアへ1回に追加するサンプル数上限
EVAL_MAX_SAMPLES = 50       # サーバ評価依頼時に使う最大サンプル数

# ==========================================
# FedDrift ベースライン
# ==========================================
# FedDrift のドリフト検出バッチサイズ。概念としては元 FedDrift の「時刻単位検出」の
# 時刻粒度(論文では500サンプル/クライアント/時刻)そのものを、独立パラメータとして取り出し
# 既定50に変えたもの。FedDrift はこの件数を溜めるごとに検出+割り当てを行う。
# 重要: FedDrift はこのバッチ完了時にのみサーバ集約するため(experiment.py:72 の fired ゲート)、
# これが FedDrift の集約(通信)間隔も兼ねる。ローカル学習ステップ数 K_STEPS とは役割が別。
# 掃引でこれを大きくすると通信量が反比例して減る(精度–通信量トレードオフの軸)。
# 小 = 検出しやすいが偽陽性↑・モデル増↑・通信↑、大 = 平滑化されるがドリフト見逃し↑。
# FedSDA は per-sample 検出なので無関係。詳細は DIFFERENCES_FROM_FEDDRIFT.md §3。
FEDDRIFT_DETECT_BATCH = 50

# ==========================================
# サーバ(クラスタリング・マージ)
# ==========================================
CROSS_EVAL_MAX_CLIENTS = 3  # クロス評価で1モデルあたりに使うクライアント数上限
CLUSTER_MIN_EVAL_N = 5      # マージ判定に必要な評価サンプルの最小数

# ==========================================
# 評価メトリクス・可視化
# ==========================================
DELAY_TOLERANCE = 100       # 真のドリフトと検出をマッチングする許容遅延(サンプル数)
PLOT_SMOOTH_WINDOW = 50     # 精度曲線の移動平均ウィンドウ
# 論文式の精度: 各時刻の学習後、次時刻コンセプトの held-out データで評価(ドリフト時刻は除外)
PAPER_TEST_SAMPLES = 200    # 1(クライアント,時刻)あたりのテストサンプル数
