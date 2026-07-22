"""実験ハイパーパラメータの一元管理。

全モジュールはここから値を参照する。実験時にコードから上書きする場合は
`from federated_drift_experiment import config; config.TOTAL_DATA_POINTS = 300` のように
モジュール属性を書き換えればよい(各モジュールは呼び出し時に参照する)。

各変数の意味・使用手法・設計上の役割は docs/hyperparameters.md に一覧化している。
括弧内は論文 (main_jp.tex / docs/fedsda-algorithm.md) の記号との対応。
"""
from .data.specs import DATASET_SPECS, get_dataset_spec

# ##########################################
# 共通パラメータ (FedSDA / FedDrift / Oblivious)
# ##########################################

# ==========================================
# 実験規模
# ==========================================
N_CLIENTS = 10              # クライアント数 C
TOTAL_DATA_POINTS = 5000    # クライアントあたりの総データ数(FedDrift: 10時刻 × 500 に合わせる)

# ==========================================
# データセット
# ==========================================
DATASET = 'blobs'          # 利用可能な名前は data/specs.py を参照
CONCEPT_SCHEDULE = 'random'  # 'random' / 'feddrift_fixed'（データ分布と独立）
CONCEPT_SCHEDULES = ('random', 'feddrift_fixed')

# 各データセット(SEA-4/CIRCLE-2/SINE-2)の生成規則は data.py と docs/differences-from-feddrift.md §1 を参照。
SEA_THRESHOLDS = {0: 9.0, 1: 8.0, 2: 7.0, 3: 9.5}   # 各概念の閾値 θ(論文 appendix 準拠。concept4=9.5 で4概念を区別)
SEA_LABEL_NOISE = 0.10     # 各概念に内在するラベルノイズ率(SEA標準10%)
CIRCLE_PARAMS = {0: (0.2, 0.5, 0.15), 1: (0.6, 0.5, 0.25)}   # CIRCLE-2 各概念の円 (cx, cy, r)

_FEATURE_DIMS = {name: spec.input_dim for name, spec in DATASET_SPECS.items()}
_DATASET_CONCEPTS = {name: spec.num_concepts for name, spec in DATASET_SPECS.items()}


def input_dim(dataset=None):
    """現在のデータセットの入力特徴次元を返す(モデルの入力層サイズに使用)。"""
    return _FEATURE_DIMS[dataset if dataset is not None else DATASET]


def num_concepts(dataset=None):
    """現在のデータセットのコンセプト数を返す。"""
    return _DATASET_CONCEPTS[dataset if dataset is not None else DATASET]


def num_classes(dataset=None):
    """データセットのクラス数を返す。"""
    return get_dataset_spec(dataset if dataset is not None else DATASET).num_classes


def dataset_spec(dataset=None):
    """データセット固有のモデル・系列設定を返す。"""
    return get_dataset_spec(dataset if dataset is not None else DATASET)


# ==========================================
# ドリフト生成
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
NEW_MODEL_TRAINING = "fixed"  # fixed / none / early_stopping
NEW_MODEL_EARLY_STOPPING_PATIENCE = 3
NEW_MODEL_EARLY_STOPPING_MIN_DELTA = 1e-4
NEW_MODEL_VALIDATION_FRACTION = 0.2
CLIENT_BATCH_SIZE = 32      # ローカル更新のミニバッチサイズ (論文 B)
UPDATES_PER_SAMPLE = 1      # 1データ点あたりの勾配更新回数 (論文 L・学習強度)。両手法共通=公平比較の予算

# 初期モデル(モデル0)の事前学習
PRETRAIN_SAMPLES = 500
PRETRAIN_EPOCHS = 10
PRETRAIN_BATCH_SIZE = 32

# ==========================================
# データ管理(クライアント)
# ==========================================
STORED_DATA_LIMIT = 50      # モデルごとの評価用データストア上限
EVAL_STORE_SAMPLE_SIZE = 20 # 評価用ストアへ1回に追加するサンプル数上限
EVAL_MAX_SAMPLES = 50       # サーバ評価依頼時に使う最大サンプル数

# ==========================================
# 評価メトリクス・可視化
# ==========================================
DELAY_TOLERANCE = 100       # 真のドリフトと検出をマッチングする許容遅延(サンプル数)
PLOT_SMOOTH_WINDOW = 50     # 精度曲線の移動平均ウィンドウ
STABLE_WINDOW = 200         # 定常精度 stable_accuracy の回復除外窓 W(各ドリフト直後 W サンプルを除外)。MIN_STABLE_PERIOD 未満


# ##########################################
# 手法固有パラメータ
# ##########################################

# ==========================================
# 逐次処理 (FedSDA / Oblivious)
# ==========================================
AGG_INTERVAL = 50           # 1ラウンドで処理するサンプル数(=集約間隔)。FedDrift は FEDDRIFT_DETECT_BATCH を使う
# ローカル更新間隔 τ(論文の「t mod τ = 0」)。τ サンプルごとに τ×UPDATES_PER_SAMPLE 回まとめて
# 更新する(総更新回数は不変)。1 = 毎サンプル更新。
LOCAL_UPDATE_TAU = 1

# ==========================================
# FedSDA: ドリフト検出と解決
# ==========================================
ADWIN_DELTA = 0.05          # ADWIN 信頼度パラメータ (delta_adwin)
ADWIN_MAX_WINDOW = 1000     # ADWIN ウィンドウ幅の上限
ADWIN_MIN_WIDTH = 10        # 検定を開始する最小ウィンドウ幅
FEDSDA_ENABLE_FORCED_DRIFT_CHECK = True  # ADWIN未検知時の保険的な損失増分チェック

E_DETECTOR_ALPHA = 0.001    # e-SRの誤警報制御値。平均誤警報間隔(ARL)を1/alpha以上にする
HDDM_DRIFT_CONFIDENCE = 0.001    # HDDM-A/Wのドリフト判定信頼度
HDDM_WARNING_CONFIDENCE = 0.005  # HDDM-A/Wの警告判定信頼度（記録用）
HDDM_W_LAMBDA = 0.05             # HDDM-WのEWMAで新しい損失へ与える重み

FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS = 1  # 新規モデルを作成してからアップロード可能になるまでの学習ラウンド数
FIFO_BUFFER_SIZE = 30       # FIFO遅延バッファ長 (N_FIFO)
RECENT_ASSIGNMENT_JOURNAL_SIZE = 0  # 0で無効。検証時は100を指定
MIN_DRIFT_DATA = 5          # ドリフト解決に必要な新概念データの最小数

# ==========================================
# FedDrift ベースライン
# ==========================================
# 検出バッチサイズ=FedDrift の1ラウンドで処理するサンプル数(検出粒度・集約間隔・学習量を兼ねる)。詳細は docs/hyperparameters.md §7
FEDDRIFT_DETECT_BATCH = 50
# 1検出バッチあたりの通信ラウンド数(論文 R)。既定1=FedSDAと予算一致。R>1で論文忠実だが更新・通信が R倍
FEDDRIFT_ROUNDS = 1
# 新規モデルをクロス評価・マージ対象から外す時刻数 W。参照実装の FedDrift は W=1。
FEDDRIFT_ISOLATION_TIMESTEPS = 1

# ==========================================
# クラスタリング (FedSDA / FedDrift)
# ==========================================
DISTANCE_THRESHOLD = 0.1    # モデル適合/マージ判定の距離閾値 (gamma_dist)。FedDrift の検出閾値にも流用
CROSS_EVAL_MAX_CLIENTS = 3  # クロス評価で1モデルあたりに使うクライアント数上限
CLUSTER_MIN_EVAL_N = 5      # マージ判定に必要な評価サンプルの最小数
# FedDriftなど、方式を明示して使うサーバの共通クラスタリング戦略。
# 'complete'=論文の max-linkage、'connected'=従来の閾値グラフ連結成分。
CLUSTER_LINKAGE = 'complete'
