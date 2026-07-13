# FedDrift 元論文との相違点

本実装（FedSDA 実験コード）と FedDrift 元論文
（Jothimurugesan et al., 2023）および公開実装 `microsoft/FedDrift` との細かい相違点を
一元的にまとめる。照合元は `FedDrift_paper/`（論文 tex、特に
`appendix-expt-setup.tex`）と `FedDrift/`（clone した参照コード）。

区分:
- ✅ **忠実**（論文・参照コードと一致することを確認済み）
- ⚠️ **意図的な相違**（オンライン per-sample 設定など、本研究の設計判断による）
- ❌ **未実装**（本研究のスコープ外）

---

## 1. データセット定義

FedDrift 由来の 3 データセット（sine / circle / sea）の**生成規則は論文・参照コードに
忠実**。実装は `FedSDA/data.py`。

| dataset | 生成規則 | 参照との照合 | 判定 |
|---|---|---|---|
| **sine** | x₁,x₂~U[0,1]²、概念0: y=1 iff x₂≤sin(x₁)、概念1: 反転 | 参照 `sine/data_loader.py::generate_sine_sample` と完全一致 | ✅ |
| **circle** | x₁,x₂~U[0,1]²、円 (0.2,0.5,0.15)/(0.6,0.5,0.25) の**外側=1**（z>0） | 参照 `circle/data_loader.py::generate_circle_sample` と z 式・極性・中心/半径すべて一致 | ✅ |
| **sea** | x₁,x₂,x₃~U[0,10]（x₃ ノイズ特徴）、y=1 iff x₁+x₂≤θ、θ={9,8,7,9.5}、10% ラベルノイズ | 論文 appendix の定義と一致 | ✅ |
| **blobs**（既定） | 独自の 2 次元合成（ガウス2塊×ラベル反転 / 同心円×ラベル反転） | **論文に存在しない独自データ** | ⚠️ |

### 注記

- **sea の閾値**: 参照コードは同梱 CSV（`concept1..4.csv`）を読むが、`concept4.csv` は
  実測 θ≈9.0 で別概念と潰れるズレがある。本実装は**論文 appendix の正準定義
  θ_D=9.5** を採用し、4 概念が区別されるようにしている（`config.SEA_THRESHOLDS`）。
  信号は論文どおり x₁+x₂（x₃ はノイズ）。
- **ノイズ**: ラベルノイズ 10% は論文どおり **sea のみ**。sine / circle にはノイズなし。
- **blobs**: FedDrift には無い独自データ。概念差が劇的（ラベル反転＋共変量シフト）で、
  ドリフトが最も検出しやすいケースとして用いている。論文の CIRCLE/SINE とは別物。
- ❌ **MNIST-2/4・FMoW** は未実装（合成データの sine/circle/sea と独自 blobs のみ）。

---

## 2. 実験セットアップ（⚠️ 意図的な相違）

データ定義は忠実だが、ドリフトの**出し方・処理粒度**は論文と異なる。これは提案手法が
**オンライン per-sample 適応**を扱うための設計判断であり、バグではない。

| 項目 | 論文 / FedDrift | 本実装 | 実装箇所 |
|---|---|---|---|
| ドリフト系列 | 10 時刻 × 500 サンプル/クライアント、**固定 staggered パターン**、切替は時刻境界のみ | `MIN_STABLE_PERIOD`(300) 経過後、毎サンプル確率 `DRIFT_PROB`(0.0015) で**ランダム**切替、連続ストリーム `TOTAL_DATA_POINTS`(5000)/クライアント | `data.py::make_concept_schedules` |
| 切替回数 | 2 概念は原則 **1 回**（default→第2概念） | **再帰的に複数回**（2 概念なら 0↔1 を反復） | 同上 |
| 処理粒度 | バッチ（500/時刻）単位で検出 | FedSDA: **per-sample 逐次**（ADWIN）／FedDrift ベースライン: 固定検出バッチ `FEDDRIFT_DETECT_BATCH`(50)（時刻粒度 `K_STEPS` から分離） | `clients/` |
| 学習の反復 | 固定 500 サンプルに対し R=100 ラウンド（≒反復エポック） | ストリームを**単一パス**（各サンプルは一度だけ到着） | `experiment.py` |

この「切替回数」の相違は recovery 分析に直結する: 本実装は 1 ランで**多数のドリフト
イベント**が生じ、それを Δ=0 起点に集計する（論文は原則 1 回）。

---

## 3. 学習ハイパーパラメータ

論文 `appendix-expt-setup.tex` の Table（training-params）との対応。

| 記号 | 説明 | 論文（合成データ） | 本実装 | 判定 |
|---|---|---|---|---|
| K | 1 ラウンドあたりのローカルステップ数 | 50 | `K_STEPS=50` | ✅ |
| η | 学習率（SINE/CIRCLE/SEA） | 10⁻² | `BASE_LR=0.01` | ✅ |
| — | Adam weight_decay / amsgrad | 10⁻³ / True | `WEIGHT_DECAY=1e-3` / `AMSGRAD=True` | ✅ |
| B | ミニバッチサイズ | 50 | `CLIENT_BATCH_SIZE=32` | ⚠️ |
| R | 1 時刻あたりの通信ラウンド数 | 100 | `R_ROUNDS=1` | ⚠️ |

R の相違は §2 の「単一パス・ストリーミング」に由来する（固定バッチを反復学習しない）。

---

## 4. 評価指標

| 項目 | 論文 | 本実装 |
|---|---|---|
| 主指標 | 各時刻 τ の学習後、τ+1 の（次概念）held-out で test accuracy | `paper_accuracy`（次時刻 held-out・ドリフト時刻除外）で論文式を再現。加えて prequential `accuracy` と recovery 曲線 acc(Δ) も算出 |
| 検出閾値 δ | アルゴリズム×データセットごとに δ∈{0.02,…,0.20} の**ベスト値**を採用 | 既定 `ADWIN_DELTA=0.05` / `DISTANCE_THRESHOLD`。`run_pareto_sweep.py` で δ を掃引し感度を提示 |

`paper_accuracy` は FedDrift 再実装で論文値（SINE ≈97.4%）にほぼ一致することを確認済み。

---

## 5. 実装上の設計判断

- **サーバ側モデル間距離の評価**: FedDrift 公開実装は aggregator が全クライアントの
  生データ（`all_data`）を中央に集めて距離を計算する**シミュレーション近道**を取る。
  本実装はこれに倣わず、評価対象モデルをデータ保持クライアントへ配布して現地評価させ、
  **集約統計量 (n, Σℓ, Σℓ²) のみ**を返す federated な形にしている
  （サブサンプリング `CROSS_EVAL_MAX_CLIENTS` / `EVAL_MAX_SAMPLES` による近似）。
  詳細と根拠は [ALGORITHM.md](ALGORITHM.md) §3 を参照。
- **FedDrift は再実装**であり、参照コードをそのまま呼んでいるわけではない。sea も CSV を
  読まず生成器で論文定義を再現している。

---

## 6. FedSDA 独自要素（FedDrift には無い提案部分）

以下は「相違」というより本研究の**貢献**。FedDrift ベースラインとの対比のため列挙する。

- **ADWIN による統計的ドリフト検出**（FedDrift の固定閾値・バッチ検出の置き換え）
- **FIFO 遅延バッファ**（`FIFO_BUFFER_SIZE`）による検知遅延中のコンセプト混合防止と
  事後分割

詳細は [ALGORITHM.md](ALGORITHM.md) を参照。
