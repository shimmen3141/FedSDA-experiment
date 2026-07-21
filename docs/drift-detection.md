# FedSDAのオンラインドリフト検出器

FedSDAは、現在のモデルが各サンプルに対して出した損失
`ℓ_t = |h_t(x_t) - y_t| ∈ [0, 1]`を逐次監視する。検出対象は概念そのものではなく、
**現在のモデルに対する損失分布の変化**である。教師ありデータストリームでは、その変化を
概念ドリフトの代理信号として扱う。

現在は、ADWIN、bounded mean向け混合Shiryaev–Roberts型e-detector、HDDM-A/Wを実装している。
本書ではいずれかを基準とせず、FedSDAへ接続可能なオンライン検出器として整理する。

## 1. ADWIN

ADWIN（ADaptive WINdowing）は可変長ウィンドウ`W`を保持し、候補分割
`W = W_0 · W_1`ごとに古い側と新しい側の平均を比較する。実装では、すべての候補分割について

```text
|mean(W_0) - mean(W_1)| > ε_cut(δ, |W_0|, |W_1|, variance(W))
```

を調べ、閾値超過が最大の分割をドリフト点として採用する。検知後は古い側`W_0`を削除し、
新しい側`W_1`を残す。FedSDAはこの分割位置をFIFOバッファへ対応付け、検知遅延中のデータを
旧概念と新概念へ事後分割する。

- 入力: `[0, 1]`の損失系列
- 方向: 平均の上昇・低下の両方（平均差の絶対値）
- 主パラメータ: `ADWIN_DELTA`。小さいほど保守的
- 実装: `federated_drift_experiment/adwin.py`の`FullScanADWIN`
- 原論文: [Learning from Time-Changing Data with Adaptive Windowing](https://www.cs.upc.edu/~gavalda/papers/adwin06.pdf)

本実装は分割点をFIFO処理へ渡す必要があるため、全候補分割を明示的に走査する。
圧縮バケットを用いる高速な標準実装とは計算量が異なる。

### 強制ドリフトチェック

ADWIN系の既定モードには、ADWINが未発火でも見逃しを補う保険的チェックがある。
ADWINの現在の窓幅に対応するFIFO末尾について現行モデルの平均損失を再計算し、モデル履歴の
平均損失より`DISTANCE_THRESHOLD`以上悪化していればドリフト解決を開始する。これはADWINの
統計検定ではなくヒューリスティックな別経路である。

`FEDSDA_ENABLE_FORCED_DRIFT_CHECK=False`でADWIN系モードの強制チェックだけを無効化できる。
既定値は`True`なので既存結果は変わらない。e-SR系モードではe値の誤警報制御を迂回しないよう、
この設定にかかわらず常に無効である。

## 2. bounded mean向け混合Shiryaev–Roberts型e-detector

混合e-SR detectorは、定常時の条件付き平均損失が基準値`m`以下という帰無仮説

```text
E[ℓ_t | 過去] ≤ m
```

に対し、平均損失の上昇をオンラインに検出する。各候補変化点`k`と賭け率`λ`について、
観測ごとに次のe-value増分を掛け合わせる。

```text
L_t(λ) = 1 + λ(ℓ_t / m - 1),    0 < λ < 1
```

帰無仮説の下では`E[L_t(λ) | 過去] ≤ 1`となる。未知の変化量に一つの`λ`を合わせる代わりに、
実装では`λ ∈ {0.05, 0.1, 0.2, 0.4, 0.8}`を等重みで混合する。さらに各時点から
新しい候補変化点を開始し、それらをShiryaev–Roberts型に合計する。統合した証拠が
`1 / α`以上になった時点で検知し、寄与が最大の候補変化点をFIFOの分割位置として返す。

- 入力: `[0, 1]`の損失系列
- 方向: 基準平均からの損失上昇のみ
- 主パラメータ: `E_DETECTOR_ALPHA`
- 実装: `federated_drift_experiment/e_detector.py`の`BoundedMeanEDetector`
- 適用モード: `FedSDA_NoCached_ESR`、`FedSDA_Cached_ESR`
- 理論的背景: [E-detectors: a nonparametric framework for online changepoint detection](https://arxiv.org/abs/2203.03532)

`α`は実験期間全体の誤警報確率ではない。e-detectorの閾値を`1/α`としたとき、適切な帰無仮説の
下で平均誤警報時間（ARL）を少なくとも`1/α`と解釈する。したがって、既定値`α=0.001`は
ARL 1000以上を狙う設定であり、「誤警報確率0.1%」を意味しない。

### FedSDA実装での基準平均

検出器をリセットするとき、現在モデルの履歴損失平均を`m`として固定する。数値的不安定性を
避けるため下限を設ける。モデルのオンライン更新によって定常時の損失平均も変わり得るため、
この推定値が常に条件付き平均の上限であるとは限らない。従って、現実装における厳密なARL保証は
この仮定が成立する場合に限られる。保証を強める候補として、較正区間から求めた平均の信頼上限を
`m`に使う方法がある。

e-detectorの証拠を迂回する判定経路を作らないため、ESR系ではFedSDAの保険的な
強制ドリフトチェックとクラス別ADWINを併用しない。

### 全体・正解クラス別e-SR混合

ClassESRでは、全体損失と二値の正解クラス別損失を監視する3個のe-SRへ各`1/3`の固定重みを
割り当て、その混合e値を`1/α`と比較する。クラス別検出器は該当クラスのサンプルだけで時刻を進める。
検知時は寄与最大の成分を選び、クラス成分ならそのクラス系列の候補開始位置を元のサンプル位置へ
戻してFIFOを分割する。

クラス別e-SRにも全体モデル統計から得た基準平均を使う。ARL保証をクラス別成分まで厳密に
解釈するには、各クラスの定常時条件付き平均も対応する基準以下という追加仮定が必要である。

## 3. HDDM-A / HDDM-W

HDDM（Hoeffding Drift Detection Method）は、有界系列の平均損失上昇をオンラインに検出する。
HDDM-Aは累積平均とHoeffding境界を使い、過去の低損失区間と現在までの平均差が境界を超えたときに
検知する。HDDM-Wは平均をEWMAへ置き換え、重み付き和に対するMcDiarmid型境界を使う。

- 入力: `[0, 1]`の損失系列
- 方向: 損失上昇のみ（FedSDAでは低下をドリフトとして扱わない）
- 主パラメータ: `HDDM_DRIFT_CONFIDENCE`
- HDDM-W固有パラメータ: `HDDM_W_LAMBDA`
- 実装: `federated_drift_experiment/hddm.py`の`HDDMA` / `HDDMW`
- 適用モード: `FedSDA_{NoCached,Cached}_HDDMA`、`FedSDA_{NoCached,Cached}_HDDMW`
- 原論文: [Online and Non-Parametric Drift Detection Methods Based on Hoeffding's Bounds](https://doi.org/10.1109/TKDE.2014.2345382)
- 著者公開PDF: [RIUMA institutional repository](https://riuma.uma.es/xmlui/bitstream/10630/25767/1/HDDM-TKDE--publicado-RIUMA.pdf)

HDDM-Aは平均を一様に扱うため急激な変化の比較対象、HDDM-Wは最近の損失を強く反映するため
緩やかな変化の比較対象になる。どちらも定数個の統計量だけを更新するため、現行の全分割走査ADWINや
全候補e-SRより1更新あたりの計算量が小さい。警告境界も計算するが、FedSDAのモデル切替は
ドリフト境界を超えた場合だけ行う。検出器単体の比較を明確にするため、強制チェックは併用しない。

ClassHDDMAでは全体損失と正解クラス別損失を並列監視し、各系列に同じ
`HDDM_DRIFT_CONFIDENCE`と警告confidenceを適用する。Bonferroni型の多重検定補正を
行う場合は、目的とする全体confidenceを`クラス数+1`で割った値を実験時に設定する。
クラス系列が検知した場合は、その系列のHDDM-Aが推定した新概念側サンプル数を元のストリーム位置へ
戻してFIFOを分割する。保持する位置はFIFO長までなので、損失統計のO(1)更新を維持する。

## 4. 手法の比較

| 観点 | ADWIN | 混合e-SR detector | HDDM-A/W |
|---|---|---|---|
| 統計量 | 可変窓の前後平均差 | 基準平均を超える逐次的証拠 | 累積平均 / EWMAの差 |
| 検出方向 | 両方向 | 損失上昇の片方向 | 損失上昇の片方向 |
| 変化時刻 | 採択した窓分割 | 寄与最大の候補開始点 | 最小境界 / EWMA切替点 |
| 検知後 | 古い窓を削除 | 新基準で再初期化 | 統計量を再初期化 |
| 制御量 | `δ`に基づく分割検定閾値 | `α`に基づくARL閾値 | 信頼度に基づくHoeffding/McDiarmid境界 |
| 1更新の実装コスト | 窓内の全分割を走査 | 候補変化点数×賭け率数 | O(1) |

`ADWIN_DELTA`と`E_DETECTOR_ALPHA`は統計的意味が異なるため、同じ数値に揃えても公平にはならない。
比較実験では、同一の損失系列、FIFO処理、学習、サーバ処理を使用し、検出精度、誤検知、検知遅延、
計算量を併記する。ただし既存のADWINモードは強制チェックも使用するため、検出器単体の比較には
強制チェックを無効にしたADWIN対照条件を別途設ける必要がある。

## 5. FedSDA各モードにおける位置づけ

モード名は通信プロトコル（NoCached/Cached）と検出器を独立した軸として表す。

| モード | 実装状況 | ローカル検出器 | 補助判定 | 通信・サーバ処理 |
|---|---|---|---|---|
| `FedSDA_NoCached_ADWIN` | 実装済み | 全体損失ADWIN | 強制ドリフトチェックあり | NoCached |
| `FedSDA_NoCached_ClassADWIN` | 実装済み | 全体損失ADWIN + 正解クラス別ADWIN | 強制ドリフトチェックあり | NoCached |
| `FedSDA_NoCached_ESR` | 実装済み | bounded mean向け混合e-SR detector | 強制チェックなし | NoCached |
| `FedSDA_NoCached_ClassESR` | 実装済み | 全体 + 正解クラス別e-SRの固定重み混合 | 強制チェックなし | NoCached |
| `FedSDA_NoCached_HDDMA` / `HDDMW` | 実装済み | 全体損失HDDM-A / HDDM-W | 強制チェックなし | NoCached |
| `FedSDA_NoCached_ClassHDDMA` | 実装済み | 全体 + 正解クラス別HDDM-A | 強制チェックなし | NoCached |
| `FedSDA_Cached_ADWIN` | 実装済み | 全体損失ADWIN | 強制ドリフトチェックあり | Cached |
| `FedSDA_Cached_ClassADWIN` | 実装済み | 全体損失ADWIN + 正解クラス別ADWIN | 強制ドリフトチェックあり | Cached |
| `FedSDA_Cached_ESR` | 実装済み | bounded mean向け混合e-SR detector | 強制チェックなし | Cached |
| `FedSDA_Cached_ClassESR` | 実装済み | 全体 + 正解クラス別e-SRの固定重み混合 | 強制チェックなし | Cached |
| `FedSDA_Cached_HDDMA` / `HDDMW` | 実装済み | 全体損失HDDM-A / HDDM-W | 強制チェックなし | Cached |
| `FedSDA_Cached_ClassHDDMA` | 実装済み | 全体 + 正解クラス別HDDM-A | 強制チェックなし | Cached |

同じ検出器名のNoCached/Cachedを比較すると通信プロトコルの影響を、同じ通信プロトコル内で
検出器名を比較すると検出器構成の影響を調べられる。

ただし、ADWIN系は強制ドリフトチェックを持ち、ESR系は統計的な証拠を迂回する補助判定を
無効にしている。従って両者はADWIN単体とe-SR単体の純粋な比較ではなく、
**実際に採用した検出パイプライン全体の比較**である。検出器単体を比較する場合は、全体損失ADWIN
だけを使い、`FEDSDA_ENABLE_FORCED_DRIFT_CHECK=False`にして実行する。

## 6. 変化点推定の評価

警報時刻と推定変化開始位置は異なる。FedSDAは検知時にFIFOを新旧概念へ分けるため、警報の早さに
加えて分割位置の正確さも記録する。真のドリフト後`DELAY_TOLERANCE`以内の警報を対応付け、推定位置
から真の位置を引いた誤差について次を返す。

- `change_point_mae`: 推定位置誤差の絶対値平均
- `change_point_bias`: 符号付き平均。正なら新概念の開始を遅く、負なら早く推定
- `change_point_estimate_count`: 評価できた対応数

これは検出フローを変更しない評価基盤なので、独立したFedSDAバージョンにはしない。raw `.npz`にも
警報位置と推定開始位置を保存し、FIFO分割純度などを事後分析できるようにする。
