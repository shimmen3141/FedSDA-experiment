# 基準版16の固定値の感度（計画、2026-11-01）

- 目的: 論文の主張「データごとの閾値の調整が要らない」に対して、FedHASP（基準版16）が全条件で同じ値に固定している3つの値を、1つずつ半分と倍に変えたときの精度と資源の変化を測る。査読で予想される「固定値への感度」の問いに答える。
- 背景: 過去の版ではT240・FIFO60などを試したが、基準版16では測っていない。2026-11-01にユーザーが実施を決めた。

## 振る値（ほかの値は基準版16の既定のまま）

| 値 | 既定 | 振る値 | 定義の場所 |
|---|---|---|---|
| T（Fixed-Shareの分け直しの時間の尺度。MDLの切替の確率1/Tにも同じ値が使われる） | 120 | 60、240 | `fedsda_final_configuration_run_settings.py`の`_FIXED_SHARE_WEIGHT_REDISTRIBUTION_TIME_SCALE_SAMPLES`。`fedsda_run_client.py`が`switching_probability=1/T`に使う |
| α（e-SRの誤警報の制御） | 0.001 | 0.01、0.0001 | 同`_E_SR_FALSE_ALARM_CONTROL_ALPHA` |
| FIFOの容量（帰属を保留する標本の数。W ≤ 容量＋1） | 30 | 15、60 | 同`_PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES` |

- Tは、予測の重みとMDLの切替の符号長の両方で同じ値を使う設計なので、両方を一緒に変える（論文の定義どおり）。
- 変化区間の最小の長さ（5件）、候補の学習の設定、adapterのrankは振らない。

## 方法

- 研究ツール（`tools/research_provisional_candidate_expert/`）の変種として作り、製品の経路と既定は変えない。変種は`product-default`系（組E）の上で、実行中だけ設定の値を差し替える。変種の名前は`product-default-time-scale-60`・`-240`、`product-default-esr-alpha-0.01`・`-0.0001`、`product-default-fifo-capacity-15`・`-60`。
- 差し替えが効いていることを、runの中で実際に使われた値（Fixed-Shareの率、MDLの切替の確率、検出器のα、FIFOの容量）の記録で確かめる。既定の値を指定した場合（例: Tを120）に`product-default`と全項目で一致することを検査する。
- 条件: 10データ（SINE2、CIRCLE2、SEA2、SEA4、MNIST2、MNIST4、fmnist2、fmnist4、rotmnist2、rotmnist4）×集約間隔50・500×seed 0〜9、10 client×5000標本。6変種×200 run＝1,200 run。研究室サーバ（uranium、64 workers）。間隔は、論文の頑健さの主張の両端の50と500にする（全間隔の2,400 runは時間がかかるため）。
- 比べる相手: 同じ条件の基準版16の既定。研究ツールの`product-default`を同じ実行で流す（200 run）か、製品の既存の結果（`fedsda-every-sample-change-interval-search`の6データと新しい4データの結果）を使う。研究ツールの`product-default`と製品の結果が一致することは、IMPROVE-032・033で確かめている。

## まとめ方

- データ×間隔の20セルごとに、既定との精度の差（10 seedの平均、勝ち/負け）、通信とMACの比、警報の回数、新しいモデルの採用の回数。
- 全体の値は20セルの平均を等しい重みで平均する。データをまたぐ平均で、あるデータの悪化を別のデータの改善で打ち消さない。
- 論文では、「値を半分・倍にしても精度の変化は小さい（またはどのデータで効くか）」を、表か図で示す。許容の幅は事前に決めず、結果を見てユーザーが判断する。
