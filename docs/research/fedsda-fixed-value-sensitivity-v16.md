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

## 計画レビュー（GPT-6 Luna、effort medium）の反映（2026-11-01）

1回目FAIL（Major 3、Minor 1）。全部反映する（上と食い違うときは、こちらを優先する）。

- 差し替えの時点: 定数を差し替えるのではなく、設定を組み立てる関数（`build_final_configuration_fedsda_run_settings`）の戻り値の設定のオブジェクトで、α・FIFOの容量・Tを置き換える（`dataclasses.replace`など。設定の検査を通す）。`switching_probability=1/T`は、clientを組み立てるときに設定から計算されるので、設定のTを置き換えれば届く。研究ツールの`run.py`の中で、この差し替えをどこで効かせるかを実装の報告に書く。
- FIFOの容量に連動する値: 製品と研究ツールの両方で、容量に連動する値（研究ツールの`run.py`が容量を渡している帰属の保留の最小の標本数、探索や格子の窓など）がすべて置き換えた容量から決まるようにする。旧い容量（30）のまま残る値が1つでもあれば、実行時に例外にして止める（報告だけで済ませない）。
- 検査: 記録した値だけでなく、実行の経路で実際に使われた値を、消費する側（Fixed-Shareの更新、MDLの選択に渡る`switching_probability`、検出器のα、帰属の保留の列の容量と、容量に連動する値）で直接読み取り、指定の値と照合する。「変種の結果が既定と異なる」は必須の検査にせず、参考として報告する。
- 主張の範囲（Minor）: 間隔50と500の両端、seed 0〜9での感度の調査として扱い、論文でもその範囲に限って書く。
- 2回目FAIL（Major 2）。反映: (1) 依頼文の本文の「報告するだけ」の記述を、上の「置き換えた容量から決め、旧い容量が残れば止める」に直した。(2) 判定の方法を事前に決める: 「小さい」という二値の結論は出さず、値ごとに、20セルの精度の差（10 seedの対の平均と、対の差のt分布による95%信頼区間）、差の絶対値の最大、信頼区間が0を含まないセルの数、全20セルの等しい重みの平均を示す。論文の文は、これらの数値をそのまま述べる形にする（「Tを半分・倍にすると、精度の差は20セルで−a〜+b pp」のように）。比べる相手に既存の製品の結果を使う場合は、同じ条件（データ、間隔、seed、client数、標本数）と基準版16の値であることを、runの識別子と記録の`behavior_baseline_version`・設定で確かめる（研究ツールの`product-default`を同じ実行で流すなら、その200 runを使う）。
- 3回目FAIL（Major 1: 「感度が小さい」と結論する基準が無い）。反映（主張を判定する基準を事前に決める）: 論文で「固定値をデータごとに調整しなくてよい」と主張するのは、各変種（6つ）で、既定の値のFedHASPが比較手法の最良（FedDriftの調整したδ、新しい4データでは公式の既定のδと、調整が済めば調整したδ。FedAvgの2形）を上回っていたセル（10 seedの平均）のすべてで、変えた値のFedHASPも比較手法の最良を上回るときとする。この基準を満たさない変種があれば、その値とセルを論文に書き、主張をその範囲に狭める。数値（差、95%信頼区間、最大の絶対差）は、この判定とは別に、そのまま報告する。基準の理由: 論文の主張は「データごとの調整なしで比較手法を上回る」であり、固定値を変えてもその結論が保たれるかが判定の対象になるため。
- 4回目FAIL（Major 1: 比較相手と条件の照合を事前に固定していない）。反映（集計の側の取り決めで、実装には影響しない）:
  - 既定のFedHASP: この実験で同じ実行に研究ツールの`product-default`（10データ×間隔50・500×seed 0〜9、200 run）を入れ、それを使う。
  - 比較手法（判定に使うもの、事前に固定）: 6データのFedDriftは`results/feddrift-tuned-delta-v15`のうち`results/feddrift-delta-tuning-v14/selected-delta-tolerance-0.1pp.csv`で選んだδのrun。新しい4データのFedDriftは`results/new-datasets-baselines/feddrift`（公式の既定のδ）。調整したδの結果がこの集計より先に出ても、判定には使わず、参考として別に示す。FedAvgは`results/method-comparison-v15/20261029_204550_fedavg-seeds0-9`と`results/new-datasets-baselines`の`fedavg`・`fedavg-local-updates-per-aggregation`。比較手法の最良は、セルごとに、これらの10 seedの平均の最大。
  - 条件の照合: どの結果も、`run_summary.csv`の`dataset_name`・`server_aggregation_interval_per_client_samples`・`random_seed`・`client_count`・`per_client_sample_count`が一致する行だけを使い、セルごとに10 seedがそろわなければ集計を止める。
