# データ特性とドリフト検出への影響

> 文書の役割: 実験条件の解釈資料
>
> 対象と採用状況: 各データセットの特性と検出への影響を扱う。性能差を解釈するための背景資料として使用する。
>
> 最終構成の正本: [FedSDAの最終提案構成](../overview/proposed-method.md)

本書は、合成・画像データの構成から予想される検出難度と、実験で繰り返し観測された傾向を
整理する。個別実験の優劣を固定的に主張するものではなく、追加手法を設計するときの仮説として使う。

## Sine2

二つの概念は正弦境界のラベルを反転した関係にあり、概念切替時には入力領域のほぼ全体で
正解ラベルが変わる。クラス比率も比較的均衡している。

- 損失上昇が広域かつ急激に現れ、全体損失の逐次検出器が反応しやすい。
- 旧概念と新概念を一つのバッチへ混ぜる影響が大きい。
- FIFOで新旧データを分けるFedSDAの利点を確認しやすい。

## Circle2

二つの概念は中心・半径の異なる円であり、両概念とも円の外側が多数派になる。概念間で
ラベルが異なる領域は約26.7%で、変化は局所的である。

- 多数の不変サンプルが全体平均損失の変化を薄める。
- 全体検出器だけでは遅延・見逃しが生じやすく、クラス別検出やモデル間比較が有力になる。
- 異なる領域を得意とするモデルを過剰に統合すると、局所的な相補性を失う可能性がある。
- 評価ストアが多数派へ偏ると、クロス評価でも概念差を捉えにくい。

## SEA2とSEA4

SEA2は閾値8と9の二概念である。特徴量を一様分布とすると、両概念でラベルが異なる領域は

\[
P(8 < x_1+x_2 \leq 9)=\frac{9^2-8^2}{200}=0.085
\]

で約8.5%に限られ、さらにラベルノイズの影響を受ける。したがって、統計的変化を検出できても、
別モデルを作成・維持する実用的利益が小さい場合がある。

SEA4はより離れた閾値を含むため、遷移によって変化領域の広さが異なる。SEA2とSEA4の結果を
一括して「SEAへの強さ」と解釈せず、概念集合と遷移ごとに検出難度を考える必要がある。

## MNIST2

MNIST2は10クラスのうち主にラベル1と2の対応だけが変化する。影響を直接受けるサンプルが
全体の一部であるため、全体損失では信号が希釈される。

- 正解クラス別検出は全体検出より変化を直接監視できる。
- 少数の検知区間だけで新規モデルを強く学習すると、影響を受けないクラスの忘却を招き得る。
- 再来概念では、警報後に既存モデルを再利用できるかが精度と遅延へ大きく影響する。
- モデルを細分化しすぎると学習データが分散するため、モデル数の多さが精度へ直結しない。

## Fashion-MNISTのラベルの入れ替えとMNISTの回転

既存の6データに加え、次の4データをFedSDA・FedDrift・FedAvgで使える。入力は28×28の行優先の784画素、
クラス数は10。元の画素をfloat32で255で割り、概念0の標本で事前学習する。

| 名前 | 元の学習用画像 | 概念0 | 概念1 | 概念2 | 概念3 |
|---|---|---|---|---|---|
| `fmnist2` | Fashion-MNIST（60000件） | 元のラベル | 1↔2 | — | — |
| `fmnist4` | Fashion-MNIST（60000件） | 元のラベル | 1↔2 | 3↔4 | 5↔6 |
| `rotmnist2` | MNIST | 0° | 180° | — | — |
| `rotmnist4` | MNIST | 0° | 90° | 180° | 270° |

Fashion-MNISTでは画像を変えず、概念ごとに表の1組だけラベルを入れ替える。
クラスの意味は1 Trouser↔2 Pullover、3 Dress↔4 Coat、5 Sandal↔6 Shirt。
根拠は[公式FedDriftのラベル交換の実装](https://github.com/microsoft/FedDrift/blob/e4ac32fe/fedml_api/data_preprocessing/MNIST/data_loader_cont.py)の
`MNIST_Data.generate_sample`と、FedDrift論文の付録のMNISTの定義であり、元の画像だけをFashion-MNISTへ替える。
[FedCCFA（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/942e820be4aa112509b3a281ff398851-Paper-Conference.pdf)の
Fashion-MNISTのラベル交換も同じクラスの組を使う。

回転は画像だけを反時計回りに回し、ラベルを変えない。根拠は公式FedDriftの
[`18e7fbe`の`data_loader_cont.py`](https://github.com/microsoft/FedDrift/blob/18e7fbe/fedml_api/data_preprocessing/MNIST/data_loader_cont.py#L74-L83)と
[`62ad4e2`の同ファイル](https://github.com/microsoft/FedDrift/blob/62ad4e2/fedml_api/data_preprocessing/MNIST/data_loader_cont.py#L165-L171)。
前者は78行で第1概念にk=0、79〜83行で第2概念を180°としてk=2を渡す。
181〜183行は`np.reshape(x, (28, 28))`→`np.rot90(x2d, k=rotation_k)`→`rx2d.flatten()`。
後者は76〜77行で概念行列の値kをそのまま渡し、168〜170行で同じ回転を行う。
このため、2概念は公式の明示した0°・180°、4概念はk=0〜3の0°・90°・180°・270°とする。
`62ad4e2`に値0・1の行列を与えた場合の0°・90°とは区別する。
4角度は[IFCA（NeurIPS 2020）](https://arxiv.org/abs/2006.04088)のRotated MNISTとも同じ
（IFCAの2クラスタのRotated CIFARは0°・180°）。

本実装のIDX読込みは行優先であり、`numpy.rot90(pixels.reshape(28, 28), k=r).reshape(784)`と同じ画素順を返す。
公式の[MNISTの説明](https://github.com/microsoft/FedDrift/blob/e4ac32fe/data/MNIST/README.md)はIDXを元データとし、
平らな784次元の画像を入力とする。ただし公式の配布zipの平坦化処理自体はリポジトリに無く、
そのzipも取得していないため、元のIDXの行優先と同じ並びである点はこの説明に基づく。

### 先行実装と異なる共通条件

概念の変換以外は今の`mnist2`・`mnist4`にそろえ、3手法で同じ入力条件を使う。

- 標本は学習用画像から無作為に復元抽出する。画像の位置の選択にNumPy乱数を1回使う。
  公式FedDriftはseed 100で並べ替えた画像を順に使い、使い切ると先頭へ戻る。
- 既定の概念の時系列とclientへの分け方は今のMNISTと同じランダムな変化点を使う。
  公式FedDriftの時刻の境界ごとの変化点行列とは異なる。
- ネットワークはMLP（隠れ層1568）、学習率0.001で、3手法とも今のMNISTと同じ。
  FedDriftの回転の版とFedCCFA・IFCAのCNNとは異なるので、論文の精度と直接比べない。
- FedDriftのδの既定は4データとも0.10。新データ固有の公式値は無いため、公式の
  [`default_deltas`のMNISTの値](https://github.com/microsoft/FedDrift/blob/e4ac32fe/fedml_api/distributed/fedavg_ens/FedAvgEnsDataLoader.py#L1274)を使う。

ラベルの入れ替えでは入力画像の分布がそのままでも正解が変わる。一方、回転では正解を保って画素の位置が変わる。
同じ概念数でも異なる適応の問題なので、結果はデータごとに示す。これらの追加で既存6データの値や基準版は変更しない。

### Fashion-MNISTの配置

[公式README](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data)の配布元は
`http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/`。次の学習用2ファイルだけを使う。
gzipのままリポジトリ直下の`data/fashion-mnist/`へ置く。別の場所なら`FDE_FASHION_MNIST_DATA_DIR`で指定する。
回転のMNISTは今の`data/mnist/`または`FDE_MNIST_DATA_DIR`を使う。

| ファイル | 公式のMD5（gzipファイル） |
|---|---|
| `train-images-idx3-ubyte.gz` | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | `25c81989df183df01b3e8a0aad5dffbe` |

別環境への配置例（Linux、リポジトリ直下で実行）:

```bash
mkdir -p data/fashion-mnist
curl --fail --location http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz --output data/fashion-mnist/train-images-idx3-ubyte.gz
curl --fail --location http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz --output data/fashion-mnist/train-labels-idx1-ubyte.gz
md5sum --check <<'MD5'
8d4fb7e6c68d591d4c3dfef9ec88bf0d  data/fashion-mnist/train-images-idx3-ubyte.gz
25c81989df183df01b3e8a0aad5dffbe  data/fashion-mnist/train-labels-idx1-ubyte.gz
MD5
```

公式の配布URLはHTTPである。取得後のMD5照合を必須とし、両方が`OK`になったファイルだけを使う。
Windowsでは`Get-FileHash -Algorithm MD5 data/fashion-mnist/*.gz`で上の値と照合する。
読込み時には既存MNISTと同じIDXの検査を行い、MD5は再検査しない。ファイルが無い・形式が違う場合は拒否し、自動取得しない。
画像ファイルはGit管理へ追加しない。

## STAGGER・Agrawal・Random RBFの再来（`stagger3`・`agrawal4`・`randomrbf4`）

単一ストリームの比較（River）で標準的に使われる3つの生成器を、全手法（FedSDA・FedDrift・FedAvg・DWM）のデータとして足した
（2026-11-20、spec `stagger-agrawal-random-rbf-recurring-datasets`）。概念の中身は、単一ストリームの版
（`src/single_stream_hard_assignment_soft_prediction/river_synthetic_recurring_streams.py`）が使うRiver 0.27.0の
`river.datasets.synth`と同じにし、製品はRiverに依存しない。

| データ | 入力 | 概念 | クラス | 概念cの中身（River） | 特徴の尺度 |
|---|---|---|---|---|---|
| `stagger3` | 3（size・color・shape） | 3 | 2 | `STAGGER(classification_function=c)`。0: sizeがsmallかつcolorがred、1: colorがgreenまたはshapeがcircle、2: sizeがmediumかlarge | 値/2（0・0.5・1） |
| `agrawal4` | 9（salary・commission・age・elevel・car・zipcode・hvalue・hyears・loan） | 4 | 2 | `Agrawal(classification_function=c, perturbation=0.05)`。関数0〜3（年齢・給与・学歴の規則） | 固定の範囲で0〜1へ線形に写す |
| `randomrbf4` | 10 | 4 | 2 | `RandomRBF(seed_model=c, n_classes=2, n_features=10, n_centroids=50)`。概念ごとに50個の中心 | 写さない |

- 標本: runが貸すNumPyの`RandomState`で、Riverと同じ式・同じ順序で作る（Pythonの`random()`を`random_sample()`、
  `randint(a, b)`を`randint(a, b + 1)`、`gauss(0, 1)`を`standard_normal()`に置き換える）。特徴はSEAと同じくfloat32へ丸める。
  Random RBFの中心はrunのseedに依らず、Riverの`seed_model=c`と同じ値（Pythonの`random.Random(c)`）。
- Agrawalの尺度: salary (x−20000)/130000、commission x/75000、age (x−20)/60、elevel x/4、car (x−1)/19、zipcode x/8、
  hvalue x/900000、hyears (x−1)/29、loan x/500000。範囲は摂動0.05の出力が取りうる値を覆う。特徴の桁が1〜5×10⁵まで違い、
  そのままでは神経網の学習が成り立ちにくいため写す。STAGGERも0〜1へそろえる。SEA（0〜10）は既存のとおり写さない。
- Random RBFは値に上限・下限が無いので写さない。実測（概念ごとに`RandomState(0)`で生成器を作り直し、その概念の2万標本の
  全10特徴の値をまとめて数える。4概念の範囲）: 最小−2.20〜−1.46、最大2.45〜2.72、1%点−0.27〜−0.19、99%点1.24〜1.28。
- Riverとの一致は、Riverのある環境（WSLの`~/river-env`）のtest
  （`tests/single_stream_hard_assignment_soft_prediction/test_recurring_synthetic_generators_match_river.py`）で確かめる:
  分類の関数（STAGGERは27通りの全入力、Agrawalは摂動なしの1万標本）、Random RBFの全中心、各概念2万標本のクラス1の割合と
  各特徴の平均（差が差の標準誤差の4倍以内）。

### 開示する差
- 標本の乱数の列: 製品はrunのNumPyの乱数で作るので、分布はRiverと同じだが、値の列は単一ストリームの版の系列と違う。
- 概念の時系列とclientへの分け方: ほかの製品のデータと同じ（ランダムな変化点、全手法で共通）。単一ストリームの版の1本の系列とは違う。
- Agrawalのhvalueの摂動: Riverは`_perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)`を
  `(val, val_min, val_max, val_range)`として受けるので、hvalueは`(9 - zipcode) * 100000`か0になる
  （MOAの`perturbValue(val, range, min, max)`の引数の並びと食い違うためと見られる）。単一ストリームの版と同じ定義にするため、このまま再現した。
  分類の関数0〜3はhvalueを使わないので、ラベルには影響しない。
- 特徴の尺度: 神経網のためSTAGGER・Agrawalを0〜1へ写す。単一ストリームの木の版は写していない（木は尺度に依らない）。
- FedDriftのδの既定0.10は公式の値ではなく仮定である。公式FedDriftの表（`FedAvgEnsDataLoader.py`の`default_deltas`:
  sea 0.04・sine 0.20・circle 0.10・MNIST 0.10）に3データは無い。本規模の比較の前に、評価に使わないseedで調整する
  （[fmnist・rotmnistの調整](../research/feddrift-delta-tuning-new-datasets-v16-20261101.md)と同じ手順）。
- ネットワークと学習率は、全手法でほかの合成データ（`sea4`など）と同じ（隠れ層(32, 32)、学習率0.01）。

## 手法比較での注意

FedSDAはサンプル単位の損失変化を起点にモデル操作を行う。一方、FedDriftは検出バッチごとに
保持モデルを比較する。このため、ランダムなバッチ途中ドリフトはFedSDAの目的に沿う一方、
FedDriftには不利になり得る。`random`と`feddrift_fixed`の両スケジュールを用い、次を分けて示す。

1. バッチ境界と無関係な現実寄りの変化への適応性能。
2. FedDrift原論文に近い固定スケジュールでの再現・公平比較。
3. 検出精度だけでなく、モデル操作、通信量、モデル数、学習量断片化を含む実用上の差。
