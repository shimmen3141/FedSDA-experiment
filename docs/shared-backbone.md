# 共有バックボーン＋概念別ヘッド

## 実験要素の位置づけ

共有表現の実験は、次の独立した関心を組み合わせて構成する。ただし、概念上の独立性と
現行コードで選択可能な組合せは区別する。

| 層 | 主な選択肢 | 変更対象 | 依存関係 |
|---|---|---|---|
| モデル構造 | `independent` / `shared_backbone` / `residual_adapter` | パラメータの共有・概念固有範囲 | 複数モデルを持つ手法に適用 |
| 共有表現のローカル学習 | `sequential` / `joint` / `frozen` | 共有部と概念固有部の更新則 | 共有表現を持つモデル構造だけで有効 |
| joint勾配統合 | `mean` / `pcgrad` | 概念別損失から得た共有部勾配の統合 | `joint`学習だけで有効 |
| 予測ルーティング | `hard` / `restarting_soft` / `protected_soft` | 正解観測前の予測の選択・混合 | モデル構造には原理上依存しない |
| 集約後のルーティング再較正 | `none` / `fifo_replay`等 | SoftRoutingの累積損失 | 共有表現とSoftRoutingの両方が必要 |

したがって、`joint`と`frozen`はモデル構造に従属する学習オプションである。一方、SoftRoutingは
独立モデルにも共有モデルにも適用できる予測オプションであり、実際に独立モデル用の
`FedSDA_NoCached_ClassESR_RestartingSoftRouting`も実装されている。`fifo_replay`はSoftRouting一般ではなく、
サーバ集約によって共有表現が同時に変化する場合だけ必要になる再較正である。

現行コードでは共有表現modeを`NoCached + ClassESR + Restarting SoftRouting`として実装しているため、
共有モデルでhard routingや他検出器を選ぶ組合せは未実装である。これはモデル構造上の必然ではなく、
検証済みの実装範囲を限定するための制約である。コード上の正確な依存関係は`docs/options.md`を参照する。

```text
モデル構造
├─ independent
├─ shared_backbone
└─ residual_adapter
   │
   └─ 共有表現の学習: sequential / joint / frozen

予測ルーティング（モデル構造から原理上独立）
├─ hard
├─ restarting_soft
└─ protected_soft
   │
   └─ 共有表現との組合せだけ再較正: none / fifo_replay / ...
```

## モデル構造

### 完全共有＋概念別head

`FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting`は全隠れ層を共有し、出力headだけを
概念別にする。共有量と計算再利用は最大になるが、概念間の負の転移や共有表現の容量不足が起こり得る。

### 評価済みのAffine・部分共有adapter

部分共有方式は、全共有で生じた概念間の負の転移を抑えつつ、入力に近い低層表現の学習量と通信量を
共有するために評価した方式である。合成データの二層MLPでは、元の第1隠れ層を共有部、第2隠れ層を
概念別adapterとして使った。MNIST2/MNIST4の一層MLPでは、共有隠れ層の後に特徴ごとのscaleとbiasから
なる概念別Affine adapterを置いた。このadapterには幅のハイパーパラメータがなく、概念ごとの大きな
全結合層の複製を避けられる。

通信と集約では、共有部をクライアントごとに一度だけ転送・FedAvgし、adapterとheadをモデルIDごとに
転送・FedAvgした。既存の`compute_head_*`指標ではadapterとheadを合わせた概念固有部分を数えた。

しかし5シード比較では、Residual adapterに対する精度・通信量上の固有の優位がなく、特にMNIST4で
大きく劣った。線形headの直前に置くAffine変換はheadへ吸収可能であり、概念固有の非線形表現を十分に
増やせない。このため実験modeと実装は削除し、比較済みの不採用案として本節だけを残す。

### 低ランク残差adapter

`FedSDA_NoCached_ResidualAdapter_ClassESR`と
`FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting`は全隠れ層を共有し、概念別の低ランク
非線形残差adapterと出力headを持つ。前者は現行モデルだけで予測するhard routingの対照方式、後者は
全保持モデルを混合する方式である。共有特徴`z`を概念`c`ごとに次のように補正する。

\[
z_c = z + U_c\operatorname{ReLU}(V_c z)
\]

`U_c`のweightとbiasをゼロ初期化するため、学習開始時は`z_c = z`となり、完全共有方式と厳密に同じ
予測から開始する。学習後は概念固有の非線形変換を獲得でき、線形headへ吸収されてしまうAffine
adapterの制約を避けられる。rankは`SHARED_ADAPTER_RANK`（CLIでは`--shared-adapter-rank`、既定8）で
指定し、実際のrankは特徴次元を上限とする。共有バックボーンはクライアントごとに一度、残差adapterとheadは
モデルIDごとに通信・FedAvgする。

## 目的

`FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting`は、既存の
`FedSDA_NoCached_ClassESR_RestartingSoftRouting`から検出・モデル作成・クラスタリングを
変えず、複数概念モデルの特徴抽出部だけを共有する構造的な派生方式である。

独立モデルでは概念ごとにMLP全体を保持・学習・推論する。共有方式では、クライアント内で
一つの特徴抽出バックボーンと、モデルIDごとの出力ヘッドを持つ。これにより次を狙う。

- モデルごとに分断されていた表現学習量を共有する。
- SoftRoutingで全モデルを評価するとき、特徴抽出を1サンプルにつき1回へ減らす。
- 概念別ヘッドを残し、概念ごとの専門化と予測混合を維持する。
- バックボーンを1回だけ転送し、モデル数に比例する重複通信を減らす。

## FedSDAと組み合わせる意味

共有バックボーン自体は一般的なmulti-task learningや個人化連合学習にも適用でき、FedSDAだけで
成立する構造ではない。一方、タスク境界も概念IDも与えられないストリームで「何を共有し、何を
概念別にするか」を決めるには、FedSDAの高粒度な検出・データ分割・モデル管理が次の役割を持つ。

1. サンプル単位の統計的検出が、固定バッチ境界に依存せず概念別head・adapterを作る契機を与える。
2. 検出区間とモデル別データストアが、概念固有部分へ異なる概念のデータが混入する量を抑える。
3. 既存モデル再利用により、再帰概念では新しいheadを増やさず、既存の概念固有部分へ戻れる。
4. クロス評価とクラスタリングが、クライアントごとに検出時刻が異なる概念モデルのIDを対応付け、
   同じ概念のadapter・headだけをサーバで集約できる。
5. 共有バックボーンは各概念に分断されていた表現学習量を統合し、概念別adapter・headはFedSDAが
   発見した細粒度な差を保持する。
6. SoftRoutingは検出・切替直後の概念帰属が不確かな期間に複数モデルを混合し、hardなモデル管理と
   実際の予測の間を連続的につなぐ。

この組合せの相乗効果は、「共有表現で全概念の学習量を利用すること」と「統計的に分割された
概念固有部分で負の転移を抑えること」を同時に狙える点にある。逆に、検出や分割が不正確なら
adapter・headの専門化も崩れ、共有部には相反する勾配が入る。そのため共有方式はFedSDAと無関係な
付加機能ではなく、FedSDAの検出精度・再利用・クラスタリング品質と相互作用する提案候補として
評価する。ただし、共有バックボーンそのものをFedSDA固有の新規性として主張してはならない。

## ローカル学習方式

`--shared-backbone-training`で、正式採用済みモデルの通常ローカル学習を選択する。

- `sequential`（既定）: 各概念ヘッドを順番に学習し、そのたびに共有バックボーンも更新する。
  既存結果との互換方式だが、バックボーンの更新回数が参加ヘッド数に比例し、更新順にも依存する。
- `joint`: 各ヘッドから同数のミニバッチを抽出し、全損失をサンプル数で加重平均する。全ヘッドへ
  勾配を伝播したうえで、バックボーンは共同更新1回、各ヘッドは1回ずつ更新する。ヘッド数に
  よる共有部の過剰更新と順序依存を避ける本命方式である。
- `frozen`: `joint`と同じデータ構成でバックボーンを固定し、ヘッドだけを更新する診断方式である。
  `sequential`の悪化が共有表現への継続的な干渉に由来するかを切り分けるために使う。

`joint`と`frozen`は学習則を変えるため高速化オプションではなく、独立した実験条件として扱う。
CSV・NPZには`shared_backbone_training`を保存し、既定以外ではPareto凡例にも方式を表示する。

### 概念間の勾配競合とPCGrad型更新

`joint`では、各概念ヘッドの損失から共有バックボーンに対する勾配を個別に求める。
`--shared-backbone-gradient-strategy`で統合方式を選択する。

- `mean`（既定）: サンプル数で加重した平均勾配を用いる。従来の`joint`更新と同じである。
- `pcgrad`: 二つの概念勾配の内積が負の場合だけ、一方から他方と競合する射影成分を除去してから
  サンプル数で加重平均する。競合しない勾配は変更せず、新しい数値閾値も追加しない。

診断と更新を切り分けるため、`mean`でも概念勾配対を計測する。次をCSV・NPZへ保存する。

- `backbone_gradient_pair_count`: 比較可能だった概念勾配対の数。
- `backbone_gradient_conflict_count` / `backbone_gradient_conflict_rate`: cosine類似度が負だった数と割合。
- `backbone_gradient_cosine_mean`: 全勾配対の平均cosine類似度。
- `backbone_gradient_negative_cosine_mean`: 競合した勾配対だけの平均cosine類似度。
- `backbone_gradient_applied_conflict_rate` / `backbone_gradient_applied_cosine_mean`:
  選択した統合方式を適用した後の概念勾配対に残る競合率と平均cosine類似度。`mean`では適用前と一致し、
  `pcgrad`では射影後を表す。
- `backbone_gradient_update_cosine_mean`: 通常の重み付きmean更新と実際に適用した更新の方向一致度。
- `backbone_gradient_update_norm_ratio_mean`: mean更新に対する実適用更新のノルム比。
- `backbone_gradient_update_delta_ratio_mean`: mean更新からの差分ノルムをmean更新ノルムで割った相対変形量。

適用前の指標は方式間で同じ診断対象を比較するために残し、適用後の指標はPCGradが競合をどこまで除去し、
更新をどれだけ変形したかを調べるために併記する。

これにより、PCGradの精度差だけでなく、Circle2の負の転移と勾配競合率が対応するかを検証できる。
PCGradはローカルoptimizerへ渡す共有勾配だけを変更し、通信内容・サーバ集約・概念別head更新・
SoftRoutingは変更しない。一方、概念ごとの共有勾配を得るため、通常の平均より計算時間は増える。

実際の更新回数は次の指標で確認できる。

- `compute_backbone_optimizer_steps_total`: 共有バックボーンoptimizerの更新回数。
- `compute_head_optimizer_steps_total`: 概念別ヘッドoptimizerの更新回数。
- `compute_optimizer_steps_total`: 従来との比較用に維持する論理的な概念モデル更新回数。

## SoftRoutingの文脈

予測レイヤー全体と`global` / `predicted_class` / `meta_predicted_class`の関係は
[soft-routing.md](soft-routing.md)を参照する。この節では共有表現更新との相互作用に焦点を当てる。

`--soft-routing-context`は、AdaHedgeが蓄積するモデル別損失の共有範囲を指定する。

- `global`は全入力で一つの損失履歴を共有する従来方式である。
- `predicted_class`は、まず従来の大域ルータで事前予測クラスを求め、その予測クラス専用の
  AdaHedgeで最終混合重みを決める。正解ラベルは重み決定後の更新にだけ使うため、予測時の
  ラベル漏洩はない。
- `meta_predicted_class`は、global mixtureと予測クラス別AdaHedgeのleaderを、さらに文脈別の
  上位AdaHedgeで混合して実予測に用いる。

文脈方式は新しい数値ハイパーパラメータを持たない。共有表現がサーバ集約で変化した場合、
大域ルータは指定された再較正方式に従い、予測クラス別ルータは古い表現に依存する証拠を破棄して
次ラウンド内で学び直す。これは正解クラス別oracle診断をそのまま使う方式ではなく、実運用時に
観測可能な事前予測だけで文脈を構成する方式である。

### Shadow meta-router診断

`predicted_class`では実予測とは独立した診断として、`meta_predicted_class`では実予測器として、
上位AdaHedgeを予測文脈ごとに持つ。
上位ルータの専門家は、全入力で証拠を共有する`global mixture`と、文脈別AdaHedgeが選んだ
`contextual leader`の2つである。各標本では既に計算済みの出力を混合し、正解取得後に両候補の
有界損失で更新する。したがって追加のモデルforwardや通信は発生しない。

この診断は、global mixtureを基本としながら特定文脈だけcontextual leaderへ切り替える二段階方式に
改善余地があるかを測る。診断結果を受け、同じ計算を実予測へ用いる
`meta_predicted_class`も選択可能である。

## 集約後のルーティング再較正

共有バックボーンをサーバで集約すると、同じ概念ヘッドでも予測関数が変化する。一方、
AdaHedgeの累積損失は集約前の予測関数に対する証拠であり、そのまま保持すると現在のモデル間の
優劣を正しく表さない場合がある。

`--shared-backbone-routing-recalibration`で次を選択する。

- `none`（既定）: 累積損失を維持し、従来の共有バックボーン結果と同じ挙動にする。
- `aggregation_restart`: 各サーバ集約・配布の直後に累積損失とmixability gapだけを初期化する。
  モデル、検出器、学習データ、概念切替状態は変更しない。再較正周期には既存の集約間隔`A`を
  用いるため、新しい数値ハイパーパラメータは増えない。
- `fifo_replay`: 集約後モデルをFIFO内の最新データで再評価し、古い累積損失を、その評価から
  時系列順に再構築した証拠で置き換える。FIFOは検出遅延を吸収するために保持され、まだ過去の
  概念モデルへ確定投入されていない区間であるため、現在分布を表す較正集合として利用する。
  FIFOが空、または保持モデルが1個だけなら、有効な既存証拠を一律に消さず何もしない。
- `leader_change_replay`: 集約前までの累積損失が選ぶleaderと、集約後モデルをFIFOで評価した
  最良モデルを比較する。両者が同じなら累積損失とmixability gapをそのまま維持し、異なる場合だけ
  `fifo_replay`を行う。モデル集合が変わった場合は旧証拠を安全に対応付けられないためreplayする。
  同率時は現行モデルを優先する。新しい数値閾値は追加しない。
- `persistent_leader_change_replay`: FIFO全体で選んだchallengerが、時系列順に重複なく分けた
  前半・後半の両方で旧leaderよりpaired lossが小さい場合だけreplayする。片方だけの一時的な
  改善では旧証拠を維持する。新規モデル作成の`forward_persistent`と同じ持続性原則を用い、
  新しい数値閾値や検証窓長を追加しない。モデル集合が変わった場合は通常のreplayを行う。

再始動回数はCSVの`routing_aggregation_restart_count`と、NPZの
`routing_aggregation_restart_counts`で確認できる。この方式は特に`joint`で生じた
「oracle accuracyは高いがSoftRoutingが良いモデルを回収できない」問題を切り分けるための
実験条件であり、既定値にはしていない。

再較正を実施した回数と、再評価に実際に用いた標本数は、それぞれ
`routing_aggregation_recalibration_count`と
`routing_aggregation_recalibration_sample_count`で確認できる。`fifo_replay`は通信を追加せず、
集約・配布済みのローカルモデルと観測済みFIFOデータだけを使う。ただし、集約ごとに最大で
`N_FIFO × 保持モデル数`のヘッド評価が増える。この計算は`routing_recalibration_*`および
`compute_backbone_examples_total`、`compute_head_examples_total`へ含める。
`leader_change_replay`の判定回数と、leaderが変わらず証拠を維持した回数は、それぞれ
`routing_aggregation_recalibration_check_count`と
`routing_aggregation_recalibration_skip_count`へ記録する。replayを省略した場合でもleader比較の
FIFO評価は必要なため、計算量は`fifo_replay`とほぼ同じである。

## 仮モデル

正式採用済みの全概念ヘッドは同じバックボーンを参照する。`sequential`または`joint`では
共有部の更新が以降の全ヘッドへ反映される。ヘッドのパラメータはモデルIDごとに独立する。

仮モデルとforward検証用shadowは、採否判定前には独立バックボーンで学習する。これにより、
棄却された候補が既存モデルの共有表現を変更する情報漏洩を防ぐ。候補が正式採用された場合だけ、
候補の学習済みバックボーンを共有部へ反映し、新しい概念別ヘッドを接続する。

## サーバ集約と通信

NoCachedサーバは各クライアントから次を集約する。

1. 共有バックボーンをクライアントごとに1回アップロードし、そのクライアントが各モデルへ
   割り当てたデータ数の合計でFedAvgする。
2. 概念別ヘッドをモデルIDごとにアップロードし、そのモデルのデータ数でFedAvgする。
3. 配布時も共有バックボーンはクライアントごとに1回、ヘッドはモデルIDごとに送る。

クロス評価では同ラウンドの集約済み候補を評価する既存NoCachedプロトコルを維持するため、
候補の完全なバックボーン＋ヘッドを評価先へ送る。将来、評価先で同じ集約済みバックボーンを
一時キャッシュすればさらに削減できるが、今回の方式には含めない。

従来の`comm_models_*`は論理モデル転送回数として維持する。ただし構造が異なる方式間では、
次の指標を用いる。

- `comm_parameter_values_*`: 実際に転送したパラメータ値数。
- `comm_bytes_*`: dtypeを反映した実転送バイト数。
- `final_parameter_values` / `final_parameter_bytes`: 共有部の重複を除いた最終保持容量。
- `compute_backbone_examples_total`: 特徴抽出部が処理した延べサンプル数。
- `compute_head_examples_total`: 概念別ヘッドが処理した延べサンプル数。

## 比較上の注意

この方式はモデル構造、ローカル学習の共有範囲、FedAvgの単位を同時に変更するため、既存方式の
単なる高速実装ではなく独立した提案候補として扱う。比較ではaccuracy・stable accuracyだけでなく、
パラメータ転送量、最終保持容量、バックボーン計算量、ヘッド計算量を併記する。

## 既存研究との関係と研究上の位置づけ

特徴抽出部を共有し、出力部だけを対象別に分ける発想自体は新規ではない。中央集約型の
[Multi-Task Learning](https://doi.org/10.1023/A:1007379606734)におけるhard parameter sharingを
基礎として、個人化連合学習でも次のような関連手法がある。

- [FedPer](https://arxiv.org/abs/1912.00818)は、共通のbase層とクライアント固有の
  personalization層を分離する。
- [FedRep](https://proceedings.mlr.press/v139/collins21a.html)は、クライアント間で共有する
  representationとクライアント固有headを交互に学習する。
- [FedBABU](https://openreview.net/forum?id=HuaYQfggn5u)は、連合学習中にbodyを学習し、
  headを個人化段階で調整する。
- [FedCR](https://proceedings.mlr.press/v202/zhang23w.html)は、クライアント間の共通表現を
  学習しつつ、各クライアントが個別predictorを持つ。

これらの個人化連合学習では、分離単位は原則として「クライアント」である。本実装では、
バックボーンはクライアント内の複数概念モデル間で共有され、headはサンプル単位の検出によって
動的に発見された「概念」ごとに分かれる。headのモデルIDはサーバのクロス評価・クラスタリングを
経てクライアント間でも対応付けられ、予測時にはSoftRoutingで複数headをオンライン混合する。

非定常環境により近い関連研究として、
[FedWeIT](https://proceedings.mlr.press/v139/yoon21b.html)は、モデルをglobal federated parameterと
sparse task-specific parameterへ分解し、クライアント間で過去タスクの知識を選択的に転移する。
一方、本実装は既知のタスク境界や固定タスクIDを前提にせず、連続ストリームから変化を検出し、
概念モデルの生成・再利用・統合とルーティングを同じオンライン処理内で行う点が異なる。

したがって研究上の主張は「共有バックボーンそのもの」ではなく、次の組合せに置くべきである。

1. バッチ境界に依存しないサンプル単位の未知概念ドリフト検出。
2. 動的に増減・統合される概念別headと、クライアント間での概念ID対応付け。
3. 共有表現が集約で変化する状況に対応したオンラインルーティングと選択的再較正。
4. 独立モデル方式に対する精度、負の転移、計算量、保持容量、通信量の実証比較。

特に共有表現は、概念間の正の転移と計算・通信削減をもたらす一方、概念差の大きいデータでは
負の転移や表現容量不足を起こし得る。この利得と干渉をドリフト・モデル管理と同時に評価することが、
既存のクライアント個人化型手法との差を明確にする。
