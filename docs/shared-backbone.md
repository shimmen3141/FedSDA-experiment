# 共有バックボーン＋概念別ヘッド

## 共有範囲

共有表現には、次の二つの実験modeがある。

- `FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting`: 全隠れ層を共有し、出力headだけを概念別にする。
- `FedSDA_NoCached_PartialSharedAdapter_ClassESR_RestartingSoftRouting`: 先頭隠れ層だけを共有し、後段adapterと出力headを概念別にする。

部分共有方式は、全共有で生じた概念間の負の転移を抑えつつ、入力に近い低層表現の学習量と通信量を共有するための方式である。
合成データの二層MLPでは、元の第1隠れ層を共有部、第2隠れ層を概念別adapterとして使う。
MNIST2/MNIST4の一層MLPでは、共有隠れ層の後に特徴ごとのscaleとbiasからなる概念別adapterを置く。
このアフィンadapterには幅のハイパーパラメータがなく、概念ごとの大きな全結合層の複製を避けられる。

通信と集約では、共有部をクライアントごとに一度だけ転送・FedAvgし、adapterとheadをモデルIDごとに転送・FedAvgする。
既存の`compute_head_*`指標は比較可能性を保つため、部分共有方式ではadapterとheadを合わせた概念固有部分を数える。

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

実際の更新回数は次の指標で確認できる。

- `compute_backbone_optimizer_steps_total`: 共有バックボーンoptimizerの更新回数。
- `compute_head_optimizer_steps_total`: 概念別ヘッドoptimizerの更新回数。
- `compute_optimizer_steps_total`: 従来との比較用に維持する論理的な概念モデル更新回数。

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
