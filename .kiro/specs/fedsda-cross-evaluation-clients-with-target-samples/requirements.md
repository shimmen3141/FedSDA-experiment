# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を設計・評価している研究者（ユーザー）。
- 今の状況: サーバのクロス評価は、対象のモデルjを「保有する」clientから最大3つを無作為に選び、候補のモデルをjの標本で評価させる。「保有する」には、配布されたjのパラメータを持つだけで、jの標本（評価標本・学習データ）を持たないclientも含まれる。そうしたclientは標本を0件しか返さないので、登録の後の同期では、対の損失の統計が下限（5件）に届かず、判定の値を持てない。統合は、構造上ほぼ登録の同期でしか起きず、同じ概念の重複が残る（CIRCLE2では、サーバが持つモデルの数が多く、通信が増える）。旧から引き継いだ挙動（[LEGACY-022](../../../docs/research/implementation-findings/legacy-022-cross-evaluation-selects-clients-without-target-samples.md)）。
- 何を変えるか: クロス評価で、対象のモデルを評価させるclientの候補を、その対象の標本を返せるclient（clientが標本を選ぶときと同じ条件）に限る扱いを、製品の経路に正式な設定として足し、既定にする。基準版12を足す。

## Introduction
2026-10-23、研究ツールの変種`product-default-sample-holding-cross-evaluation`（基準版11の既定の上で、評価のclientを絞る）の本規模（研究室サーバ、6データ×間隔4つ×seed 0〜9、MNIST4はseed 0〜19）で、精度の差は−0.14〜+0.11pp（24条件のうち15で既定以上。MNIST2は4間隔とも+0.02〜+0.11pp、MNIST4は20 seedで−0.01〜+0.04pp）、下りはCIRCLE2で0.67〜0.81倍、SINE2で0.86〜0.93倍、CIRCLE2の最終のモデル数は5.1〜6.9→2.5〜2.9になった（[IMPROVE-028](../../../docs/research/improvement-candidates/improve-028-evidence-accumulating-model-consolidation.md)の9）。ユーザーは、これを不具合の修正として、本規模の確認の後に独立したspecで既定にすると決めた（「ほとんど不具合の修正だと思うので、どこかの段階では対応したい」、2026-10-23に「その順番で進めてください」）。testは、重要なもの・後で効くもの・複雑なものに絞る（ユーザー指示）。

設定の値と呼び方の対応:

| 設定`cross_evaluation_client_selection_policy`の値 | この文書での呼び方 |
|---|---|
| `clients_holding_target_model` | 「対象のモデルを保有するclient」（今の扱い。基準版11まで） |
| `clients_with_target_model_samples` | 「対象のモデルの標本を返せるclient」（新しい既定。基準版12） |

用語:
- **対象のモデルの標本を返せるclient**: 対象のモデルjを保有モデルとして持ち（正誤の比較にjが要る）、かつ、jの評価標本を下限（6件）以上持つか、jが現在の学習帰属で、jの学習データを下限（11件）以上持つclient。標本の条件は、clientが、クロス評価の依頼に、標本を選ぶ規則（`evaluate_candidate_model_on_target_model_samples`）と同じ。研究ツールの変種は標本の条件だけを使うが、最終構成では、配布で全clientが全グローバルモデルを保有するので、2つは同じ候補を作る見込み（Requirement 5.1の照合で確かめる）。
- **対象のモデルを保有するclient**: 今の候補（評価標本・学習データ・保有モデルのどれかにjを持つか、jが現在の学習帰属のclient。`FedsdaRunClient.get_cross_evaluation_held_model_ids`）。

## Boundary Context
- 対象: クロス評価で評価させるclientの選び方の設定（新しい項目）と計算、clientの「標本を返せるか」の判定（標本の選び方と同じ規則を1か所で持つ）、最終構成の既定、構成の扱い（候補検証ありの構成は今の扱いのまま）、旧と比べる設定の入口、基準版のgolden（版1〜11の明示と、版12の追加）、既定に依るtest、研究ツールの変種の扱い（既定が変わった後も、意味が変わらないようにする）、文書。
- 対象外: クロス評価の評価そのもの（標本の選び方、上限、正誤と損失の集計）、クラスタリングの判定と閾値、統合、クラスタリングを行う同期の規則（新しいモデルの登録がある同期だけ）、通信量の数え方（メッセージの数は、評価させたclientの数で数える今の規則のまま）、FedDrift・FedAvg、固定旧と旧golden、研究ツールの拒否権の変種。

## Requirements
### Requirement 1: クロス評価で、対象の標本を返せるclientだけを評価させる扱いを選べる
**Objective:** As a 研究者, I want 標本を返せないclientを評価に選ばない扱い, so that 登録の後の同期でも、対の判定に必要な証拠が集まり、同じ概念の重複が統合される。
#### Acceptance Criteria
1. The FedSDA system shall FedSDAの参加者の設定に、クロス評価で評価させるclientの選び方を選ぶ項目を持つ（値は「対象のモデルを保有するclient」——今の扱い——と「対象のモデルの標本を返せるclient」の2つ）。
2. When 「対象のモデルの標本を返せるclient」が選ばれている, the FedSDA system shall 各対象のモデルについて、評価させるclientの候補を、そのモデルの標本を返せるclientに限り、候補の列を、クロス評価に渡されたclientの列（`run_clients`）の順で作る（今の扱いと同じ順）。候補の数が上限（1つのモデルを評価するclientの上限）以下なら、候補の全部を、その順で評価させ、乱数を消費しない。上限を超えるときだけ、今と同じ乱数で、その候補の列から上限の数を`sample`する。候補が無い対象のモデルは、どのclientにも評価させない（その組のメッセージは0件）。
3. The FedSDA system shall 「標本を返せるか」の判定の標本の条件を、clientがクロス評価の依頼に標本を選ぶ規則と同じ1つの定義から作る（2つの規則が食い違わない）。判定はclientの状態を変えず、乱数を消費しない。
4. When 「対象のモデルを保有するclient」が選ばれている, the FedSDA system shall 基準版11と同じ値・同じ乱数の消費・同じ通信量・同じ計算回数で処理する。
5. When 「対象のモデルの標本を返せるclient」が選ばれている, the FedSDA system shall クロス評価の記録（`client_cross_evaluation`）に、評価した標本の数が0件の記録を作らない（候補は対象のモデルを保有し、5件以上の標本を選ぶので、評価は必ず行われる）。

### Requirement 2: 既定を「対象のモデルの標本を返せるclient」にする
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定を「対象のモデルの標本を返せるclient」にし、候補検証ありの構成（`fedsda-with-candidate-validation`）を除く、FedSDAの全部の構成が引き継ぐ。候補検証ありの構成は、今の扱い（「対象のモデルを保有するclient」）を明示する（候補検証との組合せは未検証のため）。全部の構成の組立てが例外を出さず、実験のコマンドの引数を変えない。

### Requirement 3: これまでの基準と照合を保つ
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、変わらず成功させる。旧と比べる設定の入口と、実旧との照合に使う設定の組は、今の扱いを明示する。
2. The validation shall 基準版1〜11のgoldenを、今の扱いで、値を変えずに照合し続ける（全部の版を、既存の版ごとの設定の組立てと照合のtestで確かめる）。

### Requirement 4: 新しい既定を基準版12として足す
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版12のgoldenとして、Windowsの基準環境とWSLの両方で足す（ケースは版11と同じ22ケース）。
2. The validation shall 版11と12の差を、両方の版を実行した結果を比べて確かめる: FedDrift・FedAvgのケースと、候補検証ありの構成のケースは、指標・最終のパラメータ・記録の表・経路の記録（版12で足す項目を取り除いたもの）が版11と同じ。新しい既定を引き継ぐFedSDAのケースは、設定が新しい項目だけ違う。
3. The validation shall 経路の記録に、クロス評価の記録の行の数と、そのうち評価した標本が0件の行の数を足し（版12から。FedSDAのケースだけ）、版12で新しい既定を引き継ぐFedSDAのケース（版12のFedSDAのケースのうち、候補検証ありの構成で実行する`fedsda-sine2-events`・`fedsda-mnist2-decisions`・`fedsda-sine2-reference-reuse`・`fedsda-sine2-alternative-reference-reuse`を除く全部）で、0件の行の数が0であることを確かめる（必須の経路。記録の行が0のケースも、0件の行の数は0）。そのうち、統合が起きる`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`では、クロス評価の記録の行が1以上であることも必須の経路として確かめる（比べ方が空でない）。版11のgoldenの条件で、標本が0件の評価が起きているFedSDAのケースが少なくとも1つあることを、版11の実行で確かめる。版11までの必須の経路は、版12でも求める。

### Requirement 5: 研究で評価した値と同じであることを確かめ、研究ツールの意味を保つ
#### Acceptance Criteria
1. The validation shall 研究ツールの変種（`product-default-sample-holding-cross-evaluation`。基準版11の既定の上で評価のclientを絞る）と、新しい扱いを選んだ製品の全体runが、同じ条件で、精度・通信量（下り・上りのバイト数と、下り・上りのメッセージの数。クロス評価の依頼と返信のメッセージは、評価させたclientの数で数えるので、候補の選び方の違いを捉える）・予測の記録の要約・最終のパラメータの要約・計算回数の指標（`model_computation_counts`と`preparation_model_computation_counts`）が一致し、クロス評価の呼出しごと・対象のモデルごとの、上限を当てる前の候補のclientの列と、実際に評価させたclientの列が一致し、標本の条件を満たすが対象のモデルを保有しないclientが一度も無いことを、複数のデータセット（MNISTとSINE2・CIRCLE2・SEAのうち少なくとも1つずつ）で確かめる（既定を変える前に、基準版11の既定の上で）。
2. The research tool shall 既定を変えた後も、既存の変種が、基準版10の既定の上で評価したときと同じ意味を持つようにする（研究ツールの基点の固定に、新しい項目の今の扱いを足す）。`product-default`で始まる変種は、製品の既定（基準版12）を使う。

### Requirement 6: 文書を合わせる
#### Acceptance Criteria
1. The documentation shall 手法の説明（クロス評価のclientの選び方）、基準の文書（版12）、実験の入口、研究記録（LEGACY-022の状態、IMPROVE-028）、再開案内で、新しい扱いを書く。
