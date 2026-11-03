# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法（FedAvg、FedAvg+Window、Adaptive-FedAvg、IFCA+Window、FedDrift、Oracle）は、学習の割り当ても予測も1つのモデルに結び付けるか、単一のモデルのものだけである。先行研究の調査（`docs/research/FedHASP の先行研究チェック：分散・再帰的 concept drift 下の federated learning.md`、main checkout）で、学習も予測も柔らかく行う系統（FedEM、FedRC）がFedHASPに近いと判定された。
- 2026-11-04のユーザー判断: 有望な比較手法を1つ足す。FedEM（Marfoq et al., NeurIPS 2021）を、FedDriftの実装の上へ窓を足して移す。公式の実装は`C:\Users\yshin\Research\FedSDA-experiment\FedEM`（`omarfoq/fedem`、`996fadd`）。問題設定が異なるので、合わせ方で不当に弱くしない。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、FedEMの4構成（初期化2通り×学習の標本の範囲2通り）を足す。予備の実行で、論文の表に載せる1構成を選ぶ。既存の構成と既定は変えない。

## Boundary Context
- 対象: 新しい4構成、1時刻のround数の掃引の値を4構成でも使えるようにすること、記録、test、文書、予備の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、FedHASP（`fedsda`）、研究ツール。予測の重みを集約の間に更新するなど、FedEMに無い仕組みは足さない。

## Requirements
### Requirement 1: FedEMの4構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-fedem-window`、`feddrift-fedem-window-pretrained-component0`、`feddrift-fedem-history`、`feddrift-fedem-history-pretrained-component0`を持つ。どれも、成分モデルの数 $M$ をデータのコンセプトの数（`dataset_definitions`の`concept_count`）とし、runの最初に $M$ 個のグローバルモデルを持つ（$M$ を正しく与えることはFedEMに有利な条件で、文書と論文に書く）。
2. The system shall モデルの初期値を、IFCA+Windowの2構成と同じ規則にする。`-pretrained-component0`の付かない2構成は $M$ 個とも独立に乱数で初期化し（FedEMの公式の実装の初期化に当たる）、付く2構成はID 0だけを全手法に共通の事前学習した初期モデルの写しにし、ID 1〜$M-1$は前者と同じ乱数の初期化にする。4構成とも初期の $\pi_c$ は一様。初期化でrunの乱数を消費しない。
3. The system shall 各clientに、成分の重み $\pi_c$（長さ $M$、初期値は一様）を持たせる。各roundの学習の前に、そのclientの学習の標本の各標本 $i$ について、事後の重み $q_{i,m}\propto\pi_{c,m}\exp(-\ell_m(x_i,y_i))$（$\ell_m$ はモデル $m$ の交差エントロピーの損失、$m$ について正規化）を求め（E-step）、$\pi_c$ を $q_{i,m}$ の平均にする（公式の`MixtureClient`の`update_sample_weights`・`update_learners_weights`）。平均を取る標本は、その時刻に届いた標本とする（`-window`では公式と同じ。`-history`での理由はdesign.md）。
4. The system shall 各roundで、全 $M$ 個のモデルのそれぞれを、そのclientの学習の標本で、損失を $q_{i,m}$ で重み付けて学習する（M-step、公式の`learner.fit_batch`の重み付きの損失）。学習の予算・学習率・optimizer・round数は、FedDriftの構成`feddrift`と同じ規則にする（具体的な接続はdesign.md）。
5. The system shall サーバで、各成分モデルを、clientの学習の標本の件数の割合で重み付けて平均する（公式の`CentralizedAggregator`の`clients_weights`。$q$ の和では重み付けない）。全clientが全 $M$ 個のモデルを上りと下りで送り、通信を数える。$\pi_c$ と $q$ は送らない。
6. The system shall 各標本を、その時点のモデルの予測確率を $\pi_c$ で重み付けて混ぜた確率で予測する（公式の`evaluate_iterator`）。$\pi_c$ は次のroundのE-stepまで変えない。
7. The system shall 学習の標本の範囲を構成ごとに次のとおりにする。`-window`の構成: E-stepとM-stepの対象は、その時刻の標本だけ（窓1時刻。時刻の初めに空にする）。`-history`の構成: E-stepとM-stepの対象は、そのclientがrunの初めから受け取った全標本。どちらの構成でも、予測に使う $\pi_c$ は、その時刻に届いた標本の $q$ の平均（要求1.3）。
8. The system shall 1時刻のround数の掃引の値`--feddrift-local-training-round-counts`を、4構成でも使えるようにする（既定は1）。既存の2構成の扱いと、ほかの構成の拒否は変えない。
9. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 記録
#### Acceptance Criteria
1. The system shall 各時刻・各clientの $\pi_c$（最後のroundのE-stepの後）と、その時刻の標本の $q$ の最大の成分を記録する（表の名前と列はdesign.md）。

### Requirement 3: test
#### Acceptance Criteria
1. The validation shall E-stepと重み付きの損失が、公式の実装の式（`client.py`の`MixtureClient`、`learners/learner.py`の`fit_batch`）と同じ値になることを、公式の該当部分の写しとの照合で確かめる。
2. The validation shall 4構成の初期値が要求1.2のとおりであることを確かめる。
3. The validation shall 小さい全体run（design.mdに定める固定条件）で、学習の標本の範囲、全モデルの学習と集約、集約の重み、予測の混合、通信、$\pi_c$ の記録を確かめる。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 4: 文書と予備の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、4構成と公式のFedEMとの違い（design.mdの「公式のFedEMとの違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、予備の実行（design.md）のコマンドをユーザーに渡し、集計を示す。表に載せる構成とround数はユーザーが決める。
