# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedAvg、FedAvg+Window、Adaptive-FedAvg、IFCA+Window、FedDrift、DriftSurf、Oracle（FedDrift）。1つのモデルの手法はこの設定では原理的に弱く、IFCA+WindowはSINE2以外で1つのクラスタに退化した。FedDriftの論文（AISTATS 2023）の表1で、複数のモデルを持つ比較手法のうち強かったCFL+Window（SINE-2で96.9、CIRCLE-2で96.0）が未実装。2026-11-03のユーザー判断で足す。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、FedDriftの公式の実装の`softcluster`＋`cfl_0.1_win-1`（Clustered Federated Learning、Sattler et al.）に準じた構成`feddrift-cfl-window`を1つ足す。IFCAの構成（`.kiro/specs/feddrift-ifca-configuration/`）で足した学習の窓（`training_sample_window=current_timestep`）を再利用する。既存の構成と既定は変えない。

## Boundary Context
- 対象: 新しい1構成、それを守るtest、文書、本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、研究ツール、ほかの比較手法の構成。

## Requirements
### Requirement 1: CFL+Windowの構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-cfl-window`を持つ。runの最初はモデル1つ（ID 0、全手法に共通の事前学習の初期モデル）で、全clientをID 0のクラスタに割り当てる（公式の`cluster_init`）。モデルの上限はデータのコンセプトの数 $K$（`dataset_definitions`の`concept_count`。公式の`concept_num`と同じ扱い）。
2. The system shall clientのクラスタへの割り当てを時刻をまたいで保ち（公式の`cluster_cfl_init`が前の時刻の割り当てを写す）、各clientは割り当てられたモデルで予測し、そのモデルを今の時刻の標本だけで学習する（窓1時刻、`win-1`）。clientの割り当てが変わるのは、次の分割のときだけである。
3. When サーバが集約する, the system shall 公式の`cluster_cfl`と同じ規則で、使われている各クラスタ（モデルIDの昇順）について、そのクラスタのclientのうち学習の標本が1件以上のclientのパラメータの更新（clientの局所の学習の後のパラメータ−配布したパラメータ）から、平均の更新のノルムと最大の更新のノルムを求め、閾値の更新・分割の判定・2分割（余弦類似度の完全連結の階層的クラスタリング）・分割の条件（$\sqrt{(1-\alpha_{\mathrm{cross}})/2}>\gamma$、$\gamma=0.1$）を適用する（具体はdesign.md）。
4. When 分割が起きる, the system shall 分割したクラスタのモデルと新しいモデルを、どちらも全手法に共通の事前学習の初期モデルの写しにし（公式の乱数の初期化から変える点。design.mdの「元の実装との違い」の(c)）、片方のclientを元のIDに、もう片方を新しいIDに割り当てる。分割が1つでも起きたroundは、全モデルの集約を行わず、そのroundのclientの更新を1つも反映しない（公式どおり）。新しいIDは0〜$K-1$を順に払い出し、モデルの数が上限 $K$ に達していれば分割しない（公式の`find_unused_model_capped`、`model_num=concept_num`）。
5. The system shall 学習のround数、学習の対象の規則、件数の条件、集約の重み、通信の数え方、学習率を、IFCAの構成と同じくFedDriftの構成`feddrift`の規則にする（具体はdesign.md）。FedDriftの変化の検出・新しいモデルの作成・統合は行わない。
6. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 構成が構成の一覧に含まれ、組立てが例外を出さず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 分割の判定の部品を、公式の`cluster_cfl`・`cfl_util_*`を写した参照の実装（2分割は公式の`AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)`と同じ結果になる実装）と、design.mdに定める入力の列で照合する。
3. The validation shall 小さい全体run（design.mdに定める固定条件）と、分割が決定的に起きる入力でのサーバの集約の経路のtest（必須）で、割り当ての時刻をまたぐ保持と分割の後の切り替え・学習の窓（その時刻の標本だけ）・予測に使ったモデル・分割の記録・分割したroundの集約の省略（clientの更新が反映されない）・分割で使うモデルの初期値を確かめる。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と本規模の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、構成と、FedDriftの論文・公式の実装のCFL+Windowとの違い（design.mdの「元の実装との違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、まず予備の実行（design.mdの「本規模の実行」、24 run）のコマンドをユーザーに渡し、1時刻1roundで分割が機能するかの集計をユーザーに示す。ユーザーが進めると決めたら、10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本（400 run）のコマンドを渡す。$\gamma$ はFedDriftの論文の値0.1に固定し、調整しない。コマンドには目的・ラベル・workersを付け、開始と結果を実験の台帳に記録する。集計はdesign.mdの「本規模の集計」。
