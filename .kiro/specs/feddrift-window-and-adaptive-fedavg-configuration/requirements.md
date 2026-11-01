# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedDrift・FedAvg・Oracle・IFCA+Window（2026-11-01のユーザー判断で、余裕があればAdaptive-FedAvgを足す）。比較手法が少ないことが弱点で、FedDriftの公式の実装にあり、流用できて実装の費用が低い手法を足している。FedDriftの論文（AISTATS 2023）の主表は、Window（単一のモデルを直近1時刻のデータで学習）とAdaptive-FedAvg（Canonaco et al., IJCNN 2021。単一のモデルで、サーバが学習率を更新の変動に応じて調整）を比較している。公式の実装は`FedDrift/fedml_api/distributed/fedavg_ens/`の`FedAvgEnsAggregatorAda.py`・`FedAvgEnsTrainerAda.py`と、`FedAvgEnsDataLoader.py`の`AdaState`（公式の設定はWindowが`win-1`、Adaptive-FedAvgが`ada`＋`win-1_iter`）。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、IFCAの構成で足した学習の窓（1時刻）を再利用し、単一のモデルの構成を2つ足す。(1) Window（optimizer・学習率はFedDriftと同じ）、(2) Adaptive-FedAvg（(1)の上で、SGDとサーバの学習率の調整（`AdaState`の移植））。初期の学習率を掃引の軸として足す。既存の構成と既定は変えない。
- Windowを足す理由（2026-11-01、ユーザーと主担当の判断）: Adaptive-FedAvgとの差が学習率の調整だけなので実装の追加がほぼ無い。FedDriftの論文の主表で、Windowは7データのうち5つでAdaptive-FedAvgより高く、「単一のモデルで古いデータを忘れるだけ」の基準になる。

## Boundary Context
- 対象: 新しい2構成、初期の学習率の掃引の軸、それを守るtest、文書、調整と本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、研究ツール、Oracle・IFCAの構成。

## Requirements
### Requirement 1: 単一のモデル・窓1時刻の2構成に共通の挙動
#### Acceptance Criteria
1. The system shall 構成`feddrift-window`と`feddrift-adaptive-fedavg-window`を持つ。どちらも、runの最初に1個のグローバルモデルを持ち、その初期値は全手法に共通の事前学習した初期モデルの写しにする。
2. The system shall 全clientの全標本をそのモデル（ID 0）へ帰属させ、予測もそのモデルで行う。FedDriftの変化の検出・新しいモデルの作成・クラスタリング・統合は行わない。
3. The system shall 学習の標本を、今の時刻の標本だけにする（窓1時刻、IFCAの構成の`training_sample_window=current_timestep`と同じ規則）。学習のround数・件数の条件・集約の重み・通信の数え方は、FedDriftの構成`feddrift`と同じ規則にする（具体的な接続はdesign.mdに定める）。
4. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: Windowの学習
#### Acceptance Criteria
1. The system shall `feddrift-window`の局所の学習のoptimizerと学習率を、FedDriftの構成`feddrift`の既定（Adam、AMSGrad、weight decay 0.001、学習率はデータごとの既定。SINE2は0.01、MNIST4は0.001など）のままにする（FedDriftの論文の付録で、Adaptive-FedAvg以外の手法は同じ学習率を使う）。

### Requirement 3: Adaptive-FedAvgの学習率
#### Acceptance Criteria
1. The system shall `feddrift-adaptive-fedavg-window`の局所の学習を、momentumとweight decayの無いSGDにし、学習率を、サーバがその時刻の学習の前に決めた値 $\eta$ にする。
2. The system shall サーバが、各時刻の集約の後に、集約したモデルのパラメータから、design.mdの「学習率の更新の式」（公式の`AdaState.update`の移植、$\beta_1=\beta_2=\beta_3=0.5$）で次の時刻の $\eta$ を求める。最初の時刻の学習は初期の学習率 $\eta_0$ を使う。
3. The system shall $\eta_0$ を構成の設定の値（既定0.01、FedDriftの論文の付録の値）にし、掃引のオプション`--adaptive-fedavg-initial-learning-rates`で値を並べられるようにする（掃引の規約に従う）。

### Requirement 4: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 2構成が構成の一覧に含まれ、組立てが例外を出さず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 学習率の更新の部品が、公式の`AdaState`をtestの中に写した参照の実装と、design.mdに定める入力の列で一致することを確かめる。
3. The validation shall 小さい全体run（design.mdに定める固定条件）で、2構成それぞれの学習率（Windowは固定、Adaptive-FedAvgは調整）・学習の窓・モデルの数・統合を記録から確かめる（design.md）。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 5: 文書と調整・本規模の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、2構成と、FedDriftの論文・公式の実装のWindow・Adaptive-FedAvgとの違い（design.mdの「論文・公式の実装との違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、Windowの本規模の評価、Adaptive-FedAvgの $\eta_0$ の調整（design.mdの「調整」）と本規模の評価のコマンドをユーザーに渡す。集計はdesign.mdの「本規模の集計」。
