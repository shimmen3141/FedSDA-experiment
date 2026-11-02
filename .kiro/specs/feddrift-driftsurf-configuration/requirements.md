# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedAvg・FedAvg+Window・Adaptive-FedAvg・IFCA+Window・FedDrift・Oracle。論文の関連研究で引用しているDriftSurf（Tahmasbi et al., ICML 2021）は、FedDriftの論文の主表で「FedAvgの上でサーバに置いた従来の変化への適応」の検出の系統の代表として比べられており、公式の実装がある（`FedDrift/fedml_api/distributed/fedavg_ens/`の`FedAvgEnsDataLoader.py`の`DriftSurfState`・`DriftSurf_data_loader`、`FedAvgEnsAggregatorDriftSurf.py`。設定`dsurf`）。2026-11-03のユーザー判断で、spec→実装（GPT-6.1-sol）で足す。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、DriftSurfの構成を1つ足す。サーバが、予測・安定・反応の役割の大域モデルを持ち、全clientの最新の時刻のデータでの正解率から公式の`run_ds_algo`の状態遷移で、学習するモデルと予測するモデルを決める。δを掃引の軸として足す。既存の構成と既定は変えない。

## Boundary Context
- 対象: 新しい1構成、δの掃引の軸、それを守るtest、文書、調整と本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、研究ツール、ほかの比較手法の構成。

## Requirements
### Requirement 1: DriftSurfの構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-driftsurf`を持つ。サーバは役割`pred`（予測）・`stab`（安定）・`reac`（反応）の大域モデルを持ち、runの最初は`pred`と`stab`を全手法に共通の事前学習した初期モデルの写しにする（design.md）。
2. The system shall 全clientが、同じ大域モデル（サーバが決めた予測の役割のモデル）で予測する。client別のモデルの割り当て、FedDriftの変化の検出・作成・クラスタリング・統合は行わない。
3. When 各時刻の終わりに全clientのその時刻の標本がそろう, the system shall 学習の前に、各clientが役割のモデルのその時刻の標本での正解数と件数を数え、サーバが全clientで合計した正解率で、公式の`DriftSurfState.run_ds_algo`の状態遷移（design.mdの「状態遷移」）を行う。
4. The system shall 各役割のモデルを、その役割の学習の時刻の窓（直近 $w=10$ 時刻、公式の`win_len`）に入る時刻の標本で、FedDriftの学習・集約の規則で学習する（design.md）。
5. The system shall δを構成の設定の値にし（既定0.10）、掃引のオプション`--driftsurf-accuracy-drop-thresholds`で値を並べられるようにする（掃引の規約に従う）。反応の状態の長さ $r=3$ と窓 $w=10$ は固定する（FedDriftの論文の付録と公式の既定）。
6. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 構成が構成の一覧に含まれ、組立てが例外を出さず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 状態遷移の部品が、公式の`DriftSurfState.run_ds_algo`をtestの中に写した参照の実装（正解率の計算だけを差し替える）と、design.mdに定める正解率の列で、状態・学習する役割・予測の役割・窓・`acc_best`・昇格が一致することを確かめる。
3. The validation shall 小さい全体run（design.mdに定める固定条件）で、正解率の合計・状態遷移・学習の窓・予測のモデル・モデルの数を記録から確かめる。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と調整・本規模の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、構成と、FedDriftの論文・公式の実装のDriftSurfとの違い（design.mdの「論文・公式の実装との違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、δの調整（design.mdの「調整」）と本規模の評価のコマンドをユーザーに渡す。集計はdesign.mdの「本規模の集計」。
