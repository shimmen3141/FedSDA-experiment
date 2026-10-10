# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装は、全体runの間のモデルの計算を、外側から数えられる（model-computation-measurement）。共有部・概念固有部を通った標本数、全結合層の積和演算の数（順伝播は実測、逆伝播は見積り）、optimizerの更新回数、検出器の計数が、run全体の合計として得られる。goldenの33指標は、すべて照合済み。

ユーザーの決定（2026-10-10）: 論文で言いたいことは、(a)「FedSDAは、FedDriftより、clientの計算が少ない」（結果に従う）、(b)「clientが保有する1モデルあたりの計算量」（保有モデルが多いだけの増加と区別する。モデルの数は、常に変わる）、(c)「モデル数が増えても、計算が増えにくい」。サーバの計算にも触れる。単位は、積和演算にそろえる。

足りないもの:

1. **サーバの計算。** サーバは、モデルの順伝播を行わない（評価はclientが行う）。サーバの計算は、パラメータの算術である: 集約（clientのパラメータの加重和）、統合（クラスタのメンバーのパラメータの加重平均）、クラスタリングの診断のパラメータ距離。計数がない。
2. **保有モデル数。** 「1モデルあたり」で割るための、client・標本ごとの保有モデル数の記録がない。
3. **ラウンドごとの系列。** 計算量と保有モデル数の関係（傾き）を見るための、ラウンドごとの計算量がない。
4. **まとめ。** 上の主張に対応する量（合計、標本あたり、モデル×標本あたり）を、計数から計算する処理がない。

調査で確かめたこと:

- 集約（runtime/server_model_registration_and_aggregation.py）は、clientごとに、共有部を1回と、参加モデルの概念固有部を、件数で重み付けて足す（`_add_weighted_parameters`）。足したパラメータは、通信量の「上りのパラメータの値の数」に数えたものと、同じである（上りのパラメータを数えるのは、集約だけ）。
- 統合（runtime/model_clustering_and_consolidation.py）は、統合するクラスタごとに、重みが正のメンバーの、完全なパラメータ（共有部＋概念固有部）を、重み付けて足す。パラメータ距離は、モデルの対ごとに、概念固有部の、差の2乗和と、2つの2乗和を計算する（診断用。判定には使わない）。
- 旧は、ラウンドごとの、clientの計数の増分（`round_client_*`。用途別を含む）と、ラウンドの後の保有モデル数（`round_client_held_model_count`）を保存する。旧の予測の計数（`prediction_examples`）は、標本ごとに、保有する全モデルを通すので、client・標本ごとの保有モデル数の合計に等しい。これらを、照合に使える。旧には、サーバの計算の計数はない。

## Introduction

研究者が、新実装の全体runから、手法の比較に使う計算量（clientとサーバ、標本あたり、保有モデルあたり、ラウンドごと）を得られるようにする。

## Boundary Context

- **In scope**: サーバのパラメータの積和演算の数（集約、統合、診断のパラメータ距離）。clientの、標本ごとの保有モデル数の記録。計測つきの全体runの、ラウンドごとのモデルの計算（ローカルの処理と、同期の別）。計算量のまとめ（合計と、標本あたり・モデル×標本あたりの値）と、指標への追加。
- **Out of scope**: 結果の保存（NPZ・CSV）と図、回帰（傾きの推定）。FedDriftの計測（手法の移植の後）。実行時間。検出器の計算の、積和演算への換算。通信量（既存）。sine2以外のdataset。
- **Adjacent expectations**: 計測、集約、統合、clientの標本処理の結果（状態、乱数、指標）は、変えない。固定旧実装とgoldenは、変えない。

## Requirements

### Requirement 1: サーバのパラメータの積和演算

**Objective:** As a 研究者, I want サーバの計算を、clientと同じ単位で得たい, so that サーバの負荷にも触れられる

#### Acceptance Criteria

1. When 集約を行ったとき, the Client Model Aggregation shall 重み付きの和へ足したパラメータの値の数（clientごとの共有部と、参加モデルの概念固有部。値1つの「重みを掛けて足す」を1と数える）を、結果に含める。
2. When クラスタリングと統合を行ったとき, the Model Consolidation shall 統合の加重平均で、重み付きの和へ足したパラメータの値の数と、診断のパラメータ距離の積和演算の数（対ごとに、概念固有部の値の数の3倍——差の2乗和と、2つの2乗和）を、別々に、結果に含める。
3. The 計数 shall 割り算、写し、通信、損失統計の平均を、含めない。
4. The 追加 shall 集約と統合の結果（グローバルモデル、統計、通信量、診断の記録、乱数）を変えない。

### Requirement 2: 保有モデル数の記録

**Objective:** As a 研究者, I want client・標本ごとの保有モデル数を得たい, so that 「1モデルあたりの計算量」を、保有していた期間に応じて計算できる

#### Acceptance Criteria

1. When clientが標本1件を処理したとき, the FedSDA Run Client shall その標本の予測の時点（処理の前）の保有モデル数を、標本の順に、1件記録する。
2. If 標本の処理が失敗したとき, the FedSDA Run Client shall 記録を足さない。
3. The 記録の合計 shall 実旧の、予測でモデルへ入力した標本の数（`prediction_examples`）と一致する。

### Requirement 3: ラウンドごとのモデルの計算

**Objective:** As a 研究者, I want ラウンドごとの計算量を得たい, so that 計算量と保有モデル数の関係を見られる

#### Acceptance Criteria

1. When 計測つきの全体runを実行したとき, the Measured Run Execution shall ラウンドごとに、ローカルの処理（標本の処理と、区間末の学習）の間のモデルの計算と、同期（サーバの同期の間に、clientが行う評価と再較正）の間のモデルの計算を、別々に返す。
2. The Measured Run Execution shall 最後のラウンドの後（終端の処理）のモデルの計算も返す。ラウンドごとの値と、終端の値の合計は、run全体のモデルの計算（準備の後から終わりまで）と一致する。
3. The ラウンドごとの値 shall 実旧のラウンドごとの計数の増分（全clientの合計。ローカル＝予測・検出・統計・初期化・学習、同期＝クロス評価・再較正）と一致する。
4. The 追加 shall 全体runの結果（状態、乱数、指標）を変えない。

### Requirement 4: 計算量のまとめ

**Objective:** As a 研究者, I want 主張に対応する量を、同じ式で得たい, so that 手法とdatasetの間で、比べられる

#### Acceptance Criteria

1. When モデルの計算の計数・サーバの計数・処理した標本数・保有モデル数の合計を与えたとき, the Computation Cost Summary shall 次を返す: clientの順伝播の積和演算の数（推論＋学習）、clientの逆伝播の見積り、共有部と概念固有部の内訳、サーバの積和演算の数（集約＋統合。診断は別）、平均の保有モデル数、標本1件あたり・保有モデル×標本1件あたりの、clientの積和演算の数（順伝播だけと、逆伝播を含むもの）。
2. If 分母（処理した標本数、保有モデル数の合計）が0のとき, the Computation Cost Summary shall その比を「なし」（NaN）にする。
3. If 入力が、決まった型・範囲でないとき, the Computation Cost Summary shall 拒否する。
4. When 全体runの指標を導出するとき, the FedSDA Run Metric Derivation shall サーバの計数、処理した標本数、保有モデル数の合計を、指標へ含め、モデルの計算の計数が渡されたときは、計算量のまとめも含める。

### Requirement 5: 実旧との照合と、新実装だけで動くこと

#### Acceptance Criteria

1. When goldenの条件と、小さい条件で実行したとき, the 照合 shall 保有モデル数の合計（旧の予測の計数）、ラウンドごとのモデルの計算（旧のラウンドごとの計数）、集約の積和演算の数（通信量の、上りのパラメータの値の数）を、実旧の値と比べる。
2. The 統合とパラメータ距離の計数 shall 診断の記録と、モデルの層の形からの手計算と一致する。
3. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、計算量のまとめを得られる。
4. The 追加 shall 依存の許可の範囲に収め、goldenの33指標の照合を、そのまま通す。
