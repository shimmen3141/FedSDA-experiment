# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedDriftとFedAvgだけで、査読で弱いと見られやすい（2026-11-01のユーザー判断）。FedDriftの論文（AISTATS 2023）は、真のコンセプトのIDを知っている理想の手法Oracleを上限として載せている。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、学習の割り当てを真のコンセプトで行う構成`feddrift-oracle-concept-assignment`を足す。FedDriftの複数モデルの学習・集約・通信の扱いはそのまま使い、割り当てだけを真のコンセプトにする。既存の構成と既定は変えない。
- 背景: 2026-11-01のユーザー判断（OracleとIFCA、余裕があればAdaptive-FedAvgを、順にspecを作りGPT-6.1-solに実装させる）。

## Boundary Context
- 対象: 新しい構成、その構成を守るtest、文書、本規模の実行のコマンド。
- 対象外: FedSDA・FedDrift・FedAvgの既存の構成の設定・値・runの識別子・golden、研究ツール。IFCAとAdaptive-FedAvgは別のspec。

## Requirements
### Requirement 1: 真のコンセプトで割り当てる構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-oracle-concept-assignment`を持つ。データのコンセプトの数を $K$ とすると、runの最初に $K$ 個のグローバルモデルを持ち、モデル $k$ をコンセプト $k$ に対応させる。全部のモデルは、全手法に共通の事前学習した初期モデルの写しから始める。
2. When クライアントがラベルを見た標本を学習に回す, the system shall その標本を、その標本の真のコンセプトに対応するモデルの学習の標本にする（1つの時刻の中でコンセプトが変わっても、標本ごとに振り分ける）。
3. When クライアントが標本を予測する, the system shall そのクライアントが直前にラベルを見た標本の真のコンセプトに対応するモデルで予測する（コンセプトのIDはラベルと一緒に明かされ、まだ見ていない標本のコンセプトは使わない）。最初の標本は、最初のコンセプトのモデルで予測する。
4. The system shall FedDriftの変化の検出・新しいモデルの作成・統合（クラスタリング）を行わない。学習（時刻ごとのR回のround、学習の対象のモデル、標本数の重み）と集約、通信の数え方、事前学習、学習率などは、FedDriftの構成`feddrift`と同じにする。
5. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 構成の一覧に含まれ、組立てが例外を出さず、goldenの事例にしない構成の一覧（`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 小さい全体run（2コンセプトのデータと4コンセプトのデータ、各1つ）で、(a) 各標本の学習の行き先のモデルが、その標本の真のコンセプトのモデルであること、(b) 各標本の予測に使ったモデルが、そのクライアントの直前の標本の真のコンセプトのモデルであること（最初の標本は最初のコンセプトのモデル）、(c) モデルの数が $K$ のままで、作成・統合が起きないこと、を記録から確かめる。
3. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と本規模の実行
#### Acceptance Criteria
1. The documentation shall 実験の入口（構成の一覧）と、FedDriftとの対応の文書（`docs/experiments/differences-from-feddrift.md`）に、新しい構成と、FedDriftの論文のOracleとの違い（逐次の評価で、予測には直前の標本のコンセプトを使う）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本のコマンドをユーザーに渡す。結果は、FedHASP（基準版16）・FedDrift・FedAvgとデータ・間隔・seedで対応させて集計し、実験の台帳に記録する。
