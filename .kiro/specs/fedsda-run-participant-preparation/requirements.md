# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: clientの5つの操作（`FedsdaRunClient`）と、サーバの1ラウンドの同期（`synchronize_models_in_server_round`）は、それぞれ、実旧と一致している。単一runの実行の枠（`execute_stream_protocol_run`。固定条件の検査→初期準備→概念列と観測列の生成→区間の進行→終端）も、移植済みである。しかし、実行の枠が求める、サーバの操作の実体と、runごとの参加者の初期準備（factory）がないので、最終構成のFedSDAの全体runを、新実装だけで実行できない。

変えたいこと: 最終構成のFedSDAの、サーバの操作の実体と、参加者の初期準備（初期モデルの事前学習→サーバの準備→全clientの組立て）を作り、実行の枠で全体runを実行できるようにする。全体runの結果（サーバと全clientの最終状態）を、実旧の全体runと照合する。

調査で確かめたこと（固定旧748c3aa）:

- 旧の全体run（`experiment.py`の`run_random_drift_experiment`）: 3つの乱数をseedで初期化→`_setup_server_and_clients`（事前学習→サーバの生成と、初期モデルのパラメータ・統計の登録→clientを0から順に生成し、サーバへ登録）→概念列の生成→観測列の生成→ラウンドごとに`_run_per_sample_timestep`（標本位置・clientの順の標本処理→全clientの保留中の学習→状態の報告→送信できるモデルを持つclientがいるかの照会→`run_round`→全clientの送信待ちの進行）→全clientの未完了の候補検証の回収→サーバの終端処理（最終構成のサーバでは、何もしない）→指標の計算と保存。
- 新の実行の枠は、この順（初期準備→概念列→観測列→区間の進行→終端）を、SINEの供給について、旧と同じ乱数の消費で行う（移植済み。旧の概念列・観測列との照合済み）。
- 旧は、サーバと全clientが、1つのPythonの乱数（module全体の`random`）を共有する。
- 旧は、1つの値（`distance_threshold`）を、clientの「許容する平均損失の増加」と、サーバのクラスタリングの閾値の、両方に使う。
- 旧のclientは、標本ごとに、真の概念IDを受け取る（診断だけに使う: 真の概念別の診断証拠、割当概念の計数、標本ごとの記録）。新の実行の枠の契約は、clientへ真の概念を渡さない（既存の要求）。新のclientは、真の概念を渡されなければ、これらの診断を行わない。
- 指標の計算と保存（`compute_metrics`ほか）と、goldenとの照合は、次のspecで扱う（調査の結果、2つに分けた）。

## Introduction

研究者が、最終構成のFedSDAの全体runを、新実装だけで、実行の枠から実行できるようにする。結果は、サーバと全clientのownerに残り、次のspec（指標の導出とgoldenとの照合）が読む。

## Boundary Context

- **In scope**: 実行の枠の契約を満たす、サーバの操作の実体（状態の報告、同期、終端）と、サーバのownerの組立て。runごとの参加者の初期準備（設定の束、factory: 事前学習→サーバ→全client）。全体runの、実旧の全体runとの照合（最終状態）。
- **Out of scope**: 指標の導出・保存・goldenとの照合（次のspec）。Linuxでの再現性の確認（次のspec）。実行の枠の契約の変更（真の概念をclientへ渡すこと）。完全なrun設定の、保存表現・preset・掃引からの組立て。SINE以外のdataset。
- **Adjacent expectations**: 実行の枠、事前学習、clientの組立て、サーバの1ラウンドの同期は、移植済みの部品をそのまま使い、中の処理を変えない。

## Requirements

### Requirement 1: サーバの操作

**Objective:** As a 研究者, I want サーバが、実行の枠の3つの操作を、旧と同じ処理で行ってほしい, so that 実行の枠から、サーバの同期を進められる

#### Acceptance Criteria

1. When 同期の前の記録を求められたとき, the Run Server shall 上りの軽量メッセージ数へ、clientの数を足す。
2. When 同期を求められたとき, the Run Server shall サーバの1ラウンドの同期を、渡されたラウンドと、「新規モデルの登録が可能か」をクラスタリングの有効・無効として、行う。同期の結果を、ラウンドの順に保持する。
3. When 開始済みの通信の終端を求められたとき, the Run Server shall 状態を変えない（最終構成のサーバには、終端まで持ち越す処理がない）。
4. The Run Server shall グローバルモデル・通信量・クロス評価の記録・クラスタリングの記録のownerを、読取り用に公開する。

### Requirement 2: 参加者の初期準備

**Objective:** As a 研究者, I want runごとの参加者が、旧と同じ順と乱数の消費で準備されてほしい, so that 全体runが、旧と同じ初期状態から始まる

#### Acceptance Criteria

1. When 初期準備を求められたとき, the Participant Factory shall 初期モデルを事前学習し（runの乱数源と標本生成器を使う）、その結果のパラメータと損失統計でグローバルモデルのownerを作り、clientを、IDが0から、固定条件のclient数だけ、順に組み立て、サーバを作って、参加者として返す。
2. The Participant Factory shall サーバと全clientへ、runの同じPythonの乱数生成器を渡す。
3. The Participant Factory shall 初期準備のたびに、新しい参加者（新しいowner）を作り、前のrunの参加者と状態を共有しない。直前に準備した参加者を、読取り用に公開する。
4. When 設定の事前検査を求められたとき, the Participant Factory shall 設定の束を再検査する。乱数を消費せず、参加者を作らない。

### Requirement 3: 設定の束

**Objective:** As a 研究者, I want 全体runに要る設定が、1つの束で、生成時に確かめられてほしい, so that 不正な設定が、runの途中まで進んでから見つかることがない

#### Acceptance Criteria

1. The 設定の束 shall clientの設定の束、事前学習の条件、モデルの構造の設定、隠れ層の幅、クラスタリングの判定の基準、1つのモデルを評価するclientの上限を持ち、生成時に、型と範囲を確かめる。
2. If クラスタリングの閾値が、clientの「許容する平均損失の増加」と違う値のとき, the 設定の束 shall 拒否する（最終FedSDA構成では、1つの値を両方に使う）。
3. The 設定の束 shall 事前学習のoptimizerの設定を、別に持たない（clientの設定の束の、作り直すモデルのoptimizerの設定を使う）。

### Requirement 4: 実旧の全体runとの一致

**Objective:** As a 研究者, I want 新の全体runが、旧の全体runと同じ最終状態になってほしい, so that 次のspecで、指標とgoldenの照合へ進める

#### Acceptance Criteria

1. When 同じ固定条件（seed、client数、標本数、集約間隔、概念列の条件）と設定を与えたとき, the 全体run shall 実旧の全体run（旧の全体の流れを、旧の部品を旧の順に呼んで再現したもの）と、概念列、観測列、サーバの全状態（グローバルモデル、統計、次の正式ID、来歴、通信量）、クロス評価とクラスタリングの診断の記録、各clientの全状態、runのPythonとNumPyの乱数の最終状態が一致する。
2. The 対照 shall 複数のseedと、集約間隔・学習の間隔が違う条件で、警報、候補の採用、新規モデルの登録、クラスタリング、統合、終端での未完了の候補検証の回収を通る。
3. When 真の概念を渡さずに全体runを実行したとき, the 全体run shall 真の概念に依存する診断（真の概念別の診断証拠、割当概念の計数、標本ごとの記録の概念、クラスタリングの真の概念の一致）を除いて、真の概念を渡した全体runと、同じ状態になる。
4. When 同じ固定条件で全体runを2回実行したとき, the 全体run shall 同じ結果になり、2回めの参加者は、1回めの参加者と状態を共有しない。

### Requirement 5: 拒否

**Objective:** As a 研究者, I want 不正な入力が、状態を変える前に拒否されてほしい, so that 準備の途中の状態が残らない

#### Acceptance Criteria

1. If 設定の束、固定条件、乱数源、標本生成器が、決まった型でないとき、またはdatasetがSINEでないとき, the Participant Factory shall 乱数を消費する前に拒否する。
2. If サーバの操作の引数（ラウンド、登録が可能かのフラグ、完了したラウンド数）が、決まった型と範囲でないとき, the Run Server shall 状態を変える前に拒否する。
3. The 初期準備 shall 並行した呼出しでの結果を保証しない。

### Requirement 6: 新実装だけで動くことと依存の向き

**Objective:** As a 研究者, I want 全体runが、旧実装なしで動いてほしい, so that 後続のspecへ進める

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、factoryと実行の枠で全体runを実行し、区間の進行の件数と、サーバとclientの状態の対応を確かめられる。
2. The 初期準備 shall 新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
